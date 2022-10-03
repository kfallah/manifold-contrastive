"""
Transport operator header that estimates the manifold path between a pair of points.

@Filename    transop_header.py
@Author      Kion
@Created     09/07/22
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.models.modules import NNCLRProjectionHead, NNMemoryBankModule
from model.config import LossConfig
from model.contrastive.config import TransportOperatorConfig
from model.manifold.l1_inference import infer_coefficients
from model.manifold.transop import TransOp_expm
from model.manifold.vi_encoder import VIEncoder
from model.public.ntx_ent_loss import NTXentLoss
from model.type import DistributionData, HeaderInput, HeaderOutput, ModelOutput
from torch.cuda.amp import autocast


class TransportOperatorHeader(nn.Module):
    def __init__(
        self,
        transop_cfg: TransportOperatorConfig,
        loss_cfg: LossConfig,
        backbone_feature_dim: int,
        enable_momentum: bool,
    ):
        super(TransportOperatorHeader, self).__init__()
        self.transop_cfg = transop_cfg
        self.failed_iters = 0

        if self.transop_cfg.projection_type == "None":
            self.projector = nn.Identity()
            feature_dim = backbone_feature_dim
        elif self.transop_cfg.projection_type == "MLP":
            self.projector = NNCLRProjectionHead(
                backbone_feature_dim, self.transop_cfg.projection_hidden_dim, self.transop_cfg.projection_out_dim
            )
            feature_dim = self.transop_cfg.projection_out_dim
        elif self.transop_cfg.projection_type == "Linear":
            self.projector = nn.Linear(backbone_feature_dim, self.transop_cfg.projection_out_dim)
            feature_dim = self.transop_cfg.projection_out_dim
        else:
            raise NotImplementedError

        self.transop = TransOp_expm(M=self.transop_cfg.dictionary_size, N=feature_dim)

        self.coefficient_encoder = None
        if self.transop_cfg.enable_variational_inference:
            self.coefficient_encoder = VIEncoder(self.transop_cfg, feature_dim, self.transop_cfg.dictionary_size)

        self.transop_ema, self.enc_ema = None, None
        if enable_momentum:
            self.transop_ema = copy.deepcopy(self.transop)
            self.projector_ema = copy.deepcopy(self.projector)
            if self.transop_cfg.enable_variational_inference:
                self.enc_ema = copy.deepcopy(self.coefficient_encoder)

        self.ntx_loss = None
        if transop_cfg.use_ntxloss_sampling:
            self.ntx_loss = NTXentLoss(
                memory_bank_size=loss_cfg.memory_bank_size,
                temperature=loss_cfg.ntxent_temp,
                normalize=loss_cfg.ntxent_normalize,
                loss_type=loss_cfg.ntxent_logit,
                reduction="none",
            )

        self.nn_memory_bank = None
        if self.transop_cfg.enable_nn_point_pair:
            self.nn_memory_bank = NNMemoryBankModule(size=self.transop_cfg.nn_memory_bank_size)

    def update_momentum_network(self, momentum_rate: float, model_out: ModelOutput) -> None:
        assert self.transop_ema is not None
        assert model_out.distribution_data is not None and model_out.feature_1 is not None
        with torch.no_grad():
            z0, z1 = self.projector(model_out.feature_0), self.projector(model_out.feature_1)
            c = model_out.distribution_data.samples
            old_loss = F.mse_loss(model_out.prediction_0, model_out.prediction_1)
            with autocast(enabled=False):
                new_z1_hat = (
                    self.transop(
                        z0.detach().float().unsqueeze(-1) / self.transop_cfg.latent_scale, c.detach()
                    ).squeeze(dim=-1)
                    * self.transop_cfg.latent_scale
                )
            new_loss = F.mse_loss(new_z1_hat, z1)
            if new_loss > old_loss:
                self.transop.psi.data = self.transop_ema.psi.data
                self.projector = copy.deepcopy(self.projector_ema)
                if self.transop_cfg.enable_variational_inference:
                    self.coefficient_encoder = copy.deepcopy(self.enc_ema)
                self.failed_iters += 1
            self.transop_ema = copy.deepcopy(self.transop)
            self.projector_ema = copy.deepcopy(self.projector)
            if self.transop_cfg.enable_variational_inference:
                self.enc_ema = copy.deepcopy(self.coefficient_encoder)

    def get_param_groups(self):
        param_list = [
            {
                "params": self.transop.parameters(),
                "lr": self.transop_cfg.transop_lr,
                "weight_decay": self.transop_cfg.transop_weight_decay,
                "disable_layer_adaptation": True,
            },
            {
                "params": self.projector.parameters(),
                "lr": self.transop_cfg.projection_network_lr,
                "weight_decay": self.transop_cfg.projection_network_weight_decay,
            },
        ]
        if self.coefficient_encoder is not None:
            param_list.append(
                {
                    "params": self.coefficient_encoder.parameters(),
                    "lr": self.transop_cfg.variational_encoder_lr,
                    "weight_decay": self.transop_cfg.variational_encoder_weight_decay,
                    "disable_layer_adaptation": True,
                }
            )
        return param_list

    def forward(self, header_input: HeaderInput) -> HeaderOutput:
        x0, x1 = header_input.x_0, header_input.x_1
        feat_0, feat_1 = header_input.feature_0, header_input.feature_1
        distribution_data = None

        # In the case where only a single point pair is provided
        if feat_1 is None:
            return HeaderOutput(self.projector(feat_0), feat_1, self.projector(feat_0), feat_1, distribution_data)

        # Detach the predictions in the case where we dont want gradient going to the backbone
        if self.transop_cfg.detach_feature:
            feat_0, feat_1 = feat_0.detach(), feat_1.detach()

        # Project both features
        z0, z1 = self.projector(feat_0), self.projector(feat_1)

        # either use the nearnest neighbor bank or the projected feature to make the prediction
        if self.transop_cfg.enable_nn_point_pair:
            z1_use = self.nn_memory_bank(z1.detach(), update=True)
        else:
            z1_use = z1

        # Infer coefficients for point pair
        if self.coefficient_encoder is None:
            # Use FISTA for exact inference
            with autocast(enabled=False):
                _, c = infer_coefficients(
                    z0.float().detach() / self.transop_cfg.latent_scale,
                    z1_use.float().detach() / self.transop_cfg.latent_scale,
                    self.transop.get_psi().float(),
                    self.transop_cfg.lambda_prior,
                    max_iter=self.transop_cfg.fista_num_iterations,
                    num_trials=self.transop_cfg.iter_variational_samples,
                    device=z0.device,
                )
            distribution_data = DistributionData(c, None, None, None, None)
        else:
            # Use a variational approach
            if self.transop_cfg.variational_use_features:
                # Use the features to estimate coefficients
                distribution_data = self.coefficient_encoder(
                    z0 / self.transop_cfg.latent_scale, z1_use / self.transop_cfg.latent_scale
                )
            else:
                # Use the original images to estimate coefficients
                distribution_data = self.coefficient_encoder(x0, x1)

            # If using best of many loss, use this trick to only differentiate through the coefficient with the lowest
            # L2 error.
            with torch.no_grad():
                noise_list = []
                loss_list = []
                for _ in range(
                    self.transop_cfg.total_variational_samples // self.transop_cfg.iter_variational_samples
                ):

                    # Generate a new noise sample
                    u = self.coefficient_encoder.draw_noise_samples(
                        len(z1_use), self.transop_cfg.iter_variational_samples, z1_use.device
                    )
                    c = self.coefficient_encoder.reparameterize(
                        distribution_data.shift.unsqueeze(1).repeat(1, self.transop_cfg.iter_variational_samples, 1),
                        distribution_data.log_scale.unsqueeze(1).repeat(
                            1, self.transop_cfg.iter_variational_samples, 1
                        ),
                        u,
                    )

                    # Estimate z1 with transport operators
                    with autocast(enabled=False):
                        z1_hat = (
                            self.transop(z0.detach().float().unsqueeze(-1) / self.transop_cfg.latent_scale, c.detach())
                            .squeeze(dim=-1)
                            .transpose(0, 1)
                        ) * self.transop_cfg.latent_scale

                    # Perform max ELBO sampling to find the highest likelihood coefficient for each entry in the batch
                    if self.transop_cfg.use_ntxloss_sampling:
                        transop_loss = torch.stack(
                            [
                                self.ntx_loss(z1_hat[samp], z1_use)
                                for samp in range(self.transop_cfg.iter_variational_samples)
                            ]
                        ).T
                        # Account for duplicate from both views in ntxloss
                        transop_loss = 0.5 * (transop_loss[: len(z1_use)] + transop_loss[len(z1_use) :])
                    else:
                        transop_loss = (
                            F.mse_loss(
                                z1_hat,
                                z1_use.repeat(len(z1_hat), *torch.ones(z1_use.dim(), dtype=int)).detach(),
                                reduction="none",
                            )
                            .mean(dim=-1)
                            .transpose(0, 1)
                        )
                    noise_list.append(u)
                    loss_list.append(transop_loss)

                # Pick the best sample
                noise_list = torch.cat(noise_list, dim=1)
                loss_list = torch.cat(loss_list, dim=1)
                max_elbo = torch.argmin(loss_list, dim=1).detach()

                # Pick out best noise sample for each batch entry for reparameterization
                noise = noise_list[torch.arange(len(z0)), max_elbo]

            # Reparameterize with best noise sample (prevents backprop wrt all samples)
            c = self.coefficient_encoder.reparameterize(distribution_data.shift, distribution_data.log_scale, noise)
            distribution_data = DistributionData(c, *distribution_data[1:])

        # Matrix exponential not supported with float16
        with autocast(enabled=False):
            z1_hat = (
                self.transop(z0.float().unsqueeze(-1) / self.transop_cfg.latent_scale, c).squeeze(dim=-1)
                * self.transop_cfg.latent_scale
            )

        return HeaderOutput(z0, z1, z1_hat, z1_use, distribution_data=distribution_data)
