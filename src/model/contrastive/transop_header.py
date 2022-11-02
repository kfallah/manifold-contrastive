"""
Transport operator header that estimates the manifold path between a pair of points.

@Filename    transop_header.py
@Author      Kion
@Created     09/07/22
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.models.modules import NNCLRProjectionHead, NNMemoryBankModule
from model.config import LossConfig
from model.contrastive.config import TransportOperatorConfig
from model.manifold.l1_inference import infer_coefficients
from model.manifold.transop import TransOp_expm
from model.manifold.vi_encoder import VIEncoder
from model.type import DistributionData, HeaderInput, HeaderOutput
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
                backbone_feature_dim,
                self.transop_cfg.projection_hidden_dim,
                self.transop_cfg.projection_out_dim,
            )
            feature_dim = self.transop_cfg.projection_out_dim
        elif self.transop_cfg.projection_type == "Linear":
            self.projector = nn.Linear(
                backbone_feature_dim, self.transop_cfg.projection_out_dim
            )
            feature_dim = self.transop_cfg.projection_out_dim
        else:
            raise NotImplementedError

        if self.transop_cfg.enable_splicing:
            feature_dim = self.transop_cfg.splice_dim

        self.transop = TransOp_expm(
            M=self.transop_cfg.dictionary_size,
            N=feature_dim,
            stable_init=self.transop_cfg.stable_operator_initialization,
        )

        self.coefficient_encoder = None
        if self.transop_cfg.enable_variational_inference:
            self.coefficient_encoder = VIEncoder(
                self.transop_cfg.vi_cfg,
                feature_dim,
                self.transop_cfg.dictionary_size,
                self.transop_cfg.lambda_prior,
            )

        """
        TODO: Add support back for ntx_loss sampling
        self.ntx_loss = None
        if transop_cfg.use_ntxloss_sampling:
            self.ntx_loss = NTXentLoss(
                memory_bank_size=loss_cfg.memory_bank_size,
                temperature=loss_cfg.ntxent_temp,
                normalize=loss_cfg.ntxent_normalize,
                loss_type=loss_cfg.ntxent_logit,
                reduction="none",
            )
        """

        self.nn_memory_bank = None
        if self.transop_cfg.enable_nn_point_pair:
            self.nn_memory_bank = NNMemoryBankModule(
                size=self.transop_cfg.nn_memory_bank_size
            )

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
                    "lr": self.transop_cfg.vi_cfg.variational_encoder_lr,
                    "weight_decay": self.transop_cfg.vi_cfg.variational_encoder_weight_decay,
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
            return HeaderOutput(
                feat_0,
                feat_1,
                feat_0,
                feat_1,
                distribution_data,
            )

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

        # Splice input into sequence if enabled
        if self.transop_cfg.enable_splicing:
            feat_dim = z0.shape[-1]
            z0 = (
                torch.stack(torch.split(z0, self.transop_cfg.splice_dim, dim=-1))
                .transpose(0, 1)
                .reshape(-1, self.transop_cfg.splice_dim)
            )
            z1_use = (
                torch.stack(torch.split(z1_use, self.transop_cfg.splice_dim, dim=-1))
                .transpose(0, 1)
                .reshape(-1, self.transop_cfg.splice_dim)
            )

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
                    num_trials=1,
                    device=z0.device,
                )
            distribution_data = DistributionData(None, None, None, c)
        else:
            # Use a variational approach
            if self.transop_cfg.vi_cfg.encode_features:
                # Use the features to estimate coefficients
                distribution_data = self.coefficient_encoder(
                    z0 / self.transop_cfg.latent_scale,
                    z1_use / self.transop_cfg.latent_scale,
                    self.transop,
                )
            else:
                # Use the original images to estimate coefficients
                distribution_data = self.coefficient_encoder(x0, x1, self.transop)
            c = distribution_data.samples

        # Matrix exponential not supported with float16
        with autocast(enabled=False):
            z1_hat = (
                self.transop(
                    z0.float().unsqueeze(-1) / self.transop_cfg.latent_scale, c
                ).squeeze(dim=-1)
                * self.transop_cfg.latent_scale
            )

        # Put all the outputs back together again if spliced
        if self.transop_cfg.enable_splicing:
            z0 = z0.reshape(-1, feat_dim)
            z1 = z1.reshape(-1, feat_dim)
            z1_hat = z1_hat.reshape(-1, feat_dim)
            z1_use = z1_use.reshape(-1, feat_dim)

        return HeaderOutput(z0, z1, z1_hat, z1_use, distribution_data=distribution_data)
