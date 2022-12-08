"""
Transport operator header that estimates the manifold path between a pair of points.

@Filename    transop_header.py
@Author      Kion
@Created     09/07/22
"""
import torch
import torch.nn as nn
from lightly.models.modules import NNMemoryBankModule
from torch.cuda.amp import autocast

from model.contrastive.config import TransportOperatorConfig
from model.manifold.l1_inference import infer_coefficients
from model.manifold.transop import TransOp_expm
from model.manifold.vi_encoder import VIEncoder
from model.type import DistributionData, HeaderInput, HeaderOutput


class TransportOperatorHeader(nn.Module):
    def __init__(
        self,
        transop_cfg: TransportOperatorConfig,
        backbone_feature_dim: int,
    ):
        super(TransportOperatorHeader, self).__init__()
        self.transop_cfg = transop_cfg

        if self.transop_cfg.enable_splicing:
            backbone_feature_dim = self.transop_cfg.splice_dim

        self.transop = TransOp_expm(
            M=self.transop_cfg.dictionary_size,
            N=backbone_feature_dim,
            stable_init=self.transop_cfg.stable_operator_initialization,
        )

        self.coefficient_encoder = None
        if self.transop_cfg.enable_variational_inference:
            self.coefficient_encoder = VIEncoder(
                self.transop_cfg.vi_cfg,
                backbone_feature_dim,
                self.transop_cfg.dictionary_size,
                self.transop_cfg.lambda_prior,
            )

        self.nn_memory_bank = None
        if self.transop_cfg.enable_nn_point_pair:
            self.nn_memory_bank = NNMemoryBankModule(size=self.transop_cfg.nn_memory_bank_size)

    def get_param_groups(self):
        param_list = [
            {
                "params": self.transop.parameters(),
                "lr": self.transop_cfg.transop_lr,
                "weight_decay": self.transop_cfg.transop_weight_decay,
                "disable_layer_adaptation": True,
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

    def sample_augmentation(
        self, z: torch.Tensor, distribution_data: DistributionData, detach_coeff: bool = False
    ) -> torch.Tensor:
        c = self.coefficient_encoder.sample_prior(z, distribution_data)

        if detach_coeff:
            c = c.detach()
        psi_use = self.transop.psi.detach()
        T = torch.einsum("bm,mpk->bpk", c, psi_use)
        return (torch.matrix_exp(T) @ z.unsqueeze(-1)).squeeze(-1)

    def forward(self, header_input: HeaderInput) -> HeaderOutput:
        header_out = {}
        curr_iter = header_input.curr_iter
        x0, x1 = header_input.x_0, header_input.x_1
        z0, z1 = header_input.feature_0, header_input.feature_1
        distribution_data = None

        # Detach the predictions in the case where we dont want gradient going to the backbone
        # or when we do alternating minimization.
        if (
            (
                self.transop_cfg.enable_alternating_min
                and (header_input.curr_iter // self.transop_cfg.alternating_min_step) % 2 == 0
            )
            or self.transop_cfg.detach_feature
            or curr_iter < (self.transop_cfg.start_iter)
        ):
            z0, z1 = z0.detach(), z1.detach()

        # either use the nearnest neighbor bank or the projected feature to make the prediction
        z0 = z0[: self.transop_cfg.batch_size]
        if self.transop_cfg.enable_nn_point_pair:
            z1_use = self.nn_memory_bank(z1[: self.transop_cfg.batch_size].detach(), update=True).detach()
        else:
            z1_use = z1[: self.transop_cfg.batch_size]

        if self.coefficient_encoder:
            z0_coeff = self.coefficient_encoder.layer_norm(z0.detach())
            z1_use_coeff = self.coefficient_encoder.layer_norm(z1_use.detach())
            if self.transop_cfg.enable_splicing:
                z0_coeff = TransportOperatorHeader.splice_input(z0_coeff, self.transop_cfg.splice_dim)
                z1_use_coeff = TransportOperatorHeader.splice_input(z1_use_coeff, self.transop_cfg.splice_dim)

        # Splice input into sequence if enabled
        if self.transop_cfg.enable_splicing:
            feat_dim = z0.shape[-1]
            z0 = TransportOperatorHeader.splice_input(z0, self.transop_cfg.splice_dim)
            z1_use = TransportOperatorHeader.splice_input(z1_use, self.transop_cfg.splice_dim)

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
                    z0_coeff / self.transop_cfg.latent_scale,
                    z1_use_coeff / self.transop_cfg.latent_scale,
                    self.transop,
                )
            else:
                # Use the original images to estimate coefficients
                distribution_data = self.coefficient_encoder(x0, x1, self.transop)
            c = distribution_data.samples

        transop_grad = (
            not (
                self.transop_cfg.enable_alternating_min
                and (header_input.curr_iter // self.transop_cfg.alternating_min_step) % 2 != 0
            )
        )
        # Matrix exponential not supported with float16
        with autocast(enabled=False):
            z1_hat = (
                self.transop(
                    z0.float().unsqueeze(-1) / self.transop_cfg.latent_scale, c, transop_grad=transop_grad
                ).squeeze(dim=-1)
                * self.transop_cfg.latent_scale
            )

        # Put all the outputs back together again if spliced
        if self.transop_cfg.enable_splicing:
            z0 = z0.reshape(-1, feat_dim)
            z1_hat = z1_hat.reshape(-1, feat_dim)
            z1_use = z1_use.reshape(-1, feat_dim)

        header_out["transop_z0"] = z0
        header_out["transop_z1"] = z1_use
        header_out["transop_z1hat"] = z1_hat

        return HeaderOutput(header_out, distribution_data=distribution_data)

    @staticmethod
    def splice_input(x: torch.Tensor, splice_dim: int) -> torch.Tensor:
        return torch.stack(torch.split(x, splice_dim, dim=-1)).transpose(0, 1).reshape(-1, splice_dim)
