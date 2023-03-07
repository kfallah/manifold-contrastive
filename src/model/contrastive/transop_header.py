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
from model.manifold.vi_encoder import CoefficientEncoder
from model.type import DistributionData, HeaderInput, HeaderOutput


class TransportOperatorHeader(nn.Module):
    def __init__(
        self,
        transop_cfg: TransportOperatorConfig,
        backbone_feature_dim: int,
    ):
        super(TransportOperatorHeader, self).__init__()
        self.cfg = transop_cfg

        if self.cfg.enable_block_diagonal:
            backbone_feature_dim = self.cfg.block_dim

        self.transop = TransOp_expm(
            M=self.cfg.dictionary_size,
            N=backbone_feature_dim,
            stable_init=self.cfg.stable_operator_initialization,
            real_range=self.cfg.real_range_initialization,
            imag_range=self.cfg.image_range_initialization,
        )

        self.coefficient_encoder = None
        if self.cfg.enable_variational_inference:
            self.coefficient_encoder = CoefficientEncoder(
                self.cfg.vi_cfg,
                backbone_feature_dim,
                self.cfg.dictionary_size,
                self.cfg.lambda_prior,
            )

        self.nn_memory_bank = None
        if self.cfg.enable_nn_point_pair:
            self.nn_memory_bank = NNMemoryBankModule(size=self.cfg.nn_memory_bank_size)

    def get_param_groups(self):
        param_list = [
            {
                "params": self.transop.parameters(),
                "lr": self.cfg.transop_lr,
                "eta_min": 1e-4,
                "weight_decay": self.cfg.transop_weight_decay,
                "disable_layer_adaptation": True,
            },
        ]
        if self.coefficient_encoder is not None:
            param_list.append(
                {
                    "params": self.coefficient_encoder.parameters(),
                    "lr": self.cfg.vi_cfg.variational_encoder_lr,
                    "eta_min": 1e-4,
                    "weight_decay": self.cfg.vi_cfg.variational_encoder_weight_decay,
                    "disable_layer_adaptation": True,
                }
            )
        return param_list

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
                self.cfg.enable_alternating_min
                and (header_input.curr_iter // self.cfg.alternating_min_step) % 2 == 0
            )
            or curr_iter < self.cfg.fine_tune_iter
        ):
            z0, z1 = z0.detach(), z1.detach()

        # either use the nearnest neighbor bank or the projected feature to make the prediction
        z0 = z0[: self.cfg.batch_size]
        if self.cfg.enable_nn_point_pair:
            z1_use = self.nn_memory_bank(z1.detach(), update=True).detach()[: self.cfg.batch_size]
        else:
            z1_use = z1[: self.cfg.batch_size]

        # If enabled, impose block diagonal constraint on operators by breaking up features into
        # sequence of length b (e.g., b=64)
        if self.cfg.enable_block_diagonal:
            z0 = torch.stack(torch.split(z0, self.cfg.block_dim, dim=-1)).transpose(0, 1)
            z1_use = torch.stack(torch.split(z1_use, self.cfg.block_dim, dim=-1)).transpose(0, 1)

        # Infer coefficients for point pair
        if not self.cfg.enable_variational_inference:
            # Use FISTA for exact inference
            with autocast(enabled=False):
                _, c = infer_coefficients(
                    z0.float().detach(),
                    z1_use.float().detach(),
                    self.transop.get_psi().float(),
                    self.cfg.lambda_prior,
                    max_iter=self.cfg.fista_num_iterations,
                    num_trials=1,
                    device=z0.device,
                )
            distribution_data = DistributionData(None, None, None, c)
        else:
            distribution_data = self.coefficient_encoder(z0.detach(), z1_use.detach(), self.transop, curr_iter)
            c = distribution_data.samples

            if self.cfg.enable_vi_refinement:
                z0_flat, z1_flat = z0.reshape(-1, z0.shape[-1]), z1_use.reshape(-1, z1_use.shape[-1])
                with autocast(enabled=False):
                    _, c_refine = infer_coefficients(
                        z0_flat.float().detach(),
                        z1_flat.float().detach(),
                        self.transop.get_psi().float(),
                        self.cfg.vi_refinement_lambda,
                        max_iter=self.cfg.fista_num_iterations,
                        num_trials=1,
                        lr=1e-2,
                        decay=0.99,
                        device=z0.device,
                        c_init=c.clone().detach().reshape(1, -1, self.cfg.dictionary_size),
                        c_scale=self.cfg.fista_var_reg_scale if self.cfg.enable_fista_var_reg else None,
                        c_scale_weight=self.cfg.fista_var_reg_weight,
                    )

                # Sometimes FISTA can result in NaN values in inference, handle them here.
                c = c_refine.detach().clamp(min=-2.0, max=2.0).nan_to_num(nan=0.01).reshape(c.shape)
                distribution_data = DistributionData(*distribution_data[:-1], c)

        # Whether or not to compute gradient through the transport operator
        transop_grad = (
            not (
                self.cfg.enable_alternating_min
                and (header_input.curr_iter // self.cfg.alternating_min_step) % 2 != 0
            )
            and curr_iter > self.cfg.start_iter
        )
        # Matrix exponential not supported with float16
        with autocast(enabled=False):
            z1_hat = (
                self.transop(
                    z0.float().unsqueeze(-1), c, transop_grad=transop_grad
                ).squeeze(dim=-1)
            )

        header_out["transop_z0"] = z0
        header_out["transop_z1"] = z1_use
        header_out["transop_z1hat"] = z1_hat
        
        if self.cfg.vi_cfg.enable_det_prior:
            prior_c = distribution_data.prior_params["shift"]
            z1_det_hat = (
                self.transop(
                    z0.detach().float().unsqueeze(-1), prior_c, transop_grad=False
                ).squeeze(dim=-1)
            )    
            header_out["transop_z1_det_hat"] = z1_det_hat     

        return HeaderOutput(header_out, distribution_data=distribution_data)
