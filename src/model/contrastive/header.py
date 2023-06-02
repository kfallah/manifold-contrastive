"""
Wrapper for different contrastive headers.

@Filename    header.py
@Author      Kion
@Created     08/31/22
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from model.contrastive.config import ContrastiveHeaderConfig
from model.contrastive.projection_header import ProjectionHeader
from model.contrastive.transop_header import TransportOperatorHeader
from model.type import HeaderInput, HeaderOutput


class ContrastiveHeader(nn.Module):
    def __init__(
        self,
        header_cfg: ContrastiveHeaderConfig,
        backbone_feature_dim: int,
    ):
        super(ContrastiveHeader, self).__init__()
        self.header_cfg = header_cfg

        self.projection_header = None
        if self.header_cfg.enable_projection_header:
            self.projection_header = ProjectionHeader(self.header_cfg.projection_header_cfg, backbone_feature_dim)

        self.transop_header = None
        if self.header_cfg.enable_transop_header:
            self.transop_header = TransportOperatorHeader(self.header_cfg.transop_header_cfg, backbone_feature_dim)

    def load_model_state(self, state_path: str) -> None:
        header_weights = torch.load(state_path, map_location="cuda:0")["model_state"]
        for name, param in header_weights.items():
            if self.projection_header is not None:
                proj_name = name.replace("contrastive_header.projection_header.", "")
                if proj_name in self.projection_header.state_dict():
                    if isinstance(param, nn.Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    self.projection_header.state_dict()[proj_name].copy_(param)

            if self.transop_header is not None:
                to_name = name.replace("contrastive_header.transop_header.", "")
                if to_name in self.transop_header.state_dict():
                    if isinstance(param, nn.Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    self.transop_header.state_dict()[to_name].copy_(param)

    def forward(self, header_input: HeaderInput, nn_queue: nn.Module = None) -> HeaderOutput:
        aggregate_header_out = {}
        curr_iter = header_input.curr_iter

        distribution_data = None
        if self.transop_header is not None:
            header_out = self.transop_header(header_input, nn_queue)
            distribution_data = header_out.distribution_data
            aggregate_header_out.update(header_out.header_dict)

        if self.header_cfg.enable_transop_augmentation:
            enc = self.transop_header.coefficient_encoder
            transop = self.transop_header.transop
            z0 = header_input.feature_0
            if self.transop_header.cfg.enable_direct:
                z0 = z0[:, :self.transop_header.cfg.block_dim]
            c0 = enc.prior_sample(z0.detach(), curr_iter=curr_iter, distribution_params=distribution_data.prior_params)
            with autocast(enabled=False):
                z0_aug = transop(z0.float(), c0, transop_grad=self.header_cfg.enable_transop_prior_grad)
            aggregate_header_out["z0_aug"] = z0_aug
        elif self.header_cfg.enable_mixup_augmentation:
            z0, z1 = header_input.feature_0, header_input.feature_1
            mixup = torch.rand(len(z0), device=z0.device)
            z0_aug = mixup*z1 + (1-mixup)*z0
            aggregate_header_out["z0_aug"] = z0_aug
        elif self.header_cfg.enable_gaussian_augmentation:
            z0, z1 = header_input.feature_0, header_input.feature_1
            dist = torch.linalg.norm(z0 - z1, dim=-1).detach()
            noise = torch.randn(len(z0), device=z0.device) * torch.sqrt(dist)
            z0_aug = z0 + noise
            aggregate_header_out["z0_aug"] = z0_aug
        else:
            aggregate_header_out["z0_aug"] = header_input.feature_0

        if self.projection_header is not None:
            header_out = self.projection_header(header_input, nn_queue)
            aggregate_header_out.update(header_out.header_dict)

        return HeaderOutput(aggregate_header_out, distribution_data)

    def get_param_groups(self):
        param_list = []
        if self.projection_header is not None:
            param_list += [{"params": self.projection_header.parameters()}]
        if self.transop_header is not None:
            param_list += self.transop_header.get_param_groups()

        return param_list

    @staticmethod
    def initialize_header(
        header_cfg: ContrastiveHeaderConfig,
        backbone_feature_dim: int,
    ) -> "ContrastiveHeader":
        return ContrastiveHeader(header_cfg, backbone_feature_dim)
