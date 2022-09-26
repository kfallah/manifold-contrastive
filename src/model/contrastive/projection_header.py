"""
SimCLR header with different options for projection head.

Example models:
SimCLR
MoCoV2

@Filename    projection_header.py
@Author      Kion
@Created     09/02/22
"""

import copy
import math

import torch
import torch.nn as nn
from lightly.models.modules.heads import MoCoProjectionHead, SimCLRProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from model.contrastive.config import ProjectionHeaderConfig
from model.type import HeaderInput, HeaderOutput


class RandomProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(RandomProjection, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        random_proj = torch.randn((len(x), self.out_dim, self.in_dim), device=x.device, dtype=x.dtype) * (
            1 / math.sqrt(self.out_dim)
        )
        return (random_proj @ x.unsqueeze(-1)).squeeze(-1)


class ProjectionHeader(nn.Module):
    def __init__(
        self,
        proj_cfg: ProjectionHeaderConfig,
        backbone_feature_dim: int,
        enable_momentum: bool = False,
    ):
        super(ProjectionHeader, self).__init__()
        self.proj_cfg = proj_cfg
        self.enable_momentum = enable_momentum
        self.projector = None
        if self.proj_cfg.projection_type == "MLP":
            if self.proj_cfg.header_name == "SimCLR":
                self.projector = SimCLRProjectionHead(
                    backbone_feature_dim, self.proj_cfg.hidden_dim, self.proj_cfg.output_dim
                )
            elif self.proj_cfg.header_name == "MoCo":
                self.projector = MoCoProjectionHead(
                    backbone_feature_dim, self.proj_cfg.hidden_dim, self.proj_cfg.output_dim
                )
            else:
                raise NotImplementedError
        elif self.proj_cfg.projection_type == "Linear":
            self.projector = nn.Linear(backbone_feature_dim, self.proj_cfg.output_dim, bias=False)
        elif self.proj_cfg.projection_type == "RandomProjection":
            self.projector = RandomProjection(backbone_feature_dim, self.proj_cfg.output_dim)
        elif self.proj_cfg.projection_type == "None" or self.proj_cfg.projection_type == "Direct":
            self.projector = nn.Identity()
        else:
            raise NotImplementedError

        if self.enable_momentum:
            self.momentum_projector = copy.deepcopy(self.projector)
            deactivate_requires_grad(self.momentum_projector)

    def update_momentum_network(self, momentum_rate: float) -> None:
        assert self.enable_momentum
        update_momentum(self.projector, self.momentum_projector, momentum_rate)

    def forward(self, header_input: HeaderInput) -> HeaderOutput:
        pred_0 = self.projector(header_input.feature_0)
        pred_1 = None
        if header_input.feature_1 is not None:
            if self.enable_momentum:
                pred_1 = self.momentum_projector(header_input.feature_1)
            else:
                pred_1 = self.projector(header_input.feature_1)

        # For DirectCLR only use a subset of the features for prediction
        if self.proj_cfg.projection_type == "Direct":
            pred_0 = pred_0[..., : self.proj_cfg.direct_proj_num_dim]
            if pred_1 is not None:
                pred_1 = pred_1[..., : self.proj_cfg.direct_proj_num_dim]

        return HeaderOutput(header_input.feature_0, header_input.feature_1, pred_0, pred_1, distribution_data=None)
