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
from lightly.models.modules.heads import (BarlowTwinsProjectionHead,
                                          BYOLProjectionHead)

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
    ):
        super(ProjectionHeader, self).__init__()
        self.proj_cfg = proj_cfg
        self.projector = None
        if self.proj_cfg.projection_type == "MLP":
            if self.proj_cfg.header_name == "SimCLR":
                self.projector = nn.Sequential(
                    nn.Linear(backbone_feature_dim, self.proj_cfg.hidden_dim),
                    nn.BatchNorm1d(self.proj_cfg.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.proj_cfg.hidden_dim, self.proj_cfg.output_dim),
                )
            elif self.proj_cfg.header_name == "VICReg":
                self.projector = BarlowTwinsProjectionHead(
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

        self.final_bn = None
        if self.proj_cfg.enable_final_batchnorm:
            self.final_bn = nn.BatchNorm1d(self.proj_cfg.output_dim)

    def forward(self, header_input: HeaderInput, nn_queue: nn.Module = None) -> HeaderOutput:
        header_out = {}
        z0, z1 = header_input.feature_0, header_input.feature_1
        if nn_queue is not None:
            z1 = nn_queue(z0.detach(), update=False).detach()
        proj = self.projector(torch.cat([z0, z1]))

        if self.proj_cfg.enable_final_batchnorm:
            proj = self.final_bn(proj)

        proj_0 = proj[: len(z0)]
        proj_1 = proj[len(z1) :]

        # For DirectCLR only use a subset of the features for prediction
        if self.proj_cfg.projection_type == "Direct":
            proj_0 = proj_0[..., : self.proj_cfg.direct_proj_num_dim]
            proj_1 = proj_1[..., : self.proj_cfg.direct_proj_num_dim]

        header_out["proj_00"] = proj_0
        header_out["proj_01"] = proj_1

        return HeaderOutput(header_out, None)
