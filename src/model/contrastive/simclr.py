"""
SimCLR header with different options for projection head.

@Filename    simclr.py
@Author      Kion
@Created     09/02/22
"""

import math

import torch
import torch.nn as nn
from lightly.models.modules.heads import SimCLRProjectionHead
from model.contrastive.config import SimCLRHeaderConfig


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


class SimCLRHeader(nn.Module):
    def __init__(self, simclr_cfg: SimCLRHeaderConfig, backbone_feature_dim: int):
        super(SimCLRHeader, self).__init__()
        self.simclr_cfg = simclr_cfg
        self.projector = None
        if self.simclr_cfg.projection_type == "MLP":
            self.projector = SimCLRProjectionHead(
                backbone_feature_dim, self.simclr_cfg.hidden_dim, self.simclr_cfg.output_dim
            )
        elif self.simclr_cfg.projection_type == "Linear":
            self.projector = nn.Linear(backbone_feature_dim, self.simclr_cfg.output_dim)
        elif self.simclr_cfg.projection_type == "RandomProjection":
            self.projector = RandomProjection(backbone_feature_dim, self.simclr_cfg.output_dim)
        elif self.simclr_cfg.projection_type == "None" or self.simclr_cfg.projection_type == "Direct":
            self.projector = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, feat) -> torch.Tensor:
        pred = self.projector(feat)
        # For DirectCLR only use a subset of the features for prediction
        if self.simclr_cfg.projection_type == "Direct":
            pred = pred[..., : self.simclr_cfg.direct_proj_num_dim]
        return pred
