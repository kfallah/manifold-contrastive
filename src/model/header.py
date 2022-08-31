"""
Wrapper for different contrastive headers.

@Filename    header.py
@Author      Kion
@Created     08/31/22
"""

import torch
import torch.nn as nn
from lightly.models.modules.heads import SimCLRProjectionHead

from model.contrastive.config import ContrastiveHeaderConfig, SimCLRHeaderConfig


class ContrastiveHeader(nn.Module):
    def __init__(self, header_cfg: ContrastiveHeaderConfig, backbone_feature_dim: int):
        super(ContrastiveHeader, self).__init__()
        self.config = header_cfg
        if header_cfg.header_name == "SimCLR":
            simclr_cfg = SimCLRHeaderConfig(header_cfg)
            self.header = SimCLRProjectionHead(backbone_feature_dim, simclr_cfg.hidden_dim, simclr_cfg.output_dim)
        else:
            raise NotImplementedError

    def forward(self, x, z) -> torch.Tensor:
        prediction_out = None
        if self.header_cfg.header_name == "SimCLR":
            prediction_out = self.header(z)
        return prediction_out

    @staticmethod
    def initialize_header(header_cfg: ContrastiveHeaderConfig, backbone_feature_dim: int) -> "ContrastiveHeader":
        return ContrastiveHeader(header_cfg, backbone_feature_dim)
