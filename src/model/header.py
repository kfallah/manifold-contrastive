"""
Wrapper for different contrastive headers.

@Filename    header.py
@Author      Kion
@Created     08/31/22
"""

import torch
import torch.nn as nn

from model.contrastive.config import ContrastiveHeaderConfig, SimCLRHeaderConfig
from model.contrastive.simclr import SimCLRHeader


class ContrastiveHeader(nn.Module):
    def __init__(self, header_cfg: ContrastiveHeaderConfig, backbone_feature_dim: int):
        super(ContrastiveHeader, self).__init__()
        self.header_cfg = header_cfg

        self.simclr_header = None
        if header_cfg.header_name == "SimCLR":
            self.simclr_header = SimCLRHeader(header_cfg, backbone_feature_dim)
        else:
            raise NotImplementedError

    def forward(self, x, z) -> torch.Tensor:
        prediction_out = None
        if self.header_cfg.header_name == "SimCLR":
            prediction_out = self.simclr_header(z)
        return prediction_out

    @staticmethod
    def initialize_header(header_cfg: ContrastiveHeaderConfig, backbone_feature_dim: int) -> "ContrastiveHeader":
        return ContrastiveHeader(header_cfg, backbone_feature_dim)
