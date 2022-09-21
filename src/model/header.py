"""
Wrapper for different contrastive headers.

@Filename    header.py
@Author      Kion
@Created     08/31/22
"""

import torch.nn as nn

from model.config import LossConfig
from model.contrastive.config import ContrastiveHeaderConfig
from model.contrastive.projection_header import ProjectionHeader
from model.contrastive.transop_header import TransportOperatorHeader
from model.type import HeaderInput, HeaderOutput, ModelOutput


class ContrastiveHeader(nn.Module):
    def __init__(
        self,
        header_cfg: ContrastiveHeaderConfig,
        loss_cfg: LossConfig,
        backbone_feature_dim: int,
        enable_momentum: bool = False,
    ):
        super(ContrastiveHeader, self).__init__()
        self.header_cfg = header_cfg

        self.projection_header = None
        self.transop_header = None
        if header_cfg.header_name == "SimCLR" or header_cfg.header_name == "MoCo":
            self.projection_header = ProjectionHeader(header_cfg, backbone_feature_dim, enable_momentum)
        elif header_cfg.header_name == "TransOp":
            self.transop_header = TransportOperatorHeader(header_cfg, loss_cfg, backbone_feature_dim, enable_momentum)
        else:
            raise NotImplementedError

    def update_momentum_network(self, momentum_rate: float, model_out: ModelOutput) -> None:
        if self.projection_header is not None:
            self.projection_header.update_momentum_network(momentum_rate)
        if self.transop_header is not None:
            self.transop_header.update_momentum_network(momentum_rate, model_out)

    def forward(self, header_input: HeaderInput) -> HeaderOutput:
        prediction_out = None
        if self.projection_header is not None:
            prediction_out = self.projection_header(header_input)
        elif self.transop_header is not None:
            prediction_out = self.transop_header(header_input)
        return prediction_out

    def get_param_groups(self):
        if self.projection_header is not None:
            return [{"params": self.projection_header.parameters()}]
        elif self.transop_header is not None:
            return self.transop_header.get_param_groups()

    @staticmethod
    def initialize_header(
        header_cfg: ContrastiveHeaderConfig,
        loss_cfg: LossConfig,
        backbone_feature_dim: int,
        enable_momentum: bool = False,
    ) -> "ContrastiveHeader":
        return ContrastiveHeader(header_cfg, loss_cfg, backbone_feature_dim, enable_momentum)
