"""
Wrapper for different contrastive headers.

@Filename    header.py
@Author      Kion
@Created     08/31/22
"""

import torch.nn as nn
from lightly.models.utils import batch_unshuffle

from model.contrastive.config import ContrastiveHeaderConfig
from model.contrastive.projection_header import ProjectionHeader
from model.contrastive.projection_prediction_header import ProjectionPredictionHeader
from model.contrastive.transop_header import TransportOperatorHeader
from model.type import HeaderInput, HeaderOutput


class ContrastiveHeader(nn.Module):
    def __init__(
        self,
        header_cfg: ContrastiveHeaderConfig,
        backbone_feature_dim: int,
        enable_momentum: bool = False,
    ):
        super(ContrastiveHeader, self).__init__()
        self.header_cfg = header_cfg

        self.projection_header = None
        if self.header_cfg.enable_projection_header:
            self.projection_header = ProjectionHeader(
                self.header_cfg.projection_header_cfg, backbone_feature_dim, enable_momentum
            )

        self.transop_header = None
        if self.header_cfg.enable_transop_header:
            self.transop_header = TransportOperatorHeader(self.header_cfg.transop_header_cfg, backbone_feature_dim)

        self.proj_pred_header = None
        if self.header_cfg.enable_proj_pred_header:
            self.proj_pred_header = ProjectionPredictionHeader(
                self.header_cfg.proj_pred_header_cfg, backbone_feature_dim
            )

    def update_momentum_network(self, momentum_rate: float) -> None:
        if self.projection_header is not None:
            self.projection_header.update_momentum_network(momentum_rate)

    def unshuffle_outputs(self, shuffle_idx, header_out: HeaderOutput) -> HeaderOutput:
        if self.projection_header is not None:
            header_dict = header_out.header_dict
            header_dict["proj_01"] = batch_unshuffle(header_dict["proj_01"], shuffle_idx)

        return HeaderOutput(header_dict, header_out.distribution_data)

    def forward(self, header_input: HeaderInput) -> HeaderOutput:
        aggregate_header_out = {}

        distribution_data = None
        if self.transop_header is not None:
            header_out = self.transop_header(header_input)
            distribution_data = header_out.distribution_data
            aggregate_header_out.update(header_out.header_dict)

        if self.projection_header is not None:
            header_out = self.projection_header(header_input)
            aggregate_header_out.update(header_out.header_dict)

            if self.header_cfg.enable_transop_augmentation:
                feat_0, feat_1 = header_input.feature_0, header_input.feature_1
                if self.header_cfg.transop_header_cfg.enable_splicing:
                    feat_0 = TransportOperatorHeader.splice_input(
                        feat_0, self.header_cfg.transop_header_cfg.splice_dim
                    )
                feat_aug = self.transop_header.sample_augmentation(feat_0.detach(), distribution_data)
                if self.header_cfg.transop_header_cfg.enable_splicing:
                    feat_aug = feat_aug.reshape(feat_1.shape)
                    feat_0 = feat_0.reshape(feat_1.shape)
                aug_header_input = HeaderInput(*header_input[:-2], feat_aug, feat_1.detach())
                aug_header_out = self.projection_header(aug_header_input)
                aggregate_header_out["proj_10"] = aug_header_out.header_dict["proj_00"]
                aggregate_header_out["proj_11"] = aug_header_out.header_dict["proj_01"]

        if self.proj_pred_header is not None:
            header_out = self.proj_pred_header(header_input)
            aggregate_header_out.update(header_out.header_dict)

        return HeaderOutput(aggregate_header_out, distribution_data)

    def get_param_groups(self):
        param_list = []
        if self.projection_header is not None:
            param_list += [{"params": self.projection_header.parameters()}]
        if self.transop_header is not None:
            param_list += self.transop_header.get_param_groups()
        if self.proj_pred_header is not None:
            param_list += [{"params": self.proj_pred_header.parameters()}]

        return param_list

    @staticmethod
    def initialize_header(
        header_cfg: ContrastiveHeaderConfig,
        backbone_feature_dim: int,
        enable_momentum: bool = False,
    ) -> "ContrastiveHeader":
        return ContrastiveHeader(header_cfg, backbone_feature_dim, enable_momentum)