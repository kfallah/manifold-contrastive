"""
Backbone network whose feature space will be used for self-supervised learning.

@Filename    backbone.py
@Author      Kion
@Created     08/31/22
"""

import torch
import torch.nn as nn

from model.config import BackboneConfig


class Backbone(nn.Module):
    def __init__(self, backbone_cfg: BackboneConfig, backbone_network: nn.Module):
        super(Backbone, self).__init__()
        self.config = backbone_cfg
        self.backbone_network = backbone_network

    def forward(self, x):
        return self.backbone_network(x)

    @staticmethod
    def initialize_backbone(backbone_cfg: BackboneConfig) -> "Backbone":
        backbone_network = torch.hub.load(
            "pytorch/vision:v0.10.0", backbone_cfg.hub_model_name, pretrained=backbone_cfg.pretrained
        )
        backbone_network.fc = nn.Identity()
        return Backbone(backbone_cfg, backbone_network)
