"""
Backbone network whose feature space will be used for self-supervised learning.

@Filename    backbone.py
@Author      Kion
@Created     08/31/22
"""

import torch
import torch.nn as nn
from lightly.models.utils import deactivate_requires_grad

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
        backbone_feature_dim = backbone_network.fc.in_features
        backbone_network.fc = nn.Identity()
        network = Backbone(backbone_cfg, backbone_network)
        if backbone_cfg.load_backbone:
            backbone_weights = torch.load(f"../../model_zoo/{backbone_cfg.load_backbone}")["model_state"]
            own_state = network.state_dict()
            for name, param in backbone_weights.items():
                name = name.replace("backbone.", "")
                if name not in own_state:
                    continue
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)
        if backbone_cfg.freeze_backbone:
            deactivate_requires_grad(network)

        return network, backbone_feature_dim
