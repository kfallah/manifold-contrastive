"""
Backbone network whose feature space will be used for self-supervised learning.

@Filename    backbone.py
@Author      Kion
@Created     08/31/22
"""

import os

import torch
import torch.nn as nn
from lightly.models.utils import deactivate_requires_grad

from model.config import BackboneConfig
from model.public.wide_resnet import WideResnetFMPT


class Backbone(nn.Module):
    def __init__(self, backbone_cfg: BackboneConfig, backbone_network: nn.Module):
        super(Backbone, self).__init__()
        self.config = backbone_cfg
        self.backbone_network = backbone_network

    def forward(self, x):
        return self.backbone_network(x)

    @staticmethod
    def initialize_backbone(backbone_cfg: BackboneConfig, dataset_name: str) -> "Backbone":
        if backbone_cfg.hub_model_name.split(":")[0] == "torchhub":
            arch_name = backbone_cfg.hub_model_name.split(":")[1]
            backbone_network = torch.hub.load("pytorch/vision:v0.10.0", arch_name, pretrained=backbone_cfg.pretrained)
            backbone_feature_dim = backbone_network.fc.in_features
            backbone_network.fc = nn.Identity()

            if dataset_name == "CIFAR10" or dataset_name == "CIFAR100" or dataset_name == "STL10":
                backbone_network.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
                if dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
                    backbone_network.maxpool = nn.Identity()
        elif backbone_cfg.hub_model_name == "wresnet-28-2":
            backbone_network = WideResnetFMPT(10, k=2, n=28)
            backbone_feature_dim = backbone_network.fc.in_features
            backbone_network.fc = nn.Identity()

        network = Backbone(backbone_cfg, backbone_network)

        if backbone_cfg.load_backbone:
            backbone_weights = torch.load(backbone_cfg.load_backbone, map_location="cuda:0")["model_state"]
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
