"""
Main model wrapper that initializes the model used for training.

@Filename    model.py
@Author      Kion
@Created     08/31/22
"""

from typing import Tuple

import torch
import torch.nn as nn

from model.backbone import Backbone
from model.config import ModelConfig
from model.header import ContrastiveHeader


class Model(nn.Module):
    def __init__(self, backbone, contrastive_header):
        super(Model, self).__init__()
        self.backbone = backbone
        self.contrastive_header = contrastive_header

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Take input image and get prediction from contrastive header.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            The input image, the encoded features from the backbone, and the prediction provided by the header.
        """
        features = self.backbone(x)
        prediction = self.contrastive_header(x, features)
        return (x, features, prediction)

    @staticmethod
    def initialize_model(model_cfg: ModelConfig) -> "Model":
        backbone = Backbone.initialize_backbone(model_cfg.backbone_cfg)
        contrastive_header = ContrastiveHeader.initialize_header(model_cfg.header_cfg)
        return Model(backbone, contrastive_header)
