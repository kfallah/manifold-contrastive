"""
Main model wrapper that initializes the model used for training.

@Filename    model.py
@Author      Kion
@Created     08/31/22
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from lightly.loss import NTXentLoss

from model.backbone import Backbone
from model.config import ModelConfig
from model.header import ContrastiveHeader


class Model(nn.Module):
    def __init__(self, model_cfg: ModelConfig, backbone, contrastive_header):
        super(Model, self).__init__()
        self.model_cfg = model_cfg
        self.loss_cfg = model_cfg.loss_cfg
        self.backbone = backbone
        self.contrastive_header = contrastive_header

        self.criterion = {}
        if self.loss_cfg.ntxent_loss_active:
            self.criterion["ntxent_loss"] = NTXentLoss(temperature=self.loss_cfg.ntxent_temp)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Take input image and get prediction from contrastive header.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            Returns the encoded features from the backbone and the prediction provided by the header.
        """
        features = self.backbone(x)
        prediction = self.contrastive_header(x, features)
        return (features, prediction)

    def compute_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        feat0: torch.Tensor,
        feat1: torch.Tensor,
        pred0: torch.Tensor,
        pred1: torch.Tensor,
    ) -> Tuple[float, Dict[str, float]]:
        loss_meta = {}
        total_loss = 0.0

        if self.loss_cfg.ntxent_loss_active:
            ntxent_loss = self.criterion["ntxent_loss"](pred0, pred1)
            total_loss += ntxent_loss
            loss_meta["ntxent_loss"] = ntxent_loss.item()

        return total_loss, loss_meta

    @staticmethod
    def initialize_model(model_cfg: ModelConfig) -> "Model":
        backbone, backbone_feature_dim = Backbone.initialize_backbone(model_cfg.backbone_cfg)
        contrastive_header = ContrastiveHeader.initialize_header(model_cfg.header_cfg, backbone_feature_dim)
        return Model(model_cfg, backbone, contrastive_header)
