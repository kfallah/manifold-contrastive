"""
Main model wrapper that initializes the model used for training.

@Filename    model.py
@Author      Kion
@Created     08/31/22
"""

from typing import Dict, List, NamedTuple, Tuple

import torch
import torch.nn as nn
from lightly.loss import NTXentLoss

from model.backbone import Backbone
from model.config import ModelConfig
from model.header import ContrastiveHeader


class ModelOutput(NamedTuple):
    # List of batch of input images, where each list entry is a different view
    x_list: List[torch.Tensor]
    # List of indices for each entry in the batch
    x_idx: List
    # List of features from backbone encoder
    feature_list: List[torch.Tensor]
    # List of predictions from contrastive header
    prediction_list: List[torch.Tensor]


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

    def forward(self, x_list: List[torch.Tensor], x_idx: List) -> Tuple[ModelOutput, Dict[str, float], float]:
        """Take input image and get prediction from contrastive header.

        Args:
            x_list (List[torch.Tensor]): List of Tensors where each entry is a batch of different views of the image.
            x_idx (List): Indices of each entry in the batch.

        Returns:
            Returns the encoded features from the backbone and the prediction provided by the header.
        """
        if self.model_cfg.concat_different_views:
            x_cat = torch.cat(x_list)
            feature = self.backbone(x_cat)
            prediction = self.contrastive_header(x_cat, feature)
            feature_list = list(torch.split(feature, len(x_list[0])))
            prediction_list = list(torch.split(prediction, len(x_list[0])))
        else:
            feature_list = [self.backbone(x) for x in x_list]
            prediction_list = [self.contrastive_header(x, features) for (x, features) in zip(x_list, feature_list)]
        output = ModelOutput(x_list, x_idx, feature_list, prediction_list)
        loss_meta, total_loss = self.compute_loss(output)
        return (output, loss_meta, total_loss)

    def compute_loss(self, model_output: ModelOutput) -> Tuple[Dict[str, float], float]:
        loss_meta = {}
        total_loss = 0.0

        if self.loss_cfg.ntxent_loss_active:
            assert len(model_output.prediction_list) == 2
            ntxent_loss = self.criterion["ntxent_loss"](
                model_output.prediction_list[0], model_output.prediction_list[1]
            )
            total_loss += ntxent_loss
            loss_meta["ntxent_loss"] = ntxent_loss.item()

        return loss_meta, total_loss

    @staticmethod
    def initialize_model(model_cfg: ModelConfig) -> "Model":
        backbone, backbone_feature_dim = Backbone.initialize_backbone(model_cfg.backbone_cfg)
        contrastive_header = ContrastiveHeader.initialize_header(model_cfg.header_cfg, backbone_feature_dim)
        return Model(model_cfg, backbone, contrastive_header)
