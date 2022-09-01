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
    # Dimensions [B x V x H x W x C]
    x_list: torch.Tensor
    # List of indices for each entry in the batch
    # Dimensions [B]
    x_idx: torch.Tensor
    # List of features from backbone encoder
    # Dimensions [B x V x D]
    feature_list: torch.Tensor
    # List of predictions from contrastive header
    # Dimensions [B x V x D]
    prediction_list: torch.Tensor


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

    def forward(self, x_list: torch.Tensor, x_idx: List) -> Tuple[ModelOutput]:
        """Take input image and get prediction from contrastive header.

        Args:
            x_list (torch.Tensor): [B x V x H x W x C] List of Tensors where each entry is a batch of different views
                                    of the image.
            x_idx (torch.Tensor): [B] Indices of each entry in the batch.

        Returns:
            Returns the encoded features from the backbone and the prediction provided by the header.
        """
        if self.model_cfg.concat_different_views:
            feature = self.backbone(x_list)  # [B x V x D]
            prediction = self.contrastive_header(x_list, feature)  # [B x V x D]
        else:
            x_view_first = x_list.transpose(0, 1)  # [V x B x H x W x C]
            feature = torch.stack([self.backbone(x) for x in x_view_first])
            prediction = torch.stack(
                [self.contrastive_header(x, features) for (x, features) in zip(x_view_first, feature)]
            )
        model_output = ModelOutput(x_list, x_idx, feature.transpose(0, 1), prediction.transpose(0, 1))
        return model_output

    def compute_loss(self, model_output: ModelOutput) -> Tuple[Dict[str, float], float]:
        loss_meta = {}
        total_loss = 0.0

        if self.loss_cfg.ntxent_loss_active:
            assert model_output.prediction_list.shape[1] == 2
            ntxent_loss = self.criterion["ntxent_loss"](
                model_output.prediction_list[:, 0], model_output.prediction_list[:, 1]
            )
            total_loss += ntxent_loss
            loss_meta["ntxent_loss"] = ntxent_loss.item()

        return loss_meta, total_loss

    @staticmethod
    def initialize_model(model_cfg: ModelConfig, devices: List[int]) -> "Model":
        backbone, backbone_feature_dim = Backbone.initialize_backbone(model_cfg.backbone_cfg)
        contrastive_header = ContrastiveHeader.initialize_header(model_cfg.header_cfg, backbone_feature_dim)
        model = Model(model_cfg, backbone, contrastive_header)
        model = model.to(devices[0])
        if len(devices) > 1:
            model = torch.nn.DataParallel(model, device_ids=devices)
        return model
