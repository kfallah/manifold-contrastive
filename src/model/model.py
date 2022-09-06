"""
Main model wrapper that initializes the model used for training.

@Filename    model.py
@Author      Kion
@Created     08/31/22
"""

import copy
from typing import Dict, List, NamedTuple, Tuple

import torch
import torch.nn as nn
from lightly.loss import NTXentLoss
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)

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

        self.momentum_backbone = None
        self.momentum_header = None
        if self.model_cfg.enable_momentum_network:
            self.momentum_backbone = copy.deepcopy(self.backbone)
            self.momentum_header = copy.deepcopy(self.contrastive_header)
            deactivate_requires_grad(self.momentum_backbone)
            deactivate_requires_grad(self.momentum_header)

        self.criterion = {}
        if self.loss_cfg.ntxent_loss_active:
            self.criterion["ntxent_loss"] = NTXentLoss(
                memory_bank_size=self.loss_cfg.memory_bank_size, temperature=self.loss_cfg.ntxent_temp
            )

    def forward(self, x_list: torch.Tensor, x_idx: List) -> Tuple[ModelOutput]:
        """Take input image and get prediction from contrastive header.

        Args:
            x_list (torch.Tensor): [B x V x H x W x C] List of Tensors where each entry is a batch of different views
                                    of the image.
            x_idx (torch.Tensor): [B] Indices of each entry in the batch.

        Returns:
            Returns the encoded features from the backbone and the prediction provided by the header.
        """
        if not self.model_cfg.enable_momentum_network:
            if self.model_cfg.concat_different_views:
                feature = self.backbone(x_list)  # [B x V x D]
                prediction = self.contrastive_header(x_list, feature)  # [B x V x D]
            else:
                x_view_first = x_list.transpose(0, 1)  # [V x B x H x W x C]
                feature = torch.stack([self.backbone(x) for x in x_view_first])
                prediction = torch.stack(
                    [self.contrastive_header(x, features) for (x, features) in zip(x_view_first, feature)]
                )
            feature, prediction = feature.transpose(0, 1), prediction.transpose(0, 1)
        else:
            # Apply momentum encoder to the negative samples
            feature = self.backbone(x_list[:, 0])
            prediction = self.contrastive_header(x_list[:, 0], feature)

            # Only encode negatives if we are given a second view
            if x_list.shape[1] > 1:
                x_negative, shuffle = batch_shuffle(x_list[:, 1])
                feature_neg = self.momentum_backbone(x_negative)
                prediction_neg = self.momentum_header(x_negative, feature_neg)
                feature_neg, prediction_neg = batch_unshuffle(feature_neg, shuffle), batch_unshuffle(
                    prediction_neg, shuffle
                )
                feature = torch.stack([feature, feature_neg]).transpose(0, 1)
                prediction = torch.stack([prediction, prediction_neg]).transpose(0, 1)

        model_output = ModelOutput(x_list, x_idx, feature, prediction)
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

    def update_momentum_network(self) -> None:
        assert self.momentum_backbone is not None and self.momentum_header is not None
        update_momentum(self.backbone, self.momentum_backbone, self.model_cfg.momentum_network_update_rate)
        update_momentum(self.contrastive_header, self.momentum_header, self.model_cfg.momentum_network_update_rate)

    @staticmethod
    def initialize_model(model_cfg: ModelConfig, devices: List[int]) -> "Model":
        backbone, backbone_feature_dim = Backbone.initialize_backbone(model_cfg.backbone_cfg)
        contrastive_header = ContrastiveHeader.initialize_header(model_cfg.header_cfg, backbone_feature_dim)
        model = Model(model_cfg, backbone, contrastive_header)
        model = model.to(devices[0])
        if len(devices) > 1:
            model = torch.nn.DataParallel(model, device_ids=devices)
        return model
