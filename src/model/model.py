"""
Main model wrapper that initializes the model used for training.

@Filename    model.py
@Author      Kion
@Created     08/31/22
"""

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from model.type import HeaderInput, HeaderOutput, ModelOutput


class Model(nn.Module):
    def __init__(self, model_cfg: ModelConfig, backbone, contrastive_header):
        super(Model, self).__init__()
        self.model_cfg = model_cfg
        self.loss_cfg = model_cfg.loss_cfg
        self.backbone = backbone
        self.contrastive_header = contrastive_header

        self.momentum_backbone = None
        self.momentum_header = None
        if self.model_cfg.enable_backbone_momentum:
            self.momentum_backbone = copy.deepcopy(self.backbone)
            deactivate_requires_grad(self.momentum_backbone)

        self.criterion = {}
        if self.loss_cfg.ntxent_loss_active:
            self.criterion["ntxent_loss"] = NTXentLoss(
                memory_bank_size=self.loss_cfg.memory_bank_size, temperature=self.loss_cfg.ntxent_temp
            )

    def unshuffle_outputs(
        self, shuffle_idx, header_input: HeaderInput, header_out: HeaderOutput
    ) -> Tuple[HeaderInput, HeaderOutput]:
        x_1 = batch_unshuffle(header_input.x_1, shuffle_idx)
        feature_1 = batch_unshuffle(header_input.feature_1, shuffle_idx)
        prediction_1 = batch_unshuffle(header_out.prediction_1, shuffle_idx)
        header_input = HeaderInput(
            header_input.x_0,
            x_1,
            header_input.x_idx,
            header_input.feature_0,
            feature_1,
        )
        header_out = HeaderOutput(header_out.prediciton_0, prediction_1, header_out.distribution_data)
        return header_input, header_out

    def forward(self, x: torch.Tensor, x_idx: List) -> Tuple[ModelOutput]:
        """Take input image and get prediction from contrastive header.

        Args:
            x_list (torch.Tensor): [B x V x H x W x C] List of Tensors where each entry is a batch of different views
                                    of the image.
            x_idx (torch.Tensor): [B] Indices of each entry in the batch.

        Returns:
            Returns the encoded features from the backbone and the prediction provided by the header.
        """
        # Only a single view of an image is used
        if x.shape[1] == 1:
            feature_0 = self.backbone(x[:, 0]).unsqueeze(1)
            header_input = HeaderInput(x, None, x_idx, feature_0, None)
        # Two views of an image are provided
        elif x.shape[1] == 2:
            # No momentum encoder used for the backbone
            if not self.model_cfg.enable_backbone_momentum:
                feature_0, feature_1 = self.backbone(x[:, 0]), self.backbone(x[:, 1])
                x_1 = x[:, 1]
            else:
                feature_0 = self.backbone(x[:, 0])
                x_1, shuffle_idx = batch_shuffle(x[:, 1])
                feature_1 = self.momentum_backbone(x_1)
            header_input = HeaderInput(x[:, 0].unsqueeze(1), x_1.unsqueeze(1), x_idx, feature_0, feature_1)
        # All other cases are not currently supported
        else:
            raise NotImplementedError

        header_out = self.contrastive_header(header_input)
        # Unshuffle the second view in case of a momentum network
        if self.model_cfg.enable_backbone_momentum and x.shape[1] == 2:
            header_input, header_out = self.unshuffle_outputs(shuffle_idx, header_input, header_out)

        return ModelOutput(*header_input, *header_out)

    def compute_loss(self, model_output: ModelOutput) -> Tuple[Dict[str, float], float]:
        loss_meta = {}
        total_loss = 0.0

        if self.loss_cfg.ntxent_loss_active:
            assert model_output.prediction_1 is not None
            ntxent_loss = self.criterion["ntxent_loss"](model_output.prediction_0, model_output.prediction_1)
            total_loss += ntxent_loss
            loss_meta["ntxent_loss"] = ntxent_loss.item()
        if self.loss_cfg.kl_loss_active:
            assert model_output.distribution_data is not None
            distr = model_output.distribution_data
            scale = torch.exp(distr.log_scale)
            logscale_prior = torch.log(distr.scale_prior)
            kl_loss = (distr.shift.abs() / distr.scale_prior) + logscale_prior - distr.log_scale - 1
            kl_loss += (scale / distr.scale_prior) * (-(distr.shift.abs() / scale)).exp()
            kl_loss = kl_loss.sum(dim=-1).mean()
            total_loss += self.loss_cfg.kl_loss_weight * kl_loss
            loss_meta["kl_loss"] = kl_loss.item()
        if self.loss_cfg.transop_loss_active:
            assert self.model_cfg.header_cfg.header_name == "TransOp"
            if model_output.prediction_0_feat_detached is None or model_output.prediction_1_feat_detached is None:
                z0, z1 = model_output.prediction_0, model_output.prediction_1
            else:
                z0, z1 = model_output.prediction_0_feat_detached, model_output.prediction_1_feat_detached

            transop_loss = F.mse_loss(z0, z1)
            if self.loss_cfg.transop_loss_weight > 0:
                total_loss += self.loss_cfg.transop_loss_weight * transop_loss
            loss_meta["transop_loss"] = transop_loss.item()

        return loss_meta, total_loss

    def update_momentum_network(self, model_out: ModelOutput) -> None:
        if self.model_cfg.enable_backbone_momentum:
            update_momentum(self.backbone, self.momentum_backbone, self.model_cfg.backbone_momentum_update_rate)
        if self.model_cfg.enable_header_momentum:
            self.contrastive_header.update_momentum_network(self.model_cfg.header_momentum_update_rate, model_out)

    def get_param_groups(self):
        return [{"params": self.backbone.parameters()}] + self.contrastive_header.get_param_groups()

    @staticmethod
    def initialize_model(model_cfg: ModelConfig, devices: List[int]) -> "Model":
        backbone, backbone_feature_dim = Backbone.initialize_backbone(model_cfg.backbone_cfg)
        contrastive_header = ContrastiveHeader.initialize_header(
            model_cfg.header_cfg,
            backbone_feature_dim,
            model_cfg.enable_header_momentum,
        )
        model = Model(model_cfg, backbone, contrastive_header)
        model = model.to(devices[0])
        if len(devices) > 1:
            model = torch.nn.DataParallel(model, device_ids=devices)
        return model
