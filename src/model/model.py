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
from lightly.models.utils import batch_shuffle, batch_unshuffle, deactivate_requires_grad, update_momentum

from model.backbone import Backbone
from model.config import ModelConfig
from model.contrastive.header import ContrastiveHeader
from model.loss import Loss
from model.type import HeaderInput, HeaderOutput, ModelOutput


class Model(nn.Module):
    def __init__(self, model_cfg: ModelConfig, backbone, contrastive_header, backbone_feat_dim):
        super(Model, self).__init__()
        self.model_cfg = model_cfg
        self.loss = Loss(model_cfg)
        self.backbone = backbone
        self.contrastive_header = contrastive_header

        self.momentum_backbone = None
        if self.model_cfg.enable_backbone_momentum:
            self.momentum_backbone = copy.deepcopy(self.backbone)
            deactivate_requires_grad(self.momentum_backbone)

    def unshuffle_outputs(
        self, shuffle_idx, header_input: HeaderInput, header_out: HeaderOutput
    ) -> Tuple[HeaderInput, HeaderOutput]:
        x_1 = batch_unshuffle(header_input.x_1, shuffle_idx)
        feature_1 = batch_unshuffle(header_input.feature_1, shuffle_idx)
        header_input = HeaderInput(
            header_input.curr_iter,
            header_input.x_0,
            x_1,
            header_input.x_idx,
            header_input.feature_0,
            feature_1,
        )
        header_out = self.contrastive_header.unshuffle_outputs(shuffle_idx, header_out)
        return header_input, header_out

    def forward(self, x: torch.Tensor, x_idx: List, curr_iter: int) -> ModelOutput:
        """Take input image and get prediction from contrastive header.

        Args:
            x_list (torch.Tensor): V x [B x H x W x C] List of Tensors where each entry is a batch of different views
                                    of the image. V=1 returns features only, V=2 performs contrastive and manifold
                                    headers.
            x_idx (torch.Tensor): [B] Indices of each entry in the batch.

        Returns:
            Returns the encoded features from the backbone and the prediction provided by the header.
        """
        # Only a single view of an image is used, for evaluation purposes
        if x.shape[1] == 1:
            feature_0 = self.backbone(x[:, 0])
            header_input = HeaderInput(curr_iter, x, None, x_idx, feature_0, None)
            return ModelOutput(header_input, None)
        # Two views of an image are provided
        elif x.shape[1] == 2:
            if not self.model_cfg.enable_backbone_momentum:
                # No momentum encoder used for the backbone
                feature_0, feature_1 = self.backbone(x[:, 0]), self.backbone(x[:, 1])
                x_1 = x[:, 1]
            else:
                # Use momentum encoder for the positive pair
                feature_0 = self.backbone(x[:, 0])
                x_1, shuffle_idx = batch_shuffle(x[:, 1])
                feature_1 = self.momentum_backbone(x_1)
            header_input = HeaderInput(curr_iter, x[:, 0], x_1, x_idx, feature_0, feature_1)
        # All other cases are not currently supported
        else:
            raise NotImplementedError

        header_out = self.contrastive_header(header_input)
        # Unshuffle the second view in case of a momentum network
        if self.model_cfg.enable_backbone_momentum:
            header_input, header_out = self.unshuffle_outputs(shuffle_idx, header_input, header_out)

        return ModelOutput(header_input, header_out)

    def compute_loss(self, curr_idx: int, model_output: ModelOutput) -> Tuple[Dict[str, float], float]:
        args_dict = {}
        if self.model_cfg.loss_cfg.real_eig_reg_active:
            args_dict["psi"] = self.contrastive_header.transop_header.transop.psi
        return self.loss.compute_loss(curr_idx, model_output, args_dict)

    def update_momentum_network(self) -> None:
        if self.model_cfg.enable_backbone_momentum:
            update_momentum(
                self.backbone,
                self.momentum_backbone,
                self.model_cfg.backbone_momentum_update_rate,
            )
        if self.model_cfg.enable_header_momentum:
            self.contrastive_header.update_momentum_network(self.model_cfg.header_momentum_update_rate)

    def get_param_groups(self):
        return [{"params": self.backbone.parameters()}] + self.contrastive_header.get_param_groups()

    @staticmethod
    def initialize_model(model_cfg: ModelConfig, dataset_name: str, devices: List[int]) -> "Model":
        backbone, backbone_feature_dim = Backbone.initialize_backbone(model_cfg.backbone_cfg, dataset_name)
        contrastive_header = ContrastiveHeader.initialize_header(
            model_cfg.header_cfg,
            backbone_feature_dim,
            model_cfg.enable_header_momentum,
        )
        if model_cfg.backbone_cfg.load_backbone:
            contrastive_header.load_model_state(model_cfg.backbone_cfg.load_backbone)
        model = Model(model_cfg, backbone, contrastive_header, backbone_feature_dim)
        model = model.to(devices[0])
        if len(devices) > 1:
            model = torch.nn.DataParallel(model, device_ids=devices)
        return model
