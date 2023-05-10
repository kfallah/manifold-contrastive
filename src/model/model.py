"""
Main model wrapper that initializes the model used for training.

@Filename    model.py
@Author      Kion
@Created     08/31/22
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

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
        self.backbone_feat_dim = backbone_feat_dim
        self.contrastive_header = contrastive_header

    def forward(self, x: torch.Tensor, curr_iter: int, nn_queue: nn.Module = None) -> ModelOutput:
        """Take input image and get prediction from contrastive header.

        Args:
            x_list (torch.Tensor): V x [B x H x W x C] List of Tensors where each entry is a batch of different views
                                    of the image. V=1 returns features only, V=2 performs contrastive and manifold
                                    headers.

        Returns:
            Returns the encoded features from the backbone and the prediction provided by the header.
        """
        # Only a single view of an image is used, for evaluation purposes
        if x.shape[1] == 1:
            feature_0 = self.backbone(x[:, 0])
            header_input = HeaderInput(curr_iter, x, None, feature_0, None, None)
            return ModelOutput(header_input, None)
        # Two or three views of an image are provided
        else:
            feature_0, feature_1 = self.backbone(x[:, 0]), self.backbone(x[:, 1])
            feature_2 = None
            if x.shape[1] == 3:
                feature_2 = self.backbone(x[:, 2])
            header_input = HeaderInput(curr_iter, x[:, 0], x[:, 1], feature_0, feature_1, feature_2)

        header_out = self.contrastive_header(header_input, nn_queue)

        return ModelOutput(header_input, header_out)

    def compute_loss(self, curr_idx: int, model_output: ModelOutput) -> Tuple[Dict[str, float], float]:
        args_dict = {}
        if self.model_cfg.loss_cfg.real_eig_reg_active:
            args_dict["psi"] = self.contrastive_header.transop_header.transop.psi
        if self.model_cfg.header_cfg.enable_projection_header:
            args_dict["proj"] = self.contrastive_header.projection_header.projector
        return self.loss.compute_loss(curr_idx, model_output, args_dict)

    def get_param_groups(self):
        return [{"params": self.backbone.parameters()}] + self.contrastive_header.get_param_groups()

    @staticmethod
    def initialize_model(model_cfg: ModelConfig, dataset_name: str, devices: List[int]) -> "Model":
        backbone, backbone_feature_dim = Backbone.initialize_backbone(model_cfg.backbone_cfg, dataset_name)
        contrastive_header = ContrastiveHeader.initialize_header(
            model_cfg.header_cfg,
            backbone_feature_dim,
        )
        if model_cfg.backbone_cfg.load_backbone:
            contrastive_header.load_model_state(model_cfg.backbone_cfg.load_backbone)
        model = Model(model_cfg, backbone, contrastive_header, backbone_feature_dim)
        model = model.to(devices[0])
        if len(devices) > 1:
            model = torch.nn.DataParallel(model, device_ids=devices)
        return model
