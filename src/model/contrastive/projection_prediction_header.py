"""
Header that performs both a projection and a prediction header. The projection step replaces
the features from the backbone when the model is training.

Example models:
NNCLR
BYOL
SimSiam

@Filename    projection_prediction_header.py
@Author      Kion
@Created     09/26/22
"""

import torch.nn as nn
from lightly.models.modules import (
    NNCLRPredictionHead,
    NNCLRProjectionHead,
    NNMemoryBankModule,
)
from model.contrastive.config import ProjectionPredictionHeaderConfig
from model.type import HeaderInput, HeaderOutput


class ProjectionPredictionHeader(nn.Module):
    def __init__(
        self,
        proj_cfg: ProjectionPredictionHeaderConfig,
        backbone_feature_dim: int,
    ):
        super(ProjectionPredictionHeader, self).__init__()
        self.proj_cfg = proj_cfg
        self.projector = None
        if self.proj_cfg.header_name == "NNCLR":
            self.projector = NNCLRProjectionHead(
                backbone_feature_dim,
                self.proj_cfg.proj_hidden_dim,
                self.proj_cfg.proj_output_dim,
            )
        else:
            raise NotImplementedError

        self.predictor = None
        if self.proj_cfg.prediction_type == "MLP":
            if self.proj_cfg.header_name == "NNCLR":
                self.predictor = NNCLRPredictionHead(
                    self.proj_cfg.proj_output_dim,
                    self.proj_cfg.pred_hidden_dim,
                    self.proj_cfg.pred_output_dim,
                )
            else:
                raise NotImplementedError
        elif self.proj_cfg.prediction_type == "Linear":
            self.predictor = nn.Linear(
                self.proj_cfg.proj_output_dim, self.proj_cfg.pred_output_dim, bias=False
            )
        elif self.proj_cfg.prediction_type == "None":
            self.predictor = nn.Identity()
        else:
            raise NotImplementedError

        self.nn_memory_bank = None
        if self.proj_cfg.enable_nn_bank:
            self.nn_memory_bank = NNMemoryBankModule(size=self.proj_cfg.nn_memory_bank_size)

    def update_momentum_network(self, momentum_rate: float) -> None:
        raise NotImplementedError

    def forward(self, header_input: HeaderInput) -> HeaderOutput:
        header_out = {}
        z0 = self.projector(header_input.feature_0)
        z1 = self.projector(header_input.feature_1)
        p0 = self.predictor(z0)
        p1 = self.predictor(z1)

        if self.proj_cfg.enable_nn_bank:
            z0 = self.nn_memory_bank(z0.detach(), update=False).detach()
            z1 = self.nn_memory_bank(z1.detach(), update=True).detach()

        header_out['proj_00'] = z0
        header_out['proj_01'] = p1
        header_out['proj_10'] = z1
        header_out['proj_11'] = p0

        return HeaderOutput(header_out, distribution_data=None)
