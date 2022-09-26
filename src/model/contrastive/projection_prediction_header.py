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
                backbone_feature_dim, self.proj_cfg.proj_hidden_dim, self.proj_cfg.proj_output_dim
            )
        else:
            raise NotImplementedError

        self.predictor = None
        if self.proj_cfg.prediction_type == "MLP":
            if self.proj_cfg.header_name == "NNCLR":
                self.predictor = NNCLRPredictionHead(
                    self.proj_cfg.proj_output_dim, self.proj_cfg.pred_hidden_dim, self.proj_cfg.pred_output_dim
                )
            else:
                raise NotImplementedError
        elif self.proj_cfg.projection_type == "Linear":
            self.predictor = nn.Linear(self.proj_cfg.proj_output_dim, self.proj_cfg.pred_output_dim, bias=False)
        elif self.proj_cfg.projection_type == "None":
            self.predictor = nn.Identity()
        else:
            raise NotImplementedError

        self.nn_memory_bank = None
        if self.proj_cfg.enable_nn_bank:
            self.nn_memory_bank = NNMemoryBankModule(size=self.proj_cfg.nn_memory_bank_size)

    def update_momentum_network(self, momentum_rate: float) -> None:
        raise NotImplementedError

    def forward(self, header_input: HeaderInput) -> HeaderOutput:
        z0 = self.projector(header_input.feature_0)
        p0 = self.predictor(z0)
        z1, p1 = None, None
        if header_input.feature_1 is not None:
            z1 = self.projector(header_input.feature_1)
            p1 = self.predictor(z1)
        if not self.training or header_input.feature_1 is None:
            return HeaderOutput(header_input.feature_0, header_input.feature_1, p0, p1, distribution_data=None)

        if self.proj_cfg.enable_nn_bank:
            z0 = self.nn_memory_bank(z0, update=False)
            z1 = self.nn_memory_bank(z1, update=True)

        return HeaderOutput(z0, z1, p0, p1, distribution_data=None)
