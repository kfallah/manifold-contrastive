"""
Wrapper that handles all computations related to computing loss functions.

@Filename    loss.py
@Author      Kion
@Created     10/03/22
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import LossConfig, ModelConfig
from model.manifold.reparameterize import compute_kl
from model.public.ntx_ent_loss import NTXentLoss
from model.type import ModelOutput


class Loss(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super(Loss, self).__init__()
        self.model_cfg = model_cfg
        self.loss_cfg = model_cfg.loss_cfg

        self.criterion = {}
        if self.loss_cfg.ntxent_loss_active:
            self.criterion["ntxent_loss"] = NTXentLoss(
                memory_bank_size=self.loss_cfg.memory_bank_size,
                temperature=self.loss_cfg.ntxent_temp,
                normalize=self.loss_cfg.ntxent_normalize,
                loss_type=self.loss_cfg.ntxent_logit,
                detach_off_logit=self.loss_cfg.ntxent_detach_off_logit,
            )

    def compute_loss(self, model_output: ModelOutput) -> Tuple[Dict[str, float], float]:
        loss_meta = {}
        total_loss = 0.0

        if self.loss_cfg.ntxent_loss_active:
            assert model_output.prediction_1 is not None

            if self.loss_cfg.ntxent_symmetric:
                ntxent_loss = 0.5 * (
                    self.criterion["ntxent_loss"](
                        model_output.feature_0, model_output.prediction_1
                    )
                    + self.criterion["ntxent_loss"](
                        model_output.feature_1, model_output.prediction_0
                    )
                )
            else:
                ntxent_loss = self.criterion["ntxent_loss"](
                    model_output.prediction_0, model_output.prediction_1
                )
            total_loss += ntxent_loss
            loss_meta["ntxent_loss"] = ntxent_loss.item()
        if self.loss_cfg.kl_loss_active:
            assert model_output.distribution_data is not None
            kl_loss = compute_kl(
                self.model_cfg.header_cfg.variational_inference_config.distribution,
                model_output.distribution_data.encoder_params,
                model_output.distribution_data.prior_params,
            )
            total_loss += self.loss_cfg.kl_loss_weight * kl_loss
            loss_meta["kl_loss"] = kl_loss.item()
        if self.loss_cfg.transop_loss_active:
            assert (
                model_output.prediction_0 is not None
                and model_output.prediction_1 is not None
            )
            z1_hat, z1 = model_output.prediction_0, model_output.prediction_1

            transop_loss = F.mse_loss(z1_hat, z1)
            if self.loss_cfg.transop_loss_weight > 0:
                total_loss += self.loss_cfg.transop_loss_weight * transop_loss
            loss_meta["transop_loss"] = transop_loss.item()

        return loss_meta, total_loss
