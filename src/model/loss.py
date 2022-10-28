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
from torch.cuda.amp import autocast

from model.config import ModelConfig
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

    def compute_loss(
        self, model_output: ModelOutput, args_dict
    ) -> Tuple[Dict[str, float], float]:
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
                self.model_cfg.header_cfg.vi_cfg.distribution,
                model_output.distribution_data.encoder_params,
                model_output.distribution_data.prior_params,
            )
            total_loss += self.loss_cfg.kl_loss_weight * kl_loss
            loss_meta["kl_loss"] = kl_loss.item()
        if self.loss_cfg.hyperkl_loss_active:
            assert model_output.distribution_data is not None
            hyperkl_loss = compute_kl(
                self.model_cfg.header_cfg.vi_cfg.distribution,
                model_output.distribution_data.prior_params,
                model_output.distribution_data.hyperprior_params,
            )
            total_loss += self.loss_cfg.hyperkl_loss_weight * hyperkl_loss
            loss_meta["hyperkl_loss"] = hyperkl_loss.item()
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
        if self.loss_cfg.real_eig_reg_active:
            assert "psi" in args_dict.keys()
            psi = args_dict["psi"]
            eig_loss = (torch.real(torch.linalg.eigvals(psi)) ** 2).sum()
            loss_meta["real_eig_loss"] = eig_loss.item()
            total_loss += self.loss_cfg.real_eig_reg_weight * eig_loss
        if self.loss_cfg.cyclic_reg_active:
            assert (
                model_output.projection_0 is not None
                and model_output.distribution_data is not None
                and "psi" in args_dict.keys()
            )
            z0 = model_output.projection_0
            c = model_output.distribution_data.samples
            psi = args_dict["psi"]
            with autocast(enabled=False):
                T = torch.matrix_exp(torch.einsum("bm,mpk->bpk", c * 5, psi))
            z0_extended = (T @ z0.unsqueeze(dim=-1)).squeeze(dim=-1)
            cyclic_loss = F.mse_loss(z0, z0_extended)
            loss_meta["cyclic_loss"] = cyclic_loss.item()
            total_loss += self.loss_cfg.cyclic_reg_weight * cyclic_loss

        return loss_meta, total_loss
