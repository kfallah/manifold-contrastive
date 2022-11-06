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
from model.contrastive.transop_header import TransportOperatorHeader
from model.manifold.reparameterize import compute_kl
from model.public.ntx_ent_loss import NTXentLoss
from model.type import ModelOutput


class Loss(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super(Loss, self).__init__()
        self.model_cfg = model_cfg
        self.loss_cfg = model_cfg.loss_cfg
        self.kl_warmup = 0

        self.criterion = {}
        if self.loss_cfg.ntxent_loss_active:
            self.criterion["ntxent_loss"] = NTXentLoss(
                memory_bank_size=self.loss_cfg.memory_bank_size,
                temperature=self.loss_cfg.ntxent_temp,
                normalize=self.loss_cfg.ntxent_normalize,
                loss_type=self.loss_cfg.ntxent_logit,
                detach_off_logit=self.loss_cfg.ntxent_detach_off_logit,
            )

    def get_kl_weight(self, curr_iter: int):
        if self.loss_cfg.kl_weight_warmup == "None":
            return self.loss_cfg.kl_loss_weight
        elif self.loss_cfg.kl_weight_warmup == "Linear":
            self.kl_warmup += 1e-5
            if self.kl_warmup > 1.0:
                self.kl_warmup = 1.0
            return self.loss_cfg.kl_loss_weight * self.kl_warmup
        else:
            raise NotImplementedError()

    def compute_loss(
        self, curr_iter: int, model_output: ModelOutput, args_dict
    ) -> Tuple[Dict[str, float], float]:
        loss_meta = {}
        total_loss = 0.0

        if self.loss_cfg.ntxent_loss_active:
            assert model_output.prediction_1 is not None

            if self.loss_cfg.ntxent_symmetric:
                ntxent_loss = 0.5 * (
                    self.criterion["ntxent_loss"](
                        model_output.projection_0, model_output.prediction_1
                    )
                    + self.criterion["ntxent_loss"](
                        model_output.prediction_0, model_output.projection_1
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
            kl_weight = self.get_kl_weight(curr_iter)
            total_loss += kl_weight * kl_loss
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
            if self.model_cfg.header_cfg.enable_splicing:
                z1_hat = TransportOperatorHeader.splice_input(z1_hat, self.model_cfg.header_cfg.splice_dim)
                z1 = TransportOperatorHeader.splice_input(z1, self.model_cfg.header_cfg.splice_dim)
            c = model_output.distribution_data.samples
            # Only take loss over values where transport significantly occured to prevent feature collapse
            weight = ~(c.abs() < 1e-2).all(dim=-1)
            transop_loss = (weight.unsqueeze(-1) * F.mse_loss(z1_hat, z1, reduction='none')).mean()
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
