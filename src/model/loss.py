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

from model.config import ModelConfig
from model.manifold.reparameterize import compute_kl
from model.public.ntx_ent_loss import contrastive_loss, lie_nt_xent_loss
from model.type import ModelOutput


class Loss(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super(Loss, self).__init__()
        self.model_cfg = model_cfg
        self.loss_cfg = model_cfg.loss_cfg
        self.kl_warmup = 0

    def get_kl_weight(self, curr_iter: int):
        if self.loss_cfg.kl_weight_warmup == "None":
            return self.loss_cfg.kl_loss_weight
        elif self.loss_cfg.kl_weight_warmup == "Linear":
            self.kl_warmup += 5e-5
            if self.kl_warmup > 1.0:
                self.kl_warmup = 1.0
            return self.loss_cfg.kl_loss_weight * self.kl_warmup
        elif self.loss_cfg.kl_weight_warmup == "Exponential":
            self.kl_warmup += 1
            if self.kl_warmup > 5000:
                self.kl_warmup = 5000
            ratio = self.kl_warmup / 5000
            kl_weight = (5e-3) ** (1 - ratio)
            return self.loss_cfg.kl_loss_weight * kl_weight
        elif self.loss_cfg.kl_weight_warmup == "Cyclic":
            mod_iter = curr_iter % 50000
            weight = 1.0
            if mod_iter >= 40000:
                weight = 1 - ((mod_iter - 40000) / 10000)
            return weight * self.loss_cfg.kl_loss_weight
        elif self.loss_cfg.kl_weight_warmup == "Annealed":
            self.kl_warmup += 1
            if self.kl_warmup > 100000:
                self.kl_warmup = 100000
            temp = (1 * (100000 - self.kl_warmup) + 0.001 * self.kl_warmup) / 100000
            return temp * self.loss_cfg.kl_loss_weight
        else:
            raise NotImplementedError()

    def compute_loss(self, curr_iter: int, model_output: ModelOutput, args_dict) -> Tuple[Dict[str, float], float]:
        loss_meta = {}
        header_out = model_output.header_output
        header_dict = header_out.header_dict
        total_loss = 0.0

        # Contrastive loss terms
        if self.loss_cfg.ntxent_loss_active:
            ntxent_loss = contrastive_loss(header_dict["proj_00"], header_dict["proj_01"], self.loss_cfg.ntxent_temp)
            total_loss += self.loss_cfg.ntxent_loss_weight * ntxent_loss
            loss_meta["ntxent_loss"] = ntxent_loss.item()

        # InfoNCE Lie Loss on transport operator estimates
        if self.loss_cfg.ntxent_lie_loss_active and curr_iter >= self.loss_cfg.ntxent_lie_loss_start_iter:
            # Contrast point pair, using additional negative found via feature augmentation
            z0, z1, zaug = model_output.header_input.feature_0, model_output.header_input.feature_1, header_dict["z0_aug"]

            if self.loss_cfg.ntxent_lie_loss_mse:
                lie_loss = lie_nt_xent_loss(
                    z0, z1, zaug,
                    mse=True,
                    temperature=self.loss_cfg.ntxent_lie_temp,
                )
            else:
                lie_loss = lie_nt_xent_loss(
                    F.normalize(z0, dim=-1),
                    F.normalize(z1, dim=-1),
                    F.normalize(zaug, dim=-1),
                    temperature=self.loss_cfg.ntxent_lie_temp,
                )
            loss_meta["ntxent_lie_loss"] = lie_loss.item()
            total_loss += self.loss_cfg.ntxent_lie_loss_weight * lie_loss

        # Transport operator loss terms
        if self.loss_cfg.transop_loss_active:
            z1_hat, z1 = header_dict["transop_z1hat"], header_dict["transop_z1"]
            transop_loss = F.mse_loss(z1_hat, z1.detach())
            loss_meta["transop_loss"] = transop_loss.item()
            total_loss += self.loss_cfg.transop_loss_weight * transop_loss

        if self.loss_cfg.c_refine_loss_active:
            c_vi = header_out.distribution_data.encoder_params["shift"]
            c_loss = F.mse_loss(c_vi, header_out.distribution_data.samples.detach())
            loss_meta["c_pred"] = self.loss_cfg.c_refine_loss_weight * c_loss.item()

        if self.loss_cfg.enable_shift_l2:
            enc_shift = header_out.distribution_data.encoder_params["shift"]
            enc_shift_l2 = (enc_shift**2).sum(dim=-1).mean()
            total_loss += self.loss_cfg.shift_l2_weight * enc_shift_l2
            loss_meta["shift_l2"] = enc_shift_l2.item()
            if self.loss_cfg.enable_prior_shift_l2:
                prior_shift = header_out.distribution_data.prior_params["shift"]
                prior_shift_l2 = (prior_shift**2).sum(dim=-1).mean()
                total_loss += self.loss_cfg.shift_l2_weight * prior_shift_l2
                loss_meta["prior_shift_l2"] = prior_shift_l2.item()

        if self.loss_cfg.real_eig_reg_active:
            assert "psi" in args_dict.keys()
            psi_use = args_dict["psi"]
            psi_use = psi_use.reshape(-1, psi_use.shape[-1], psi_use.shape[-1])
            psi_use = psi_use[torch.randperm(len(psi_use))[:10]]
            eig_loss = (torch.real(torch.linalg.eigvals(psi_use)) ** 2).sum()
            loss_meta["real_eig_loss"] = eig_loss.item()
            total_loss += self.loss_cfg.real_eig_reg_weight * eig_loss

        # Variational Inference loss terms
        if self.loss_cfg.kl_loss_active:
            assert header_out.distribution_data is not None
            kl_loss = compute_kl(
                self.model_cfg.header_cfg.transop_header_cfg.vi_cfg.distribution,
                header_out.distribution_data.encoder_params,
                header_out.distribution_data.prior_params,
                self.loss_cfg.kl_detach_shift,
            ).mean()
            kl_weight = self.get_kl_weight(curr_iter)
            total_loss += kl_weight * kl_loss
            loss_meta["kl_loss"] = kl_loss.item()

        return loss_meta, total_loss
