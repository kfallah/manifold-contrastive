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
from lightly.loss.vicreg_loss import VICRegLoss

from model.config import ModelConfig
from model.manifold.reparameterize import compute_kl
from model.public.ntx_ent_loss import NTXentLoss, lie_nt_xent_loss
from model.type import ModelOutput


class Loss(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super(Loss, self).__init__()
        self.model_cfg = model_cfg
        self.loss_cfg = model_cfg.loss_cfg
        self.kl_warmup = 0

        self.ntxent_loss = None
        if self.loss_cfg.ntxent_loss_active:
            self.ntxent_loss = NTXentLoss(
                memory_bank_size=self.loss_cfg.memory_bank_size,
                temperature=self.loss_cfg.ntxent_temp,
                normalize=self.loss_cfg.ntxent_normalize,
                loss_type=self.loss_cfg.ntxent_logit,
                detach_off_logit=self.loss_cfg.ntxent_detach_off_logit,
            )

        self.vicreg_loss = None
        if self.loss_cfg.vicreg_loss_active:
            self.vicreg_loss = VICRegLoss(
                lambda_param=self.loss_cfg.vicreg_inv_weight,
                mu_param=self.loss_cfg.vicreg_var_weight,
                nu_param=self.loss_cfg.vicreg_cov_weight,
            )

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
            if self.kl_warmup > 10000:
                self.kl_warmup = 10000
            ratio = self.kl_warmup / 10000
            kl_weight = (5e-3) ** (1 - ratio)
            return self.loss_cfg.kl_loss_weight * kl_weight
        elif self.loss_cfg.kl_weight_warmup == "Cyclic":
            total_iters = 100000
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
            ntxent_loss = self.ntxent_loss.nt_xent(header_dict["proj_00"], header_dict["proj_01"])
            if self.loss_cfg.ntxent_symmetric:
                ntxent_loss = 0.5 * (
                    ntxent_loss + self.ntxent_loss.nt_xent(header_dict["proj_10"], header_dict["proj_11"])
                )
            total_loss += self.loss_cfg.ntxent_loss_weight * ntxent_loss
            loss_meta["ntxent_loss"] = ntxent_loss.item()

        # Self-supervised loss from vicreg
        if self.loss_cfg.vicreg_loss_active:
            z0, z1 = header_dict["proj_00"], header_dict["proj_01"]
            vicreg_loss = self.vicreg_loss(z0, z1)
            total_loss += self.loss_cfg.vicreg_loss_weight * vicreg_loss
            loss_meta["vicreg_loss"] = vicreg_loss.item()

        # InfoNCE Lie Loss on transport operator estimates
        if self.loss_cfg.ntxent_lie_loss_active and curr_iter > self.loss_cfg.ntxent_lie_loss_start_iter:
            z0, z1, z1hat = header_dict["transop_z0"], header_dict["transop_z1"], header_dict["transop_z1"]
            z0, z1, z1hat = z1.reshape(len(z1), -1), z1_hat.reshape(len(z1), -1), z0.reshape(len(z1), -1)
            proj = args_dict["proj"]
            z0_proj, z1_proj, z1hat_proj= proj(z0), proj(z1), proj(z1hat)
            lie_loss = lie_nt_xent_loss(
                F.normalize(z0_proj, dim=-1),
                F.normalize(z1_proj, dim=-1),
                F.normalize(z1hat_proj, dim=-1),
                temperature=self.loss_cfg.ntxent_lie_temp
            )
            loss_meta["infonce_lie_loss"] = lie_loss.item()     
            total_loss += self.loss_cfg.ntxent_lie_loss_weight * lie_loss

        # Transport operator loss terms
        if self.loss_cfg.transop_loss_active:
            z1_hat, z1 = header_dict["transop_z1hat"], header_dict["transop_z1"]
            if self.loss_cfg.transop_loss_fn == "mse":
                transop_loss = F.mse_loss(z1_hat, z1.detach(), reduction="none").mean()
                if self.loss_cfg.transop_symmetric:
                    z0_hat, z0 = header_dict["transop_z0hat"], header_dict["transop_z0"]
                    transop_loss = 0.5*transop_loss + 0.5*F.mse_loss(z0_hat, z0.detach(), reduction="none").mean()
                loss_meta["transop_loss"] = transop_loss.item()     
            elif self.loss_cfg.transop_loss_fn == "cos":
                transop_loss = (-(F.cosine_similarity(z1, z1_hat, dim=-1)**2)).exp().mean()
                loss_meta["transop_cos"] = transop_loss.item()      
            elif self.loss_cfg.transop_loss_fn == "huber":
                transop_loss = F.huber_loss(z1_hat, z1, reduction="none", delta=0.5).mean()
                loss_meta["transop_huber"] = transop_loss.item()
            elif self.loss_cfg.transop_loss_fn == "ratio":
                z0 = header_dict["transop_z0"]
                transop_loss = F.mse_loss(z1_hat, z1.detach(), reduction="none").mean(dim=-1)
                loss_meta["transop_loss"] = transop_loss.mean().item()
                transop_loss /= F.mse_loss(z0, z1.detach(), reduction="none").mean(dim=-1) + 1.0e-4
                transop_loss = transop_loss.mean()
                loss_meta["transop_ratio"] = transop_loss.item()
            elif self.loss_cfg.transop_loss_fn == "diff":
                z0 = header_dict["transop_z0"]
                transop_loss = F.mse_loss(z1_hat, z1, reduction="none").mean(dim=-1)
                loss_meta["transop_loss"] = transop_loss.mean().item()
                transop_loss -= 0.01 * F.mse_loss(z0, z1, reduction="none").mean(dim=-1)
                transop_loss = transop_loss.mean()
                loss_meta["transop_diff"] = transop_loss.item()
            else:
                raise NotImplementedError
            total_loss += self.loss_cfg.transop_loss_weight * transop_loss

        if self.loss_cfg.c_refine_loss_active:
            c_vi = header_out.distribution_data.encoder_params["shift"]
            c_loss = F.mse_loss(c_vi, header_out.distribution_data.samples.detach())
            loss_meta["c_pred"] = self.loss_cfg.c_refine_loss_weight * c_loss.item()

        if self.loss_cfg.c_l2_active:
            c = header_out.distribution_data.samples
            c_l2 = (c**2).mean()
            loss_meta["c_l2"] = self.loss_cfg.c_l2_weight * c_l2

        if self.loss_cfg.enable_shift_l2:
            enc_shift = header_out.distribution_data.encoder_params["shift"]
            enc_shift_l2 = (enc_shift**2).sum(dim=-1).mean()
            total_loss += self.loss_cfg.shift_l2_weight * enc_shift_l2
            loss_meta["shift_l2"] = enc_shift_l2.item()

        if self.loss_cfg.det_prior_loss_active:
            z1_det_hat, z1 = header_dict["transop_z1_det_hat"], header_dict["transop_z1"].detach()
            det_transop_loss = F.mse_loss(z1_det_hat, z1, reduction="none").mean()
            total_loss += self.loss_cfg.transop_loss_weight * det_transop_loss
            loss_meta["det_transop_loss"] = det_transop_loss.item()

            prior_shift = header_out.distribution_data.prior_params["shift"]
            det_shift_l2 = (prior_shift**2).sum(dim=-1).mean()
            total_loss += self.loss_cfg.det_prior_l2_weight * det_shift_l2
            loss_meta["det_l2_loss"] = det_shift_l2.item()

        if self.loss_cfg.real_eig_reg_active:
            assert "psi" in args_dict.keys()
            psi = args_dict["psi"]
            eig_loss = (torch.real(torch.linalg.eigvals(psi)) ** 2).sum()
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
