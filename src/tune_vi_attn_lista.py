import logging
import os
import time
import warnings
from dataclasses import MISSING, dataclass

import hydra
import numpy as np
import omegaconf
import torch
import torch.distributions as distr
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import wandb
from dataloader.contrastive_dataloader import get_dataloader
from model.manifold.l1_inference import infer_coefficients
from model.manifold.transop import TransOp_expm
from model.model import Model
from model.public.linear_warmup_cos_anneal import LinearWarmupCosineAnnealingLR

warnings.filterwarnings("ignore")


@dataclass
class AttentionConfig:
    feature_dim: int = 512
    num_heads: int = 8
    psi_hidden_dim: int = 2048
    ff_hidden_dim: int = 2048
    residual_hidden_dim: int = 2048
    dropout: float = 0.1


@dataclass
class CoeffEncoderConfig:
    variational: bool = True

    use_fista: bool = True
    num_fista_iter: int = 50

    use_attn_lista: bool = False
    attention_cfg: AttentionConfig = AttentionConfig()

    feature_dim: int = 512
    hidden_dim1: int = 2048
    hidden_dim2: int = 2048
    scale_prior: float = 0.02
    shift_prior: float = 0.0
    threshold: float = 0.032

    logging_interval: int = 500
    lr: float = 0.01
    opt_type: str = "SGD"
    batch_size: int = 128
    kl_weight: float = 1.0e-5
    weight_decay: float = 1.0e-5
    grad_acc_iter: int = 1

    enable_c_l2: bool = False
    c_l2_weight: float = 1.0e-3

    enable_shift_l2: bool = False
    shift_l2_weight: float = 1.0e-2

    enable_pred_loss: bool = False
    pred_loss_weight: float = 10.0

    enable_max_sample: bool = True
    max_sample_l1_penalty: float = 1.0e-3
    total_num_samples: int = 20
    samples_per_iter: int = 20

    deterministic_pretrain: bool = False
    num_det_pretrain_iters: int = 20000

    learn_prior: bool = False
    no_shift_prior: bool = False


@dataclass
class TransportOperatorConfig:
    train_transop: bool = False
    start_iter: int = 1000
    dict_size: int = 100

    lr: float = 0.1
    weight_decay: float = 1.0e-6

    init_real_range: float = 1.0e-4
    init_imag_range: float = 5.0


@dataclass
class ExperimentConfig:
    exp_name: str = MISSING
    exp_dir: str = MISSING
    enable_wandb: bool = False
    seed: int = 0

    num_epochs: int = 300
    run_dir: str = "../../results/Transop_sep-loss/"
    device: str = "cuda:0"

    vi_cfg: CoeffEncoderConfig = CoeffEncoderConfig()
    transop_cfg: TransportOperatorConfig = TransportOperatorConfig()


class SparseCodeAttn(nn.Module):
    def __init__(self, cfg: AttentionConfig, dict_size: int):
        super(SparseCodeAttn, self).__init__()
        self.cfg = cfg
        self.dict_size = dict_size

        self.psi_extract = nn.Sequential(
            nn.Linear(64**2, cfg.psi_hidden_dim),
            nn.LayerNorm(cfg.psi_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.psi_hidden_dim, cfg.feature_dim),
        )
        self.residual_extract = nn.Sequential(
            nn.Linear(64, cfg.residual_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.residual_hidden_dim, cfg.feature_dim),
        )

        self.attn = nn.MultiheadAttention(
            cfg.feature_dim, num_heads=cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.pre_norm = nn.LayerNorm(cfg.feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.ff_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_hidden_dim, cfg.feature_dim),
        )
        self.post_norm = nn.LayerNorm(cfg.feature_dim)
        self.c_proj = nn.Linear(cfg.feature_dim, dict_size)

    def forward(self, r, psi):
        # compute residual
        r_feat = self.residual_extract(r)
        # extract features for operators
        psi_feat = self.psi_extract(psi.reshape(self.dict_size, -1))
        psi_feat = psi_feat.unsqueeze(0).repeat(len(r), 1, 1)
        # Run through attention layer
        r_attn, _ = self.attn(r_feat, psi_feat, psi_feat)
        r_feat = self.pre_norm(r_feat + r_attn)
        r_mlp = self.mlp(r_feat)
        r_feat = self.post_norm(r_mlp + r_mlp)
        return self.c_proj(r_feat)


class VIEncoder(nn.Module):
    def __init__(self, cfg: CoeffEncoderConfig, dict_size: int):
        super(VIEncoder, self).__init__()
        self.feat_extract = nn.Sequential(
            nn.Linear(64, cfg.hidden_dim1),
            nn.LeakyReLU(),
            nn.Linear(cfg.hidden_dim1, cfg.hidden_dim2),
            nn.LeakyReLU(),
            nn.Linear(cfg.hidden_dim2, cfg.feature_dim),
        )

        if cfg.use_attn_lista:
            self.attn1 = SparseCodeAttn(cfg.attention_cfg, dict_size)

        self.cfg = cfg
        self.dict_size = dict_size
        if cfg.variational:
            self.scale = nn.Linear(cfg.feature_dim, dict_size)
        self.shift = nn.Linear(cfg.feature_dim, dict_size)

    def soft_threshold(self, z: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(torch.abs(z) - lambda_) * torch.sign(z)

    def kl_loss(self, scale, shift, prior_scale=None, prior_shift=None):
        if prior_scale is None:
            prior_scale = torch.ones_like(scale) * self.cfg.scale_prior
        if prior_shift is None:
            prior_shift = torch.ones_like(shift) * self.cfg.shift_prior * torch.sign(shift).detach()
        if self.cfg.no_shift_prior:
            shift = shift.detach()
        encoder = distr.Laplace(shift, scale)
        prior = distr.Laplace(prior_shift, prior_scale)
        return distr.kl_divergence(encoder, prior)

    def reparameterize(self, noise, log_scale, shift):
        # Reparameterize
        scale = torch.exp(log_scale)
        eps = -scale * torch.sign(noise) * torch.log((1.0 - 2.0 * torch.abs(noise)).clamp(min=1e-6, max=1e6))
        c = shift + eps

        # Threshold
        c_thresh = self.soft_threshold(eps.detach(), self.cfg.threshold)
        non_zero = torch.nonzero(c_thresh, as_tuple=True)
        c_thresh[non_zero] = shift[non_zero].detach() + c_thresh[non_zero]
        c = c + c_thresh - c.detach()
        return c

    def log_distr(self, curr_iter, scale, shift):
        distr = {
            "distr/avg_enc_scale": scale.mean(),
            "distr/min_enc_scale": scale.min(),
            "distr/max_enc_scale": scale.max(),
            "distr/avg_enc_shift": shift.abs().mean(),
            "distr/min_enc_shift": shift.abs().min(),
            "distr/max_enc_shift": shift.abs().max(),
        }
        wandb.log(distr, step=curr_iter)

    def sample(self, x, curr_iter=0):
        z = self.feat_extract(x)

        if not self.cfg.variational:
            shift = self.shift(z)
            return shift, torch.tensor(0.0), (torch.zeros_like(shift), shift)

        log_scale, shift = self.scale(z), self.shift(z)
        log_scale += torch.log(torch.ones_like(log_scale) * self.cfg.scale_prior)
        log_scale, shift = log_scale.clamp(min=-20, max=2), shift.clamp(min=-5, max=5)

        # Reparameterization
        noise = torch.rand_like(log_scale) - 0.5
        c = self.reparameterize(noise, log_scale, shift)

        # KL loss
        kl = self.kl_loss(log_scale.exp(), shift, None, None).sum(dim=-1).mean()

        # Log distribution params
        if curr_iter % self.cfg.logging_interval == 0:
            self.log_distr(curr_iter, log_scale.exp(), shift)

        return c, (kl, log_scale, shift)

    def forward(self, x0, x1, psi, curr_iter=0):
        c, (kl, log_scale, shift) = self.sample(x0, curr_iter)

        # Refine distribution params with LISTA
        if self.cfg.use_attn_lista:
            c_final = c.clone()
            r_list = []
            for i in range(1):
                T = torch.matrix_exp(torch.einsum("bsm,mpk->bspk", c_final, psi))
                r = x1 - (T @ x0.unsqueeze(-1)).squeeze(-1)
                r_list.append((r**2).mean().item())
                c_update = self.attn1(r, psi)
                # c_final = self.soft_threshold(c_final + c_update, 0.1)

                # Straight through estimator
                c_final = c + c_update
                c_refine = self.soft_threshold(c_final.detach(), 0.2)
                c_final = c_final + c_refine - c_final.detach()

        if self.cfg.use_fista:
            x0_flat, x1_flat = x0.reshape(-1, x0.shape[-1]), x1.reshape(-1, x1.shape[-1])
            _, c_fista = infer_coefficients(
                x0_flat.float().detach(),
                x1_flat.float().detach(),
                psi.float(),
                0.1,
                max_iter=self.cfg.num_fista_iter,
                num_trials=1,
                lr=5e-2,
                decay=0.99,
                device=x0_flat.device,
                c_init=c.clone().detach().reshape(1, -1, self.dict_size),
            )
            c_final = c_fista.reshape(c.shape).detach()
            # c_final = c + (c_fista.reshape(c.shape) - c).detach()

        return c_final, kl, (log_scale, shift)


def log_transop(psi, curr_iter):
    psi_norm = (psi**2).sum(dim=(-1, -2))
    L = torch.linalg.eigvals(psi)
    eig_real = L.real
    eig_imag = L.imag

    wandb.log(
        {
            "psi/avg_fnorm": psi_norm.mean(),
            "psi/max_fnorm": psi_norm.max(),
            "psi/min_fnorm": psi_norm.min(),
            "psi/avg_real_eig": eig_real.abs().mean(),
            "psi/max_real_eig": eig_real.abs().max(),
            "psi/min_real_eig": eig_real.abs().min(),
            "psi/avg_imag_eig": eig_imag.abs().mean(),
            "psi/max_imag_eig": eig_imag.abs().max(),
            "psi/min_imag_eig": eig_imag.abs().min(),
        },
        step=curr_iter,
    )


def init_models(exp_cfg):
    current_checkpoint = 400
    device_idx = [0]

    # Set the default device
    default_device = torch.device(exp_cfg.device)
    # Load config
    cfg = omegaconf.OmegaConf.load(exp_cfg.run_dir + ".hydra/config.yaml")
    cfg.model_cfg.backbone_cfg.load_backbone = None

    # Load model
    model = Model.initialize_model(cfg.model_cfg, cfg.train_dataloader_cfg.dataset_cfg.dataset_name, device_idx)
    state_dict = torch.load(
        exp_cfg.run_dir + f"checkpoints/checkpoint_epoch{current_checkpoint}.pt", map_location=default_device
    )
    model.load_state_dict(state_dict["model_state"])
    # Manually override directory for dataloaders
    cfg.train_dataloader_cfg.dataset_cfg.dataset_dir = "../../datasets"
    cfg.train_dataloader_cfg.batch_size = exp_cfg.vi_cfg.batch_size
    # Load dataloaders
    _, train_dataloader = get_dataloader(cfg.train_dataloader_cfg)

    # Initialize models
    if exp_cfg.transop_cfg.train_transop:
        psi_init = TransOp_expm(
            exp_cfg.transop_cfg.dict_size,
            64,
            stable_init=True,
            real_range=exp_cfg.transop_cfg.init_real_range,
            imag_range=exp_cfg.transop_cfg.init_imag_range,
        ).to(default_device)
        psi = psi_init.psi
    else:
        psi = model.contrastive_header.transop_header.transop.get_psi()
    backbone = model.backbone.to(default_device)
    nn_bank = model.contrastive_header.transop_header.nn_memory_bank
    encoder = VIEncoder(exp_cfg.vi_cfg, exp_cfg.transop_cfg.dict_size).to(default_device)

    return train_dataloader, backbone, nn_bank, psi, encoder


def train(exp_cfg, train_dataloader, backbone, nn_bank, psi, encoder):
    param_list = [
        {
            "params": encoder.parameters(),
            "lr": exp_cfg.vi_cfg.lr,
            "eta_min": 1e-4,
            "weight_decay": exp_cfg.vi_cfg.weight_decay,
            "disable_layer_adaptation": True,
        },
    ]
    if exp_cfg.transop_cfg.train_transop:
        param_list.append(
            {
                "params": psi,
                "lr": exp_cfg.transop_cfg.lr,
                "eta_min": 1e-4,
                "weight_decay": exp_cfg.transop_cfg.weight_decay,
                "disable_layer_adaptation": True,
            }
        )
    if exp_cfg.vi_cfg.opt_type == "SGD":
        opt = torch.optim.SGD(
            param_list, lr=exp_cfg.vi_cfg.lr, nesterov=True, momentum=0.9, weight_decay=exp_cfg.vi_cfg.weight_decay
        )
    elif exp_cfg.vi_cfg.opt_type == "AdamW":
        opt = torch.optim.AdamW(param_list, lr=exp_cfg.vi_cfg.lr, weight_decay=exp_cfg.vi_cfg.weight_decay)
    else:
        raise NotImplementedError()
    iters_per_epoch = len(train_dataloader)
    scheduler = LinearWarmupCosineAnnealingLR(
        opt, warmup_epochs=20 * iters_per_epoch, max_epochs=iters_per_epoch * exp_cfg.num_epochs, eta_min=1e-4
    )

    transop_loss_save = []
    kl_loss_save = []
    dw_loss_save = []
    c_save = []
    iter_time = []
    for i in range(exp_cfg.num_epochs):
        for idx, batch in enumerate(train_dataloader):
            curr_iter = i * len(train_dataloader) + idx
            pre_time = time.time()
            x0, x1 = batch[0][0], batch[0][1]
            x0, x1 = x0.to(exp_cfg.device), x1.to(exp_cfg.device)

            with torch.no_grad():
                z0, z1 = backbone(x0), backbone(x1)
            z1 = nn_bank(z1.detach(), update=True).detach()

            z0 = torch.stack(torch.split(z0, 64, dim=-1)).transpose(0, 1)
            z1 = torch.stack(torch.split(z1, 64, dim=-1)).transpose(0, 1)

            c, kl_loss, (log_scale, shift) = encoder(z0, z1, psi.detach(), curr_iter)
            if exp_cfg.transop_cfg.train_transop and curr_iter < exp_cfg.transop_cfg.start_iter:
                psi_use = psi.detach()
            else:
                psi_use = psi
            T = torch.matrix_exp(torch.einsum("bsm,mpk->bspk", c, psi_use))
            z1_hat = (T @ z0.unsqueeze(-1)).squeeze(-1)
            transop_loss = torch.nn.functional.mse_loss(z1_hat, z1, reduction="none")

            loss = transop_loss.mean() + exp_cfg.vi_cfg.kl_weight * kl_loss
            if exp_cfg.vi_cfg.enable_c_l2:
                l2_reg = (c**2).sum(dim=-1).mean()
                loss += exp_cfg.vi_cfg.c_l2_weight * l2_reg
            if exp_cfg.vi_cfg.enable_shift_l2:
                l2_reg = (shift**2).sum(dim=-1).mean()
                loss += exp_cfg.vi_cfg.shift_l2_weight * l2_reg
            if exp_cfg.vi_cfg.enable_pred_loss:
                pred_loss = torch.nn.functional.mse_loss(c, shift)
                loss += exp_cfg.vi_cfg.pred_loss_weight * pred_loss

            (loss / exp_cfg.vi_cfg.grad_acc_iter).backward()
            if curr_iter % exp_cfg.vi_cfg.logging_interval == 0:
                enc_grad = torch.cat([param.grad.data.reshape(-1).detach().cpu() for param in encoder.parameters()])
                wandb.log(
                    {
                        "grad/avg_vi_grad": enc_grad.abs().mean(),
                        "grad/max_vi_grad": enc_grad.abs().max(),
                        "grad/med_vi_grad": torch.median(enc_grad.abs()),
                        "grad/avg_to_grad": psi.grad.abs().mean() if psi.grad is not None else 0.0,
                        "grad/max_to_grad": psi.grad.abs().max() if psi.grad is not None else 0.0,
                        "grad/med_to_grad": torch.median(psi.grad.abs()) if psi.grad is not None else 0.0,
                    },
                    step=curr_iter,
                )

            if curr_iter % exp_cfg.vi_cfg.grad_acc_iter == 0:
                torch.nn.utils.clip_grad_norm_(psi, 0.1)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.1)
                opt.step()
                scheduler.step()
                opt.zero_grad()

            transop_loss_save.append(transop_loss.mean().item())
            kl_loss_save.append(kl_loss.item())
            dw_bw_points = torch.nn.functional.mse_loss(z0, z1, reduction="none").mean(dim=-1)
            dw_loss_save.append((transop_loss.mean(dim=-1) / dw_bw_points).mean().item())
            c_save.append(c.detach().cpu())
            iter_time.append(time.time() - pre_time)

            if curr_iter % exp_cfg.vi_cfg.logging_interval == 0:
                last_c = torch.cat(c_save[-exp_cfg.vi_cfg.logging_interval :])
                c_nz = torch.count_nonzero(last_c, dim=-1).float().mean()
                c_mag = last_c[last_c.abs() > 0].abs().mean()
                log.info(
                    f"Iter {curr_iter:6n} -- TO loss: {np.mean(transop_loss_save[-exp_cfg.vi_cfg.logging_interval:]):.3E},"
                    + f" KL loss: {np.mean(kl_loss_save[-exp_cfg.vi_cfg.logging_interval:]):.3E},"
                    + f" dist improve: {np.mean(dw_loss_save[-exp_cfg.vi_cfg.logging_interval:]):.3E},"
                    + f" c nonzero: {c_nz:.3f},"
                    + f" c mag: {c_mag:.3f},"
                    + f" time: {np.mean(iter_time[-exp_cfg.vi_cfg.logging_interval:]):.2f}sec"
                )

                wandb.log(
                    {
                        "loss/transop": np.mean(transop_loss_save[-exp_cfg.vi_cfg.logging_interval :]),
                        "loss/kl": np.mean(kl_loss_save[-exp_cfg.vi_cfg.logging_interval :]),
                        "loss/dist_improve": np.mean(dw_loss_save[-exp_cfg.vi_cfg.logging_interval :]),
                        "meta/c_nonzero": c_nz,
                        "meta/c_mag": c_mag,
                        "meta/iter_time": np.mean(iter_time[-exp_cfg.vi_cfg.logging_interval :]),
                    },
                    step=curr_iter,
                )

                if exp_cfg.transop_cfg.train_transop:
                    log_transop(psi.detach().cpu(), curr_iter)
            if curr_iter % 5000 == 0:
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "psi": psi,
                        "optimizer": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "curr_iter": curr_iter,
                        "curr_epoch": i,
                    },
                    os.getcwd() + f"/model_state{curr_iter}.pt",
                )


def register_configs() -> None:
    cs.store(name="exp_cfg", node=ExperimentConfig)


@hydra.main(version_base=None, config_path="../config_vi", config_name="default_vi")
def main(exp_cfg: ExperimentConfig) -> None:
    wandb.init(
        project="vi-tune",
        entity="kfallah",
        mode="online" if exp_cfg.enable_wandb else "disabled",
        settings=wandb.Settings(start_method="thread"),
        config=OmegaConf.to_container(exp_cfg, resolve=True, throw_on_missing=True),
    )

    # Set random seeds
    torch.manual_seed(exp_cfg.seed)
    np.random.seed(exp_cfg.seed)

    log.info("Initializing Models...")
    train_dataloader, backbone, nn_bank, psi, encoder = init_models(exp_cfg)
    log.info("Training Variational Encoder...")
    train(exp_cfg, train_dataloader, backbone, nn_bank, psi, encoder)


if __name__ == "__main__":
    log = logging.getLogger(__name__)
    cs = ConfigStore.instance()
    register_configs()

    main()
