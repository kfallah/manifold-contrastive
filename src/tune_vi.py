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
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from dataloader.contrastive_dataloader import get_dataloader
from model.config import ModelConfig
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
    dropout: float = 0.1


@dataclass
class CoeffEncoderConfig:
    variational: bool = True
    attention_cfg: AttentionConfig = AttentionConfig()

    feature_dim: int = 512
    hidden_dim1: int = 2048
    hidden_dim2: int = 2048
    scale_prior: float = 0.02
    shift_prior: float = 0.0
    threshold: float = 0.032

    logging_interval: int = 500
    lr: float = 0.03
    opt_type: str = "SGD"
    batch_size: int = 128
    kl_weight: float = 5.0e-4
    weight_decay: float = 1.0e-5
    grad_acc_iter: int = 1

    enable_thresh_warmup: bool = False

    wasserstein_loss: bool = False

    enable_c_l2: bool = False
    c_l2_weight: float = 1.0e-3

    enable_shift_l2: bool = False
    shift_l2_weight: float = 1.0e-2

    enable_max_sample: bool = False
    max_sample_start_iter: int = 2000
    max_sample_l1_penalty: float = 1.0e-3
    total_num_samples: int = 100
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

    init_real_range: float = 0.0001
    init_imag_range: float = 5.0


@dataclass
class ExperimentConfig:
    exp_name: str = MISSING
    exp_dir: str = MISSING
    enable_wandb: bool = False
    seed: int = 0

    num_epochs: int = 300
    run_dir: str = "../../results/Transop_sep-loss/"
    load_dir: str = ""
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
            nn.LeakyReLU(),
            nn.Linear(cfg.psi_hidden_dim, cfg.feature_dim),
        )

        self.attn = nn.MultiheadAttention(
            cfg.feature_dim, num_heads=cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.pre_norm = nn.LayerNorm(cfg.feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.ff_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.ff_hidden_dim, cfg.feature_dim),
        )
        self.post_norm = nn.LayerNorm(cfg.feature_dim)

    def forward(self, z, psi):
        # extract features for operators
        psi_feat = self.psi_extract(psi.reshape(self.dict_size, -1))
        psi_feat = psi_feat.unsqueeze(0).repeat(len(z), 1, 1)
        # Run through attention layer
        z_attn, _ = self.attn(z, psi_feat, psi_feat)
        z = self.pre_norm(z + z_attn)
        z_mlp = self.mlp(z)
        z = self.post_norm(z + z_mlp)
        return z


class VIEncoder(nn.Module):
    def __init__(self, cfg: CoeffEncoderConfig, dict_size: int):
        super(VIEncoder, self).__init__()
        self.feat_extract = nn.Sequential(
            nn.Linear(128, cfg.hidden_dim1),
            nn.LeakyReLU(),
            nn.Linear(cfg.hidden_dim1, cfg.hidden_dim2),
            nn.LeakyReLU(),
            nn.Linear(cfg.hidden_dim2, cfg.feature_dim),
        )

        self.attn1 = SparseCodeAttn(cfg.attention_cfg, dict_size)
        # self.attn2 = SparseCodeAttn(cfg.attention_cfg, dict_size)

        self.cfg = cfg
        self.dict_size = dict_size
        if cfg.variational:
            self.scale = nn.Linear(cfg.feature_dim, dict_size)
        self.shift = nn.Linear(cfg.feature_dim, dict_size)

        if self.cfg.learn_prior:
            self.prior_scale = nn.Sequential(
                nn.Linear(64, cfg.hidden_dim1),
                nn.LeakyReLU(),
                nn.Linear(cfg.hidden_dim1, cfg.hidden_dim2),
                nn.LeakyReLU(),
                nn.Linear(cfg.hidden_dim2, dict_size),
            )

        if self.cfg.enable_thresh_warmup:
            self.thresh_warmup = 0.0
        else:
            self.thresh_warmup = 1.0

    def soft_threshold(self, z: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(torch.abs(z) - lambda_) * torch.sign(z)

    def kl_loss(self, scale, shift, prior_scale=None, prior_shift=None, kl_scale=None):
        if prior_scale is None:
            prior_scale = torch.ones_like(scale) * self.cfg.scale_prior
        if prior_shift is None:
            prior_shift = torch.ones_like(shift) * self.cfg.shift_prior * torch.sign(shift).detach()
            if kl_scale is not None:
                prior_shift *= kl_scale
            # prior_shift = shift.clone().detach()
        if self.cfg.no_shift_prior:
            shift = shift.detach()

        if self.cfg.wasserstein_loss:
            w2 = ((shift - prior_shift) ** 2).sum(dim=-1) + ((scale - prior_scale) ** 2).sum(dim=-1)
            return w2

        encoder = distr.Laplace(shift, scale)
        prior = distr.Laplace(prior_shift, prior_scale)
        return distr.kl_divergence(encoder, prior)

    def reparameterize(self, noise, log_scale, shift):
        # Reparameterize
        scale = torch.exp(log_scale)
        eps = -scale * torch.sign(noise) * torch.log((1.0 - 2.0 * torch.abs(noise)).clamp(min=1e-6, max=1e6))
        c = shift + eps

        # Threshold
        c_thresh = self.soft_threshold(eps.detach(), self.cfg.threshold * self.thresh_warmup)
        non_zero = torch.nonzero(c_thresh, as_tuple=True)
        c_thresh[non_zero] = shift[non_zero].detach() + c_thresh[non_zero]
        c = c + c_thresh - c.detach()
        return c

    def max_elbo_sample(self, log_scale, shift, psi, x0, x1):
        with torch.no_grad():
            noise_list = []
            loss_list = []
            l1_list = []
            log_scale_expanded = log_scale.unsqueeze(0).repeat(self.cfg.samples_per_iter, 1, 1, 1)
            shift_expanded = shift.unsqueeze(0).repeat(self.cfg.samples_per_iter, 1, 1, 1)
            for _ in range(self.cfg.total_num_samples // self.cfg.samples_per_iter):
                noise = torch.rand_like(log_scale_expanded) - 0.5
                c = self.reparameterize(noise, log_scale_expanded, shift_expanded)
                T = torch.matrix_exp(torch.einsum("sblm,mpk->sblpk", c, psi))
                x1_hat = (T @ x0.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
                transop_loss = torch.nn.functional.mse_loss(
                    x1_hat, x1.unsqueeze(0).repeat(self.cfg.samples_per_iter, 1, 1, 1), reduction="none"
                ).mean(dim=-1)

                noise_list.append(noise)
                loss_list.append(transop_loss)
                l1_list.append(c.abs().sum(dim=-1))

            noise_list = torch.cat(noise_list, dim=0)
            loss_list = torch.cat(loss_list, dim=0)
            l1_list = torch.cat(l1_list, dim=0)
            max_elbo = torch.argmin(loss_list + self.cfg.max_sample_l1_penalty * l1_list, dim=0).detach()
            # Pick out best noise sample for each batch entry for reparameterization
            max_elbo = max_elbo.reshape(-1)
            n_samples, n_batch, n_seq, n_feat = noise_list.shape
            noise_list = noise_list.reshape(n_samples, -1, n_feat)
            optimal_noise = noise_list[max_elbo, torch.arange(n_batch * n_seq)]
            return optimal_noise.reshape(n_batch, n_seq, n_feat)

    def log_distr(self, curr_iter, scale, shift, prior_scale):
        distr = {
            "distr/avg_enc_scale": scale.mean(),
            "distr/min_enc_scale": scale.min(),
            "distr/max_enc_scale": scale.max(),
            "distr/avg_enc_shift": shift.abs().mean(),
            "distr/min_enc_shift": shift.abs().min(),
            "distr/max_enc_shift": shift.abs().max(),
        }
        if self.cfg.learn_prior:
            distr.update(
                {
                    "distr/avg_prior_scale": prior_scale.mean(),
                    "distr/min_prior_scale": prior_scale.min(),
                    "distr/max_prior_scale": prior_scale.max(),
                }
            )

        wandb.log(distr, step=curr_iter)

    def forward(self, curr_iter, x0, x1, psi):
        # Extract features for attention
        # z0, z1 = self.feat_extract(x0), self.feat_extract(x1)
        z = self.feat_extract(torch.cat((x0, x1), dim=-1))
        z = self.attn1(z, psi)
        # z = self.attn2(z, psi)

        if self.cfg.enable_thresh_warmup:
            if not self.cfg.deterministic_pretrain or curr_iter > self.cfg.num_det_pretrain_iters:
                self.thresh_warmup += 5.0e-5
            if self.thresh_warmup > 1.0:
                self.thresh_warmup = 1.0

        if not self.cfg.variational:
            shift = self.shift(z)
            return shift, torch.tensor(0.0), (torch.zeros_like(shift), shift)

        log_scale, shift = self.scale(z), self.shift(z)
        log_scale += torch.log(torch.ones_like(log_scale) * self.cfg.scale_prior)
        log_scale, shift = log_scale.clamp(min=-100, max=2), shift.clamp(min=-5, max=5)

        # Reparameterization
        if self.cfg.deterministic_pretrain and curr_iter < self.cfg.num_det_pretrain_iters:
            c = shift.clone()
        else:
            if self.cfg.enable_max_sample and curr_iter > self.cfg.max_sample_start_iter:
                noise = self.max_elbo_sample(log_scale, shift, psi, x0, x1)
            else:
                noise = torch.rand_like(log_scale) - 0.5
            c = self.reparameterize(noise, log_scale, shift)

        # Compute KL and find prior params if needed
        prior_scale = None
        kl_scale = None
        # psi_mag = (psi.reshape(-1, self.dict_size) ** 2).sum(dim=0)
        # psi_median = torch.median(psi_mag)
        # kl_scale = torch.exp(5 * torch.log((psi_median / psi_mag))).detach().clamp(min=0.1, max=10)
        if self.cfg.learn_prior:
            prior_log_scale = self.prior_scale(x0)
            prior_log_scale += torch.log(torch.ones_like(prior_log_scale) * self.cfg.scale_prior)
            prior_scale = prior_log_scale.clamp(min=-100, max=2).exp()
        kl = self.kl_loss(log_scale.exp(), shift, prior_scale, None, kl_scale).sum(dim=-1).mean()

        # Log distribution params
        if curr_iter % self.cfg.logging_interval == 0:
            self.log_distr(curr_iter, log_scale.exp(), shift, prior_scale)

        return c, kl, (log_scale, shift)


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

    if len(exp_cfg.load_dir) > 0:
        state = torch.load(exp_cfg.load_dir, map_location=default_device)
        encoder.load_state_dict(state["encoder"])
        psi = state["psi"].to(default_device)

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
                "eta_min": 1e-6,
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
    n_epochs = exp_cfg.num_epochs + (exp_cfg.vi_cfg.num_det_pretrain_iters // iters_per_epoch)
    scheduler = LinearWarmupCosineAnnealingLR(
        opt, warmup_epochs=20 * iters_per_epoch, max_epochs=iters_per_epoch * n_epochs, eta_min=1e-4
    )

    transop_loss_save = []
    kl_loss_save = []
    dw_loss_save = []
    c_save = []
    iter_time = []
    for i in range(n_epochs):
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

            c, kl_loss, (log_scale, shift) = encoder(curr_iter, z0, z1, psi.detach())
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
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 10.0)
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
