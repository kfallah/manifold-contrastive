import logging
import os
import time
import warnings
from dataclasses import MISSING, dataclass

import hydra
import numpy as np
import torch
import torch.distributions as distr
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import wandb
from dataloader.config import DataLoaderConfig, DatasetConfig
from dataloader.ssl_dataloader import get_dataset
from model.manifold.transop import TransOp_expm
from model.public.linear_warmup_cos_anneal import LinearWarmupCosineAnnealingLR

warnings.filterwarnings("ignore")


@dataclass
class CoeffEncoderConfig:
    variational: bool = True
    use_nn_point_pair: bool = False

    feature_dim: int = 512
    hidden_dim1: int = 2048
    hidden_dim2: int = 2048
    scale_prior: float = 0.01
    shift_prior: float = 0.0
    threshold: float = 0.01

    logging_interval: int = 500
    lr: float = 0.0001
    opt_type: str = "AdamW"
    kl_weight: float = 1.0e-5
    weight_decay: float = 1.0e-5
    grad_acc_iter: int = 1

    enable_c_l2: bool = False
    c_l2_weight: float = 1.0e-3

    enable_shift_l2: bool = True
    shift_l2_weight: float = 5.0e-3

    enable_max_sample: bool = True
    max_sample_start_iter: int = 0
    total_num_samples: int = 100
    samples_per_iter: int = 50

    learn_prior: bool = True
    no_shift_prior: bool = True


@dataclass
class TransportOperatorConfig:
    train_transop: bool = True
    start_iter: int = 0
    dict_size: int = 64
    batch_size: int = 64
    random_filter_count: int = 32

    lr: float = 0.001
    weight_decay: float = 1.0e-2

    init_real_range: float = 0.0001
    init_imag_range: float = 6.0


@dataclass
class ExperimentConfig:
    exp_name: str = MISSING
    exp_dir: str = MISSING
    enable_wandb: bool = False
    seed: int = 0

    num_epochs: int = 500
    device: str = "cuda:0"

    vi_cfg: CoeffEncoderConfig = CoeffEncoderConfig()
    transop_cfg: TransportOperatorConfig = TransportOperatorConfig()
    data_cfg: DataLoaderConfig = DataLoaderConfig(
        dataset_cfg=DatasetConfig(dataset_name="CIFAR10", num_classes=10, image_size=32),
        num_workers=32,
        train_batch_size=512,
    )


def contrastive_loss(x0, x1, tau, norm=True):
    # https://github.com/google-research/simclr/blob/master/objective.py
    bsize = x0.shape[0]
    target = torch.arange(bsize, device=x1.device)
    eye_mask = torch.eye(bsize, device=x1.device) * 1e9
    if norm:
        x0 = F.normalize(x0, p=2, dim=1)
        x1 = F.normalize(x1, p=2, dim=1)
    logits00 = x0 @ x0.t() / tau - eye_mask
    logits11 = x1 @ x1.t() / tau - eye_mask
    logits01 = x0 @ x1.t() / tau
    logits10 = x1 @ x0.t() / tau
    return (
        F.cross_entropy(torch.cat([logits01, logits00], dim=1), target)
        + F.cross_entropy(torch.cat([logits10, logits11], dim=1), target)
    ) / 2


class VIEncoder(nn.Module):
    def __init__(self, cfg: CoeffEncoderConfig, dict_size: int):
        super(VIEncoder, self).__init__()
        self.feat_extract = nn.Sequential(
            nn.Linear(32, cfg.hidden_dim1),
            nn.LeakyReLU(),
            nn.Linear(cfg.hidden_dim1, cfg.hidden_dim2),
            nn.LeakyReLU(),
            nn.Linear(cfg.hidden_dim2, cfg.feature_dim),
        )

        self.cfg = cfg
        self.dict_size = dict_size
        self.threshold = cfg.threshold
        if cfg.variational:
            self.scale = nn.Linear(cfg.feature_dim, dict_size)
        self.shift = nn.Linear(cfg.feature_dim, dict_size)

        if self.cfg.learn_prior:
            self.prior_feat = nn.Sequential(
                nn.Linear(16, cfg.hidden_dim1),
                nn.LeakyReLU(),
                nn.Linear(cfg.hidden_dim1, cfg.hidden_dim2),
                nn.LeakyReLU(),
                nn.Linear(cfg.hidden_dim2, cfg.feature_dim),
            )
            self.prior_scale = nn.Linear(cfg.feature_dim, dict_size)
            # self.prior_shift = nn.Linear(cfg.feature_dim, dict_size)

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

        encoder = distr.Laplace(shift, scale)
        prior = distr.Laplace(prior_shift, prior_scale)
        return distr.kl_divergence(encoder, prior)

    def sample(self, x0):
        assert self.cfg.learn_prior
        z_prior = self.prior_feat(x0)
        log_scale = self.prior_scale(z_prior)
        log_scale += torch.log(torch.ones_like(log_scale) * self.cfg.scale_prior)
        log_scale = log_scale.clamp(min=-100, max=2)
        noise = torch.rand_like(log_scale) - 0.5
        shift = torch.zeros_like(log_scale)
        return self.reparameterize(noise, log_scale, shift)

    def reparameterize(self, noise, log_scale, shift):
        # Reparameterize
        scale = torch.exp(log_scale)
        eps = -scale * torch.sign(noise) * torch.log((1.0 - 2.0 * torch.abs(noise)).clamp(min=1e-6, max=1e6))
        c = shift + eps

        # Threshold
        c_thresh = self.soft_threshold(eps.detach(), self.threshold)
        non_zero = torch.nonzero(c_thresh, as_tuple=True)
        c_thresh[non_zero] = shift[non_zero].detach() + c_thresh[non_zero]
        c = c + c_thresh - c.detach()
        return c

    def max_elbo_sample(self, log_scale, shift, psi, x0, x1):
        with torch.no_grad():
            noise_list = []
            loss_list = []
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

            noise_list = torch.cat(noise_list, dim=0)
            loss_list = torch.cat(loss_list, dim=0)
            max_elbo = torch.argmin(loss_list, dim=0).detach()
            # Pick out best noise sample for each batch entry for reparameterization
            max_elbo = max_elbo.reshape(-1)
            n_samples, n_batch, n_seq, n_feat = noise_list.shape
            noise_list = noise_list.reshape(n_samples, -1, n_feat)
            optimal_noise = noise_list[max_elbo, torch.arange(n_batch * n_seq)]
            return optimal_noise.reshape(n_batch, n_seq, n_feat)

    def log_distr(self, curr_iter, scale, shift, prior_scale, prior_shift):
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
                    "prior/avg_prior_scale": prior_scale.mean(),
                    "prior/min_prior_scale": prior_scale.min(),
                    "prior/max_prior_scale": prior_scale.max(),
                    # "prior/avg_prior_mean": prior_shift.abs().mean(),
                    # "prior/min_prior_mean": prior_shift.abs().min(),
                    # "prior/max_prior_mean": prior_shift.abs().max(),
                }
            )

        wandb.log(distr, step=curr_iter)

    def forward(self, curr_iter, x0, x1, psi):
        z = self.feat_extract(torch.cat((x0, x1), dim=-1))

        if not self.cfg.variational:
            shift = self.shift(z)
            return shift, torch.tensor(0.0), (torch.zeros_like(shift), shift)

        log_scale, shift = self.scale(z), self.shift(z)
        log_scale += torch.log(torch.ones_like(log_scale) * self.cfg.scale_prior)
        log_scale, shift = log_scale.clamp(min=-100, max=2), shift.clamp(min=-5, max=5)

        # Reparameterization
        if self.cfg.enable_max_sample and curr_iter > self.cfg.max_sample_start_iter:
            noise = self.max_elbo_sample(log_scale, shift, psi, x0, x1)
        else:
            noise = torch.rand_like(log_scale) - 0.5
        c = self.reparameterize(noise, log_scale, shift)

        # Compute KL and find prior params if needed
        prior_scale = None
        prior_shift = None
        kl_scale = None
        if self.cfg.learn_prior:
            z_prior = self.prior_feat(x0)
            prior_log_scale = self.prior_scale(z_prior)
            prior_log_scale += torch.log(torch.ones_like(log_scale) * self.cfg.scale_prior)
            prior_log_scale = prior_log_scale.clamp(min=-100, max=2)
            prior_scale = prior_log_scale.exp()
        kl = self.kl_loss(log_scale.exp(), shift, prior_scale, prior_shift, kl_scale).sum(dim=-1).mean()

        # Log distribution params
        if curr_iter % self.cfg.logging_interval == 0:
            self.log_distr(curr_iter, log_scale.exp(), shift, prior_scale, prior_shift)

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
    default_device = torch.device(exp_cfg.device)
    # Load config
    dataset = get_dataset(exp_cfg.data_cfg)
    train_dataloader = dataset.train_dataloader

    transop = TransOp_expm(
        exp_cfg.transop_cfg.dict_size,
        16,
        stable_init=True,
        real_range=exp_cfg.transop_cfg.init_real_range,
        imag_range=exp_cfg.transop_cfg.init_imag_range,
    ).to(default_device)

    backbone = torchvision.models.resnet18()
    backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    backbone.to(default_device)
    backbone = nn.Sequential(*backbone._modules.values())

    projection = nn.Sequential(
        nn.Linear(512, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 64),
        nn.BatchNorm1d(64),
    ).to(default_device)
    encoder = VIEncoder(exp_cfg.vi_cfg, exp_cfg.transop_cfg.dict_size).to(default_device)

    return train_dataloader, backbone, transop.psi, encoder, projection


def train(exp_cfg, train_dataloader, backbone, psi, encoder, projector):
    param_list = [
        {
            "params": encoder.parameters(),
            "lr": exp_cfg.vi_cfg.lr,
            "eta_min": 1e-6,
            "weight_decay": exp_cfg.vi_cfg.weight_decay,
            "disable_layer_adaptation": True,
        },
        {
            "params": list(backbone.parameters()) + list(projector.parameters()),
            "lr": 0.003,
            "weight_decay": 1.0e-5,
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
    n_epochs = exp_cfg.num_epochs
    scheduler = LinearWarmupCosineAnnealingLR(
        opt, warmup_epochs=10 * iters_per_epoch, max_epochs=iters_per_epoch * n_epochs, eta_min=5e-5
    )

    transop_loss_save = []
    kl_loss_save = []
    infonce_loss_save = []
    dw_loss_save = []
    c_save = []
    iter_time = []
    for i in range(n_epochs):
        for idx, batch in enumerate(train_dataloader):
            curr_iter = i * len(train_dataloader) + idx
            pre_time = time.time()
            x0, x1 = batch[0][0], batch[0][1]
            x0, x1 = x0.to(exp_cfg.device), x1.to(exp_cfg.device)

            # TRANSOP LOSS
            z0, z1 = backbone[:-2](x0).reshape(len(x0), -1, 16), backbone[:-2](x1).reshape(len(x1), -1, 16)
            filter_idx = torch.randperm(z0.shape[1])[: exp_cfg.transop_cfg.random_filter_count]
            z0_use, z1_use = (
                z0[: exp_cfg.transop_cfg.batch_size, filter_idx],
                z1[: exp_cfg.transop_cfg.batch_size, filter_idx],
            )
            # if exp_cfg.vi_cfg.use_nn_point_pair:
            #    z1 = nn_bank(z1.detach(), update=True).detach()

            c, kl_loss, (log_scale, shift) = encoder(curr_iter, z0_use.detach(), z1_use.detach(), psi.detach())
            T = torch.matrix_exp(torch.einsum("bsm,mpk->bspk", c, psi))
            z1_hat = (T @ z0_use.unsqueeze(-1)).squeeze(-1)
            transop_loss = torch.nn.functional.mse_loss(z1_hat, z1_use, reduction="none")

            loss = transop_loss.mean() + exp_cfg.vi_cfg.kl_weight * kl_loss
            if exp_cfg.vi_cfg.enable_c_l2:
                l2_reg = (c**2).sum(dim=-1).mean()
                loss += exp_cfg.vi_cfg.c_l2_weight * l2_reg
            if exp_cfg.vi_cfg.enable_shift_l2:
                l2_reg = (shift**2).sum(dim=-1).mean()
                loss += exp_cfg.vi_cfg.shift_l2_weight * l2_reg

            if exp_cfg.vi_cfg.learn_prior:
                z0_aug = z0.clone()
                c_aug = encoder.sample(z0[:, filter_idx].detach())
                T = torch.matrix_exp(torch.einsum("bsm,mpk->bspk", c_aug, psi.detach()))
                z0_aug[:, filter_idx] = (T @ z0[:, filter_idx].unsqueeze(-1)).squeeze(-1)
            else:
                z0_aug = z0

            # CONTRASTIVE LOSS
            z0f, z1f = (
                backbone[-2:](z0_aug.reshape(len(z0), -1, 4, 4)).squeeze(),
                backbone[-2:](z1.reshape(len(z1), -1, 4, 4)).squeeze(),
            )
            h = projector(torch.cat([z0f.squeeze(), z1f.squeeze()]))
            h0, h1 = h[: len(z0)], h[len(z0) :]
            infonce_loss = contrastive_loss(h0, h1, 0.5)
            loss += infonce_loss

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
            infonce_loss_save.append(infonce_loss.item())
            dw_bw_points = torch.nn.functional.mse_loss(z0_use, z1_use, reduction="none").mean(dim=-1) + 1e-6
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
                    + f" InfoNCE loss: {np.mean(infonce_loss_save[-exp_cfg.vi_cfg.logging_interval:]):.3E},"
                    + f" dist improve: {np.mean(dw_loss_save[-exp_cfg.vi_cfg.logging_interval:]):.3E},"
                    + f" c nonzero: {c_nz:.3f},"
                    + f" c mag: {c_mag:.3f},"
                    + f" time: {np.mean(iter_time[-exp_cfg.vi_cfg.logging_interval:]):.2f}sec"
                )

                wandb.log(
                    {
                        "loss/transop": np.mean(transop_loss_save[-exp_cfg.vi_cfg.logging_interval :]),
                        "loss/kl": np.mean(kl_loss_save[-exp_cfg.vi_cfg.logging_interval :]),
                        "loss/infonce": np.mean(infonce_loss_save[-exp_cfg.vi_cfg.logging_interval :]),
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


@hydra.main(version_base=None, config_path="../config_spatial", config_name="base")
def main(exp_cfg: ExperimentConfig) -> None:
    wandb.init(
        project="spatial-transop",
        mode="online" if exp_cfg.enable_wandb else "disabled",
        settings=wandb.Settings(start_method="thread"),
        config=OmegaConf.to_container(exp_cfg, resolve=True, throw_on_missing=True),
    )

    # Set random seeds
    torch.manual_seed(exp_cfg.seed)
    np.random.seed(exp_cfg.seed)

    log.info("Initializing Models...")
    train_dataloader, backbone, psi, encoder, projector = init_models(exp_cfg)
    log.info("Training Variational Encoder...")
    train(exp_cfg, train_dataloader, backbone, psi, encoder, projector)


if __name__ == "__main__":
    log = logging.getLogger(__name__)
    cs = ConfigStore.instance()
    register_configs()
    main()
