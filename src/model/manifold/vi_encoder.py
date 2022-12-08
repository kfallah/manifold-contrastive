from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from model.contrastive.config import VariationalEncoderConfig
from model.manifold.reparameterize import draw_noise_samples, reparameterize
from model.type import DistributionData


class VIEncoder(nn.Module):
    def __init__(
        self,
        vi_cfg: VariationalEncoderConfig,
        input_size: int,
        dictionary_size: int,
        lambda_prior: float,
        full_size: int = 512,
    ):
        super(VIEncoder, self).__init__()

        if vi_cfg.use_warmpup:
            self.warmup = 0.01
        else:
            self.warmup = 1.0
        self.vi_cfg = vi_cfg
        feat_dim = self.vi_cfg.feature_dim
        self.dictionary_size = dictionary_size
        self.lambda_prior = lambda_prior

        self.layer_norm = nn.LayerNorm(full_size)
        self.initialize_encoder_params(input_size, feat_dim, dictionary_size)
        self.initialize_prior_params(input_size, feat_dim, dictionary_size)

    def initialize_encoder_params(self, input_size, feat_dim, dict_size):
        if self.vi_cfg.encoder_type == "MLP":
            in_size = input_size if self.vi_cfg.share_encoder else ((2 * input_size) + 1)
            self.enc_feat_extract = nn.Sequential(
                nn.Linear(in_size, 4 * feat_dim),
                nn.BatchNorm1d(4 * feat_dim),
                nn.LeakyReLU(),
                nn.Linear(4 * feat_dim, feat_dim),
            )
        else:
            raise NotImplementedError

        if (
            self.vi_cfg.distribution == "Laplacian"
            or self.vi_cfg.distribution == "Laplacian+Gamma"
            or self.vi_cfg.distribution == "Gaussian"
            or self.vi_cfg.distribution == "Gaussian+Gamma"
        ):
            self.enc_scale = nn.Linear(feat_dim, dict_size)
            self.enc_shift = nn.Linear(feat_dim, dict_size)

        if self.vi_cfg.distribution == "Laplacian+Gamma" or self.vi_cfg.distribution == "Gaussian+Gamma":
            # self.enc_gamma_a = nn.Linear(feat_dim, dict_size)
            self.enc_gamma_b = nn.Linear(feat_dim, dict_size)

    def initialize_prior_params(self, input_size, feat_dim, dict_size):
        self.prior_params = {}
        if (
            self.vi_cfg.distribution == "Laplacian"
            or self.vi_cfg.distribution == "Laplacian+Gamma"
            or self.vi_cfg.distribution == "Gaussian"
            or self.vi_cfg.distribution == "Gaussian+Gamma"
        ):
            self.prior_params["logscale"] = torch.log(torch.tensor(self.vi_cfg.scale_prior))
            self.prior_params["shift"] = 0.0

        if self.vi_cfg.distribution == "Laplacian+Gamma" or self.vi_cfg.distribution == "Gaussian+Gamma":
            self.prior_params["gamma_a"] = 2.0
            self.prior_params["gamma_b"] = 2.0 / self.lambda_prior

        if self.vi_cfg.prior_type == "Learned":
            self.prior_feat_extract = nn.Sequential(
                nn.Linear(input_size, 4 * feat_dim),
                nn.LayerNorm(4 * feat_dim),
                # nn.BatchNorm1d(2 * input_size),
                nn.GELU(),
                nn.Linear(4 * feat_dim, 2 * feat_dim),
                nn.LayerNorm(2 * feat_dim),
                # nn.BatchNorm1d(2 * input_size),
                nn.GELU(),
                nn.Linear(2 * feat_dim, feat_dim),
            )

            if (
                self.vi_cfg.distribution == "Laplacian"
                or self.vi_cfg.distribution == "Laplacian+Gamma"
                or self.vi_cfg.distribution == "Gaussian"
                or self.vi_cfg.distribution == "Gaussian+Gamma"
            ):
                self.prior_scale = nn.Linear(feat_dim, dict_size)
                self.prior_shift = nn.Linear(feat_dim, dict_size)

            if self.vi_cfg.distribution == "Laplacian+Gamma" or self.vi_cfg.distribution == "Gaussian+Gamma":
                # self.prior_gamma_a = nn.Linear(feat_dim, dict_size)
                self.prior_gamma_b = nn.Linear(feat_dim, dict_size)

    def sample_prior(self, z, distribution_data: DistributionData) -> torch.Tensor:
        distr = distribution_data.prior_params
        if len(distr["shift"]) == len(z):
            distr_params = distr
        else:
            if self.vi_cfg.prior_type == "Fixed":
                assert NotImplementedError
                distr_params = self.prior_params
            elif self.vi_cfg.prior_type == "Learned":
                distr_params = {}
                z_prior = self.prior_feat_extract(z)
                distr_params["logscale"] = self.prior_scale(z_prior).clamp(max=2)
                distr_params["logscale"] += torch.log(
                    torch.ones_like(distr_params["logscale"]) * self.vi_cfg.scale_prior
                )
                distr_params["shift"] = self.prior_shift(z_prior).clamp(min=-1, max=1)

                if self.vi_cfg.distribution == "Laplacian+Gamma" or self.vi_cfg.distribution == "Gaussian+Gamma":
                    # prior_gamma_a = self.prior_gamma_a(z_prior).exp()
                    prior_gamma_b = self.prior_gamma_b(z_prior).exp().clamp(min=1e-4, max=1e4) * (
                        2 / self.lambda_prior
                    )
                    prior_gamma_a = torch.ones_like(prior_gamma_b) * 2.0
                    distr_params["gamma_a"] = prior_gamma_a
                    distr_params["gamma_b"] = prior_gamma_b

        noise = draw_noise_samples(self.vi_cfg.distribution, distr_params["shift"].shape, z.device)
        c = reparameterize(
            self.vi_cfg.distribution,
            distr_params,
            noise,
            self.lambda_prior,
            self.warmup,
            self.vi_cfg.normalize_coefficients,
            self.vi_cfg.normalize_mag,
        )
        return c

    def get_distribution_params(self, x0, x1) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        encoder_params = {}
        if self.vi_cfg.encoder_type == "MLP":
            if self.vi_cfg.share_encoder:
                z0 = self.enc_feat_extract(x0)
                z1 = self.enc_feat_extract(x1)
                z_enc = torch.max(z0, z1)
            else:
                dist = (torch.linalg.norm(x0 - x1, dim=-1) ** 2).unsqueeze(-1)
                z_enc = self.enc_feat_extract(torch.cat((x0, x1, dist), dim=1))
        else:
            raise NotImplementedError()

        if (
            self.vi_cfg.distribution == "Laplacian"
            or self.vi_cfg.distribution == "Laplacian+Gamma"
            or self.vi_cfg.distribution == "Gaussian"
            or self.vi_cfg.distribution == "Gaussian+Gamma"
        ):
            encoder_params["logscale"] = self.enc_scale(z_enc).clamp(max=2)
            encoder_params["logscale"] += torch.log(
                torch.ones_like(encoder_params["logscale"]) * self.vi_cfg.scale_prior
            )
            # encoder_params["shift"] = self.enc_shift(z_enc).clamp(min=-1, max=1)
            encoder_params["shift"] = self.enc_shift(z_enc)

        if self.vi_cfg.distribution == "Laplacian+Gamma" or self.vi_cfg.distribution == "Gaussian+Gamma":
            # gamma_a = self.enc_gamma_a(z_enc).exp()
            gamma_b = self.enc_gamma_b(z_enc).exp().clamp(min=1e-4, max=1e4) * (2 / self.lambda_prior)
            gamma_a = 2 * torch.ones_like(gamma_b)
            encoder_params["gamma_a"] = gamma_a
            encoder_params["gamma_b"] = gamma_b

        prior_params = {}
        hyperprior_params = {}
        if (
            self.vi_cfg.distribution == "Laplacian"
            or self.vi_cfg.distribution == "Laplacian+Gamma"
            or self.vi_cfg.distribution == "Gaussian"
            or self.vi_cfg.distribution == "Gaussian+Gamma"
        ):
            hyperprior_params["logscale"] = torch.ones_like(encoder_params["logscale"]) * self.prior_params["logscale"]
            hyperprior_params["shift"] = torch.zeros_like(encoder_params["shift"])
        if self.vi_cfg.distribution == "Laplacian+Gamma" or self.vi_cfg.distribution == "Gaussian+Gamma":
            hyperprior_params["gamma_a"] = torch.ones_like(encoder_params["gamma_a"]) * self.prior_params["gamma_a"]
            hyperprior_params["gamma_b"] = torch.ones_like(encoder_params["gamma_b"]) * self.prior_params["gamma_b"]

        if self.vi_cfg.prior_type == "Learned":
            z_prior = self.prior_feat_extract(x0)
            if (
                self.vi_cfg.distribution == "Laplacian"
                or self.vi_cfg.distribution == "Laplacian+Gamma"
                or self.vi_cfg.distribution == "Gaussian"
                or self.vi_cfg.distribution == "Gaussian+Gamma"
            ):
                prior_params["logscale"] = self.prior_scale(z_prior).clamp(max=2)
                prior_params["logscale"] += torch.log(
                    torch.ones_like(prior_params["logscale"]) * self.vi_cfg.scale_prior
                )
                prior_params["shift"] = self.prior_shift(z_prior).clamp(min=-1, max=1)

            if self.vi_cfg.distribution == "Laplacian+Gamma" or self.vi_cfg.distribution == "Gaussian+Gamma":
                # prior_gamma_a = self.prior_gamma_a(z_prior).exp()
                prior_gamma_b = self.prior_gamma_b(z_prior).exp().clamp(min=1e-4, max=1e4) * (2 / self.lambda_prior)
                prior_gamma_a = torch.ones_like(prior_gamma_b) * 2.0
                prior_params["gamma_a"] = prior_gamma_a
                prior_params["gamma_b"] = prior_gamma_b
        else:
            prior_params = hyperprior_params.copy()

        return encoder_params, prior_params, hyperprior_params

    def draw_samples(self, z0, z1, encoder_params, transop):
        if self.vi_cfg.total_samples == 1:
            noise = draw_noise_samples(self.vi_cfg.distribution, encoder_params["shift"].shape, z1.device)
        else:
            # If using best of many loss, use this trick to only differentiate through
            # the coefficient with the lowest L2 error.
            with torch.no_grad():
                noise_list = []
                loss_list = []
                for _ in range(self.vi_cfg.total_samples // self.vi_cfg.per_iter_samples):
                    # Generate a new noise sample
                    shape = (
                        self.vi_cfg.per_iter_samples,
                        *encoder_params["shift"].shape,
                    )
                    u = draw_noise_samples(self.vi_cfg.distribution, shape, z1.device)
                    c = reparameterize(
                        self.vi_cfg.distribution,
                        encoder_params,
                        u,
                        self.lambda_prior,
                        self.warmup,
                        self.vi_cfg.normalize_coefficients,
                        self.vi_cfg.normalize_mag,
                    )

                    # Estimate z1 with transport operators
                    with autocast(enabled=False):
                        z1_hat = transop(z0.detach().float().unsqueeze(-1), c.detach()).squeeze(dim=-1)

                        transop_loss = F.mse_loss(
                            z1_hat,
                            z1.repeat(len(z1_hat), *torch.ones(z1.dim(), dtype=int)).detach(),
                            reduction="none",
                        ).mean(dim=-1)
                    noise_list.append(u)
                    loss_list.append(transop_loss)

                # Pick the best sample
                noise_list = torch.cat(noise_list, dim=0)
                loss_list = torch.cat(loss_list, dim=0)
                max_elbo = torch.argmin(loss_list, dim=0).detach()

                # Pick out best noise sample for each batch entry for reparameterization
                noise = noise_list[max_elbo, torch.arange(len(z0))]
        c = reparameterize(
            self.vi_cfg.distribution,
            encoder_params,
            noise,
            self.lambda_prior,
            self.warmup,
            self.vi_cfg.normalize_coefficients,
            self.vi_cfg.normalize_mag,
        )
        return c

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, transop: nn.Module):
        if self.training:
            self.warmup += 2e-4
            if self.warmup > 1.0:
                self.warmup = 1.0

        encoder_params, prior_params, hyperprior_params = self.get_distribution_params(x0, x1)
        samples = self.draw_samples(x0, x1, encoder_params, transop)

        return DistributionData(encoder_params, prior_params, hyperprior_params, samples)
