from json import encoder
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.contrastive.config import VariationalEncoderConfig
from model.manifold.reparameterize import compute_kl, draw_noise_samples, reparameterize
from model.type import DistributionData
from torch.cuda.amp import autocast


class VIEncoder(nn.Module):
    def __init__(
        self,
        vi_cfg: VariationalEncoderConfig,
        input_size: int,
        dictionary_size: int,
        lambda_prior: float,
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

        self.initialize_encoder_params(input_size, feat_dim, dictionary_size)
        self.initialize_prior_params(input_size, feat_dim, dictionary_size)

    def initialize_encoder_params(self, input_size, feat_dim, dict_size):
        if self.vi_cfg.encoder_type == "MLP":
            self.enc_feat_extract = nn.Sequential(
                nn.Linear(input_size, 2 * input_size),
                nn.BatchNorm1d(2 * input_size),
                nn.ReLU(),
                nn.Linear(2 * input_size, feat_dim // 2),
            )
            self.enc_aggregate = nn.Sequential(
                nn.Linear(feat_dim, 2 * feat_dim),
                nn.BatchNorm1d(2 * feat_dim),
                nn.ReLU(),
                nn.Linear(2 * feat_dim, feat_dim),
            )
        else:
            raise NotImplementedError

        if (
            self.vi_cfg.distribution == "Laplacian"
            or self.vi_cfg.distribution == "Laplacian+Gamma"
        ):
            self.enc_scale = nn.Linear(feat_dim, dict_size)
            self.enc_shift = nn.Linear(feat_dim, dict_size)

        if self.vi_cfg.distribution == "Laplacian+Gamma":
            self.enc_gamma_a = nn.Linear(feat_dim, dict_size)
            self.enc_gamma_b = nn.Linear(feat_dim, dict_size)

    def initialize_prior_params(self, input_size, feat_dim, dict_size):
        self.prior_params = {}
        if (
            self.vi_cfg.distribution == "Laplacian"
            or self.vi_cfg.distribution == "Laplacian+Gamma"
        ):
            self.prior_params["logscale"] = torch.log(
                torch.tensor(self.vi_cfg.scale_prior)
            )
            self.prior_params["shift"] = 0.0
        if self.vi_cfg.distribution == "Laplacian+Gamma":
            self.prior_params["gamma_a"] = 2.0
            self.prior_params["gamma_b"] = 2.0 / self.lambda_prior

        if self.vi_cfg.prior_type == "Learned":
            raise NotImplementedError()

    def get_distribution_params(
        self, x0, x1
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        encoder_params = {}
        if self.vi_cfg.encoder_type == "MLP":
            z0, z1 = self.enc_feat_extract(x0), self.enc_feat_extract(x1)
            z_enc = self.enc_aggregate(torch.cat((z0, z1), dim=1))
        else:
            raise NotImplementedError()

        if (
            self.vi_cfg.distribution == "Laplacian"
            or self.vi_cfg.distribution == "Laplacian+Gamma"
        ):
            encoder_params["logscale"] = self.enc_scale(z_enc)
            encoder_params["shift"] = self.enc_shift(z_enc)

        if self.vi_cfg.distribution == "Laplacian+Gamma":
            gamma_a = self.enc_gamma_a(z_enc)
            gamma_b = self.enc_gamma_b(z_enc)
            gamma_a.detach().clamp_(min=1e-4, max=1e4)
            gamma_b.detach().clamp_(min=1e-4, max=1e4)
            encoder_params["gamma_a"] = gamma_a
            encoder_params["gamma_b"] = gamma_b

        prior_params = {}
        hyperprior_params = {}
        if (
            self.vi_cfg.distribution == "Laplacian"
            or self.vi_cfg.distribution == "Laplacian+Gamma"
        ):
            prior_params["logscale"] = (
                torch.ones_like(encoder_params["logscale"])
                * self.prior_params["logscale"]
            )
            prior_params["shift"] = torch.zeros_like(encoder_params["shift"])
        if self.vi_cfg.distribution == "Laplacian+Gamma":
            prior_params["gamma_a"] = (
                torch.ones_like(encoder_params["gamma_a"])
                * self.prior_params["gamma_a"]
            )
            prior_params["gamma_b"] = (
                torch.ones_like(encoder_params["gamma_b"])
                * self.prior_params["gamma_b"]
            )

        if self.vi_cfg.prior_type == "Learned":
            raise NotImplementedError()

        return encoder_params, prior_params, hyperprior_params

    def draw_samples(self, z0, z1, encoder_params, transop):
        if self.vi_cfg.total_samples == 1:
            noise = draw_noise_samples(
                self.vi_cfg.distribution, encoder_params["shift"].shape, z1.device
            )
        else:
            # If using best of many loss, use this trick to only differentiate through
            # the coefficient with the lowest L2 error.
            with torch.no_grad():
                noise_list = []
                loss_list = []
                for _ in range(
                    self.vi_cfg.total_samples // self.vi_cfg.per_iter_samples
                ):
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
                    )

                    # Estimate z1 with transport operators
                    with autocast(enabled=False):
                        z1_hat = transop(
                            z0.detach().float().unsqueeze(-1), c.detach()
                        ).squeeze(dim=-1)

                        transop_loss = F.mse_loss(
                            z1_hat,
                            z1.repeat(
                                len(z1_hat), *torch.ones(z1.dim(), dtype=int)
                            ).detach(),
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
        )
        return c

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, transop: nn.Module):
        self.warmup += 1e-3
        if self.warmup > 1.0:
            self.warmup = 1.0

        encoder_params, prior_params, hyperprior_params = self.get_distribution_params(
            x0, x1
        )
        samples = self.draw_samples(x0, x1, encoder_params, transop)

        return DistributionData(encoder_params, prior_params, samples)
