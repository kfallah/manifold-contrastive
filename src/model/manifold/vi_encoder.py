from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from model.contrastive.config import VariationalEncoderConfig
from model.manifold.reparameterize import draw_noise_samples, reparameterize
from model.type import DistributionData


class CoefficientEncoder(nn.Module):
    def __init__(
        self,
        vi_cfg: VariationalEncoderConfig,
        input_size: int,
        dictionary_size: int,
        lambda_prior: float,
    ):
        super(CoefficientEncoder, self).__init__()

        self.vi_cfg = vi_cfg
        feat_dim = self.vi_cfg.feature_dim
        self.dictionary_size = dictionary_size
        self.lambda_prior = lambda_prior

        self.initialize_encoder_params(input_size, feat_dim, dictionary_size)
        self.initialize_prior_params(input_size, feat_dim, dictionary_size)

    def initialize_encoder_params(self, input_size, feat_dim, dict_size):
        self.enc_feat_extract = nn.Sequential(
                nn.Linear(2*input_size, 4 * feat_dim),
                nn.LeakyReLU(),
                nn.Linear(4 * feat_dim, 4 * feat_dim),
                nn.LeakyReLU(),
                nn.Linear(4 * feat_dim, feat_dim),
            )
        self.enc_scale = nn.Linear(feat_dim, dict_size)
        self.enc_shift = nn.Linear(feat_dim, dict_size)

    def initialize_prior_params(self, input_size, feat_dim, dict_size):
        self.prior_params = {}
        self.prior_params["logscale"] = torch.log(torch.tensor(self.vi_cfg.scale_prior))
        self.prior_params["shift"] = 0.0

        if self.vi_cfg.enable_learned_prior:
            self.prior_feat_extract = nn.Sequential(
                nn.Linear(input_size, 4 * feat_dim),
                nn.LeakyReLU(),
                nn.Linear(4 * feat_dim, 2 * feat_dim),
                nn.LeakyReLU(),
                nn.Linear(2 * feat_dim, feat_dim),
            )
            self.prior_scale = nn.Linear(feat_dim, dict_size)
            if self.vi_cfg.enable_prior_shift:
                self.prior_shift = nn.Linear(feat_dim, dict_size)

    # Return encoder, prior, and hyperprior params. In case where the prior is fixed and not learned,
    # the hyperprior is equal to the pror.
    def get_distribution_params(self, x0, x1) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        encoder_params = {}
        z_enc = self.enc_feat_extract(torch.cat((x0, x1), dim=-1))

        encoder_params["logscale"] = self.enc_scale(z_enc).clamp(min=-100, max=2)
        encoder_params["logscale"] += torch.log(
            torch.ones_like(encoder_params["logscale"]) * self.vi_cfg.scale_prior
        )
        encoder_params["shift"] = self.enc_shift(z_enc).clamp(min=-5.0, max=5.0)

        prior_params = {}
        hyperprior_params = {}
        hyperprior_params["logscale"] = torch.ones_like(encoder_params["logscale"]) * self.prior_params["logscale"]
        hyperprior_params["shift"] = torch.zeros_like(encoder_params["shift"])
        if self.vi_cfg.enable_learned_prior:
            z_prior = self.prior_feat_extract(x0)
            prior_params["logscale"] = self.prior_scale(z_prior).clamp(max=2)
            prior_params["logscale"] += torch.log(
                torch.ones_like(prior_params["logscale"]) * self.vi_cfg.scale_prior
            )
            if self.vi_cfg.enable_prior_shift:
                prior_params["shift"] = self.prior_shift(z_prior).clamp(min=-2.0, max=2.0)
        else:
            prior_params = hyperprior_params.copy()

        return encoder_params, prior_params, hyperprior_params

    def max_elbo_sample(self, encoder_params, psi, x0, x1) -> torch.Tensor:
        logscale, shift = encoder_params['logscale'], encoder_params['shift']
        with torch.no_grad():
            noise_list = []
            loss_list = []
            l1_list = []
            logscale_expanded = logscale.unsqueeze(0).repeat(self.vi_cfg.samples_per_iter, 1, 1, 1)
            for _ in range(self.vi_cfg.total_num_samples // self.vi_cfg.samples_per_iter):
                noise = draw_noise_samples(self.vi_cfg.distribution, logscale_expanded.shape, logscale_expanded.device)
                c = reparameterize(self.vi_cfg.distribution, encoder_params, noise, self.lambda_prior)
                T = torch.matrix_exp(torch.einsum("sblm,mpk->sblpk", c, psi))
                x1_hat = (T @ x0.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
                transop_loss = torch.nn.functional.mse_loss(
                    x1_hat, x1.unsqueeze(0).repeat(self.vi_cfg.samples_per_iter, 1, 1, 1), reduction="none"
                ).mean(dim=-1)

                noise_list.append(noise)
                loss_list.append(transop_loss)
                l1_list.append(c.abs().sum(dim=-1))

            noise_list = torch.cat(noise_list, dim=0)
            loss_list = torch.cat(loss_list, dim=0)
            l1_list = torch.cat(l1_list, dim=0)
            max_elbo = torch.argmin(loss_list + self.vi_cfg.max_sample_l1_penalty * l1_list, dim=0).detach()
            # Pick out best noise sample for each batch entry for reparameterization
            max_elbo = max_elbo.reshape(-1)
            n_samples, n_batch, n_seq, n_feat = noise_list.shape
            noise_list = noise_list.reshape(n_samples, -1, n_feat)
            optimal_noise = noise_list[max_elbo, torch.arange(n_batch * n_seq)]
            return optimal_noise.reshape(n_batch, n_seq, n_feat)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, transop: nn.Module, curr_iter = 0):
        encoder_params, prior_params, hyperprior_params = self.get_distribution_params(x0, x1)

        if self.vi_cfg.enable_max_sampling and curr_iter > self.vi_cfg.max_sample_start_iter:
            noise = self.max_elbo_sample(encoder_params, transop.psi.detach(), x0, x1)
        else:
            noise = draw_noise_samples(self.vi_cfg.distribution, encoder_params['shift'].shape, x0.device)
        samples = reparameterize(self.vi_cfg.distribution, encoder_params, noise, self.lambda_prior)

        return DistributionData(encoder_params, prior_params, hyperprior_params, samples)
