from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from model.contrastive.config import VariationalEncoderConfig
from model.manifold.l1_inference import infer_coefficients
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

        if self.vi_cfg.enable_thresh_warmup:
            self.thresh_warmup = 0.0
        else:
            self.thresh_warmup = 1.0

        self.initialize_encoder_params(input_size, feat_dim, dictionary_size)
        self.initialize_prior_params(input_size, feat_dim, dictionary_size)

    def initialize_encoder_params(self, input_size, feat_dim, dict_size):
        # If FISTA is the encoder, do not initialize any model weights
        if self.vi_cfg.enable_fista_enc:
            return

        if self.vi_cfg.encode_point_pair:
            enc_input = input_size * 2
        else:
            enc_input = input_size

        enc_out = feat_dim
        if self.vi_cfg.enable_det_enc:
            enc_out = dict_size

        self.enc_feat_extract = nn.Sequential(
            nn.Linear(enc_input, 4 * feat_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * feat_dim, 4 * feat_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * feat_dim, enc_out),
        )

        self.enc_scale = None
        self.enc_shift = None
        if not self.vi_cfg.enable_det_enc:
            self.enc_scale = nn.Linear(feat_dim, dict_size)
            self.enc_shift = nn.Linear(feat_dim, dict_size)

    def initialize_prior_params(self, input_size, feat_dim, dict_size):
        self.prior_params = {}
        self.prior_params["logscale"] = torch.log(torch.tensor(self.vi_cfg.scale_prior))
        self.prior_params["shift"] = torch.tensor(self.vi_cfg.shift_prior)

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

    def fista_encoder(self, x0, x1, psi, prior_params, curr_iter):
        if self.vi_cfg.enable_max_sampling and curr_iter > self.vi_cfg.max_sample_start_iter:
            noise = self.max_elbo_sample(prior_params, psi.detach(), x0, x1)
        else:
            noise = draw_noise_samples(self.vi_cfg.distribution, prior_params["shift"].shape, x0.device)
        c = reparameterize(self.vi_cfg.distribution, prior_params, noise, self.thresh_warmup * self.lambda_prior)

        x0_flat, x1_flat = x0.reshape(-1, x0.shape[-1]), x1.reshape(-1, x1.shape[-1])
        with autocast(enabled=False):
            _, c_refine = infer_coefficients(
                x0_flat.float().detach(),
                x1_flat.float().detach(),
                psi.float().detach(),
                self.vi_cfg.fista_lambda,
                max_iter=self.vi_cfg.fista_num_iters,
                num_trials=1,
                lr=1e-2,
                decay=0.99,
                device=x0.device,
                c_init=c.clone().detach().reshape(1, -1, self.dictionary_size),
                c_scale=self.vi_cfg.fista_var_reg_scale if self.vi_cfg.enable_fista_var_reg else None,
                c_scale_weight=self.vi_cfg.fista_var_reg_weight,
            )

        # Sometimes FISTA can result in NaN values in inference, handle them here.
        c = c_refine.detach().clamp(min=-2.0, max=2.0).nan_to_num(nan=0.01).reshape(c.shape)
        return {"shift": c, "logscale": torch.ones_like(c) * self.prior_params["logscale"]}

    def get_prior_params(self, x0, curr_iter=100000, x1=None):
        prior_params = {}
        hyperprior_params = {}
        if len(x0.shape) > 2:
            hyperprior_params["logscale"] = (
                torch.ones((*x0.shape[:2], self.dictionary_size), device=x0.device) * self.prior_params["logscale"]
            )
            hyperprior_params["shift"] = (
                torch.ones((*x0.shape[:2], self.dictionary_size), device=x0.device) * self.prior_params["shift"]
            )
        else:
            hyperprior_params["logscale"] = (
                torch.ones((len(x0), self.dictionary_size), device=x0.device) * self.prior_params["logscale"]
            )
            hyperprior_params["shift"] = (
                torch.ones((len(x0), self.dictionary_size), device=x0.device) * self.prior_params["shift"]
            )
        if self.vi_cfg.enable_learned_prior:
            z_prior = self.prior_feat_extract(x0)
            prior_params["logscale"] = self.prior_scale(z_prior).clamp(max=3)
            prior_params["logscale"] += torch.log(
                torch.ones_like(prior_params["logscale"]) * self.vi_cfg.scale_prior
            )
            if self.vi_cfg.enable_prior_shift:
                prior_params["shift"] = self.prior_shift(z_prior).clamp(min=-5.0, max=5.0)
            else:
                prior_params["shift"] = hyperprior_params["shift"]

            if self.vi_cfg.enable_prior_warmup:
                warmup = min(curr_iter / self.vi_cfg.prior_warmup_iters, 1.0)
                #warmup = (5e-2) ** (1 - warmup)
                prior_params["logscale"] = (
                    warmup * prior_params["logscale"] + (1 - warmup) * hyperprior_params["logscale"]
                )
                prior_params["shift"] = (
                    warmup * prior_params["shift"]
                    + (1 - warmup)
                    * hyperprior_params["shift"]
                    * torch.sign(warmup * prior_params["shift"]).detach()
                )
        else:
            prior_params = hyperprior_params.copy()
        return prior_params, hyperprior_params

    # Return encoder, prior, and hyperprior params. In case where the prior is fixed and not learned,
    # the hyperprior is equal to the pror.
    def get_distribution_params(
        self, x0, x1, psi, curr_iter
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        prior_params, hyperprior_params = self.get_prior_params(x0, curr_iter=curr_iter, x1=x1)

        encoder_params = {}
        if self.vi_cfg.enable_fista_enc:
            encoder_params = self.fista_encoder(x0, x1, psi, prior_params, curr_iter)
        elif self.vi_cfg.enable_det_enc:
            encoder_params["logscale"] = torch.log(torch.ones_like(prior_params["logscale"]) * self.vi_cfg.scale_prior)
            if self.vi_cfg.encode_point_pair:
                enc_input = torch.cat((x0, x1), dim=-1)
            else:
                enc_input = x0
            encoder_params["shift"] = self.enc_feat_extract(enc_input).clamp(min=-5.0, max=5.0)
        else:
            if self.vi_cfg.encode_point_pair:
                enc_input = torch.cat((x0, x1), dim=-1)
            else:
                enc_input = x0

            z_enc = self.enc_feat_extract(enc_input)
            encoder_params["logscale"] = self.enc_scale(z_enc).clamp(min=-100, max=3)
            encoder_params["logscale"] += torch.log(
                torch.ones_like(encoder_params["logscale"]) * self.vi_cfg.scale_prior
            )
            encoder_params["shift"] = self.enc_shift(z_enc).clamp(min=-5.0, max=5.0)
            encoder_params["shift"] += torch.ones_like(encoder_params["shift"]) * self.vi_cfg.shift_prior

        return encoder_params, prior_params, hyperprior_params

    def prior_sample(self, x, curr_iter=100000, distribution_params=None) -> torch.Tensor:
        if distribution_params is None:
            distribution_params, _ = self.get_prior_params(x, curr_iter=curr_iter)
        noise = draw_noise_samples(self.vi_cfg.distribution, distribution_params["logscale"].shape, x.device)
        samples = reparameterize(
            self.vi_cfg.distribution, distribution_params, noise, self.thresh_warmup * self.lambda_prior
        )
        return samples

    def max_elbo_sample(self, encoder_params, transop, x0, x1) -> torch.Tensor:
        with torch.no_grad():
            noise_list = []
            loss_list = []
            l1_list = []
            noise_shape = [self.vi_cfg.samples_per_iter] + list(encoder_params['logscale'].shape)
            for _ in range(self.vi_cfg.total_num_samples // self.vi_cfg.samples_per_iter):
                noise = draw_noise_samples(self.vi_cfg.distribution, noise_shape, x0.device)
                c = reparameterize(
                    self.vi_cfg.distribution, encoder_params, noise, self.thresh_warmup * self.lambda_prior
                )
                # s x b x d
                x1_hat = transop(x0.float(), c, transop_grad=False)
                # s x b
                transop_loss = torch.nn.functional.mse_loss(
                    x1_hat, x1.unsqueeze(0).expand(len(x1_hat), -1, -1), reduction="none"
                ).mean(dim=-1)

                noise_list.append(noise.squeeze(1))
                loss_list.append(transop_loss)
                l1_list.append(c.abs().sum(dim=-1).squeeze(1))

            # S x b x m
            noise_list = torch.cat(noise_list, dim=0)
            # S x b
            loss_list = torch.cat(loss_list, dim=0)
            # S x b
            l1_list = torch.cat(l1_list, dim=0)
            # b
            max_elbo = torch.argmin(loss_list + self.vi_cfg.max_sample_l1_penalty * l1_list, dim=0).detach()
            optimal_noise = noise_list[max_elbo, torch.arange(len(max_elbo))]
            return optimal_noise

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, transop: nn.Module, curr_iter=0):
        if self.training:
            self.thresh_warmup += 5.0e-5
            if self.thresh_warmup > 1.0:
                self.thresh_warmup = 1.0

        encoder_params, prior_params, hyperprior_params = self.get_distribution_params(
            x0, x1, transop.psi.detach(), curr_iter
        )

        if self.vi_cfg.enable_fista_enc or self.vi_cfg.enable_det_enc:
            samples = encoder_params["shift"]
        else:
            if self.vi_cfg.enable_max_sampling and curr_iter >= self.vi_cfg.max_sample_start_iter:
                noise = self.max_elbo_sample(encoder_params, transop, x0, x1)
            else:
                noise = draw_noise_samples(self.vi_cfg.distribution, encoder_params["shift"].shape, x0.device)
            samples = reparameterize(
                self.vi_cfg.distribution, encoder_params, noise, self.thresh_warmup * self.lambda_prior
            )

        return DistributionData(encoder_params, prior_params, hyperprior_params, samples)
