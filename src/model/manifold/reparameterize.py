from base64 import encode
from typing import Dict, Tuple

import torch
from torch.distributions import gamma as gamma


def soft_threshold(z: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu(torch.abs(z) - lambda_) * torch.sign(z)


def draw_noise_samples(distribution: str, shape: Tuple[int], device: torch.device):
    if distribution == "Laplacian" or distribution == "Laplacian+Gamma":
        return torch.rand(shape, device=device) - 0.5
    if distribution == "Gaussian" or distribution == "Gaussian+Gamma":
        return torch.randn(shape, device=device)


def reparameterize(
    distribution: str,
    distribution_params: Dict[str, torch.Tensor],
    noise: torch.Tensor,
    lambda_prior: float,
    warmup: float = 1.0,
    normalize_coefficients: bool = False,
    normalize_mag: float = 1.0,
):
    if distribution == "Laplacian" or distribution == "Laplacian+Gamma":
        assert (
            "logscale" in distribution_params.keys()
            and "shift" in distribution_params.keys()
        )
        logscale, shift = distribution_params["logscale"], distribution_params["shift"]
        if len(noise.shape) >= 3:
            logscale = logscale.view(1, *logscale.shape).expand(len(noise), -1, -1)
            shift = shift.view(1, *shift.shape).expand(len(noise), -1, -1)
        scale = torch.exp(logscale)
        eps = (
            -scale
            * torch.sign(noise)
            * torch.log((1.0 - 2.0 * torch.abs(noise)).clamp(min=1e-6, max=1e6))
        )
        c = shift + eps
    if distribution == "Gaussian" or distribution == "Gaussian+Gamma":
        assert (
            "logscale" in distribution_params.keys()
            and "shift" in distribution_params.keys()
        )
        logscale, shift = distribution_params["logscale"], distribution_params["shift"]
        if len(noise.shape) >= 3:
            logscale = logscale.view(1, *logscale.shape).expand(len(noise), -1, -1)
            shift = shift.view(1, *shift.shape).expand(len(noise), -1, -1)
        scale = torch.exp(0.5 * logscale)
        eps = scale * noise
        c = shift + eps

    if distribution == "Laplacian+Gamma" or distribution == "Gaussian+Gamma":
        assert (
            "gamma_a" in distribution_params.keys()
            and "gamma_b" in distribution_params.keys()
        )
        gamma_a, gamma_b = (
            distribution_params["gamma_a"],
            distribution_params["gamma_b"],
        )
        gamma_distr = gamma.Gamma(gamma_a, gamma_a / (gamma_b + 1e-6))
        if len(noise.shape) >= 3:
            lambda_ = gamma_distr.rsample([len(noise)])
        else:
            lambda_ = gamma_distr.rsample()
    else:
        lambda_ = torch.ones_like(eps) * lambda_prior

    # We do this weird detaching pattern because in certain cases we want gradient to flow through lambda_
    # In the case where lambda_ is constant, this is the same as c_thresh.detach() in the final line.
    c_thresh = soft_threshold(eps.detach() * warmup, lambda_)
    non_zero = torch.nonzero(c_thresh, as_tuple=True)
    c_thresh[non_zero] = (warmup * shift[non_zero].detach()) + c_thresh[non_zero]
    c = c + c_thresh - c.detach()
    if normalize_coefficients:
        c = torch.nn.functional.normalize(c, dim=-1) * normalize_mag
    return c


def compute_kl(
    distribution: str,
    encoder_params: Dict[str, torch.Tensor],
    prior_params: Dict[str, torch.Tensor],
):
    kl_loss = 0.0
    if distribution == "Laplacian" or distribution == "Laplacian+Gamma":
        assert "shift" in encoder_params.keys() and "logscale" in encoder_params.keys()
        assert "shift" in prior_params.keys() and "logscale" in prior_params.keys()
        encoder_shift, encoder_logscale = (
            encoder_params["shift"],
            encoder_params["logscale"],
        )
        prior_shift, prior_logscale = prior_params["shift"], prior_params["logscale"]
        encoder_scale, prior_scale = torch.exp(encoder_logscale), torch.exp(
            prior_logscale
        )
        laplace_kl = (
            ((encoder_shift - prior_shift).abs() / prior_scale)
            + prior_logscale
            - encoder_logscale
            - 1
        )
        laplace_kl += (encoder_scale / prior_scale) * (
            -((encoder_shift - prior_shift).abs() / encoder_scale)
        ).exp()
        kl_loss += laplace_kl.sum(dim=-1).mean()
    if distribution == "Gaussian" or distribution == "Gaussian+Gamma":
        assert "shift" in encoder_params.keys() and "logscale" in encoder_params.keys()
        assert "shift" in prior_params.keys() and "logscale" in prior_params.keys()
        encoder_shift, encoder_logscale = (
            encoder_params["shift"],
            encoder_params["logscale"],
        )
        prior_shift, prior_logscale = prior_params["shift"], prior_params["logscale"]
        encoder_scale = torch.exp(encoder_logscale)
        prior_scale = torch.exp(prior_logscale)
        gauss_kl = (encoder_scale + ((encoder_shift - prior_shift) ** 2)) / (
            2 * prior_scale
        )
        gauss_kl += 0.5 * (prior_logscale - encoder_logscale - 1)
        kl_loss += gauss_kl.sum(dim=-1).mean()

    if distribution == "Laplacian+Gamma" or distribution == "Gaussian+Gamma":
        assert "gamma_a" in encoder_params.keys() and "gamma_b" in encoder_params.keys()
        assert "gamma_a" in prior_params.keys() and "gamma_b" in prior_params.keys()
        enc_gamma_a, enc_gamma_b = encoder_params["gamma_a"], encoder_params["gamma_b"]
        prior_gamma_a, prior_gamma_b = prior_params["gamma_a"], prior_params["gamma_b"]
        gamma_enc = gamma.Gamma(enc_gamma_a, enc_gamma_a / (enc_gamma_b + 1e-6))
        gamma_prior = gamma.Gamma(prior_gamma_a, prior_gamma_a / (prior_gamma_b + 1e-6))
        gamma_kl = torch.distributions.kl.kl_divergence(gamma_enc, gamma_prior).mean()
        kl_loss += gamma_kl

    return kl_loss
