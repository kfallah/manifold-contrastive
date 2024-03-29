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
):
    if distribution == "Laplacian":
        assert "logscale" in distribution_params.keys() and "shift" in distribution_params.keys()
        logscale, shift = distribution_params["logscale"], distribution_params["shift"]
        scale = torch.exp(logscale)
        eps = -scale * torch.sign(noise) * torch.log((1.0 - 2.0 * torch.abs(noise)).clamp(min=1e-6, max=1e6))
        c = shift + eps
    if distribution == "Gaussian":
        assert "logscale" in distribution_params.keys() and "shift" in distribution_params.keys()
        logscale, shift = distribution_params["logscale"], distribution_params["shift"]
        scale = torch.exp(0.5 * logscale)
        eps = scale * noise
        c = shift + eps

    # In the case where there are multiple noise samples per datum
    # Used for max elbo sampling
    if len(noise.shape) != len(logscale.shape):
        repeat_len = torch.ones(len(logscale.shape), dtype=int)
        logscale = logscale.view(1, *logscale.shape).repeat(len(noise), *repeat_len)
        shift = shift.view(1, *shift.shape).repeat(len(noise), *repeat_len)

    # We do this weird detaching pattern because in certain cases we want gradient to flow through lambda_
    # In the case where lambda_ is constant, this is the same as c_thresh.detach() in the final line.
    c_thresh = soft_threshold(eps.detach(), lambda_prior)
    non_zero = torch.nonzero(c_thresh, as_tuple=True)
    c_thresh[non_zero] = shift[non_zero].detach() + c_thresh[non_zero]
    c = c + c_thresh - c.detach()
    return c


def compute_kl(
    distribution: str,
    encoder_params: Dict[str, torch.Tensor],
    prior_params: Dict[str, torch.Tensor],
    detach_shift: bool = False,
):
    kl_loss = 0.0
    assert "shift" in encoder_params.keys() and "logscale" in encoder_params.keys()
    assert "shift" in prior_params.keys() and "logscale" in prior_params.keys()
    encoder_shift, encoder_logscale = (
        encoder_params["shift"],
        encoder_params["logscale"],
    )
    if detach_shift:
        encoder_shift = encoder_shift.detach()
    prior_shift, prior_logscale = prior_params["shift"], prior_params["logscale"]
    encoder_scale, prior_scale = torch.exp(encoder_logscale), torch.exp(prior_logscale)

    if distribution == "Laplacian":
        encoder_scale, prior_scale = torch.exp(encoder_logscale), torch.exp(prior_logscale)
        laplace_kl = ((encoder_shift - prior_shift).abs() / prior_scale) + prior_logscale - encoder_logscale - 1
        laplace_kl += (encoder_scale / prior_scale) * (-((encoder_shift - prior_shift).abs() / encoder_scale)).exp()
        kl_loss += laplace_kl.sum(dim=-1)

    if distribution == "Gaussian":
        gauss_kl = (encoder_scale + ((encoder_shift - prior_shift) ** 2)) / (2 * prior_scale)
        gauss_kl += 0.5 * (prior_logscale - encoder_logscale - 1)
        kl_loss += gauss_kl.sum(dim=-1)


    return kl_loss
