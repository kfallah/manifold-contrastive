import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_threshold(c, zeta):
    return F.relu(torch.abs(c) - zeta) * torch.sign(c)


def compute_loss(c, x0, x1, psi):
    T = torch.einsum("tbm,mpk->tbpk", c, psi)
    x1_hat = (torch.matrix_exp(T) @ x0.unsqueeze(-1)).squeeze(-1)
    loss = F.mse_loss(x1_hat, x1, reduction="none")
    return loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def infer_coefficients(x0, x1, psi, zeta, max_iter=300, tol=1e-5, num_trials=100, device="cpu", c_init=0.1):
    c = nn.Parameter(
        torch.mul(torch.randn((num_trials, len(x0), len(psi)), device=device), c_init), requires_grad=True
    )
    c_opt = torch.optim.SGD([c], lr=5e-2, nesterov=False, momentum=0.9)
    opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(c_opt, gamma=0.985)
    batch_size = len(x1)
    x1 = x1.repeat(num_trials, *torch.ones(x1.dim(), dtype=int))
    change = torch.ones((num_trials, len(x0)), device=device) * (1.0e99)
    change_idx = change > tol
    k = 0

    while k < max_iter and change_idx.any():
        old_coeff = c.clone()

        c_opt.zero_grad()
        loss = compute_loss(c, x0, x1, psi)
        (loss.mean(dim=(-1)) * change_idx).sum().backward()
        c_opt.step()
        opt_scheduler.step()

        with torch.no_grad():
            c.data = soft_threshold(c, get_lr(c_opt) * zeta)
        change = torch.norm(c.data - old_coeff, dim=(-1)) / (torch.norm(old_coeff, dim=(-1)) + 1e-9)
        change_idx = change > tol
        k += 1

    trial_idx = torch.argmin(
        loss.detach().cpu().mean(dim=(-1)) + zeta * torch.abs(c.detach().cpu()).sum(dim=-1), dim=0
    )
    c = c[trial_idx, torch.arange(batch_size)].data
    return (loss.detach().cpu(), get_lr(c_opt), k), c.data


def infer_coefficients_smoothed(
    x0, x1, psi, zeta, max_iter=800, tol=1e-6, num_trials=100, device="cpu", init_sigma=1, c_scale=2
):
    """
    Applies the smoothing to inference from (Sohl-Dickstein 2017). Currently only works with 1 operator.
    A bit sus because it relies on complex-valued autograd.
    """
    c = nn.Parameter((torch.randn((num_trials, len(x0)), device=device)) * c_scale, requires_grad=True)
    c_opt = torch.optim.SGD([c], lr=1e-2, nesterov=False, momentum=0.9)
    opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(c_opt, gamma=0.995)
    change = torch.ones((num_trials, len(x0)), device=device) * (1.0e99)
    change_idx = change > tol
    dim = x1.shape[1]
    k, sigma = 0, init_sigma
    l, U = np.linalg.eig(psi.detach().cpu().numpy())
    l, U = torch.tensor(l, device=device, dtype=torch.cfloat), torch.tensor(U, device=device, dtype=torch.cfloat)
    x0, x1 = x0.unsqueeze(0), x1.unsqueeze(0)
    loss_list, c_list = [], []

    while k < max_iter and change_idx.any():
        old_coeff = c.clone()

        c_opt.zero_grad()
        eig = torch.exp(c[:, :, None, None] * l[None, None, :, None]) * torch.eye(dim, device=device)[None, None]
        if sigma > 0:
            eig *= torch.exp((sigma / 2) * l[None, None, :, None] ** 2) * torch.eye(dim, device=device)[None, None]
        T = torch.real(U @ eig @ U.conj().T)

        loss = torch.linalg.norm(x1 - T @ x0, axis=(-1, -2)) ** 2
        loss_list.append(loss.detach().cpu())
        c_list.append(c.detach().cpu())
        (loss.mean(dim=(-1, -2)) * change_idx).sum().backward()
        c_opt.step()
        opt_scheduler.step()
        with torch.no_grad():
            c.data = soft_threshold(c, get_lr(c_opt) * zeta)

        change = torch.abs(c.data - old_coeff) / (torch.abs(old_coeff) + 1e-9)
        change_idx = change > tol
        k += 1
        sigma *= 0.99

    loss_list, c_list = torch.stack(loss_list), torch.stack(c_list)
    trial_idx = torch.argmin(loss_list[-1] + zeta * torch.abs(c.detach().cpu()).sum(), dim=0)
    c = c[trial_idx, torch.arange(x1.shape[1])].data
    loss_list = loss_list[:, trial_idx, torch.arange(x1.shape[1])]
    c_list = c_list[:, trial_idx, torch.arange(x1.shape[1])]

    return (loss_list, get_lr(c_opt), k, c_list), c


def compute_arc_length(psi, c, t, x0, device="cpu"):
    batch_size = x0.shape[0]
    A = psi[None] * c[:, None, None]
    arc_len = torch.zeros((batch_size), device=device)
    t_int = t[1] - t[0]
    for t_use in t:
        Tx = torch.matrix_exp(A * t_use) @ x0.unsqueeze(-1)
        A_Tx = torch.matmul(A, Tx).squeeze()
        arc_len = arc_len + t_int * torch.norm(A_Tx, dim=1)

    return arc_len.detach().cpu().numpy()
