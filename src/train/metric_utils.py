"""
Utility functions used for computing metrics.

@Filename    metric_utils.py
@Author      Kion
@Created     09/01/22
"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.manifold import TSNE


def plot_tsne(features: np.array, labels: np.array) -> Figure:
    fig = plt.figure(figsize=(8, 8))
    feat_embed = TSNE(n_components=2, init="random", perplexity=3).fit_transform(
        features
    )
    for label in np.unique(labels):
        label_idx = labels == label
        plt.scatter(*feat_embed[label_idx].T)
    return fig


def plot_log_spectra(features: np.array) -> Figure:
    fig = plt.figure(figsize=(8, 8))
    _, s, _ = np.linalg.svd(features)
    log_spectra = np.log(s)
    plt.plot(log_spectra)
    plt.xlabel("Dimension", fontsize=18)
    plt.ylabel("Log Singular Value", fontsize=18)
    return fig, log_spectra


def sweep_psi_path_plot(psi: torch.tensor, z0: np.array, c_mag: int) -> Figure:
    z = z0[0].float().to(psi.device)[:psi.shape[-1]]

    # z = model.backbone(x_gpu[0])[0]
    # z = torch.tensor(z0[0][0]).to(default_device)
    # psi = model.contrastive_header.transop_header.transop.psi
    psi_norm = (psi.reshape(len(psi), -1) ** 2).sum(dim=-1)
    psi_idx = torch.argsort(psi_norm)
    latent_dim = len(z)

    fig, ax = plt.subplots(nrows=15, ncols=3, figsize=(16, 35))
    plt.subplots_adjust(hspace=0.4, top=0.9)

    for i in range(ax.size):
        row = int(i / 3)
        column = int(i % 3)
        curr_psi = psi_idx[-(i + 1)]

        coeff = torch.linspace(-c_mag, c_mag, 30, device=psi.device)
        T = torch.matrix_exp(coeff[:, None, None] * psi[None, curr_psi])
        z1_hat = (T @ z).squeeze(dim=-1)

        for z_dim in range(latent_dim):
            ax[row, column].plot(
                np.linspace(-c_mag, c_mag, 30),
                z1_hat[:, z_dim].detach().cpu().numpy(),
            )
        ax[row, column].title.set_text(
            f"Psi {curr_psi} - F-norm: {psi_norm[curr_psi]:.2E}"
        )

    return fig


def transop_plots(
    coefficients: np.array, psi: torch.tensor, z0: np.array
) -> Dict[str, Figure]:
    psi_norms = ((psi.reshape(len(psi), -1)) ** 2).sum(dim=-1).detach().cpu().numpy()
    count_nz = np.zeros(len(psi) + 1, dtype=int)
    total_nz = np.count_nonzero(coefficients, axis=1)
    for z in range(len(total_nz)):
        count_nz[total_nz[z]] += 1
    number_operator_uses = np.count_nonzero(coefficients, axis=0) / len(coefficients)

    psi_mag_fig = plt.figure(figsize=(20, 4))
    plt.bar(np.arange(len(psi)), psi_norms, width=1)
    plt.xlabel("Transport Operator Index", fontsize=18)
    plt.ylabel("F-Norm", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("F-Norm of Transport Operators", fontsize=20)

    coeff_use_fig = plt.figure(figsize=(20, 4))
    plt.bar(np.arange(len(psi) + 1), count_nz, width=1)
    plt.xlabel("Number of Coefficients Used per Point Pair", fontsize=18)
    plt.ylabel("Occurences", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Number of Non-Zero Coefficients", fontsize=20)

    psi_use_fig = plt.figure(figsize=(20, 4))
    plt.bar(np.arange(len(psi)), number_operator_uses, width=1)
    plt.xlabel("Percentage of Point Pairs an Operator is Used For", fontsize=18)
    plt.ylabel("% Of Point Pairs", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Transport Operator Index", fontsize=20)

    psi_eig_plt = plt.figure(figsize=(8, 8))
    L = torch.linalg.eigvals(psi.detach())
    plt.scatter(
        torch.real(L).detach().cpu().numpy(), torch.imag(L).detach().cpu().numpy()
    )
    plt.xlabel("Real Components of Eigenvalues", fontsize=18)
    plt.ylabel("Imag Components of Eigenvalues", fontsize=18)

    psi_sweep_sub1c_fig = sweep_psi_path_plot(psi.detach(), z0, 0.1)
    psi_sweep_1c_fig = sweep_psi_path_plot(psi.detach(), z0, 1)
    psi_sweep_5c_fig = sweep_psi_path_plot(psi.detach(), z0, 5)

    figure_dict = {
        "psi_mag_iter": psi_mag_fig,
        "coeff_use_iter": coeff_use_fig,
        "psi_use_iter": psi_use_fig,
        "psi_eig_plt": psi_eig_plt,
        "psi_sweep_sub1c_fig": psi_sweep_sub1c_fig,
        "psi_sweep_1c": psi_sweep_1c_fig,
        "psi_sweep_5c": psi_sweep_5c_fig,
    }

    return figure_dict
