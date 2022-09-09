"""
Utility functions used for computing metrics.

@Filename    metric_utils.py
@Author      Kion
@Created     09/01/22
"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.manifold import TSNE


def plot_tsne(features: np.array, labels: np.array) -> Figure:
    fig = plt.figure(figsize=(8, 8))
    feat_embed = TSNE(n_components=2, init="random", perplexity=3).fit_transform(features)
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


def transop_plots(coefficients: np.array, psi: np.array) -> Dict[str, Figure]:
    psi_norms = ((psi.reshape(len(psi), -1))**2).sum(dim=-1).detach().cpu().numpy()
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

    figure_dict = {
        f"psi_mag_iter": psi_mag_fig,
        f"coeff_use_iter": coeff_use_fig,
        f"psi_use_iter": psi_use_fig,
    }

    return figure_dict
