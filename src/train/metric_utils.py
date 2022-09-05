"""
Utility functions used for computing metrics.

@Filename    metric_utils.py
@Author      Kion
@Created     09/01/22
"""

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
