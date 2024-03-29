"""
Module responsible for logging all metrics related to training.

@Filename    metric_logger.py
@Author      Kion
@Created     09/01/22
"""

import logging
import math
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from model.model import Model, ModelOutput
from train.config import MetricLoggerConfig
from train.metric_utils import plot_log_spectra, plot_tsne, transop_plots

log = logging.getLogger(__name__)


class MetricLogger:
    def __init__(
        self,
        metric_logger_cfg: MetricLoggerConfig,
        model: Model,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        self.cfg = metric_logger_cfg
        self.model = model
        self.scheduler = scheduler
        self.metadata_avg = {}
        self.feature_cache = []
        self.label_cache = []
        self.c_cache = []

    def enable_feature_cache(self):
        return self.cfg.enable_log_spectra_plot or self.cfg.enable_tsne_plot or self.cfg.enable_transop_logging

    def save_figure(self, figure_name: str, fig: Figure):
        save_path = os.getcwd() + "/figures/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig(save_path + figure_name + ".png", bbox_inches="tight")

    def log_metrics(
        self,
        curr_iter: int,
        curr_epoch: int,
        model_output: ModelOutput,
        labels: torch.Tensor,
        loss_metadata: Dict[str, float],
    ) -> None:
        if self.cfg.enable_wandb_logging:
            assert wandb.run is not None, "wandb not initialized!"
        if self.enable_feature_cache():
            self.feature_cache.extend(model_output.header_input.feature_0.detach().cpu().numpy())
            self.feature_cache = self.feature_cache[-self.cfg.feature_cache_size :]
            self.label_cache.extend(labels.numpy())
            self.label_cache = self.label_cache[-self.cfg.feature_cache_size :]

            if self.cfg.enable_transop_logging:
                current_c = model_output.header_output.distribution_data.samples.detach().cpu().numpy()
                self.c_cache.extend(current_c.reshape(-1, current_c.shape[-1]))
                self.c_cache = self.c_cache[-self.cfg.feature_cache_size :]
        metrics = {}

        # Logging training heuristics
        if self.cfg.enable_loss_logging:
            if len(self.metadata_avg) == 0:
                # In the first iteration use the loss metadata directly
                self.metadata_avg = dict(("loss/" + key, value) for (key, value) in loss_metadata.items())
            else:
                # Get moving average for the iterations between the loss metadata
                for key in loss_metadata.keys():
                    if "loss/" + key not in self.metadata_avg.keys():
                        self.metadata_avg["loss/" + key] = 0.0
                    self.metadata_avg["loss/" + key] += loss_metadata[key] / self.cfg.loss_log_freq

            if curr_iter % self.cfg.loss_log_freq == 0:
                metrics.update(self.metadata_avg)
                if self.cfg.enable_console_logging:
                    format_loss = [f"{key}: {self.metadata_avg['loss/' + key]:.3E}" for key in loss_metadata.keys()]
                    log.info(f"[Epoch {curr_epoch}, iter {curr_iter}]: " + ", ".join(format_loss))
                    # Reset all values to zero for the next loop
                    self.metadata_avg = dict.fromkeys(self.metadata_avg, 0.0)

        # Logging optimizer LR
        if self.cfg.enable_optimizer_logging and curr_iter % self.cfg.optimizer_log_freq == 0:
            optim_lr = self.scheduler.get_last_lr()[0]
            metrics["meta/optim_lr"] = optim_lr
        # Logging collapse level
        if self.cfg.enable_collapse_logging and curr_iter % self.cfg.collapse_log_freq == 0:
            feat = torch.tensor(np.array(self.feature_cache)).float()
            feat_norm = torch.nn.functional.normalize(feat, dim=1)
            feat_var = torch.std(feat_norm, dim=0)
            feat_collapse = max(0.0, 1 - math.sqrt(len(feat_var)) * feat_var.mean().item())
            metrics["meta/feat_collapse"] = feat_collapse
            _, S, _ = torch.linalg.svd(feat, full_matrices=False)
            norm_S = S / S.sum()
            effective_rank = (-norm_S * torch.log(norm_S)).sum()
            stable_rank = (S**2).sum() / (max(S) ** 2)
            metrics["meta/effective_rank"] = effective_rank
            metrics["meta/stable_rank"] = stable_rank

            if self.cfg.enable_console_logging:
                log.info(
                    f"[Epoch {curr_epoch}, iter {curr_iter}]: Feature collapse: {feat_collapse:.2f}, Effective Rank: {effective_rank:.2f}, Stable Rank: {stable_rank:.2f}"
                )

        if self.cfg.enable_transop_logging and curr_iter % self.cfg.transop_log_freq == 0:
            assert (
                self.model.model_cfg.header_cfg.enable_transop_header
                and model_output.header_output.distribution_data is not None
            )
            header_dict = model_output.header_output.header_dict
            psi = self.model.contrastive_header.transop_header.transop.get_psi()
            # Construct full dimension matrix from block diagonal
            if len(psi.shape) >= 4:
                psi = torch.stack([torch.block_diag(*psi[i]) for i in range(len(psi))])
            c = np.array(self.c_cache)
            coeff_nz = np.count_nonzero(c, axis=0)
            nz_tot = np.count_nonzero(coeff_nz)
            total_nz = np.count_nonzero(c, axis=1)
            avg_feat_norm = np.linalg.norm(np.array(self.feature_cache), axis=-1).mean()
            dist_bw_point_pairs = F.mse_loss(header_dict["transop_z1"], header_dict["transop_z0"]).item()
            transop_dist = (
                F.mse_loss(
                    header_dict["transop_z1"],
                    header_dict["transop_z1hat"],
                    reduction="none",
                ).sum(dim=-1)
                / (
                    F.mse_loss(
                        header_dict["transop_z1"],
                        header_dict["transop_z0"],
                        reduction="none",
                    ).sum(dim=-1)
                    + 1e-6
                )
            ).mean(dim=-1)
            mean_dist_improvement = transop_dist.mean().item()

            psi_mag = torch.norm(psi.data.reshape(len(psi.data), -1), dim=-1)
            to_metrics = {
                "transop/avg_transop_mag": psi_mag.mean(),
                "transop/total_transop_used": nz_tot,
                "transop/avg_transop_used": total_nz.mean(),
                "transop/avg_coeff_mag": np.abs(c[np.abs(c) > 0]).mean(),
                "transop/avg_feat_norm": avg_feat_norm,
                "transop/dist_bw_point_pairs": dist_bw_point_pairs,
                "transop/mean_dist_improvement": mean_dist_improvement,
            }
            if self.cfg.enable_console_logging:
                log.info(
                    f"[TO iter {curr_iter}]:"
                    + f", dist improve: {mean_dist_improvement:.3E}"
                    + f", avg # to used: {total_nz.mean():.2f}/{len(psi)}"
                    + f", avg coeff mag: {to_metrics['transop/avg_coeff_mag']:.3f}"
                    + f", dist bw pp: {dist_bw_point_pairs:.3E}"
                    + f", average to mag: {psi_mag.mean():.3E}"
                    + f", avg feat norm: {avg_feat_norm:.2E}"
                )
                if self.model.model_cfg.header_cfg.transop_header_cfg.enable_variational_inference:
                    distr_data = model_output.header_output.distribution_data
                    scale = torch.exp(distr_data.encoder_params["logscale"])
                    shift = distr_data.encoder_params["shift"]
                    log.info(
                        f"[Encoder params]: "
                        + f"min scale: {scale.abs().min():.2E}"
                        + f", max scale: {scale.abs().max():.2E}"
                        + f", mean scale: {scale.mean():.2E}"
                        + f", min shift: {shift.abs().min():.2E}"
                        + f", max shift: {shift.abs().max():.2E}"
                        + f", mean shift: {shift.abs().mean():.2E}"
                    )
                    if self.model.model_cfg.header_cfg.transop_header_cfg.vi_cfg.enable_learned_prior:
                        prior_scale = torch.exp(distr_data.prior_params["logscale"])
                        prior_shift = distr_data.prior_params["shift"]
                        log.info(
                            f"[Prior params]: "
                            + f"min scale: {prior_scale.abs().min():.3E}"
                            + f", max scale: {prior_scale.abs().max():.3E}"
                            + f", mean scale: {prior_scale.mean():.3E}"
                            + f", min shift: {prior_shift.abs().min():.3E}"
                            + f", max shift: {prior_shift.abs().max():.3E}"
                            + f", mean shift: {prior_shift.abs().mean():.3E}"
                        )

            # Generate transport operator plots
            fig_dict = transop_plots(c, psi, self.feature_cache[-1])
            for fig_name in fig_dict.keys():
                if self.cfg.enable_wandb_logging:
                    wandb.log({"transop_plt/" + fig_name: wandb.Image(fig_dict[fig_name])}, step=curr_iter)
                if self.cfg.enable_local_figure_saving:
                    self.save_figure(f"{fig_name}{curr_iter}", fig_dict[fig_name])
                plt.close(fig_dict[fig_name])

            metrics.update(to_metrics)

        # t-SNE Plotting of backbone features
        if (
            self.cfg.enable_tsne_plot
            and (self.cfg.enable_wandb_logging or self.cfg.enable_local_figure_saving)
            and curr_iter % self.cfg.tsne_plot_freq == 0
        ):
            tsne_fig = plot_tsne(np.array(self.feature_cache), np.array(self.label_cache))
            if self.cfg.enable_wandb_logging:
                wandb.log({"feat_plt/tsne": wandb.Image(tsne_fig)}, step=curr_iter)
            if self.cfg.enable_local_figure_saving:
                self.save_figure(f"tsne_iter{curr_iter}", tsne_fig)

        # Plot log spectra to determine if dimensional collapse occurs.
        if (
            self.cfg.enable_log_spectra_plot
            and (self.cfg.enable_wandb_logging or self.cfg.enable_local_figure_saving)
            and curr_iter % self.cfg.log_spectra_plot_freq == 0
        ):
            feat_norm = torch.nn.functional.normalize(torch.tensor(np.array(self.feature_cache)).float(), dim=1)
            feat_cov = np.cov(feat_norm.numpy().T)
            feat_log_spectra_plot, feat_log_spectra = plot_log_spectra(feat_cov)
            metrics["feat_log_spectra"] = feat_log_spectra
            if self.cfg.enable_wandb_logging:
                wandb.log(
                    {"feat_plt/feat_log_spectra_plot": wandb.Image(feat_log_spectra_plot)},
                    step=curr_iter,
                )
            if self.cfg.enable_local_figure_saving:
                self.save_figure(f"feat_log_spectra_iter{curr_iter}", feat_log_spectra_plot)

        if self.cfg.enable_wandb_logging:
            wandb.log(metrics, step=curr_iter)

    @staticmethod
    def initialize_metric_logger(
        metric_logger_cfg: MetricLoggerConfig,
        model: Model,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        return MetricLogger(metric_logger_cfg, model, scheduler)
