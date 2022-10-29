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
        self.pred_cache = []
        self.label_cache = []
        self.c_cache = []

    def enable_feature_cache(self):
        return (
            self.cfg.enable_log_spectra_plot
            or self.cfg.enable_tsne_plot
            or self.cfg.enable_transop_logging
        )

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
            self.feature_cache.extend(model_output.feature_0.detach().cpu().numpy())
            self.feature_cache = self.feature_cache[-self.cfg.feature_cache_size :]
            self.pred_cache.extend(model_output.prediction_0.detach().cpu().numpy())
            self.pred_cache = self.pred_cache[-self.cfg.feature_cache_size :]
            self.label_cache.extend(labels.numpy())
            self.label_cache = self.label_cache[-self.cfg.feature_cache_size :]

            if self.cfg.enable_transop_logging:
                self.c_cache.extend(
                    model_output.distribution_data.samples.detach().cpu().numpy()
                )
                self.c_cache = self.c_cache[-self.cfg.feature_cache_size :]
        metrics = {}

        # Logging training heuristics
        if self.cfg.enable_loss_logging:
            if len(self.metadata_avg) == 0:
                # In the first iteration use the loss metadata directly
                self.metadata_avg = loss_metadata
            else:
                # Get moving average for the iterations between the loss metadata
                for key in loss_metadata.keys():
                    self.metadata_avg[key] += (
                        loss_metadata[key] / self.cfg.loss_log_freq
                    )

            if curr_iter % self.cfg.loss_log_freq == 0:
                metrics.update(self.metadata_avg)
                if self.cfg.enable_console_logging:
                    format_loss = [
                        f"{key}: {self.metadata_avg[key]:.3E}"
                        for key in self.metadata_avg.keys()
                    ]
                    log.info(
                        f"[Epoch {curr_epoch}, iter {curr_iter}]: "
                        + ", ".join(format_loss)
                    )
                    # Reset all values to zero for the next loop
                    self.metadata_avg = dict.fromkeys(self.metadata_avg, 0.0)

        # Logging optimizer LR
        if (
            self.cfg.enable_optimizer_logging
            and curr_iter % self.cfg.optimizer_log_freq == 0
        ):
            optim_lr = self.scheduler.get_last_lr()
            metrics["optim_lr"] = optim_lr
            if self.cfg.enable_console_logging:
                log.info(
                    f"[Epoch {curr_epoch}, iter {curr_iter}]: Optimizer LR: {optim_lr}"
                )

        # Logging collapse level
        if (
            self.cfg.enable_collapse_logging
            and curr_iter % self.cfg.collapse_log_freq == 0
        ):
            feat_norm = torch.nn.functional.normalize(
                torch.tensor(np.array(self.feature_cache)).float(), dim=1
            )
            feat_var = torch.std(feat_norm, dim=0)
            feat_collapse = max(
                0.0, 1 - math.sqrt(len(feat_var)) * feat_var.mean().item()
            )
            pred_norm = torch.nn.functional.normalize(
                torch.tensor(np.array(self.pred_cache)).float(), dim=1
            )
            pred_var = torch.std(pred_norm, dim=0)
            pred_collapse = max(0.0, 1 - math.sqrt(len(pred_var)) * pred_var.mean())
            metrics["feat_collapse"] = feat_collapse
            metrics["pred_collapse"] = pred_collapse
            if self.cfg.enable_console_logging:
                log.info(
                    f"[Epoch {curr_epoch}, iter {curr_iter}]: Feature collapse: {feat_collapse:.2f}"
                    + f", Prediction collapse: {pred_collapse:.2f}"
                )

        if (
            self.cfg.enable_transop_logging
            and curr_iter % self.cfg.transop_log_freq == 0
        ):
            assert (
                self.model.model_cfg.header_cfg.header_name == "TransOp"
                and model_output.distribution_data is not None
            )
            psi = self.model.contrastive_header.transop_header.transop.get_psi()
            c = np.array(self.c_cache)
            coeff_nz = np.count_nonzero(c, axis=0)
            nz_tot = np.count_nonzero(coeff_nz)
            total_nz = np.count_nonzero(c, axis=1)
            avg_feat_norm = np.linalg.norm(np.array(self.feature_cache), axis=-1).mean()
            transop_loss = F.mse_loss(
                model_output.prediction_0, model_output.prediction_1
            ).item()
            dist_bw_point_pairs = F.mse_loss(
                model_output.prediction_0, model_output.prediction_1
            ).item()
            failed_iters = (
                self.model.contrastive_header.transop_header.failed_iters
                / self.cfg.transop_log_freq
            )
            self.model.contrastive_header.transop_header.failed_iters = 0

            psi_mag = torch.norm(psi.data.reshape(len(psi.data), -1), dim=-1)
            to_metrics = {
                "avg_transop_mag": psi_mag.mean(),
                "total_transop_used": nz_tot,
                "avg_transop_used": total_nz.mean(),
                "avg_coeff_mag": np.abs(c[np.abs(c) > 0]).mean(),
                "avg_feat_norm": avg_feat_norm,
                "transop_loss": transop_loss,
                "dist_bw_point_pairs": dist_bw_point_pairs,
            }
            if self.cfg.enable_console_logging:
                log.info(
                    f"[Transport Operator iter {curr_iter}]: transop loss: {transop_loss:.3E}"
                    + f", dist bw point pairs: {dist_bw_point_pairs:.3E}"
                    + f", average transop mag: {psi_mag.mean():.3E}"
                    + f", total # operators used: {nz_tot}/{len(psi)}"
                    + f", avg # operators used: {total_nz.mean()}/{len(psi)}"
                    + f", avg feat norm: {avg_feat_norm:.2E}"
                    + f", avg coeff mag: {to_metrics['avg_coeff_mag']:.2f}"
                    + f", % failed iters: {100. * failed_iters:.2f}"
                )
                if self.model.model_cfg.header_cfg.enable_variational_inference:
                    scale = torch.exp(
                        model_output.distribution_data.encoder_params["logscale"]
                    )
                    shift = model_output.distribution_data.encoder_params["shift"]
                    log.info(
                        f"[Encoder params]: "
                        + f"min scale: {scale.abs().min():.3E}"
                        + f", max scale: {scale.abs().max():.3E}"
                        + f", mean scale: {scale.mean():.3E}"
                        + f", min shift: {shift.abs().min():.3E}"
                        + f", max shift: {shift.abs().max():.3E}"
                        + f", mean shift: {shift.mean():.3E}"
                    )
                    if "Gamma" in self.model.model_cfg.header_cfg.vi_cfg.distribution:
                        enc_gamma_a = model_output.distribution_data.encoder_params[
                            "gamma_a"
                        ]
                        enc_gamma_b = model_output.distribution_data.encoder_params[
                            "gamma_b"
                        ]
                        log.info(
                            "[Encoder Gamma]: "
                            + f"min gamma_a: {enc_gamma_a.abs().min():.3E}"
                            + f", max gamma_a: {enc_gamma_a.abs().max():.3E}"
                            + f", mean gamma_a: {enc_gamma_a.mean():.3E}"
                            + f", min gamma_b: {enc_gamma_b.abs().min():.3E}"
                            + f", max gamma_b: {enc_gamma_b.abs().max():.3E}"
                            + f", mean gamma_b: {enc_gamma_b.mean():.3E}"
                        )
                        if (
                            self.model.model_cfg.header_cfg.vi_cfg.prior_type
                            == "Learned"
                        ):
                            prior_scale = torch.exp(
                                model_output.distribution_data.prior_params["logscale"]
                            )
                            prior_shift = model_output.distribution_data.prior_params[
                                "shift"
                            ]
                            prior_gamma_a = model_output.distribution_data.prior_params[
                                "gamma_a"
                            ]
                            prior_gamma_b = model_output.distribution_data.prior_params[
                                "gamma_b"
                            ]
                            log.info(
                                f"[Prior params]: "
                                + f"min scale: {prior_scale.abs().min():.3E}"
                                + f", max scale: {prior_scale.abs().max():.3E}"
                                + f", mean scale: {prior_scale.mean():.3E}"
                                + f", min shift: {prior_shift.abs().min():.3E}"
                                + f", max shift: {prior_shift.abs().max():.3E}"
                                + f", mean shift: {prior_shift.mean():.3E}"
                            )
                            log.info(
                                "[Prior Gamma]: "
                                + f"min gamma_a: {prior_gamma_a.abs().min():.3E}"
                                + f", max gamma_a: {prior_gamma_a.abs().max():.3E}"
                                + f", mean gamma_a: {prior_gamma_a.mean():.3E}"
                                + f", min gamma_b: {prior_gamma_b.abs().min():.3E}"
                                + f", max gamma_b: {prior_gamma_b.abs().max():.3E}"
                                + f", mean gamma_b: {prior_gamma_b.mean():.3E}"
                            )

            # Generate transport operator plots
            fig_dict = transop_plots(c, psi, model_output.projection_0)
            for fig_name in fig_dict.keys():
                if self.cfg.enable_wandb_logging:
                    wandb.log(
                        {fig_name: wandb.Image(fig_dict[fig_name])}, step=curr_iter
                    )
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
            tsne_fig = plot_tsne(
                np.array(self.feature_cache), np.array(self.label_cache)
            )
            if self.cfg.enable_wandb_logging:
                wandb.log({"tsne": wandb.Image(tsne_fig)}, step=curr_iter)
            if self.cfg.enable_local_figure_saving:
                self.save_figure(f"tsne_iter{curr_iter}", tsne_fig)

        # Plot log spectra to determine if dimensional collapse occurs.
        if (
            self.cfg.enable_log_spectra_plot
            and (self.cfg.enable_wandb_logging or self.cfg.enable_local_figure_saving)
            and curr_iter % self.cfg.log_spectra_plot_freq == 0
        ):
            feat_norm = torch.nn.functional.normalize(
                torch.tensor(np.array(self.feature_cache)).float(), dim=1
            )
            feat_cov = np.cov(feat_norm.numpy().T)
            feat_log_spectra_plot, feat_log_spectra = plot_log_spectra(feat_cov)
            pred_norm = torch.nn.functional.normalize(
                torch.tensor(np.array(self.pred_cache)).float(), dim=1
            )
            pred_cov = np.cov(pred_norm.numpy().T)
            pred_log_spectra_plot, pred_log_spectra = plot_log_spectra(pred_cov)
            metrics["feat_log_spectra"] = feat_log_spectra
            metrics["pred_log_spectra"] = pred_log_spectra
            if self.cfg.enable_wandb_logging:
                wandb.log(
                    {"feat_log_spectra_plot": wandb.Image(feat_log_spectra_plot)},
                    step=curr_iter,
                )
                wandb.log(
                    {"pred_log_spectra_plot": wandb.Image(pred_log_spectra_plot)},
                    step=curr_iter,
                )
            if self.cfg.enable_local_figure_saving:
                self.save_figure(
                    f"feat_log_spectra_iter{curr_iter}", feat_log_spectra_plot
                )
                self.save_figure(
                    f"pred_log_spectra_iter{curr_iter}", pred_log_spectra_plot
                )

        if self.cfg.enable_wandb_logging:
            wandb.log(metrics, step=curr_iter)

    @staticmethod
    def initialize_metric_logger(
        metric_logger_cfg: MetricLoggerConfig,
        model: Model,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        return MetricLogger(metric_logger_cfg, model, scheduler)
