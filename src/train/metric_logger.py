"""
Module responsible for logging all metrics related to training.

@Filename    metric_logger.py
@Author      Kion
@Created     09/01/22
"""

import logging
from typing import Dict

import torch
import wandb
from model.model import Model, ModelOutput

from train.config import MetricLoggerConfig

log = logging.getLogger(__name__)


class MetricLogger:
    def __init__(
        self, metric_logger_cfg: MetricLoggerConfig, model: Model, scheduler: torch.optim.lr_scheduler._LRScheduler
    ):
        self.cfg = metric_logger_cfg
        self.model = model
        self.scheduler = scheduler

    def log_metrics(
        self, curr_iter: int, curr_epoch: int, model_output: ModelOutput, loss_metadata: Dict[str, float]
    ) -> None:
        if self.cfg.enable_wandb_logging:
            assert wandb.run is not None, "wandb not initialized!"

        if self.cfg.enable_loss_logging and curr_iter % self.cfg.loss_log_freq == 0:
            if self.cfg.enable_wandb_logging:
                wandb.log(loss_metadata)
            if self.cfg.enable_console_logging:
                format_loss = [f"{key}: {loss_metadata[key]:.3E}" for key in loss_metadata.keys()]
                log.info(f"[Epoch {curr_epoch}, iter {curr_iter}]: " + ", ".join(format_loss))
        if self.cfg.enable_optimizer_logging and curr_iter % self.cfg.optimizer_log_freq == 0:
            optim_lr = self.scheduler.get_last_lr()
            if self.cfg.enable_wandb_logging:
                wandb.log({"optim_lr": optim_lr})
            if self.cfg.enable_console_logging:
                log.info(f"Optimizer LR: {optim_lr}")

    @staticmethod
    def initialize_metric_logger(
        metric_logger_cfg: MetricLoggerConfig, model: Model, scheduler: torch.optim.lr_scheduler._LRScheduler
    ):
        return MetricLogger(metric_logger_cfg, model, scheduler)
