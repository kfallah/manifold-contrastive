"""
Module containing implementations for all optimizers and schedulers.

@Filename    optimizer.py
@Author      Kion
@Created     09/01/22
"""

import torch
import torch.nn as nn
from train.config import OptimizerConfig, SchedulerConfig

from model.public.lars import LARS


def initialize_optimizer(config: OptimizerConfig, model_params: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer == "Adam":
        return torch.optim.Adam(model_params, lr=config.initial_lr, weight_decay=config.weight_decay)
    elif config.optimizer == "LARS":
        return LARS(
            model_params,
            lr=config.initial_lr,
            weight_decay=config.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )
    else:
        raise NotImplementedError


def initialize_scheduler(
    config: SchedulerConfig, num_epochs: int, num_iters: int, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    if config.scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters, eta_min=0, last_epoch=-1)
    else:
        raise NotImplementedError
