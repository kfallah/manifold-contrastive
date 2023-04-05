"""
Module containing implementations for all optimizers and schedulers.

@Filename    optimizer.py
@Author      Kion
@Created     09/01/22
"""

import numpy as np
import torch
import torch.nn as nn

from model.public.lars import LARS
from model.public.linear_warmup_cos_anneal import LinearWarmupCosineAnnealingLR
from train.config import OptimizerConfig, SchedulerConfig


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    step_mult = min((step / total_steps), 1.0)
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step_mult * np.pi))


def initialize_optimizer(config: OptimizerConfig, model_params: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer == "SGD":
        return torch.optim.SGD(
            model_params,
            lr=config.initial_lr,
            weight_decay=config.weight_decay,
            momentum=0.9 if config.enable_nesterov else 0.0,
            nesterov=config.enable_nesterov,
        )
    elif config.optimizer == "Adam":
        return torch.optim.Adam(
            model_params,
            lr=config.initial_lr,
            weight_decay=config.weight_decay,
            # betas=(0.8, 0.999),
        )
    elif config.optimizer == "AdamW":
        return torch.optim.AdamW(
            model_params,
            lr=config.initial_lr,
            weight_decay=config.weight_decay,
            # betas=(0.8, 0.999),
        )
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
    config: SchedulerConfig,
    opt_config: OptimizerConfig,
    num_epochs: int,
    num_iters: int,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler._LRScheduler:
    if config.scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters, eta_min=0, last_epoch=-1)
    elif config.scheduler == "LinearWarmupCosineAnnealingLR":
        iters_per_epoch = num_iters / num_epochs
        return LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=config.warmup_epochs * iters_per_epoch, max_epochs=num_iters, eta_min=1e-6
        )
    elif config.scheduler == "CosineAnnealingMinLR":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: get_lr(
                step,
                num_iters,
                opt_config.initial_lr,
                1e-3,
            ),
        )
    elif config.scheduler == "MultiStepLR":
        iters_per_epoch = num_iters / num_epochs
        milestones = [int(num_iters - 50 * iters_per_epoch), int(num_iters - 25 * iters_per_epoch)]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    else:
        raise NotImplementedError
