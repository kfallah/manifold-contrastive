"""
Main trainer wrapper that contains all training state variables and performs a single training epoch.

@Filename    trainer.py
@Author      Kion
@Created     08/31/22
"""

from typing import Dict

import torch
import torch.nn as nn
from experiment import ExperimentConfig
from model.model import Model

from train.config import TrainerConfig


class Trainer(nn.Module):
    def __init__(self, trainer_cfg: TrainerConfig, model: Model):
        super(Trainer, self).__init__()

        # Initialize optimizer and scheduler

    def train_epoch(self, epoch: int, train_dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        model.train()
        return {}

    @staticmethod
    def initialize_trainer(exp_cfg: ExperimentConfig, model: Model) -> "Trainer":
