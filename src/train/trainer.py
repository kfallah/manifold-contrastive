"""
Main trainer wrapper that contains all training state variables and performs a single training epoch.

@Filename    trainer.py
@Author      Kion
@Created     08/31/22
"""

from typing import Dict

import torch
import torch.nn as nn
import wandb
from model.model import Model
from torch.cuda.amp import GradScaler, autocast

from train.config import TrainerConfig


class Trainer(nn.Module):
    def __init__(self, trainer_cfg: TrainerConfig, model: Model, device: torch.device):
        super(Trainer, self).__init__()
        self.trainer_cfg = trainer_cfg
        self.model = model
        self.device = device

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 7324)
        self.scaler = GradScaler(enabled=self.trainer_cfg.use_amp)

    def train_epoch(self, epoch: int, train_dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        with autocast(enabled=self.trainer_cfg.use_amp):
            self.model.train()
            for idx, (x0, x1), _, _ in enumerate(train_dataloader):
                x0, x1 = x0.to(self.device), x1.to(self.device)

                # Send inputs through model
                feat0, pred0 = self.model(x0)
                feat1, pred1 = self.model(x1)
                # Compute loss
                total_loss, loss_metadata = self.model.compute_loss(x0, x1, feat0, feat1, pred0, pred1)

                # Backpropagate loss
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                if idx % self.trainer_cfg.logging_freq == 0:
                    wandb.log(loss_metadata)
        return loss_metadata

    @staticmethod
    def initialize_trainer(trainer_cfg: TrainerConfig, model: Model, device: torch.device) -> "Trainer":
        return Trainer(trainer_cfg, model, device)
