"""
Main trainer wrapper that contains all training state variables and performs a single training epoch.

@Filename    trainer.py
@Author      Kion
@Created     08/31/22
"""

from typing import Dict

import torch
import torch.nn as nn
from model.model import Model
from model.optimizer import initialize_optimizer, initialize_scheduler
from torch.cuda.amp import GradScaler, autocast

from train.config import TrainerConfig
from train.metric_logger import MetricLogger


class Trainer(nn.Module):
    def __init__(self, trainer_cfg: TrainerConfig, model: Model, device: torch.device):
        super(Trainer, self).__init__()
        self.trainer_cfg = trainer_cfg
        self.model = model
        self.device = device

        # Initialize optimizer and scheduler
        self.optimizer = initialize_optimizer(self.trainer_cfg.optimizer_cfg, self.model.parameters())
        self.scheduler = initialize_scheduler(
            self.trainer_cfg.scheduler_cfg, self.trainer_cfg.num_epochs, self.optimizer
        )
        self.scaler = GradScaler(enabled=self.trainer_cfg.use_amp)

        # Initialize metric logger
        self.metric_logger = MetricLogger.initialize_metric_logger(
            self.trainer_cfg.metric_logger_cfg, self.model, self.scheduler
        )

    def train_epoch(self, epoch: int, train_dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.train()
        for idx, batch in enumerate(train_dataloader):
            x_list = list(batch[0])
            x_gpu = [x.to(self.device) for x in x_list]
            x_idx = batch[2]

            with autocast(enabled=self.trainer_cfg.use_amp):
                # Send inputs through model
                model_output, loss_metadata, total_loss = self.model(x_gpu, x_idx)

            # Backpropagate loss
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.metric_logger.log_metrics(idx + (epoch * len(train_dataloader)), epoch, model_output, loss_metadata)

        return loss_metadata

    def save_model(self, curr_epoch: int, save_path: str) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "current_epoch": curr_epoch,
            },
            save_path + f"checkpoint_epoch{curr_epoch}.pt",
        )

    @staticmethod
    def initialize_trainer(trainer_cfg: TrainerConfig, model: Model, device: torch.device) -> "Trainer":
        return Trainer(trainer_cfg, model, device)

    @staticmethod
    def load_trainer(load_path: str, trainer_cfg: TrainerConfig, model: Model, device: torch.device) -> "Trainer":
        model_dict = torch.load(load_path)

        trainer = Trainer.initialize_trainer(trainer_cfg, model, device)
        trainer.optimizer.load_state_dict(model_dict["optimizer"])
        trainer.scheduler.load_state_dict(model_dict["scheduler"])
        model.load_state_dict(model_dict["model_state"])
        return trainer
