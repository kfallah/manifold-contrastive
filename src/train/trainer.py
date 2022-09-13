"""
Main trainer wrapper that contains all training state variables and performs a single training epoch.

@Filename    trainer.py
@Author      Kion
@Created     08/31/22
"""

import os
import time
from typing import Dict

import torch
import torch.nn as nn
from model.model import Model
from model.optimizer import initialize_optimizer, initialize_scheduler
from torch.cuda.amp import GradScaler, autocast

from train.config import TrainerConfig
from train.metric_logger import MetricLogger


class Trainer(nn.Module):
    def __init__(self, trainer_cfg: TrainerConfig, model: Model, num_train_iters: int, device: torch.device):
        super(Trainer, self).__init__()
        self.trainer_cfg = trainer_cfg
        self.model = model
        self.device = device

        # Initialize optimizer and scheduler
        self.optimizer = initialize_optimizer(self.trainer_cfg.optimizer_cfg, self.get_model().get_param_groups())
        self.scheduler = initialize_scheduler(
            self.trainer_cfg.scheduler_cfg, self.trainer_cfg.num_epochs, num_train_iters, self.optimizer
        )
        self.scaler = GradScaler(enabled=self.trainer_cfg.use_amp)

        # Initialize metric logger
        self.metric_logger = MetricLogger.initialize_metric_logger(
            self.trainer_cfg.metric_logger_cfg, self.get_model(), self.scheduler
        )

    def get_model(self) -> Model:
        if isinstance(self.model, nn.parallel.DataParallel):
            return self.model.module
        else:
            return self.model

    def train_epoch(self, epoch: int, train_dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.train()
        for idx, batch in enumerate(train_dataloader):
            pre_time = time.time()
            x_list = list(batch[0])
            # Tensor of input images of shape [B x V x H x W x C]
            x_gpu = torch.stack([x.to(self.device) for x in x_list]).transpose(0, 1)
            x_idx = torch.Tensor([int(idx) for idx in batch[2]])

            # Update momentum networks if they are enabled
            self.get_model().update_momentum_network()

            with autocast(enabled=self.trainer_cfg.use_amp):
                # Send inputs through model
                model_output = self.model(x_gpu, x_idx)
                loss_metadata, total_loss = self.get_model().compute_loss(model_output)

            # Backpropagate loss
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            loss_metadata["iter_time"] = time.time() - pre_time
            self.metric_logger.log_metrics(
                idx + (epoch * len(train_dataloader)), epoch, model_output, batch[1], loss_metadata
            )

        return loss_metadata

    def save_model(self, curr_epoch: int, save_path: str, save_best=False) -> None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_best:
            save_path = save_path + "best_eval_checkpoint.pt"
        else:
            save_path = save_path + f"checkpoint_epoch{curr_epoch}.pt"
        torch.save(
            {
                "model_state": self.get_model().state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "current_epoch": curr_epoch,
            },
            save_path,
        )

    @staticmethod
    def initialize_trainer(
        trainer_cfg: TrainerConfig, model: Model, num_train_iters: int, device: torch.device
    ) -> "Trainer":
        return Trainer(trainer_cfg, model, num_train_iters, device)

    @staticmethod
    def load_trainer(
        load_path: str, trainer_cfg: TrainerConfig, model: Model, num_train_iters: int, device: torch.device
    ) -> "Trainer":
        model_dict = torch.load(load_path)

        trainer = Trainer.initialize_trainer(trainer_cfg, model, num_train_iters, device)
        trainer.optimizer.load_state_dict(model_dict["optimizer"])
        trainer.scheduler.load_state_dict(model_dict["scheduler"])
        trainer.get_model().load_state_dict(model_dict["model_state"])
        return trainer
