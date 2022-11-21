"""
Train a linear classifier on top of the frozen weights of a backbone using all training labels. This serves as a proxy
to determine how well the features will perform in a transfer learning task.

@Filename    linear_probe.py
@Author      Kion
@Created     09/05/22
"""


from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from eval.config import LinearProbeConfig
from eval.type import EvalRunner, EvaluationInput
from eval.utils import num_correct
from model.optimizer import initialize_optimizer, initialize_scheduler


class LinearProbeEval(EvalRunner):
    def initialize_training_modules(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_classes: int = 10,
    ) -> None:
        self.model = deepcopy(model)
        self.linear_head = nn.Linear(512, num_classes).to(device)
        num_epoch_iters = len(train_loader)

        self.optimizer = initialize_optimizer(self.get_config().optimizer_cfg, self.linear_head.parameters())
        self.scheduler = initialize_scheduler(
            self.get_config().scheduler_cfg,
            self.get_config().optimizer_cfg,
            self.get_config().num_epochs,
            num_epoch_iters * self.get_config().num_epochs,
            self.optimizer,
        )
        self.scaler = GradScaler(enabled=self.get_config().use_amp)

    def train_epoch(self, train_dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
        epoch_loss = []
        self.model.train()
        for idx, batch in enumerate(train_dataloader):
            x, y, batch_idx = batch
            x, y = x.unsqueeze(1).to(device), y.to(device)

            with autocast(enabled=self.get_config().use_amp):
                # Send inputs through model
                model_out = self.model(x, batch_idx, idx)
                feat = model_out.header_input.feature_0
                y_logit = self.linear_head(feat).squeeze(1)
                loss = F.cross_entropy(y_logit, y)

            epoch_loss.append(loss.item())

            # Backpropagate loss
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
        return np.mean(epoch_loss)

    def val(self, val_dataloader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, float, float]:
        self.model.eval()
        num_top1_correct = 0
        num_top5_correct = 0
        total = 0
        val_loss = []
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                x, y, batch_idx = batch
                x, y = x.unsqueeze(1).to(device), y.to(device)

                model_out = self.model(x, batch_idx, idx)
                feat = model_out.header_input.feature_0
                y_logit = self.linear_head(feat).squeeze(1)
                loss = F.cross_entropy(y_logit, y)
                val_loss.append(loss.item())

                batch_top1, batch_top5 = num_correct(y_logit, y, topk=(1, 5))
                num_top1_correct += batch_top1.item()
                num_top5_correct += batch_top5.item()
                total += len(x)

        num_top1_acc = num_top1_correct / total
        num_top5_acc = num_top5_correct / total
        return num_top1_acc, num_top5_acc, np.mean(val_loss)

    def run_eval(
        self,
        train_eval_input: EvaluationInput,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        **kwargs
    ) -> Tuple[Dict[str, float], float]:
        self.initialize_training_modules(train_eval_input.model, train_dataloader, device, kwargs["num_classes"])

        for epoch in range(self.get_config().num_epochs):
            _ = self.train_epoch(train_dataloader, device)

        # Save the validation performance at the end of training
        val_top1_acc, val_top5_acc, val_avg_loss = self.val(val_dataloader, device)

        # Add full metrics, as well as final metrics, to the metrics log
        metrics = {
            "val_top1_acc": val_top1_acc,
            "val_top5_acc": val_top5_acc,
            "val_avg_loss_list": val_avg_loss,
        }

        return metrics, val_top1_acc

    def get_config(self) -> LinearProbeConfig:
        return self.cfg
