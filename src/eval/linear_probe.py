"""
Train a linear classifier on top of the frozen weights of a backbone using all training labels. This serves as a proxy
to determine how well the features will perform in a transfer learning task.

@Filename    linear_probe.py
@Author      Kion
@Created     09/05/22
"""


from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.config import LinearProbeConfig
from eval.type import EvalRunner, EvaluationInput
from eval.utils import num_correct
from model.optimizer import initialize_optimizer, initialize_scheduler


class LinearProbeEval(EvalRunner):
    def initialize_training_modules(
        self,
        train_eval_input: EvaluationInput, 
        device: torch.device,
        num_classes: int = 10,
    ) -> None:
        backbone_feat_dim = train_eval_input.feature_list.shape[1]
        self.clf = nn.Linear(backbone_feat_dim, num_classes).to(device)
        num_epoch_iters = len(train_eval_input.feature_list) // self.get_config().num_epochs

        self.optimizer = initialize_optimizer(self.get_config().optimizer_cfg, self.clf.parameters())
        self.scheduler = initialize_scheduler(
            self.get_config().scheduler_cfg,
            self.get_config().optimizer_cfg,
            self.get_config().num_epochs,
            num_epoch_iters * self.get_config().num_epochs,
            self.optimizer,
        )
        self.criterion = nn.CrossEntropyLoss().to(device)

    def run_eval(
        self,
        train_eval_input: EvaluationInput,
        val_eval_input: EvaluationInput,
        device: torch.device,
        **kwargs
    ) -> Tuple[Dict[str, float], float]:
        self.initialize_training_modules(train_eval_input, device, kwargs["num_classes"])

        x_train, y_train = train_eval_input.feature_list, train_eval_input.labels
        x_val, y_val = val_eval_input.feature_list, val_eval_input.labels
        x_train, y_train = x_train.to(device), y_train.to(device)
        x_val, y_val = x_val.to(device), y_val.to(device)
    
        for e in range(self.get_config().num_epochs):
            perm = torch.randperm(len(x_train)).view(-1, 500)
            for idx in perm:
                self.optimizer.zero_grad()
                self.criterion(self.clf(x_train[idx]), y_train[idx]).backward()
                self.optimizer.step()
                self.scheduler.step()

        y_pred = self.clf(x_val)
        pred_top = y_pred.topk(max([1, 5]), 1, largest=True, sorted=True).indices
        acc = {
            t: (pred_top[:, :t] == y_val[..., None]).float().sum(1).mean().cpu().item()
            for t in [1, 5]
        }

        # Add full metrics, as well as final metrics, to the metrics log
        metrics = {
            "val_top1_acc": acc[1],
            "val_top5_acc": acc[5],
        }

        return metrics, acc[1]

    def get_config(self) -> LinearProbeConfig:
        return self.cfg
