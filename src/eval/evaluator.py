"""
Main evaluator wrapper that runs all validation metrics to measure the success of a training run.

@Filename    evaluator.py
@Author      Kion
@Created     08/31/22
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from dataloader.config import DataLoaderConfig
from dataloader.utils import get_weak_aug_dataloader
from eval.clustering import ClusteringEval
from eval.config import EvaluatorConfig
from eval.knn_acc import KNNEval
from eval.linear_probe import LinearProbeEval
from eval.type import EvalRunner
from eval.utils import encode_features
from model.model import Model


class Evaluator(nn.Module):
    def __init__(self, eval_cfg: EvaluatorConfig, model: Model, device: torch.device):
        super(Evaluator, self).__init__()
        self.eval_cfg = eval_cfg
        self.model = model
        self.device = device

        self.eval_runners: List[EvalRunner] = []
        if eval_cfg.clustering_eval_cfg.enable_runner:
            self.eval_runners.append(ClusteringEval(eval_cfg.clustering_eval_cfg))
        if eval_cfg.knn_eval_cfg.enable_runner:
            self.eval_runners.append(KNNEval(eval_cfg.knn_eval_cfg))
        if eval_cfg.linear_probe_eval_cfg.enable_runner:
            self.eval_runners.append(LinearProbeEval(eval_cfg.linear_probe_eval_cfg))

    def eval_needed(self, epoch: int) -> bool:
        eval_runner_list = [epoch % eval_runner.get_eval_freq() == 0 for eval_runner in self.eval_runners]
        return np.any(eval_runner_list)

    def run_eval(
        self,
        epoch: int,
        train_dataloader: torch.utils.data.DataLoader,
        train_dataloader_cfg: DataLoaderConfig,
        eval_dataloader: torch.utils.data.DataLoader,
        last_epoch: bool = False,
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        if not self.eval_needed(epoch) and not last_epoch:
            return None
        eval_metrics = {}

        weak_aug_train_dataloader = get_weak_aug_dataloader(train_dataloader, train_dataloader_cfg)
        train_eval_input = encode_features(self.model, weak_aug_train_dataloader, self.device)
        val_eval_input = encode_features(self.model, eval_dataloader, self.device)

        checkpoint_metric = 0.0
        for eval_runner in self.eval_runners:
            if epoch > 0 and (epoch % eval_runner.get_eval_freq() == 0 or last_epoch):
                metric_metadata, key_metric_value = eval_runner.run_eval(
                    train_eval_input=train_eval_input,
                    val_eval_input=val_eval_input,
                    train_dataloader=weak_aug_train_dataloader,
                    val_dataloader=eval_dataloader,
                    num_classes=train_dataloader_cfg.dataset_cfg.num_classes,
                    device=self.device,
                )
                eval_metrics.update(metric_metadata)
                if eval_runner.cfg.use_for_best_checkpoint:
                    checkpoint_metric = key_metric_value

        return checkpoint_metric, eval_metrics

    @staticmethod
    def initialize_evaluator(eval_cfg: EvaluatorConfig, model: Model, device: torch.device) -> "Evaluator":
        return Evaluator(eval_cfg, model, device)
