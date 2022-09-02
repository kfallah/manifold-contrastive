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
from model.model import Model

from eval.clustering import ClusteringEval
from eval.config import EvaluatorConfig
from eval.knn_acc import KNNEval
from eval.type import EvalRunner, EvaluationInput


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

    def eval_needed(self, epoch: int) -> bool:
        eval_runner_list = [epoch % eval_runner.get_eval_freq() == 0 for eval_runner in self.eval_runners]
        return np.any(eval_runner_list)

    def run_eval(
        self, epoch: int, eval_dataloader: torch.utils.data.DataLoader
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        if not self.eval_needed(epoch):
            return None
        eval_metrics = {}

        self.model.eval()
        x_eval = []
        labels = []
        x_idx = []
        feature_list = []
        prediction_list = []
        for _, batch in enumerate(eval_dataloader):
            x, batch_label, batch_idx = batch
            x_gpu = x.to(self.device).unsqueeze(1)
            batch_idx = torch.Tensor([int(idx) for idx in batch_idx])
            model_output = self.model(x_gpu, batch_idx)

            x_eval.append(x.detach().cpu())
            labels.append(batch_label.detach().cpu())
            x_idx.append(batch_idx.detach().cpu())
            feature_list.append(model_output.feature_list.detach().cpu())
            prediction_list.append(model_output.prediction_list.detach().cpu())
        # Flatten all encoded data to a single tensor
        x_eval = torch.cat(x_eval)
        labels = torch.cat(labels)
        x_idx = torch.cat(x_idx)
        feature_list = torch.cat(feature_list).squeeze(1)
        prediction_list = torch.cat(prediction_list).squeeze(1)
        # Create evaluation input from all encoded data
        eval_input = EvaluationInput(self.model, x_eval, labels, x_idx, feature_list, prediction_list)

        checkpoint_metric = 0.0
        for eval_runner in self.eval_runners:
            if epoch % eval_runner.get_eval_freq() == 0:
                metric_metadata, key_metric_value = eval_runner.run_eval(eval_input)
                eval_metrics.update(metric_metadata)
                if eval_runner.cfg.use_for_best_checkpoint:
                    checkpoint_metric = key_metric_value

        return checkpoint_metric, eval_metrics

    @staticmethod
    def initialize_evaluator(eval_cfg: EvaluatorConfig, model: Model, device: torch.device) -> "Evaluator":
        return Evaluator(eval_cfg, model, device)
