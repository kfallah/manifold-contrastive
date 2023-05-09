"""
Main evaluator wrapper that runs all validation metrics to measure the success of a training run.

@Filename    evaluator.py
@Author      Kion
@Created     08/31/22
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

import wandb
from dataloader.base import Dataset
from eval.clustering import ClusteringEval
from eval.config import EvaluatorConfig
from eval.knn_acc import KNNEval
from eval.linear_probe import LinearProbeEval
from eval.nn_aug import AugmentationNNEval
from eval.semi_sup_probe import SemiSupProbeEval
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
        if eval_cfg.aug_nn_eval_cfg.enable_runner:
            self.eval_runners.append(AugmentationNNEval(eval_cfg.aug_nn_eval_cfg))
        if eval_cfg.semisup_probe_eval_cfg.enable_runner:
            self.eval_runners.append(SemiSupProbeEval(eval_cfg.semisup_probe_eval_cfg))

    def eval_needed(self, epoch: int) -> bool:
        eval_runner_list = [epoch % eval_runner.get_eval_freq() == 0 for eval_runner in self.eval_runners]
        return np.any(eval_runner_list)

    def run_eval(
        self,
        epoch: int,
        dataset: Dataset,
        last_epoch: bool = False,
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        if not self.eval_needed(epoch) and not last_epoch:
            return None
        eval_metrics = {}

        eval_dataloader = dataset.eval_dataloader
        val_dataloader = dataset.val_dataloader
        train_eval_input = encode_features(self.model, eval_dataloader, self.device)
        val_eval_input = encode_features(self.model, val_dataloader, self.device)

        checkpoint_metric = 0.0
        for eval_runner in self.eval_runners:
            if epoch % eval_runner.get_eval_freq() == 0 or last_epoch:
                metric_metadata, key_metric_value, figures = eval_runner.run_eval(
                    train_eval_input=train_eval_input,
                    val_eval_input=val_eval_input,
                    num_classes=dataset.cfg.dataset_cfg.num_classes,
                    device=self.device,
                    dataset=dataset,
                )
                eval_metrics.update(metric_metadata)
                if eval_runner.cfg.use_for_best_checkpoint:
                    checkpoint_metric = key_metric_value

                for fig_name in figures.keys():
                    wandb.log({"eval/" + fig_name: wandb.Image(figures[fig_name])})
                    if self.eval_cfg.save_figure_local:
                        save_path = os.getcwd() + "/figures/"
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        figures[fig_name].savefig(save_path + fig_name + ".png", bbox_inches="tight")

                    plt.close(figures[fig_name])

        return checkpoint_metric, eval_metrics

    @staticmethod
    def initialize_evaluator(eval_cfg: EvaluatorConfig, model: Model, device: torch.device) -> "Evaluator":
        return Evaluator(eval_cfg, model, device)
