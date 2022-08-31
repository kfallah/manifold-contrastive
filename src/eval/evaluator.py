"""
Main evaluator wrapper that runs all validation metrics to measure the success of a training run.

@Filename    evaluator.py
@Author      Kion
@Created     08/31/22
"""
from typing import Dict

import torch.nn as nn
from experiment import ExperimentConfig
from model.model import Model

from eval.config import EvaluatorConfig


class Evaluator(nn.Module):
    def __init__(self, eval_cfg: EvaluatorConfig, model: Model):
        super(Evaluator, self).__init__()

        # Initialize optimizer and scheduler

    @staticmethod
    def initialize_evaluator(exp_cfg: ExperimentConfig, model: Model) -> "Evaluator":
        return Evaluator(exp_cfg.eval_cfg, model)

    def run_eval(epoch: int, eval_dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        return {}
