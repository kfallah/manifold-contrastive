"""
Main evaluator wrapper that runs all validation metrics to measure the success of a training run.

@Filename    evaluator.py
@Author      Kion
@Created     08/31/22
"""
from typing import Dict

import torch
import torch.nn as nn
from model.model import Model

from eval.config import EvaluatorConfig


class Evaluator(nn.Module):
    def __init__(self, eval_cfg: EvaluatorConfig, model: Model):
        super(Evaluator, self).__init__()

        # Initialize optimizer and scheduler

    @staticmethod
    def initialize_evaluator(eval_cfg: EvaluatorConfig, model: Model, device: torch.device) -> "Evaluator":
        return Evaluator(eval_cfg, model)

    def run_eval(self, epoch: int, eval_dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        return 0, {}
