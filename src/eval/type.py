"""
Contain all data types used by the evaluator modules.

@Filename    type.py
@Author      Kion
@Created     09/01/22
"""

from typing import Dict, NamedTuple, Tuple

import torch

from eval.config import EvalRunnerConfig
from model.model import Model


class EvaluationInput(NamedTuple):
    # Model used for evaluation
    model: Model
    # All the input images used for evaluation [N x H x W x 3] (expected on CPU)
    x: torch.Tensor
    # Indices for input data from dataloader [N] (expected on CPU)
    x_idx: list
    # Labels for all of the validation samples [N] (expected on CPU)
    labels: torch.Tensor
    # Encoded features from backbone for the val dataloader [N x D] (expected on CPU)
    feature_list: torch.Tensor


class EvalRunner:
    def __init__(self, cfg: EvalRunnerConfig):
        super(EvalRunner, self).__init__()
        self.cfg = cfg

    def run_eval(self, **kwargs) -> Tuple[Dict[str, float], float]:
        raise NotImplementedError

    def get_eval_freq(self) -> int:
        return self.cfg.eval_freq

    def get_config(self) -> EvalRunnerConfig:
        raise NotImplementedError
