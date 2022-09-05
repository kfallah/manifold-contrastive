"""
Train a linear classifier on top of the frozen weights of a backbone using all training labels. This serves as a proxy
to determine how well the features will perform in a transfer learning task.

@Filename    linear_probe.py
@Author      Kion
@Created     09/05/22
"""


from typing import Dict, Tuple

from eval.config import LinearProbeConfig
from eval.type import EvalRunner, EvaluationInput


class LinearProbeEval(EvalRunner):
    def run_eval(
        self, train_eval_input: EvaluationInput, val_eval_input: EvaluationInput
    ) -> Tuple[Dict[str, float], float]:

        return {}, 0

    def get_config(self) -> LinearProbeConfig:
        return LinearProbeConfig(self.cfg)
