"""
Perform k-Nearest Neighbors to determine whether points in the latent space match examples from the same class.

@Filename    knn_acc.py
@Author      Kion
@Created     09/01/22
"""


from typing import Dict, Tuple

from eval.config import ClusteringEvalConfig
from eval.type import EvalRunner, EvaluationInput


class KNNEval(EvalRunner):
    def run_eval(self, eval_input: EvaluationInput) -> Tuple[Dict[str, float], float]:
        knn_metrics = {}
        feature_knn = 0
        return knn_metrics, feature_knn

    def get_config(self) -> ClusteringEvalConfig:
        return ClusteringEvalConfig(self.cfg)
