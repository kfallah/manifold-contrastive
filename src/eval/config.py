"""
Class that contains all config DataClasses for evaluation.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""

from dataclasses import MISSING, dataclass


@dataclass
class ClusteringEvalConfig:
    num_points_cluster: int = 5000


@dataclass
class EvaluatorConfig:
    eval_frequency: int = MISSING
    # k-NN accuracy in the feature space of test features
    enable_clustering_eval: bool = False
    clustering_eval_freq: int = 50
    clustering_eval_cfg: ClusteringEvalConfig = ClusteringEvalConfig()
