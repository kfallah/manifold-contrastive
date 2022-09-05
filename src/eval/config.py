"""
Class that contains all config DataClasses for evaluation.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""

from dataclasses import dataclass


@dataclass
class EvalRunnerConfig:
    # Whether to run this metric or not
    enable_runner: bool = False
    # Whether to use this metric as the key metric for saving the best checkpoint
    use_for_best_checkpoint: bool = False
    # Evaluation frequency in epochs
    eval_freq: int = 50


@dataclass
class LinearProbeConfig(EvalRunnerConfig):
    num_epochs: int = 100


@dataclass
class ClusteringEvalConfig(EvalRunnerConfig):
    num_points_cluster: int = 5000
    num_clusters: int = 10


@dataclass
class KNNEvalConfig(EvalRunnerConfig):
    k: int = 20


@dataclass
class EvaluatorConfig:
    # Clustering accuracy in the feature space of test features
    clustering_eval_cfg: ClusteringEvalConfig = ClusteringEvalConfig()
    # k-NN accuracy in the feature space of test features
    knn_eval_cfg: KNNEvalConfig = KNNEvalConfig()
