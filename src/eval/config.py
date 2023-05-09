"""
Class that contains all config DataClasses for evaluation.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""

from dataclasses import dataclass

from train.config import OptimizerConfig, SchedulerConfig


@dataclass
class EvalRunnerConfig:
    # Whether to run this metric or not
    enable_runner: bool = False
    # Whether to use this metric as the key metric for saving the best checkpoint
    use_for_best_checkpoint: bool = False
    # Evaluation frequency in epochs
    eval_freq: int = 50


@dataclass
class AugmentationNN(EvalRunnerConfig):
    num_images: int = 20
    num_augs: int = 8


@dataclass
class LinearProbeConfig(EvalRunnerConfig):
    optimizer_cfg: OptimizerConfig = OptimizerConfig(initial_lr=0.2, weight_decay=0.0, enable_nesterov=True)
    scheduler_cfg: SchedulerConfig = SchedulerConfig()
    num_epochs: int = 100


@dataclass
class SemiSupConfig(EvalRunnerConfig):
    num_iters: int = 10000
    batchsize_label: int = 32
    batchsize_unlabel: int = 224
    labels_per_class: int = 5
    num_trials: int = 5
    manifold_aug: bool = False


@dataclass
class ClusteringEvalConfig(EvalRunnerConfig):
    num_points_cluster: int = 5000
    num_clusters: int = 10


@dataclass
class KNNEvalConfig(EvalRunnerConfig):
    k: int = 5


@dataclass
class EvaluatorConfig:
    # Whether to locally save eval figures
    save_figure_local: bool = True
    # Clustering accuracy in the feature space of test features
    clustering_eval_cfg: ClusteringEvalConfig = ClusteringEvalConfig()
    # k-NN accuracy in the feature space of test features
    knn_eval_cfg: KNNEvalConfig = KNNEvalConfig()
    # Linear classification probe on top of frozen backbone features.
    linear_probe_eval_cfg: LinearProbeConfig = LinearProbeConfig()
    # Plot the nearest neighbor of augmentations sampled from the prior.
    aug_nn_eval_cfg: AugmentationNN = AugmentationNN()
    # Perform semi-supervised learning in the feature space by either using
    # consistency regularizations or feature augmentations from the operators
    semisup_probe_eval_cfg: SemiSupConfig = SemiSupConfig()
