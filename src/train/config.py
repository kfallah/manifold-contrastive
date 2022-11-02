"""
Class that contains all config DataClasses for the trainer.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""

from dataclasses import dataclass


@dataclass
class MetricLoggerConfig:
    enable_wandb_logging: bool = True
    enable_console_logging: bool = True
    enable_local_figure_saving: bool = False
    feature_cache_size: int = 5000

    enable_loss_logging: bool = True
    loss_log_freq: int = 100
    enable_optimizer_logging: bool = False
    optimizer_log_freq: int = 500
    enable_collapse_logging: bool = False
    collapse_log_freq: int = 500
    enable_transop_logging: bool = False
    transop_log_freq: int = 1000

    enable_tsne_plot: bool = False
    tsne_plot_freq: int = 5000
    enable_log_spectra_plot: bool = False
    log_spectra_plot_freq: int = 5000


@dataclass
class OptimizerConfig:
    optimizer: str = "SGD"
    initial_lr: float = 0.1
    weight_decay: float = 1e-6
    enable_nesterov: bool = False


@dataclass
class SchedulerConfig:
    scheduler: str = "CosineAnnealingLR"
    warmup_epochs: int = 10


@dataclass
class TrainerConfig:
    optimizer_cfg: OptimizerConfig = OptimizerConfig()
    scheduler_cfg: SchedulerConfig = SchedulerConfig()
    metric_logger_cfg: MetricLoggerConfig = MetricLoggerConfig()
    num_epochs: int = 300
    grad_accumulation_iters: int = 1
    save_interval: int = 50
    use_amp: bool = False
