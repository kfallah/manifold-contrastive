"""
Class that contains all config DataClasses for the trainer.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""

from dataclasses import MISSING, dataclass


@dataclass
class MetricLoggerConfig:
    enable_wandb_logging: bool = True
    enable_console_logging: bool = True
    enable_local_figure_saving: bool = False
    feature_cache_size: int = 5000

    enable_loss_logging: bool = True
    loss_log_freq: int = 100
    enable_optimizer_logging: bool = False
    optimizer_log_freq: int = 100
    enable_collapse_logging: bool = False
    collapse_log_freq: int = 100

    enable_tsne_plot: bool = False
    tsne_plot_freq: int = 2000
    enable_log_spectra_plot: bool = False
    log_spectra_plot_freq: int = 2000


@dataclass
class OptimizerConfig:
    optimizer: str = MISSING
    initial_lr: float = MISSING
    weight_decay: float = MISSING


@dataclass
class SchedulerConfig:
    scheduler: str = MISSING


@dataclass
class TrainerConfig:
    optimizer_cfg: OptimizerConfig = MISSING
    scheduler_cfg: SchedulerConfig = MISSING
    metric_logger_cfg: MetricLoggerConfig = MetricLoggerConfig()
    num_epochs: int = 300
    save_interval: int = 50
    use_amp: bool = False
