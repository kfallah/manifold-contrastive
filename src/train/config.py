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

    enable_loss_logging: bool = True
    loss_log_freq: int = 100
    enable_optimizer_logging: bool = False
    optimizer_log_freq: int = 100


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
    use_amp: bool = False
