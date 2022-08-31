"""
Class that contains all config DataClasses for the trainer.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""

from dataclasses import MISSING, dataclass


@dataclass
class OptimizerConfig:
    optimizer: str = MISSING
    initial_lr: float = MISSING


@dataclass
class SchedulerConfig:
    scheduler: str = MISSING
    decay: float = MISSING


@dataclass
class TrainerConfig:
    optimizer_cfg: OptimizerConfig = MISSING
    scheduler_cfg: SchedulerConfig = MISSING
    num_epochs: int = 300
    use_amp: bool = False
    logging_freq: int = 100
