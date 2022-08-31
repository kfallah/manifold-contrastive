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
    num_epochs: int = MISSING
    use_amp: bool = MISSING
    optimizer_cfg: OptimizerConfig = MISSING
    scheduler_cfg: SchedulerConfig = MISSING
