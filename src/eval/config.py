"""
Class that contains all config DataClasses for evaluation.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""

from dataclasses import dataclass


@dataclass
class EvaluatorConfig:
    eval_frequency: int = MISSING
