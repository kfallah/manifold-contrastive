"""
Class that contains all config DataClasses for the dataloaders.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""

from dataclasses import MISSING, dataclass
from typing import Tuple


@dataclass
class DatasetConfig:
    # Supported values: ["CIFAR10", "CIFAR100"]
    dataset_name: str = MISSING
    num_classes: int = MISSING
    dataset_dir: str = "../../datasets"
    image_size: int = 224


@dataclass
class CollateFunctionConfig:
    # Supported values: ["LIGHTLY_IMAGE", "NONE"]
    collate_fn_type: str = MISSING
    cj_prob: float = 0.4
    cj_bright: float = 0.4
    cj_contrast: float = 0.4
    cj_sat: float = 0.4
    cj_hue: float = 0.1
    min_scale: float = 0.2
    gausian_blur: float = 0.0
    gray_scale: float = 0.1
    horiz_flip: float = 0.5
    normalize_mean: Tuple[float] = (0.4914, 0.4822, 0.4465)
    normalize_std: Tuple[float] = (0.2023, 0.1994, 0.2010)


@dataclass
class DataLoaderConfig:
    dataset_cfg: DatasetConfig = MISSING
    collate_fn_cfg: CollateFunctionConfig = MISSING
    train: bool = True
    batch_size: int = 64
    shuffle: bool = False
    persistent_workers: bool = True
    num_workers: int = 0
