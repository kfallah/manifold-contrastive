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
    # Supported values: ["CIFAR10", "CIFAR100", "STL10", "TinyImagenet"]
    dataset_name: str = MISSING
    num_classes: int = MISSING
    dataset_dir: str = "../../datasets"
    image_size: int = 64


@dataclass
class SSLAugmentationConfig:
    cj_prob: float = 0.8
    cj_bright: float = 0.4
    cj_contrast: float = 0.4
    cj_sat: float = 0.4
    cj_hue: float = 0.1
    min_scale: float = 0.2
    gray_scale: float = 0.1
    horiz_flip: float = 0.5
    normalize_mean: Tuple[float] = (0.4914, 0.4822, 0.4465)
    normalize_std: Tuple[float] = (0.2023, 0.1994, 0.2010)


@dataclass
class DataLoaderConfig:
    dataset_cfg: DatasetConfig = MISSING
    ssl_aug_cfg: SSLAugmentationConfig = SSLAugmentationConfig()
    train: bool = True
    train_batch_size: int = 1024
    eval_batch_size: int = 1000
    val_batch_size: int = 1000
    persistent_workers: bool = True
    num_workers: int = 0
