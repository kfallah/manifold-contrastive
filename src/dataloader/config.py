"""
Class that contains all config DataClasses for the dataloaders.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""

from dataclasses import MISSING, dataclass


@dataclass
class DatasetConfig:
    # Supported values: ["CIFAR10"]
    dataset_name: str = MISSING
    dataset_dir: str = "../../datasets"
    image_size: int = 224


@dataclass
class CollateFunctionConfig:
    # Supported values: ["LIGHTLY_IMAGE", "NONE"]
    collate_fn_type: str = MISSING
    cj_prob: float = 0.5
    gausian_blur: float = 0.0


@dataclass
class DataLoaderConfig:
    dataset_cfg: DatasetConfig = MISSING
    collate_fn_cfg: CollateFunctionConfig = MISSING
    train: bool = True
    batch_size: int = 64
    shuffle: bool = False
    persistent_workers: bool = True
    num_workers: int = 0
