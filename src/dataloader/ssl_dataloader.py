"""
Use wrapper from W-MSE codebase to create a self-supervised dataloader for
contrastive learning.

Repurposed code from:
https://github.com/htdt/self-supervised/blob/d9662c8d07dafd194a9045375f4f6aa09f5b03e9/datasets/transforms.py

@Filename    contrastive_dataloader.py
@Author      Kion
@Created     08/31/22
"""

from dataloader.base import Dataset
from dataloader.cifar10 import CIFAR10
from dataloader.cifar100 import CIFAR100
from dataloader.config import DataLoaderConfig, SSLAugmentationConfig
from dataloader.stl10 import STL10
from dataloader.tinyimagenet import TinyImagenet

DATASETS = {"CIFAR10": CIFAR10, "CIFAR100": CIFAR100, "STL10": STL10, "TinyImagenet": TinyImagenet}

def get_dataset(cfg: DataLoaderConfig) -> Dataset:
    return DATASETS[cfg.dataset_cfg.dataset_name](cfg)