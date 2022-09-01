"""
Use wrapper from lightly.ai to create a self-supervised dataloader for
contrastive learning.

@Filename    contrastive_dataloader.py
@Author      Kion
@Created     08/31/22
"""

from typing import Callable, Tuple

import lightly.data as data
import torchvision
from torch.utils.data import DataLoader, Dataset

from dataloader.config import CollateFunctionConfig, DataLoaderConfig, DatasetConfig


def get_dataset(config: DatasetConfig, train: bool = True) -> Dataset:
    """Return torchvision dataset for a given config.

    Args:
        config (DatasetConfig): Dataset config
        train (bool): Whether to load the training dataset.

    Returns:
        Dataset: Returns the torchvision dataset.
    """
    if config.dataset_name == "CIFAR10":
        transform = None if config.image_size == 32 else torchvision.transforms.Resize(config.image_size)
        dataset = torchvision.datasets.CIFAR10(config.dataset_dir, transform=transform, train=train, download=False)
    else:
        raise NotImplementedError
    return dataset


def get_collate_fn(config: CollateFunctionConfig, image_size: int) -> Callable:
    """Return a lightly collate function from the provided config.

    Args:
        config (CollateFunctionConfig): Dataset config.
        image_size (int): Image size in pixels.

    Returns:
        collate_fn: Returns the callable collate function
    """
    if config.collate_fn_type == "LIGHTLY_IMAGE":
        collate_fn = data.collate.ImageCollateFunction(
            image_size, cj_prob=config.cj_prob, gaussian_blur=config.gausian_blur
        )
    elif config.collate_fn_type == "NONE":
        collate_fn = None
    else:
        raise NotImplementedError
    return collate_fn


def get_dataloader(config: DataLoaderConfig) -> Tuple[Dataset, DataLoader]:
    """
    For a given DataLoaderConfig, return the respective dataset and dataloader.
    Augmentations are selected from the provided config.


    Args:
        config (DataLoaderConfig): DataLoader config from hydra.

    Returns:
        Tuple[Dataset, DataLoader]: Returns both the dataset and the
        dataloader.
    """
    pytorch_dataset = get_dataset(config.dataset_cfg, config.train)
    lightly_dataset = data.LightlyDataset.from_torch_dataset(pytorch_dataset)
    collate_fn = get_collate_fn(config.collate_fn_cfg, config.dataset_cfg.image_size)
    dataloader = DataLoader(
        lightly_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        persistent_workers=config.persistent_workers,
    )

    return (pytorch_dataset, dataloader)
