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
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from dataloader.config import (CollateFunctionConfig, DataLoaderConfig,
                               DatasetConfig)


def get_dataset(config: DatasetConfig, enable_collate: bool = True, train: bool = True) -> Dataset:
    """Return torchvision dataset for a given config.

    Args:
        config (DatasetConfig): Dataset config
        enable_collate (bool): Whether a collate function is enabled with the dataset.
        train (bool): Whether to load the training dataset.

    Returns:
        Dataset: Returns the torchvision dataset.
    """
    # Return transform for conversion to lightly dataset
    transform = None
    transform_list = [] if enable_collate else [T.ToTensor()]

    if config.dataset_name == "CIFAR10":
        if config.image_size != 32:
            # Prepend transform so it comes before ToTensor()
            transform_list.insert(0, T.Resize(config.image_size))
        transform = T.Compose(transform_list)
        dataset = torchvision.datasets.CIFAR10(config.dataset_dir, transform=transform, train=train, download=True)
    elif config.dataset_name == "CIFAR100":
        if config.image_size != 32:
            # Prepend transform so it comes before ToTensor()
            transform_list.insert(0, T.Resize(config.image_size))
        transform = T.Compose(transform_list)
        dataset = torchvision.datasets.CIFAR100(config.dataset_dir, transform=transform, train=train, download=True)
    elif config.dataset_name == "ImageNet":
        transform = T.Compose(transform_list)
        split = "train" if train else "val"
        directory = config.dataset_dir + "/" + split
        dataset = torchvision.datasets.ImageFolder(directory, transform=transform)
    else:
        raise NotImplementedError
    return dataset, transform


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
            image_size,
            cj_prob=config.cj_prob,
            cj_bright=config.cj_bright,
            cj_contrast=config.cj_contrast,
            cj_sat=config.cj_sat,
            cj_hue=config.cj_hue,
            min_scale=config.min_scale,
            gaussian_blur=config.gausian_blur,
            normalize={"mean": list(config.normalize_mean), "std": list(config.normalize_std)}
            if config.normalize_mean
            else None,
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
    pytorch_dataset, transform = get_dataset(
        config.dataset_cfg, config.collate_fn_cfg.collate_fn_type != "NONE", config.train
    )
    lightly_dataset = data.LightlyDataset.from_torch_dataset(pytorch_dataset, transform=transform)
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
