"""
Utility functions for dataloaders.

@Filename    utils.py
@Author      Kion
@Created     09/02/22
"""

import copy

import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataloader.config import DataLoaderConfig


def get_unaugmented_dataloader(dataloader: DataLoader) -> DataLoader:
    dataset = copy.deepcopy(dataloader.dataset)
    dataset.transform = T.Compose([dataset.transform, T.ToTensor()])
    return DataLoader(
        dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
        persistent_workers=dataloader.persistent_workers,
    )


def get_weak_aug_dataloader(dataloader: DataLoader, dataloader_cfg: DataLoaderConfig) -> DataLoader:
    dataset = copy.deepcopy(dataloader.dataset)
    dataset.transform = T.Compose(
        [
            dataset.transform,
            T.RandomResizedCrop(dataloader_cfg.dataset_cfg.image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )
    return DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        drop_last=False,
        num_workers=dataloader.num_workers,
        persistent_workers=dataloader.persistent_workers,
    )
