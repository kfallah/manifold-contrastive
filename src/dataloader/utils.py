"""
Utility functions for dataloaders.

@Filename    utils.py
@Author      Kion
@Created     09/02/22
"""

import copy

import lightly.data as data
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataloader.config import DataLoaderConfig


def get_weak_aug_dataloader(dataloader: DataLoader, dataloader_cfg: DataLoaderConfig) -> DataLoader:
    # Very hideous way of doing this, but STL is unique that it has a different split for just labels
    if dataloader_cfg.dataset_cfg.dataset_name == "STL10":
        dataset = torchvision.datasets.STL10(
            root=dataloader_cfg.dataset_cfg.dataset_dir, 
            transform=dataloader.dataset.transform, 
            split="train",
            download=True
        )
        dataset = data.LightlyDataset.from_torch_dataset(dataset, transform=dataset.transform)
    else:
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
