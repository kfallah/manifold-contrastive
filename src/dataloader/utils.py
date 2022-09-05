"""
Utility functions for dataloaders.

@Filename    utils.py
@Author      Kion
@Created     09/02/22
"""

import copy

import torchvision.transforms as T
from torch.utils.data import DataLoader


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
