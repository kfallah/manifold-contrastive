# Repurposed from dataset structure at:
# https://github.com/htdt/self-supervised/blob/master/datasets/base.py

import random
from abc import ABCMeta, abstractmethod
from functools import lru_cache

import torch

from dataloader.config import DataLoaderConfig
from dataloader.transform import MultiSample, get_base_augmentation, get_ssl_augmentation, get_weak_augmentation


# Dataset that returns images with nearest neighbor
class NaturalTransformationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, nn_path):
        super(NaturalTransformationDataset, self)
        self.dataset = dataset
        self.nn_graph = torch.load(nn_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x0, label = self.dataset.__getitem__(idx)
        neighbor = random.randrange(len(self.nn_graph[idx]))
        x1 = self.dataset.__getitem__(int(self.nn_graph[idx, neighbor]))
        return (x0, x1[0], label)


class Dataset(metaclass=ABCMeta):
    def __init__(self, cfg: DataLoaderConfig):
        self.cfg = cfg

    def get_ssl_transform(self):
        if self.cfg.ssl_aug_cfg.enable_weak_z0_aug:
            transform = [
                get_ssl_augmentation(self.cfg.ssl_aug_cfg, self.cfg.dataset_cfg.image_size),
                get_ssl_augmentation(self.cfg.ssl_aug_cfg, self.cfg.dataset_cfg.image_size),
                get_weak_augmentation(self.cfg.ssl_aug_cfg, self.cfg.dataset_cfg.image_size),
            ]
        else:
            transform = 2 * [get_ssl_augmentation(self.cfg.ssl_aug_cfg, self.cfg.dataset_cfg.image_size)]
        t = MultiSample(transform)
        return t

    def get_base_transform(self):
        return get_base_augmentation(self.cfg.ssl_aug_cfg)

    def get_base_transform(self):
        return get_base_augmentation(self.cfg.ssl_aug_cfg)

    @abstractmethod
    def train_dataset(self):
        """
        Dataset used for self-supervised pretraining, likely uses two augmented views of an
        image.
        """
        raise NotImplementedError

    @abstractmethod
    def eval_dataset(self):
        """
        Dataset used for downstream evaluation of contrastive features, likely uses standard
        augmentations.
        """
        raise NotImplementedError

    @abstractmethod
    def val_dataset(self):
        """
        Dataset used for validation metrics, likely uses no augmentations
        """
        raise NotImplementedError

    @property
    @lru_cache()
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset(),
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.cfg.persistent_workers,
        )

    @property
    @lru_cache()
    def eval_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.eval_dataset(),
            batch_size=self.cfg.eval_batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.cfg.persistent_workers,
        )

    @property
    @lru_cache()
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset(),
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.cfg.persistent_workers,
        )
