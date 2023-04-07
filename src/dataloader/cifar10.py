import torchvision

from dataloader.base import Dataset, NaturalTransformationDataset


class CIFAR10(Dataset):
    def train_dataset(self):
        t = self.get_ssl_transform()
        dataset = torchvision.datasets.CIFAR10(
            root=self.cfg.dataset_cfg.dataset_dir, train=True, download=True, transform=t
        )
        if self.cfg.dataset_cfg.enable_nn_pair:
            dataset = NaturalTransformationDataset(dataset, self.cfg.dataset_cfg.nn_pair_path)
        return dataset

    def eval_dataset(self):
        """
        Dataset used for downstream evaluation of contrastive features, likely uses standard
        augmentations.
        """
        t = self.get_base_transform()
        return torchvision.datasets.CIFAR10(
            root=self.cfg.dataset_cfg.dataset_dir, train=True, download=True, transform=t
        )

    def val_dataset(self):
        """
        Dataset used for validation metrics, likely uses no augmentations
        """
        t = self.get_base_transform()
        return torchvision.datasets.CIFAR10(
            root=self.cfg.dataset_cfg.dataset_dir, train=False, download=True, transform=t
        )
