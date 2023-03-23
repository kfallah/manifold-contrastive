import torchvision
import torchvision.transforms as T

from dataloader.base import Dataset


class TinyImagenet(Dataset):
    def train_dataset(self):
        t = self.get_ssl_transform()
        directory = self.cfg.dataset_cfg.dataset_dir + "/train"
        return torchvision.datasets.ImageFolder(directory, transform=t)

    def eval_dataset(self):
        """
        Dataset used for downstream evaluation of contrastive features, likely uses standard
        augmentations.
        """
        t = self.get_base_transform()
        directory = self.cfg.dataset_cfg.dataset_dir + "/train"
        return torchvision.datasets.ImageFolder(directory, transform=t)

    def val_dataset(self):
        """
        Dataset used for validation metrics, likely uses no augmentations
        """
        t = self.get_base_transform()
        directory = self.cfg.dataset_cfg.dataset_dir + "/val"
        return torchvision.datasets.ImageFolder(directory, transform=t)
