import torchvision
import torchvision.transforms as T

from dataloader.base import Dataset


class STL10(Dataset):

    def test_transform(self):
        return T.Compose(
            [T.Resize(70, interpolation=3), T.CenterCrop(64), self.get_base_transform()]
        )

    def train_dataset(self):
        t = self.get_ssl_transform()
        return torchvision.datasets.STL10(root=self.cfg.dataset_cfg.dataset_dir, split="train+unlabeled", download=True, transform=t)

    def eval_dataset(self):
        """
        Dataset used for downstream evaluation of contrastive features, likely uses standard
        augmentations.
        """
        t = self.test_transform()
        return torchvision.datasets.STL10(root=self.cfg.dataset_cfg.dataset_dir, split="train", download=True, transform=t)

    def val_dataset(self):
        """
        Dataset used for validation metrics, likely uses no augmentations
        """
        t = self.test_transform()
        return torchvision.datasets.STL10(root=self.cfg.dataset_cfg.dataset_dir, split="test", download=True, transform=t)
