import torchvision.transforms as T

from dataloader.config import DataLoaderConfig, SSLAugmentationConfig


def get_base_augmentation(cfg: SSLAugmentationConfig):
    return T.Compose([
        T.ToTensor(), T.Normalize(cfg.normalize_mean, cfg.normalize_std)
    ])

def get_weak_augmentation(cfg: SSLAugmentationConfig, img_size: int):
    """Return a torchvision transform that gives two views of an image.

    Args:
        config (SSLAugmentationConfig): Augmentation config.
        image_size (int): Image size in pixels.

    Returns:
        torchvision.transform: Transform that gives two views of the image.
    """
    return T.Compose(
        [
            T.RandomResizedCrop(
                img_size,
                scale=(0.5, 1.0),
                ratio=(0.75, (4 / 3)),
                interpolation=3,
            ),
            get_base_augmentation(cfg),
        ]
    )

def get_ssl_augmentation(cfg: SSLAugmentationConfig, img_size: int):
    """Return a torchvision transform that gives two views of an image.

    Args:
        config (SSLAugmentationConfig): Augmentation config.
        image_size (int): Image size in pixels.

    Returns:
        torchvision.transform: Transform that gives two views of the image.
    """
    return T.Compose(
        [
            T.RandomApply(
                [T.ColorJitter(cfg.cj_bright, cfg.cj_contrast, cfg.cj_sat, cfg.cj_hue)], p=cfg.cj_prob
            ),
            T.RandomGrayscale(p=cfg.gray_scale),
            T.RandomResizedCrop(
                img_size,
                scale=(cfg.min_scale, 1.0),
                ratio=(0.75, (4 / 3)),
                interpolation=3,
            ),
            T.RandomHorizontalFlip(p=cfg.horiz_flip),
            get_base_augmentation(cfg),
        ]
    )

class MultiSample:
    """ generates n samples with augmentation """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return tuple(t(x) for t in self.transform)