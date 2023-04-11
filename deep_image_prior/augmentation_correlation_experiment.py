"""

"""
import argparse
import omegaconf
import torchvision
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from train_layer_inverse import load_backbone_model
from methods.load import load_transop_model, deactivate_requires_grad
import torch

default_device = torch.device("cuda:0")

def apply_image_augmentation(image, intensity, img_size=(32, 32)):
    # Load the default augmentation
    return T.Compose([
        T.RandomResizedCrop(
            img_size,
            scale=(intensity, intensity), # Use scale as intensity
            ratio=(1.0, 1.0),
            interpolation=3,
        ),
    ])(image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_augmented_images",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--input_image_shape",
        default=(32, 32),
    )
    parser.add_argument(
        "--num_base_images",
        default=100,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="pretrained/prioraug_vi_100s_l0.01_dbd_eigreg5e-6",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="pretrained/config.yaml",
    )

    args = parser.parse_args()
    # Load the config
    config = omegaconf.OmegaConf.load(args.config_path)
    config.model_cfg.backbone_cfg.load_backbone = None
    # # Load up a backbone model
    # backbone = load_backbone_model(
    #     "../results/simclr_cifar10/cfg_simclr_cifar10.yaml",
    #     "../results/simclr_cifar10/simclr_cifar10.pt",
    #     "penultimate",
    #     input_image_shape=args.input_image_shape,
    # ).to(default_device)
    # Load the pair encoder
    print("Loading transport operator")
    transop_model = load_transop_model(
        config.model_cfg,
        dataset_name="CIFAR10",
        device=default_device,
        checkpoint_path=args.checkpoint_path
    )
    backbone = transop_model.backbone.backbone_network
    transop = transop_model.contrastive_header.transop_header.transop
    vi_encoder = transop_model.contrastive_header.transop_header.coefficient_encoder

    def compute_vi_coefficients(image_a, image_b):
        # Embed the images
        z_a = backbone(image_a.unsqueeze(0).to(default_device))
        z_b = backbone(image_b.unsqueeze(0).to(default_device))
        # Compute the transport operator coefficients
        coeffs = vi_encoder(z_a, z_b, transop).samples.squeeze()

        return coeffs
        
    # Select a random CIFAR image
    train_cifar10 = torchvision.datasets.CIFAR10(
        "datasets",
        train=True,
        transform=T.Compose([
            T.Resize(args.input_image_shape),
            T.ToTensor(),
        ]),
        download=True
    )
    # Uniformly sample augmentation intensities
    augmentation_intensities = np.linspace(0.5, 1.0, args.num_augmented_images)
    # Apply N augmentations to the image, storing their intensities. 
    # augmented_images = []
    intensities = []
    coeff_mags = []
    print("Computing coefficients for augmented images...")
    for image_index in tqdm(range(args.num_base_images)):
        base_image = train_cifar10[np.random.randint(len(train_cifar10))][0]
        for augmentation_index in range(args.num_augmented_images):
            intensity = augmentation_intensities[augmentation_index]
            # Augment the image
            augmented_image = apply_image_augmentation(base_image, augmentation_index)
            # Compute coefficients between the base image and the augmented image
            coeffs = compute_vi_coefficients(base_image, augmented_image).detach().cpu().numpy()
            coeff_mag = np.linalg.norm(coeffs)
            intensities.append(intensity)
            coeff_mags.append(coeff_mag)
    # Make a matplotlib scatter plot
    plt.figure()
    plt.scatter(intensities, coeff_mags)
    plt.xlabel("Augmentation Intensity")
    plt.ylabel("VI Coefficient Magnitude")
    # Add the correlation to the legend
    corr = np.corrcoef(intensities, coeff_mags)[0, 1]
    plt.legend([f"Correlation: {corr:.3f}"])
    plt.savefig("plots/augmentation_correlation.png")