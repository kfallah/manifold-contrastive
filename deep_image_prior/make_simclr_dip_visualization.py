import random
import os
from tqdm import tqdm
import torch
import numpy as np
import torchvision
import torchvision.transforms.transforms as T
import torch.nn as nn
import argparse
from matplotlib import pyplot as plt
import wandb
import omegaconf
import sys

sys.path.append("../src")

from model.public.dip_ae import dip_ae, get_noise
from methods.load import load_backbone_model
from methods.deep_image_prior import compute_dip_image
from methods.plotting import plot_multiple_dip_images

default_device = torch.device('cuda:0')

if __name__ == "__main__":
    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--last_layer", default="layer3")
    parser.add_argument("--learning_rate", default=1e-3)
    parser.add_argument("--number_of_images", default=2)
    parser.add_argument("--fixed_noise", default=True)
    args = parser.parse_args()

    print("Loading resnet backbone")
    backbone = load_backbone_model(
        "../results/simclr_cifar10/cfg_simclr_cifar10.yaml",
        "../results/simclr_cifar10/simclr_cifar10.pt",
        args.last_layer
    )
    print("Loading CIFAR10 dataset")
    cifar10 = torchvision.datasets.CIFAR10(
        "datasets",
        train=True,
        transform=T.Compose([
            T.Resize(256),
            # T.RandomCrop(224, 4),
            T.ToTensor(),
        ]),
        download=True
    )

    input_images = []
    dip_images = []
    for image_index in range(args.number_of_images):
        wandb.init(
            project="deep-image-prior",
            config={
                "last_layer": args.last_layer,
                "fixed_noise": args.fixed_noise,
                "learning_rate": args.learning_rate,
            }
        )
        input_image = cifar10[image_index][0].unsqueeze(0)
        print(f"Input image shape: {input_image.shape}")
        # input_image = T.Resize(32)(input_image)
        print(f"Resized image shape: {input_image.shape}")
        input_z = backbone(input_image.to(default_device)).detach().cpu()
        print(f"Input z shape: {input_z.shape}")
        # Compute the DIP image
        dip_image = compute_dip_image(
            input_z,
            input_image,
            backbone, 
            fixed_noise=args.fixed_noise,
            learning_rate=args.learning_rate,
        ).detach().cpu()

        input_images.append(input_image)
        dip_images.append(dip_image)

        wandb.log({
            "input_image": wandb.Image(input_image),
            "dip_image": wandb.Image(dip_image),
        })

        wandb.finish()

    plot_multiple_dip_images(
        input_images, 
        dip_images,
        save_path="plots/simclr_dip_comparisons_resnet_18_layer3.png"
    )