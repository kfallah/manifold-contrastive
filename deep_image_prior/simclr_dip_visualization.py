import sys
import random
import os
from tqdm import tqdm
import argparse
sys.path.append(os.path.dirname(os.getcwd()) + "/src/")

import torch
import numpy as np
import torchvision

import torchvision.transforms.transforms as T
from model.public.dip_ae import dip_ae, get_noise
from lightly.models.utils import deactivate_requires_grad

import torch.nn as nn

from matplotlib import pyplot as plt

import wandb
import omegaconf
from utils import load_transop_model
from make_individual_dip import load_resnet_backbone

default_device = torch.device('cuda:0')

def plot_multiple_dip_images(
    original_images,
    dip_images,
    save_path="dip_comparisons.png",
    image_shape=(32, 32)
):
    fig, axs = plt.subplots(len(original_images), 2, figsize=(len(original_images), 2))
    for index in range(len(original_images)):
        original_image = original_images[index]
        dip_image = dip_images[index]
        # Resize images to the proper shape
        dip_image = T.Resize(image_shape)(dip_image)
        original_image = T.Resize(image_shape)(original_image)
        # dip_image = np.resize(dip_image, image_shape)
        # original_image = np.resize(original_image, image_shape)
        # Convert to numpy and permute the channels
        dip_image = dip_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        original_image = original_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        # Display the images
        axs[index, 0].imshow(original_image)
        axs[index, 0].axis('off')
        axs[index, 1].imshow(dip_image)
        axs[index, 1].axis('off')

    plt.savefig(save_path)

def make_dip_ae(z_use):
    
    """
    net = dip_ae(
        32, 
        3, 
        num_channels_down=[8, 16, 32],
        num_channels_up=[8, 16, 32],
        num_channels_skip=[4, 4, 4],
        filter_size_down=[3, 5, 7], 
        filter_size_up=[3, 5, 7],
        upsample_mode='nearest', 
        downsample_mode='avg',
        need_sigmoid=False, 
        pad='zero', 
        act_fun='LeakyReLU'
    ).type(z_use.type()).to(default_device)
    """

    """
    # Give me a unet architecture for cifar10 32 x32
    net = dip_ae(
        32,
        3,
        num_channels_down=[32, 32, 32],
        num_channels_up=[32, 32, 32],
        num_channels_skip=[4, 4, 4],
        filter_size_down=[3, 3, 3],
        filter_size_up=[3, 3, 3],
        upsample_mode='nearest',
        downsample_mode='avg',
        need_sigmoid=False,
        pad='zero',
        act_fun='LeakyReLU'
    ).type(z_use.type()).to(default_device)
    """
    net = dip_ae(
        32, 
        3, 
        num_channels_down=[16, 32, 64, 128, 128, 128],
        num_channels_up=[16, 32, 64, 128, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4, 4],
        filter_size_down=[7, 7, 5, 5, 3, 3], 
        filter_size_up=[7, 7, 5, 5, 3, 3],
        upsample_mode='nearest', 
        downsample_mode='avg',
        need_sigmoid=False, 
        pad='zero', 
        act_fun='LeakyReLU'
    ).type(z_use.type()).to(default_device)

    return net

def compute_dip_image(
    input_z, 
    input_x, 
    mse_lambda=1.0,
    learning_rate=1e-3,
    fixed_noise=False,
):
    print("Computing Deep Image Prior")
    # avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
    z_use = torch.tensor(input_z).to(default_device)
    # z_use_avg = avg_pool(z_use.reshape(-1, 7, 7))[..., 0, 0]
    net = make_dip_ae(z_use)
    net_input = get_noise(32, 256).type(z_use.type()).detach().to(default_device)
    print(f"Net input shape: {net_input.shape}")
    if fixed_noise:
        opt = torch.optim.Adam(
            list(net.parameters()),
            lr=learning_rate
        ) 
    else:
        opt = torch.optim.Adam(
            list(net.parameters()) + list(net_input),
            lr=learning_rate
        )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)

    for j in tqdm(range(3000)):
        x_hat = net(net_input) # [:, :, :224, :224]
        z_hat = backbone(x_hat)

        loss = mse_lambda * torch.nn.functional.mse_loss(z_hat[0], z_use)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        if j % 10 == 0:
            wandb.log({
                "loss": loss.item(),
                "x_loss": torch.norm(
                    x_hat.detach().cpu() - input_x.detach().cpu(),
                    p=1
                ).item(),
            })

    x_hat[x_hat > 1] = 1
    x_hat[x_hat < 0] = 0

    return x_hat[0]

def load_backbone_model(config_path, checkpoint_path, final_layer="layer3"):
    # Load the config
    config = omegaconf.OmegaConf.load(config_path)
    config.model_cfg.backbone_cfg.load_backbone = None
    # Load the tansport operator model
    print("Loading the model...")
    model = load_transop_model(
        config.model_cfg,
        dataset_name="CIFAR10",
        device=default_device,
        checkpoint_path=checkpoint_path
    )

    backbone = model.backbone.backbone_network
    """
    backbone.avgpool = torch.nn.Identity()
    backbone.fc = torch.nn.Identity()
    """
    if final_layer == "layer3":
        backbone = nn.Sequential(*backbone._modules.values())[:-3]
    elif final_layer == "layer4":
        backbone = nn.Sequential(*backbone._modules.values())[:-2]

    backbone = nn.Sequential(
        nn.AvgPool2d(8, 8),
        backbone,
    )

    deactivate_requires_grad(backbone)

    return backbone

if __name__ == "__main__":
    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--last_layer", default="layer4")
    parser.add_argument("--learning_rate", default=1e-3)
    parser.add_argument("--fixed_noise", default=True)
    args = parser.parse_args()

    #backbone_load = torch.load("../results/SimCLR-Linear_09-13-2022_14-08-55/checkpoints/checkpoint_epoch299.pt")
    print("Loading resnet backbone")
    backbone = load_backbone_model(
        "../results/simclr_cifar10/cfg_simclr_cifar10.yaml",
        "../results/simclr_cifar10/simclr_cifar10.pt",
        args.last_layer
    )
    print("Loading CIFAR10 dataset")
    cifar10 = torchvision.datasets.CIFAR10(
        "../datasets",
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
    for image_index in range(2):
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
        save_path="plots/simclr_dip_comparisons_resnet_18_layer4.png"
    )