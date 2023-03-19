import sys
import random
import os
from tqdm import tqdm
sys.path.append(os.path.dirname(os.getcwd()) + "/src/")

import torch
import numpy as np
import torchvision

import torchvision.transforms.transforms as T
from model.public.dip_ae import dip_ae, get_noise
from lightly.models.utils import deactivate_requires_grad

from matplotlib import pyplot as plt

default_device = torch.device('cuda:0')

def plot_dip_image(
    original_image, 
    dip_image, 
    save_path="dip_comparison.png",
    image_shape=(32, 32)
):
    fig, axs = plt.subplots(1, 2)
    # Resize images to the proper shape
    dip_image = T.Resize(image_shape)(dip_image)
    original_image = T.Resize(image_shape)(original_image)
    # dip_image = np.resize(dip_image, image_shape)
    # original_image = np.resize(original_image, image_shape)
    # Convert to numpy and permute the channels
    dip_image = dip_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    original_image = original_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    # Display the images
    axs[0].imshow(original_image)
    axs[1].imshow(dip_image)
    plt.savefig(save_path)

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

def load_resnet_backbone():
    backbone = torchvision.models.resnet50(pretrained=True).to(default_device)
    backbone.eval()
    backbone.fc = torch.nn.Identity()
    backbone.avgpool = torch.nn.Identity()
    # backbone.layer4 = torch.nn.Identity()
    deactivate_requires_grad(backbone)

    return backbone

def compute_dip_image(input_z, mse_lambda=1.0):
    print("Computing Deep Image Prior")
    avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    z_use = torch.tensor(input_z).to(default_device)
    z_use_avg = avg_pool(z_use.reshape(-1, 7, 7))[..., 0, 0]

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

    net_input = get_noise(32, 256).type(z_use.type()).detach().to(default_device)
    opt = torch.optim.Adam(list(net.parameters()) + list(net_input), lr=1e-3)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)

    for j in tqdm(range(3000)):
        x_hat = net(net_input)[:, :, :224, :224]
        z_hat = backbone(x_hat)
        z_avg = avg_pool(z_use.reshape(-1, 7, 7))[..., 0, 0]

        loss = torch.nn.functional.mse_loss(z_hat[0], z_use)
        loss = loss + mse_lambda * torch.nn.functional.mse_loss(z_hat[0], z_use)

        opt.zero_grad()
        loss.backward()
        opt.step()
        #scheduler.step()

        if j % 500 == 0:
            print(loss.item())

    x_hat[x_hat > 1] = 1
    x_hat[x_hat < 0] = 0

    return x_hat[0]

if __name__ == "__main__":
    #backbone_load = torch.load("../results/SimCLR-Linear_09-13-2022_14-08-55/checkpoints/checkpoint_epoch299.pt")
    print("Loading resnet backbone")
    backbone = load_resnet_backbone()
    print("Loading CIFAR10 dataset")
    cifar10 = torchvision.datasets.CIFAR10(
        "../datasets",
        train=True,
        transform=T.Compose([
            T.Resize(256),
            T.RandomCrop(224, 4),
            T.ToTensor(),
        ]),
        download=True
    )
    input_images = []
    dip_images = []
    for image_index in range(3):
        input_image = cifar10[image_index][0].unsqueeze(0)
        input_z = backbone(input_image.to(default_device)).detach().cpu()
        # Compute the DIP image
        dip_image = compute_dip_image(input_z).detach().cpu()

        input_images.append(input_image)
        dip_images.append(dip_image)

    plot_multiple_dip_images(
        input_images, 
        dip_images,
        save_path="dip_comparisons_layer4.png"
    )