import sys
sys.path.append("../../src/")

from dataloader.ssl_dataloader import get_dataset
import os
import argparse
# from dataloader.contrastive_dataloader import get_dataloader
# from dataloader.utils import get_unaugmented_dataloader
from eval.utils import encode_features
from model.model import Model

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from lightly.models.utils import deactivate_requires_grad
import matplotlib.pyplot as plt
import omegaconf
import numpy as np

default_device = torch.device('cuda:0')
    
def load_transop_model(
    model_cfg, 
    dataset_name="CIFAR10", 
    device=default_device,
    checkpoint_path=None,
    device_index=[0]
):
    print(os.path.exists(checkpoint_path))
    # Load model
    model = Model.initialize_model(model_cfg, dataset_name=dataset_name, devices=device_index)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['model_state'], strict=False)

    return model

def compute_operator_path_range(
    model,
    operator_index=0,
    device=default_device,
    num_samples=1000,
):
    """
        Computes the distribution of an operator's
        coefficients and returns the range of 5th to 95th 
        percentiles. 
    """
    # Get a bunch of random image samples
    # TODO fix this
    return (-3, 3)

def compute_operator_path_samples(
    initial_z,
    model,
    path_range=None,
    operator_index=0,
    num_samples=10,
    device=default_device,
):
    """
        Sample points along a manifold operator path
    """
    if path_range is None:
        path_range = compute_operator_path_range(
            model,
            operator_index=operator_index,
        )
    initial_z = initial_z.to(device)
    print(f"initial_z.shape: {initial_z.shape}")
    transop = model.contrastive_header.transop_header.transop
    # Get the operator
    psi = transop.get_psi()
    print("psi.shape: ", psi.shape)
    psi_operator = psi[:, operator_index]
    psi_operator = psi_operator.unsqueeze(0).to(device)
    num_coefficients = psi.shape[1]
    print(f"num_coefficients: {num_coefficients}")
    print(f"psi_operator.shape: {psi_operator.shape}")
    # Compute samples in operator time domain for given path range
    time_domain_samples = torch.linspace(
        path_range[0],
        path_range[1],
        num_samples,
    ).to(device)
    time_domain_samples = time_domain_samples[:, None, None, None]
    print(f"time_domain_samples.shape: {time_domain_samples.shape}")
    # Apply the operator transformation
    # NOTE: I am only applying a single operator, so I don't need to sum
    # Transform initial_z given the coefficients
    T = torch.matrix_exp(time_domain_samples * psi_operator)
    operator_path_samples = (
        T @ initial_z.reshape(1, 8, -1, 1)
    ).reshape(num_samples, -1)
    assert list(operator_path_samples.shape) == [num_samples, 512]

    return operator_path_samples

def compute

def plot_operator_path_samples(
    initial_z_indices,
    model,
    embeddings,
    path_range=None,
    operator_index=0,
    num_samples=5,
    true_path_num_samples=500,
    save_path="nn_path_visualizations.png",
):
    """
        Plots the nearest images to the points along a
        manifold path for a given transport operator.
    """
    if not isinstance(initial_z_indices, list):
        initial_z_indices = [initial_z_indices]
    initial_z = []
    for initial_z_index in initial_z_indices:
        initial_z.append(embeddings.feature_list[initial_z_index])

    if path_range is None:
        path_range = compute_operator_path_range(
            model,
            operator_index=operator_index,
        )
    fig, axs = plt.subplots(
        len(initial_z),
        num_samples + 1, 
        figsize=((num_samples + 1) * 1.5, len(initial_z)),
        dpi=300
    )
    for z_index in range(len(initial_z)):
        init_z = initial_z[z_index]
        # Compute points uniformly sampled along the manifold path
        operator_path_samples = compute_operator_path_samples(
            init_z,
            model,
            path_range=path_range,
            operator_index=operator_index,
            num_samples=num_samples,
        )
        # Compute the nearest neighbors and images
        nn_samples, images = compute_manifold_path_nearest_images(
            operator_path_samples,
            embeddings, 
        )
        # Plot the initial image
        initial_image = embeddings.x[initial_z_indices[z_index]]
        image = initial_image.permute(1, 2, 0) # [:, :, [2, 1, 0]]
        image = (image - image.min()) / (image.max() - image.min())
        axs[z_index, 0].imshow(image)
        # axs[index].set_title(f"Path {index}")
        axs[z_index, 0].axis("off")
        axs[z_index, 0].set_title("Initial Image")
        # Plot the images
        # axs = image_fig.subplots(1, num_samples)
        for index in range(num_samples):
            # print(images[index].shape)
            image = images[index].permute(1, 2, 0) # [:, :, [2, 1, 0]]
            image = (image - image.min()) / (image.max() - image.min())
            axs[z_index, index + 1].imshow(image)
            # axs[index].set_title(f"Path {index}")
            axs[z_index, index + 1].axis("off")

    plt.savefig(save_path)

if __name__ == "__main__":
    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="prioraug_vi_100s_l0.01_dbd_eigreg5e-6",
    )
    parser.add_argument(
        "--num_paths",
        default=5,
    )
    parser.add_argument(
        "--num_samples",
        default=10,
    )
    parser.add_argument(
        "--resnet_features_path",
        default="resnet_features.pt",
    )
    parser.add_argument(
        "--plot_save_path",
        default="nn_path_visualizations.png",
    )

    args = parser.parse_args()
    # Load the config
    config = omegaconf.OmegaConf.load(args.config_path)
    config.model_cfg.backbone_cfg.load_backbone = None
    # with open(args.config_path, 'r') as stream:
    #    config = yaml.safe_load(stream)
    # Load the tansport operator model
    print("Loading the model...")
    model = load_transop_model(
        config.model_cfg,
        dataset_name="CIFAR10",
        device=default_device,
        checkpoint_path=args.checkpoint_path
    )
    # Load the resnet model
    backbone = model.backbone
    # backbone = load_resnet_backbone(
    #     "CIFAR10",
    #     arch_name="resnet18",
    #     backbone_cfg=config.model_cfg.backbone_cfg,
    # )
    # Load an image dataset
    # cifar10 = load_cifar10()
    # Embed the entire cifar10 datase
    # Manually override directory for dataloaders
    # config.train_dataloader_cfg.dataset_cfg.dataset_dir = "../../datasets"
    # config.train_dataloader_cfg.batch_size = 32
    # config.eval_dataloader_cfg.dataset_cfg.dataset_dir = "../../datasets"
    # Load dataloaders
    # train_dataset, train_dataloader = get_dataloader(config.train_dataloader_cfg)
    # eval_dataset, eval_dataloader = get_dataloader(config.eval_dataloader_cfg)
    # unaugmented_train_dataloader = get_unaugmented_dataloader(train_dataloader)
    # Load dataloaders
    config.dataloader_cfg.dataset_cfg.dataset_dir = "../../datasets"
    config.dataloader_cfg.train_batch_size = 32
    dataset = get_dataset(config.dataloader_cfg)
    train_dataloader = dataset.eval_dataloader
    test_dataloader = dataset.val_dataloader
    # Get encoding of entire dataset
    print("Encoding the dataset...")
    embeddings = encode_features(
        model,
        train_dataloader,
    #    unaugmented_train_dataloader, 
        default_device
    )
    # embeddings = embeddings.feature_list
    # Make the plot
    print("Plotting operator path samples")
    # Select random z from embeddings
    print(dataset)
    init_z_inds = []
    for i in range(args.num_paths):
        init_z_inds.append(
            np.random.randint(0, len(embeddings.feature_list))
        )
        
    plot_operator_path_samples(
        init_z_inds,
        model,
        embeddings,
        operator_index=0,
        num_samples=args.num_samples,
        save_path=args.plot_save_path,
        path_range=(-3, 3)
    )
