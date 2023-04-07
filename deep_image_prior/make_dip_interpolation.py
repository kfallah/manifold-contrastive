"""
    Interpolates the latent manifold for a given example using the 
    transport operator model, and visualizes the modified embeddings 
    using the Deep Image Prior (DIP). 
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from lightly.models.utils import deactivate_requires_grad
import matplotlib.pyplot as plt
import omegaconf
import numpy as np
import argparse
import os
import sys
sys.path.append("../src/")

from dataloader.ssl_dataloader import get_dataset
# from dataloader.contrastive_dataloader import get_dataloader
# from dataloader.utils import get_unaugmented_dataloader
from eval.utils import encode_features
from model.model import Model
from methods.load import load_backbone_model, load_transop_model
from methods.plotting import plot_operator_path_samples
from methods.manifold_interpolation import compute_interpolation_dip_images
from methods.manifold_interpolation import compute_operator_path_samples, compute_operator_path_range

default_device = torch.device('cuda:0')

if __name__ == "__main__":
    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="../results/simclr_cifar10/cfg_simclr_cifar10.yaml",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="../results/simclr_cifar10/simclr_cifar10.pt",
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
        "--plot_save_path",
        default="interpolation_paths.png",
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
    print(model)
    backbone = model.backbone.backbone_network
    print(f"Loaded model with backbone: {backbone}")
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
        default_device
    )
    # Make the plot
    print("Plotting operator path samples")
    # Select random z from embeddings
    init_z_inds = []
    input_images = []
    interpolation_images = []
    for i in range(args.num_paths):
        init_z_ind = np.random.randint(0, len(embeddings.feature_list))
        input_image = embeddings.x[init_z_ind]
        init_z = embeddings.feature_list[init_z_ind]

        operator_path_samples = compute_operator_path_samples(
            init_z,
            model,
            path_range=compute_operator_path_range(model),
            operator_index=0,
            num_samples=args.num_paths,
            device=default_device,
        )
    
        ims = compute_interpolation_dip_images(
            operator_path_samples,
            input_image,
            backbone,
            mse_lambda=1.0,
            learning_rate=1e-3,
            fixed_noise=False,
            return_network=False,
        )
        input_images.append(input_image)
        interpolation_images.append(ims)

    plot_operator_path_samples(
        input_images,
        inetpolation_images,
        save_path=args.plot_save_path,
    )