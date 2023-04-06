import sys
sys.path.append("../../src/")

from dataloader.ssl_dataloader import get_dataset

import argparse
from eval.utils import encode_features
import omegaconf
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from make_nn_path_visualizations import load_transop_model

default_device = "cuda:0"

def plot_vi_distribution(
    model,
    embeddings,
    num_samples=1000,
    operator_index=0
):
    coefficients_list = []
    for sample_index in range(num_samples):
        # Sample two random features
        feature_index_1 = np.random.randint(0, embeddings.feature_list.shape[0])
        feature_index_2 = np.random.randint(0, embeddings.feature_list.shape[0])
        feature_1 = embeddings.feature_list[feature_index_1].to(default_device)
        feature_2 = embeddings.feature_list[feature_index_2].to(default_device)
        # Compute the vi coefficient
        coeff_enc = model.contrastive_header.transop_header.coefficient_encoder
        coefficients = coeff_enc(
            feature_1.unsqueeze(0),
            feature_2.unsqueeze(0),
            model.contrastive_header.transop_header.transop
        )
        # print(f"coefficients.shape: {coefficients.samples.shape}")
        samples = coefficients.samples.squeeze()
        coeffs = samples[operator_index].detach().cpu().numpy()
        coefficients_list.append(coeffs)
        # coeff_mean = torch.mean(samples, dim=0)
        # coeff_means.append(coeff_mean)
    # coeff_means = torch.stack(coeff_means)
    coefficients_list = np.stack(coefficients_list)
    # Plot the distribution
    plt.figure()
    sns.distplot(coefficients_list)
    plt.savefig("vi_distribution.png")

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
        default=50,
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

    plot_vi_distribution(
        model,
        embeddings,
        num_samples=1000
    )