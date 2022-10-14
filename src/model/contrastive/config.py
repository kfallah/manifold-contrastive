"""
Config for all the different contrastive headers.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""
from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="exp_cfg/model_cfg/header_cfg", name="simclr", node=ProjectionHeaderConfig
    )
    cs.store(
        group="exp_cfg/model_cfg/header_cfg",
        name="proj_pred",
        node=ProjectionPredictionHeaderConfig,
    )
    cs.store(
        group="exp_cfg/model_cfg/header_cfg",
        name="transop_header",
        node=TransportOperatorConfig,
    )


@dataclass
class ContrastiveHeaderConfig:
    header_name: str = "DefaultHeader"


@dataclass
class ProjectionPredictionHeaderConfig(ContrastiveHeaderConfig):
    # Whether to use a NN memory bank to store positive examples (used by NNCLR)
    enable_nn_bank: bool = False
    nn_memory_bank_size: int = 65536
    prediction_type: str = "MLP"
    proj_hidden_dim: int = 2048
    proj_output_dim: int = 256
    pred_hidden_dim: int = 2048
    pred_output_dim: int = 128
    direct_pred_num_dim: int = 64


@dataclass
class ProjectionHeaderConfig(ContrastiveHeaderConfig):
    projection_type: str = "MLP"
    hidden_dim: int = 2048
    output_dim: int = 128
    direct_proj_num_dim: int = 64


@dataclass
class TransportOperatorConfig(ContrastiveHeaderConfig):
    dictionary_size: int = 128
    lambda_prior: float = 1e-1
    transop_lr: float = 1e-2
    transop_weight_decay: float = 1e-6
    detach_feature: bool = False
    # Scale point pairs before inferring coefficients and applying transop
    latent_scale: float = 1.0

    # Settings for projection
    projection_type: str = "None"
    projection_hidden_dim: int = 2048
    projection_out_dim: int = 128
    projection_network_lr: float = 3e-4
    projection_network_weight_decay: float = 1e-6

    # Option to use NN to find point pairs
    enable_nn_point_pair: bool = False
    nn_memory_bank_size: int = 65536

    # Config for variational network
    enable_variational_inference: bool = True
    variational_distribution: str = "Laplacian"
    variational_prior_type: str = "Fixed"
    variational_encoder_lr: float = 3e-4
    variational_encoder_weight_decay: float = 1e-6
    iter_variational_samples: int = 10
    total_variational_samples: int = 100
    use_ntxloss_sampling: bool = False
    use_warmpup: bool = False
    variational_scale_prior: float = 0.1
    variational_encoder_type: str = "mlp"
    variational_feature_dim: int = 256
    variational_use_features: bool = True
    # Config for exact inference
    fista_num_iterations: int = 800
