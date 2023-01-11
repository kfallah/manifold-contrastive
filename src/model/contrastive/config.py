"""
Config for all the different contrastive headers.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""
from dataclasses import dataclass


@dataclass
class ProjectionPredictionHeaderConfig:
    # Whether to use a NN memory bank to store positive examples (used by NNCLR)
    header_name: str = "NNCLR"
    enable_nn_bank: bool = False
    nn_memory_bank_size: int = 65536
    prediction_type: str = "MLP"
    proj_hidden_dim: int = 2048
    pred_hidden_dim: int = 2048
    pred_output_dim: int = 128
    direct_pred_num_dim: int = 64


@dataclass
class ProjectionHeaderConfig:
    header_name: str = "SimCLR"
    projection_type: str = "MLP"
    hidden_dim: int = 2048
    output_dim: int = 128
    direct_proj_num_dim: int = 64


@dataclass
class VariationalEncoderConfig:
    variational_encoder_lr: float = 3e-4
    variational_encoder_weight_decay: float = 1e-6

    distribution: str = "Laplacian"
    prior_type: str = "Fixed"
    scale_prior: float = 0.02
    encoder_type: str = "MLP"
    feature_dim: int = 256
    encode_features: bool = True
    share_encoder: bool = True
    encode_position: bool = False

    use_warmpup: bool = False
    normalize_coefficients: bool = False
    normalize_mag: float = 1.0
    per_iter_samples: int = 10
    total_samples: int = 100


@dataclass
class TransportOperatorConfig:
    start_iter: int = 50000
    fine_tune_iter: int = 100000
    dictionary_size: int = 200
    lambda_prior: float = 0.04
    transop_lr: float = 1e-3
    transop_weight_decay: float = 1e-5

    stable_operator_initialization: bool = True
    real_range_initialization: float = 1e-4
    image_range_initialization: float = 0.3

    detach_feature: bool = False
    batch_size: int = 256
    # Scale point pairs before inferring coefficients and applying transop
    latent_scale: float = 1.0

    # Option to splice input to create BDM constraint on transop
    enable_splicing: bool = False
    # Only use the top block in the BDM
    enable_direct: bool = False
    splice_dim: int = 128

    # Option for alternating minimization between
    enable_alternating_min: bool = False
    # Number of steps to alternate between updating the backbone and the transop/encoder
    alternating_min_step: int = 200

    # Option to use NN to find point pairs
    enable_nn_point_pair: bool = False
    nn_memory_bank_size: int = 65536

    # Config for variational network
    enable_variational_inference: bool = True
    vi_cfg: VariationalEncoderConfig = VariationalEncoderConfig()
    enable_vi_refinement: bool = False
    vi_refinement_lambda: float = 0.1
    # Config for exact inference
    fista_num_iterations: int = 800


@dataclass
class ContrastiveHeaderConfig:
    enable_projection_header: bool = False
    projection_header_cfg: ProjectionHeaderConfig = ProjectionHeaderConfig()

    enable_transop_header: bool = False
    enable_transop_augmentation: bool = False
    transop_header_cfg: TransportOperatorConfig = TransportOperatorConfig()

    enable_proj_pred_header: bool = False
    proj_pred_header_cfg: ProjectionPredictionHeaderConfig = ProjectionPredictionHeaderConfig()
