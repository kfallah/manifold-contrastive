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
    hidden_dim: int = 1024
    output_dim: int = 128
    direct_proj_num_dim: int = 64
    enable_final_batchnorm: bool = True


@dataclass
class VariationalEncoderConfig:
    variational_encoder_lr: float = 0.001
    variational_encoder_weight_decay: float = 1e-5

    distribution: str = "Laplacian"
    scale_prior: float = 0.02
    shift_prior: float = 0.0
    feature_dim: int = 512
    # Whether the encoder should use x0 and x1 or just x0
    encode_point_pair: bool = True
    # warmup the threshold parameter -- start with dense coefficients and anneal to sparse
    enable_thresh_warmup: bool = False
    # Use attention layer to refine features before drawing samples
    enable_enc_attn: bool = False

    enable_max_sampling: bool = True
    max_sample_l1_penalty: float = 1.0e-2
    max_sample_start_iter: int = 50000
    total_num_samples: int = 20
    samples_per_iter: int = 20

    # Learn the prior, as in a CVAE
    # Required for transop augmentations
    enable_learned_prior: bool = False
    enable_prior_shift: bool = False
    enable_prior_block_encoding: bool = False
    # Use a deterministic prior for shift
    enable_det_prior: bool = False

    # Use deterministic encoder
    enable_det_enc: bool = False

    # whether to use FISTA for the encoder instead of a DNN.
    enable_fista_enc: bool = False
    fista_lambda: float = 0.1
    fista_num_iters: int = 40
    # Enable variance regularization to prevent L1 collapse with FISTA
    enable_fista_var_reg: bool = False
    fista_var_reg_scale: float = 0.01
    fista_var_reg_weight: float = 0.1


@dataclass
class TransportOperatorConfig:
    # What iteration to start training the transport operator/VI
    start_iter: int = 50000
    # What iteration to start co-adapting the backbone weights with the operator
    fine_tune_iter: int = 100000
    # Default dictionary size
    dictionary_size: int = 100
    # Default threshold degree by variational encoder
    lambda_prior: float = 0.02
    # Learning rate for operators
    transop_lr: float = 0.1
    transop_weight_decay: float = 1e-6
    # Use the negative coefficients from z0 to z1
    # to transport from z1 to z0
    symmmetric_transport: bool = False

    stable_operator_initialization: bool = True
    real_range_initialization: float = 0.0001
    image_range_initialization: float = 6.0

    batch_size: int = 128

    # Option to splice input to create BDM constraint on transop
    enable_block_diagonal: bool = True
    block_dim: int = 64
    # Use a separate dictionary for each block of features
    enable_dict_per_block: bool = False

    # Option for alternating minimization between
    enable_alternating_min: bool = False
    # Number of steps to alternate between updating the backbone and the transop/encoder
    alternating_min_step: int = 20

    # Config for variational network
    enable_variational_inference: bool = True
    vi_cfg: VariationalEncoderConfig = VariationalEncoderConfig()
    # Config for exact inference
    fista_num_iterations: int = 20


@dataclass
class ContrastiveHeaderConfig:
    enable_projection_header: bool = False
    projection_header_cfg: ProjectionHeaderConfig = ProjectionHeaderConfig()

    enable_transop_header: bool = False
    enable_transop_augmentation: bool = False
    transop_header_cfg: TransportOperatorConfig = TransportOperatorConfig()

    enable_proj_pred_header: bool = False
    proj_pred_header_cfg: ProjectionPredictionHeaderConfig = ProjectionPredictionHeaderConfig()
