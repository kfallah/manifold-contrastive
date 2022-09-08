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
    cs.store(group="exp_cfg/model_cfg/header_cfg", name="simclr", node=ProjectionHeaderConfig)
    cs.store(group="exp_cfg/model_cfg/header_cfg", name="transop", node=TransportOperatorConfig)


@dataclass
class ContrastiveHeaderConfig:
    header_name: str = "DefaultHeader"


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
    transop_weight_decay: float = 1e-6
    num_coefficient_samples: int = 10

    # Config for variational network
    enable_variational_inference: bool = True
    use_warmpup: bool = False
    variational_scale_prior: float = 0.1
    variational_encoder_type: str = "mlp"
    variational_feature_dim: int = 256
    # Config for exact inference
    fista_num_iterations: int = 800
