"""
Config for all the different contrastive headers.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""
from dataclasses import dataclass
from typing import Tuple

from hydra.core.config_store import ConfigStore


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="exp_cfg/model_cfg/header_cfg", name="simclr", node=SimCLRHeaderConfig)


@dataclass
class ContrastiveHeaderConfig:
    header_name: str = "DefaultHeader"


@dataclass
class SimCLRHeaderConfig(ContrastiveHeaderConfig):
    projection_type: str = "MLP"
    hidden_dim: int = 2048
    output_dim: int = 128
    direct_proj_num_dim: int = 64
