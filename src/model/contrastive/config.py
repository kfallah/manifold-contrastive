"""
Config for all the different contrastive headers.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""
from dataclasses import MISSING, dataclass

from hydra.core.config_store import ConfigStore


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="exp_cfg/model_cfg/header_cfg", name="simclr", node=SimCLRHeaderConfig)


@dataclass
class ContrastiveHeaderConfig:
    header_name: str = "DefaultHeader"


@dataclass
class SimCLRHeaderConfig(ContrastiveHeaderConfig):
    hidden_dim: int = 2048
    output_dim: int = 128
