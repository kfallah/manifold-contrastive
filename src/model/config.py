"""
Class that contains all config DataClasses for models.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""

from dataclasses import dataclass

from model.contrastive.config import ContrastiveHeaderConfig


@dataclass
class BackboneConfig:
    # PyTorch hub model name https://pytorch.org/hub/research-models
    hub_model_name: str = "resnet18"
    pretrained: bool = False


@dataclass
class ModelConfig:
    backbone_cfg: BackboneConfig = BackboneConfig()
    header_cfg: ContrastiveHeaderConfig = ContrastiveHeaderConfig()
