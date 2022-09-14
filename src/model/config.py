"""
Class that contains all config DataClasses for models.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""

from dataclasses import dataclass

from model.contrastive.config import ContrastiveHeaderConfig


@dataclass
class LossConfig:
    ntxent_loss_active: bool = True
    ntxent_temp: float = 0.07
    kl_loss_active: bool = False
    kl_loss_weight: float = 0.1
    transop_loss_active: bool = False
    transop_loss_weight: float = 1.0
    memory_bank_size: int = 0


@dataclass
class BackboneConfig:
    # PyTorch hub model name https://pytorch.org/hub/research-models
    hub_model_name: str = "resnet18"
    pretrained: bool = False


@dataclass
class ModelConfig:
    backbone_cfg: BackboneConfig = BackboneConfig()
    header_cfg: ContrastiveHeaderConfig = ContrastiveHeaderConfig()
    loss_cfg: LossConfig = LossConfig()
    # Whether to concatenate different views of a batch of images and feed them into the model all at once
    concat_different_views: bool = False
    # Whether to shuffle input batches (to prevent overfitting to the ordering of the positive/negative pairs)
    enable_batch_shuffle: bool = False
    # Whether to use momentum networks
    enable_backbone_momentum: bool = False
    backbone_momentum_update_rate: float = 0.99
    enable_header_momentum: bool = False
    header_momentum_update_rate: float = 0.99
