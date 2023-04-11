"""
Class that contains all config DataClasses for models.

@Filename    config.py
@Author      Kion
@Created     08/31/22
"""

from dataclasses import dataclass
from typing import Optional

from model.contrastive.config import ContrastiveHeaderConfig


@dataclass
class LossConfig:
    ntxent_loss_active: bool = True
    ntxent_loss_weight: float = 1.0
    ntxent_temp: float = 0.07
    ntxent_normalize: bool = True
    ntxent_logit: str = "cos"
    ntxent_detach_off_logit: bool = False
    ntxent_symmetric: bool = False

    ntxent_lie_loss_active: bool = False
    ntxent_lie_loss_weight: float = 1.0
    ntxent_lie_loss_start_iter: int = 0
    ntxent_lie_temp: float = 0.07

    vicreg_loss_active: bool = False
    vicreg_loss_weight: float = 1.0
    vicreg_inv_weight: float = 25.0
    vicreg_var_weight: float = 25.0
    vicreg_cov_weight: float = 1.0
    vicreg_block_dim: int = -1

    kl_loss_active: bool = False
    kl_loss_weight: float = 0.1
    kl_detach_shift: bool = False
    kl_weight_warmup: str = "None"

    real_eig_reg_active: bool = False
    real_eig_reg_weight: float = 1e-5

    transop_loss_active: bool = False
    transop_loss_weight: float = 1.0
    transop_loss_fn: str = "mse"
    transop_loss_ce_temp: float = 1e-1
    transop_symmetric: bool = False
    memory_bank_size: int = 0

    c_refine_loss_active: bool = False
    c_refine_loss_weight: float = 10.0

    c_l2_active: bool = False
    c_l2_weight: float = 1.0e-3

    enable_shift_l2: bool = False
    enable_prior_shift_l2: bool = False
    shift_l2_weight: float = 1.0e-3

    det_prior_loss_active: bool = False
    det_prior_l2_weight: float = 1.0e-3


@dataclass
class BackboneConfig:
    # PyTorch hub model name https://pytorch.org/hub/research-models
    hub_model_name: str = "resnet18"
    pretrained: bool = False
    load_backbone: Optional[str] = None
    freeze_backbone: bool = False


@dataclass
class ModelConfig:
    backbone_cfg: BackboneConfig = BackboneConfig()
    header_cfg: ContrastiveHeaderConfig = ContrastiveHeaderConfig()
    loss_cfg: LossConfig = LossConfig()
