"""
Contains all typing information relevant to models.

@Filename    type.py
@Author      Kion
@Created     09/07/22
"""

from typing import Callable, Dict, NamedTuple, Optional

import torch


class DistributionData(NamedTuple):
    # Dictionary containing params for encoder
    encoder_params: Dict[str, torch.Tensor]
    # Dictionary containing params for prior (can be fixed if not learned)
    prior_params: Dict[str, torch.Tensor]
    # Samples from variational distribution
    samples: Optional[torch.Tensor] = None


class ModelOutput(NamedTuple):
    # Augmentation pairs of image. The other view is optional in the case where augmentations are not applied to
    # images.
    # Dimensions [B x H x W x C]
    x_0: torch.Tensor
    x_1: Optional[torch.Tensor] = None
    # List of indices for each entry in the batch
    # Dimensions [B]
    x_idx: Optional[torch.Tensor] = None
    # List of features from backbone encoder
    # Dimensions [B x D]
    feature_0: Optional[torch.Tensor] = None
    feature_1: Optional[torch.Tensor] = None
    # Projections from the header
    projection_0: Optional[torch.Tensor] = None
    projection_1: Optional[torch.Tensor] = None
    # Predictions from contrastive header
    # Dimensions [B x D']
    prediction_0: Optional[torch.Tensor] = None
    prediction_1: Optional[torch.Tensor] = None
    # Optional distribution data if using a variational model
    distribution_data: Optional[DistributionData] = None


class HeaderInput(NamedTuple):
    # Positive and negative augmentation pair
    # Dimensions [B x H x W x C]
    x_0: torch.Tensor
    x_1: torch.Tensor
    # List of indices for each entry in the batch
    # Dimensions [B]
    x_idx: torch.Tensor
    # List of features from backbone encoder
    # Dimensions [B x D]
    feature_0: torch.Tensor
    feature_1: Optional[torch.Tensor]


class HeaderOutput(NamedTuple):
    # Projections from the header
    projection_0: torch.Tensor
    projection_1: torch.Tensor
    # Predictions from the header
    # Dimensions [B x D']
    prediciton_0: torch.Tensor
    prediction_1: torch.Tensor
    # Optional distribution data if using a variational model
    distribution_data: Optional[DistributionData]
