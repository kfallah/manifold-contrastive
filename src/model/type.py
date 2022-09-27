"""
Contains all typing information relevant to models.

@Filename    type.py
@Author      Kion
@Created     09/07/22
"""

from typing import NamedTuple, Optional

import torch


class DistributionData(NamedTuple):
    # Samples from variational distribution
    samples: torch.Tensor
    # Distribution parameter associated with scale from encoder
    log_scale: torch.Tensor
    # Distribution parameter associated with shift from encoder
    shift: torch.Tensor
    # Prior scale
    scale_prior: torch.Tensor
    # Prior shift
    shift_prior: torch.Tensor


class ModelOutput(NamedTuple):
    # Augmentation pairs of image. The other view is optional in the case where augmentations are not applied to
    # images.
    # Dimensions [B x H x W x C]
    x_0: torch.Tensor
    x_1: Optional[torch.Tensor]
    # List of indices for each entry in the batch
    # Dimensions [B]
    x_idx: torch.Tensor
    # List of features from backbone encoder
    # Dimensions [B x D]
    feature_0: torch.Tensor
    feature_1: Optional[torch.Tensor]
    # List of predictions from contrastive header
    # Dimensions [B x D']
    prediction_0: torch.Tensor
    prediction_1: Optional[torch.Tensor]
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
    # Features in the case where additional processing is performed.
    feature_0: torch.Tensor
    feature_1: torch.Tensor
    # Prediction for the positive pair from the header
    # Dimensions [B x D']
    prediciton_0: torch.Tensor
    # Prediciton from the negative pair from the header
    # Dimensions [B x D']
    prediction_1: torch.Tensor
    # Optional distribution data if using a variational model
    distribution_data: Optional[DistributionData]
