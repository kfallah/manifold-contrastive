"""
Contains all typing information relevant to models.

@Filename    type.py
@Author      Kion
@Created     09/07/22
"""

from typing import Dict, NamedTuple, Optional

import torch


class DistributionData(NamedTuple):
    # Dictionary containing params for encoder
    encoder_params: Dict[str, torch.Tensor]
    # Dictionary containing params for prior (can be fixed if not learned)
    prior_params: Dict[str, torch.Tensor]
    # Dictionary containing hyperprior params, useful for learned prior
    hyperprior_params: Dict[str, torch.Tensor]
    # Samples from variational distribution
    samples: Optional[torch.Tensor] = None


class HeaderInput(NamedTuple):
    # Current iteration
    curr_iter: int
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
    # Outputs from the header
    header_dict: Dict[str, torch.Tensor]
    # Optional distribution data if using a variational model
    distribution_data: Optional[DistributionData] = None


class ModelOutput(NamedTuple):
    header_input: HeaderInput
    header_output: Optional[HeaderOutput] = None
