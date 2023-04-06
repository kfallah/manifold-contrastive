"""
    Tools for loading various models
"""
import os
import torch
from lightly.models.utils import deactivate_requires_grad
import omegaconf
import torch.nn as nn

import sys
sys.path.append(os.path.dirname(os.getcwd()) + "/src/")

from model.model import Model

default_device = torch.device('cuda:0')

def load_transop_model(
    model_cfg, 
    dataset_name="CIFAR10", 
    device=default_device,
    checkpoint_path=None,
    device_index=[0]
):
    print(os.path.exists(checkpoint_path))
    # Load model
    model = Model.initialize_model(model_cfg, dataset_name=dataset_name, devices=device_index)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['model_state'], strict=False)

    return model

def load_backbone_model(
    config_path, 
    checkpoint_path, 
    final_layer="layer3",
    upscale_factor=8
):
    """
        Loads a trained backbone model
    """
    # Load the config
    config = omegaconf.OmegaConf.load(config_path)
    config.model_cfg.backbone_cfg.load_backbone = None
    # Load the tansport operator model
    print("Loading the model...")
    model = load_transop_model(
        config.model_cfg,
        dataset_name="CIFAR10",
        device=default_device,
        checkpoint_path=checkpoint_path
    )
    # Unpack the backbone
    backbone = model.backbone.backbone_network
    # Truncate the backbone to a given layer
    if final_layer == "layer3":
        backbone = nn.Sequential(*backbone._modules.values())[:-3]
    elif final_layer == "layer4":
        backbone = nn.Sequential(*backbone._modules.values())[:-2]
    # Add an average pooling layer to downsample the image 
    backbone = nn.Sequential(
        nn.AvgPool2d(upscale_factor, upscale_factor),
        backbone,
    )

    deactivate_requires_grad(backbone)

    return backbone

def load_pretrained_resnet(
    last_layer="fc", 
    resnet_type="resnet50", 
    weights_path=None
):
    """
        Loads a pretrained resnet model from 
        the model hub.
    """
    if resnet_type == "resnet50":
        backbone = torchvision.models.resnet50(pretrained=True).to(default_device)
    else:
        backbone = torchvision.models.resnet18(pretrained=True).to(default_device)

    if not weights_path is None:
        backbone.load_state_dict(torch.load(weights_path))

    backbone.eval()

    if last_layer == "layer4":
        backbone.avgpool = torch.nn.Identity()
    elif last_layer == "layer3":
        backbone.avgpool = torch.nn.Identity()
        backbone.layer4 = torch.nn.Identity()

    backbone.fc = torch.nn.Identity()
    deactivate_requires_grad(backbone)

    return backbone