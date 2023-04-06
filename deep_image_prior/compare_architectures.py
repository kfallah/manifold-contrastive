from make_individual_dip import load_resnet_backbone
from utils import load_transop_model
import omegaconf
import torch
print("SimCLR Architecture")
last_layer = "layer4"
default_device = "cuda"
config_path = "../results/simclr_cifar10/cfg_simclr_cifar10.yaml"
checkpoint_path = "../results/simclr_cifar10/simclr_cifar10.pt"
# Load the config
config = omegaconf.OmegaConf.load(config_path)
config.model_cfg.backbone_cfg.load_backbone = None
# Load the tansport operator model
model = load_transop_model(
    config.model_cfg,
    dataset_name="CIFAR10",
    device=default_device,
    checkpoint_path=checkpoint_path
)
backbone = model.backbone.backbone_network
backbone.avgpool = torch.nn.Identity()
backbone.fc = torch.nn.Identity()
print(backbone)

print("Resnet Architecture")
backbone = load_resnet_backbone(
    last_layer=last_layer,
    resnet_type="resnet18"
)
print(backbone)