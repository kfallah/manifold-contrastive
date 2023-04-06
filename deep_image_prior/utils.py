import os
import torch
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