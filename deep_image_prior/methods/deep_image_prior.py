import wandb
from tqdm import tqdm
import torch

import sys
sys.path.append("../../src/")

from model.public.dip_ae import dip_ae, get_noise

default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def make_dip_ae(z_use):
    """
        Makes the UNet autoencoder architecture for DIP. 
        NOTE: This architecture is for the 256x256 images.
    """ 
    net = dip_ae(
        32, 
        3, 
        num_channels_down=[16, 32, 64, 128, 128, 128],
        num_channels_up=[16, 32, 64, 128, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4, 4],
        filter_size_down=[7, 7, 5, 5, 3, 3], 
        filter_size_up=[7, 7, 5, 5, 3, 3],
        upsample_mode='nearest', 
        downsample_mode='avg',
        need_sigmoid=False, 
        pad='zero', 
        act_fun='LeakyReLU'
    ).type(z_use.type()).to(default_device)

    return net

def compute_dip_image(
    input_z, 
    input_x,
    backbone,
    mse_lambda=1.0,
    learning_rate=1e-3,
    fixed_noise=False,
    return_network=False,
):
    """
        Computes the deep image prior for a given 
    """
    print("Computing Deep Image Prior")
    z_use = torch.tensor(input_z).to(default_device)
    # Make the DIP network
    net = make_dip_ae(z_use)
    # Make the noise
    net_input = get_noise(32, 256).type(z_use.type()).detach().to(default_device)
    print(f"Net input shape: {net_input.shape}")
    if fixed_noise:
        opt = torch.optim.Adam(
            list(net.parameters()),
            lr=learning_rate
        ) 
    else:
        opt = torch.optim.Adam(
            list(net.parameters()) + list(net_input),
            lr=learning_rate
        )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)

    for j in tqdm(range(3000)):
        x_hat = net(net_input) # [:, :, :224, :224]
        z_hat = backbone(x_hat)

        loss = mse_lambda * torch.nn.functional.mse_loss(z_hat[0], z_use)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        if j % 10 == 0:
            if wandb.run is not None:
                wandb.log({
                    "loss": loss.item(),
                    "x_loss": torch.norm(
                        x_hat.detach().cpu() - input_x.detach().cpu(),
                        p=1
                    ).item(),
                })

    x_hat[x_hat > 1] = 1
    x_hat[x_hat < 0] = 0

    return x_hat[0]

