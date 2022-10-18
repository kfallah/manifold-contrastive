import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.transforms as T
import wandb
from matplotlib import pyplot as plt
from torchvision.models import resnet18

from model.autoencoder import ConvDecoder
from model.contrastive.config import TransportOperatorConfig
from model.manifold.l1_inference import infer_coefficients
from model.manifold.transop import TransOp_expm
from model.manifold.vi_encoder import VIEncoder
from train.metric_utils import transop_plots

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run_label", required=True, type=str, help="Label for run")
parser.add_argument("-z", "--zeta", default=0.04, type=float, help="L1 penalty")
parser.add_argument("-g", "--gamma", default=1e-5, type=float, help="WD penalty")
parser.add_argument("-w", "--kl_weight", default=1e-1, type=float, help="KL weight")
parser.add_argument("-pl", "--psi_lr", default=1e-3, type=float, help="Psi LR")
parser.add_argument("-nl", "--net_lr", default=3e-4, type=float, help="VI LR")
parser.add_argument("-s", "--n_samples", default=100, type=int, help="VI samples")
parser.add_argument("-t", "--temp", default=1, type=float, help="Sampling temp")
parser.add_argument("-b1", "--beta1", default=0.9, type=float, help="Beta1")
parser.add_argument("-b2", "--beta2", default=0.999, type=float, help="Beta2")

parser.add_argument("-T", "--final_temp", default=1, type=float, help="Final temp")
parser.add_argument(
    "--image_input", action="store_true", default=False, help="VI use images"
)
parser.add_argument(
    "--disable_wandb", action="store_true", default=False, help="Disable W&B"
)
parser.add_argument(
    "--temp_warmup", action="store_true", default=False, help="Warmup temp"
)
parser.add_argument("--l2_sq_reg", action="store_true", default=False, help="Use L2 Sq")
parser.add_argument(
    "--stable_init", action="store_true", default=False, help="Stable init"
)
args = parser.parse_args()

# Config #
dict_size = 200
run_label = args.run_label
zeta = args.zeta
gamma = args.gamma
kl_weight = args.kl_weight
psi_lr = args.psi_lr
net_lr = args.net_lr
save_freq = 1000
log_freq = 1000
latent_scale = 14.1
use_vi = True
default_device = torch.device("cuda:1")
total_num_samples = args.n_samples
use_features = True
temp = args.temp
temp_warmup = args.temp_warmup
final_temp = args.final_temp
l2_sq_reg = args.l2_sq_reg
beta1 = args.beta1
beta2 = args.beta2
stable_init = args.stable_init

total_epoch = 1000

logging.basicConfig(
    filename=f"transop_vi_mae_{run_label}.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logging.info(args)

wandb.init(
    project="mae-variational-sampling",
    entity="kfallah",
    mode="disabled" if args.disable_wandb else "online",
    settings=wandb.Settings(start_method="thread"),
    config=args,
)
logging.info(f"Initializing experiment {wandb.run.name}...")

# Data loader #
cifar10 = torchvision.datasets.CIFAR10(
    "datasets",
    train=True,
    transform=T.Compose(
        [
            T.Resize(48),
            T.RandomCrop(32, 4),
            T.ToTensor(),
            T.Normalize(
                mean=[0.50707516, 0.48654887, 0.44091784],
                std=[0.26733429, 0.25643846, 0.27615047],
            ),
        ]
    ),
)


# Dataset that returns images with nearest neighbor
class NaturalTransformationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super(NaturalTransformationDataset, self)
        self.dataset = dataset
        self.nn_graph = torch.arange(len(self.dataset))[:, None]

    def set_nn_graph(self, nn_graph):
        self.nn_graph = nn_graph

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x0, label = self.dataset.__getitem__(idx)
        neighbor = random.randrange(len(self.nn_graph[idx]))
        x1 = self.dataset.__getitem__(int(self.nn_graph[idx, neighbor]))
        return (x0, x1[0], label)


train_dataset = NaturalTransformationDataset(cifar10)
train_dataset.nn_graph = np.load("results/cifar10_resnet18_nn.npy")
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=256, shuffle=True, num_workers=20
)


# Model Initialization #
backbone = resnet18(pretrained=False).to(default_device)
backbone.fc = torch.nn.Linear(512, 128).to(default_device)
decoder = ConvDecoder(128, 3, 32).to(default_device)
transop = TransOp_expm(dict_size, 128, var=1, stable_init=stable_init).to(
    default_device
)

if os.path.exists("results/pretrained/cifar10_resnet18_ae.pth"):
    ae_weights = torch.load(
        "results/pretrained/cifar10_resnet18_ae.pth", map_location=default_device
    )
    backbone.load_state_dict(ae_weights["encoder"])
    decoder.load_state_dict(ae_weights["decoder"])
    backbone.eval()
    decoder.eval()

if use_vi:
    cfg = TransportOperatorConfig(
        dictionary_size=dict_size,
        lambda_prior=zeta,
        variational_use_features=use_features,
        variational_scale_prior=0.02,
        variational_feature_dim=256,
    )
    vi = VIEncoder(cfg, 128, dict_size).to(default_device)
    to_opt = torch.optim.Adam(
        [
            {"params": transop.parameters()},
            {"params": vi.parameters(), "weight_decay": 1e-6, "lr": net_lr},
        ],
        lr=psi_lr,
        betas=(beta1, beta2),
        weight_decay=0.0 if l2_sq_reg else gamma,
    )
else:
    to_opt = torch.optim.AdamW(transop.parameters(), lr=psi_lr, weight_decay=gamma)
to_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    to_opt, total_epoch * len(train_dataloader)
)

psi_list = []
c_list = []
to_loss_list = []
recon_loss_list = []
pair_dist_list = []
kl_loss_list = []

# Training #
for epoch in range(total_epoch):
    for idx, batch in enumerate(train_dataloader):
        pre_time = time.time()
        x0, x1, y = batch
        x0, x1 = x0.to(default_device), x1.to(default_device)

        z0, z1 = backbone(x0) / latent_scale, backbone(x1) / latent_scale
        x0_hat, x1_hat = decoder(latent_scale * z0), decoder(latent_scale * z1)
        recon_loss = 0.5 * (F.mse_loss(x1_hat, x1) + F.mse_loss(x0_hat, x0))

        z0, z1 = z0.detach(), z1.detach()

        if temp_warmup:
            temp = temp * 0.9996
            if temp < final_temp:
                temp = final_temp

        if use_vi:
            if use_features:
                distribution_data = vi(z0, z1)
            else:
                distribution_data = vi(x0, x1)
            if total_num_samples > 1:
                with torch.no_grad():
                    loss_list = []
                    noise_list = []
                    for _ in range(total_num_samples // 10):
                        # Generate a new noise sample
                        u = vi.draw_noise_samples(len(z1), 10, z1.device)
                        c_temp = vi.reparameterize(
                            distribution_data.shift.unsqueeze(1).repeat(1, 10, 1),
                            distribution_data.log_scale.unsqueeze(1).repeat(1, 10, 1),
                            u,
                        )
                        z1_temp = (
                            transop(z0.unsqueeze(-1).detach(), c_temp.detach())
                            .squeeze(dim=-1)
                            .transpose(0, 1)
                        )
                        transop_loss = (
                            F.mse_loss(
                                z1_temp,
                                z1.repeat(
                                    len(z1_temp), *torch.ones(z1.dim(), dtype=int)
                                ).detach(),
                                reduction="none",
                            )
                            .mean(dim=-1)
                            .transpose(0, 1)
                        )
                        noise_list.append(u)
                        loss_list.append(transop_loss)

                    # Pick the best sample
                    noise_list = torch.cat(noise_list, dim=1)
                    loss_list = torch.cat(loss_list, dim=1)
                    # max_elbo = torch.argmin(loss_list, dim=1)
                    iwae_distr = torch.distributions.categorical.Categorical(
                        logits=-1 * loss_list / temp
                    )
                    iwae_samp = iwae_distr.sample()
                    u = noise_list[torch.arange(len(z0)), iwae_samp]
            else:
                u = vi.draw_noise_samples(len(z1), 1, z1.device)[:, 0]
            c = vi.reparameterize(
                distribution_data.shift,
                distribution_data.log_scale,
                u,
            )

            scale = torch.exp(distribution_data.log_scale)
            logscale_prior = torch.log(distribution_data.scale_prior)
            kl_loss = (
                (distribution_data.shift.abs() / distribution_data.scale_prior)
                + logscale_prior
                - distribution_data.log_scale
                - 1
            )
            kl_loss += (scale / distribution_data.scale_prior) * (
                -(distribution_data.shift.abs() / scale)
            ).exp()
            kl_loss = kl_loss.sum(dim=-1).mean()
        else:
            _, c = infer_coefficients(
                z0.detach(),
                z1.detach(),
                transop.get_psi(),
                zeta,
                800,
                device=default_device,
                num_trials=1,
                c_init=0.02,
            )
        z1_hat = transop(z0.unsqueeze(-1), c).squeeze(-1)

        to_loss = F.mse_loss(z1_hat, z1)
        to_opt.zero_grad()
        (to_loss + kl_weight * kl_loss).backward()
        psi_grad = transop.psi.grad.clone()
        to_opt.step()
        to_scheduler.step()

        if l2_sq_reg:
            with torch.no_grad():
                transop.psi.data -= (
                    (transop.psi.data**2) * gamma * to_opt.param_groups[0]["lr"]
                )

        to_loss_list.append(to_loss.item())
        recon_loss_list.append(recon_loss.item())
        kl_loss_list.append(kl_loss.item())
        pair_dist_list.append(F.mse_loss(z0, z1).item())
        c_list.append(c.detach().cpu())

        if (idx + (epoch * len(train_dataloader))) % log_freq == 0 and idx > 0:
            pair_dist = F.mse_loss(z0, z1).item()
            logging.info(
                f"[Epoch {epoch} Iter {idx + len(train_dataloader)*(epoch)}] [Distance bw Pairs: {np.array(pair_dist_list)[-log_freq:].mean():.6E}]"
                + f" [TransOp Loss {np.array(to_loss_list)[-log_freq:].mean():.6E}]  [Recon Loss {np.array(recon_loss_list)[-log_freq:].mean():.6E}]"
                + f" [KL Loss {np.array(kl_loss_list)[-log_freq:].mean():.6E}]"
                + f" [Time {time.time() - pre_time:.2f} sec]"
            )
            psi_norm = (transop.psi.detach() ** 2).sum(dim=(-1, -2))
            psi_order = torch.flip(torch.argsort(psi_norm), dims=(0,))
            coeff_np = torch.cat(c_list[-log_freq:]).detach().numpy()
            coeff_nz = np.count_nonzero(coeff_np, axis=0)
            nz_tot = np.count_nonzero(coeff_nz)
            count_nz = np.zeros(dict_size + 1, dtype=int)
            total_nz = np.count_nonzero(coeff_np, axis=1)
            avg_feat_norm = np.linalg.norm(z0.detach().cpu().numpy(), axis=-1).mean()
            for z in range(len(total_nz)):
                count_nz[total_nz[z]] += 1
            logging.info("Non-zero elements per bin: {}".format(count_nz))
            logging.info(f"Top ten psi F-norm: {psi_norm[psi_order[:10]]}")
            logging.info(
                f"Top ten psi grad: {(psi_grad.detach() ** 2).sum(dim=(-1,-2))[psi_order[:10]]}"
            )
            logging.info("Non-zero by coefficient #: {}".format(nz_tot))
            logging.info(
                f"Total # operators used: {nz_tot}/{dict_size}"
                + f", avg # operators used: {total_nz.mean()}/{dict_size}"
                + f", avg feat norm: {avg_feat_norm:.2E}"
                + f", avg coeff mag: {np.abs(coeff_np[np.abs(coeff_np) > 0]).mean():.3E}"
            )
            logging.info(
                f"Avg operator F-norms: {psi_norm.mean().item():.3E}"
                + f", temp: {temp:.3E}"
                + f", avg sample entropy: {iwae_distr.entropy().mean():.3E}"
            )

            if not args.disable_wandb:
                wandb.log(
                    {
                        "pairwise_dist": np.array(pair_dist_list)[-log_freq:].mean(),
                        "transop_loss": np.array(to_loss_list)[-log_freq:].mean(),
                        "recon_loss": np.array(recon_loss_list)[-log_freq:].mean(),
                        "kl_loss": np.array(kl_loss_list)[-log_freq:].mean(),
                        "iter_time": time.time() - pre_time,
                        "avg_coeff_mag": np.abs(coeff_np[np.abs(coeff_np) > 0]).mean(),
                        "avg_feat_norm": avg_feat_norm,
                        "avg_transop_fnorm": psi_norm.mean().item(),
                        "avg_num_transop_used": total_nz.mean(),
                        "avg_entropy": iwae_distr.entropy().mean(),
                        "temp": temp,
                    },
                    step=(idx + (epoch * len(train_dataloader))),
                )

                fig_dict = transop_plots(
                    coeff_np,
                    transop.psi.detach().cpu(),
                    z0.detach().cpu().numpy()[0],
                )
                for fig_name in fig_dict.keys():
                    wandb.log(
                        {fig_name: wandb.Image(fig_dict[fig_name])},
                        step=(idx + (epoch * len(train_dataloader))),
                    )
                    plt.close(fig_dict[fig_name])

        if (idx + (epoch * len(train_dataloader))) % save_freq == 0:
            psi_list.append(transop.psi.data.cpu())
            torch.save(
                {
                    "psi": psi_list,
                    "encoder": backbone.state_dict(),
                    "decoder": decoder.state_dict(),
                    "vi": vi.state_dict() if use_vi else None,
                    "z0": z0.detach().cpu(),
                    "z1": z1.detach().cpu(),
                    "to_loss": to_loss_list,
                    "recon_loss": recon_loss_list,
                },
                f"results/transop_vi_mae_{run_label}.pt",
            )
