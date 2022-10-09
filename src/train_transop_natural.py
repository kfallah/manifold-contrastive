import logging
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.transforms as T
from torchvision.models import resnet18

from model.manifold.l1_inference import infer_coefficients
from model.manifold.transop import TransOp_expm

# Config #
dict_size = 50
run_number = 1
zeta = 0.1
gamma = 1e-6
psi_lr = 1e-3
save_freq = 40
default_device = torch.device("cuda:0")

logging.basicConfig(
    filename=f"transop_dummy{run_number}.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)

# Data loader #
cifar10 = torchvision.datasets.CIFAR10(
    "datasets",
    train=True,
    transform=T.Compose(
        [
            T.Resize(256),
            T.RandomCrop(224, 4),
            T.RandomHorizontalFlip(),
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
    train_dataset, batch_size=96, shuffle=True, num_workers=20
)


# Model Initialization #
backbone = resnet18(pretrained=True).to(default_device)
backbone.fc = torch.nn.Identity()
transop = TransOp_expm(dict_size, 512, var=0.05).to(default_device)

opt = torch.optim.Adam(
    transop.parameters(), lr=psi_lr, weight_decay=gamma, betas=(0.5, 0.99)
)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9985)

psi_list = []
loss_list = []
# Training #
for epoch in range(100):
    for idx, batch in enumerate(train_dataloader):
        pre_time = time.time()
        x0, x1, y = batch
        x0, x1 = x0.to(default_device), x1.to(default_device)

        with torch.no_grad():
            z0, z1 = backbone(x0) / 24.0, backbone(x1) / 24.0
        _, c = infer_coefficients(
            z0,
            z1,
            transop.get_psi(),
            zeta,
            800,
            device=default_device,
            num_trials=1,
            c_init=0.01,
        )
        z1_hat = transop(z0.unsqueeze(-1), c).squeeze(-1)

        loss = F.mse_loss(z1_hat, z1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        loss_list.append(loss.item())

        if idx % save_freq == 0:
            psi_list.append(transop.psi.data.cpu())
            pair_dist = F.mse_loss(z0, z1).item()
            psi_norm = (transop.psi**2).mean(dim=0).sum().item()
            coeff_np = c.detach().cpu().numpy()
            coeff_nz = np.count_nonzero(coeff_np, axis=0)
            nz_tot = np.count_nonzero(coeff_nz)
            count_nz = np.zeros(dict_size + 1, dtype=int)
            total_nz = np.count_nonzero(coeff_np, axis=1)
            avg_feat_norm = np.linalg.norm(z0.detach().cpu().numpy(), axis=-1).mean()

            for z in range(len(total_nz)):
                count_nz[total_nz[z]] += 1

            logging.info(
                f"[Iter {idx + len(train_dataloader)*(epoch)}] [Distance bw Pairs: {pair_dist:.6E}]"
                + f" [TransOp Loss {np.mean(loss_list):.6E}]"
                + f" [Time {time.time() - pre_time:.2f} sec]"
            )
            logging.info("Non-zero elements per bin: {}".format(count_nz))
            logging.info("Non-zero by coefficient #: {}".format(nz_tot))
            logging.info(
                f"Total # operators used: {nz_tot}/{len(transop.psi)}"
                + f", avg # operators used: {total_nz.mean()}/{len(transop.psi)}"
                + f", avg feat norm: {avg_feat_norm:.2E}"
                + f", avg coeff mag: {np.abs(coeff_np[np.abs(coeff_np) > 0]).mean():.3E}"
            )
            logging.info("Avg operator F-norms: {:.3E}".format(psi_norm))

            torch.save(
                {
                    "psi": psi_list,
                    "z0": z0.detach().cpu(),
                    "z1": z1.detach().cpu(),
                },
                f"results/transop_dummy{run_number}.pt",
            )

            loss_list = []
