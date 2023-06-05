# %%
import argparse
import copy
import os
import random
import sys
import time
import warnings

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.models.utils import deactivate_requires_grad, update_momentum

import wandb
from dataloader.ssl_dataloader import get_dataset
from dataloader.transform import (MultiSample, get_ssl_augmentation,
                                  get_weak_augmentation)
from model.config import ModelConfig
from model.manifold.reparameterize import compute_kl
from model.model import Model
from model.public.linear_warmup_cos_anneal import LinearWarmupCosineAnnealingLR
from model.public.ntx_ent_loss import lie_nt_xent_loss
from model.type import HeaderInput
from train.metric_utils import transop_plots

#sys.path.append(os.path.dirname(os.getcwd()) + "/src/")



warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="Finetune SSL model")

parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use")
parser.add_argument("--labels_per_class", type=int, default=20, help="Labels per class") 
parser.add_argument("--seed", type=int, default=1337, help="Number of trials to run") 
parser.add_argument("--n_trials", type=int, default=5, help="Number of trials to run") 
parser.add_argument("--n_iters", type=int, default=5000, help="Number of iters to run for")

parser.add_argument("--lr", type=float, default=5.0e-4, help="Learning rate")
parser.add_argument("--wd", type=float, default=5.0e-4, help="Weight Decay")
parser.add_argument("--bsl", type=int, default=32, help="Labeled batchsize")
parser.add_argument("--bsu", type=int, default=480, help="Unlabeled batchsize")


parser.add_argument("--con_weight", type=float, default=1.0, help="Consistency weight")

parser.add_argument("--enable_ctt", default=True, action='store_true', help="Whether to use contrastive loss")
parser.add_argument("--ctt_weight", type=float, default=0.1, help="Contrastive weight")

parser.add_argument("--enable_transop", default=True, action='store_true', help="Whether to use transop model")
parser.add_argument("--transop_aug", default=True, action='store_true', help="Whether to apply transop prior augs")
parser.add_argument("--detach_aug", default=True, action='store_true', help="Whether to detach augmentations from prior and transop")
parser.add_argument("--disable_thresh", default=True, action='store_true', help="Whether to disable thresholding and max ELBO sampling")
parser.add_argument("--transop_weight", type=float, default=1.0, help="Transop loss weight")
parser.add_argument("--transop_lr", type=float, default=5.0e-4, help="Transop LR")
parser.add_argument("--transop_wd", type=float, default=1.0e-3, help="Transop weight decay")
parser.add_argument("--kl_weight", type=float, default=1.0e-3, help="KL Div weight")

args = parser.parse_args()

device_idx = [2,3]

if args.dataset == 'stl10':
    # ManifoldCLR
    ckpt_path = "/home/kfallah/manifold-contrastive/results/pretrained/stl10_vi-thresh_proj/model_weight.pt"
    cfg_path = "/home/kfallah/manifold-contrastive/results/pretrained/stl10_vi-thresh_proj/config.yaml"
else:
    ## CIFAR10 ##
    ckpt_path = "/home/kfallah/manifold-contrastive/results/cifar10_vi-thresh0.01_dict32_kl1e-3/checkpoints/checkpoint_epoch999.pt"
    cfg_path = "/home/kfallah/manifold-contrastive/results/cifar10_vi-thresh0.01_dict32_kl1e-3/.hydra/config.yaml"
args.ckpt_path = ckpt_path

# %% [markdown]
# ## Boilerplate needed for notebooks

# %%
# Set the default device
default_device = torch.device("cuda:2")
# Load config
cfg = omegaconf.OmegaConf.load(cfg_path)
cfg.model_cfg.backbone_cfg.load_backbone = None

# CUSTOM CONFIG FOR THIS RUN #

# Load model
default_model_cfg = ModelConfig()
model = Model.initialize_model(cfg.model_cfg, cfg.dataloader_cfg.dataset_cfg.dataset_name, [device_idx[0]])
state_dict = torch.load(ckpt_path, map_location=default_device)
model.load_state_dict(state_dict['model_state'], strict=True)
# Manually override directory for dataloaders
cfg.dataloader_cfg.dataset_cfg.dataset_dir = "/home/kfallah/manifold-contrastive/datasets"
# Load dataloaders
dataset = get_dataset(cfg.dataloader_cfg)
test_dataloader = dataset.val_dataloader

# Load transport operators
backbone = model.backbone.backbone_network
proj = model.contrastive_header.projection_header.projector
transop, coeff_enc = (None, None)
if model.contrastive_header.transop_header is not None:
    transop = model.contrastive_header.transop_header.transop
    coeff_enc = model.contrastive_header.transop_header.coefficient_encoder

# %%
class SSLDataLoader(object):
    def __init__(self, labeled_dset, unlabeled_dset, bsl, bsu, num_workers):
        sampler_lab = InfBatchSampler(labeled_dset, bsl)
        sampler_unlab = InfBatchSampler(unlabeled_dset, bsu)

        wl = max(int(num_workers * bsl / (bsl + bsu)), min(1, num_workers))
        wu = num_workers - wl
        self.labeled_dset = torch.utils.data.DataLoader(labeled_dset, batch_sampler=sampler_lab, num_workers=wl)
        self.unlabeled_dset = torch.utils.data.DataLoader(unlabeled_dset, batch_sampler=sampler_unlab, num_workers=wu)

        self.labeled_iter = iter(self.labeled_dset)
        self.unlabeled_iter = iter(self.unlabeled_dset)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data_l = next(self.labeled_iter)
        except StopIteration:
            self.labeled_iter = iter(self.labeled_dset)
            data_l = next(self.labeled_iter)

        try:
            data_u = next(self.unlabeled_iter)
        except StopIteration:
            self.unlabeled_iter = iter(self.unlabeled_dset)
            data_u = next(self.unlabeled_iter)

        return data_l, data_u


class InfBatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size):
        super(InfBatchSampler, self).__init__(data_source)

        self.N = len(data_source)
        self.batch_size = batch_size if batch_size < self.N else self.N
        self.L = self.N // self.batch_size

    def __iter__(self):
        while True:
            idx = np.random.permutation(self.N)
            for i in range(self.L):
                yield idx[i * self.batch_size : (i + 1) * self.batch_size]

    def __len__(self):
        return sys.maxsize

# %%
def get_dataloader(labels_per_class = 10, seed=0, cifar10=False):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    trainl_dataset = copy.deepcopy(dataset.eval_dataloader.dataset)
    trainu_dataset = copy.deepcopy(dataset.train_dataloader.dataset)

    if cifar10:
        data, labels = trainl_dataset.data, np.array(trainl_dataset.targets)
    else:
        data, labels = trainl_dataset.data, trainl_dataset.labels
    data_list, label_list = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        choice = np.random.choice(idx, labels_per_class, replace=False)
        data_list.append(data[choice])
        label_list.append(labels[choice])
    data, labels = np.concatenate(data_list), np.concatenate(label_list)
    trainl_dataset.data = data
    if cifar10:
        trainl_dataset.targets = labels
    else:
        trainl_dataset.labels = labels

    trainl_dataset.transform = MultiSample(
        [get_weak_augmentation(cfg.dataloader_cfg.ssl_aug_cfg, cfg.dataloader_cfg.dataset_cfg.image_size)]
    )
    trainu_dataset.transform = MultiSample(
        [get_weak_augmentation(cfg.dataloader_cfg.ssl_aug_cfg, cfg.dataloader_cfg.dataset_cfg.image_size),
        get_ssl_augmentation(cfg.dataloader_cfg.ssl_aug_cfg, cfg.dataloader_cfg.dataset_cfg.image_size),
        get_ssl_augmentation(cfg.dataloader_cfg.ssl_aug_cfg, cfg.dataloader_cfg.dataset_cfg.image_size)]
    )

    return SSLDataLoader(trainl_dataset, trainu_dataset, args.bsl, args.bsu, 32)

# %%
for exp_iter in range(args.n_trials):
    wandb.init(
        project="ssl_finetune",
        config=args,
    )
    print(f"Exp iter {exp_iter+1}")
    ssl_dataloader = get_dataloader(args.labels_per_class, seed=args.seed, cifar10='cifar10' in ckpt_path)

    clf = nn.Linear(512, 10).to(default_device)
    backbone_train = torch.nn.DataParallel(copy.deepcopy(backbone), device_ids=device_idx)
    backbone_ema = torch.nn.DataParallel(copy.deepcopy(backbone), device_ids=device_idx)
    clf_ema = copy.deepcopy(clf)
    deactivate_requires_grad(backbone_ema)
    deactivate_requires_grad(clf_ema)
    named_params = [param for param in backbone_train.named_parameters() if param[1].requires_grad]
    # Remove weight decay from batch norm layers
    wd_params, non_wd_params = [], []
    for name, param in named_params:
        if "bn" in name:
            non_wd_params.append(param)
        else:
            wd_params.append(param)
    param_groups = [{
        'params': wd_params + list(clf.parameters())
    },
    {
        'params': non_wd_params,
        'weight_decay': 0.0
    },
    ]     
    if args.enable_ctt:
        proj_train = copy.deepcopy(proj)
        param_groups += [{
            'params': proj_train.parameters()
        }]        
    if args.enable_transop:
        transop_train = torch.nn.DataParallel(copy.deepcopy(model.contrastive_header.transop_header), device_ids=device_idx)
        transop_train.module.cfg.batch_size = args.bsu // len(device_idx)
        transop_train.module.coefficient_encoder.vi_cfg.enable_prior_warmup = False
        param_groups +=  [{
            'params': transop_train.module.coefficient_encoder.parameters(),
        },
        {
            'params': transop_train.module.transop.parameters(),
            'lr': args.transop_lr,
            'weight_decay': args.transop_wd
        }] 
        
        list(transop_train.parameters())
        if args.disable_thresh:
            transop_train.module.coefficient_encoder.lambda_prior = 0.0
            transop_train.module.coefficient_encoder.vi_cfg.enable_max_sampling = False
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.wd)
    scheduler =  LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=200, max_epochs=args.n_iters, eta_min=0.0)
    criterion = torch.nn.functional.cross_entropy
    curr_iter = 0

    backbone_train.train()
    for _, data in enumerate(ssl_dataloader):
        ctt_loss_list = []
        transop_loss_list = []
        kl_loss_list = []
        di_list = []
        clf_loss_list = []
        con_loss_list = []
        c_enc_list = []
        c_prior_list = []
        time_list = []
        if curr_iter >= args.n_iters:
            break

        pre_time = time.time()
        (xl, yl), (xu, _) = data
        x0, x1, xweak = xu[0], xu[1], xu[2]
        xl, yl = xl[0].to(default_device), yl.to(default_device)
        x0, x1, xweak = x0.to(default_device), x1.to(default_device),  xweak.to(default_device)

        # Clf loss
        y_pred = clf(backbone_train(xl))
        clf_loss = criterion(y_pred, yl)

        if args.enable_transop or args.enable_ctt:
            z0, z1 = backbone_train(x0), backbone_train(x1)

        # TO loss
        transop_loss, kl_loss = torch.tensor(0.), torch.tensor(0.)
        if args.enable_transop:
            # Transop forward pass
            transop_input = HeaderInput(100000, x0, x1, z0, z1)
            transop_out = transop_train(transop_input)
            z1, z1_hat = transop_out.header_dict['transop_z1'], transop_out.header_dict['transop_z1hat']
            transop_loss = F.mse_loss(z1.detach(), z1_hat, reduction='none')
            dist = F.mse_loss(z0, z1, reduction='none')
            di = (transop_loss.sum(-1) / (dist.sum(-1) + 1e-3)).mean()
            di_list.append(di.item())

            # KL loss
            encoder_params, prior_params = transop_out.distribution_data.encoder_params, transop_out.distribution_data.prior_params
            c_enc = transop_out.distribution_data.samples.detach().cpu()
            c_enc_list.append(c_enc)
            kl_loss = compute_kl("Laplacian", encoder_params, prior_params).mean()

        # Contrastive loss
        ctt_loss = torch.tensor(0.)
        if args.enable_ctt:
            if args.enable_transop:
                c_aug = transop_train.module.coefficient_encoder.prior_sample(z0.detach())
                z0_aug = transop_train.module.transop(z0, c_aug)
            else:
                z0_aug = z0.clone()
            h0, h1 = proj_train(z0_aug), proj_train(z1)
            ctt_loss = lie_nt_xent_loss(
                        F.normalize(h0, dim=-1), 
                        F.normalize(h1, dim=-1), 
                        temperature=0.5,
                    )

        # Consistency Regularization 
        zweak = backbone_train(xweak)
        if not args.transop_aug:
            zstrong = zweak.clone()
        else:
            c = transop_train.module.coefficient_encoder.prior_sample(zweak.detach())
            c_prior_list.append(c.detach().cpu())
            if args.detach_aug:
                zstrong = transop_train.module.transop(zweak, c.detach(), transop_grad=False)
            else:
                zstrong = transop_train.module.transop(zweak, c)

        xu_aug_logits = clf(zstrong)           
        xu_prob = torch.softmax(clf(zweak).detach(), dim=-1)
        wu, yu = torch.max(xu_prob, dim=1)
        wu = (wu > 0.95).detach()
        loss_con = criterion(xu_aug_logits, yu, reduction="none")
        loss_con = torch.sum(loss_con[wu]) / len(wu)

        optimizer.zero_grad()
        (clf_loss + args.transop_weight*transop_loss.mean() + args.con_weight*loss_con + args.ctt_weight*ctt_loss).backward()
        if args.enable_transop:
            torch.nn.utils.clip_grad_norm_(transop_train.module.transop.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(transop_train.module.coefficient_encoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        update_momentum(backbone_train, backbone_ema, m=0.999)
        update_momentum(clf, clf_ema, m=0.999)

        clf_loss_list.append(clf_loss.item())
        transop_loss_list.append(transop_loss.mean().item())
        con_loss_list.append(loss_con.item())
        kl_loss_list.append(kl_loss.item())
        ctt_loss_list.append(ctt_loss.item())
        time_list.append(time.time() - pre_time)

        # Partial Eval
        if curr_iter % 100 == 0:
            wandb_dict = {
                'meta/time': np.mean(time_list),
                'meta/selected': (sum(wu)/len(xweak)),
                'meta/z_norm': (zweak.detach().cpu()**2).sum(-1).mean().item(),
                'train/clf': np.mean(clf_loss_list),
                'train/con': np.mean(con_loss_list),
                'train/to': np.mean(transop_loss_list),
                'train/kl': np.mean(kl_loss_list),
                'train/ctt': np.mean(ctt_loss_list),
                'train/di': np.mean(di_list)
            }
            print(f"Iter {curr_iter} -- clf loss: {np.mean(clf_loss_list):.3E}, con loss: {np.mean(con_loss_list):.3E}, ctt loss: {np.mean(ctt_loss_list):.3E}, to loss: {np.mean(transop_loss_list):.3E}, kl: {np.mean(kl_loss_list):.3E}, di: {np.mean(di_list):.3E}, time: {np.mean(time_list):.2f}sec ")
            if args.enable_transop and args.transop_aug:
                c_enc = torch.stack(c_enc_list)
                c_prior = torch.stack(c_prior_list)
                psi = transop_train.module.transop.psi.data.detach().cpu()
                psi_norm = (psi.reshape(len(psi), -1)**2).sum(-1).mean()
                print(f"Iter {curr_iter} -- c_enc: {c_enc[c_enc.abs() > 0].abs().mean():.2E} c_prior: {c_prior[c_prior.abs() > 0].abs().mean():.2E}, psi_norm: {psi_norm.item():.3E}")
                wandb_dict.update({
                    'meta/c_enc': c_enc[c_enc.abs() > 0].abs().mean(),
                    'meta/c_prior': c_prior[c_prior.abs() > 0].abs().mean(),
                    'meta/psi_norm': psi_norm.item()
                })

                if curr_iter % 2000 == 0:
                    if len(psi.shape) >= 4:
                        psi = torch.stack([torch.block_diag(*psi[i]) for i in range(len(psi))])
                    fig_dict = transop_plots(c_enc.reshape(-1, c_enc.shape[-1]).numpy(), psi, z0[0].detach().cpu().numpy())
                    for fig_name in fig_dict.keys():
                        wandb.log({"transop_plt/" + fig_name: wandb.Image(fig_dict[fig_name])}, step=curr_iter)
            transop_loss_list = []
            di_list = []
            clf_loss_list = []
            con_loss_list = []
            c_enc_list = []
            c_prior_list = []
            time_list = []
            wandb.log(wandb_dict, curr_iter)

        # Full Eval
        if curr_iter % 1000 == 0 or (curr_iter+1) == args.n_iters:
            backbone.eval()
            backbone_ema.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for idx, batch in enumerate(test_dataloader):
                    x, y = batch
                    x, y = x.to(default_device), y.to(default_device)
                    #z = backbone(x)
                    #y_pred = clf(z)
                    z = backbone_ema(x)
                    y_pred = clf_ema(z)
                    y_pred = y_pred.topk(1, 1, largest=True, sorted=True).indices
                    total += len(y_pred)
                    correct += (y_pred[:, 0] == y).sum().item()
            wandb.log({'eval/acc': (correct / total)*100}, curr_iter)
            print(f"EVAL Iter {curr_iter} -- acc: {(correct / total)*100:.2f}%")
            backbone.train()
        curr_iter += 1
    wandb.finish(quiet=True)
    args.seed = args.seed + 1
    print()
    print()

