# %%
import copy
import math
import random
import warnings

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.models.utils import deactivate_requires_grad, update_momentum

from dataloader.ssl_dataloader import get_dataset
from eval.utils import encode_features
from model.config import ModelConfig
from model.model import Model

warnings.filterwarnings("ignore")

num_trials = 50

# Augmentation type
# CHOICES: None, Transop, Mixup, Featmatch
feat_aug = "Featmatch"
labels_per_class = 5

thresh = 0.95
con_weight = 1.0
num_iters = 10000
h_dim = 2048
seed = 0
lr = 5e-4
no_thresh = True
alpha = 0.5

# %%
# CIFAR10
# Proj
ckpt_path = "/home/kfallah/manifold-contrastive/results/cifar10_vi-thresh0.01_dict32_kl1e-3/checkpoints/checkpoint_epoch999.pt"
cfg_path = "/home/kfallah/manifold-contrastive/results/cifar10_vi-thresh0.01_dict32_kl1e-3/.hydra/config.yaml"

# STL 10
# ckpt_path = "/home/kfallah/manifold-contrastive/results/pretrained/stl10_vi-thresh_proj/model_weight.pt"
# cfg_path = "/home/kfallah/manifold-contrastive/results/pretrained/stl10_vi-thresh_proj/config.yaml"

# ckpt_path = "/home/kfallah/manifold-contrastive/results/pretrained/stl10_vi_dict64_kl1e-5_to10_z0-neg/model_weight.pt"
# cfg_path = "/home/kfallah/manifold-contrastive/results/pretrained/stl10_vi_dict64_kl1e-5_to10_z0-neg/config.yaml"


print("Starting SSL experiment...")
print(ckpt_path)
print("Aug: ", feat_aug)
print("N trials: ", num_trials)
print("labels: ", labels_per_class)
print("thresh: ", thresh)
print("con weight: ", con_weight)
print("hidden dim: ", h_dim)
print("n iters: ", num_iters)
print("lr: ", lr)
print("no thresh: ", no_thresh)
print("alpha: ", alpha)
print()

device_idx = [0]
# Set the default device
default_device = torch.device("cuda:0")
# Load config
cfg = omegaconf.OmegaConf.load(cfg_path)
cfg.model_cfg.backbone_cfg.load_backbone = None

# CUSTOM CONFIG FOR THIS RUN #

# Load model
default_model_cfg = ModelConfig()
model = Model.initialize_model(cfg.model_cfg, cfg.dataloader_cfg.dataset_cfg.dataset_name, device_idx)
state_dict = torch.load(ckpt_path, map_location=default_device)
model.load_state_dict(state_dict['model_state'], strict=False)
# Manually override directory for dataloaders
if "tin" in ckpt_path:
    cfg.dataloader_cfg.dataset_cfg.dataset_dir = "/home/kfallah/manifold-contrastive/datasets/tiny-imagenet-200"
else:
    cfg.dataloader_cfg.dataset_cfg.dataset_dir = "/home/kfallah/manifold-contrastive/datasets"
cfg.dataloader_cfg.train_batch_size = 500
num_classes = cfg.dataloader_cfg.dataset_cfg.num_classes
# Load dataloaders
dataset = get_dataset(cfg.dataloader_cfg)
train_dataloader = dataset.eval_dataloader
test_dataloader = dataset.val_dataloader

# Set all random seeds before encoding dataset
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Get encoding of entire dataset
train_eval_input = encode_features(model, train_dataloader, default_device)
test_eval_input = encode_features(model, test_dataloader, default_device)

# Load transport operators
backbone = model.backbone.backbone_network
transop, coeff_enc = (None, None)
if model.contrastive_header.transop_header is not None:
    transop = model.contrastive_header.transop_header.transop
    coeff_enc = model.contrastive_header.transop_header.coefficient_encoder
if no_thresh:
    coeff_enc.lambda_prior = 0.0

# Eval values
x, z, y = train_eval_input.x, train_eval_input.feature_list, train_eval_input.labels
x_test, z_test, y_test = test_eval_input.x, test_eval_input.feature_list, test_eval_input.labels

# %%
class AttenHead(nn.Module):
    def __init__(self, fdim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.fatt = fdim//num_heads

        for i in range(num_heads):
            setattr(self, f'embd{i}', nn.Linear(fdim, self.fatt))
        for i in range(num_heads):
            setattr(self, f'fc{i}', nn.Linear(2*self.fatt, self.fatt))
        self.fc = nn.Linear(self.fatt*num_heads, fdim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, fx_in, fp_in):
        fp_in = fp_in.squeeze(0)
        d = math.sqrt(self.fatt)

        Nx = len(fx_in)
        f = torch.cat([fx_in, fp_in])
        f = torch.stack([getattr(self, f'embd{i}')(f) for i in range(self.num_heads)])  # head x N x fatt
        fx, fp = f[:, :Nx], f[:, Nx:]

        w = self.dropout(F.softmax(torch.matmul(fx, torch.transpose(fp, 1, 2)) / d, dim=2))  # head x Nx x Np
        fa = torch.cat([torch.matmul(w, fp), fx], dim=2)  # head x Nx x 2*fatt
        fa = torch.stack([F.relu(getattr(self, f'fc{i}')(fa[i])) for i in range(self.num_heads)])  # head x Nx x fatt
        fa = torch.transpose(fa, 0, 1).reshape(Nx, -1)  # Nx x fdim
        fx = F.relu(fx_in + self.fc(fa))  # Nx x fdim
        w = torch.transpose(w, 0, 1)  # Nx x head x Np

        return fx, w

def get_feature_split(labels_per_class, features, labels, num_classes=10, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    labeled_idx = []
    for i in range(num_classes):
        idx = torch.where(labels == i)[0]
        subset_idx = np.random.choice(idx.detach().cpu().numpy(), labels_per_class, replace=False)
        labeled_idx.append(subset_idx)
    labeled_idx = np.concatenate(labeled_idx)
    unlabel_idx = np.arange(len(features))[~np.in1d(np.arange(len(features)), labeled_idx)]

    zl, yl = features[labeled_idx], labels[labeled_idx]
    zul, yul = features[unlabel_idx], labels[unlabel_idx]

    return (zl, yl), (zul, yul)

# %%
acc_list = []

for exp_iter in range(num_trials):
    print(f"Exp iter {exp_iter+1}, seed {exp_iter + seed}")
    (train_zl, train_yl), (train_zul, train_yul) = get_feature_split(labels_per_class, z, y, num_classes=num_classes, seed=exp_iter + seed)

    clf = nn.Sequential(
        nn.Linear(512, h_dim),
        nn.LeakyReLU(),
        nn.Linear(h_dim, num_classes)
    ).to(default_device)
    clf_ema = copy.deepcopy(clf)
    deactivate_requires_grad(clf_ema)
    params = list(clf.parameters())
    attn = None
    if feat_aug == "Featmatch":
        attn = AttenHead(512, 4).to(default_device)
        params += list(attn.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=5.0e-4)
    criterion = torch.nn.functional.cross_entropy
    curr_iter = 0

    while curr_iter < num_iters:
        label_idx, unlabel_idx = torch.randperm(len(train_zl))[:32], torch.randperm(len(train_zul))[:480]
        zl, yl = train_zl[label_idx].to(default_device), train_yl[label_idx].to(default_device)
        zu = train_zul[unlabel_idx].to(default_device)

        # # TO AUG
        if feat_aug == "Transop":
            with torch.no_grad():
                c = coeff_enc.prior_sample(zu.detach())
                zu_aug = transop(zu, c)
        elif feat_aug == "Mixup":
            with torch.no_grad():
                mix_idx = torch.randperm(len(train_zul))[:480]
                zu_mix = train_zul[mix_idx].to(default_device)
                #mixup = torch.rand(len(zu_mix), device=zu_mix.device).unsqueeze(-1)
                mixup = torch.tensor(np.random.beta(alpha, alpha, len(zu_mix)), device=zu_mix.device).float().unsqueeze(-1)
                zu_aug = mixup*zu + (1-mixup)*zu_mix
        elif feat_aug == "Featmatch":
            zu_aug, wx = attn(zu, zl.unsqueeze(0))
        else:
            zu_aug = zu.clone()

        y_pred = clf(zl)
        loss = criterion(y_pred, yl)

        # Consistency Regularization 
        xu_aug_logits = clf(zu_aug)  
        if feat_aug == "Mixup":
            yaug = torch.softmax(xu_aug_logits, dim=-1)
            yu, ymix = torch.softmax(clf_ema(zu).detach(), dim=-1), torch.softmax(clf_ema(zu_mix).detach(), dim=-1)
            ylab = (mixup*yu + (1-mixup)*ymix).detach()
            loss_con = F.mse_loss(yaug, ylab)
            wu = np.ones(len(zu))
        else:       
            # TODO: potentially change to clf_ema for pseudo-labels  
            xu_prob = torch.softmax(clf(zu).detach(), dim=-1)
            wu, yu = torch.max(xu_prob, dim=1)
            wu = (wu > thresh).detach()
            loss_con = criterion(xu_aug_logits, yu, reduction="none")
            loss_con = torch.sum(loss_con[wu]) / len(wu)

        optimizer.zero_grad()
        (loss + con_weight*loss_con).backward()
        optimizer.step()

        update_momentum(clf, clf_ema, m=0.999)

        if (curr_iter%1000==0) or (curr_iter+1) == num_iters:
            correct = 0
            total = 0
            with torch.no_grad():
                for k in range(len(z_test) // 1000):
                    y_pred = clf_ema(z_test[k*1000:(k+1)*1000].to(default_device)).detach().cpu()
                    y_pred = y_pred.topk(1, 1, largest=True, sorted=True).indices
                    total += len(y_pred)
                    correct += (y_pred[:, 0] == y_test[k*1000:(k+1)*1000]).float().sum().item()
            acc = correct / total
            print(f"Iter {curr_iter} -- acc: {acc*100:.2f}%, selected: {(sum(wu)/len(zu))*100:.2f}%, z norm: {(z**2).sum(axis=-1).mean():.2f}")
        curr_iter += 1
    print()
    print()
    acc_list.append(acc*100)
print(acc_list)
print(f"Mean acc: {np.mean(acc_list):.2f}, stddev: {np.std(acc_list):.2f}")


