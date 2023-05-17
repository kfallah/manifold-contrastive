"""
    This is the main file for running an experiment testing
    the efficacy of running SIMCLR on the neuroscience dataset. 
"""
import argparse
import os
import random
import sys
import time
import warnings

sys.path.append(os.getcwd() + "/src/")


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluation import (
    embed_v4_data,
    evaluate_IT_explained_variance,
    evaluate_linear_classifier,
    evaluate_logistic_regression,
    evaluate_pose_change_regression,
    evaluate_pose_regression,
    transop_plots,
    tsne_plot,
)
from linear_warmup_cos_anneal import LinearWarmupCosineAnnealingLR
from tqdm import tqdm

import wandb
from datasets import get_dataset, pose_dims
from model.contrastive.config import VariationalEncoderConfig
from model.manifold.reparameterize import compute_kl
from model.manifold.transop import TransOp_expm
from model.manifold.vi_encoder import CoefficientEncoder

warnings.filterwarnings("ignore")


class ContrastiveHead(torch.nn.Module):
    """
    Contrastive head MLP.

    For a baseline, we can use two simple linear layers.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, batchnorm=False):
        super().__init__()
        layers = [
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        ]
        if batchnorm:
            layers.insert(1, torch.nn.BatchNorm1d(hidden_dim))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SkipBlock(torch.nn.Module):
    def __init__(self, dim, norm=False):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.ln = None
        if norm:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        out = x + self.layer(x)
        if self.ln is not None:
            out = self.ln(out)
        return out


class Backbone(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=2, norm=False):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.skip_list = nn.ModuleList(SkipBlock(hidden_dim, norm=norm) for _ in range(num_hidden_layers))
        self.decode = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        z = self.enc(x)
        for skip_layer in self.skip_list:
            z = skip_layer(z)
        return self.decode(z)

    """
        def linear_block(dim_in, dim_out):
            layers = [torch.nn.Linear(dim_in, dim_out)]
            if batchnorm:
                layers.append(torch.nn.BatchNorm1d(dim_out))
            layers.append(torch.nn.LeakyReLU())
            return layers

        extra_hidden_layers = []
        for _ in range(num_hidden_layers - 1):  # already have one
            extra_hidden_layers.extend(linear_block(hidden_dim, hidden_dim))

        self.model = torch.nn.Sequential(
            *linear_block(input_dim, hidden_dim),
            *extra_hidden_layers,
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)
    """


# ======================== Training code ========================


class SimCLRTrainer:
    def __init__(
        self,
        v4_train,
        v4_test,
        label_train,
        label_test,
        objectid_train,
        objectid_test,
        pose_train,
        pose_test,
        it_train,
        it_test,
        args,
    ):
        self.v4_train = v4_train
        self.v4_test = v4_test
        self.label_train = label_train
        self.label_test = label_test
        self.objectid_train = objectid_train
        self.objectid_test = objectid_test
        self.pose_train = pose_train
        self.pose_test = pose_test
        self.it_train = it_train
        self.it_test = it_test
        self.args = args

        self.train_idx = {}
        for i in range(len(v4_train)):
            label = self.objectid_train[i]
            idx = torch.where(self.objectid_train == label)[0]
            # Add all neighbors except the object itself
            self.train_idx[i] = torch.tensor([nn.item() for nn in idx if nn.item() != i])

        self.test_idx = {}
        for i in range(len(v4_test)):
            label = self.objectid_test[i]
            idx = torch.where(self.objectid_test == label)[0]
            # Add all neighbors except the object itself
            self.test_idx[i] = torch.tensor([nn.item() for nn in idx if nn.item() != i])

    def nxent_loss(self, out_1, out_2, out_3=None, temperature=0.07, mse=False, eps=1e-6):
        """
        DOES NOT assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        out_3: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        # out_3_dist: [batch_size * world_size, dim]
        # out: [2 * batch_size, dim]
        # out_dist: [3 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        if out_3 is not None:
            out_dist = torch.cat([out_1, out_2, out_3], dim=0)
        else:
            out_dist = torch.cat([out_1, out_2], dim=0)

        # cov and sim: [2 * batch_size, 3 * batch_size * world_size]
        # neg: [2 * batch_size]
        if mse:
            cov = -((out.unsqueeze(1) - out_dist.unsqueeze(0)) ** 2).mean(dim=-1)
        else:
            cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature).clamp(min=1e-18)
        neg = sim.sum(dim=-1)

        if not mse:
            row_sub = torch.exp(torch.norm(out, dim=-1) / temperature)
            neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        if mse:
            pos = -((out_1 - out_2) ** 2).mean(dim=-1)
            pos = torch.exp(pos / temperature).clamp(min=1e-18)
        else:
            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature).clamp(min=1e-18)
        pos = torch.cat([pos, pos], dim=0)
        loss = -torch.log(pos / (neg + eps)).mean()
        if loss < 0.0:
            print("Lie Contrastive loss can't be negative")
            raise ValueError("Lie Contrastive loss can't be negative")
        return loss

    def run_simclr(self, backbone, contrastive_head, args, manifold_model=None):
        """
        Train the model using simclr.

        Implementation details:

        1. We do contrastive learning in the output space of the model.
        2. As input we use V4 data from brainscore.
        3. We select positive pairs from different "presentations" of the same image ("stimulus")
        """
        param_groups = [
            {
                "params": list(backbone.parameters()) + list(contrastive_head.parameters()),
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            }
        ]

        transop, coeff_enc = None, None
        if args.enable_manifoldclr:
            transop, coeff_enc = manifold_model
            param_groups += [
                {
                    "params": transop.parameters(),
                    "lr": 1.0e-3,
                    "weight_decay": args.to_wd,
                }
            ]
            param_groups += [
                {
                    "params": coeff_enc.parameters(),
                    "lr": 1.0e-4,
                    "weight_decay": 1.0e-5,
                }
            ]

        model_name = "SimCLR" if manifold_model is None else "ManifoldCLR"
        print(f"Training the model using {model_name}")
        # Make an optimizer
        if args.optimizer == "adam":
            optim = torch.optim.AdamW(
                param_groups,
                lr=args.learning_rate,
            )
            eta_min = 1.0e-6
        elif args.optimizer == "sgd":
            optim = torch.optim.SGD(
                param_groups,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            eta_min = 1.0e-3
        else:
            raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")

        iters_per_epoch = len(self.v4_train) // args.batch_size
        scheduler = LinearWarmupCosineAnnealingLR(
            optim, warmup_epochs=100 * iters_per_epoch, max_epochs=args.num_epochs * iters_per_epoch, eta_min=eta_min
        )

        # Run the trianing
        epoch_bar = tqdm(range(args.num_epochs))
        for epoch in epoch_bar:
            to_loss_list = []
            kl_loss_list = []
            train_loss_list = []
            pairwise_dist = []
            dist_improve_list = []
            feat_norm = []
            epoch_bar.set_description(f"Epoch {epoch}")
            # Go through the train dataset
            backbone.train()
            contrastive_head.train()
            indices_perm = torch.randperm(len(self.v4_train))
            for i in range(len(self.v4_train) // args.batch_size):
                # Load batch
                current_idx = indices_perm[i * args.batch_size : (i + 1) * args.batch_size]
                x0 = self.v4_train[current_idx].to(args.device)
                x1_idx = torch.tensor([random.choice(self.train_idx[inst_idx.item()]) for inst_idx in current_idx])
                x1 = self.v4_train[x1_idx].to(args.device)

                # Encode features using the backbone
                z0, z1 = backbone(x0), backbone(x1)

                # Handle ManifoldCLR options
                transop_loss, kl_loss, shiftl2_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
                if manifold_model is not None:
                    # Manifold loss
                    distribution_data = coeff_enc(z0.detach(), z1.detach(), transop, i + epoch * iters_per_epoch)
                    c = distribution_data.samples
                    z1_hat = transop(z0.float(), c)

                    if args.enable_shiftl2:
                        shift = distribution_data.encoder_params["shift"]
                        shiftl2_loss = (shift**2).sum(-1).mean()

                    # KL loss
                    transop_loss = F.mse_loss(z1_hat, z1.detach(), reduction="none")
                    kl_loss = compute_kl(
                        "Laplacian", distribution_data.encoder_params, distribution_data.prior_params
                    ).mean()

                    # Prior augmentation
                    c0 = coeff_enc.prior_sample(
                        z0.detach(),
                        curr_iter=i + epoch * iters_per_epoch,
                        distribution_params=distribution_data.prior_params,
                    )
                    z0_aug = transop(z0.float(), c0)
                    h0_unaug = contrastive_head(z0)
                else:
                    z0_aug = z0.clone()

                # Pass through contrastive head
                h0, h1 = contrastive_head(z0_aug), contrastive_head(z1)

                # Compute the info nce loss in the output space
                if args.no_contrastive_head:
                    train_loss = self.nxent_loss(
                        h0, h1, out_3=h0_unaug if (args.enable_manifoldclr and args.z0_neg) else None, mse=True
                    )
                else:
                    train_loss = self.nxent_loss(F.normalize(h0, dim=-1), F.normalize(h1, dim=-1))
                # Backprop
                optim.zero_grad()
                (
                    train_loss
                    + args.to_weight * transop_loss.mean()
                    + args.kl_weight * kl_loss
                    + args.shiftl2_weight * shiftl2_loss
                ).backward()

                if args.enable_manifoldclr:
                    torch.nn.utils.clip_grad_norm_(coeff_enc.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(transop.parameters(), 0.1)

                optim.step()
                scheduler.step()

                dist = F.mse_loss(z0, z1, reduction="none")
                train_loss_list.append(train_loss.item())
                to_loss_list.append(transop_loss.mean().item())
                kl_loss_list.append(kl_loss.item())
                pairwise_dist.append(dist.mean().item())
                feat_norm.append((z1**2).sum(-1).mean().item())
                dist_improve_list.append((transop_loss.sum(-1) / (dist.sum(-1) + 1e-6)).mean().item())
            wandb_dict = {
                "train/train_loss": np.mean(train_loss_list),
                "train/pairwise_dist": np.mean(pairwise_dist),
                "train/feat_mag": np.mean(feat_norm),
            }

            # Add transport operator logging
            if args.enable_manifoldclr:
                encoder_logscale, encoder_shift = (
                    distribution_data.encoder_params["logscale"],
                    distribution_data.encoder_params["shift"],
                )
                prior_logscale, prior_shift = (
                    distribution_data.prior_params["logscale"],
                    distribution_data.prior_params["shift"],
                )
                wandb_dict.update(
                    {
                        "transop/transop_loss": np.mean(to_loss_list),
                        "transop/kl_loss": np.mean(kl_loss_list),
                        "transop/dist_improve": np.mean(dist_improve_list),
                        "transop/mean_psi_norm": (transop.psi.reshape(args.dict_size, -1) ** 2).sum(-1).mean().item(),
                        "transop_vi/c_enc_mag": c[c.abs() > 0].detach().abs().cpu().mean(),
                        "transop_vi/c_prior_mag": c0[c0.abs() > 0].detach().abs().cpu().mean(),
                        "transop_vi/mean_enc_scale": encoder_logscale.exp().mean().item(),
                        "transop_vi/mean_enc_shift": encoder_shift.abs().mean().item(),
                        "transop_vi/mean_prior_scale": prior_logscale.exp().mean().item(),
                        "transop_vi/mean_prior_shift": prior_shift.abs().mean().item(),
                    }
                )

            if epoch % args.eval_frequency == 0 or epoch == (args.num_epochs - 1):
                print(f"Running evaluation")
                backbone.eval()
                contrastive_head.eval()
                if args.eval_logistic_regression:
                    evaluate_logistic_regression(backbone, self.animals_train_dataset, self.animals_test_dataset, args)

                train_feat = embed_v4_data(self.v4_train, backbone, args.device)
                test_feat = embed_v4_data(self.v4_test, backbone, args.device)

                tsne = tsne_plot(train_feat, self.label_train)
                train_category_acc, train_category_fscore = evaluate_linear_classifier(
                    train_feat, self.label_train, train_feat, self.label_train, args
                )
                test_category_acc, train_category_fscore = evaluate_linear_classifier(
                    train_feat, self.label_train, test_feat, self.label_test, args
                )
                wandb_dict.update(
                    {
                        "eval/train_cat_linear_acc": train_category_acc,
                        "eval/train_cat_linear_fscore": train_category_fscore,
                        "eval/test_cat_linear_acc": test_category_acc,
                        "eval/test_cat_linear_fscore": train_category_fscore,
                        "figs/tsne": wandb.Image(tsne),
                    }
                )

                if args.eval_object_id_linear:
                    object_acc, object_fscore = evaluate_linear_classifier(
                        train_feat, self.objectid_train, test_feat, self.objectid_test, args
                    )
                    object_id_tsne = tsne_plot(train_feat, self.objectid_train)
                    wandb_dict.update(
                        {
                            "eval/obj_linear_acc": object_acc,
                            "eval/obj_linear_fscore": object_fscore,
                            "figs/obj_tsne": wandb.Image(object_id_tsne),
                        }
                    )

                if args.eval_pose_regression:
                    # pose regression: assuming this is what pose change regression would
                    # converge to with many pairs
                    pose_r2, pose_r = evaluate_pose_regression(
                        train_feat, self.pose_train, test_feat, self.pose_test, args
                    )
                    wandb_dict["eval/R2_mean"] = pose_r2[0]
                    wandb_dict["eval/R2_median"] = pose_r2[1]
                    wandb_dict["eval/R_mean"] = pose_r[0]
                    wandb_dict["eval/R_median"] = pose_r[1]
                    for i, dim in enumerate(pose_dims):
                        wandb_dict[f"eval_pose/R2_{dim}"] = pose_r2[2 + i]
                        wandb_dict[f"eval_pose/R_{dim}"] = pose_r[2 + i]

                    if args.enable_manifoldclr:
                        # pose change regression
                        pose_change_r2, pose_change_r = evaluate_pose_change_regression(
                            manifold_model,
                            train_feat,
                            self.train_idx,
                            self.pose_train,
                            test_feat,
                            self.test_idx,
                            self.pose_test,
                            args,
                        )
                        wandb_dict["eval/diff_R2_mean"] = pose_change_r2[0]
                        wandb_dict["eval/diff_R2_median"] = pose_change_r2[1]
                        wandb_dict["eval/diff_R_mean"] = pose_change_r[0]
                        wandb_dict["eval/diff_R_median"] = pose_change_r[1]
                        for i, dim in enumerate(pose_dims):
                            wandb_dict[f"eval_pose/diff_R2_{dim}"] = pose_change_r2[2 + i]
                            wandb_dict[f"eval_pose/diff_R_{dim}"] = pose_change_r[2 + i]

                if args.eval_explained_variance:
                    raise NotImplementedError("Explained variance not implemented for the new dataset structure")
                    evaluate_IT_explained_variance(
                        backbone, self.neuroid_train_dataset, self.neuroid_eval_dataset, args
                    )

                if args.enable_manifoldclr:
                    # TODO: modify passing psi if incorporating block diagonal constraint
                    fig_plots = transop_plots(
                        c.detach().cpu().numpy(), transop.psi[:, 0].detach().cpu(), z0.detach().cpu().numpy()
                    )
                    for fig_name in fig_plots.keys():
                        wandb_dict.update({"transop_plt/" + fig_name: wandb.Image(fig_plots[fig_name])})

                torch.save(
                    {
                        "args": args,
                        "backbone": backbone.state_dict(),
                        "header": contrastive_head.state_dict(),
                        "transop": transop.state_dict() if args.enable_manifoldclr else None,
                        "coeff_enc": coeff_enc.state_dict() if args.enable_manifoldclr else None,
                    },
                    args.save_dir + f"/model_weights_epoch{epoch}.pt",
                )

            wandb.log(wandb_dict, step=epoch)

def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simclr on the neuroscience dataset")

    parser.add_argument("--dataset", type=str, default="brainscore", help="Dataset to use")
    parser.add_argument("--seed", type=int, default=747, help="Random seed")  # Dim of V4 data
    parser.add_argument("--input_dim", type=int, default=88, help="Input dimension")  # Dim of V4 data
    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="Hidden dimension of contrastive head"
    )  # Dim of V4 data
    parser.add_argument(
        "--backbone_output_dim", type=int, default=64, help="Output dimension of backbone head"
    )  # Dim of V4 data
    parser.add_argument(
        "--contrastive_output_dim", type=int, default=64, help="Output dimension of contrastive head"
    )  # Dim of V4 data
    parser.add_argument(
        "--contrastive_head_batchnorm",
        type=str2bool,
        default=True,
        help="Whether to use batchnorms after layers in contrastive head and backbone",
    )
    parser.add_argument(
        "--backbone_batchnorm",
        type=str2bool,
        default=True,
        help="Whether to use batchnorms after layers in contrastive head and backbone",
    )
    parser.add_argument(
        "--no_contrastive_head",
        type=str2bool,
        default=True,
        help="Whether or not to include a contrastive head",
    )
    parser.add_argument("--num_hidden_layers", type=int, default=6, help="Number of hidden layers in backbone")
    # NOTE: just doing eval every epoch rn
    # parser.add_argument("--eval_readout_frequency", type=int, default=600, help="Number of epochs between evaluating linear readout")
    parser.add_argument("--save_dir", type=str, default="./neuro_results/", help="Path to save the model")
    parser.add_argument("--num_epochs", type=int, default=10000, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5.0e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use")
    parser.add_argument("--temperature", type=float, default=1e-1, help="Temperature for info nce loss")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--dataset_type", type=str, default="pose", help="Type of dataset used")
    parser.add_argument("--average_trials", type=str2bool, default=True, help="Whether or not to average across trials")
    parser.add_argument("--dataset_split_version", default="stimulus")
    parser.add_argument("--average_downsample_factor", type=int, default=50, help="Factor to downsample average by")
    parser.add_argument("--ignore_cache", type=str2bool, default=False, help="Whether or not to ignore the cache")
    parser.add_argument(
        "--eval_explained_variance", type=str2bool, default=False, help="Whether or not to evaluate explained variance"
    )
    parser.add_argument(
        "--eval_logistic_regression", type=str2bool, default=False, help="Whether or not to evaluate logistic regression"
    )
    parser.add_argument(
        "--eval_object_id_linear", type=str2bool, default=True, help="Whether or not to evaluate object id linear"
    )
    parser.add_argument(
        "--eval_pose_regression", type=str2bool, default=True, help="Whether or not to evaluate pose regression"
    )
    parser.add_argument(
        "--eval_pose_change_regr_n_pairs",
        type=int,
        default=5000,
        help="Number of pairs to use for pose change regression training and test",
    )
    parser.add_argument(
        "--eval_regression_model", type=str, default="svr", help="Regression model to fit for prediction."
    )
    parser.add_argument("--eval_frequency", default=500)

    # ManifoldCLR args
    parser.add_argument("--enable_manifoldclr", type=str2bool, default=True, help="Enable ManifoldCLR")
    parser.add_argument("--dict_size", type=int, default=32, help="Dictionary size")
    parser.add_argument("--z0_neg", type=str2bool, default=True, help="Whether to use z0 as a negative.")
    parser.add_argument("--to_weight", type=float, default=1.0e1, help="Transop loss weight")
    parser.add_argument("--to_wd", type=float, default=1.0e-5, help="Transop loss weight")
    parser.add_argument("--kl_weight", type=float, default=1.0e-6, help="KL Div weight")
    parser.add_argument("--threshold", type=float, default=0.0, help="Reparam threshold")
    parser.add_argument("--max_elbo", type=str2bool, default=False, help="Max elbo sampling for enc inference")
    parser.add_argument("--enable_shiftl2", type=str2bool, default=False, help="Enable shift l2 loss")
    parser.add_argument("--shiftl2_weight", type=float, default=1.0e-3, help="Shift l2 loss weight")
    parser.add_argument("--run_name", type=str, default="default", help="runname")

    args = parser.parse_args()
    args.save_dir = args.save_dir + args.run_name

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print("Created directory for figures at {}".format(args.save_dir))

    print(f"Running simclr experiment in {args.save_dir}")
    # Initialize wandb
    wandb.init(
        project="neuro_simclr",
        config=args,
    )
    if args.run_name != "default":
        wandb.run.name = args.run_name

    # Load dataset
    assert args.dataset_split_version in ["stimulus", "old"]
    train_data, test_data = get_dataset(
        args.average_trials,
        args.average_downsample_factor,
        args.seed,
        ignore_cache=args.ignore_cache,
        dataset_split_version=args.dataset_split_version,
    )
    (v4_train, it_train, label_train, objectid_train, pose_train) = train_data
    (v4_test, it_test, label_test, objectid_test, pose_test) = test_data
    # train_dataloader, test_dataloader = load_dataloaders(train_dataset, test_dataset, args)
    # Initialize the model
    if args.no_contrastive_head:
        contrastive_head = nn.Identity()
    else:
        contrastive_head = ContrastiveHead(
            input_dim=args.backbone_output_dim,
            hidden_dim=1024,
            output_dim=args.contrastive_output_dim,
            batchnorm=args.contrastive_head_batchnorm,
        ).to(args.device)
    # NOTE: Not sure what shapes this should be
    backbone = Backbone(
        input_dim=args.input_dim,  # TBD
        hidden_dim=args.hidden_dim,
        output_dim=args.backbone_output_dim,
        num_hidden_layers=args.num_hidden_layers,
        norm=args.backbone_batchnorm,
    ).to(args.device)
    print(backbone)

    manifold_model = None
    if args.enable_manifoldclr:
        threshold = 0.0
        transop = TransOp_expm(
            M=args.dict_size,
            N=args.backbone_output_dim,
            stable_init=True,
            real_range=1.0e-4,
            imag_range=5.0,
            dict_count=1,
        ).to(args.device)
        vi_cfg = VariationalEncoderConfig(
            scale_prior=0.005,
            shift_prior=0.01,
            enable_learned_prior=True,
            enable_prior_shift=True,
            enable_prior_warmup=True,
            prior_warmup_iters=500,
            enable_max_sampling=args.max_elbo,
            max_sample_start_iter=0,
            samples_per_iter=20,
            total_num_samples=20,
        )
        coeff_enc = CoefficientEncoder(vi_cfg, args.backbone_output_dim, args.dict_size, args.threshold).to(
            args.device
        )
        manifold_model = (transop, coeff_enc)

    # The backbone output =/= contrastive header here (in terms of dimension)
    # Run trianing
    trainer = SimCLRTrainer(
        v4_train,
        v4_test,
        label_train,
        label_test,
        objectid_train,
        objectid_test,
        pose_train,
        pose_test,
        it_train,
        it_test,
        args,
    )
    trainer.run_simclr(
        backbone,
        contrastive_head,
        # (train_dataloader, test_dataloader),
        args,
        manifold_model=manifold_model,
    )

    wandb.finish()
