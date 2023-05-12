"""
    This is the main file for running an experiment testing
    the efficacy of running SIMCLR on the neuroscience dataset. 
"""
import argparse
import random
import time
import warnings

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
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU())

    def forward(self, x):
        return x + self.layer(x)


class Backbone(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=2, norm=False):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.skip_list = nn.ModuleList(SkipBlock(hidden_dim, hidden_dim) for _ in range(num_hidden_layers))
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
        self.args = args

        self.train_idx, self.test_idx = {}, {}
        for i in np.unique(self.objectid_train):
            idx = torch.where(self.objectid_train == i)[0]
            self.train_idx[i] = idx
            idx = torch.where(self.objectid_test == i)[0]
            self.test_idx[i] = idx

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
        if manifold_model is not None:
            transop, coeff_enc = manifold_model
            param_groups += [
                {
                    "params": transop.parameters(),
                    "lr": 1.0e-3,
                    "weight_decay": 1.0e-3,
                }
            ]
            param_groups += [
                {
                    "params": coeff_enc.parameters(),
                    "lr": 1.0e-4,
                    "weight_decay": 1.0e-5,
                }
            ]

        print("Training the model using simclr")
        # Make an optimizer
        if args.optimizer == "adam":
            optim = torch.optim.AdamW(
                param_groups,
                lr=args.learning_rate,
            )
            eta_min = 1.0e-5
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
            optim, warmup_epochs=10 * iters_per_epoch, max_epochs=args.num_epochs * iters_per_epoch, eta_min=eta_min
        )

        # Run the trianing
        epoch_bar = tqdm(range(args.num_epochs))
        for epoch in epoch_bar:
            train_loss_list = []
            pairwise_dist = []
            feat_norm = []
            epoch_bar.set_description(f"Epoch {epoch}")
            # Go through the train dataset
            indices_perm = torch.randperm(len(self.v4_train))
            for i in range(len(self.v4_train) // args.batch_size):
                # Load batch
                x0 = self.v4_train[indices_perm[i * args.batch_size : (i + 1) * args.batch_size]].to(args.device)
                y0 = self.objectid_train[indices_perm[i * args.batch_size : (i + 1) * args.batch_size]]
                # TODO: remove current index as a candidate for point pair
                x1_idx = torch.tensor([random.choice(self.train_idx[y_inst.item()]) for y_inst in y0])
                x1 = self.v4_train[x1_idx].to(args.device)

                # Encode features using the backbone
                z0, z1 = backbone(x0), backbone(x1)
                h0, h1 = contrastive_head(z0), contrastive_head(z1)
                # Compute the info nce loss in the output space
                if args.no_contrastive_head:
                    train_loss = self.nxent_loss(h0, h1, mse=True)
                else:
                    train_loss = self.nxent_loss(F.normalize(h0, dim=-1), F.normalize(h1, dim=-1))
                # Backprop
                optim.zero_grad()
                train_loss.backward()
                optim.step()
                scheduler.step()

                train_loss_list.append(train_loss.item())
                pairwise_dist.append(F.mse_loss(z0, z1).item())
                feat_norm.append((z1**2).sum(-1).mean().item())
            wandb_dict = {
                "train/train_loss": np.mean(train_loss_list),
                "train/pairwise_dist": np.mean(pairwise_dist),
                "train/feat_mag": np.mean(feat_norm),
            }

            if epoch % args.eval_frequency == 0 or epoch == (args.num_epochs - 1):
                print(f"Running evaluation")
                if args.eval_logistic_regression:
                    evaluate_logistic_regression(backbone, self.animals_train_dataset, self.animals_test_dataset, args)

                train_feat = embed_v4_data(self.v4_train, backbone, args.device)
                test_feat = embed_v4_data(self.v4_test, backbone, args.device)
                train_acc, train_fscore = evaluate_linear_classifier(
                    train_feat, self.label_train, train_feat, self.label_train, args
                )
                test_acc, test_fscore = evaluate_linear_classifier(
                    train_feat, self.label_train, test_feat, self.label_test, args
                )
                # Eval object id linear
                if args.eval_object_id_linear:
                    acc, fscore = evaluate_linear_classifier(
                        train_feat, self.objectid_train, test_feat, self.objectid_test, args
                    )
                    wandb_dict.update({
                        "eval/linear_acc_objid": acc, 
                        "eval/linear_fscore_objid": fscore, 
                    })
                # Make a tnse plot based on category
                category_tsne = tsne_plot(train_feat, self.label_train)
                # Make a tsne plot based on object id
                object_id_tsne = tsne_plot(train_feat, self.objectid_train)

                if args.eval_pose_regression:
                    # pose regression: assuming this is what pose change regression would
                    # converge to with many pairs
                    pose_r2 = evaluate_pose_regression(train_feat, self.pose_train, test_feat, self.pose_test, args)
                    wandb_dict['eval/pose_R2_mean'] = pose_r2[0]
                    wandb_dict['eval/pose_R2_median'] = pose_r2[1]
                    for i, dim in enumerate(pose_dims):
                        wandb_dict[f'eval/pose_R2_{dim}'] = pose_r2[2+i]

                    if manifold_model is not None:
                        # pose change regression
                        pose_change_r2 = evaluate_pose_change_regression(
                            manifold_model, train_feat, self.pose_train, test_feat, self.pose_test, args
                        )
                        wandb_dict['eval/pose_change_R2_mean'] = pose_change_r2[0]
                        wandb_dict['eval/pose_change_R2_median'] = pose_change_r2[1]
                        for i, dim in enumerate(pose_dims):
                            wandb_dict[f'eval/pose_change_R2_{dim}'] = pose_change_r2[2+i]

                if args.eval_explained_variance:
                    raise NotImplementedError("Explained variance not implemented for the new dataset structure")
                    evaluate_IT_explained_variance(
                        backbone, self.neuroid_train_dataset, self.neuroid_eval_dataset, args
                    )

                wandb_dict.update({
                    "eval/train_linear_acc": train_acc, 
                    "eval/train_linear_fscore": train_fscore, 
                    "eval/test_linear_acc": test_acc,
                    "eval/test_linear_fscore": test_fscore,
                    "figs/category_tsne": wandb.Image(category_tsne),
                    "figs/object_id_tsne": wandb.Image(object_id_tsne),
                })

            wandb.log(wandb_dict, step=epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simclr on the neuroscience dataset")

    parser.add_argument("--dataset", type=str, default="brainscore", help="Dataset to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")  # Dim of V4 data
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
        type=bool,
        default=True,
        help="Whether to use batchnorms after layers in contrastive head and backbone",
    )
    parser.add_argument(
        "--backbone_batchnorm",
        type=bool,
        default=False,
        help="Whether to use batchnorms after layers in contrastive head and backbone",
    )
    parser.add_argument(
        "--no_contrastive_head",
        type=bool,
        default=True,
        help="Whether or not to include a contrastive head",
    )
    parser.add_argument("--num_hidden_layers", type=int, default=4, help="Number of hidden layers in backbone")
    # NOTE: just doing eval every epoch rn
    # parser.add_argument("--eval_readout_frequency", type=int, default=600, help="Number of epochs between evaluating linear readout")
    parser.add_argument("--model_save_path", type=str, default="model_weights/model.pt", help="Path to save the model")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use")
    parser.add_argument("--temperature", type=float, default=7e-2, help="Temperature for info nce loss")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--dataset_type", type=str, default="pose", help="Type of dataset used")
    parser.add_argument("--average_trials", type=bool, default=False, help="Whether or not to average across trials")
    parser.add_argument("--average_downsample_factor", type=int, default=5, help="Factor to downsample average by")
    parser.add_argument("--ignore_cache", type=bool, default=False, help="Whether or not to ignore the cache")
    parser.add_argument(
        "--eval_explained_variance", type=bool, default=False, help="Whether or not to evaluate explained variance"
    )
    parser.add_argument(
        "--eval_logistic_regression", type=bool, default=False, help="Whether or not to evaluate logistic regression"
    )
    parser.add_argument(
        "--eval_object_id_linear", type=bool, default=True, help="Whether or not to evaluate object id linear"
    )
    parser.add_argument("--eval_pose_regression", type=bool, default=True, help="Whether or not to evaluate pose regression")
    parser.add_argument("--eval_pose_change_regr_n_pairs", type=int, default=100000, help="Number of pairs to use for pose change regression training and test")
    parser.add_argument("--eval_frequency", default=100)

    # ManifoldCLR args
    parser.add_argument("--enable_manifoldclr", type=bool, default=False, help="Enable ManifoldCLR")
    parser.add_argument("--dict_size", type=int, default=10, help="Dictionary size")
    parser.add_argument("--kl_weight", type=float, default=1.0e-5, help="KL Div weight")
    parser.add_argument("--threshold", type=float, default=0.0, help="Reparam threshold")

    args = parser.parse_args()

    print("Running simclr experiment")
    # Initialize wandb
    wandb.init(
        project="neuro_simclr",
        config=args,
    )

    # Load dataset
    train_data, test_data = get_dataset(
        args.average_trials, 
        args.average_downsample_factor,
        args.seed,
        ignore_cache=args.ignore_cache,
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
        args,
    )
    trainer.run_simclr(
        backbone,
        contrastive_head,
        # (train_dataloader, test_dataloader),
        args,
    )

    wandb.finish()
