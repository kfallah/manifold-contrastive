"""
    This is the main file for running an experiment testing
    the efficacy of running SIMCLR on the neuroscience dataset. 
"""
import argparse
import random
import time

import brainscore
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluation import (
    embed_v4_data,
    evaluate_IT_explained_variance,
    evaluate_linear_classifier,
    evaluate_logistic_regression,
)
from linear_warmup_cos_anneal import LinearWarmupCosineAnnealingLR
from torch.utils.data import Dataset
from tqdm import tqdm

import datasets
import wandb
from datasets import generate_brainscore_train_test_split


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


class Backbone(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=2, batchnorm=True):
        super().__init__()

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


# ======================== Training code ========================


class SimCLRTrainer:
    def __init__(
        self,
        v4_train,
        v4_test,
        label_train,
        label_test,
        it_train,
        it_test,
        args,
    ):
        self.v4_train = v4_train
        self.v4_test = v4_test
        self.label_train = label_train
        self.label_test = label_test
        self.it_train = it_train
        self.it_test = it_test
        self.args = args

        self.train_idx, self.test_idx = {}, {}
        for i in np.unique(self.label_train):
            idx = torch.where(self.label_train == i)[0]
            self.train_idx[i] = idx
            idx = torch.where(self.label_test == i)[0]
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

    def run_simclr(self, backbone, contrastive_head, args):
        """
        Train the model using simclr.

        Implementation details:

        1. We do contrastive learning in the output space of the model.
        2. As input we use V4 data from brainscore.
        3. We select positive pairs from different "presentations" of the same image ("stimulus")
        """
        print("Training the model using simclr")
        # Make an optimizer
        if args.optimizer == "adam":
            optim = torch.optim.AdamW(
                list(backbone.parameters()) + list(contrastive_head.parameters()),
                lr=args.learning_rate,
            )
            eta_min = 1.0e-5
        elif args.optimizer == "sgd":
            optim = torch.optim.SGD(
                list(backbone.parameters()) + list(contrastive_head.parameters()),
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
        train_losses = []
        for epoch in epoch_bar:
            epoch_bar.set_description(f"Epoch {epoch}")
            # Go through the train dataset
            indices_perm = torch.randperm(len(self.v4_train))
            for i in range(len(self.v4_train) // args.batch_size):
                # Load batch
                x0 = self.v4_train[indices_perm[i * args.batch_size : (i + 1) * args.batch_size]].to(args.device)
                y0 = self.label_train[indices_perm[i * args.batch_size : (i + 1) * args.batch_size]]
                # TODO: remove current index as a candidate for point pair
                x1_idx = torch.tensor([random.choice(self.train_idx[y_inst.item()]) for y_inst in y0])
                x1 = self.v4_train[x1_idx].to(args.device)

                # Encode features using the backbone
                a_contrast_out = contrastive_head(backbone(x0))
                b_contrast_out = contrastive_head(backbone(x1))
                # Compute the info nce loss in the output space
                if args.no_contrastive_head:
                    train_loss = self.nxent_loss(a_contrast_out, b_contrast_out, mse=True)
                else:
                    train_loss = self.nxent_loss(
                        F.normalize(a_contrast_out, dim=-1), F.normalize(b_contrast_out, dim=-1)
                    )
                # Backprop
                optim.zero_grad()
                train_loss.backward()
                optim.step()
                scheduler.step()

                train_losses.append(train_loss.item())
            train_loss = np.mean(train_losses)

            if epoch % args.eval_frequency == 0:
                print(f"Running evaluation")
                if args.eval_logistic_regression:
                    evaluate_logistic_regression(backbone, self.animals_train_dataset, self.animals_test_dataset, args)

                train_feat = embed_v4_data(self.v4_train, backbone, args.device)
                test_feat = embed_v4_data(self.v4_test, backbone, args.device)
                evaluate_linear_classifier(train_feat, self.label_train, test_feat, self.label_test, args)

                if args.eval_explained_variance:
                    evaluate_IT_explained_variance(
                        backbone, self.neuroid_train_dataset, self.neuroid_eval_dataset, args
                    )

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simclr on the neuroscience dataset")

    parser.add_argument("--dataset", type=str, default="brainscore", help="Dataset to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")  # Dim of V4 data
    parser.add_argument("--input_dim", type=int, default=88, help="Input dimension")  # Dim of V4 data
    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="Hidden dimension of contrastive head"
    )  # Dim of V4 data
    parser.add_argument(
        "--backbone_output_dim", type=int, default=32, help="Output dimension of backbone head"
    )  # Dim of V4 data
    parser.add_argument(
        "--contrastive_output_dim", type=int, default=64, help="Output dimension of contrastive head"
    )  # Dim of V4 data
    parser.add_argument(
        "--contrastive_head_batchnorm",
        type=bool,
        default=False,
        help="Whether to use batchnorms after layers in contrastive head and backbone",
    )
    parser.add_argument(
        "--backbone_batchnorm",
        type=bool,
        default=True,
        help="Whether to use batchnorms after layers in contrastive head and backbone",
    )
    parser.add_argument(
        "--no_contrastive_head",
        type=bool,
        default=False,
        help="Whether or not to include a contrastive head",
    )
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="Number of hidden layers in backbone")
    # NOTE: just doing eval every epoch rn
    # parser.add_argument("--eval_readout_frequency", type=int, default=600, help="Number of epochs between evaluating linear readout")
    parser.add_argument("--model_save_path", type=str, default="model_weights/model.pt", help="Path to save the model")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for info nce loss")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--dataset_type", type=str, default="pose", help="Type of dataset used")
    parser.add_argument("--average_trials", type=bool, default=False, help="Whether or not to average across trials")
    parser.add_argument(
        "--eval_explained_variance", type=bool, default=False, help="Whether or not to evaluate explained variance"
    )
    parser.add_argument(
        "--eval_logistic_regression", type=bool, default=False, help="Whether or not to evaluate logistic regression"
    )
    parser.add_argument("--eval_frequency", default=50)

    args = parser.parse_args()

    print("Running simclr experiment")
    # Initialize wandb
    wandb.init(
        project="neuro_simclr",
        config=args,
    )

    # Load dataset
    neuroid_train_data, neuroid_test_data = generate_brainscore_train_test_split(random_seed=args.seed)
    if args.average_trials:
        train_avg = neuroid_train_data.multi_groupby(["category_name", "object_name", "stimulus_id"]).mean(
            dim="presentation"
        )
        v4_train = train_avg.sel(region="V4").to_numpy()
        it_train = train_avg.sel(region="IT").to_numpy()
        train_category = train_avg.coords["category_name"].to_numpy()
        _, label_train = np.unique(train_category, return_inverse=True)

        test_avg = neuroid_test_data.multi_groupby(["category_name", "object_name", "stimulus_id"]).mean(
            dim="presentation"
        )
        v4_test = test_avg.sel(region="V4").to_numpy()
        it_test = test_avg.sel(region="IT").to_numpy()
        test_category = test_avg.coords["category_name"].to_numpy()
        _, label_test = np.unique(test_category, return_inverse=True)
    else:
        v4_train = neuroid_train_data.sel(region="V4").to_numpy()
        it_train = neuroid_train_data.sel(region="IT").to_numpy()
        train_category = neuroid_train_data.coords["category_name"].to_numpy()
        _, label_train = np.unique(train_category, return_inverse=True)

        v4_test = neuroid_test_data.sel(region="V4").to_numpy()
        it_test = neuroid_test_data.sel(region="IT").to_numpy()
        test_category = neuroid_test_data.coords["category_name"].to_numpy()
        _, label_test = np.unique(test_category, return_inverse=True)
    v4_train = torch.tensor(v4_train).float()
    v4_test = torch.tensor(v4_test).float()
    label_train = torch.tensor(label_train).long()
    it_train = torch.tensor(it_train).float()
    it_test = torch.tensor(it_test).float()
    label_test = torch.tensor(label_test).long()

    # train_dataloader, test_dataloader = load_dataloaders(train_dataset, test_dataset, args)
    # Initialize the model
    if args.no_contrastive_head:
        contrastive_head = nn.Identity()
    else:
        contrastive_head = ContrastiveHead(
            input_dim=args.backbone_output_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.contrastive_output_dim,
            batchnorm=args.contrastive_head_batchnorm,
        ).to(args.device)
    # NOTE: Not sure what shapes this should be
    backbone = Backbone(
        input_dim=args.input_dim,  # TBD
        hidden_dim=args.hidden_dim,
        output_dim=args.backbone_output_dim,
        num_hidden_layers=args.num_hidden_layers,
        batchnorm=args.backbone_batchnorm,
    ).to(args.device)
    print(backbone)
    # The backbone output =/= contrastive header here (in terms of dimension)
    # Run trianing
    trainer = SimCLRTrainer(
        v4_train,
        v4_test,
        label_train,
        label_test,
        it_train,
        it_test,
        args,
    )
    trainer.run_simclr(
        backbone,
        contrastive_head,
        # (train_dataloader, test_dataloader),
        args,
    )

    wandb.finish()
