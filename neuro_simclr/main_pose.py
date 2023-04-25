"""
    This is the main file for running an experiment testing
    the efficacy of running SIMCLR on the neuroscience dataset. 
"""
import argparse
import wandb
import brainscore
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import sklearn

from datasets import AnimalClassificationDataset, TrialContrastiveDataset, PoseContrastiveDataset
from evaluation import evaluate_linear_readout

class ContrastiveHead(torch.nn.Module):
    """
        Contrastive head MLP.

        For a baseline, we can use two simple linear layers.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)

class Backbone(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.model(x)

def load_dataloaders(train_dataset, test_dataset, args):
    """
        Load the dataloaders
    """
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    return train_dataloader, test_dataloader

# ======================== Training code ========================

class SimCLRTrainer():

    def __init__(self, train_dataset, test_dataset, animals_train_dataset, animals_test_dataset, args):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.animals_train_dataset = animals_train_dataset
        self.animals_test_dataset = animals_test_dataset
        self.args = args

    def nxent_loss(self, out_1, out_2, out_3=None, temperature=0.07, mse=False, eps=1e-6):
        """
        DOES NOT assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        out_3: [batch_size, dim]
        """
        out_1 = out_1 / (out_1.norm(dim=1, keepdim=True) + eps)
        out_2 = out_2 / (out_2.norm(dim=1, keepdim=True) + eps)
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
            cov = -((out.unsqueeze(1) - out_dist.unsqueeze(0))**2).mean(dim=-1)
        else:
            cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        if not mse:
            row_sub = torch.exp(torch.norm(out, dim=-1) / temperature)
            neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        if mse:
            pos = -((out_1 - out_2)**2).mean(dim=-1)
            pos = torch.exp(pos / temperature)
        else:
            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)
        loss = -torch.log(pos / (neg + eps)).mean()
        if loss < 0.0:
            print("Lie Contrastive loss can't be negative")
            raise ValueError("Lie Contrastive loss can't be negative")
        return loss

    def run_simclr(self, backbone, contrastive_head, dataloaders, args):
        """
            Train the model using simclr.

            Implementation details:

            1. We do contrastive learning in the output space of the model.
            2. As input we use V4 data from brainscore. 
            3. We select positive pairs from different "presentations" of the same image ("stimulus")
        """
        print("Training the model using simclr")
        # Unpack the dataloaders
        train_dataloader, test_dataloader = dataloaders
        # Make an optimizer
        if args.optimizer == "adam":
            optim = torch.optim.Adam(
                list(backbone.parameters()) + list(contrastive_head.parameters()),
                lr=args.learning_rate,
            )
        elif args.optimizer == "sgd":
            optim = torch.optim.SGD(
                list(backbone.parameters()) + list(contrastive_head.parameters()),
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
            )
        else:
            raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")
        # Run the trianing
        epoch_bar = tqdm(range(args.num_epochs))
        for epoch in epoch_bar:
            epoch_bar.set_description(f"Epoch {epoch}")
            # Train the model
            train_losses = []
            for batch in tqdm(train_dataloader):
                # Unpack the batch
                item_a, item_b = batch
                item_a = item_a.to(args.device)
                item_b = item_b.to(args.device)
                # Encode features using the backbone
                a_contrast_out = contrastive_head(backbone(item_a))
                b_contrast_out = contrastive_head(backbone(item_b))
                # Compute the info nce loss in the output space
                loss = self.nxent_loss(
                    a_contrast_out,
                    b_contrast_out
                )
                train_losses.append(
                    loss.item()
                )
                # Backprop
                optim.zero_grad()
                loss.backward() 
                optim.step()
            train_loss = np.mean(train_losses)
            # Run testing
            test_losses = []
            for batch in tqdm(test_dataloader):
                # Unpack the batch
                item_a, item_b = batch
                item_a = item_a.to(args.device)
                item_b = item_b.to(args.device)
                # Encode features using the backbone
                a_contrast_out = contrastive_head(backbone(item_a))
                b_contrast_out = contrastive_head(backbone(item_b))
                # Compute the info nce loss in the output space
                loss = self.nxent_loss(
                    a_contrast_out,
                    b_contrast_out
                )
                test_losses.append(
                    loss.item()
                )
            test_loss = np.mean(test_losses)
            # Run the linear readout evaluation
            # TODO: Maybe change this from every epoch to every num iterations
            if epoch % args.eval_readout_frequency == 0:
                accuracy, fscore = evaluate_linear_readout(
                    backbone,
                    self.animals_train_dataset, 
                    self.animals_test_dataset, 
                    args
                )
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "accuracy": accuracy,   
                    "fscore": fscore,
                })
            # Save the model at the end of each epoch
            # with open(args.model_save_path, "wb") as f:
            #     state_dict = model.state_dict()
            #     torch.save(state_dict, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run simclr on the neuroscience dataset")

    parser.add_argument("--dataset", type=str, default="brainscore", help="Dataset to use")
    parser.add_argument("--input_dim", type=int, default=88, help="Input dimension") # Dim of V4 data
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension of contrastive head") # Dim of V4 data
    parser.add_argument("--output_dim", type=int, default=64, help="Output dimension of contrastive head") # Dim of V4 data
    # NOTE: just doing eval every epoch rn
    parser.add_argument("--eval_readout_frequency", type=int, default=10, help="Number of epochs between evaluating linear readout") 
    parser.add_argument("--model_save_path", type=str, default="model_weights/model.pt", help="Path to save the model")
    parser.add_argument("--num_epochs", type=int, default=2000, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use")
    parser.add_argument("--dataset_type", type=str, default="pose", help="Type of dataset used")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()
    
    print("Running simclr experiment")
    # Initialize wandb
    wandb.init(
        project="neuro_simclr",
        entity="helblazer811",
        config=args,
    )
    # Load simclr dataset
    animals_train_dataset = AnimalClassificationDataset(split="train")
    animals_test_dataset = AnimalClassificationDataset(split="test")
    if args.dataset_type == "trial":
        train_dataset = TrialContrastiveDataset(split="train")
        test_dataset = TrialContrastiveDataset(split="test")
    elif args.dataset_type == "pose":
        train_dataset = PoseContrastiveDataset(
            split="train",
            batch_size=args.batch_size,
        )
        test_dataset = PoseContrastiveDataset(
            split="test",
            batch_size=args.batch_size,
        )
    else:
        raise NotImplementedError(f"Dataset type {args.dataset_type} not implemented")
    print(f"Dataset type: {type(train_dataset)}")
    train_dataloader, test_dataloader = load_dataloaders(train_dataset, test_dataset, args)
    # Initialize the model
    contrastive_head = ContrastiveHead(
        input_dim=512,
        hidden_dim=512,
        output_dim=64,
    ).to(args.device)
    # NOTE: Not sure what shapes this should be
    backbone = Backbone(
        input_dim=args.input_dim, # TBD
        hidden_dim=512,
        output_dim=512,
    ).to(args.device)
    # The backbone output =/= contrastive header here (in terms of dimension)
    # Run trianing
    trainer = SimCLRTrainer(
        train_dataset,
        test_dataset, 
        animals_train_dataset, 
        animals_test_dataset,
        args
    )
    trainer.run_simclr(
        backbone,
        contrastive_head,
        (train_dataloader, test_dataloader),
        args
    )

    wandb.finish()