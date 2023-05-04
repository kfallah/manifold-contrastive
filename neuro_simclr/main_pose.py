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
from evaluation import evaluate_linear_readout, evaluate_nn_classifier, evaluate_IT_explained_variance
from datasets import generate_brainscore_train_test_split

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

# ======================== Training code ========================

class SimCLRTrainer():

    def __init__(
        self, 
        neuroid_train_dataset, 
        neuroid_test_dataset, 
        train_dataset, 
        test_dataset, 
        animals_train_dataset, 
        animals_test_dataset, 
        args
    ):
        self.neuroid_train_dataset = neuroid_train_dataset
        self.neuroid_test_dataset = neuroid_test_dataset
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
            optim = torch.optim.Adam(
                list(backbone.parameters()) + list(contrastive_head.parameters()),
                lr=args.learning_rate,
            )
        elif args.optimizer == "sgd":
            optim = torch.optim.SGD(
                list(backbone.parameters()) + list(contrastive_head.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")
        # Run the trianing
        iteration_bar = tqdm(range(args.num_iterations))
        for iteration in iteration_bar:
            iteration_bar.set_description(f"Iteration {iteration}")
            # Load a batch
            random_indices = np.random.choice(
                len(self.train_dataset),
                size=args.batch_size,
            )
            train_batch = [self.train_dataset[i] for i in random_indices]
            train_batch = tuple(map(torch.stack, zip(*train_batch)))
            item_a = train_batch[0].to(args.device)
            item_b = train_batch[1].to(args.device)
            # Encode features using the backbone
            a_contrast_out = contrastive_head(backbone(item_a))
            b_contrast_out = contrastive_head(backbone(item_b))
            # Compute the info nce loss in the output space
            train_loss = self.nxent_loss(
                a_contrast_out,
                b_contrast_out
            )
            # Backprop
            optim.zero_grad()
            train_loss.backward() 
            optim.step()
            # Run testing
            # Load a batch
            random_indices = np.random.choice(
                len(self.test_dataset),
                size=args.batch_size,
            )
            test_batch = [self.test_dataset[i] for i in random_indices]
            test_batch = tuple(map(torch.stack, zip(*test_batch)))
            item_a = test_batch[0].to(args.device)
            item_b = test_batch[1].to(args.device)
            # Encode features using the backbone
            a_contrast_out = contrastive_head(backbone(item_a))
            b_contrast_out = contrastive_head(backbone(item_b))
            # Compute the info nce loss in the output space
            test_loss = self.nxent_loss(
                a_contrast_out,
                b_contrast_out
            )
            # Run the linear readout evaluation
            # TODO: Maybe change this from every epoch to every num iterations
            if iteration % args.eval_readout_frequency == 0:
                accuracy, fscore = evaluate_linear_readout(
                    backbone,
                    self.animals_train_dataset, 
                    self.animals_test_dataset, 
                    args
                )
                evaluate_nn_classifier(
                    backbone,
                    self.animals_train_dataset,
                    self.animals_test_dataset,
                    args
                )
                evaluate_IT_explained_variance(
                    backbone,
                    self.neuroid_train_dataset,
                    self.neuroid_test_dataset,
                    args
                )
                wandb.log({
                    "iteration": iteration,
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
    parser.add_argument("--num_iterations", type=int, default=2000, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    # Data settings
    parser.add_argument("--dataset_type", type=str, default="pose", help="Type of dataset used")
    parser.add_argument("--average_trials", type=bool, default=True, help="Whether or not to average across trials")

    args = parser.parse_args()
    
    print("Running simclr experiment")
    # Initialize wandb
    wandb.init(
        project="neuro_simclr",
        entity="helblazer811",
        config=args,
    )
    # Load simclr dataset
    neuroid_train_dataset, neuroid_eval_dataset = generate_brainscore_train_test_split()
    animals_train_dataset = AnimalClassificationDataset(split="train")
    animals_test_dataset = AnimalClassificationDataset(split="test")
    if args.dataset_type == "trial":
        train_dataset = TrialContrastiveDataset(
            split="train"
        )
        test_dataset = TrialContrastiveDataset(split="test")
    elif args.dataset_type == "pose":
        train_dataset = PoseContrastiveDataset(
            split="train",
            average_trials=args.average_trials,
        )
        test_dataset = PoseContrastiveDataset(
            split="test",
            average_trials=args.average_trials,
        )
    else:
        raise NotImplementedError(f"Dataset type {args.dataset_type} not implemented")
    print(f"Dataset type: {type(train_dataset)}")
    # train_dataloader, test_dataloader = load_dataloaders(train_dataset, test_dataset, args)
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
        neuroid_train_dataset,
        neuroid_eval_dataset,
        train_dataset,
        test_dataset, 
        animals_train_dataset, 
        animals_test_dataset,
        args
    )
    trainer.run_simclr(
        backbone,
        contrastive_head,
        # (train_dataloader, test_dataloader),
        args
    )

    wandb.finish()