from datasets import AnimalClassificationDataset
import torch.nn as nn
from evaluation import evaluate_nn_classifier
from argparse import Namespace
import torch

if __name__ == "__main__":
    animals_train_dataset = AnimalClassificationDataset(
        split="train",
        region="V4"
    )
    animals_test_dataset = AnimalClassificationDataset(
        split="test",
        region="V4"
    )
    
    model = nn.Sequential(
        nn.Identity(),
    )

    args = Namespace(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    evaluate_nn_classifier(
        model, 
        animals_train_dataset, 
        animals_test_dataset, 
        args,
        num_epochs=100,
        batch_size=32
    )