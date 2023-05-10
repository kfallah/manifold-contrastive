"""
    This is the code for a linear readout baseline experiment. 

    Our goal is to evaluate the performance of a basic linear
    binary classifier on predicting whether or not an image is of
    an animal or not using the V4 data. 
"""
import argparse
import torch.nn as nn
import tabulate

from datasets import AllCategoryClassificationDataset
from evaluation import evaluate_linear_classifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear readout baseline experiment")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    model = nn.Sequential(
        nn.Identity(),
    )

    print("Evaluating V4")
    v4_accuracy, v4_fscore = evaluate_linear_classifier(
        model,
        AllCategoryClassificationDataset(
            split="train",
            region="V4",
        ),
        AllCategoryClassificationDataset(
            split="test",
            region="V4",
        ),
        args
    )
    print("Evaluating IT")
    it_accuracy, it_fscore = evaluate_linear_classifier(
        model,
        AllCategoryClassificationDataset(
            split="train",
            region="IT",
        ),
        AllCategoryClassificationDataset(
            split="test",
            region="IT",
        ),
        args
    )
    print("Evaluating V4 Averaged")
    v4_averaged_accuracy, v4_averaged_fscore = evaluate_linear_classifier(
        model,
        AllCategoryClassificationDataset(
            split="train",
            region="V4",
            average_presentations=True,
        ),
        AllCategoryClassificationDataset(
            split="test",
            region="V4",
            average_presentations=True,
        ),
        args
    )
    print("Evaluating IT Averaged")
    it_averaged_accuracy, it_averaged_fscore = evaluate_linear_classifier(
        model,
        AllCategoryClassificationDataset(
            split="train",
            region="IT",
            average_presentations=True,
        ),
        AllCategoryClassificationDataset(
            split="test",
            region="IT",
            average_presentations=True,
        ),
        args
    )

    # Make table with tabulate
    table = [
        ["Model", "Accuracy", "F1 Score"],
        ["V4", v4_accuracy, v4_fscore],
        ["IT", it_accuracy, it_fscore],
        ["V4 Averaged", v4_averaged_accuracy, v4_averaged_fscore],
        ["IT Averaged", it_averaged_accuracy, it_averaged_fscore],
    ]
    print(tabulate.tabulate(table, headers="firstrow", tablefmt="github"))
