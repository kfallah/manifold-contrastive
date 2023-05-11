"""
    This is the code for a linear readout baseline experiment. 

    Our goal is to evaluate the performance of a basic linear
    binary classifier on predicting whether or not an image is of
    an animal or not using the V4 data. 
"""
import argparse
import torch.nn as nn
import tabulate

from datasets import get_dataset
from evaluation import evaluate_linear_classifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear readout baseline experiment")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    model = nn.Sequential(
        nn.Identity(),
    )

    # Load up the datasets
    # Averaged
    averaged_train, averaged_test = get_dataset(average_trials=True)
    averaged_v4_train, averaged_it_train, averaged_label_train, averaged_objectid_train, averaged_pose_train = averaged_train
    averaged_v4_test, averaged_it_test, averaged_label_test, averaged_objectid_test, averaged_pose_test = averaged_test
    # Non averaged
    train, test = get_dataset(average_trials=False)
    v4_train, it_train, label_train, objectid_train, pose_train = train
    v4_test, it_test, label_test, objectid_test, pose_test = test
    # Run the linear evaluation
    print("Evaluating V4")
    v4_accuracy, v4_fscore = evaluate_linear_classifier(
        v4_train,
        label_train,
        v4_test,
        label_test,
        args
    )
    print("Evaluating IT")
    it_accuracy, it_fscore = evaluate_linear_classifier(
        it_train,
        label_train,
        it_test,
        label_test,
        args
    )
    print("Evaluating V4 Averaged")
    v4_averaged_accuracy, v4_averaged_fscore = evaluate_linear_classifier(
        averaged_v4_train,
        averaged_label_train,
        averaged_v4_test,
        averaged_label_test,
        args
    ) 
    print("Evaluating IT Averaged")
    it_averaged_accuracy, it_averaged_fscore = evaluate_linear_classifier(
        averaged_it_train,
        averaged_label_train,
        averaged_it_test,
        averaged_label_test,
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
