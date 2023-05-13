"""
    This is the code for a linear readout baseline experiment. 

    Our goal is to evaluate the performance of a basic linear
    binary classifier on predicting whether or not an image is of
    an animal or not using the V4 data. 
"""
import argparse
import torch.nn as nn
import tabulate
import numpy as np
from tqdm import tqdm

from datasets import get_dataset, get_pixel_dataset
from evaluation import evaluate_linear_classifier

def run_linear_evaluation(
    seed,
    args,
    objectid=False,
):
    # Load up the datasets
    # Averaged
    averaged_train, averaged_test = get_dataset(
        average_trials=True, 
        random_seed=args.random_seed,
        ignore_cache=True
    )
    averaged_v4_train, averaged_it_train, averaged_label_train, averaged_objectid_train, averaged_pose_train = averaged_train
    averaged_v4_test, averaged_it_test, averaged_label_test, averaged_objectid_test, averaged_pose_test = averaged_test
    # Non averaged
    train, test = get_dataset(
        average_trials=False, 
        random_seed=args.random_seed
    )
    v4_train, it_train, label_train, objectid_train, pose_train = train
    v4_test, it_test, label_test, objectid_test, pose_test = test
    # Load pixel dataset
    pixel_train, pixel_test = get_pixel_dataset(random_seed=args.random_seed)
    pixel_train, _, _ = pixel_train
    pixel_test, _, _ = pixel_test
    # Choose the right dataset
    if objectid:
        averaged_train_label = averaged_objectid_train
        averaged_test_label = averaged_objectid_test
        train_label = objectid_train
        test_label = objectid_test
    else:
        averaged_train_label = averaged_label_train
        averaged_test_label = averaged_label_test
        train_label = label_train
        test_label = label_test
    # Run each of the evals
    print("Evaluating Pixel")
    pixel_accuracy, pixel_fscore = evaluate_linear_classifier(
        pixel_train,
        averaged_train_label,
        pixel_test,
        averaged_test_label,
        args
    )
    print("Evaluating V4")
    v4_accuracy, v4_fscore = evaluate_linear_classifier(
        v4_train,
        train_label,
        v4_test,
        test_label,
        args
    )
    print("Evaluating IT")
    it_accuracy, it_fscore = evaluate_linear_classifier(
        it_train,
        train_label,
        it_test,
        test_label,
        args
    )
    print("Evaluating V4 Averaged")
    v4_averaged_accuracy, v4_averaged_fscore = evaluate_linear_classifier(
        averaged_v4_train,
        averaged_train_label,
        averaged_v4_test,
        averaged_test_label,
        args
    ) 
    print("Evaluating IT Averaged")
    it_averaged_accuracy, it_averaged_fscore = evaluate_linear_classifier(
        averaged_it_train,
        averaged_train_label,
        averaged_it_test,
        averaged_test_label,
        args
    )
    # Return the results
    return (v4_accuracy, v4_fscore, it_accuracy, it_fscore, pixel_accuracy, pixel_fscore, v4_averaged_accuracy, v4_averaged_fscore, it_averaged_accuracy, it_averaged_fscore)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear readout baseline experiment")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_trials", type=int, default=2)

    args = parser.parse_args()

    model = nn.Sequential(
        nn.Identity(),
    )
    # Make a set of random seeds
    category_scores = []
    object_id_scores = []
    seeds = np.arange(args.num_trials)
    for trial_index in tqdm(range(args.num_trials)):
        trial_seed = seeds[trial_index]
        
        # Run the linear evaluation
        cat_scores = run_linear_evaluation(
            trial_seed,
            args,
            objectid=False,
        )
        category_scores.append(cat_scores)
        obj_id_scores = run_linear_evaluation(
            trial_seed,
            args,
            objectid=True,
        )
        object_id_scores.append(obj_id_scores)
    # Compute the mean and std of the scores
    category_scores = np.array(category_scores)
    object_id_scores = np.array(object_id_scores)

    v4_accuracy_mean, v4_accuracy_std = np.mean(category_scores[:, 0]), np.std(category_scores[:, 0])
    v4_fscore_mean, v4_fscore_std = np.mean(category_scores[:, 1]), np.std(category_scores[:, 1])
    it_accuracy_mean, it_accuracy_std = np.mean(category_scores[:, 2]), np.std(category_scores[:, 2])
    it_fscore_mean, it_fscore_std = np.mean(category_scores[:, 3]), np.std(category_scores[:, 3])
    pixel_accuracy_mean, pixel_accuracy_std = np.mean(category_scores[:, 4]), np.std(category_scores[:, 4])
    pixel_fscore_mean, pixel_fscore_std = np.mean(category_scores[:, 5]), np.std(category_scores[:, 5])
    v4_averaged_accuracy_mean, v4_averaged_accuracy_std = np.mean(category_scores[:, 6]), np.std(category_scores[:, 6])
    v4_averaged_fscore_mean, v4_averaged_fscore_std = np.mean(category_scores[:, 7]), np.std(category_scores[:, 7])
    it_averaged_accuracy_mean, it_averaged_accuracy_std = np.mean(category_scores[:, 8]), np.std(category_scores[:, 8])
    it_averaged_fscore_mean, it_averaged_fscore_std = np.mean(category_scores[:, 9]), np.std(category_scores[:, 9])
    # Make table with tabulate
    print("Category Evaluation Performance")
    table = [
        ["Model", "Accuracy", "F1 Score"],
        ["V4", f"{v4_accuracy_mean} +- {v4_accuracy_std}", f"{v4_fscore_mean} +- {v4_fscore_std}"],
        ["IT", f"{it_accuracy_mean} +- {it_accuracy_std}", f"{it_fscore_mean} +- {it_fscore_std}"],
        ["Pixel", f"{pixel_accuracy_mean} +- {pixel_accuracy_std}", f"{pixel_fscore_mean} +- {pixel_fscore_std}"],
        ["V4 Averaged", f"{v4_averaged_accuracy_mean} +- {v4_averaged_accuracy_std}", f"{v4_averaged_fscore_mean} +- {v4_averaged_fscore_std}"],
        ["IT Averaged", f"{it_averaged_accuracy_mean} +- {it_averaged_accuracy_std}", f"{it_averaged_fscore_mean} +- {it_averaged_fscore_std}"],
    ]
    print(tabulate.tabulate(table, headers="firstrow", tablefmt="github"))
    print("Object ID Evaluation Performance")
    v4_accuracy_mean, v4_accuracy_std = np.mean(object_id_scores[:, 0]), np.std(object_id_scores[:, 0])
    v4_fscore_mean, v4_fscore_std = np.mean(object_id_scores[:, 1]), np.std(object_id_scores[:, 1])
    it_accuracy_mean, it_accuracy_std = np.mean(object_id_scores[:, 2]), np.std(object_id_scores[:, 2])
    it_fscore_mean, it_fscore_std = np.mean(object_id_scores[:, 3]), np.std(object_id_scores[:, 3])
    pixel_accuracy_mean, pixel_accuracy_std = np.mean(object_id_scores[:, 4]), np.std(object_id_scores[:, 4])
    pixel_fscore_mean, pixel_fscore_std = np.mean(object_id_scores[:, 5]), np.std(object_id_scores[:, 5])
    v4_averaged_accuracy_mean, v4_averaged_accuracy_std = np.mean(object_id_scores[:, 6]), np.std(object_id_scores[:, 6])
    v4_averaged_fscore_mean, v4_averaged_fscore_std = np.mean(object_id_scores[:, 7]), np.std(object_id_scores[:, 7])
    it_averaged_accuracy_mean, it_averaged_accuracy_std = np.mean(object_id_scores[:, 8]), np.std(object_id_scores[:, 8])
    it_averaged_fscore_mean, it_averaged_fscore_std = np.mean(object_id_scores[:, 9]), np.std(object_id_scores[:, 9])

    table = [
        ["Model", "Accuracy", "F1 Score"],
        ["V4", f"{v4_accuracy_mean} +- {v4_accuracy_std}", f"{v4_fscore_mean} +- {v4_fscore_std}"],
        ["IT", f"{it_accuracy_mean} +- {it_accuracy_std}", f"{it_fscore_mean} +- {it_fscore_std}"],
        ["Pixel", f"{pixel_accuracy_mean} +- {pixel_accuracy_std}", f"{pixel_fscore_mean} +- {pixel_fscore_std}"],
        ["V4 Averaged", f"{v4_averaged_accuracy_mean} +- {v4_averaged_accuracy_std}", f"{v4_averaged_fscore_mean} +- {v4_averaged_fscore_std}"],
        ["IT Averaged", f"{it_averaged_accuracy_mean} +- {it_averaged_accuracy_std}", f"{it_averaged_fscore_mean} +- {it_averaged_fscore_std}"],
    ]
    print(tabulate.tabulate(table, headers="firstrow", tablefmt="github"))