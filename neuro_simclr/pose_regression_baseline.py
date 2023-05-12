"""
    This is the code for a pose regression baseline experiment. 

    Our goal is to evaluate the performance of basic linear
    regression on predicting pose changes from the differences in
    representation between two samples. 
"""
import argparse
import numpy as np
import torch.nn as nn
import tabulate

from datasets import get_dataset, pose_dims
from evaluation import _eval_regression

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose regression baseline experiment")

    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n", type=int, default=2000, help="Maximum number of training pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    args.n = int(args.n)

    def train_test_pair_indices(train, test):
        n = args.n
        i_train_ab = rng.choice(len(train), size=n*2, replace=True)
        i_test_ab = rng.choice(len(test), size=n*2, replace=True)
        return i_train_ab[:n], i_train_ab[n:], \
            i_test_ab[:n], i_test_ab[n:]

    # Load up the datasets
    # Averaged
    avgd_train, avgd_test = get_dataset(average_trials=True, random_seed=args.seed)
    avgd_v4_train, avgd_it_train, avgd_label_train, avgd_objectid_train, avgd_pose_train = avgd_train
    avgd_v4_test, avgd_it_test, avgd_label_test, avgd_objectid_test, avgd_pose_test = avgd_test
    i_avgd_train_a, i_avgd_train_b, i_avgd_test_a, i_avgd_test_b = train_test_pair_indices(avgd_v4_train, avgd_v4_test)
    avgd_posechange_train = avgd_pose_train[i_avgd_train_a] - avgd_pose_train[i_avgd_train_b]
    avgd_posechange_test = avgd_pose_test[i_avgd_test_a] - avgd_pose_test[i_avgd_test_b]

    # Non averaged
    train, test = get_dataset(average_trials=False, random_seed=args.seed)
    v4_train, it_train, label_train, objectid_train, pose_train = train
    v4_test, it_test, label_test, objectid_test, pose_test = test
    i_train_a, i_train_b, i_test_a, i_test_b = train_test_pair_indices(v4_train, v4_test)
    posechange_train = pose_train[i_train_a] - pose_train[i_train_b]
    posechange_test = pose_test[i_test_a] - pose_test[i_test_b]

    # Run the evaluations
    print("Evaluating V4")
    v4_results = _eval_regression(
        v4_train[i_train_a] - v4_train[i_train_b],
        posechange_train,
        v4_test[i_test_a] - v4_test[i_test_b],
        posechange_test,
    )
    print("Evaluating IT")
    it_results = _eval_regression(
        it_train[i_train_a] - it_train[i_train_b],
        posechange_train,
        it_test[i_test_a] - it_test[i_test_b],
        posechange_test,
    )
    print("Evaluating V4 Averaged")
    v4_avgd_results = _eval_regression(
        avgd_v4_train[i_avgd_train_a] - avgd_v4_train[i_avgd_train_b],
        avgd_posechange_train,
        avgd_v4_test[i_avgd_test_a] - avgd_v4_test[i_avgd_test_b],
        avgd_posechange_test,
    )
    print("Evaluating IT Averaged")
    it_avgd_results = _eval_regression(
        avgd_it_train[i_avgd_train_a] - avgd_it_train[i_avgd_train_b],
        avgd_posechange_train,
        avgd_it_test[i_avgd_test_a] - avgd_it_test[i_avgd_test_b],
        avgd_posechange_test,
    )

    # predicting pose directly from individual samples
    print("Evaluating V4 no diff")
    v4_nodiff_results = _eval_regression(
        v4_train,
        pose_train,
        v4_test,
        pose_test,
    )
    print("Evaluating IT no diff")
    it_nodiff_results = _eval_regression(
        it_train,
        pose_train,
        it_test,
        pose_test,
    )
    print("Evaluating V4 Averaged no diff")
    v4_avgd_nodiff_results = _eval_regression(
        avgd_v4_train,
        avgd_pose_train,
        avgd_v4_test,
        avgd_pose_test,
    )
    print("Evaluating IT Averaged no diff")
    it_avgd_nodiff_results = _eval_regression(
        avgd_it_train,
        avgd_pose_train,
        avgd_it_test,
        avgd_pose_test,
    )
    print("\nResults for pose change using pairs:")
    # Make table with tabulate
    pose_dims_r2 = [f'{dim} R2' for dim in pose_dims]
    table = [
        ["Model", "Mean R2", "Median R2", *pose_dims_r2],
        ["V4", *v4_results],
        ["IT", *it_results],
        ["V4 Averaged", *v4_avgd_results],
        ["IT Averaged", *it_avgd_results],
    ]
    print(tabulate.tabulate(
        table, headers="firstrow", tablefmt="github",
        floatfmt=".3f"
    ))

    print("\nResults for predicting pose directly from individual samples:")
    # Make table with tabulate
    pose_dims_r2 = [f'{dim} R2' for dim in pose_dims]
    table = [
        ["Model", "Mean R2", "Median R2", *pose_dims_r2],
        ["V4", *v4_nodiff_results],
        ["IT", *it_nodiff_results],
        ["V4 Averaged", *v4_avgd_nodiff_results],
        ["IT Averaged", *it_avgd_nodiff_results],
    ]
    print(tabulate.tabulate(
        table, headers="firstrow", tablefmt="github",
        floatfmt=".3f"
    ))
