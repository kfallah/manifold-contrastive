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
from evaluation import _eval_pose_regression_from_diffs

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
        n = np.min([len(train)//2, args.n])
        i_train_ab = rng.permutation(len(train))[:n*2]
        even_test_len = len(test) - len(test) % 2
        i_test_ab = rng.permutation(even_test_len)
        return i_train_ab[:n], i_train_ab[n:], \
            i_test_ab[:even_test_len//2], i_test_ab[even_test_len//2:]

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
    v4_results = _eval_pose_regression_from_diffs(
        v4_train[i_train_a] - v4_train[i_train_b],
        posechange_train,
        v4_test[i_test_a] - v4_test[i_test_b],
        posechange_test,
    )
    # print("Evaluating IT")
    # it_accuracy, it_fscore = evaluate_linear_classifier(
    #     it_train,
    #     label_train,
    #     it_test,
    #     label_test,
    #     args
    # )
    print("Evaluating V4 Averaged")
    v4_avgd_results = _eval_pose_regression_from_diffs(
        avgd_v4_train[i_avgd_train_a] - avgd_v4_train[i_avgd_train_b],
        avgd_posechange_train,
        avgd_v4_test[i_avgd_test_a] - avgd_v4_test[i_avgd_test_b],
        avgd_posechange_test,
    )
    # v4_averaged_accuracy, v4_averaged_fscore = evaluate_linear_classifier(
    #     averaged_v4_train,
    #     averaged_label_train,
    #     averaged_v4_test,
    #     averaged_label_test,
    #     args
    # ) 
    # print("Evaluating IT Averaged")
    # it_averaged_accuracy, it_averaged_fscore = evaluate_linear_classifier(
    #     averaged_it_train,
    #     averaged_label_train,
    #     averaged_it_test,
    #     averaged_label_test,
    #     args
    # )
    # Make table with tabulate
    pose_dims_r2 = [f'{dim} R2' for dim in pose_dims]
    table = [
        ["Model", "Total R2", "Median R2", *pose_dims_r2],
        ["V4", v4_results[0], v4_results[1], *v4_results[2]],
        # ["IT", it_accuracy, it_fscore],
        ["V4 Averaged", v4_avgd_results[0], v4_avgd_results[1], *v4_avgd_results[2]],
        # ["IT Averaged", it_averaged_accuracy, it_averaged_fscore],
    ]
    print(tabulate.tabulate(table, headers="firstrow", tablefmt="github"))
