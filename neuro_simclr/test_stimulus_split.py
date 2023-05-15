import brainscore 
import numpy as np
import torch
from evaluation import evaluate_linear_classifier
import argparse
import xarray

pose_dims = ['s', 'ty', 'tz', 'rxy', 'rxz', 'ryz']

def get_stimulus_dataset_split(random_seed):
    # Load the train and test datasets split by stimuli
    # Set numpy seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    # Generates a consistent train test split of the brainscore dataset
    # along the presentation dimension
    neural_data = brainscore.get_assembly(name="dicarlo.MajajHong2015.public")
    print(neural_data)
    neural_data = neural_data.squeeze("time_bin")
    neural_data = neural_data.transpose("presentation", "neuroid")
    neural_data = neural_data.multi_groupby(["category_name", "object_name", "stimulus_id", *pose_dims]).mean(
        dim="presentation",
        keep_attrs=True
    )
    # Get the list of stimuli
    v4_data_tensor = neural_data.sel(region="V4").to_numpy()
    it_data_tensor = neural_data.sel(region="IT").to_numpy()
    indices = np.arange(len(neural_data))
    train_indices = np.random.choice(indices, size=int(len(indices) * 0.8), replace=False)
    test_indices = np.setdiff1d(indices, train_indices)
    object_val = neural_data.object_name.to_numpy()
    _, objectids = np.unique(object_val, return_inverse=True)
    pose = np.column_stack([neural_data[d].values for d in pose_dims])
    categories = neural_data.category_name.to_numpy()
    _, categories = np.unique(categories, return_inverse=True)
    # Get the test data
    test_v4_data = torch.Tensor(v4_data_tensor[test_indices])
    test_it_data = torch.Tensor(it_data_tensor[test_indices])
    test_categories = torch.Tensor(categories[test_indices]).long()
    test_objectids = torch.Tensor(objectids[test_indices]).long()
    test_pose = pose[test_indices]
    print(f"Test data shape: {test_v4_data.shape}")
    test_data = (test_v4_data, test_it_data, test_categories, test_objectids, test_pose)
    # Get the train data 
    train_v4_data = torch.Tensor(v4_data_tensor[train_indices])
    train_it_data = torch.Tensor(it_data_tensor[train_indices])
    train_categories = torch.Tensor(categories[train_indices]).long()
    train_objectids = torch.Tensor(objectids[train_indices]).long()
    print(f"Train data shape: {train_v4_data.shape}")
    train_pose = pose[train_indices]
    train_data = (train_v4_data, train_it_data, train_categories, train_objectids, train_pose)

    return train_data, test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear readout baseline experiment")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    for random_seed in [1, 2, 3, 4, 5]:
        train_data, test_data = get_stimulus_dataset_split(random_seed)
        v4_train_data, it_train_data, category_train_data, object_train_data, pose_train_data = train_data
        v4_test_data, it_test_data, category_test_data, object_test_data, pose_test_data = test_data
        # Run linear evaluation on category level with this split
        v4_accuracy, v4_fscore = evaluate_linear_classifier(
            v4_train_data,
            category_train_data,
            v4_test_data,
            category_test_data,
            args
        )
        it_accuracy, it_fscore = evaluate_linear_classifier(
            it_train_data,
            category_train_data,
            it_test_data,
            category_test_data,
            args
        )
        print("Category Level Evaluation")
        print("IT Accuracy: ", it_accuracy)
        print("IT F1 Score: ", it_fscore)
        print("V4 Accuracy: ", v4_accuracy)
        print("V4 F1 Score: ", v4_fscore)
        # Run linear evaluation on object level with this split
        v4_accuracy, v4_fscore = evaluate_linear_classifier(
            v4_train_data,
            object_train_data,
            v4_test_data,
            object_test_data,
            args
        )
        it_accuracy, it_fscore = evaluate_linear_classifier(
            it_train_data,
            object_train_data,
            it_test_data,
            object_test_data,
            args
        )
        print("Object Level Evaluation")
        print("IT Accuracy: ", it_accuracy)
        print("IT F1 Score: ", it_fscore)
        print("V4 Accuracy: ", v4_accuracy)
        print("V4 F1 Score: ", v4_fscore)
