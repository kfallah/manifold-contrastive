import warnings

import brainscore
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

def generate_brainscore_train_test_split(random_seed=42, split_percentage=0.8):
    # Set numpy seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    # Generates a consistent train test split of the brainscore dataset
    # along the presentation dimension
    neural_data = brainscore.get_assembly(name="dicarlo.MajajHong2015.public")
    neural_data = neural_data.squeeze("time_bin")
    neural_data = neural_data.transpose("presentation", "neuroid")
    # Randomly select 80% of the presentations to be the train set
    train_indices = np.random.choice(
        np.arange(len(neural_data)), size=int(len(neural_data) * split_percentage), replace=False
    )
    # Subtract train_indices to get test indices
    test_indices = np.setdiff1d(np.arange(len(neural_data)), train_indices)
    # Set the data
    train_data = neural_data[train_indices]
    test_data = neural_data[test_indices]

    return train_data, test_data

pose_dims = ['s', 'ty', 'tz', 'rxy', 'rxz', 'ryz']

def get_dataset(average_trials=False, random_seed=0):
    neuroid_train_data, neuroid_test_data = generate_brainscore_train_test_split(random_seed=random_seed)
    if average_trials:
        train_data = neuroid_train_data.multi_groupby(["category_name", "object_name", "stimulus_id"]).mean(
            dim="presentation"
        )
        test_data = neuroid_test_data.multi_groupby(["category_name", "object_name", "stimulus_id"]).mean(
            dim="presentation"
        )
    else:
        train_data, test_data = neuroid_train_data, neuroid_test_data

    v4_train = train_data.sel(region="V4").to_numpy()
    it_train = train_data.sel(region="IT").to_numpy()
    train_category = train_data.category_name.to_numpy()
    _, label_train = np.unique(train_category, return_inverse=True)
    train_object = train_data.object_name.to_numpy()
    _, objectid_train = np.unique(train_object, return_inverse=True)
    pose_train = np.column_stack([train_data[d].values for d in pose_dims])

    v4_test = test_data.sel(region="V4").to_numpy()
    it_test = test_data.sel(region="IT").to_numpy()
    test_category = test_data.category_name.to_numpy()
    _, label_test = np.unique(test_category, return_inverse=True)
    test_object = test_data.object_name.to_numpy()
    _, objectid_test = np.unique(test_object, return_inverse=True)
    pose_test = np.column_stack([test_data[d].values for d in pose_dims])

    v4_train = torch.tensor(v4_train).float()
    v4_test = torch.tensor(v4_test).float()
    label_train = torch.tensor(label_train).long()
    objectid_train = torch.tensor(objectid_train).long()
    pose_train = torch.tensor(pose_train).float()
    it_train = torch.tensor(it_train).float()
    it_test = torch.tensor(it_test).float()
    label_test = torch.tensor(label_test).long()
    objectid_test = torch.tensor(objectid_test).long()
    pose_test = torch.tensor(pose_test).float()

    return (
        (v4_train, it_train, label_train, objectid_train, pose_train),
        (v4_test, it_test, label_test, objectid_test, pose_test)
    )


if __name__ == "__main__":
    get_dataset()
