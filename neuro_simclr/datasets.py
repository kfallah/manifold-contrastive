import brainscore
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def generate_brainscore_train_test_split(random_seed=42, split_percentage=0.8):
    # Generates a consistent train test split of the brainscore dataset
    # along the presentation dimension
    neural_data = brainscore.get_assembly(name="dicarlo.MajajHong2015.public")
    neural_data = neural_data.squeeze('time_bin')
    neural_data = neural_data.transpose('presentation', 'neuroid')
    # Randomly select 80% of the presentations to be the train set
    train_indices = np.random.choice(
        np.arange(len(neural_data)), 
        size=int(len(neural_data) * split_percentage), 
        replace=False
    )
    # Subtract train_indices to get test indices
    test_indices = np.setdiff1d(
        np.arange(len(neural_data)), 
        train_indices
    )
    # Set the data
    train_data = neural_data[train_indices]
    test_data = neural_data[test_indices]

    return train_data, test_data

# ================== Evaluation datasets ==================

class StimulusIDClassificationDataset(Dataset):
    """
        Dataset to see if you can predict the stimulus id
        from a V4 vector. 
    """

    def __init__(self, split="train"):
        pass

    def load_stimulus_id_classification_dataset(self):
        pass

class AnimalClassificationDataset(Dataset):
    """
        This is the dataset class for the neuroscience dataset
    """
    
    def __init__(self, split="train", region="V4"):
        self.region = region
        if split == "train":
            neural_data, _ = generate_brainscore_train_test_split()
        else:
            _, neural_data = generate_brainscore_train_test_split()

        neuroid_data = neural_data.multi_groupby([
            'category_name', 
            'object_name', 
            'stimulus_id'
        ]).mean(dim='presentation')
        # Use the rest as test
        neuroid_data = neuroid_data.sel(region=self.region)
        # neuroid_data = neuroid_data.squeeze('time_bin')
        # neuroid_data = neuroid_data.transpose('presentation', 'neuroid')
        # Get the indices where the category is animals
        animal_labels = (neuroid_data.coords['category_name'] == 'Animals').to_numpy().astype(int)

        self.neuroid_data = neuroid_data
        self.labels = animal_labels

    def __len__(self):
        # Return the number of stimuli
        return len(self.data)

    def __getitem__(self, index):
        return torch.Tensor(self.neuroid_data[index].to_numpy()), self.labels[index]

class AllClassClassificationDataset(Dataset):
    """
        This is the dataset class for the neuroscience dataset
    """
    
    def __init__(self, split="Train", average_across_stimuli=True):
        if split == "Train":
            neural_data, _ = generate_brainscore_train_test_split()
        else:
            _, neural_data = generate_brainscore_train_test_split()

        neuroid_data = neural_data.multi_groupby([
            'category_name', 
            'object_name', 
            'stimulus_id'
        ]).mean(dim='presentation')
        # Use the rest as test
        neuroid_data = neuroid_data.sel(region='V4')
        # neuroid_data = neuroid_data.squeeze('time_bin')
        # neuroid_data = neuroid_data.transpose('presentation', 'neuroid')
        # Get the indices where the category is animals
        labels = neuroid_data.coords['category_name'].to_numpy()

        self.neuroid_data, self.labels = neuroid_data, labels

    def __len__(self):
        # Return the number of stimuli
        return len(self.data)

    def __getitem__(self, index):
        return torch.Tensor(self.neuroid_data[index].to_numpy()), self.labels[index]


# ================== Contrastive Datasets ==================

class TrialContrastiveDataset(Dataset):

    def __init__(self, split="train", random_seed=42, split_percentage=0.8):
        # Load the data from brainscore and group it
        if split == "train":
            neural_data, _ = generate_brainscore_train_test_split()
        else:
            _, neural_data = generate_brainscore_train_test_split()
 
        neuroid_data = neural_data.sel(region='V4')
        stimulus_id_list = neuroid_data.coords['stimulus_id'].to_numpy()
        self.stimulus_ids = np.unique(stimulus_id_list)
        # load stimulus to v4 map
        self.stimulus_to_v4_map = {}
        for stimulus_id in self.stimulus_ids:
            presentations = neuroid_data.sel(
                stimulus_id=stimulus_id
            )
            # presentations = presentations.squeeze('time_bin')
            # presentations = presentations.transpose('presentation', 'neuroid')
            self.stimulus_to_v4_map[stimulus_id] = torch.Tensor(
                presentations
            )

    def __len__(self):
        # Length is the number of stimuli
        return len(self.stimulus_ids)

    def __getitem__(self, index):
        # For a given stimulus index 
        stimulus_id = self.stimulus_ids[index]
        # Choose two random presentations of that stimulus
        presentations = self.stimulus_to_v4_map[stimulus_id]
        # Choose two random presentations of that stimulus
        if presentations.shape[0] < 2:
            print("Not enough presentations for stimulus_id: ", stimulus_id)
            return self.__getitem__(np.random.randint(len(self)))
        random_indices = np.random.choice(
            len(presentations),
            2,
            replace=False
        )
        item_a = torch.Tensor(presentations[random_indices[0]].to_numpy())
        item_b = torch.Tensor(presentations[random_indices[1]].to_numpy())

        return item_a, item_b

class PoseContrastiveDataset(Dataset):
    """
        Dataset for contrastive learning basedon pose. 
    """

    def __init__(
        self, 
        split="train", 
        random_seed=42, 
        split_percentage=0.8,
        average_trials=False
    ):
        # self.neuroid_data, self.labels = self.data
        # Load the data from brainscore and group it
        if split == "train":
            neural_data, _ = generate_brainscore_train_test_split()
        else:
            _, neural_data = generate_brainscore_train_test_split()

        if average_trials:
            neural_data = neural_data.multi_groupby([
                'category_name', 
                'object_name', 
                'stimulus_id'
            ]).mean(dim='presentation')
        neuroid_data = neural_data.sel(region='V4')
        # Get list of "object_name"
        object_name_list = neuroid_data.coords['object_name'].to_numpy()
        object_name_list = np.unique(object_name_list)
        self.object_names = object_name_list
        # Load up a dictionary of the form object_name -> [presentation1, presentation2, ...]
        self.presentations_map = {} 
        for object_name in self.object_names:
            presentations = neuroid_data.sel(
                object_name=object_name
            )
            # presentations.sel()
            # presentations = presentations.squeeze('time_bin')
            # presentations = presentations.transpose('presentation', 'neuroid')
            self.presentations_map[object_name] = torch.Tensor(
                presentations.to_numpy()
            )

    def __len__(self):
        # Length is the number of stimuli
        return len(self.object_names)

    def __getitem__(self, index):
        # For a given object name index 
        object_name = self.object_names[index]
        # Choose two random presentations of that stimulus
        # presentations = self.neuroid_data.sel(object_name=object_name)
        # presentations = presentations.squeeze('time_bin')
        # presentations = presentations.transpose('presentation', 'neuroid')
        # Choose two random presentations of that stimulus
        random_indices = np.random.choice(
            len(self.presentations_map[object_name]),
            2,
            replace=False
        )
        item_a = self.presentations_map[object_name][random_indices[0]]
        item_b = self.presentations_map[object_name][random_indices[1]]

        return item_a, item_b

if __name__ == "__main__":
    generate_brainscore_train_test_split()