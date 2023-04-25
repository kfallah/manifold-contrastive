import brainscore
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# ================== Evaluation datasets ==================

class AnimalClassificationDataset(Dataset):
    """
        This is the dataset class for the neuroscience dataset
    """
    
    def __init__(self, split="Train"):
        if split == "Train":
            self.data, _ = self.load_animal_classification_dataset()
        else:
            _, self.data = self.load_animal_classification_dataset()

        self.neuroid_data, self.labels = self.data

    def load_animal_classification_dataset(self, split_percentage=0.8, average_across_stimuli=True):
        print("Loading brainscore dataset")
        neural_data = brainscore.get_assembly(name="dicarlo.MajajHong2015.public")
        neuroid_data = neural_data.multi_groupby([
            'category_name', 
            'object_name', 
            'stimulus_id'
        ]).mean(dim='presentation')
        # Use the rest as test
        neuroid_data = neuroid_data.sel(region='V4')
        neuroid_data = neuroid_data.squeeze('time_bin')
        neuroid_data = neuroid_data.transpose('presentation', 'neuroid')
        # Get the indices where the category is animals
        animal_labels = (neuroid_data.coords['category_name'] == 'Animals').to_numpy().astype(int)
        # Split the data into train and test sets
        train_v4, test_v4, train_labels, test_labels = train_test_split(
            neuroid_data, 
            animal_labels, 
            test_size=1-split_percentage, 
            stratify=animal_labels,
            random_state=42 # Seed so the same split is used every time
        )
        
        return (train_v4, train_labels), (test_v4, test_labels)

    def __len__(self):
        # Return the number of stimuli
        return len(self.data)

    def __getitem__(self, index):
        return torch.Tensor(self.neuroid_data[index].to_numpy()), self.labels[index]

class AllClassClassificationDataset(Dataset):
    """
        This is the dataset class for the neuroscience dataset
    """
    
    def __init__(self, split="Train"):
        if split == "Train":
            self.data, _ = self.load_animal_classification_dataset()
        else:
            _, self.data = self.load_animal_classification_dataset()

        self.neuroid_data, self.labels = self.data

    def load_animal_classification_dataset(self, split_percentage=0.8, average_across_stimuli=True):
        print("Loading brainscore dataset")
        neural_data = brainscore.get_assembly(name="dicarlo.MajajHong2015.public")
        neuroid_data = neural_data.multi_groupby([
            'category_name', 
            'object_name', 
            'stimulus_id'
        ]).mean(dim='presentation')
        # Use the rest as test
        neuroid_data = neuroid_data.sel(region='V4')
        neuroid_data = neuroid_data.squeeze('time_bin')
        neuroid_data = neuroid_data.transpose('presentation', 'neuroid')
        # Get the indices where the category is animals
        # animal_labels = (neuroid_data.coords['category_name'] == 'Animals').to_numpy().astype(int)
        labels = neuroid_data.coords['category_name'].to_numpy()
        # Split the data into train and test sets
        train_v4, test_v4, train_labels, test_labels = train_test_split(
            neuroid_data, 
            l_labels, 
            test_size=1-split_percentage, 
            stratify=labels,
            random_state=42 # Seed so the same split is used every time
        )
        
        return (train_v4, train_labels), (test_v4, test_labels)

    def __len__(self):
        # Return the number of stimuli
        return len(self.data)

    def __getitem__(self, index):
        return torch.Tensor(self.neuroid_data[index].to_numpy()), self.labels[index]


# ================== Contrastive Datasets ==================

class TrialContrastiveDataset(Dataset):

    def __init__(self, split="train", random_seed=42, split_percentage=0.8):
        # self.neuroid_data, self.labels = self.data
        # Load the data from brainscore and group it
        print("Loading brainscore dataset")
        neural_data = brainscore.get_assembly(name="dicarlo.MajajHong2015.public")
        neuroid_data = neural_data.sel(region='V4')
        # Get list of "stimulus_id"
        stimulus_id_list = neuroid_data.coords['stimulus_id'].to_numpy()
        np.random.seed(random_seed) # Set seed for consistent splits
        stimulus_id_list = np.unique(stimulus_id_list)
        random_indices = np.random.choice(
            len(stimulus_id_list),
            int(len(stimulus_id_list) * split_percentage),
            replace=False
        )
        train_mask = np.zeros(len(stimulus_id_list), dtype=bool)
        train_mask[random_indices] = True
        test_mask = np.invert(train_mask)
        if split == "train":
            self.stimulus_ids = stimulus_id_list[train_mask]
        else:
            self.stimulus_ids = stimulus_id_list[test_mask]
        self.neuroid_data = neuroid_data

    def __len__(self):
        # Length is the number of stimuli
        return len(self.stimulus_ids)

    def __getitem__(self, index):
        # For a given stimulus index 
        stimulus_id = self.stimulus_ids[index]
        # Choose two random presentations of that stimulus
        presentations = self.neuroid_data.sel(stimulus_id=stimulus_id)
        presentations = presentations.squeeze('time_bin')
        presentations = presentations.transpose('presentation', 'neuroid')
        # Choose two random presentations of that stimulus
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
        batch_size=512
    ):
        self.batch_size = batch_size
        # self.neuroid_data, self.labels = self.data
        # Load the data from brainscore and group it
        print("Loading brainscore dataset")
        neural_data = brainscore.get_assembly(name="dicarlo.MajajHong2015.public")
        neuroid_data = neural_data.sel(region='V4')
        # Get list of "object_name"
        object_name_list = neuroid_data.coords['object_name'].to_numpy()
        np.random.seed(random_seed) # Set seed for consistent splits
        object_name_list = np.unique(object_name_list)
        random_indices = np.random.choice(
            len(object_name_list),
            int(len(object_name_list) * split_percentage),
            replace=False
        )
        train_mask = np.zeros(len(object_name_list), dtype=bool)
        train_mask[random_indices] = True
        test_mask = np.invert(train_mask)
        if split == "train":
            self.object_names = object_name_list[train_mask]
        else:
            self.object_names = object_name_list[test_mask]
        self.neuroid_data = neuroid_data

    def __len__(self):
        # Length is the number of stimuli
        return max(len(self.object_names), self.batch_size)

    def __getitem__(self, index):
        # For a given object name index 
        object_name = self.object_names[index % len(self.object_names)]
        # Choose two random presentations of that stimulus
        presentations = self.neuroid_data.sel(object_name=object_name)
        presentations = presentations.squeeze('time_bin')
        presentations = presentations.transpose('presentation', 'neuroid')
        # Choose two random presentations of that stimulus
        random_indices = np.random.choice(
            len(presentations),
            2,
            replace=False
        )
        item_a = torch.Tensor(presentations[random_indices[0]].to_numpy())
        item_b = torch.Tensor(presentations[random_indices[1]].to_numpy())

        return item_a, item_b
