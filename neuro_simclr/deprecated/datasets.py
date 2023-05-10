"""
    This is the deprecated code for loading datasets in various ways. 
"""
# ================== Evaluation datasets ==================

# Code for loading up the split that is used for the rest of the dataset objects
# neuroid_train_data, neuroid_test_data = generate_brainscore_train_test_split()

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
        self.num_classes = 2
        self.dataset_name = "animal_binary"
        self.region = region
        if split == "train":
            neural_data = neuroid_train_data
        else:
            neural_data = neuroid_test_data

        neuroid_data = neural_data.multi_groupby(["category_name", "object_name", "stimulus_id"]).mean(
            dim="presentation"
        )
        # Use the rest as test
        neuroid_data = neuroid_data.sel(region=self.region)
        # neuroid_data = neuroid_data.squeeze('time_bin')
        # neuroid_data = neuroid_data.transpose('presentation', 'neuroid')
        # Get the indices where the category is animals
        animal_labels = (neuroid_data.coords["category_name"] == "Animals").to_numpy().astype(int)

        self.neuroid_data = neuroid_data
        self.labels = animal_labels

        self.tensor_data = torch.Tensor(self.neuroid_data.to_numpy())
        # self.tensor_data = []
        # for index in tqdm(range(len(self.neuroid_data))):
        #     self.tensor_data.append(
        #         torch.Tensor(self.neuroid_data[index].to_numpy())
        #     )

    def __len__(self):
        # Return the number of stimuli
        return len(self.tensor_data)

    def __getitem__(self, index):
        return self.tensor_data[index], self.labels[index]


class AllCategoryClassificationDataset(Dataset):
    """
    This is the dataset class for the neuroscience dataset
    """

    def __init__(self, split="train", average_presentations=False, region="V4"):
        self.dataset_name = "all_category"
        self.num_classes = 8
        if split == "train":
            neural_data = neuroid_train_data
        else:
            neural_data = neuroid_test_data

        if average_presentations
            neuroid_data = neural_data.multi_groupby(["category_name", "object_name", "stimulus_id"]).mean(
                dim="presentation"
            )
        else:
            neuroid_data = neural_data
        # Use the rest as test
        neuroid_data = neuroid_data.sel(region="V4")
        # neuroid_data = neuroid_data.squeeze('time_bin')
        # neuroid_data = neuroid_data.transpose('presentation', 'neuroid')
        # Get the indices where the category is animals
        labels = neuroid_data.coords["category_name"].to_numpy()
        max_class_index = np.unique(labels)
        self.labels_to_index = {label: index for index, label in enumerate(np.unique(labels))}

        self.neuroid_data, self.labels = neuroid_data, labels
        self.tensor_data = torch.Tensor(self.neuroid_data.to_numpy())
        # self.tensor_data = []
        # for index in range(self.neuroid_data):
        #     self.tensor_data.append(
        #         torch.Tensor(self.neuroid_data[index].to_numpy())
        #     )

    def __len__(self):
        # Return the number of stimuli
        return len(self.tensor_data)

    def __getitem__(self, index):
        return self.tensor_data[index], self.labels_to_index[self.labels[index]]


# ================== Contrastive Datasets ==================


class TrialContrastiveDataset(Dataset):
    def __init__(self, split="train", random_seed=42, split_percentage=0.8):
        # Load the data from brainscore and group it
        if split == "train":
            neural_data = neuroid_train_data
        else:
            neural_data = neuroid_test_data
        neuroid_data = neural_data.sel(region="V4")
        stimulus_id_list = neuroid_data.coords["stimulus_id"].to_numpy()
        self.stimulus_ids = np.unique(stimulus_id_list)
        # store indices of presentations for each stimulus
        self.stimulus_to_presentations_idx_map = {}
        self.data = torch.Tensor(neuroid_data.values)
        for stimulus_id in self.stimulus_ids:
            presentations_idx = (neuroid_data.stimulus_id == stimulus_id).values
            presentations_idx = np.argwhere(presentations_idx).reshape((-1,))
            self.stimulus_to_presentations_idx_map[stimulus_id] = presentations_idx

    def __len__(self):
        # Length is the number of stimuli
        return len(self.stimulus_ids)

    def __getitem__(self, index):
        # For a given stimulus index
        stimulus_id = self.stimulus_ids[index]
        # get indices of presentations of that stimulus
        presentations_idx = self.stimulus_to_presentations_idx_map[stimulus_id]
        if len(presentations_idx) < 2:
            return self.__getitem__(np.random.randint(len(self)))

        # Choose two random presentations of that stimulus
        random_indices = np.random.choice(presentations_idx, 2, replace=False)
        item_a = self.data[random_indices[0]]
        item_b = self.data[random_indices[1]]

        return item_a, item_b


class NewPoseContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning basedon pose.
    """

    def __init__(self, split="train", random_seed=42, split_percentage=0.8, average_trials=False):
        # self.neuroid_data, self.labels = self.data
        # Load the data from brainscore and group it
        if split == "train":
            neural_data = neuroid_train_data
        else:
            neural_data = neuroid_test_data

        if average_trials:
            neural_data = (
                neuroid_train_data.multi_groupby(["category_name", "object_name", "stimulus_id"])
                .mean(dim="presentation")
                .sel(region="V4")
                .to_numpy()
            )
        neuroid_data = neural_data.sel(region="V4")
        # Get list of "object_name"
        object_name_list = neuroid_data.coords["object_name"].to_numpy()
        object_name_list = np.unique(object_name_list)
        self.object_names = object_name_list
        # Load up a dictionary of the form object_name -> [presentation1, presentation2, ...]
        self.presentations = []
        self.labels = []
        for idx, object_name in enumerate(self.object_names):
            obj_present = neuroid_data.sel(object_name=object_name)
            # presentations.sel()
            # presentations = presentations.squeeze('time_bin')
            # presentations = presentations.transpose('presentation', 'neuroid')
            self.presentations.append(torch.tensor(obj_present.to_numpy()).float())
            self.labels.append((torch.ones(len(obj_present)) * idx).long())
        self.presentations = torch.cat(self.presentations)
        self.labels = torch.cat(self.labels)

    def __len__(self):
        # Length is the number of stimuli
        return len(self.presentations)

    def __getitem__(self, index):
        # For a given object name index
        x0 = self.presentations[index]
        label = self.labels[index]
        idx = torch.where(self.labels == label)[0]
        rand_neighbor = np.random.choice(idx, 1, replace=False)[0]
        x1 = self.presentations[rand_neighbor]
        return x0, x1


class PoseContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning basedon pose.
    """

    def __init__(self, split="train", random_seed=42, split_percentage=0.8, average_trials=False):
        # self.neuroid_data, self.labels = self.data
        # Load the data from brainscore and group it
        if split == "train":
            neural_data = neuroid_train_data
        else:
            neural_data = neuroid_test_data

        if average_trials:
            neural_data = neural_data.multi_groupby(["category_name", "object_name", "stimulus_id"]).mean(
                dim="presentation"
            )
        neuroid_data = neural_data.sel(region="V4")
        # Get list of "object_name"
        object_name_list = neuroid_data.coords["object_name"].to_numpy()
        object_name_list = np.unique(object_name_list)
        self.object_names = object_name_list
        # Load up a dictionary of the form object_name -> [presentation1, presentation2, ...]
        self.presentations_map = {}
        for object_name in self.object_names:
            presentations = neuroid_data.sel(object_name=object_name)
            # presentations.sel()
            # presentations = presentations.squeeze('time_bin')
            # presentations = presentations.transpose('presentation', 'neuroid')
            self.presentations_map[object_name] = torch.Tensor(presentations.to_numpy())

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
        random_indices = np.random.choice(len(self.presentations_map[object_name]), 2, replace=False)
        item_a = self.presentations_map[object_name][random_indices[0]]
        item_b = self.presentations_map[object_name][random_indices[1]]

        return item_a, item_b


if __name__ == "__main__":
    # generate_brainscore_train_test_split()
    train_dataset = TrialContrastiveDataset(split="train")
