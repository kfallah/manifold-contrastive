import warnings

import brainscore
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import xarray

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

def get_pixel_dataset(
    flatten_images=True, 
    random_seed=0,
    output_resolution=32
):
    """
        Returns images and labelsfrom the dicarlo dataset
        corresponding to the splits outputted by 
        generate_brainscore_train_test_split
    """
    neuroid_train_data, neuroid_test_data = generate_brainscore_train_test_split(random_seed=random_seed)
    # Get the train and test data
    # NOTE: Always average over trials because there is just one image per stimulus (image is the stimulus)
    train_stimulus_set = neuroid_train_data.attrs['stimulus_set']
    train_data = neuroid_train_data.multi_groupby(["category_name", "object_name", "stimulus_id"]).mean(
        dim="presentation"
    )
    test_stimulus_set = neuroid_test_data.attrs['stimulus_set']
    test_data = neuroid_test_data.multi_groupby(["category_name", "object_name", "stimulus_id"]).mean(
        dim="presentation"
    )
    # Get the images
    # Helper function for transforming stimulus
    def transform_stimulus(stimulus_path):
        img = Image.open(stimulus_path)
        img = transforms.Resize((output_resolution, output_resolution))(img)
        img = transforms.ToTensor()(img)
        if flatten_images:
            img = img.flatten()
        return img
    # NOTE this may be very slow and we may want to then cache it. 
    # Load the train images
    train_stimulus_ids = train_data.stimulus_id.to_numpy()
    pixel_train = torch.stack([
        transform_stimulus(train_stimulus_set.get_stimulus(stimulus_id))
        for stimulus_id in train_stimulus_ids
    ])
    # Load test images
    test_stimulus_ids = test_data.stimulus_id.to_numpy()
    pixel_test = torch.stack([
        transform_stimulus(test_stimulus_set.get_stimulus(stimulus_id))
        for stimulus_id in test_stimulus_ids
    ])
    # Get the category labels 
    train_category = train_data.category_name.to_numpy()
    _, label_train = np.unique(train_category, return_inverse=True)

    test_category = test_data.category_name.to_numpy()
    _, label_test = np.unique(test_category, return_inverse=True)

    label_train = torch.tensor(label_train).long()
    label_test = torch.tensor(label_test).long()
    # Load object ids
    train_object = train_data.object_name.to_numpy()
    _, object_id_train = np.unique(train_object, return_inverse=True)
    object_id_train = torch.tensor(object_id_train).long()

    test_object = test_data.object_name.to_numpy()
    _, object_id_test = np.unique(test_object, return_inverse=True)
    object_id_test = torch.tensor(object_id_test).long()

    return (pixel_train, label_train, object_id_train), (pixel_test, label_test, object_id_test)

def load_cache(average_trials, ds_factor):
    """
        Loads the cache if it exists
    """
    if average_trials:
        cache_file = f"avgd-{ds_factor}-data.pt"
    else:
        cache_file = "data.pt"
    try:
        return torch.load(cache_file)
    except FileNotFoundError:
        return None

pose_dims = ['s', 'ty', 'tz', 'rxy', 'rxz', 'ryz']

def apply_averaging(neuroid_data, average_downsample_factor=5):
    v4_datas = []
    it_datas = []
    objectid_datas = []
    pose_datas = []
    category_datas = []
    # Convert data to numpy
    v4_data = neuroid_data.sel(region="V4").to_numpy()
    it_data = neuroid_data.sel(region="IT").to_numpy()
    object_val = neuroid_data.object_name.to_numpy()
    _, objectids = np.unique(object_val, return_inverse=True)
    pose = np.column_stack([neuroid_data[d].values for d in pose_dims])
    categories = neuroid_data.category_name.to_numpy()
    _, categories = np.unique(categories, return_inverse=True)
    stimulus_ids = neuroid_data.stimulus_id.to_numpy()
    _, stimulus_ids = np.unique(stimulus_ids, return_inverse=True)
    # For each unique stimulus id
    unique_stimulus_ids = np.unique(stimulus_ids)
    for stimulus_id in tqdm(unique_stimulus_ids):
        # Select the indices of presentations of the same stimulus
        stimulus_indices = np.where(stimulus_ids == stimulus_id)[0]
        num_trials = len(stimulus_indices)
        # Get the pose, objectid, and category
        pose_val = pose[stimulus_indices[0]]
        objectid = objectids[stimulus_indices[0]]
        category = categories[stimulus_indices[0]]
        # Randomly assign each trial a bin
        if average_downsample_factor == 1:
            # In this case we are doing no averaging so assign each trial to its own bin
            bin_assignments = np.arange(num_trials)
            num_bins = num_trials
        elif average_downsample_factor >= 50:
            # In this case assign every trial to the same bin
            bin_assignments = np.zeros(num_trials)
            num_bins = 1
        else:
            # NOTE there is some stochasticity to bin sizes with randint
            num_bins = max(num_trials // average_downsample_factor, 1)
            bin_assignments = np.random.randint(num_bins, size=num_trials)
        # Average over each bin
        for bin_index in range(num_bins):
            # Get the indices for this bin
            bin_indices = np.where(bin_assignments == bin_index)[0] 
            # Skip if there are no indices
            if len(bin_indices) == 0:
                continue
            # Average over the bin for each region
            bin_average_v4 = v4_data[stimulus_indices[bin_indices]].mean(axis=0)
            bin_average_it = it_data[stimulus_indices[bin_indices]].mean(axis=0)
            assert np.isfinite(bin_average_v4).all(), (bin_average_v4, v4_data[stimulus_indices[bin_indices]], num_bins)
            # Append the v4 and it data
            v4_datas.append(bin_average_v4)
            it_datas.append(bin_average_it)
            # Append the same pose, category, and object id for each
            pose_datas.append(pose_val)
            objectid_datas.append(objectid)
            category_datas.append(category)

    return v4_datas, it_datas, pose_datas, objectid_datas, category_datas

def get_dataset(average_trials=False, average_downsample_factor=50, random_seed=0, ignore_cache=False):
    if not ignore_cache:
        cache = load_cache(average_trials, average_downsample_factor) 
        if cache is not None:
            return cache
    neuroid_train_data, neuroid_test_data = generate_brainscore_train_test_split(random_seed=random_seed)

    if average_trials == False:
        average_downsample_factor = 1

    v4_train, it_train, pose_train, objectid_train, label_train = apply_averaging(
        neuroid_train_data,
        average_downsample_factor=average_downsample_factor
    )
    v4_test, it_test, pose_test, objectid_test, label_test = apply_averaging(
        neuroid_test_data,
        average_downsample_factor=average_downsample_factor
    )
    # Convert everything to troch tensors from numpy
    v4_train = torch.tensor(v4_train).float()
    v4_test = torch.tensor(v4_test).float()
    assert torch.all(torch.isfinite(v4_train))
    assert torch.all(torch.isfinite(v4_test))
    label_train = torch.tensor(label_train).long()
    objectid_train = torch.tensor(objectid_train).long()
    pose_train = torch.tensor(pose_train).float()
    it_train = torch.tensor(it_train).float()
    it_test = torch.tensor(it_test).float()
    label_test = torch.tensor(label_test).long()
    objectid_test = torch.tensor(objectid_test).long()
    pose_test = torch.tensor(pose_test).float()

    assert v4_train.shape[0] == it_train.shape[0] == len(label_train) == len(objectid_train) == len(pose_train)
    data = (
        (v4_train, it_train, label_train, objectid_train, pose_train),
        (v4_test, it_test, label_test, objectid_test, pose_test)
    )

    if average_trials:
        torch.save(data, f"avgd-{average_downsample_factor}-data.pt")
    else:
        torch.save(data, "data.pt")
    
    return data


if __name__ == "__main__":
    get_dataset()
