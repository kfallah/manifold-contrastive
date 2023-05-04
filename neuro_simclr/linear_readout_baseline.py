"""
    This is the code for a linear readout baseline experiment. 

    Our goal is to evaluate the performance of a basic linear
    binary classifier on predicting whether or not an image is of
    an animal or not using the V4 data. 
"""
import argparse
import brainscore
from compute_positive_pairs import load_brainscore_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn
import matplotlib.pyplot as plt
from evaluation import evaluate_nn_classifier

import torch.nn as nn

def load_animal_classification_dataset(split_percentage=0.8, average_across_stimuli=True):
    print("Loading brainscore dataset")
    brainscore_data = load_brainscore_data()
    neural_data = brainscore.get_assembly(name="dicarlo.MajajHong2015.public")
    neuroid_data = neural_data.multi_groupby([
        'category_name', 
        'object_name', 
        'stimulus_id'
    ]).mean(dim='presentation')
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
        stratify=animal_labels
    )
    
    return (train_v4, train_labels), (test_v4, test_labels)

def train_logistic_regression(train_data, test_data):
    (train_v4, train_labels), (test_v4, test_labels) = train_data, test_data
    # Train sklearn logistic regression model
    clf = sklearn.linear_model.LogisticRegressionCV().fit(train_v4, train_labels)
    # Evaluate the model on the test set
    score = clf.score(test_v4, test_labels)
    print("Accuracy: ", score)
    # Predict confusion matrix
    predicted_labels = clf.predict(test_v4)
    print(confusion_matrix)
    confusion_mat = confusion_matrix(test_labels, predicted_labels)
    confusion_mat = confusion_mat / confusion_mat.sum(axis=1, keepdims=True)
    print("Confusion matrix: ", confusion_mat)
    # Plot the sklearn confusion matrix
    plt.figure()
    plt.imshow(confusion_mat, cmap='Blues')
    plt.savefig("confusion_matrix.png")
    # Print fscore
    fscore = sklearn.metrics.f1_score(test_labels, predicted_labels)
    print("Fscore: ", fscore)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Linear readout baseline experiment")
    # Setup the data
    # train_data, test_data = load_animal_classification_dataset()
    # Train the logistic regression model
    # train_logistic_regression(train_data, test_data)
    animals_train_dataset = AnimalClassificationDataset(split="train")
    animals_test_dataset = AnimalClassificationDataset(split="test")
    
    model = nn.Sequential(
        nn.Identity(),
    )

    evaluate_nn_classifier(
        model,
    )
