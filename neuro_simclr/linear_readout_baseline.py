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

from evaluation import evaluate_linear_classifier

import torch.nn as nn

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

    evaluate_linear_classifier(
        model,
    )
