import sklearn 
import torch
import torch.nn as nn
import wandb
import numpy as np

def evaluate_nn_classifier(
    backbone, 
    train_dataset, 
    test_dataset, 
    args,
    num_epochs=100,
    batch_size=32
):
    """
        Trains a simple single layer linear nn classifier using
        the frozen backbone representations. 
    """
    max_class_index = -1
    train_data = []
    # Embed the train dataset of images using the backbone
    for data in train_dataset:
        image, label = data
        max_class_index = max(max_class_index, label)
        image = image.to(args.device)
        embedding = backbone(image)
        train_data.append(
            (embedding.detach().cpu(), label)
        )
    # Embed the test dataset of images using the backbone
    test_data = []
    for data in test_dataset:
        image, label = data
        image = image.to(args.device)
        embedding = backbone(image)
        test_data.append(
            (embedding.detach().cpu(), label)
        )
    # Train a single layer linear nn classifier on the train dataset
    print(f"Test data shape: {test_data[0][0].shape}")
    neural_network = nn.Sequential(
        nn.Linear(test_data[0][0].shape[-1], 2)
    ).to(args.device)
    # Train the model using an adam optimizer
    optim = torch.optim.Adam(neural_network.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
    )
    for epoch in range(100):
        optim.zero_grad()
        for data in train_data_loader:
            # Get the embeddings and labels
            embeddings, labels = data
            embeddings = embeddings.to(args.device)
            labels = labels.to(args.device)
            # Get the output from the neural network
            output = neural_network(embeddings)
            loss = loss_fn(output, labels)
            loss.backward()
            optim.step()
        # Evaluate the model on the test dataset
        with torch.no_grad():
            correct = 0
            total = 0
            labs = []
            preds = []
            for data in test_dataloader:
                embeddings, labels = data
                embeddings = embeddings.to(args.device)
                labels = labels.to(args.device)
                output = neural_network(embeddings)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                preds.extend(predicted.detach().cpu().numpy())
                labs.extend(labels.detach().cpu().numpy())
            
            fscore = sklearn.metrics.f1_score(labs, preds)

    if wandb.run is not None:
        wandb.log({
            "nn_eval_accuracy": correct/total,
            "nn_eval_fscore": fscore
        })
    print(f"Accuracy: {correct/total}")
    print(f"Fscore: {fscore}")

def evaluate_knn(backbone, train_dataset, test_dataset, args):
    # Return accuracy
    pass

def evaluate_linear_readout(backbone, train_dataset, test_dataset, args):
    """
        Evaluate the linear readout
    """
    # Embed each of the neuroid data using the backbone
    train_embeddings = []
    train_labels = []
    for data in train_dataset:
        neuroid_data, labels = data
        neuroid_data = neuroid_data.to(args.device)
        train_embeddings.append(
            backbone(neuroid_data).detach().cpu().numpy()
        )
        train_labels.append(labels)
    # Embed the test data
    test_embeddings = []
    test_labels = []
    for data in test_dataset:
        neuroid_data, labels = data
        neuroid_data = neuroid_data.to(args.device)
        test_embeddings.append(
            backbone(neuroid_data).detach().cpu().numpy()
        )
        test_labels.append(labels)
    # Train a logistic classifier on those data pairs 
    # similar to what is done in `linear_readout_baseline`
    clf = sklearn.linear_model.LogisticRegressionCV().fit(
        train_embeddings,
        train_labels
    )
    # Evaluate the model on the test set
    accuracy = clf.score(
        test_embeddings,
        test_labels
    )
    # Predict f1 score
    predicted_labels = clf.predict(test_embeddings)
    fscore = sklearn.metrics.f1_score(test_labels, predicted_labels)

    return accuracy, fscore

def evaluate_IT_explained_variance(backbone, neuroid_train_dataset, neuroid_eval_dataset, args):
    """
        Measures the explained variance of the IT data. 

        (1) Encode the representations of the given backbone.
        (2) Fit a linear regression model for each of the IT neuronal
            site recordings.
        (3) Return the R^2 score for each of the linear regression models.
    """
    v4_train_dataset = neuroid_train_dataset.sel(region='V4')
    it_train_dataset = neuroid_train_dataset.sel(region='IT')
    v4_test_dataset = neuroid_eval_dataset.sel(region='V4')
    it_test_dataset = neuroid_eval_dataset.sel(region='IT')
    # Generate embeddings from the backbone
    train_embeddings = []
    for neuroid_data in v4_train_dataset:
        neuroid_data = torch.Tensor(neuroid_data.to_numpy()).to(args.device)
        train_embeddings.append(
            backbone(neuroid_data).detach().cpu().numpy()
        )
    test_embeddings = []
    for neuroid_data in v4_test_dataset:
        neuroid_data = torch.Tensor(neuroid_data.to_numpy()).to(args.device)
        test_embeddings.append(
            backbone(neuroid_data).detach().cpu().numpy()
        )
    # Fit a linear regression model for each of the IT neuronal site recordings
    num_neuronal_sites = it_test_dataset.shape[-1]
    assert num_neuronal_sites == 168
    r2_values = []
    for neuron_site_index in range(num_neuronal_sites):
        # Select the data for the given neuronal site. 
        site_data = it_train_dataset[:, neuron_site_index]
        # Fit a linear regression model to the data
        linear_regression_model = sklearn.linear_model.LinearRegression().fit(
            train_embeddings,
            site_data
        )
        # Evaluate R^2 of each of the linear regression models 
        # on the test set
        r2 = linear_regression_model.score(
            test_embeddings,
            neuroid_eval_dataset[:, neuron_site_index]
        )
        r2_values.append(r2)
    # Log the R^2 values to wandb
    wandb.log({
        "median_IT_explained_variance": np.median(r2_values)
    })


