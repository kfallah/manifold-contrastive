import sklearn 
import torch
import torch.nn as nn
import wandb
import numpy as np
from tqdm import tqdm

def embed_v4_data(dataset, backbone, batch_size=32, one_hot=False, args=None):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
    )

    embeddings = []
    labels = []
    for data in dataloader:
        image, label = data
        if one_hot:
            label = torch.eye(dataset.num_classes)[label]
        image = image.to(args.device)
        embedding = backbone(image.unsqueeze(0)).squeeze()
        embedding = list(embedding.detach().cpu())
        embeddings.extend(embedding)
        labels.extend(label)

    embeddings = torch.stack(embeddings)
    labels = torch.stack(labels)

    return embeddings, labels

def evaluate_linear_classifier(
    backbone, 
    train_dataset, 
    test_dataset, 
    args,
    num_epochs=100,
    batch_size=32,
):
    """
        Trains a simple single layer linear nn classifier using
        the frozen backbone representations. 
    """
    num_classes = train_dataset.num_classes
    # Set the backbone to eval mode
    backbone.eval()
    max_class_index = -1
    # Embed the train dataset of images using the backbone
    train_data = embed_v4_data(train_dataset, backbone, batch_size=batch_size, one_hot=True, args=args)
    # Embed the test dataset of images using the backbone
    test_data = embed_v4_data(test_dataset, backbone, batch_size=batch_size, one_hot=False, args=args)
    # Train a single layer linear nn classifier on the train dataset
    neural_network = nn.Sequential(
        nn.Linear(test_data[0][0].shape[-1], num_classes),
    ).to(args.device)
    # Train the model using an adam optimizer
    optim = torch.optim.Adam(neural_network.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    train_data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_data),
        batch_size=batch_size,
    )
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*test_data),
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
                _, predicted = torch.max(output.data, -1)
                total += labels.size(-1)
                correct += (predicted == labels).sum().item()

                preds.extend(predicted.detach().cpu().numpy())
                labs.extend(labels.detach().cpu().numpy())

            if num_classes == 2: 
                fscore = sklearn.metrics.f1_score(labs, preds)
            else:
                fscore = sklearn.metrics.f1_score(labs, preds, average="macro")

    if wandb.run is not None:
        wandb.log({
            f"{train_dataset.dataset_name}_linear_eval_accuracy": correct/total,
            f"{train_dataset.dataset_name}_linear_eval_fscore": fscore
        })

    print(f"Accuracy: {correct/total}")
    print(f"Fscore: {fscore}")

def evaluate_knn(backbone, train_dataset, test_dataset, args):
    # Return accuracy
    pass

def evaluate_logistic_regression(backbone, train_dataset, test_dataset, args):
    """
        Evaluate the linear readout
    """
    # Put the backbone in eval mode
    backbone.eval()
    # Embed each of the neuroid data using the backbone
    train_embeddings = []
    train_labels = []
    for data in train_dataset:
        neuroid_data, labels = data
        neuroid_data = neuroid_data.to(args.device)
        if len(neuroid_data.shape) == 1:
            neuroid_data = neuroid_data.unsqueeze(0)
        backbone_out = backbone(neuroid_data).detach().cpu().numpy()
        train_embeddings.append(
            backbone_out.squeeze()
        )
        train_labels.append(labels)
    # Embed the test data
    test_embeddings = []
    test_labels = []
    for data in test_dataset:
        neuroid_data, labels = data
        neuroid_data = neuroid_data.to(args.device)
        if len(neuroid_data.shape) == 1:
            neuroid_data = neuroid_data.unsqueeze(0)

        backbone_out = backbone(neuroid_data).detach().cpu().numpy()
        test_embeddings.append(
            backbone_out.squeeze()
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

    if wandb.run is not None:
        wandb.log({
            f"{train_dataset.dataset_name}_logistic_regression_accuracy": accuracy,
            f"{train_dataset.dataset_name}_logistic_regression_fscore": fscore
        })

def evaluate_IT_explained_variance(backbone, neuroid_train_dataset, neuroid_eval_dataset, args):
    """
        Measures the explained variance of the IT data. 

        (1) Encode the representations of the given backbone.
        (2) Fit a linear regression model for each of the IT neuronal
            site recordings.
        (3) Return the R^2 score for each of the linear regression models.
    """
    # Put the backbone in eval mode
    backbone.eval()

    v4_train_dataset = neuroid_train_dataset.sel(region='V4')
    it_train_dataset = neuroid_train_dataset.sel(region='IT')
    v4_test_dataset = neuroid_eval_dataset.sel(region='V4')
    it_test_dataset = neuroid_eval_dataset.sel(region='IT')
    # Generate embeddings from the backbone
    train_embeddings = []
    for neuroid_data in v4_train_dataset:
        neuroid_data = torch.Tensor(neuroid_data.to_numpy()).to(args.device)
        neuroid_data = neuroid_data.unsqueeze(0)
        train_embeddings.append(
            backbone(neuroid_data).detach().cpu().numpy().squeeze()
        )
    test_embeddings = []
    for neuroid_data in v4_test_dataset:
        neuroid_data = torch.Tensor(neuroid_data.to_numpy()).to(args.device)
        neuroid_data = neuroid_data.unsqueeze(0)
        test_embeddings.append(
            backbone(neuroid_data).detach().cpu().numpy().squeeze()
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


