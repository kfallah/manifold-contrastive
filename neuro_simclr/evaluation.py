import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn.manifold import TSNE

import wandb


def embed_v4_data(data, backbone, device, batch_size=1000):
    embeddings = []
    with torch.no_grad():
        for i in range(len(data) // batch_size):
            x = data[i * batch_size : (i + 1) * batch_size].to(device)
            z = backbone(x)
            embeddings.append(z.detach().cpu())
        if len(data) % batch_size != 0:
            z = backbone(data[(i + 1) * batch_size :].to(device))
            embeddings.append(z.detach().cpu())
        embeddings = torch.cat(embeddings)
    return embeddings


def tsne_plot(train_data, train_label, ex_per_class=800):
    data = []
    label = []
    for i in np.unique(train_label):
        idx = torch.where(train_label == i)[0]
        data.append(train_data[idx[:ex_per_class]])
        label.append(train_label[idx[:ex_per_class]])
    data = torch.cat(data)
    label = torch.cat(label)

    fig = plt.figure(figsize=(8, 8))
    feat_embed = TSNE(n_components=2, init="random", perplexity=3).fit_transform(data)
    for i in np.unique(train_label):
        label_idx = i == label
        plt.scatter(*feat_embed[label_idx].T)
    return fig


def evaluate_linear_classifier(
    train_data,
    train_label,
    test_data,
    test_label,
    args,
):
    """
    Trains a simple single layer linear nn classifier using
    the frozen backbone representations.
    """
    num_classes = len(np.unique(train_label))
    clf = nn.Linear(train_data.shape[-1], num_classes).to(args.device)
    # Train the model using an adam optimizer
    lr_start, lr_end = 1e-2, 1e-6
    gamma = (lr_end / lr_start) ** (1 / 500)
    opt = torch.optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(500):
        indices_perm = torch.randperm(len(train_data))
        for i in range(len(train_data) // 1000):
            x = train_data[indices_perm[i * 1000 : (i + 1) * 1000]].to(args.device)
            labels = train_label[indices_perm[i * 1000 : (i + 1) * 1000]].to(args.device)
            output = clf(x)

            loss = loss_fn(output, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

    y_pred = clf(test_data.to(args.device))
    pred = y_pred.topk(1, 1, largest=True, sorted=True).indices[:, 0].detach().cpu().numpy()
    acc = (pred == test_label.numpy()).mean()

    fscore = sklearn.metrics.f1_score(test_label, pred, average="macro")

    return acc, fscore


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
        train_embeddings.append(backbone_out.squeeze())
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
        test_embeddings.append(backbone_out.squeeze())
        test_labels.append(labels)
    # Train a logistic classifier on those data pairs
    # similar to what is done in `linear_readout_baseline`
    clf = sklearn.linear_model.LogisticRegressionCV().fit(train_embeddings, train_labels)
    # Evaluate the model on the test set
    accuracy = clf.score(test_embeddings, test_labels)
    # Predict f1 score
    predicted_labels = clf.predict(test_embeddings)
    fscore = sklearn.metrics.f1_score(test_labels, predicted_labels)

    if wandb.run is not None:
        wandb.log(
            {
                f"{train_dataset.dataset_name}_logistic_regression_accuracy": accuracy,
                f"{train_dataset.dataset_name}_logistic_regression_fscore": fscore,
            }
        )


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

    v4_train_dataset = neuroid_train_dataset.sel(region="V4")
    it_train_dataset = neuroid_train_dataset.sel(region="IT")
    v4_test_dataset = neuroid_eval_dataset.sel(region="V4")
    it_test_dataset = neuroid_eval_dataset.sel(region="IT")
    # Generate embeddings from the backbone
    train_embeddings = []
    for neuroid_data in v4_train_dataset:
        neuroid_data = torch.Tensor(neuroid_data.to_numpy()).to(args.device)
        neuroid_data = neuroid_data.unsqueeze(0)
        train_embeddings.append(backbone(neuroid_data).detach().cpu().numpy().squeeze())
    test_embeddings = []
    for neuroid_data in v4_test_dataset:
        neuroid_data = torch.Tensor(neuroid_data.to_numpy()).to(args.device)
        neuroid_data = neuroid_data.unsqueeze(0)
        test_embeddings.append(backbone(neuroid_data).detach().cpu().numpy().squeeze())
    # Fit a linear regression model for each of the IT neuronal site recordings
    num_neuronal_sites = it_test_dataset.shape[-1]
    assert num_neuronal_sites == 168
    r2_values = []
    for neuron_site_index in range(num_neuronal_sites):
        # Select the data for the given neuronal site.
        site_data = it_train_dataset[:, neuron_site_index]
        # Fit a linear regression model to the data
        linear_regression_model = sklearn.linear_model.LinearRegression().fit(train_embeddings, site_data)
        # Evaluate R^2 of each of the linear regression models
        # on the test set
        r2 = linear_regression_model.score(test_embeddings, neuroid_eval_dataset[:, neuron_site_index])
        r2_values.append(r2)
    # Log the R^2 values to wandb
    wandb.log({"median_IT_explained_variance": np.median(r2_values)})
