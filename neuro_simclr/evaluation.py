import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch import Tensor

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
    raise NotImplementedError("This function is not yet implemented with the new dataset paradigm.")
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
    # clf = sklearn.linear_model.LogisticRegressionCV().fit(train_embeddings, train_labels)
    clf = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    clf.fit(train_embeddings, train_labels)
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
    raise NotImplementedError("This function is not yet implemented with the new dataset paradigm.")
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


def tnp(tensor: Tensor):
    return tensor.detach().cpu().numpy()


def _eval_regression(train_X: Tensor, train_Y: Tensor, test_X: Tensor, test_Y: Tensor):
    assert train_X.shape[0] == train_Y.shape[0]
    assert test_X.shape[0] == test_Y.shape[0]
    assert train_X.shape[1] == test_X.shape[1]
    assert train_Y.shape[1] == test_Y.shape[1]

    # Fit a linear regression model to the data
    linear_regression_model = sklearn.linear_model.LinearRegression().fit(tnp(train_X), tnp(train_Y))
    ypred = linear_regression_model.predict(tnp(test_X))
    ytrue = tnp(test_Y)

    # R^2 = 1 - u/v
    # u = sum_i (y_i - ypred_i)^2
    # v = sum_i (y_i - ymean)^2
    # sum just over rows to get per-dimension R^2
    u = np.sum((ytrue - ypred) ** 2, axis=0)
    v = np.sum((ytrue - np.mean(ytrue, axis=0)) ** 2, axis=0)
    r2 = 1 - u / v

    return np.mean(r2), np.median(r2), *r2


def evaluate_pose_regression(train_feat, train_pose, test_data, test_pose, args):
    return _eval_regression(train_feat, train_pose, test_data, test_pose)


def evaluate_pose_change_regression(manifold_model, train_data, train_pose, test_data, test_pose, args):
    rng = np.random.default_rng(args.seed)
    n = args.eval_pose_change_regr_n_pairs
    i_train_a = rng.choice(len(train_data), size=n, replace=True)
    train_a = train_data[i_train_a].detach().to(args.device)
    i_train_b = rng.choice(len(train_data), size=n, replace=True)
    train_b = train_data[i_train_b].detach().to(args.device)
    i_test_a = rng.choice(len(test_data), size=n, replace=True)
    test_a = test_data[i_test_a].detach().to(args.device)
    i_test_b = rng.choice(len(test_data), size=n, replace=True)
    test_b = test_data[i_test_b].detach().to(args.device)

    transop, coeff_enc = manifold_model

    c_train = []
    c_test = []
    for i in range(n // 1000):
        dist_data_train = coeff_enc(
            train_a[i * 1000 : (i + 1) * 1000],
            train_b[i * 1000 : (i + 1) * 1000],
            transop,
        )
        c_train.append(dist_data_train.samples)

        dist_data_test = coeff_enc(
            test_a[i * 1000 : (i + 1) * 1000],
            test_b[i * 1000 : (i + 1) * 1000],
            transop,
        )
        c_test.append(dist_data_test.samples)

    c_train = torch.cat(c_train, dim=0)
    c_test = torch.cat(c_test, dim=0)

    return _eval_regression(
        c_train,
        train_pose[i_train_b] - train_pose[i_train_a],
        c_test,
        test_pose[i_test_b] - test_pose[i_test_a],
    )


def sweep_psi_path_plot(psi: torch.tensor, z0: np.array, c_mag: int):
    z = torch.tensor(z0).float().to(psi.device)[: psi.shape[-1]]

    # z = model.backbone(x_gpu[0])[0]
    # z = torch.tensor(z0[0][0]).to(default_device)
    # psi = model.contrastive_header.transop_header.transop.psi
    psi_norm = (psi.reshape(len(psi), -1) ** 2).sum(dim=-1)
    psi_idx = torch.argsort(psi_norm)
    latent_dim = len(z)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))
    plt.subplots_adjust(hspace=0.4, top=0.9)

    for i in range(ax.size):
        row = int(i / 2)
        column = int(i % 2)
        curr_psi = psi_idx[-(i + 1)]

        coeff = torch.linspace(-c_mag, c_mag, 30, device=psi.device)
        T = torch.matrix_exp(coeff[:, None, None] * psi[None, curr_psi])
        z1_hat = (T @ z).squeeze(dim=-1)

        for z_dim in range(latent_dim):
            ax[row, column].plot(
                np.linspace(-c_mag, c_mag, 30),
                z1_hat[:, z_dim].detach().cpu().numpy(),
            )
        ax[row, column].title.set_text(f"Psi {curr_psi} - F-norm: {psi_norm[curr_psi]:.2E}")

    return fig


def transop_plots(coefficients: np.array, psi: torch.tensor, z0: np.array):
    psi_norms = ((psi.reshape(len(psi), -1)) ** 2).sum(dim=-1).detach().cpu().numpy()
    count_nz = np.zeros(len(psi) + 1, dtype=int)
    total_nz = np.count_nonzero(coefficients, axis=1)
    for z in range(len(total_nz)):
        count_nz[total_nz[z]] += 1
    number_operator_uses = np.count_nonzero(coefficients, axis=0) / len(coefficients)

    psi_mag_fig = plt.figure(figsize=(20, 4))
    plt.bar(np.arange(len(psi)), psi_norms, width=1)
    plt.xlabel("Transport Operator Index", fontsize=18)
    plt.ylabel("F-Norm", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("F-Norm of Transport Operators", fontsize=20)

    coeff_use_fig = plt.figure(figsize=(20, 4))
    plt.bar(np.arange(len(psi) + 1), count_nz, width=1)
    plt.xlabel("Number of Coefficients Used per Point Pair", fontsize=18)
    plt.ylabel("Occurences", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Number of Non-Zero Coefficients", fontsize=20)

    psi_use_fig = plt.figure(figsize=(20, 4))
    plt.bar(np.arange(len(psi)), number_operator_uses, width=1)
    plt.xlabel("Percentage of Point Pairs an Operator is Used For", fontsize=18)
    plt.ylabel("% Of Point Pairs", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Transport Operator Index", fontsize=20)

    psi_eig_plt = plt.figure(figsize=(8, 8))
    L = torch.linalg.eigvals(psi.detach())
    plt.scatter(torch.real(L).detach().cpu().numpy(), torch.imag(L).detach().cpu().numpy())
    plt.xlabel("Real Components of Eigenvalues", fontsize=18)
    plt.ylabel("Imag Components of Eigenvalues", fontsize=18)

    psi_sweep_1c_fig = sweep_psi_path_plot(psi.detach(), z0, 1)
    psi_sweep_5c_fig = sweep_psi_path_plot(psi.detach(), z0, 5)

    figure_dict = {
        "psi_mag_iter": psi_mag_fig,
        "coeff_use_iter": coeff_use_fig,
        "psi_use_iter": psi_use_fig,
        "psi_eig_plt": psi_eig_plt,
        "psi_sweep_1c": psi_sweep_1c_fig,
        "psi_sweep_5c": psi_sweep_5c_fig,
    }

    return figure_dict

if __name__ == '__main__':
    model = torch.load('model_weights_epoch9999.pt')
    # backbone, transop, coeff_enc = torch.load('model_weights_epoch9999.pt')