import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch import Tensor

import wandb


def get_model_output_on_cpu(data, model, device, batch_size=1000):
    embeddings = []
    batch_size = min(batch_size, len(data))
    with torch.no_grad():
        for i in range(len(data) // batch_size):
            x = data[i * batch_size : (i + 1) * batch_size].to(device)
            z = model(x)
            embeddings.append(z.detach().cpu())
        if len(data) % batch_size != 0:
            z = model(data[(i + 1) * batch_size :].to(device))
            embeddings.append(z.detach().cpu())
        embeddings = torch.cat(embeddings)
    return embeddings


def embed_v4_data(data, backbone, device, batch_size=1000):
    return get_model_output_on_cpu(data, backbone, device, batch_size=batch_size)


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


def evaluate_IT_explained_variance(train_feat, train_it, test_feat, test_it, args):
    """
    Measures the explained variance of the IT data.

    (1) Encode the representations of the given backbone.
    (2) Fit a linear regression model for each of the IT neuronal
        site recordings.
    (3) Return the R^2 score for each of the linear regression models.
    """
    num_neuronal_sites = test_it.shape[-1]
    assert num_neuronal_sites == 168

    return _eval_linear_regr_median_r2(train_feat, train_it, test_feat, test_it)


def eval_best_layer(eval_fn, backbone, contrastive_head, train_v4, train_Y, test_v4, test_Y, args):
    """
    Returns median r^2 value across IT sites for the best layer of the backbone
    or contrastive head.
    """
    backbone.eval()
    if contrastive_head is not None:
        contrastive_head.eval()
        ct_layers_names = [(contrastive_head, 'ct head')]
    else:
        ct_layers_names = []

    z_train, z_test = train_v4, test_v4
    r2s = []
    for layer, name in [
        (nn.Identity(), 'V4'),
        (backbone.enc, 'encode'),
        *[(backbone.skip_list[i], f'bb hidden {i+1}') for i in range(args.num_hidden_layers)],
        (backbone.decode, 'decode'),
        *ct_layers_names,
    ]:
        z_train = get_model_output_on_cpu(z_train, layer, args.device)
        z_test = get_model_output_on_cpu(z_test, layer, args.device)
        r2s.append(eval_fn(z_train, train_Y, z_test, test_Y))
        print(f'{name}: {r2s[-1]}')

    return np.max(r2s)

def eval_IT_pred_best_layer(backbone, contrastive_head, v4_train, it_train, v4_test, it_test, args):
    print('R^2\n===')
    r2 = eval_best_layer(_eval_linear_regr_median_r2, backbone, contrastive_head, v4_train, it_train, v4_test, it_test, args)
    print('\nPearson R\n=========')
    r = eval_best_layer(_eval_linear_regr_median_r, backbone, contrastive_head, v4_train, it_train, v4_test, it_test, args)
    return r2, r

def _eval_linear_regr_median_r2(train_X, train_Y, test_X, test_Y):
    regr_model = sklearn.linear_model.LinearRegression().fit(tnp(train_X), tnp(train_Y))
    ypred = regr_model.predict(tnp(test_X))
    ytrue = tnp(test_Y)
    u = np.sum((ytrue - ypred) ** 2, axis=0)
    v = np.sum((ytrue - np.mean(ytrue, axis=0)) ** 2, axis=0)
    r2 = 1 - u / v
    return np.median(r2)

def _eval_linear_regr_median_r(train_X, train_Y, test_X, test_Y):
    regr_model = sklearn.linear_model.LinearRegression().fit(tnp(train_X), tnp(train_Y))
    ypred = regr_model.predict(tnp(test_X))
    ytrue = tnp(test_Y)
    # pearson correlation
    ytrue_mean = ytrue.mean(0)
    ypred_mean = ypred.mean(0)
    r = ((ytrue - ytrue_mean) * (ypred - ypred_mean)).sum(0)
    denom = np.sqrt(((ytrue - ytrue_mean) ** 2).sum(0) * ((ypred - ypred_mean) ** 2).sum(0))
    r = r / denom
    return np.median(r)

def tnp(tensor: Tensor):
    return tensor.detach().cpu().numpy()


def _eval_regression(train_X: Tensor, train_Y: Tensor, test_X: Tensor, test_Y: Tensor, args):
    assert train_X.shape[0] == train_Y.shape[0], f"{train_X.shape} != {train_Y.shape}"
    assert test_X.shape[0] == test_Y.shape[0], f"{test_X.shape} != {test_Y.shape}"
    assert train_X.shape[1] == test_X.shape[1], f"{train_X.shape} != {test_X.shape}"
    assert train_Y.shape[1] == test_Y.shape[1], f"{train_Y.shape} != {test_Y.shape}"

    # Fit a linear regression model to the data
    if args.eval_regression_model == "linear":
        regression_model = sklearn.linear_model.LinearRegression().fit(tnp(train_X), tnp(train_Y))
        ypred = regression_model.predict(tnp(test_X))
    elif args.eval_regression_model == "svr":
        y_pred_list = []
        for pose in range(train_Y.shape[-1]):
            regression_model = sklearn.svm.SVR(kernel="linear", C=1e-3, tol=1e-5).fit(
                tnp(train_X), tnp(train_Y[..., pose])
            )
            ypred = regression_model.predict(tnp(test_X))
            y_pred_list.append(ypred)
        ypred = np.stack(y_pred_list, axis=-1)
    else:
        raise NotImplementedError

    ytrue = tnp(test_Y)

    # R^2 = 1 - u/v
    # u = sum_i (y_i - ypred_i)^2
    # v = sum_i (y_i - ymean)^2
    # sum just over rows to get per-dimension R^2
    u = np.sum((ytrue - ypred) ** 2, axis=0)
    v = np.sum((ytrue - np.mean(ytrue, axis=0)) ** 2, axis=0)
    r2 = 1 - u / v

    # pearson correlation
    ytrue_mean = ytrue.mean(0)
    ypred_mean = ypred.mean(0)
    r = ((ytrue - ytrue_mean) * (ypred - ypred_mean)).sum(0)
    denom = np.sqrt(((ytrue - ytrue_mean) ** 2).sum(0) * ((ypred - ypred_mean) ** 2).sum(0))
    r = r / denom

    return (np.mean(r2), np.median(r2), *r2), (np.mean(r), np.median(r), *r)


# def evaluate_pose_regression(train_feat, train_objectid, train_pose, test_data, test_objectid, test_pose, args):
def evaluate_pose_regression(train_feat, train_pose, test_data, test_pose, args):
    return _eval_regression(train_feat, train_pose, test_data, test_pose, args)
    # for per-object:
    r2s = []
    rs = []
    for o in torch.unique(train_objectid):
        idx = train_objectid == o
        train_feat_o = train_feat[idx]
        train_pose_o = train_pose[idx]
        idx = test_objectid == o
        test_data_o = test_data[idx]
        test_pose_o = test_pose[idx]
        r2, r = _eval_regression(train_feat_o, train_pose_o, test_data_o, test_pose_o, args)
        r2s.append(r2)
        rs.append(r)
    r2s = np.row_stack(r2s)
    rs = np.row_stack(rs)
    assert r2s.shape == (len(torch.unique(train_objectid)), 8)
    return r2s.mean(0), rs.mean(0)


def evaluate_pose_change_regression(manifold_model, train_data, train_idx, train_pose, test_data, test_idx, test_pose, args):
    rng = np.random.default_rng(args.seed)
    n = args.eval_pose_change_regr_n_pairs
    i_train_a = np.arange(len(train_data[:n]))
    train_a = train_data[:n].to(args.device)
    i_train_b = torch.tensor([rng.choice(train_idx[inst_idx.item()]) for inst_idx in i_train_a])
    train_b = train_data[i_train_b].to(args.device)

    i_test_a = np.arange(len(test_data[:n]))
    test_a = test_data[:n].to(args.device)
    i_test_b = torch.tensor([rng.choice(test_idx[inst_idx.item()]) for inst_idx in i_test_a])
    test_b = test_data[i_test_b].to(args.device)

    """
    i_train_a = rng.choice(len(train_data), size=n, replace=True)
    train_a = train_data[i_train_a].detach().to(args.device)
    i_train_b = rng.choice(len(train_data), size=n, replace=True)
    train_b = train_data[i_train_b].detach().to(args.device)
    i_test_a = rng.choice(len(test_data), size=n, replace=True)
    test_a = test_data[i_test_a].detach().to(args.device)
    i_test_b = rng.choice(len(test_data), size=n, replace=True)
    test_b = test_data[i_test_b].detach().to(args.device)
    """

    transop, coeff_enc = manifold_model

    c_train = []
    c_test = []
    for i in range((len(train_a) // 1000) - 1):
        dist_data_train = coeff_enc(
            train_a[i * 1000 : (i + 1) * 1000],
            train_b[i * 1000 : (i + 1) * 1000],
            transop,
        )
        c_train.append(dist_data_train.samples.detach().cpu())
    if len(train_a) % 1000 != 0:
        dist_data_train = coeff_enc(
            train_a[(i+1) * 1000 :],
            train_b[(i+1) * 1000 :],
            transop,
        )
        c_train.append(dist_data_train.samples.detach().cpu())

    dist_data_test = coeff_enc(
        test_a,
        test_b,
        transop,
    )
    c_test.append(dist_data_test.samples.detach().cpu())

    c_train = torch.cat(c_train, dim=0)
    c_test = torch.cat(c_test, dim=0)

    return _eval_regression(
        c_train, train_pose[i_train_b] - train_pose[i_train_a], c_test, test_pose[i_test_b] - test_pose[i_test_a], args
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


if __name__ == "__main__":
    model = torch.load("model_weights_epoch9999.pt")
    # backbone, transop, coeff_enc = torch.load('model_weights_epoch9999.pt')
