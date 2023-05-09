"""
Utility functions used for evaluator runners.

@Filename    utils.py
@Author      Kion
@Created     09/05/22
"""
import numpy as np
import torch
import torch.nn as nn

from eval.type import EvaluationInput


def get_feature_split(labels_per_class, features, labels, num_classes=10, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    labeled_idx = []
    for i in range(num_classes):
        idx = torch.where(labels == i)[0]
        subset_idx = np.random.choice(idx.detach().cpu().numpy(), labels_per_class, replace=False)
        labeled_idx.append(subset_idx)
    labeled_idx = np.concatenate(labeled_idx)
    unlabel_idx = np.arange(len(features))[~np.in1d(np.arange(len(features)), labeled_idx)]

    zl, yl = features[labeled_idx], labels[labeled_idx]
    zul, yul = features[unlabel_idx], labels[unlabel_idx]

    return (zl, yl), (zul, yul)


def num_correct(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def encode_features(
    model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device
) -> EvaluationInput:
    x_eval = []
    labels = []
    feature_list = []
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            x, batch_label = batch
            x_gpu = x.to(device).unsqueeze(1)
            model_output = model(x_gpu, 0)
            feat = model_output.header_input.feature_0

            x_eval.append(x.detach().cpu())
            labels.append(batch_label.detach().cpu())
            feature_list.append(feat.detach().cpu())
    # Flatten all encoded data to a single tensor
    x_eval = torch.cat(x_eval)
    labels = torch.cat(labels)
    feature_list = torch.cat(feature_list).squeeze(1)
    return EvaluationInput(model, x_eval, labels, feature_list)
