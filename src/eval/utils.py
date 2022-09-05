"""
Utility functions used for evaluator runners.

@Filename    utils.py
@Author      Kion
@Created     09/05/22
"""
import torch
import torch.nn as nn

from eval.type import EvaluationInput


def encode_features(
    model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device
) -> EvaluationInput:
    x_eval = []
    labels = []
    x_idx = []
    feature_list = []
    prediction_list = []
    model.eval()
    for _, batch in enumerate(data_loader):
        x, batch_label, batch_idx = batch
        x_gpu = x.to(device).unsqueeze(1)
        batch_idx = torch.Tensor([int(idx) for idx in batch_idx])
        model_output = model(x_gpu, batch_idx)

        x_eval.append(x.detach().cpu())
        labels.append(batch_label.detach().cpu())
        x_idx.append(batch_idx.detach().cpu())
        feature_list.append(model_output.feature_list.detach().cpu())
        prediction_list.append(model_output.prediction_list.detach().cpu())
    # Flatten all encoded data to a single tensor
    x_eval = torch.cat(x_eval)
    labels = torch.cat(labels)
    x_idx = torch.cat(x_idx)
    feature_list = torch.cat(feature_list).squeeze(1)
    prediction_list = torch.cat(prediction_list).squeeze(1)
    return EvaluationInput(model, x_eval, x_idx, labels, feature_list, prediction_list)
