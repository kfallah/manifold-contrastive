"""
Perform k-Nearest Neighbors to determine whether points in the latent space match examples from the same class.

@Filename    knn_acc.py
@Author      Kion
@Created     09/01/22
"""


from typing import Dict, Tuple

import scipy as sp
import torch
import torch.nn as nn

from eval.config import KNNEvalConfig
from eval.type import EvalRunner, EvaluationInput


class KNNEval(EvalRunner):
    def run_eval(
        self, train_eval_input: EvaluationInput, val_eval_input: EvaluationInput
    ) -> Tuple[Dict[str, float], float]:
        knn_metrics = {}

        norm_train_features = nn.functional.normalize(train_eval_input.feature_list)
        dist = torch.mm(val_eval_input.feature_list, norm_train_features.T).T
        _, yi = dist.topk(self.get_config().k, dim=0, largest=True, sorted=True)
        pred = torch.tensor(sp.stats.mode(train_eval_input.labels[yi], axis=0)[0])
        feat_knn_acc = pred.eq(val_eval_input.labels).sum().item() / len(val_eval_input.feature_list)
        knn_metrics["feat_knn_acc"] = feat_knn_acc

        norm_train_pred = nn.functional.normalize(train_eval_input.prediction_list)
        dist = torch.mm(val_eval_input.prediction_list, norm_train_pred.T).T
        _, yi = dist.topk(self.get_config().k, dim=0, largest=True, sorted=True)
        pred = torch.tensor(sp.stats.mode(train_eval_input.labels[yi], axis=0)[0])
        pred_knn_acc = pred.eq(val_eval_input.labels).sum().item() / len(val_eval_input.prediction_list)
        knn_metrics["pred_knn_acc"] = pred_knn_acc

        return knn_metrics, feat_knn_acc

    def get_config(self) -> KNNEvalConfig:
        return KNNEvalConfig(self.cfg)
