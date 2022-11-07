"""
Perform k-means clustering on the embedding features.

@Filename    clustering.py
@Author      Kion
@Created     09/01/22
"""


from typing import Dict, Tuple

import numpy as np
import scipy as sp
from sklearn.cluster import KMeans

from eval.config import ClusteringEvalConfig
from eval.type import EvalRunner, EvaluationInput


class ClusteringEval(EvalRunner):
    def get_purity_acc(self, pred_labels, true_labels):
        acc = 0
        num_pred_classes = np.unique(pred_labels)
        for i in num_pred_classes:
            purity_label = sp.stats.mode(true_labels[pred_labels == i])[0]
            correct_labels = purity_label == true_labels[pred_labels == i]
            acc += correct_labels.sum()
        acc = acc / len(pred_labels)
        return acc

    def run_eval(
        self, train_eval_input: EvaluationInput, val_eval_input: EvaluationInput, **kwargs
    ) -> Tuple[Dict[str, float], float]:
        cluster_metrics = {}
        # Take a linear spacing of features indices based on the number of points used for clustering
        feat_cluster_idx = np.linspace(
            0, len(val_eval_input.feature_list) - 1, self.get_config().num_points_cluster, dtype=int
        )
        # Fit k-means clustering to the selected points
        feature_cluster = KMeans(n_clusters=self.get_config().num_clusters, random_state=0).fit(
            val_eval_input.feature_list[feat_cluster_idx]
        )
        # Get the purity accuracy of the estimated clusters
        feature_cluster_acc = self.get_purity_acc(
            feature_cluster.labels_, val_eval_input.labels[feat_cluster_idx].numpy()
        )
        # Save to metadata dictionary
        cluster_metrics["feat_cluster_acc"] = feature_cluster_acc

        return cluster_metrics, feature_cluster_acc

    def get_config(self) -> ClusteringEvalConfig:
        return self.cfg
