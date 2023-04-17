"""
Perform semi-supervised learning on features by either applying consistency regularization
or by applying augmentations sampled from the manifold operators.

@Filename    semi_sup_probe.py
@Author      Kion
@Created     04/17/23
"""


from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from eval.config import SemiSupConfig
from eval.type import EvalRunner, EvaluationInput
from eval.utils import get_feature_split


class SemiSupProbeEval(EvalRunner):
    def semisup_probe(
        self,
        train_zl,
        train_yl,
        train_zu,
        val_z,
        val_y,
        device,
        num_iters,
        transop_aug: bool = False,
        coeff_enc=None,
        psi=None,
    ):
        clf = nn.Sequential(nn.Linear(512, 2048), nn.LeakyReLU(), nn.Linear(2048, 10)).to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1.0e-5)
        criterion = torch.nn.functional.cross_entropy
        curr_iter = 0

        while curr_iter < num_iters:
            label_idx, unlabel_idx = (
                torch.randperm(len(train_zl))[: self.get_config().batchsize_label],
                torch.randperm(len(train_zu))[: self.get_config().batchsize_unlabel],
            )
            zl, yl = train_zl[label_idx].to(device), train_yl[label_idx].to(device)
            zu = train_zu[unlabel_idx].to(device)

            # TO AUG
            if not transop_aug:
                zu_aug = zu.clone()
            else:
                c = coeff_enc.prior_sample(zu.detach())
                A = torch.einsum("bm,smuv->bsuv", c, psi)
                T = torch.matrix_exp(A)
                zu_aug = (T @ zu.reshape(len(zu), psi.shape[0], -1, 1)).reshape(*zu.shape)

            y_pred = clf(zl)
            loss = criterion(y_pred, yl)

            # Consistency Regularization
            xu_aug_logits = clf(zu_aug)
            xu_prob = torch.softmax(clf(zu).detach(), dim=-1)
            wu, yu = torch.max(xu_prob, dim=1)
            wu = (wu > 0.95).detach()
            loss_con = criterion(xu_aug_logits, yu, reduction="none")
            loss_con = torch.sum(loss_con[wu]) / len(wu)

            optimizer.zero_grad()
            (loss + loss_con).backward()
            optimizer.step()
            curr_iter += 1

        y_pred = clf(val_z.to(device))
        pred_top = y_pred.topk(max([1]), 1, largest=True, sorted=True).indices
        acc = (pred_top[:, 0].detach().cpu() == val_y).float().mean().item()
        return acc

    def run_eval(
        self, train_eval_input: EvaluationInput, val_eval_input: EvaluationInput, device: torch.device, **kwargs
    ) -> Tuple[Dict[str, float], float]:
        train_z, train_y = train_eval_input.feature_list, train_eval_input.labels
        val_z, val_y = val_eval_input.feature_list, val_eval_input.labels
        metrics = {}
        acc_list = []
        for i in range(self.get_config().num_trials):
            (train_zl, train_yl), (train_zu, _) = get_feature_split(
                self.get_config().labels_per_class, train_z, train_y, seed=i
            )
            acc = self.semisup_probe(train_zl, train_yl, train_zu, val_z, val_y, device, self.get_config().num_iters)
            acc_list.append(acc)
        metrics.update({"ssl_acc_mean": np.mean(acc_list), "ssl_acc_std": np.std(acc_list)})

        if self.get_config().manifold_aug:
            manifold_acc = []
            model = val_eval_input.model
            psi = model.contrastive_header.transop_header.transop.psi
            coeff_enc = model.contrastive_header.transop_header.coefficient_encoder
            for i in range(self.get_config().num_trials):
                (train_zl, train_yl), (train_zu, _) = get_feature_split(
                    self.get_config().labels_per_class, train_z, train_y, seed=i
                )
                acc = self.semisup_probe(
                    train_zl,
                    train_yl,
                    train_zu,
                    val_z,
                    val_y,
                    device,
                    self.get_config().num_iters,
                    transop_aug=True,
                    coeff_enc=coeff_enc,
                    psi=psi,
                )
                manifold_acc.append(acc)
            metrics.update(
                {
                    "ssl_man_acc_mean": np.mean(manifold_acc),
                    "ssl_man_acc_std": np.std(manifold_acc),
                }
            )

        return metrics, np.mean(acc_list), {}

    def get_config(self) -> SemiSupConfig:
        return self.cfg
