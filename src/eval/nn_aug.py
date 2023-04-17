"""
Apply manifold augmentations by sampling from the prior and plot
the nearest neighbors in the dataset.

@Filename    nn_aug.py
@Author      Kion
@Created     04/17/23
"""


import copy
import warnings
from typing import Dict, Tuple

import torch
import torchvision.transforms
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances

from eval.config import AugmentationNN
from eval.type import EvalRunner, EvaluationInput
from eval.utils import encode_features

warnings.filterwarnings("ignore")


class AugmentationNNEval(EvalRunner):
    def run_eval(
        self, train_eval_input: EvaluationInput, val_eval_input: EvaluationInput, device: torch.device, **kwargs
    ) -> Tuple[Dict[str, float], float]:
        dataset_obj = kwargs["dataset"]
        eval_dataloader = dataset_obj.eval_dataloader
        dataset = copy.deepcopy(eval_dataloader.dataset)
        dataset.transform = torchvision.transforms.ToTensor()
        unshuffle_dataloder = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=500,
            shuffle=False,
            num_workers=32,
            pin_memory=True,
            drop_last=False,
        )
        nn_features = encode_features(train_eval_input.model, unshuffle_dataloder, device)
        z, x_im = nn_features.feature_list, nn_features.x

        model = val_eval_input.model
        transop = model.contrastive_header.transop_header.transop
        coeff_enc = model.contrastive_header.transop_header.coefficient_encoder
        start_idx = 20
        num_im, num_augs = self.get_config().num_images, self.get_config().num_augs

        aug_list = torch.zeros((num_im, num_augs, 512))
        for i in range(num_im):
            for j in range(num_augs):
                z0 = z[start_idx + i].to(device)
                c = coeff_enc.prior_sample(z0.unsqueeze(0).detach()).squeeze(0) * 2
                T = torch.matrix_exp(torch.einsum("m,smuv->suv", c, transop.psi))
                zu_aug = (T @ z0.reshape(transop.psi.shape[0], -1, 1)).reshape(*z0.shape)
                aug_list[i, j] = zu_aug.detach().cpu()

        fig, ax = plt.subplots(nrows=num_im, ncols=num_augs + 1, figsize=(12, 12))

        for k in range(num_im):
            pw_dist = pairwise_distances(z, aug_list[k])
            nn_idx = torch.argmin(torch.tensor(pw_dist), dim=0)
            ax[k, 0].imshow(x_im[start_idx + k].permute(1, 2, 0))
            for i in range(num_augs):
                ax[k, i + 1].imshow(x_im[nn_idx[i]].permute(1, 2, 0))
        [axi.set_axis_off() for axi in ax.ravel()]
        plt.subplots_adjust(wspace=0.0, hspace=0.0)

        return {}, -1.0, {"aug_nn": fig}

    def get_config(self) -> AugmentationNN:
        return self.cfg