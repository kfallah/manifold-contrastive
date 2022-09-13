"""
Transport operator header that estimates the manifold path between a pair of points.

@Filename    transop_header.py
@Author      Kion
@Created     09/07/22
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.contrastive.config import TransportOperatorConfig
from model.manifold.l1_inference import infer_coefficients
from model.manifold.transop import TransOp_expm
from model.manifold.vi_encoder import VIEncoder
from model.type import DistributionData, HeaderInput, HeaderOutput
from torch.cuda.amp import autocast


class TransportOperatorHeader(nn.Module):
    def __init__(self, transop_cfg: TransportOperatorConfig, backbone_feature_dim: int):
        super(TransportOperatorHeader, self).__init__()
        self.transop_cfg = transop_cfg
        self.transop = TransOp_expm(M=self.transop_cfg.dictionary_size, N=backbone_feature_dim)

        self.coefficient_encoder = None
        if self.transop_cfg.enable_variational_inference:
            self.coefficient_encoder = VIEncoder(
                self.transop_cfg, backbone_feature_dim, self.transop_cfg.dictionary_size
            )

    def get_param_groups(self):
        param_list = [
            {
                "params": self.transop.parameters(),
                "lr": self.transop_cfg.transop_lr,
                "weight_decay": self.transop_cfg.transop_weight_decay,
                "disable_layer_adaptation": True,
            },
        ]
        if self.coefficient_encoder is not None:
            param_list.append({"params": self.coefficient_encoder.parameters()
                               "lr": self.transop_cfg.variational_encoder_lr,
                               "weight_decay": self.transop_cfg.variational_encoder_weight_decay})
        return param_list

    def forward(self, header_input: HeaderInput) -> HeaderOutput:
        z0, z1 = header_input.feature_0, header_input.feature_1
        distribution_data = None

        # If no target point was provided, return the features as a prediction
        if z1 is None:
            return HeaderOutput(z0, z1, distribution_data)

        # First infer coefficients for point pair
        if self.coefficient_encoder is None:
            c, _ = infer_coefficients(
                z0,
                z1,
                self.transop.get_psi(),
                self.transop_cfg.lambda_prior,
                max_iter=self.transop_cfg.fista_num_iterations,
                num_trials=self.transop_cfg.iter_variational_samples,
                device=z0.device,
            )
        else:
            distribution_data = self.coefficient_encoder(z0, z1)

            # If using best of many loss, use this trick to only differentiate through the coefficient with the lowest
            # L2 error.
            with torch.no_grad():
                noise_list = []
                loss_list = []
                for _ in range(
                    self.transop_cfg.total_variational_samples // self.transop_cfg.iter_variational_samples
                ):

                    # Generate a new noise sample
                    u = self.coefficient_encoder.draw_noise_samples(
                        len(z1), self.transop_cfg.iter_variational_samples, z1.device
                    )
                    c = self.coefficient_encoder.reparameterize(
                        distribution_data.shift.unsqueeze(1).repeat(1, self.transop_cfg.iter_variational_samples, 1),
                        distribution_data.log_scale.unsqueeze(1).repeat(
                            1, self.transop_cfg.iter_variational_samples, 1
                        ),
                        u,
                    )

                    # Estimate z1 with transport operators
                    with autocast(enabled=False):
                        z1_hat = self.transop(z0.detach().float().unsqueeze(-1), c.detach()).squeeze().transpose(0, 1)

                    # Perform max ELBO sampling to find the highest likelihood coefficient for each entry in the batch
                    transop_loss = (
                        F.mse_loss(
                            z1_hat,
                            z1.repeat(len(z1_hat), *torch.ones(z1.dim(), dtype=int)).detach(),
                            reduction="none",
                        )
                        .mean(dim=-1)
                        .transpose(0, 1)
                    )
                    noise_list.append(u)
                    loss_list.append(transop_loss)

                # Pick the best sample
                noise_list = torch.cat(noise_list, dim=1)
                loss_list = torch.cat(loss_list, dim=1)
                max_elbo = torch.argmin(loss_list, dim=1).detach()

                # Pick out best noise sample for each batch entry for reparameterization
                noise = noise_list[torch.arange(len(z0)), max_elbo]

            # Reparameterize with best noise sample (prevents backprop wrt all samples)
            c = self.coefficient_encoder.reparameterize(distribution_data.shift, distribution_data.log_scale, noise)
            distribution_data = DistributionData(c, *distribution_data[1:])

        # Matrix exponential not supported with float16
        with autocast(enabled=False):
            z1_hat = self.transop(z0.float().unsqueeze(-1), c).squeeze()

        return HeaderOutput(z1_hat, z1, distribution_data)
