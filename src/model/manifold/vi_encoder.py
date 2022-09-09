import torch
import torch.nn as nn
import torch.nn.functional as F
from model.contrastive.config import TransportOperatorConfig
from model.type import DistributionData
from torch.distributions import gamma as gamma


class VIEncoder(nn.Module):
    def __init__(self, transop_cfg: TransportOperatorConfig, input_size: int, dictionary_size: int):
        super(VIEncoder, self).__init__()

        if transop_cfg.use_warmpup:
            self.warmup = 0.1
        else:
            self.warmup = 1.0
        self.scale_prior = transop_cfg.variational_scale_prior
        self.enc_type = transop_cfg.variational_encoder_type
        self.lambda_ = transop_cfg.lambda_prior
        self.num_samples = transop_cfg.num_coefficient_samples
        self.feat_dim = transop_cfg.variational_feature_dim
        self.threshold = True

        if self.enc_type == "mlp":
            self.enc = nn.Sequential(
                nn.Linear(2 * input_size, 4 * input_size),
                nn.BatchNorm1d(4 * input_size),
                nn.ReLU(),
                nn.Linear(4 * input_size, 4 * input_size),
                nn.BatchNorm1d(4 * input_size),
                nn.ReLU(),
                nn.Linear(4 * input_size, self.feat_dim),
            )
        elif self.enc_type == "lstm":
            self.enc = nn.LSTM(input_size, self.feat_dim, num_layers=1)
        else:
            raise NotImplementedError

        self.scale = nn.Linear(self.feat_dim, dictionary_size)
        self.shift = nn.Linear(self.feat_dim, dictionary_size)

    def ramp_hyperparams(self):
        self.warmup = 1.0

    def soft_threshold(self, z: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
        return F.relu(torch.abs(z) - lambda_) * torch.sign(z)

    def reparameterize(self, shift, logscale, u):
        scale = torch.exp(logscale)
        eps = -scale * torch.sign(u) * torch.log((1.0 - 2.0 * torch.abs(u)).clamp(min=1e-6))

        c = shift + eps * self.warmup
        if self.threshold:
            c_thresh = self.soft_threshold(eps.detach() * self.warmup, self.lambda_)
            non_zero = torch.nonzero(c_thresh, as_tuple=True)
            c_thresh[non_zero] = shift[non_zero] + c_thresh[non_zero]
            c = c + c_thresh - c.detach()

        return c

    def forward(self, x0, x1):
        if self.enc_type == "lstm":
            z = self.enc(torch.stack((x0, x1), dim=1))[0][:, -1]
        else:
            z = self.enc(torch.cat((x0, x1), dim=-1))

        logscale, shift = self.scale(z), self.shift(z)
        u = torch.rand_like(logscale.unsqueeze(1).repeat(1, self.num_samples, 1)) - 0.5
        c = self.reparameterize(
            shift.unsqueeze(1).repeat(1, self.num_samples, 1), logscale.unsqueeze(1).repeat(1, self.num_samples, 1), u
        )
        distribution_data = DistributionData(
            # Wrap this in a list to support DataParallel
            samples=c,
            log_scale=logscale,
            shift=shift,
            scale_prior=torch.ones_like(logscale) * self.scale_prior,
            shift_prior=torch.zeros_like(shift),
            noise=u,
        )

        return c, distribution_data
