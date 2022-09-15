import torch
import torch.nn as nn
import torch.nn.functional as F
from model.contrastive.config import TransportOperatorConfig
from model.public.wide_resnet import resnet20
from model.type import DistributionData
from torch.distributions import gamma as gamma


class VIEncoder(nn.Module):
    def __init__(self, transop_cfg: TransportOperatorConfig, input_size: int, dictionary_size: int):
        super(VIEncoder, self).__init__()

        if transop_cfg.use_warmpup:
            self.warmup = 0.1
        else:
            self.warmup = 1.0
        self.transop_cfg = transop_cfg
        self.scale_prior = transop_cfg.variational_scale_prior
        self.enc_type = transop_cfg.variational_encoder_type
        self.lambda_ = transop_cfg.lambda_prior
        self.num_samples = transop_cfg.iter_variational_samples
        self.feat_dim = transop_cfg.variational_feature_dim
        self.dictionary_size = dictionary_size
        self.threshold = True

        if transop_cfg.variational_use_features:
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
        else:
            self.enc = resnet20()
            self.enc_proj = nn.Linear(20, self.feat_dim)

        self.scale = nn.Linear(self.feat_dim, dictionary_size)
        self.shift = nn.Linear(self.feat_dim, dictionary_size)

    def ramp_hyperparams(self):
        self.warmup = 1.0

    def soft_threshold(self, z: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
        return F.relu(torch.abs(z) - lambda_) * torch.sign(z)

    def draw_noise_samples(self, batch_size, num_samples, device):
        return torch.rand((batch_size, num_samples, self.dictionary_size), device=device) - 0.5

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
        self.warmup += 5e-4
        if self.warmup > 1.0:
            self.warmup = 1.0

        if self.transop_cfg.variational_use_features:
            if self.enc_type == "lstm":
                z = self.enc(torch.stack((x0, x1), dim=1))[0][:, -1]
            else:
                z = self.enc(torch.cat((x0, x1), dim=-1))
        else:
            z0, z1 = self.enc(torch.cat((x0, x1), dim=0)[:, 0]).split(len(x0), dim=0)
            z = self.enc_proj(torch.cat((z0, z1), dim=-1))

        logscale, shift = self.scale(z), self.shift(z)
        distribution_data = DistributionData(
            # Wrap this in a list to support DataParallel
            samples=None,
            log_scale=logscale,
            shift=shift,
            scale_prior=torch.ones_like(logscale) * self.scale_prior,
            shift_prior=torch.zeros_like(shift),
        )

        return distribution_data
