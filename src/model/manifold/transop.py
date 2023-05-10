import torch
import torch.nn as nn


class TransOp_expm(nn.Module):
    def __init__(self, M=6, N=3, var=1e0, stable_init=False, real_range=1.0e-4, imag_range=0.3, dict_count=1):
        super(TransOp_expm, self).__init__()
        init_var = var / N

        if stable_init:
            self.psi = nn.Parameter(torch.zeros((M, dict_count, N, N)), requires_grad=True)
            real = (torch.rand((M, dict_count)) - 0.5) * real_range
            imag = (torch.rand((M, dict_count)) - 0.5) * imag_range
            for i in range(0, N, 2):
                self.psi.data[..., i, i] = real
                self.psi.data[..., i + 1, i] = imag
                self.psi.data[..., i, i + 1] = -imag
                self.psi.data[..., i + 1, i + 1] = real
        else:
            self.psi = nn.Parameter(torch.mul(torch.randn((M, dict_count, N, N)), init_var), requires_grad=True)
        self.M = M
        self.N = N
        self.dict_count = dict_count

    def forward(self, x, c, transop_grad=True):
        if transop_grad:
            psi_use = self.psi
        else:
            psi_use = self.psi.detach()  
        T = torch.matrix_exp(torch.einsum("...m,mspk->...spk", c, psi_use))
        out = (T @ x.view(*x.shape[:-1], self.dict_count, -1, 1))
        out = out.view(*out.shape[:-3], x.shape[-1])
        return out

    def set_coefficients(self, c):
        self.c = c

    def get_psi(self):
        return self.psi.data

    def set_psi(self, psi_input):
        self.psi.data = psi_input
