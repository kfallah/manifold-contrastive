import torch
import torch.nn as nn


class TransOp_expm(nn.Module):
    def __init__(self, M=6, N=3, var=1e0, stable_init=False):
        super(TransOp_expm, self).__init__()
        init_var = var / N
        if stable_init:
            self.psi = nn.Parameter(torch.zeros((M, N, N)), requires_grad=True)
            for i in range(0, self.psi.shape[1], 2):
                real = (torch.rand(len(self.psi)) - 0.5) * 2e-4
                imag = (torch.rand(len(self.psi)) - 0.5) * 2
                self.psi.data[:, i, i] = real
                self.psi.data[:, i + 1, i] = imag
                self.psi.data[:, i, i + 1] = -imag
                self.psi.data[:, i + 1, i + 1] = real
        else:
            self.psi = nn.Parameter(torch.mul(torch.randn((M, N, N)), init_var), requires_grad=True)
        self.M = M
        self.N = N

    def forward(self, x, c, transop_grad=True):
        if transop_grad:
            psi_use = self.psi
        else:
            psi_use = self.psi.detach()
        if len(c.shape) == 2:
            T = torch.einsum("bm,mpk->bpk", c, psi_use)
        else:
            T = torch.einsum("sbm,mpk->sbpk", c, psi_use)
        out = torch.matrix_exp(T) @ x
        return out

    def set_coefficients(self, c):
        self.c = c

    def get_psi(self):
        return self.psi.data

    def set_psi(self, psi_input):
        self.psi.data = psi_input
