import torch
import torch.nn as nn


class TransOp_expm(nn.Module):
    def __init__(self, M=6, N=3, var=2e-3):
        super(TransOp_expm, self).__init__()
        self.psi = nn.Parameter(torch.mul(torch.randn((M, N, N)), var), requires_grad=True)
        self.M = M
        self.N = N

    def forward(self, x, c):
        if len(c.shape) == 2:
            T = torch.einsum("bm,mpk->bpk", c, self.psi)
        else:
            T = torch.einsum("bsm,mpk->bspk", c, self.psi)
            x = x.unsqueeze(1)
        out = torch.matrix_exp(T) @ x
        return out

    def set_coefficients(self, c):
        self.c = c

    def get_psi(self):
        return self.psi.data

    def set_psi(self, psi_input):
        self.psi.data = psi_input
