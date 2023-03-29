import torch
import torch.nn as nn


class TransOp_expm(nn.Module):
    def __init__(self, M=6, N=3, var=1e0, stable_init=False, real_range=1.0e-4, imag_range=0.3, dict_count=None):
        super(TransOp_expm, self).__init__()
        init_var = var / N
        if dict_count is None:
            size = (M, N, N)
            init_size = (M)
        else:
            size = (dict_count, M, N, N)
            init_size = (dict_count, M)

        if stable_init:
            self.psi = nn.Parameter(torch.zeros(size), requires_grad=True)
            real = (torch.rand(init_size) - 0.5) * real_range
            imag = (torch.rand(init_size) - 0.5) * imag_range
            for i in range(0, N, 2):
                self.psi.data[..., i, i] = real
                self.psi.data[..., i + 1, i] = imag
                self.psi.data[..., i, i + 1] = -imag
                self.psi.data[..., i + 1, i + 1] = real
        else:
            self.psi = nn.Parameter(torch.mul(torch.randn(size), init_var), requires_grad=True)
        self.M = M
        self.N = N
        self.dict_count = dict_count

    def forward(self, x, c, transop_grad=True):
        if transop_grad:
            psi_use = self.psi
        else:
            psi_use = self.psi.detach()
        
        if self.dict_count is None:
            if len(c.shape) == 2:
                T = torch.einsum("bm,mpk->bpk", c, psi_use)
            else:
                T = torch.einsum("bsm,mpk->bspk", c, psi_use)
            T = torch.matrix_exp(T)
            out = T @ x
        else:
            T = torch.einsum("bm,smpk->bspk", c, psi_use)
            T = torch.matrix_exp(T.reshape(-1, self.N, self.N)).reshape(len(x), self.dict_count, self.N, self.N)
            x = x.reshape(len(x), self.dict_count, self.N, 1)
            out = (T @ x).reshape(len(x), -1, 1)
        return out

    def set_coefficients(self, c):
        self.c = c

    def get_psi(self):
        return self.psi.data

    def set_psi(self, psi_input):
        self.psi.data = psi_input
