import torch
import torch.nn as nn

class SWiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.w1 = nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )
        self.w2 = nn.Parameter(
            torch.empty(d_model, d_ff, device=device, dtype=dtype)
        )
        self.w3 = nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )
        init_std = (2 / (d_model + d_ff)) ** 0.5
        nn.init.trunc_normal_(self.w1, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)

    def forward(self, x_BCA: torch.Tensor) -> torch.Tensor:
        w1x_BCF = torch.einsum('...A,FA->...F', x_BCA, self.w1)
        siluw1x_BCF = w1x_BCF * torch.sigmoid(w1x_BCF)
        w3x_BCF = torch.einsum('...A,FA->...F', x_BCA, self.w3)
        return torch.einsum('AF,...F->...A', self.w2, siluw1x_BCF * w3x_BCF)