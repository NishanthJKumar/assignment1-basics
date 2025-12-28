import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        init_std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('...I,OI -> ...O', x, self.W)
    
    def set_weights(self, w: torch.Tensor) -> None:
        self.W.data = w
