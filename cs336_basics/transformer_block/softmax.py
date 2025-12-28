import torch
import torch.nn as nn

class SoftMax(nn.Module):
    def __init__(self, device=None, dtype=None) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, dimension: int) -> torch.Tensor:
        x_diff_exp = torch.exp(x - torch.max(x, dim=dimension, keepdims=True).values)
        return x_diff_exp / torch.sum(x_diff_exp, dim=dimension, keepdims=True)