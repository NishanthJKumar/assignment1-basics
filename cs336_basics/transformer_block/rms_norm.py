import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gains = nn.Parameter(
            torch.ones(self.d_model, device=device, dtype=dtype)
        )

    def forward(self, x_BCA: torch.Tensor) -> torch.Tensor:
        in_dtype = x_BCA.dtype
        x_BCA = x_BCA.to(torch.float32)
        rms_BC = torch.sqrt(1/self.d_model * torch.sum(torch.square(x_BCA), dim=-1) + self.eps)
        g = self.gains.to(torch.float32)
        ret_BCA = (x_BCA / rms_BC[:, :, None]) * g[None, None, :]
        return ret_BCA.to(in_dtype)