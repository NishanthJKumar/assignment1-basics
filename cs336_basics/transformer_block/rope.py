import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k

        # Precompute frequency tensor: theta^(-2i/d_k) for i in [0, d_k/2)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Precompute position embeddings for max sequence length
        positions = torch.arange(max_seq_len, device=device)
        freqs = torch.outer(positions, inv_freq)  # (max_seq_len, d_k//2)
        
        # Create cos and sin cache
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        seq_len = x.shape[-2]
        
        # Get cached values for current sequence length
        # cos_cached: (max_seq_len, d_k//2) -> (seq_len, d_k//2) -> (1, seq_len, d_k//2)
        cos = self.cos_cached[:seq_len].unsqueeze(0)  # (1, seq_len, d_k//2)
        sin = self.sin_cached[:seq_len].unsqueeze(0)  # (1, seq_len, d_k//2)
        
        # Split x into even and odd dimensions
        x1 = x[..., ::2]  # (..., seq_len, d_k//2) - even indices
        x2 = x[..., 1::2]  # (..., seq_len, d_k//2) - odd indices
        
        # Apply rotation: [cos * x1 - sin * x2, sin * x1 + cos * x2]
        # Each: (..., seq_len, d_k//2)
        rotated = torch.stack([
            x1 * cos - x2 * sin,  # (..., seq_len, d_k//2)
            x1 * sin + x2 * cos   # (..., seq_len, d_k//2)
        ], dim=-1)  # (..., seq_len, d_k//2, 2)
        
        # Interleave back to original shape
        return rotated.flatten(-2)  # (..., seq_len, d_k)