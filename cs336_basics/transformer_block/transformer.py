import torch
import torch.nn as nn
from cs336_basics.transformer_block.rms_norm import RMSNorm
from cs336_basics.transformer_block.swiglu import SWiGLU
from cs336_basics.transformer_block.attention import MultiHeadedSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Layers
        self.ln1 = RMSNorm(d_model)
        self.mha = MultiHeadedSelfAttention(d_model, d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        self.swiglu = SWiGLU(d_model, d_ff)

        # Initialize weights for RMSNorm layers
        nn.init.ones_(self.ln1.gains)
        nn.init.ones_(self.ln2.gains)

        # Initialize weights for MultiHeadedSelfAttention
        init_std_mha = (2 / (d_model + d_model)) ** 0.5
        nn.init.trunc_normal_(self.mha.Q, mean=0.0, std=init_std_mha, a=-3*init_std_mha, b=3*init_std_mha)
        nn.init.trunc_normal_(self.mha.K, mean=0.0, std=init_std_mha, a=-3*init_std_mha, b=3*init_std_mha)
        nn.init.trunc_normal_(self.mha.V, mean=0.0, std=init_std_mha, a=-3*init_std_mha, b=3*init_std_mha)
        nn.init.trunc_normal_(self.mha.O, mean=0.0, std=init_std_mha, a=-3*init_std_mha, b=3*init_std_mha)

        # Initialize weights for SWiGLU
        init_std_swiglu = (2 / (d_model + d_ff)) ** 0.5
        nn.init.trunc_normal_(self.swiglu.w1, mean=0.0, std=init_std_swiglu, a=-3*init_std_swiglu, b=3*init_std_swiglu)
        nn.init.trunc_normal_(self.swiglu.w2, mean=0.0, std=init_std_swiglu, a=-3*init_std_swiglu, b=3*init_std_swiglu)
        nn.init.trunc_normal_(self.swiglu.w3, mean=0.0, std=init_std_swiglu, a=-3*init_std_swiglu, b=3*init_std_swiglu)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        # Apply first RMSNorm
        normed_features = self.ln1(in_features)

        # Apply MultiHeadedSelfAttention with RoPE
        attn_output = self.mha(normed_features, self.theta, self.max_seq_len)

        # Residual connection after attention
        residual1 = in_features + attn_output

        # Apply second RMSNorm
        normed_residual = self.ln2(residual1)

        # Apply SWiGLU
        ffn_output = self.swiglu(normed_residual)

        # Residual connection after feed-forward network
        output = residual1 + ffn_output

        return output