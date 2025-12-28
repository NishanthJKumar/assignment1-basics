import torch
import torch.nn as nn
from einops import rearrange


from cs336_basics.transformer_block.softmax import SoftMax
from cs336_basics.transformer_block.rope import RoPE

def scaled_dot_product_attention(Q_BQA: torch.Tensor, K_BKA: torch.Tensor, V_BVD: torch.Tensor, mask_BQK: torch.Tensor | None) -> torch.Tensor:
    attention_dim_size = Q_BQA.shape[-1]
    norm_att_BQK = torch.einsum('...QA,...KA->...QK', Q_BQA, K_BKA) / (attention_dim_size ** 0.5)
    if mask_BQK is not None:
        norm_att_BQK = torch.where(mask_BQK == True, norm_att_BQK, -float('inf'))
    softmax = SoftMax()
    softmax_scores_BQK = softmax(norm_att_BQK,dimension=-1)
    return torch.einsum('...QK,...KD->...QD', softmax_scores_BQK, V_BVD)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, d_in: int, d_model: int, num_heads: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.H = d_model//num_heads

        # init big q, k and V
        self.Q = nn.Parameter(
            torch.empty(self.d_model, d_in, device=device, dtype=dtype)
        )
        self.K = nn.Parameter(
            torch.empty(self.d_model, d_in, device=device, dtype=dtype)
        )
        self.V = nn.Parameter(
            torch.empty(self.d_model, d_in, device=device, dtype=dtype)
        )
        self.O = nn.Parameter(
            torch.empty(self.d_model, self.d_model, device=device, dtype=dtype)
        )
        # Make sure to use standard initialization
        init_std = (2 / (self.d_model + d_in)) ** 0.5
        nn.init.trunc_normal_(self.Q, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)
        nn.init.trunc_normal_(self.K, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)
        nn.init.trunc_normal_(self.V, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)

        o_init_std = (2 / (self.d_model + d_model)) ** 0.5
        nn.init.trunc_normal_(self.O, mean=0.0, std=init_std, a=-3*o_init_std, b=3*o_init_std)


    def forward(self, x_BCI: torch.Tensor, theta: float | None = None, max_seq_len: float | None = None) -> torch.Tensor:
        C = x_BCI.shape[-2]
        
        Q_BCA = torch.einsum('...CI,AI -> ...CA', x_BCI, self.Q)
        K_BCA = torch.einsum('...CI,AI -> ...CA', x_BCI, self.K)
        V_BCA = torch.einsum('...CI,AI -> ...CA', x_BCI, self.V)
        
        Q_BCNH = torch.reshape(Q_BCA, (-1, C, self.num_heads, self.H))
        K_BCNH = torch.reshape(K_BCA, (-1, C, self.num_heads, self.H))
        V_BCNH = torch.reshape(V_BCA, (-1, C, self.num_heads, self.H))

        Q_BNCH = rearrange(Q_BCNH, '... C N H -> ... N C H')
        K_BNCH = rearrange(K_BCNH, '... C N H -> ... N C H')
        V_BNCH = rearrange(V_BCNH, '... C N H -> ... N C H')

        if theta is not None and max_seq_len is not None:
            rope = RoPE(theta, self.H, max_seq_len)
            Q_BNCH = rope(Q_BNCH)
            K_BNCH = rope(K_BNCH)

        C = x_BCI.shape[-2]
        causal_mask = torch.tril(torch.ones((C, C))) == 1
        concated_att_BNCH = scaled_dot_product_attention(Q_BNCH, K_BNCH, V_BNCH, causal_mask)
        concated_att_BCA = rearrange(concated_att_BNCH, "... N C H -> ... C (N H)")
        return torch.einsum('AB,...CB->...CA', self.O, concated_att_BCA)

        
