import torch
import torch.nn as nn
from cs336_basics.transformer_block.embedding_layer import Embedding
from cs336_basics.transformer_block.transformer import TransformerBlock
from cs336_basics.transformer_block.rms_norm import RMSNorm
from cs336_basics.transformer_block.linear_layer import Linear

class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # Token embedding layer
        self.token_embedding = Embedding(vocab_size, d_model)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])

        # Final layer norm using RMSNorm
        self.ln_final = RMSNorm(d_model)

        # Output projection layer using custom Linear class
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Token embeddings
        token_embeddings = self.token_embedding(input_ids)

        # Pass through Transformer blocks
        x = token_embeddings
        for block in self.transformer_blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_final(x)

        # Output projection
        logits = self.lm_head(x)

        return logits