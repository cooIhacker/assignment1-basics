import torch
from torch import nn
from jaxtyping import Float, Int64
from torch import Tensor

from cs336_basics.src.model.layers import Embedding, Linear, RMSNorm, SwiGLU
from cs336_basics.src.model.attention import CausalMultiheadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = CausalMultiheadSelfAttention(d_model, num_heads, True, max_seq_len, theta, device, dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        y = x + self.mha(self.norm1(x))
        return y + self.swiglu(self.norm2(y))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.emb = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    @classmethod
    def from_config(cls, CONFIG):
        return cls(
            vocab_size=CONFIG.vocab_size,
            context_length=CONFIG.context_length,
            d_model=CONFIG.d_model,
            num_layers=CONFIG.num_layers,
            num_heads=CONFIG.num_heads,
            d_ff=CONFIG.d_ff,
            rope_theta=CONFIG.rope_theta,
            device=CONFIG.device,
            dtype=CONFIG.dtype,
        )

    def forward(self, inputs_ids: Int64[Tensor, "..."]) -> Float[Tensor, "... vocab_size"]:
        x = self.emb(inputs_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))
