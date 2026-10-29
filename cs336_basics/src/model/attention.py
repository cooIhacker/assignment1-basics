import math

import torch
from jaxtyping import Float, Bool
from torch import Tensor
from torch import nn
from einops import einsum, rearrange

from cs336_basics.src.model.layers import Linear, RotaryPositionalEmbedding


def softmax(x: Float[Tensor, " ..."], dimension: int) -> Float[Tensor, " ..."]:
    m = x.max(dim=dimension, keepdim=True).values
    denom_inv = 1 / torch.exp(x - m).sum(dim=dimension, keepdim=True)
    return torch.exp(x - m) * denom_inv


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    scale_factor: float = 1 / math.sqrt(Q.shape[-1])
    attn_scores = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys") * scale_factor
    if mask is not None:
        attn_scores.masked_fill_(~mask, float("-inf"))
    return softmax(attn_scores, -1) @ V


class CausalMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        apply_rope: bool = False,
        max_seq_len: int | None = None,
        theta: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale_factor = 1 / math.sqrt(self.d_head)
        self.W_qkv = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)
        self.apply_rope = apply_rope
        if apply_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device=device)

    def forward(
        self, x: Float[Tensor, " batch seq_len d_model"], token_positions=None
    ) -> Float[Tensor, " batch seq_len d_model"]:
        qkv: Tensor = self.W_qkv(x)
        qkv = rearrange(
            qkv,
            "batch seq_len (three num_heads d_head) -> batch num_heads seq_len d_head three",
            three=3,
            num_heads=self.num_heads,
            d_head=self.d_head,
        )
        q, k, v = qkv.unbind(dim=-1)
        if self.apply_rope:
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        attn_scores = (
            einsum(q, k, " ... q_seq_len d_head, ... k_seq_len d_head -> ... q_seq_len k_seq_len") * self.scale_factor
        )
        mask = torch.ones_like(attn_scores, dtype=bool).tril()
        attn_scores.masked_fill_(~mask, float("-inf"))
        attn_weights = softmax(attn_scores, -1)
        attn_out = einsum(attn_weights, v, " ... q_seq_len k_seq_len, ... k_seq_len d_head -> ... q_seq_len d_head")
        attn_out = rearrange(attn_out, " batch num_heads q_seq_len d_head -> batch q_seq_len (num_heads d_head)")

        return self.W_o(attn_out)
