import torch
from torch import nn
from jaxtyping import Float, Int64
from torch import Tensor
from einops import einsum, rearrange


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        emb_table = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(emb_table, 0, 1, -3, 3)
        self.emb_table = nn.Parameter(emb_table)

    def forward(self, x: Int64[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        return self.emb_table[x]


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        weight = torch.empty(size=(out_features, in_features), device=device, dtype=dtype)
        sigm = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(weight, 0, sigm**2, -3 * sigm, 3 * sigm)
        self.weight = nn.Parameter(weight)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        self.scale = 1 / d_model
        self.gain = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_type = x.dtype
        x = x.to(torch.float32)
        inv_rms_x = 1 / (self.scale * torch.sum(x**2, dim=-1, keepdim=True) + self.eps) ** 0.5
        out = x * inv_rms_x * self.gain
        return out.to(in_type)


def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return torch.sigmoid(x) * x


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        d_ff = d_ff if d_ff else ((8 / 3 * d_model) // 64) * 64
        self.W_1 = Linear(d_model, d_ff, device, dtype)
        self.W_2 = Linear(d_ff, d_model, device, dtype)
        self.W_3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.W_2(silu(self.W_1(x)) * self.W_3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        thetas = theta ** (-2 * torch.arange(d_k // 2, device=device, dtype=torch.float32) / d_k)
        lengths = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(lengths, thetas)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        x1, x2 = rearrange(x, "... (d two) -> two ... d", two=2)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        return rearrange([y1, y2], "two ... d -> ... (d two)")
