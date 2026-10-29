import torch
from jaxtyping import Float
from torch import Tensor


def cross_entropy(logits: Float[Tensor, "batch vocab_size"], targets: Float[Tensor, "batch_size"]) -> Float[Tensor, ""]:
    max_logit = logits.max(dim=-1, keepdim=True).values
    target_logits = logits[torch.arange(logits.shape[0]), targets]
    return ((max_logit - target_logits) + torch.log(torch.exp(logits - max_logit).sum(dim=-1))).mean()
