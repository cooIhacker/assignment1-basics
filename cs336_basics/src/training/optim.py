from collections.abc import Callable, Iterable
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["m"] = torch.zeros_like(p.data)
                self.state[p]["v"] = torch.zeros_like(p.data)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, betas, eps, weight_decay = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                lr_t = lr * (1 - betas[1] ** t) ** 0.5 / (1 - betas[0] ** t)
                grad = p.grad.data
                p.data -= lr * weight_decay * p.data
                state["m"] = betas[0] * state["m"] + (1 - betas[0]) * grad
                state["v"] = betas[1] * state["v"] + (1 - betas[1]) * grad**2
                p.data -= lr_t * state["m"] / (state["v"] ** 0.5 + eps)
                state["t"] = t + 1
        return loss


def get_lr_cosine_schedule(step: int, max_lr: float, min_lr: float, warmup: int, cosine_cycle: int) -> int:
    if step < warmup:
        return max_lr * step / warmup
    elif step <= cosine_cycle:
        return min_lr + 0.5 * (1 + math.cos(math.pi * (step - warmup) / (cosine_cycle - warmup))) * (max_lr - min_lr)
    else:
        return min_lr


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    g_norm = sum(torch.sum(p.grad**2) for p in parameters if p.grad is not None) ** 0.5
    if g_norm >= max_l2_norm:
        scale = max_l2_norm / (g_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad *= scale
