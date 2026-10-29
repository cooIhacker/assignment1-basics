import os
import typing

import numpy as np
import torch
from torch import nn
import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(dataset) - context_length, (batch_size,))
    x = np.stack([dataset[start : start + context_length] for start in starts])
    y = np.stack([dataset[start + 1 : start + context_length + 1] for start in starts])
    return torch.from_numpy(x).long().to(device), torch.from_numpy(y).long().to(device)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    chkpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(chkpt, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer
):
    chkpt = torch.load(src)
    model.load_state_dict(chkpt["model"])
    optimizer.load_state_dict(chkpt["optimizer"])
    return chkpt["iteration"]
