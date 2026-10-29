import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from cs336_basics.configs.schema import load_config
from cs336_basics.src.model.transformer import TransformerLM
from cs336_basics.src.training.loss import cross_entropy
from cs336_basics.src.training.optim import AdamW, gradient_clipping
from cs336_basics.src.training.utils import get_batch, save_checkpoint, load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", required=True)
parser.add_argument("--out_dir", default=None)
parser.add_argument("--checkpoint_path", default=None)
args = parser.parse_args()

config = load_config(args.config_path)
model_config = config.model
train_config = config.training
optim_config = config.optimizer

if args.out_dir is None:
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(train_config.snapshotting_path) / run_name
else:
    out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
print(f"out_dir={out_dir}")

device = model_config.device or "mps"
model = TransformerLM.from_config(model_config).to(device)
model = torch.compile(model, backend="aot_eager")

optimizer = AdamW(
    params=model.parameters(),
    lr=optim_config.lr,
    betas=optim_config.betas,
    eps=optim_config.eps,
    weight_decay=optim_config.weight_decay,
)

train_dataset = np.load(train_config.train_dataset_path, mmap_mode="r")
valid_dataset = np.load(train_config.valid_dataset_path, mmap_mode="r")

step = 0

if args.checkpoint_path is not None:
    step = load_checkpoint(args.checkpoint_path, model, optimizer) + 1

while step < train_config.num_steps:
    x, y = get_batch(train_dataset, train_config.batch_size, train_config.context_length, device)
    optimizer.zero_grad()
    logits = model(x)
    logits = rearrange(logits, "b s v -> (b s) v")
    loss = cross_entropy(logits=logits, targets=y.view(-1))
    loss.backward()
    gradient_clipping(model.parameters(), train_config.grad_norm_clip)
    optimizer.step()

    print(f"{step=}, train_loss={loss.item()}")

    if step % train_config.snapshotting_every_num_steps == 0:
        save_checkpoint(model, optimizer, step, out_dir / f"chkpt_{step}")

    if step % train_config.validation_every_num_steps == 0:
        x_valid, y_valid = get_batch(valid_dataset, train_config.batch_size, train_config.context_length, device)
        model.eval()
        with torch.no_grad():
            val_logits = rearrange(model(x_valid), "b s v -> (b s) v")
            val_loss = cross_entropy(logits=val_logits, targets=y_valid.view(-1))
            print(f"{step=}, valid_loss={val_loss.item()}")
        model.train()
    step += 1
