import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float = 10000.0
    device: str | None = None
    dtype: str | None = None


@dataclass
class TrainingConfig:
    train_dataset_path: str
    valid_dataset_path: str
    batch_size: int
    context_length: int
    num_steps: int = 100
    grad_norm_clip: float = 1.0
    validation_every_num_steps: int = 10
    snapshotting_every_num_steps: int = 10
    snapshotting_path: str = "logs"


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01


@dataclass
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("rb") as file:
        raw = tomllib.load(file)

    model_raw = dict(raw["model"])
    training_raw = dict(raw["training"])
    optimizer_raw = dict(raw["optimizer"])

    model_raw.setdefault("context_length", training_raw["context_length"])
    if "betas" in optimizer_raw:
        optimizer_raw["betas"] = tuple(optimizer_raw["betas"])

    return ExperimentConfig(
        model=ModelConfig(**model_raw),
        training=TrainingConfig(**training_raw),
        optimizer=OptimizerConfig(**optimizer_raw),
    )
