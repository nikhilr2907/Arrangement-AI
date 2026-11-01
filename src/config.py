from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import overload

import yaml

from src.data.serialization import integer, mapping, number, text


@dataclass(frozen=True)
class ModelConfig:
    feature_dims: tuple[int, int] = (768, 768)
    model_dim: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    history_bars: int = 16

    def __post_init__(self) -> None:
        if (
            self.model_dim < 1
            or self.heads < 1
            or self.model_dim % self.heads
            or self.layers < 1
            or self.history_bars < 1
            or not 0 <= self.dropout < 1
            or len(self.feature_dims) != 2
            or min(self.feature_dims) < 1
        ):
            raise ValueError("Invalid model dimensions, history or dropout")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    destination_weight: float = 1.0
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if (
            self.epochs < 1
            or self.batch_size < 1
            or self.learning_rate <= 0
            or self.weight_decay < 0
            or self.gradient_clip <= 0
            or self.destination_weight < 0
        ):
            raise ValueError("Invalid training configuration")


@overload
def load_config(path: Path, kind: type[ModelConfig]) -> ModelConfig: ...


@overload
def load_config(path: Path, kind: type[TrainingConfig]) -> TrainingConfig: ...


def load_config(
    path: Path, kind: type[ModelConfig] | type[TrainingConfig]
) -> ModelConfig | TrainingConfig:
    data = mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
    unknown = set(data) - {field.name for field in fields(kind)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    if kind is ModelConfig:
        return model_config_from_dict(data)
    defaults = TrainingConfig()
    parsed = TrainingConfig(
        epochs=integer(data.get("epochs", defaults.epochs)),
        batch_size=integer(data.get("batch_size", defaults.batch_size)),
        learning_rate=number(data.get("learning_rate", defaults.learning_rate)),
        weight_decay=number(data.get("weight_decay", defaults.weight_decay)),
        gradient_clip=number(data.get("gradient_clip", defaults.gradient_clip)),
        destination_weight=number(data.get("destination_weight", defaults.destination_weight)),
        seed=integer(data.get("seed", defaults.seed)),
        device=text(data.get("device", defaults.device)),
    )
    return parsed


def model_config_from_dict(data: dict[str, object]) -> ModelConfig:
    defaults = ModelConfig()
    raw_dims = data.get("feature_dims", defaults.feature_dims)
    if not isinstance(raw_dims, (list, tuple)) or len(raw_dims) != 2:
        raise ValueError("feature_dims must contain MIDI and audio feature widths")
    return ModelConfig(
        feature_dims=(integer(raw_dims[0]), integer(raw_dims[1])),
        model_dim=integer(data.get("model_dim", defaults.model_dim)),
        heads=integer(data.get("heads", defaults.heads)),
        layers=integer(data.get("layers", defaults.layers)),
        dropout=number(data.get("dropout", defaults.dropout)),
        history_bars=integer(data.get("history_bars", defaults.history_bars)),
    )
