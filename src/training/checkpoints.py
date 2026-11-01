from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from src.config import TrainingConfig, model_config_from_dict
from src.data.serialization import mapping
from src.features.cache import ExtractorSpec
from src.models.arrangement import ArrangementModel

SCHEMA_VERSION = 2


def validate_extractors(payload: dict[str, object], specs: Mapping[str, ExtractorSpec]) -> None:
    provenance = mapping(payload["provenance"])
    if "extractors" not in provenance:
        raise ValueError("Checkpoint has no extractor provenance")
    raw = provenance["extractors"]
    if not isinstance(raw, str):
        raise ValueError("Invalid checkpoint extractor provenance")
    trained = mapping(json.loads(raw))
    for mode, spec in specs.items():
        if mode not in trained:
            continue
        previous = mapping(trained[mode])
        # A local snapshot may move; its content revision must remain the same.
        current = asdict(spec)
        previous.pop("checkpoint", None)
        current.pop("checkpoint", None)
        if current != previous:
            raise ValueError(f"{mode} extractor differs from the checkpoint's training features")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(
    path: Path,
    model: ArrangementModel,
    optimizer: torch.optim.Optimizer,
    training: TrainingConfig,
    epoch: int,
    best_loss: float,
    modalities: set[str],
    provenance: dict[str, str],
) -> None:
    numpy_state = cast(tuple[str, NDArray[np.uint32], int, int, float], np.random.get_state())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_config": asdict(model.config),
        "training_config": asdict(training),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "modalities": sorted(modalities),
        "provenance": provenance,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[ArrangementModel, dict[str, object]]:
    payload = mapping(torch.load(path, map_location="cpu", weights_only=True))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Incompatible checkpoint; the step-context architecture requires retraining"
        )
    raw_config = mapping(payload["model_config"])
    config = model_config_from_dict(raw_config)
    model = ArrangementModel(config).to(device)
    model.load_state_dict(cast(dict[str, torch.Tensor], payload["model"]))
    model.eval()
    return model, payload


def restore_rng(payload: dict[str, object]) -> None:
    random.setstate(cast(tuple[object, ...], payload["python_rng"]))
    name, keys, pos, gaussian, cached = cast(
        tuple[str, list[int], int, int, float], payload["numpy_rng"]
    )
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(cast(torch.Tensor, payload["torch_rng"]))
    cuda_rng = cast(list[torch.Tensor], payload["cuda_rng"])
    if cuda_rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng)
