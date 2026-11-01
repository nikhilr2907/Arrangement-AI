import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import yaml

from src.cli import evaluate, generate, prepare, train
from src.config import ModelConfig, TrainingConfig, load_config
from src.contracts import PreparedEpisode
from src.data.serialization import write_json
from src.features.cache import ClipFeatures, FeatureStore
from src.features.inventory import FeatureInventory
from src.training.checkpoints import load_checkpoint, validate_extractors


def test_cli_cached_training_evaluation_and_clip_only_generation(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episodes = tmp_path / "episodes.json"
    write_json(
        episodes,
        [
            replace(episode, episode_id=split, composition_id=split, split=split)
            for split in ("train", "validation", "test")
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare", "--source", "observed", "--root", str(episodes), "--output", str(episodes)],
    )
    prepare.main()
    store = FeatureStore(tmp_path / "features")
    entries = []
    for channel in episode.request.channels:
        cached = features[(channel.clip.clip_id, "midi")]
        key = store.key(channel, cached.spec)
        store.save(key, cached)
        entries.append(
            {
                "clip_id": channel.clip.clip_id,
                "modality": "midi",
                "key": key,
                "spec": asdict(cached.spec),
            }
        )
    inventory_path = tmp_path / "inventory.json"
    write_json(inventory_path, {"cache_root": str(store.root), "entries": entries})
    model_path, training_path = tmp_path / "model.yaml", tmp_path / "training.yaml"
    model_path.write_text(yaml.safe_dump(asdict(model_config)))
    training_path.write_text(yaml.safe_dump(asdict(TrainingConfig(epochs=1, batch_size=8))))
    checkpoint_dir = tmp_path / "checkpoints"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--episodes",
            str(episodes),
            "--inventory",
            str(inventory_path),
            "--model-config",
            str(model_path),
            "--config",
            str(training_path),
            "--output",
            str(checkpoint_dir),
        ],
    )
    train.main()
    checkpoint = checkpoint_dir / "best.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--episodes",
            str(episodes),
            "--inventory",
            str(inventory_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "evaluation.json"),
            "--rollouts",
            "1",
        ],
    )
    evaluate.main()
    metrics = json.loads((tmp_path / "evaluation.json").read_text())
    assert metrics["metrics"]["decisions"] == 8
    request_path = tmp_path / "request.json"
    write_json(request_path, episode.request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--request",
            str(request_path),
            "--inventory",
            str(inventory_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "generated"),
        ],
    )
    generate.main()
    assert (tmp_path / "generated/arrangement.mid").is_file()
    result = json.loads((tmp_path / "generated/arrangement.json").read_text())
    assert len(result["result"]["playback"]) == 4
    _, payload = load_checkpoint(checkpoint, torch.device("cpu"))
    inventory = FeatureInventory(inventory_path)
    validate_extractors(payload, {"midi": replace(inventory.specs["midi"], checkpoint="relocated")})
    with pytest.raises(ValueError, match="extractor differs"):
        validate_extractors(payload, {"midi": replace(inventory.specs["midi"], revision="other")})


def test_unknown_config_option_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "wrong.yaml"
    path.write_text("batch_siz: 4")
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_config(path, TrainingConfig)
