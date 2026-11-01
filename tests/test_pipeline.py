from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import ModelConfig, TrainingConfig
from src.contracts import PreparedEpisode
from src.data.collate import DecisionCollator
from src.data.dataset import ArrangementDecisionDataset
from src.features.cache import ClipFeatures, FeatureStore
from src.inference.generator import ArrangementGenerator
from src.models.arrangement import ArrangementModel
from src.rendering.midi import export_midi
from src.training.checkpoints import load_checkpoint, seed_everything
from src.training.trainer import Trainer


def test_training_checkpoint_generation(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
    tmp_path: Path,
) -> None:
    seed_everything(4)
    dataset = ArrangementDecisionDataset([episode], model_config.history_bars)
    collator = DecisionCollator(features, model_config)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    trainer = Trainer(ArrangementModel(model_config), TrainingConfig(epochs=2, learning_rate=0.003))
    metrics = trainer.fit(loader, loader, tmp_path)
    assert len(metrics) == 2 and np.isfinite(metrics[-1]["loss"])
    loaded, payload = load_checkpoint(tmp_path / "last.pt", torch.device("cpu"))
    trainer.model.eval()
    inputs = next(iter(loader)).inputs
    assert torch.equal(trainer.model(inputs).change, loaded(inputs).change)
    result = ArrangementGenerator(loaded, features).generate(episode.request)
    assert len(result.playback) == 4
    export_midi(result, tmp_path / "generated.mid")
    assert (tmp_path / "generated.mid").stat().st_size > 0
    resumed = Trainer(loaded, replace(trainer.config, epochs=3))
    resumed.resume(payload)
    assert resumed.start_epoch == 2


def test_model_can_overfit_nontrivial_edits(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
) -> None:
    dataset = ArrangementDecisionDataset([episode], 3)
    loader = DataLoader(dataset, batch_size=8, collate_fn=DecisionCollator(features, model_config))
    trainer = Trainer(
        ArrangementModel(model_config), TrainingConfig(epochs=1, learning_rate=0.01, weight_decay=0)
    )
    initial = trainer.run(loader, False)["loss"]
    for _ in range(60):
        trainer.run(loader, True)
    final = trainer.run(loader, False)
    assert final["loss"] < initial * 0.2 and final["accuracy"] >= 0.875


def test_feature_cache_roundtrip_and_invalidation(
    episode: PreparedEpisode, features: dict[tuple[str, str], ClipFeatures], tmp_path: Path
) -> None:
    channel = episode.request.channels[0]
    cached = features[(channel.clip.clip_id, "midi")]
    store = FeatureStore(tmp_path)
    key = store.key(channel, cached.spec)
    store.save(key, cached)
    assert np.array_equal(store.load(key).values, cached.values)
    assert key != store.key(channel, replace(cached.spec, revision="different"))


def test_windows_start_with_actual_playback_state(episode: PreparedEpisode) -> None:
    example = ArrangementDecisionDataset([episode], history_bars=1)[4]
    assert len(example.history) == 1
    assert example.states[0].next_source_bar == 2


def test_spawn_loader(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
) -> None:
    loader = DataLoader(
        ArrangementDecisionDataset([episode], 3),
        batch_size=2,
        collate_fn=DecisionCollator(features, model_config),
        num_workers=1,
        multiprocessing_context="spawn",
    )
    assert sum(len(batch.targets.change) for batch in loader) == 8
