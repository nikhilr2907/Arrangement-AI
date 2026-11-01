import numpy as np
import pytest
import torch

from src.config import ModelConfig
from src.contracts import (
    ActionKind,
    ArrangementRequest,
    Channel,
    Clip,
    Note,
    PlaybackAction,
    PreparedEpisode,
)
from src.features.cache import ClipFeatures, ExtractorSpec
from src.timing import regular_grid


@pytest.fixture(autouse=True)
def deterministic_cpu() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(7)


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        feature_dims=(8, 8), model_dim=16, heads=2, layers=1, dropout=0, history_bars=3
    )


@pytest.fixture
def episode() -> PreparedEpisode:
    keys = Clip("keys", regular_grid(2), (Note(60, 0, 480), Note(67, 1920, 2400)))
    drums = Clip("drums", regular_grid(1), (Note(36, 0, 240),), is_drum=True, role=1)
    request = ArrangementRequest((Channel("keys", keys), Channel("drums", drums)), regular_grid(4))

    def start(index: int) -> PlaybackAction:
        return PlaybackAction(ActionKind.START, index)

    keep, silence = PlaybackAction(ActionKind.KEEP), PlaybackAction(ActionKind.SILENCE)
    actions = ((start(0), start(0)), (keep, start(0)), (start(0), silence), (keep, keep))
    return PreparedEpisode(
        "fixture", "composition", "train", "constructed_fixture", request, actions
    )


@pytest.fixture
def features(episode: PreparedEpisode) -> dict[tuple[str, str], ClipFeatures]:
    rng = np.random.default_rng(4)
    spec = ExtractorSpec("test_fixture", "not_pretrained", "1", 8, 4)
    return {
        (channel.clip.clip_id, "midi"): ClipFeatures(
            rng.normal(size=(len(channel.clip.grid.bars), 4, 8)).astype(np.float32), spec
        )
        for channel in episode.request.channels
    }
