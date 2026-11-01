import os
from pathlib import Path

import numpy as np
import pytest

from src.contracts import Channel, Clip, Modality, Note
from src.features.musicbert import FIELD_SIZES, MusicBertExtractor, octuple_fields, octuple_ids
from src.timing import regular_grid


def test_octuple_fields_use_pretrained_dictionary() -> None:
    note = Note(60, 0, 480, 80)
    channel = Channel("piano", Clip("piano", regular_grid(1), (note,)))
    fields = octuple_fields(note, channel, 0)
    assert fields == (0, 0, 0, 60, 16, 20, 9, 35)
    assert all(4 <= token < 4 + sum(FIELD_SIZES) for token in octuple_ids(fields))


@pytest.mark.skipif(
    os.environ.get("RUN_EXTRACTOR_TESTS") != "1", reason="Optional MusicBERT runtime integration"
)
def test_native_musicbert_adapter_with_small_local_fixture(tmp_path: Path) -> None:
    from musicbert_hf import MusicBert, MusicBertConfig

    config = MusicBertConfig(
        vocab_size=5 + sum(FIELD_SIZES),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=128,
        pad_token_id=1,
        fairseq_position_embeddings=True,
    )
    # Random weights are solely an adapter contract fixture, never a training-data fallback.
    model = MusicBert(config)
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    model.save_pretrained(tmp_path, state_dict=state)
    extractor = MusicBertExtractor(str(tmp_path), phase_bins=4)
    clip = Clip("notes", regular_grid(2), (Note(60, 0, 3000),))
    result = extractor.encode(Channel("notes", clip))
    assert result.values.shape == (2, 4, 16) and np.isfinite(result.values).all()
    assert np.any(result.values[1] != 0)


@pytest.mark.skipif(
    os.environ.get("RUN_MERT_SMOKE") != "1", reason="Downloads real MERT checkpoint"
)
def test_real_mert_checkpoint_on_audio() -> None:
    from src.features.mert import MertExtractor

    extractor = MertExtractor(phase_bins=4)
    asset = Path("test_data_samples/melody_stems_Current.wav").resolve()
    clip = Clip("audio", regular_grid(1), audio_path=asset)
    values = extractor.encode(Channel("audio", clip, Modality.AUDIO)).values
    assert values.shape == (1, 4, 768) and np.isfinite(values).all()
    assert np.std(values) > 0
