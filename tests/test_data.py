from dataclasses import replace
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf
import yaml

from src.config import ModelConfig
from src.contracts import (
    ArrangementResult,
    Clip,
    Modality,
    Note,
    PreparedEpisode,
)
from src.data.collate import DecisionCollator
from src.data.dataset import ArrangementDecisionDataset, replay
from src.data.preparation import EpisodeBuilder, LakhSource, SlakhSource, composition_fingerprint
from src.features.cache import ClipFeatures, ExtractorSpec
from src.models.arrangement import ArrangementModel
from src.rendering.audio import render_audio
from src.timing import regular_grid


def _write_midi(path: Path, bars: int, pitch: int = 60) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("note_on", note=pitch, velocity=80),
                mido.Message("note_off", note=pitch, time=480),
                mido.MetaMessage("end_of_track", time=1920 * bars - 480),
            ]
        )
    )
    midi.save(path)


def test_lakh_deduplicates_and_reports_invalid_files(tmp_path: Path) -> None:
    (tmp_path / "a.mid").write_bytes(b"invalid MIDI")
    _write_midi(tmp_path / "b.mid", 2)
    (tmp_path / "c.mid").write_bytes((tmp_path / "b.mid").read_bytes())
    source = LakhSource(tmp_path)
    assert len(list(source.iter_compositions())) == 1
    assert len(source.failures) == 1


def test_slakh_pairs_assets_aligns_short_parts_and_reports_failures(tmp_path: Path) -> None:
    folder = tmp_path / "Track00001"
    (folder / "MIDI").mkdir(parents=True)
    (folder / "stems").mkdir()
    for source_id, bars in (("S00", 1), ("S01", 2)):
        _write_midi(folder / "MIDI" / f"{source_id}.mid", bars)
        sf.write(folder / "stems" / f"{source_id}.flac", np.zeros(32000), 8000)
    metadata = {
        "stems": {source: {"midi_saved": True, "audio_rendered": True} for source in ("S00", "S01")}
    }
    (folder / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    broken = tmp_path / "Track00002"
    broken.mkdir()
    (broken / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    source = SlakhSource(tmp_path)
    compositions = list(source.iter_compositions())
    assert len(compositions) == 1 and len(source.failures) == 1
    _, clips = compositions[0]
    assert [len(clip.grid.bars) for clip in clips] == [2, 2]
    assert all(clip.audio_path is not None and clip.audio_path.is_file() for clip in clips)
    EpisodeBuilder().build(composition_fingerprint(clips), clips)


def test_exact_reuse_mining_recovers_repetition() -> None:
    grid = regular_grid(4)
    notes = tuple(Note(60 + index % 2, index * 1920, index * 1920 + 480) for index in range(4))
    clips = (
        Clip("first", grid, notes),
        Clip("second", grid, tuple(replace(n, pitch=n.pitch - 12) for n in notes)),
    )
    episode = EpisodeBuilder().build("composition", clips)
    assert [len(channel.clip.grid.bars) for channel in episode.request.channels] == [2, 2]
    _, playback = replay(episode)
    assert [row[0].source_bar for row in playback] == [0, 1, 0, 1]


def test_boundary_spanning_notes_not_mislabeled_as_silence() -> None:
    clip = Clip("sustain", regular_grid(2), (Note(60, 0, 3000),))
    with pytest.raises(ValueError, match="boundary-spanning"):
        EpisodeBuilder().build("composition", (clip, replace(clip, clip_id="other")))


def test_composition_group_is_independent_of_part_order() -> None:
    first = Clip("a", regular_grid(1), (Note(60, 0, 480),))
    second = Clip("b", regular_grid(1), (Note(48, 0, 480),), program=32)
    assert composition_fingerprint((first, second)) == composition_fingerprint((second, first))


def test_mixed_batch_backpropagates_audio_projection(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
    tmp_path: Path,
) -> None:
    first = episode.request.channels[0]
    audio_clip = replace(first.clip, audio_path=tmp_path / "unused_cached_asset.wav")
    channels = (
        replace(first, clip=audio_clip, modality=Modality.AUDIO),
        episode.request.channels[1],
    )
    mixed = replace(episode, request=replace(episode.request, channels=channels))
    original = features[(first.clip.clip_id, "midi")]
    features[(first.clip.clip_id, "audio")] = ClipFeatures(
        original.values, replace(original.spec, name="audio_fixture")
    )
    batch = DecisionCollator(features, model_config)([ArrangementDecisionDataset([mixed], 3)[0]])
    model = ArrangementModel(model_config)
    output = model(batch.inputs)
    (output.change.sum() + output.destinations.sum()).backward()
    gradient = model.source.projections[1].weight.grad
    assert gradient is not None and bool((gradient != 0).any())


@pytest.mark.parametrize("discontinuous", [False, True])
def test_audio_fades_only_at_physical_cuts(
    episode: PreparedEpisode, tmp_path: Path, discontinuous: bool
) -> None:
    from src.contracts import ActionKind, PlaybackAction

    rate = 8000
    source = np.full((rate * 6, 2), 0.2, dtype=np.float32)
    source[:, 1] = 0.4
    asset = tmp_path / "source.wav"
    sf.write(asset, source, rate, subtype="FLOAT")
    first = episode.request.channels[0]
    clip = replace(
        first.clip,
        audio_path=asset,
        audio_spans=((0.0, 2.0), (4.0, 6.0)) if discontinuous else (),
    )
    channel = replace(first, clip=clip, modality=Modality.AUDIO)
    request = replace(episode.request, channels=(channel,), grid=regular_grid(2))
    labelled = replace(
        episode,
        request=request,
        actions=((PlaybackAction(ActionKind.START, 0),), (PlaybackAction(ActionKind.KEEP),)),
    )
    _, playback = replay(labelled)
    from src.inference.state import coalesce

    render_audio(
        ArrangementResult(request, playback, coalesce(playback)), tmp_path / "render", rate
    )
    output, _ = sf.read(tmp_path / "render" / "channel_000.wav", always_2d=True)
    if discontinuous:
        assert np.allclose(output[rate * 2 - 1 : rate * 2 + 1], 0)
    else:
        assert np.allclose(output[rate * 2 - 2 : rate * 2 + 2], [[0.2, 0.4]] * 4)


def test_extractor_spec_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="finite"):
        ClipFeatures(
            np.full((1, 4, 8), np.nan, dtype=np.float32),
            ExtractorSpec("fixture", "none", "1", 8, 4),
        )
