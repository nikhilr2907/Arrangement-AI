from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from src.config import ModelConfig, TrainingConfig
from src.contracts import (
    ActionKind,
    ArrangementRequest,
    ArrangementResult,
    BarPlayback,
    Channel,
    Clip,
    Meter,
    Modality,
    Note,
    PlaybackAction,
    PlaybackState,
    PreparedEpisode,
)
from src.data.collate import DecisionCollator, destination_coordinates, source_coordinates
from src.data.dataset import ArrangementDecisionDataset, refine_episode, replay
from src.data.midi import read_midi
from src.data.serialization import read_episodes, write_json
from src.features.cache import ClipFeatures
from src.inference.candidates import build_candidates
from src.inference.generator import ArrangementGenerator
from src.inference.state import PlaybackStateMachine, coalesce
from src.models.arrangement import ArrangementModel
from src.models.encoders import RotaryBlock
from src.rendering.audio import render_audio
from src.rendering.midi import export_midi
from src.timing import regular_grid
from src.training.checkpoints import load_checkpoint
from src.training.trainer import Trainer


def fragment_episode(audio: Path | None = None) -> PreparedEpisode:
    clip = Clip(
        "fragment",
        regular_grid(1),
        (Note(60, 0, 480), Note(64, 480, 960)),
        audio_path=audio,
    )
    channel = Channel("bass", clip, Modality.AUDIO if audio else Modality.MIDI)
    request = ArrangementRequest((channel,), regular_grid(1), steps_per_bar=4)
    start, keep = PlaybackAction(ActionKind.START, 0), PlaybackAction(ActionKind.KEEP)
    return PreparedEpisode(
        "fragment",
        "composition",
        "train",
        "constructed_fixture",
        request,
        ((start,), (keep,), (start,), (keep,)),
    )


def test_repeating_fragment_inside_one_bar_renders_correct_notes(tmp_path: Path) -> None:
    episode = fragment_episode()
    _, playback = replay(episode)
    result = ArrangementResult(episode.request, playback, coalesce(playback))
    assert [(p.destination_start_phase, p.length_steps) for p in result.placements] == [
        (0, 2),
        (2, 2),
    ]
    output = tmp_path / "repeat.mid"
    export_midi(result, output)
    assert read_midi(output)[0].notes == (
        Note(60, 0, 480),
        Note(64, 480, 960),
        Note(60, 960, 1440),
        Note(64, 1440, 1920),
    )


def test_audio_repeats_source_phase_without_fading_keep_boundaries(tmp_path: Path) -> None:
    rate = 8000
    audio = np.repeat(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), rate // 2)
    asset = tmp_path / "source.wav"
    sf.write(asset, audio, rate, subtype="FLOAT")
    episode = fragment_episode(asset)
    start, keep = PlaybackAction(ActionKind.START, 0, 2), PlaybackAction(ActionKind.KEEP)
    episode = replace(episode, actions=((start,), (keep,), (start,), (keep,)))
    _, playback = replay(episode)
    render_audio(
        ArrangementResult(episode.request, playback, coalesce(playback)), tmp_path / "out", rate
    )
    rendered, _ = sf.read(tmp_path / "out/channel_000.wav")
    assert rendered[[2000, 6000, 10000, 14000], 0] == pytest.approx([0.3, 0.4, 0.3, 0.4])
    assert rendered[3999, 0] == pytest.approx(0.3)
    assert rendered[4000, 0] == pytest.approx(0.4)
    assert rendered[7999, 0] == pytest.approx(0)
    assert rendered[8000, 0] == pytest.approx(0)


@pytest.mark.parametrize("meter,steps", [(Meter(4, 4), 16), (Meter(3, 4), 12), (Meter(6, 8), 12)])
def test_source_cursor_and_end_masks_work_in_other_meters(meter: Meter, steps: int) -> None:
    grid = regular_grid(1, meter=meter)
    channel = Channel("track", Clip("source", grid))
    request = ArrangementRequest((channel,), grid, steps_per_bar=steps)
    machine = PlaybackStateMachine()
    state, item = machine.apply(
        PlaybackState(), PlaybackAction(ActionKind.START, 0, steps - 1), channel, request, 0
    )
    assert item.source_phase == steps - 1
    assert state == PlaybackState(1, 0)
    assert not machine.keep_is_valid(state, channel, request, 0, 1)
    assert len(build_candidates(PlaybackState(), channel, request, 0)) == steps
    with pytest.raises(ValueError, match="KEEP"):
        machine.apply(state, PlaybackAction(ActionKind.KEEP), channel, request, 0, phase=1)


def test_fraction_grid_is_independent_of_meter_and_rejects_incompatible_source() -> None:
    channel = Channel("track", Clip("source", regular_grid(1, meter=Meter(3, 4))))
    request = ArrangementRequest((channel,), channel.clip.grid, steps_per_bar=16)
    assert len(PlaybackStateMachine().legal_starts(channel, request, 0)) == 16
    incompatible = replace(request, grid=regular_grid(1))
    assert PlaybackStateMachine().legal_starts(channel, incompatible, 0) == ()


def test_out_of_range_phases_are_not_silently_wrapped() -> None:
    episode = fragment_episode()
    channel = episode.request.channels[0]
    machine = PlaybackStateMachine()
    assert not machine.keep_is_valid(PlaybackState(0, 4), channel, episode.request, 0)
    assert machine.legal_starts(channel, episode.request, 0, 4) == ()
    with pytest.raises(ValueError, match="source position"):
        machine.apply(
            PlaybackState(), PlaybackAction(ActionKind.START, 0, 4), channel, episode.request, 0
        )


def test_refinement_preserves_coarse_source_trajectory_and_json(
    episode: PreparedEpisode,
    tmp_path: Path,
) -> None:
    fine = refine_episode(episode, 4)
    _, coarse = replay(episode)
    _, detailed = replay(fine)
    assert [[p.source_bar for p in row] for row in detailed[::4]] == [
        [p.source_bar for p in row] for row in coarse
    ]
    assert detailed[1][0].source_phase == 1
    path = tmp_path / "steps.json"
    write_json(path, [fine])
    assert read_episodes(path) == [fine]
    with pytest.raises(ValueError, match="refinement"):
        refine_episode(fine, 3)


def test_exact_position_tuples_and_history_prefix(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
) -> None:
    fine = refine_episode(episode, 4)
    example = ArrangementDecisionDataset([fine], 3)[12]
    assert (example.bar, example.phase) == (1, 2)
    assert destination_coordinates(example, 0, 2, 3) == pytest.approx(
        [-1 / 3, 0.5, 0, -1, 0.25, 0.25]
    )
    channel = fine.request.channels[0]
    assert source_coordinates(channel, 1, 1, 4) == pytest.approx([0.5, 0.25, 1, 0, 2 / 32])
    batch = DecisionCollator(features, model_config)([example])
    assert int(batch.inputs.playback_valid.sum()) == 6
    assert int(batch.inputs.history_valid.sum()) == 6
    assert not batch.inputs.history_valid[0, -1, 2:].any()


def test_future_and_undecided_step_actions_cannot_change_model_inputs(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
) -> None:
    fine = refine_episode(episode, 4)
    dataset = ArrangementDecisionDataset([fine], 3)
    example = dataset[4]
    assert not example.committed and example.phase == 2
    other = 1 - example.channel_index
    actions = list(fine.actions)
    row = list(actions[2])
    row[other] = PlaybackAction(ActionKind.SILENCE)
    actions[2] = tuple(row)
    changed = replace(fine, actions=tuple(actions))
    collator = DecisionCollator(features, model_config)
    a, b = (
        collator([example]).inputs,
        collator([ArrangementDecisionDataset([changed], 3)[4]]).inputs,
    )
    for field in fields(a):
        assert torch.equal(getattr(a, field.name), getattr(b, field.name)), field.name
    model = ArrangementModel(model_config).eval()
    assert torch.equal(model(a).change, model(b).change)


def test_committed_other_track_changes_current_context(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
) -> None:
    fine = refine_episode(episode, 4)
    example = ArrangementDecisionDataset([fine], 3)[4]
    other = fine.request.channels[1 - example.channel_index]
    playing = BarPlayback(
        other.channel_id,
        0,
        0,
        PlaybackAction(ActionKind.START, 0, 2),
        destination_phase=2,
        source_phase=2,
        steps_per_bar=4,
    )
    off = replace(
        playing, source_bar=None, source_phase=0, action=PlaybackAction(ActionKind.SILENCE)
    )
    collator = DecisionCollator(features, model_config)
    a = collator([replace(example, committed=(playing,))]).inputs
    b = collator([replace(example, committed=(off,))]).inputs
    assert not torch.equal(a.current, b.current)
    model = ArrangementModel(model_config).eval()
    assert not torch.allclose(model(a).change, model(b).change)


def test_global_context_ignores_track_storage_order(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
) -> None:
    fine = refine_episode(episode, 4)
    first = fine.request.channels[0]
    third = replace(first, channel_id="third", clip=replace(first.clip, clip_id="third", role=2))
    fine = replace(
        fine,
        request=replace(fine.request, channels=(*fine.request.channels, third)),
        actions=tuple((*row, row[0]) for row in fine.actions),
    )
    features = {**features, ("third", "midi"): features[(first.clip.clip_id, "midi")]}
    inputs = DecisionCollator(features, model_config)(
        [ArrangementDecisionDataset([fine], 3)[12]]
    ).inputs
    swapped = replace(
        inputs,
        history_global=inputs.history_global.flip(-1),
        history_channel_valid=inputs.history_channel_valid.flip(-1),
        history_roles=inputs.history_roles.flip(-1),
        current=inputs.current.flip(-1),
        current_status=inputs.current_status.flip(-1),
        current_roles=inputs.current_roles.flip(-1),
        current_valid=inputs.current_valid.flip(-1),
    )
    model = ArrangementModel(model_config).eval()
    assert torch.allclose(model(inputs).change, model(swapped).change, atol=1e-6)
    assert torch.allclose(model(inputs).destinations, model(swapped).destinations, atol=1e-6)


def test_variable_source_padding_does_not_change_a_decision(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
) -> None:
    fine = refine_episode(episode, 4)
    dataset = ArrangementDecisionDataset([fine], 3)
    first = next(dataset[i] for i in range(2) if dataset[i].channel_index == 1)
    second = next(dataset[i] for i in range(2) if dataset[i].channel_index == 0)
    collator = DecisionCollator(features, model_config)
    alone, padded = collator([first]).inputs, collator([first, second]).inputs
    assert padded.source.size(1) > alone.source.size(1)
    model = ArrangementModel(model_config).eval()
    assert torch.allclose(model(alone).change, model(padded).change[:1], atol=1e-6)
    assert torch.allclose(
        model(alone).destinations,
        model(padded).destinations[:1, : alone.destination.size(1)],
        atol=1e-6,
    )


def test_sixteen_step_grid_has_selectable_source_positions_and_finite_gradients(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
) -> None:
    fine = refine_episode(episode, 16)
    expanded = {
        key: ClipFeatures(np.repeat(value.values, 4, axis=1), replace(value.spec, phase_bins=16))
        for key, value in features.items()
    }
    example = ArrangementDecisionDataset([fine], 3)[0]
    inputs = DecisionCollator(expanded, model_config)([example]).inputs
    expected = len(fine.request.channels[example.channel_index].clip.grid.bars) * 16
    assert int(inputs.source_valid.sum()) == expected
    assert int(inputs.destination_valid.sum()) == expected
    model = ArrangementModel(model_config)
    output = model(inputs)
    (output.change.sum() + output.destinations.sum()).backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_rope_uses_relative_time_and_handles_padding(model_config: ModelConfig) -> None:
    block = RotaryBlock(model_config).eval()
    values = torch.randn(2, 5, model_config.model_dim)
    valid = torch.tensor([[False, False, True, True, True], [False] * 5])
    positions = torch.arange(5).expand(2, -1)
    a = block(values, valid, positions, causal=True)
    b = block(values, valid, positions + 4, causal=True)
    assert torch.isfinite(a).all()
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.equal(a[1], torch.zeros_like(a[1]))


def test_fine_context_training_checkpoint_and_seeded_rollout(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
    tmp_path: Path,
) -> None:
    fine = refine_episode(episode, 4)
    dataset = ArrangementDecisionDataset([fine], 3)
    loader = DataLoader(dataset, batch_size=8, collate_fn=DecisionCollator(features, model_config))
    trainer = Trainer(ArrangementModel(model_config), TrainingConfig(epochs=1))
    trainer.fit(loader, loader, tmp_path)
    model, _ = load_checkpoint(tmp_path / "last.pt", torch.device("cpu"))
    generator = ArrangementGenerator(model, features)
    first = generator.generate(fine.request, sample=True)
    assert first == generator.generate(fine.request, sample=True)
    assert len(first.playback) == 16
    assert all(len(row) == 2 for row in first.playback)
    assert all(p.steps_per_bar == 4 and p.length_steps > 0 for p in first.placements)


def test_resolution_must_be_supported_by_cached_features(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
) -> None:
    fine = refine_episode(episode, 16)
    with pytest.raises(ValueError, match="phase bins"):
        DecisionCollator(features, model_config)([ArrangementDecisionDataset([fine], 3)[0]])
