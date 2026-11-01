from dataclasses import replace
from pathlib import Path

import mido
import pytest

from src.contracts import (
    ActionKind,
    ArrangementResult,
    Meter,
    Note,
    PlaybackAction,
    PlaybackState,
    PreparedEpisode,
    TempoChange,
)
from src.data.dataset import replay
from src.data.midi import read_midi
from src.data.serialization import read_episodes, write_json
from src.inference.candidates import build_candidates
from src.inference.state import PlaybackStateMachine, coalesce
from src.rendering.midi import export_midi
from src.timing import grid_from_events, regular_grid, seconds_to_tick, tick_to_seconds


def test_silence_does_not_advance(episode: PreparedEpisode) -> None:
    state, playback = PlaybackStateMachine().apply(
        PlaybackState(),
        PlaybackAction(ActionKind.KEEP),
        episode.request.channels[0],
        episode.request,
        0,
    )
    assert state.next_source_bar is None and playback.source_bar is None


def test_end_requires_restart_or_silence(episode: PreparedEpisode) -> None:
    channel = episode.request.channels[1]
    machine = PlaybackStateMachine()
    state = PlaybackState(1)
    assert not machine.keep_is_valid(state, channel, episode.request, 1)
    assert build_candidates(state, channel, episode.request, 1) == (
        PlaybackAction(ActionKind.SILENCE),
        PlaybackAction(ActionKind.START, 0),
    )
    with pytest.raises(ValueError, match="KEEP"):
        machine.apply(state, PlaybackAction(ActionKind.KEEP), channel, episode.request, 1)


def test_repeat_preserves_placement_boundary(episode: PreparedEpisode) -> None:
    _, playback = replay(episode)
    keys = [p for p in coalesce(playback) if p.channel_id == "keys"]
    assert [(p.source_start_bar, p.destination_start_bar, p.bar_count) for p in keys] == [
        (0, 0, 2),
        (0, 2, 2),
    ]


def test_piecewise_tempo_and_meter() -> None:
    grid = grid_from_events(
        480, 3360, [(0, Meter(3, 4)), (1440, Meter())], [TempoChange(0, 120), TempoChange(480, 60)]
    )
    assert tick_to_seconds(1440, grid) == pytest.approx(2.5)
    assert seconds_to_tick(2.5, grid) == 1440
    assert [bar.end - bar.start for bar in grid.bars] == [1440, 1920]
    with pytest.raises(ValueError, match="Mid-bar"):
        grid_from_events(480, 3840, [(0, Meter()), (100, Meter(3, 4))], [])


def test_manifest_round_trip(episode: PreparedEpisode, tmp_path: Path) -> None:
    path = tmp_path / "episodes.json"
    write_json(path, [episode])
    assert read_episodes(path) == [episode]


def test_sustained_note_is_not_retriggered_on_keep(
    episode: PreparedEpisode, tmp_path: Path
) -> None:
    first = episode.request.channels[0]
    clip = replace(first.clip, notes=(Note(60, 0, 3000),))
    request = replace(episode.request, channels=(replace(first, clip=clip),), grid=regular_grid(2))
    actions = ((PlaybackAction(ActionKind.START, 0),), (PlaybackAction(ActionKind.KEEP),))
    labelled = replace(episode, request=request, actions=actions)
    _, playback = replay(labelled)
    result = ArrangementResult(request, playback, coalesce(playback))
    output = tmp_path / "sustain.mid"
    export_midi(result, output)
    loaded = read_midi(output)
    assert loaded[0].notes == (Note(60, 0, 3000),)


def test_crop_entry_retriggers_entering_sustain(episode: PreparedEpisode, tmp_path: Path) -> None:
    first = episode.request.channels[0]
    request = replace(
        episode.request,
        channels=(replace(first, clip=replace(first.clip, notes=(Note(60, 0, 3000),))),),
        grid=regular_grid(1),
    )
    labelled = replace(episode, request=request, actions=((PlaybackAction(ActionKind.START, 1),),))
    _, playback = replay(labelled)
    output = tmp_path / "crop.mid"
    export_midi(ArrangementResult(request, playback, coalesce(playback)), output)
    assert read_midi(output)[0].notes == (Note(60, 0, 1080),)


def test_parser_resolves_sustain_pedal(tmp_path: Path) -> None:
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend(
        [
            mido.Message("control_change", control=64, value=127),
            mido.Message("note_on", note=60, velocity=80),
            mido.Message("note_off", note=60, time=240),
            mido.Message("control_change", control=64, value=0, time=240),
        ]
    )
    path = tmp_path / "pedal.mid"
    midi.save(path)
    assert read_midi(path)[0].notes == (Note(60, 0, 480),)
