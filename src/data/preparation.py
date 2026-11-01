from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import yaml

from src.contracts import (
    ActionKind,
    ArrangementRequest,
    Channel,
    Clip,
    Controller,
    Modality,
    Note,
    PlaybackAction,
    PreparedEpisode,
)
from src.data.midi import crop_controllers, read_midi
from src.data.serialization import mapping
from src.timing import grid_from_events, source_audio_seconds

logger = logging.getLogger(__name__)


class ArrangementSource(Protocol):
    def iter_compositions(self) -> Iterator[tuple[str, tuple[Clip, ...]]]: ...


def composition_fingerprint(clips: tuple[Clip, ...]) -> str:
    rows = sorted(
        (
            clip.program,
            clip.is_drum,
            tuple((n.pitch, n.onset / clip.grid.ppq, n.offset / clip.grid.ppq) for n in clip.notes),
        )
        for clip in clips
    )
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def split_for(composition_id: str) -> str:
    bucket = int(hashlib.sha256(composition_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


class LakhSource:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.failures: list[tuple[str, str]] = []

    def iter_compositions(self) -> Iterator[tuple[str, tuple[Clip, ...]]]:
        seen: set[str] = set()
        for path in sorted(self.root.rglob("*")):
            if path.suffix.lower() not in {".mid", ".midi"}:
                continue
            try:
                clips = read_midi(path)
            except (OSError, ValueError, EOFError, KeyError) as error:
                self.failures.append((str(path), str(error)))
                logger.warning("Skipping %s: %s", path, error)
                continue
            fingerprint = composition_fingerprint(clips)
            if fingerprint not in seen:
                seen.add(fingerprint)
                yield fingerprint, clips


class SlakhSource:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.failures: list[tuple[str, str]] = []

    def iter_compositions(self) -> Iterator[tuple[str, tuple[Clip, ...]]]:
        for metadata in sorted(self.root.rglob("metadata.yaml")):
            if "omitted" in metadata.parts:
                continue
            try:
                clips = self._read_composition(metadata)
            except (OSError, ValueError, EOFError, KeyError, yaml.YAMLError) as error:
                self.failures.append((str(metadata), str(error)))
                logger.warning("Skipping %s: %s", metadata.parent, error)
                continue
            if clips:
                yield composition_fingerprint(clips), clips

    def _read_composition(self, metadata: Path) -> tuple[Clip, ...]:
        data = mapping(yaml.safe_load(metadata.read_text(encoding="utf-8")))
        clips: list[Clip] = []
        for source_id, source_raw in mapping(data["stems"]).items():
            source = mapping(source_raw)
            if not source.get("midi_saved", True) or not source.get("audio_rendered", True):
                continue
            parts = read_midi(metadata.parent / "MIDI" / f"{source_id}.mid")
            audio = metadata.parent / "stems" / f"{source_id}.flac"
            if len(parts) != 1 or not audio.is_file():
                raise ValueError(f"Unpaired Slakh source: {metadata.parent.name}/{source_id}")
            clips.append(
                replace(
                    parts[0],
                    clip_id=f"{metadata.parent.name}:{source_id}",
                    audio_path=audio.resolve(),
                )
            )
        if not clips:
            return ()
        reference = max(clips, key=lambda clip: clip.grid.bars[-1].end)
        if any(
            clip.grid.ppq != reference.grid.ppq
            or clip.grid.bars != reference.grid.bars[: len(clip.grid.bars)]
            or clip.grid.tempos != reference.grid.tempos
            for clip in clips
        ):
            raise ValueError("Slakh source timing maps disagree")
        return tuple(replace(clip, grid=reference.grid) for clip in clips)


def _bar_notes(clip: Clip, index: int) -> tuple[tuple[int, int, int, int], ...]:
    bar = clip.grid.bars[index]
    return tuple(
        (note.pitch, note.onset - bar.start, note.offset - bar.start, note.velocity)
        for note in clip.notes
        if note.onset < bar.end and note.offset > bar.start
    )


def _mine_channel(clip: Clip) -> tuple[Clip, tuple[PlaybackAction, ...]]:
    signatures = [_bar_notes(clip, index) for index in range(len(clip.grid.bars))]
    unique: dict[tuple[object, ...], int] = {}
    source_indices: list[int] = []
    mapped: list[int | None] = []
    for index, signature in enumerate(signatures):
        if not signature:
            mapped.append(None)
            continue
        bar = clip.grid.bars[index]
        key = (bar.meter, signature, crop_controllers(clip.controllers, bar.start, bar.end, 0))
        if key not in unique:
            unique[key] = len(source_indices)
            source_indices.append(index)
        mapped.append(unique[key])
    if not source_indices:
        raise ValueError("Cannot mine an empty part")
    # Only bars with no crossing notes are independently reusable in exact-bar mining.
    if any(
        start < 0 or end > clip.grid.bars[index].end - clip.grid.bars[index].start
        for index in source_indices
        for _, start, end, _ in signatures[index]
    ):
        raise ValueError(
            "Exact bar mining excludes boundary-spanning notes; use observed placements"
        )
    ppq = clip.grid.ppq
    notes: list[Note] = []
    controllers: list[Controller] = []
    cursor = 0
    from src.contracts import Meter, TempoChange

    meter_events: list[tuple[int, Meter]] = []
    for source in source_indices:
        bar = clip.grid.bars[source]
        meter_events.append((cursor, bar.meter))
        controllers.extend(crop_controllers(clip.controllers, bar.start, bar.end, cursor))
        notes.extend(
            Note(pitch, cursor + start, cursor + end, velocity)
            for pitch, start, end, velocity in signatures[source]
        )
        cursor += bar.end - bar.start
    grid = grid_from_events(ppq, cursor, meter_events, (TempoChange(0, clip.grid.tempos[0].bpm),))
    if len(clip.grid.tempos) > 1:
        raise ValueError("Exact-bar reuse preparation currently requires constant tempo")
    audio_spans = (
        tuple(source_audio_seconds(clip, index) for index in source_indices)
        if clip.audio_path
        else ()
    )
    mined = replace(
        clip, grid=grid, notes=tuple(notes), audio_spans=audio_spans, controllers=tuple(controllers)
    )
    actions: list[PlaybackAction] = []
    expected: int | None = None
    for source_position in mapped:
        if source_position == expected:
            action = PlaybackAction(ActionKind.KEEP)
        elif source_position is None:
            action = PlaybackAction(ActionKind.SILENCE)
        else:
            action = PlaybackAction(ActionKind.START, source_position)
        actions.append(action)
        expected = None if source_position is None else source_position + 1
    return mined, tuple(actions)


class EpisodeBuilder:
    def build(self, composition_id: str, clips: tuple[Clip, ...]) -> PreparedEpisode:
        if len(clips) < 2:
            raise ValueError("Arrangement training requires at least two parts")
        if any(clip.grid != clips[0].grid for clip in clips):
            raise ValueError("Composition parts must share a musical grid")
        mined = [
            _mine_channel(replace(clip, clip_id=f"{composition_id}:{index}"))
            for index, clip in enumerate(clips)
        ]
        channels = tuple(Channel(f"channel_{i}", clip) for i, (clip, _) in enumerate(mined))
        actions = tuple(
            tuple(channel_actions[t] for _, channel_actions in mined)
            for t in range(len(clips[0].grid.bars))
        )
        return PreparedEpisode(
            composition_id,
            composition_id,
            split_for(composition_id),
            "inferred_exact_bar_reuse",
            ArrangementRequest(channels, clips[0].grid),
            actions,
        )


def paired_episode(episode: PreparedEpisode, modality: Modality) -> PreparedEpisode:
    if modality == Modality.AUDIO and any(
        c.clip.audio_path is None for c in episode.request.channels
    ):
        raise ValueError("A paired audio episode requires audio for every source clip")
    channels = tuple(replace(c, modality=modality) for c in episode.request.channels)
    return replace(episode, request=replace(episode.request, channels=channels))
