from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path


class Modality(StrEnum):
    MIDI = "midi"
    AUDIO = "audio"


class ActionKind(IntEnum):
    KEEP = 0
    SILENCE = 1
    START = 2


@dataclass(frozen=True)
class Meter:
    numerator: int = 4
    denominator: int = 4

    def __post_init__(self) -> None:
        if self.numerator < 1 or self.denominator < 1:
            raise ValueError("Meter values must be positive")
        if self.denominator & (self.denominator - 1):
            raise ValueError("Meter denominator must be a power of two")


@dataclass(frozen=True)
class TempoChange:
    tick: int
    bpm: float

    def __post_init__(self) -> None:
        import math

        if self.tick < 0 or not math.isfinite(self.bpm) or self.bpm <= 0:
            raise ValueError("Tempo requires a nonnegative tick and positive finite BPM")


@dataclass(frozen=True)
class BarSpan:
    start: int
    end: int
    meter: Meter = Meter()

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError("Bar boundaries must be nonnegative and increasing")


@dataclass(frozen=True)
class MusicalGrid:
    ppq: int
    bars: tuple[BarSpan, ...]
    tempos: tuple[TempoChange, ...] = (TempoChange(0, 120.0),)

    def __post_init__(self) -> None:
        if self.ppq < 1 or not self.bars or not self.tempos or self.tempos[0].tick != 0:
            raise ValueError("A grid requires PPQ, bars, and a tempo starting at zero")
        if self.bars[0].start != 0:
            raise ValueError("Source grids start at tick zero; represent pickups explicitly")
        for previous, current in zip(self.bars, self.bars[1:], strict=False):
            if previous.end != current.start:
                raise ValueError("Bars must be contiguous")
        if any(a.tick >= b.tick for a, b in zip(self.tempos, self.tempos[1:], strict=False)):
            raise ValueError("Tempo changes must be strictly ordered")


@dataclass(frozen=True)
class Note:
    pitch: int
    onset: int
    offset: int
    velocity: int = 80

    def __post_init__(self) -> None:
        if not (0 <= self.pitch <= 127 and 1 <= self.velocity <= 127):
            raise ValueError("Invalid MIDI pitch or velocity")
        if not 0 <= self.onset < self.offset:
            raise ValueError("A note requires onset < offset")


@dataclass(frozen=True)
class Controller:
    tick: int
    control: int
    value: int

    def __post_init__(self) -> None:
        if self.tick < 0 or not 0 <= self.control <= 127 or not 0 <= self.value <= 127:
            raise ValueError("Invalid MIDI controller")


@dataclass(frozen=True)
class Clip:
    clip_id: str
    grid: MusicalGrid
    notes: tuple[Note, ...] = ()
    audio_path: Path | None = None
    program: int = 0
    is_drum: bool = False
    role: int = 0
    audio_spans: tuple[tuple[float, float], ...] = ()
    controllers: tuple[Controller, ...] = ()

    def __post_init__(self) -> None:
        if not self.clip_id or not 0 <= self.program <= 127 or not 0 <= self.role < 16:
            raise ValueError("Invalid clip identity, instrument or role")
        if any(note.offset > self.grid.bars[-1].end for note in self.notes):
            raise ValueError("Notes extend beyond the clip grid")
        if self.audio_spans and (
            len(self.audio_spans) != len(self.grid.bars)
            or any(not 0 <= a < b for a, b in self.audio_spans)
        ):
            raise ValueError("Audio span mapping must cover every source bar")


@dataclass(frozen=True)
class Channel:
    channel_id: str
    clip: Clip
    modality: Modality = Modality.MIDI

    def __post_init__(self) -> None:
        if not self.channel_id:
            raise ValueError("Channel identity cannot be empty")
        if self.modality == Modality.AUDIO and self.clip.audio_path is None:
            raise ValueError("Audio channels require an audio asset")


@dataclass(frozen=True)
class PlaybackAction:
    kind: ActionKind
    source_bar: int | None = None
    source_phase: int = 0

    def __post_init__(self) -> None:
        if self.kind == ActionKind.START:
            if self.source_bar is None or self.source_bar < 0:
                raise ValueError("START requires a nonnegative source bar")
        elif self.source_bar is not None or self.source_phase:
            raise ValueError("Only START has a source position")
        if self.source_phase < 0:
            raise ValueError("Source phase cannot be negative")


@dataclass(frozen=True)
class PlaybackState:
    next_source_bar: int | None = None
    next_source_phase: int = 0

    def __post_init__(self) -> None:
        if self.next_source_bar is not None and self.next_source_bar < 0:
            raise ValueError("Playback cursors cannot be negative")
        if self.next_source_phase < 0 or (self.next_source_bar is None and self.next_source_phase):
            raise ValueError("Invalid source phase")


@dataclass(frozen=True)
class BarPlayback:
    channel_id: str
    destination_bar: int
    source_bar: int | None
    action: PlaybackAction
    probability: float = 1.0
    destination_phase: int = 0
    source_phase: int = 0
    steps_per_bar: int = 1


@dataclass(frozen=True)
class Placement:
    channel_id: str
    source_start_bar: int
    destination_start_bar: int
    bar_count: int
    source_start_phase: int = 0
    destination_start_phase: int = 0
    steps_per_bar: int = 1
    step_count: int | None = None

    @property
    def length_steps(self) -> int:
        return (
            self.step_count if self.step_count is not None else self.bar_count * self.steps_per_bar
        )


@dataclass(frozen=True)
class ArrangementRequest:
    channels: tuple[Channel, ...]
    grid: MusicalGrid
    seed: int = 0
    steps_per_bar: int = 1

    def __post_init__(self) -> None:
        ids = [channel.channel_id for channel in self.channels]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("An arrangement requires uniquely named channels")
        if self.steps_per_bar < 1:
            raise ValueError("steps_per_bar must be positive")
        for grid in (self.grid, *(channel.clip.grid for channel in self.channels)):
            if any((bar.end - bar.start) % self.steps_per_bar for bar in grid.bars):
                raise ValueError("Decision steps must be exactly representable in musical ticks")


@dataclass(frozen=True)
class PreparedEpisode:
    episode_id: str
    composition_id: str
    split: str
    provenance: str
    request: ArrangementRequest
    actions: tuple[tuple[PlaybackAction, ...], ...]

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("Unknown dataset split")
        if len(self.actions) != len(self.request.grid.bars) * self.request.steps_per_bar:
            raise ValueError("Action rows must match destination decision steps")
        if any(len(row) != len(self.request.channels) for row in self.actions):
            raise ValueError("Each action row must cover all channels")
        if any(
            action.source_phase >= self.request.steps_per_bar
            for row in self.actions
            for action in row
        ):
            raise ValueError("Action source phase is outside the decision grid")


@dataclass(frozen=True)
class ArrangementResult:
    request: ArrangementRequest
    playback: tuple[tuple[BarPlayback, ...], ...]
    placements: tuple[Placement, ...]
