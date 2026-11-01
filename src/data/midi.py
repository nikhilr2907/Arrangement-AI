from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import mido

from src.contracts import Clip, Controller, Meter, Note, TempoChange
from src.timing import grid_from_events


def read_midi(path: Path) -> tuple[Clip, ...]:
    """Read distinct MIDI parts, resolving pedal sustains and rejecting unsupported bends."""
    midi = mido.MidiFile(path)
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError("Asynchronous MIDI or SMPTE timing is unsupported")
    meters: dict[int, Meter] = {0: Meter()}
    tempos: dict[int, TempoChange] = {0: TempoChange(0, 120.0)}
    parts: list[tuple[str, int, bool, tuple[Note, ...], tuple[Controller, ...]]] = []
    end_tick = 0
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        programs: dict[int, int] = defaultdict(int)
        active: dict[tuple[int, int], deque[tuple[int, int]]] = defaultdict(deque)
        sustained: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        pedal: dict[int, bool] = defaultdict(bool)
        notes: dict[int, list[Note]] = defaultdict(list)
        controls: dict[int, list[Controller]] = defaultdict(list)
        name = f"track_{track_index}"
        for message in track:
            tick += int(message.time)
            if message.type == "track_name":
                name = str(message.name)
            elif message.type == "set_tempo":
                tempos[tick] = TempoChange(tick, float(mido.tempo2bpm(message.tempo)))
            elif message.type == "time_signature":
                meters[tick] = Meter(int(message.numerator), int(message.denominator))
            elif message.type == "program_change":
                channel = int(message.channel)
                if notes[channel] or any(
                    key[0] == channel and value for key, value in active.items()
                ):
                    if programs[channel] != message.program:
                        raise ValueError(
                            "Program changes within a sounding part require separate assets"
                        )
                programs[channel] = int(message.program)
            elif message.type == "pitchwheel" and message.pitch != 0:
                raise ValueError("Pitch bends require a controller-preserving backend")
            elif message.type == "control_change":
                channel = int(message.channel)
                if message.control != 64:
                    controls[channel].append(
                        Controller(tick, int(message.control), int(message.value))
                    )
                    continue
                down = message.value >= 64
                if pedal[channel] and not down:
                    for pitch, start, velocity in sustained.pop(channel, []):
                        if tick > start:
                            notes[channel].append(Note(pitch, start, tick, velocity))
                pedal[channel] = down
            elif message.type == "note_on" and message.velocity > 0:
                active[(int(message.channel), int(message.note))].append(
                    (tick, int(message.velocity))
                )
            elif message.type in {"note_on", "note_off"}:
                channel, pitch = int(message.channel), int(message.note)
                if active[(channel, pitch)]:
                    start, velocity = active[(channel, pitch)].popleft()
                    if pedal[channel]:
                        sustained[channel].append((pitch, start, velocity))
                    elif tick > start:
                        notes[channel].append(Note(pitch, start, tick, velocity))
        if any(active.values()):
            raise ValueError("Unclosed note-on events")
        for channel, held in sustained.items():
            for pitch, start, velocity in held:
                if tick > start:
                    notes[channel].append(Note(pitch, start, tick, velocity))
        end_tick = max(end_tick, tick)
        for channel, events in notes.items():
            if events:
                parts.append(
                    (
                        f"{track_index}:{channel}:{name}",
                        programs[channel],
                        channel == 9,
                        tuple(sorted(events, key=lambda n: (n.onset, n.pitch, n.offset))),
                        tuple(controls[channel]),
                    )
                )
    if not parts:
        raise ValueError("MIDI file has no complete note events")
    grid = grid_from_events(
        int(midi.ticks_per_beat),
        max(end_tick, 1),
        sorted(meters.items()),
        tuple(tempos[t] for t in sorted(tempos)),
    )
    return tuple(
        Clip(
            f"{path.stem}:{name}",
            grid,
            notes,
            program=program,
            is_drum=drum,
            role=1 if drum else 2 if 32 <= program <= 39 else 0,
            controllers=controllers,
        )
        for name, program, drum, notes, controllers in parts
    )


def crop_controllers(
    controls: tuple[Controller, ...], start: int, end: int, target: int, ratio: float = 1.0
) -> tuple[Controller, ...]:
    """Replay controller state at entry and retain changes inside the source crop."""
    initial = {event.control: event.value for event in controls if event.tick <= start}
    result = [Controller(target, control, value) for control, value in sorted(initial.items())]
    result.extend(
        Controller(target + round((event.tick - start) * ratio), event.control, event.value)
        for event in controls
        if start < event.tick < end
    )
    return tuple(result)
