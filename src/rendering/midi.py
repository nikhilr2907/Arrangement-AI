from pathlib import Path

import mido

from src.contracts import ArrangementResult, Modality
from src.data.midi import crop_controllers
from src.timing import step_tick


def export_midi(result: ArrangementResult, path: Path) -> None:
    request = result.request
    midi = mido.MidiFile(ticks_per_beat=request.grid.ppq)
    tempo_track = mido.MidiTrack()
    midi.tracks.append(tempo_track)
    meta_events = [
        (tempo.tick, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo.bpm)))
        for tempo in request.grid.tempos
    ]
    previous_meter = None
    for bar in request.grid.bars:
        if bar.meter != previous_meter:
            meta_events.append(
                (
                    bar.start,
                    mido.MetaMessage(
                        "time_signature",
                        numerator=bar.meter.numerator,
                        denominator=bar.meter.denominator,
                    ),
                )
            )
            previous_meter = bar.meter
    cursor = 0
    for tick, event in sorted(meta_events, key=lambda item: item[0]):
        tempo_track.append(event.copy(time=tick - cursor))
        cursor = tick
    non_drum = [index for index in range(16) if index != 9]
    pitched_count = 0
    for channel in request.channels:
        if channel.modality != Modality.MIDI:
            continue
        if not channel.clip.is_drum and pitched_count >= len(non_drum):
            raise ValueError("Standard MIDI export supports at most fifteen pitched channels")
        port = 9 if channel.clip.is_drum else non_drum[pitched_count]
        pitched_count += int(not channel.clip.is_drum)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=channel.channel_id))
        track.append(mido.Message("program_change", channel=port, program=channel.clip.program))
        events: list[tuple[int, int, int, int]] = []
        controls: list[tuple[int, int, int, int]] = []
        clip = channel.clip
        for placement in result.placements:
            if placement.channel_id != channel.channel_id:
                continue
            steps = request.steps_per_bar
            source_step = placement.source_start_bar * steps + placement.source_start_phase
            destination_step = (
                placement.destination_start_bar * steps + placement.destination_start_phase
            )
            start = step_tick(clip.grid, source_step, steps)
            end = step_tick(clip.grid, source_step + placement.length_steps, steps)
            target = step_tick(request.grid, destination_step, steps)
            controls.extend(
                (event.tick, 2, event.control, event.value)
                for event in crop_controllers(
                    clip.controllers, start, end, target, request.grid.ppq / clip.grid.ppq
                )
            )
            for note in clip.notes:
                left, right = max(note.onset, start), min(note.offset, end)
                if right <= left:
                    continue
                onset = target + round((left - start) * request.grid.ppq / clip.grid.ppq)
                offset = target + round((right - start) * request.grid.ppq / clip.grid.ppq)
                if offset > onset:
                    events.extend(
                        ((onset, 1, note.pitch, note.velocity), (offset, 0, note.pitch, 0))
                    )
        cursor = 0
        for tick, kind, pitch, velocity in sorted(
            events + controls, key=lambda event: (event[0], {0: 0, 2: 1, 1: 2}[event[1]], event[2])
        ):
            track.append(
                mido.Message(
                    "control_change",
                    channel=port,
                    control=pitch,
                    value=velocity,
                    time=tick - cursor,
                )
                if kind == 2
                else mido.Message(
                    "note_on" if kind else "note_off",
                    channel=port,
                    note=pitch,
                    velocity=velocity,
                    time=tick - cursor,
                )
            )
            cursor = tick
        track.append(
            mido.MetaMessage("end_of_track", time=max(0, request.grid.bars[-1].end - cursor))
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)
