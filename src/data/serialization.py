from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import cast

from src.contracts import (
    ActionKind,
    ArrangementRequest,
    BarSpan,
    Channel,
    Clip,
    Controller,
    Meter,
    Modality,
    MusicalGrid,
    Note,
    PlaybackAction,
    PreparedEpisode,
    TempoChange,
)


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Expected a JSON object with string keys")
    return cast(dict[str, object], value)


def sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Expected a JSON array")
    return cast(list[object], value)


def integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Expected an integer")
    return value


def text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected a string")
    return value


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a number")
    return float(value)


def grid_from_dict(value: object) -> MusicalGrid:
    data = mapping(value)
    bars: list[BarSpan] = []
    for raw in sequence(data["bars"]):
        bar = mapping(raw)
        meter = mapping(bar["meter"])
        bars.append(
            BarSpan(
                integer(bar["start"]),
                integer(bar["end"]),
                Meter(integer(meter["numerator"]), integer(meter["denominator"])),
            )
        )
    tempos = tuple(
        TempoChange(integer(mapping(t)["tick"]), number(mapping(t)["bpm"]))
        for t in sequence(data["tempos"])
    )
    return MusicalGrid(integer(data["ppq"]), tuple(bars), tempos)


def clip_from_dict(value: object, base: Path) -> Clip:
    data = mapping(value)
    audio = data.get("audio_path")
    notes = tuple(
        Note(
            integer(mapping(n)["pitch"]),
            integer(mapping(n)["onset"]),
            integer(mapping(n)["offset"]),
            integer(mapping(n)["velocity"]),
        )
        for n in sequence(data.get("notes", []))
    )
    drum = data.get("is_drum", False)
    if not isinstance(drum, bool):
        raise ValueError("is_drum must be boolean")
    return Clip(
        text(data["clip_id"]),
        grid_from_dict(data["grid"]),
        notes,
        (base / text(audio)).resolve() if audio is not None else None,
        integer(data.get("program", 0)),
        drum,
        integer(data.get("role", 0)),
        tuple(
            (number(sequence(span)[0]), number(sequence(span)[1]))
            for span in sequence(data.get("audio_spans", []))
        ),
        tuple(
            Controller(
                integer(mapping(event)["tick"]),
                integer(mapping(event)["control"]),
                integer(mapping(event)["value"]),
            )
            for event in sequence(data.get("controllers", []))
        ),
    )


def request_from_dict(value: object, base: Path) -> ArrangementRequest:
    data = mapping(value)
    channels: list[Channel] = []
    for raw in sequence(data["channels"]):
        channel = mapping(raw)
        channels.append(
            Channel(
                text(channel["channel_id"]),
                clip_from_dict(channel["clip"], base),
                Modality(text(channel.get("modality", "midi"))),
            )
        )
    return ArrangementRequest(
        tuple(channels), grid_from_dict(data["grid"]), integer(data.get("seed", 0))
    )


def episode_from_dict(value: object, base: Path) -> PreparedEpisode:
    data = mapping(value)
    actions = tuple(
        tuple(
            PlaybackAction(
                ActionKind(integer(mapping(action)["kind"])),
                integer(mapping(action)["source_bar"])
                if mapping(action).get("source_bar") is not None
                else None,
            )
            for action in sequence(row)
        )
        for row in sequence(data["actions"])
    )
    return PreparedEpisode(
        text(data["episode_id"]),
        text(data["composition_id"]),
        text(data["split"]),
        text(data["provenance"]),
        request_from_dict(data["request"], base),
        actions,
    )


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, default=_json_default, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_episodes(path: Path) -> list[PreparedEpisode]:
    return [
        episode_from_dict(value, path.parent)
        for value in sequence(json.loads(path.read_text(encoding="utf-8")))
    ]


def read_request(path: Path) -> ArrangementRequest:
    return request_from_dict(json.loads(path.read_text(encoding="utf-8")), path.parent)
