from bisect import bisect_right
from collections.abc import Sequence

from src.contracts import BarSpan, Clip, Meter, MusicalGrid, TempoChange


def regular_grid(
    bars: int, bpm: float = 120.0, meter: Meter | None = None, ppq: int = 480
) -> MusicalGrid:
    meter = meter or Meter()
    length, remainder = divmod(ppq * meter.numerator * 4, meter.denominator)
    if bars < 1 or length < 1 or remainder:
        raise ValueError("Grid requires complete bars representable at the selected PPQ")
    spans = tuple(BarSpan(i * length, (i + 1) * length, meter) for i in range(bars))
    return MusicalGrid(ppq, spans, (TempoChange(0, bpm),))


def tick_to_seconds(tick: int, grid: MusicalGrid) -> float:
    if tick < 0:
        raise ValueError("Ticks cannot be negative")
    seconds = 0.0
    for index, tempo in enumerate(grid.tempos):
        if tick <= tempo.tick:
            break
        end = grid.tempos[index + 1].tick if index + 1 < len(grid.tempos) else tick
        seconds += (min(tick, end) - tempo.tick) * 60.0 / (grid.ppq * tempo.bpm)
    return seconds


def seconds_to_tick(seconds: float, grid: MusicalGrid) -> int:
    import math

    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Seconds must be finite and nonnegative")
    for index, tempo in enumerate(grid.tempos):
        start_seconds = tick_to_seconds(tempo.tick, grid)
        if index + 1 == len(grid.tempos) or seconds < tick_to_seconds(
            grid.tempos[index + 1].tick, grid
        ):
            return tempo.tick + round((seconds - start_seconds) * grid.ppq * tempo.bpm / 60)
    raise ValueError("Invalid tempo map")


def bar_seconds(index: int, grid: MusicalGrid) -> tuple[float, float]:
    bar = grid.bars[index]
    return tick_to_seconds(bar.start, grid), tick_to_seconds(bar.end, grid)


def source_audio_seconds(clip: Clip, index: int) -> tuple[float, float]:
    return clip.audio_spans[index] if clip.audio_spans else bar_seconds(index, clip.grid)


def compatible(
    source: MusicalGrid, source_bar: int, target: MusicalGrid, target_bar: int, audio: bool
) -> bool:
    if not 0 <= source_bar < len(source.bars):
        return False
    src, dst = source.bars[source_bar], target.bars[target_bar]
    if (
        src.meter != dst.meter
        or (src.end - src.start) * target.ppq != (dst.end - dst.start) * source.ppq
    ):
        return False
    if audio:
        a, b = bar_seconds(source_bar, source)
        c, d = bar_seconds(target_bar, target)
        return abs((b - a) - (d - c)) <= 0.002
    return True


def grid_from_events(
    ppq: int,
    end_tick: int,
    meters: Sequence[tuple[int, Meter]],
    tempos: Sequence[TempoChange],
) -> MusicalGrid:
    meter_map = dict(meters)
    meter_map.setdefault(0, Meter())
    tempo_map = {tempo.tick: tempo for tempo in tempos}
    tempo_map.setdefault(0, TempoChange(0, 120.0))
    boundaries = sorted(meter_map)
    bars: list[BarSpan] = []
    cursor = 0
    while cursor < end_tick:
        meter = meter_map[boundaries[bisect_right(boundaries, cursor) - 1]]
        length, remainder = divmod(ppq * meter.numerator * 4, meter.denominator)
        if remainder or not length:
            raise ValueError("Meter cannot be represented at this PPQ")
        if any(cursor < position < cursor + length for position in boundaries):
            raise ValueError("Mid-bar meter changes are unsupported")
        bars.append(BarSpan(cursor, cursor + length, meter))
        cursor += length
    return MusicalGrid(ppq, tuple(bars), tuple(tempo_map[t] for t in sorted(tempo_map)))
