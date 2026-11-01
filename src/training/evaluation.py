from src.contracts import (
    ActionKind,
    ArrangementRequest,
    ArrangementResult,
    BarPlayback,
    PlaybackAction,
    PlaybackState,
    PreparedEpisode,
)
from src.data.dataset import replay
from src.inference.state import PlaybackStateMachine, coalesce


def loop_baseline(request: ArrangementRequest) -> ArrangementResult:
    machine = PlaybackStateMachine()
    states = [PlaybackState() for _ in request.channels]
    rows: list[tuple[BarPlayback, ...]] = []
    for bar in range(len(request.grid.bars)):
        row: list[BarPlayback] = []
        for index, channel in enumerate(request.channels):
            state = states[index]
            if state.next_source_bar is not None and machine.keep_is_valid(
                state, channel, request, bar
            ):
                action = PlaybackAction(ActionKind.KEEP)
            else:
                starts = machine.legal_starts(channel, request, bar)
                action = (
                    PlaybackAction(ActionKind.START, starts[0])
                    if starts
                    else PlaybackAction(ActionKind.SILENCE)
                )
            states[index], item = machine.apply(state, action, channel, request, bar)
            row.append(item)
        rows.append(tuple(row))
    playback = tuple(rows)
    return ArrangementResult(request, playback, coalesce(playback))


def rollout_metrics(
    result: ArrangementResult, target: PreparedEpisode | None = None
) -> dict[str, float]:
    items = [item for row in result.playback for item in row]
    count = max(1, len(items))
    metrics = {
        "silence_ratio": sum(item.source_bar is None for item in items) / count,
        "change_ratio": sum(item.action.kind != ActionKind.KEEP for item in items) / count,
        "mean_run_bars": sum(p.bar_count for p in result.placements)
        / max(1, len(result.placements)),
    }
    if target is not None:
        _, expected = replay(target)
        pairs = [
            (actual, truth)
            for actual_row, target_row in zip(result.playback, expected, strict=True)
            for actual, truth in zip(actual_row, target_row, strict=True)
        ]
        metrics["source_accuracy"] = sum(a.source_bar == b.source_bar for a, b in pairs) / count
        metrics["presence_accuracy"] = (
            sum((a.source_bar is None) == (b.source_bar is None) for a, b in pairs) / count
        )
    return metrics
