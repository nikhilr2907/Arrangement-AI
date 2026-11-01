from src.contracts import ActionKind, ArrangementRequest, Channel, PlaybackAction, PlaybackState
from src.inference.state import PlaybackStateMachine


def build_candidates(
    state: PlaybackState, channel: Channel, request: ArrangementRequest, bar: int, phase: int = 0
) -> tuple[PlaybackAction, ...]:
    machine = PlaybackStateMachine()
    destinations = [PlaybackAction(ActionKind.SILENCE)] if state.next_source_bar is not None else []
    destinations.extend(
        PlaybackAction(ActionKind.START, *divmod(source, request.steps_per_bar))
        for source in machine.legal_starts(channel, request, bar, phase)
        if state.next_source_bar is None
        or source != (state.next_source_bar * request.steps_per_bar + state.next_source_phase)
    )
    return tuple(destinations)
