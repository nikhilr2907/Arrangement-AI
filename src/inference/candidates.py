from src.contracts import ActionKind, ArrangementRequest, Channel, PlaybackAction, PlaybackState
from src.inference.state import PlaybackStateMachine


def build_candidates(
    state: PlaybackState, channel: Channel, request: ArrangementRequest, bar: int
) -> tuple[PlaybackAction, ...]:
    machine = PlaybackStateMachine()
    destinations = [PlaybackAction(ActionKind.SILENCE)] if state.next_source_bar is not None else []
    destinations.extend(
        PlaybackAction(ActionKind.START, source)
        for source in machine.legal_starts(channel, request, bar)
        if source != state.next_source_bar
    )
    return tuple(destinations)
