from src.contracts import (
    ActionKind,
    ArrangementRequest,
    BarPlayback,
    Channel,
    Modality,
    Placement,
    PlaybackAction,
    PlaybackState,
)
from src.timing import compatible


class PlaybackStateMachine:
    def legal_starts(
        self, channel: Channel, request: ArrangementRequest, bar: int
    ) -> tuple[int, ...]:
        return tuple(
            source
            for source in range(len(channel.clip.grid.bars))
            if compatible(
                channel.clip.grid, source, request.grid, bar, channel.modality == Modality.AUDIO
            )
        )

    def keep_is_valid(
        self, state: PlaybackState, channel: Channel, request: ArrangementRequest, bar: int
    ) -> bool:
        return state.next_source_bar is None or state.next_source_bar in self.legal_starts(
            channel, request, bar
        )

    def apply(
        self,
        state: PlaybackState,
        action: PlaybackAction,
        channel: Channel,
        request: ArrangementRequest,
        bar: int,
        probability: float = 1.0,
    ) -> tuple[PlaybackState, BarPlayback]:
        if action.kind == ActionKind.KEEP:
            if not self.keep_is_valid(state, channel, request, bar):
                raise ValueError("KEEP would run past the clip or into incompatible timing")
            source = state.next_source_bar
        elif action.kind == ActionKind.SILENCE:
            source = None
        else:
            if action.source_bar not in self.legal_starts(channel, request, bar):
                raise ValueError("START points to an invalid source bar")
            source = action.source_bar
        next_state = PlaybackState(None if source is None else source + 1)
        return next_state, BarPlayback(channel.channel_id, bar, source, action, probability)


def coalesce(playback: tuple[tuple[BarPlayback, ...], ...]) -> tuple[Placement, ...]:
    placements: list[Placement] = []
    if not playback:
        return ()
    for channel_index in range(len(playback[0])):
        for row in playback:
            item = row[channel_index]
            if item.source_bar is None:
                continue
            previous = placements[-1] if placements else None
            if (
                previous is not None
                and previous.channel_id == item.channel_id
                and item.action.kind == ActionKind.KEEP
                and previous.destination_start_bar + previous.bar_count == item.destination_bar
                and previous.source_start_bar + previous.bar_count == item.source_bar
            ):
                placements[-1] = Placement(
                    previous.channel_id,
                    previous.source_start_bar,
                    previous.destination_start_bar,
                    previous.bar_count + 1,
                )
            else:
                placements.append(
                    Placement(item.channel_id, item.source_bar, item.destination_bar, 1)
                )
    return tuple(placements)
