from dataclasses import replace

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
from src.timing import compatible_step


class PlaybackStateMachine:
    def legal_starts(
        self, channel: Channel, request: ArrangementRequest, bar: int, phase: int = 0
    ) -> tuple[int, ...]:
        steps = request.steps_per_bar
        if not 0 <= bar < len(request.grid.bars) or not 0 <= phase < steps:
            return ()
        return tuple(
            source
            for source in range(len(channel.clip.grid.bars) * steps)
            if compatible_step(
                channel.clip,
                source,
                request.grid,
                bar * steps + phase,
                steps,
                channel.modality == Modality.AUDIO,
            )
        )

    def keep_is_valid(
        self,
        state: PlaybackState,
        channel: Channel,
        request: ArrangementRequest,
        bar: int,
        phase: int = 0,
    ) -> bool:
        if (
            not 0 <= bar < len(request.grid.bars)
            or not 0 <= phase < request.steps_per_bar
            or not 0 <= state.next_source_phase < request.steps_per_bar
        ):
            return False
        if state.next_source_bar is None:
            return True
        source = state.next_source_bar * request.steps_per_bar + state.next_source_phase
        return source in self.legal_starts(channel, request, bar, phase)

    def apply(
        self,
        state: PlaybackState,
        action: PlaybackAction,
        channel: Channel,
        request: ArrangementRequest,
        bar: int,
        probability: float = 1.0,
        phase: int = 0,
    ) -> tuple[PlaybackState, BarPlayback]:
        steps = request.steps_per_bar
        if not 0 <= bar < len(request.grid.bars):
            raise ValueError("Destination bar is outside the musical grid")
        if not 0 <= phase < steps:
            raise ValueError("Destination phase is outside the decision grid")
        if action.kind == ActionKind.KEEP:
            if not self.keep_is_valid(state, channel, request, bar, phase):
                raise ValueError("KEEP would run past the clip or into incompatible timing")
            source_bar, source_phase = state.next_source_bar, state.next_source_phase
        elif action.kind == ActionKind.SILENCE:
            source_bar, source_phase = None, 0
        else:
            source = (action.source_bar or 0) * steps + action.source_phase
            if action.source_phase >= steps or source not in self.legal_starts(
                channel, request, bar, phase
            ):
                raise ValueError("START points to an invalid source position")
            source_bar, source_phase = action.source_bar, action.source_phase
        if source_bar is None:
            next_state = PlaybackState()
        else:
            next_bar, next_phase = divmod(source_bar * steps + source_phase + 1, steps)
            next_state = PlaybackState(next_bar, next_phase)
        return next_state, BarPlayback(
            channel.channel_id, bar, source_bar, action, probability, phase, source_phase, steps
        )


def coalesce(playback: tuple[tuple[BarPlayback, ...], ...]) -> tuple[Placement, ...]:
    placements: list[Placement] = []
    if not playback:
        return ()
    for channel_index in range(len(playback[0])):
        for row in playback:
            item = row[channel_index]
            if item.source_bar is None:
                continue
            steps = item.steps_per_bar
            previous = placements[-1] if placements else None
            if (
                previous is not None
                and previous.channel_id == item.channel_id
                and item.action.kind == ActionKind.KEEP
                and previous.destination_start_bar * steps
                + previous.destination_start_phase
                + previous.length_steps
                == item.destination_bar * steps + item.destination_phase
                and previous.source_start_bar * steps
                + previous.source_start_phase
                + previous.length_steps
                == item.source_bar * steps + item.source_phase
            ):
                count = previous.length_steps + 1
                placements[-1] = replace(
                    previous,
                    bar_count=(count + steps - 1) // steps,
                    step_count=count if steps > 1 else None,
                )
            else:
                placements.append(
                    Placement(
                        item.channel_id,
                        item.source_bar,
                        item.destination_bar,
                        1,
                        item.source_phase,
                        item.destination_phase,
                        steps,
                        1 if steps > 1 else None,
                    )
                )
    return tuple(placements)
