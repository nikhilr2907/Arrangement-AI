from __future__ import annotations

import random
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from torch.utils.data import Dataset

from src.contracts import (
    ActionKind,
    ArrangementRequest,
    BarPlayback,
    PlaybackAction,
    PlaybackState,
    PreparedEpisode,
)
from src.inference.state import PlaybackStateMachine


@dataclass(frozen=True)
class DecisionExample:
    request: ArrangementRequest
    bar: int
    channel_index: int
    states: tuple[PlaybackState, ...]
    history: tuple[tuple[BarPlayback, ...], ...]
    committed: tuple[BarPlayback, ...]
    target: PlaybackAction | None = None
    phase: int = 0


def replay(
    episode: PreparedEpisode,
) -> tuple[tuple[tuple[PlaybackState, ...], ...], tuple[tuple[BarPlayback, ...], ...]]:
    machine = PlaybackStateMachine()
    states = tuple(PlaybackState() for _ in episode.request.channels)
    snapshots: list[tuple[PlaybackState, ...]] = []
    history: list[tuple[BarPlayback, ...]] = []
    for step, actions in enumerate(episode.actions):
        bar, phase = divmod(step, episode.request.steps_per_bar)
        snapshots.append(states)
        transitions = [
            machine.apply(state, action, channel, episode.request, bar, phase=phase)
            for state, action, channel in zip(
                states, actions, episode.request.channels, strict=True
            )
        ]
        states = tuple(state for state, _ in transitions)
        history.append(tuple(item for _, item in transitions))
    return tuple(snapshots), tuple(history)


class ArrangementDecisionDataset(Dataset[DecisionExample]):
    def __init__(
        self,
        episodes: Sequence[PreparedEpisode],
        history_bars: int = 16,
        seed: int = 0,
        available_modalities: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.episodes = tuple(episodes)
        self.history_bars = history_bars
        self.seed = seed
        self.epoch = 0
        self.available_modalities = available_modalities or {}
        self._replay_cache: dict[
            int, tuple[tuple[tuple[PlaybackState, ...], ...], tuple[tuple[BarPlayback, ...], ...]]
        ] = {}
        self.ends: list[int] = []
        total = 0
        for episode in episodes:
            total += len(episode.actions) * len(episode.request.channels)
            self.ends.append(total)
        if not total:
            raise ValueError("No prepared arrangement decisions")

    def __len__(self) -> int:
        return self.ends[-1]

    def _replay(
        self, episode_index: int
    ) -> tuple[tuple[tuple[PlaybackState, ...], ...], tuple[tuple[BarPlayback, ...], ...]]:
        if episode_index not in self._replay_cache:
            if len(self._replay_cache) >= 8:
                self._replay_cache.pop(next(iter(self._replay_cache)))
            self._replay_cache[episode_index] = replay(self.episodes[episode_index])
        return self._replay_cache[episode_index]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> DecisionExample:
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect_right(self.ends, index)
        offset = index - (self.ends[episode_index - 1] if episode_index else 0)
        episode = self.episodes[episode_index]
        if self.available_modalities:
            from src.contracts import Modality

            rng = random.Random(f"modalities:{self.seed}:{self.epoch}:{episode.episode_id}")
            mixed_channels = tuple(
                replace(
                    c,
                    modality=Modality(
                        rng.choice(
                            self.available_modalities.get(c.clip.clip_id, (c.modality.value,))
                        )
                    ),
                )
                for c in episode.request.channels
            )
            episode = replace(episode, request=replace(episode.request, channels=mixed_channels))
        count = len(episode.request.channels)
        step, order_index = divmod(offset, count)
        steps = episode.request.steps_per_bar
        bar, phase = divmod(step, steps)
        order = list(range(count))
        random.Random(f"{self.seed}:{self.epoch}:{episode.episode_id}:{step}").shuffle(order)
        channel = order[order_index]
        states, playback = self._replay(episode_index)
        return DecisionExample(
            episode.request,
            bar,
            channel,
            states[step],
            playback[max(0, bar - self.history_bars) * steps : step],
            tuple(playback[step][i] for i in order[:order_index]),
            episode.actions[step][channel],
            phase,
        )


def refine_episode(episode: PreparedEpisode, steps_per_bar: int) -> PreparedEpisode:
    """Expand coarse actions without inventing edits between observed boundaries."""
    previous = episode.request.steps_per_bar
    if steps_per_bar < previous or steps_per_bar % previous:
        raise ValueError("New resolution must be an integer refinement of the existing grid")
    factor = steps_per_bar // previous
    actions: list[tuple[PlaybackAction, ...]] = []
    for row in episode.actions:
        actions.append(
            tuple(replace(action, source_phase=action.source_phase * factor) for action in row)
        )
        actions.extend(
            tuple(PlaybackAction(ActionKind.KEEP) for _ in row) for _ in range(factor - 1)
        )
    return replace(
        episode,
        request=replace(episode.request, steps_per_bar=steps_per_bar),
        actions=tuple(actions),
    )
