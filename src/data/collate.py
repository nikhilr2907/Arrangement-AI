import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import torch
from torch import Tensor

from src.config import ModelConfig
from src.contracts import ActionKind, Channel, PlaybackAction
from src.data.batch import DecisionBatch, DecisionTargets, ModelInputs
from src.data.dataset import DecisionExample
from src.features.cache import ClipFeatures, FeatureIdentity
from src.inference.candidates import build_candidates
from src.inference.state import PlaybackStateMachine


def source_coordinates(channel: Channel, bar: int | None, phase: int, steps: int) -> list[float]:
    if bar is None:
        return [0.0] * 5
    count = len(channel.clip.grid.bars)
    fraction = phase / steps
    return [
        bar / count,
        fraction,
        math.sin(2 * math.pi * fraction),
        math.cos(2 * math.pi * fraction),
        count / 32,
    ]


def destination_coordinates(
    example: DecisionExample, bar: int, phase: int, history_bars: int
) -> list[float]:
    meter = example.request.grid.bars[bar].meter
    fraction = phase / example.request.steps_per_bar
    return [
        (bar - example.bar) / history_bars,
        fraction,
        math.sin(2 * math.pi * fraction),
        math.cos(2 * math.pi * fraction),
        meter.numerator / 16,
        meter.denominator / 16,
    ]


class _FeatureBank:
    def __init__(
        self, features: Mapping[FeatureIdentity, ClipFeatures], config: ModelConfig, steps: int
    ) -> None:
        self.features, self.config, self.steps = features, config, steps
        self.bins = next(iter(features.values())).spec.phase_bins if features else 16
        if self.bins % steps:
            raise ValueError("Feature phase bins must be a multiple of steps_per_bar")
        self.midi = [
            np.zeros((self.bins, config.feature_dims[0]), dtype=np.float32) for _ in range(2)
        ]
        self.audio = [
            np.zeros((self.bins, config.feature_dims[1]), dtype=np.float32) for _ in range(2)
        ]
        self.metadata = [[float(mode), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for mode in (2, 3)]
        self.references: dict[tuple[str, str, int], int] = {}

    def reference(self, channel: Channel, source: int | None, phase: int = 0) -> int:
        if source is None:
            return 0
        if not 0 <= source < len(channel.clip.grid.bars) or not 0 <= phase < self.steps:
            return 1
        identity = (channel.clip.clip_id, channel.modality.value)
        key = (*identity, source)
        if key not in self.references:
            features = self.features[identity]
            mode = int(channel.modality.value == "audio")
            if (
                features.spec.feature_dim != self.config.feature_dims[mode]
                or features.spec.phase_bins != self.bins
            ):
                raise ValueError("Feature dimensions/preprocessing do not match the model")
            if len(features.values) != len(channel.clip.grid.bars):
                raise ValueError("Source bar count differs from feature cache")
            self.references[key] = len(self.metadata)
            self.midi.append(features.values[source] if mode == 0 else np.zeros_like(self.midi[0]))
            self.audio.append(
                features.values[source] if mode == 1 else np.zeros_like(self.audio[0])
            )
            bar, count = channel.clip.grid.bars[source], len(channel.clip.grid.bars)
            self.metadata.append(
                [
                    float(mode),
                    float(channel.clip.role),
                    source / count,
                    count / 32,
                    bar.meter.numerator / 16,
                    bar.meter.denominator / 16,
                    (bar.end - bar.start) / channel.clip.grid.ppq / 8,
                    channel.clip.grid.tempos[0].bpm / 240,
                ]
            )
        return 2 + (self.references[key] - 2) * self.steps + phase

    def tensors(self) -> tuple[Tensor, Tensor, Tensor]:
        return (
            torch.from_numpy(np.stack(self.midi)),
            torch.from_numpy(np.stack(self.audio)),
            torch.tensor(self.metadata),
        )


def _allocate(
    batch: int, history: int, channels: int, choices: int, sources: int, bank: _FeatureBank
) -> ModelInputs:
    midi, audio, metadata = bank.tensors()
    steps = bank.steps
    length = (history + 1) * steps
    return ModelInputs(
        feature_midi=midi,
        feature_audio=audio,
        feature_meta=metadata,
        resolution=torch.tensor(steps),
        history_global=torch.zeros((batch, history + 1, steps, channels), dtype=torch.long),
        history_channel_valid=torch.zeros((batch, history + 1, steps, channels), dtype=torch.bool),
        history_valid=torch.zeros((batch, history + 1, steps), dtype=torch.bool),
        history_meta=torch.zeros((batch, history + 1, steps, 6)),
        history_roles=torch.zeros((batch, channels), dtype=torch.long),
        playback_tokens=torch.zeros((batch, length), dtype=torch.long),
        playback_destination=torch.zeros((batch, length, 6)),
        playback_source=torch.zeros((batch, length, 5)),
        playback_valid=torch.zeros((batch, length), dtype=torch.bool),
        source=torch.zeros((batch, sources), dtype=torch.long),
        source_valid=torch.zeros((batch, sources), dtype=torch.bool),
        current=torch.zeros((batch, max(channels - 1, 1)), dtype=torch.long),
        current_status=torch.zeros((batch, max(channels - 1, 1)), dtype=torch.long),
        current_roles=torch.zeros((batch, max(channels - 1, 1)), dtype=torch.long),
        current_valid=torch.zeros((batch, max(channels - 1, 1)), dtype=torch.bool),
        keep=torch.zeros(batch, dtype=torch.long),
        query_meta=torch.zeros((batch, 15)),
        destination=torch.zeros((batch, choices), dtype=torch.long),
        destination_meta=torch.zeros((batch, choices, 3)),
        destination_valid=torch.zeros((batch, choices), dtype=torch.bool),
        keep_valid=torch.zeros(batch, dtype=torch.bool),
    )


def _fill_history(
    inputs: ModelInputs, row: int, example: DecisionExample, bank: _FeatureBank, history: int
) -> None:
    channels = example.request.channels
    inputs.history_roles[row, : len(channels)] = torch.tensor([c.clip.role for c in channels])
    for previous in example.history:
        item = previous[example.channel_index]
        slot = history + item.destination_bar - example.bar
        if not 0 <= slot <= history:
            continue
        phase = item.destination_phase
        meta = torch.tensor(destination_coordinates(example, item.destination_bar, phase, history))
        inputs.history_valid[row, slot, phase] = True
        inputs.history_meta[row, slot, phase] = meta
        for index, playback in enumerate(previous):
            inputs.history_global[row, slot, phase, index] = bank.reference(
                channels[index], playback.source_bar, playback.source_phase
            )
            inputs.history_channel_valid[row, slot, phase, index] = True
        position = slot * bank.steps + phase
        inputs.playback_tokens[row, position] = (
            0 if item.source_bar is None else 1 if item.action.kind == ActionKind.START else 2
        )
        inputs.playback_destination[row, position] = meta
        inputs.playback_source[row, position] = torch.tensor(
            source_coordinates(
                channels[example.channel_index], item.source_bar, item.source_phase, bank.steps
            )
        )
        inputs.playback_valid[row, position] = True


def _fill_current(
    inputs: ModelInputs, row: int, example: DecisionExample, bank: _FeatureBank
) -> None:
    machine = PlaybackStateMachine()
    committed = {item.channel_id: item for item in example.committed}
    slot = 0
    for index, channel in enumerate(example.request.channels):
        if index == example.channel_index:
            continue
        actual = committed.get(channel.channel_id)
        if actual is not None:
            reference = bank.reference(channel, actual.source_bar, actual.source_phase)
        elif machine.keep_is_valid(
            example.states[index], channel, example.request, example.bar, example.phase
        ):
            state = example.states[index]
            reference = bank.reference(channel, state.next_source_bar, state.next_source_phase)
        else:
            reference = 1
        inputs.current[row, slot] = reference
        inputs.current_status[row, slot] = int(actual is not None)
        inputs.current_roles[row, slot] = channel.clip.role
        inputs.current_valid[row, slot] = True
        slot += 1


def _fill_query(
    inputs: ModelInputs, row: int, example: DecisionExample, bank: _FeatureBank, history: int
) -> None:
    channel = example.request.channels[example.channel_index]
    state = example.states[example.channel_index]
    keep_valid = PlaybackStateMachine().keep_is_valid(
        state, channel, example.request, example.bar, example.phase
    )
    inputs.keep_valid[row] = keep_valid
    inputs.keep[row] = (
        bank.reference(channel, state.next_source_bar, state.next_source_phase) if keep_valid else 1
    )
    inputs.query_meta[row] = torch.tensor(
        destination_coordinates(example, example.bar, example.phase, history)
        + source_coordinates(channel, state.next_source_bar, state.next_source_phase, bank.steps)
        + [
            float(state.next_source_bar is not None),
            float(keep_valid),
            channel.clip.role / 16,
            (example.bar + example.phase / bank.steps) / len(example.request.grid.bars),
        ]
    )
    for source in range(len(channel.clip.grid.bars) * bank.steps):
        bar, phase = divmod(source, bank.steps)
        inputs.source[row, source] = bank.reference(channel, bar, phase)
        inputs.source_valid[row, source] = True


def _fill_candidates(
    inputs: ModelInputs,
    row: int,
    example: DecisionExample,
    candidates: Sequence[PlaybackAction],
    bank: _FeatureBank,
) -> None:
    channel = example.request.channels[example.channel_index]
    total = len(channel.clip.grid.bars) * bank.steps
    for index, action in enumerate(candidates):
        inputs.destination[row, index] = bank.reference(
            channel, action.source_bar, action.source_phase
        )
        position = (action.source_bar or 0) * bank.steps + action.source_phase
        inputs.destination_meta[row, index] = torch.tensor(
            [
                float(action.kind == ActionKind.START),
                position / total,
                float(action.kind == ActionKind.SILENCE),
            ]
        )
        inputs.destination_valid[row, index] = True


def _fill_target(
    targets: DecisionTargets,
    row: int,
    target: PlaybackAction | None,
    candidates: Sequence[PlaybackAction],
    keep_valid: bool,
) -> None:
    if target is None:
        return
    changed = target.kind != ActionKind.KEEP
    targets.change[row] = float(changed)
    targets.gate_valid[row] = keep_valid and bool(candidates)
    if changed:
        if target not in candidates:
            raise ValueError("Target is redundant or unreachable; canonicalise preparation labels")
        targets.destinations[row, candidates.index(target)] = True
        targets.destination_valid[row] = True
    elif not keep_valid:
        raise ValueError("Ground-truth KEEP is not legal")


class DecisionCollator:
    def __init__(
        self, features: Mapping[FeatureIdentity, ClipFeatures], config: ModelConfig
    ) -> None:
        self.features, self.config = features, config

    def __call__(self, examples: list[DecisionExample]) -> DecisionBatch:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        steps = examples[0].request.steps_per_bar
        if any(example.request.steps_per_bar != steps for example in examples):
            raise ValueError("A batch must use the same decision resolution")
        bank = _FeatureBank(self.features, self.config, steps)
        candidate_sets = [
            build_candidates(
                e.states[e.channel_index],
                e.request.channels[e.channel_index],
                e.request,
                e.bar,
                e.phase,
            )
            for e in examples
        ]
        count = len(examples)
        choices = max(1, max(map(len, candidate_sets)))
        channels = max(len(e.request.channels) for e in examples)
        sources = max(
            len(e.request.channels[e.channel_index].clip.grid.bars) * steps for e in examples
        )
        inputs = _allocate(count, self.config.history_bars, channels, choices, sources, bank)
        targets = DecisionTargets(
            torch.zeros(count),
            torch.zeros(count, dtype=torch.bool),
            torch.zeros((count, choices), dtype=torch.bool),
            torch.zeros(count, dtype=torch.bool),
        )
        for row, (example, candidates) in enumerate(zip(examples, candidate_sets, strict=True)):
            _fill_query(inputs, row, example, bank, self.config.history_bars)
            _fill_history(inputs, row, example, bank, self.config.history_bars)
            _fill_current(inputs, row, example, bank)
            _fill_candidates(inputs, row, example, candidates, bank)
            _fill_target(targets, row, example.target, candidates, bool(inputs.keep_valid[row]))
        midi, audio, metadata = bank.tensors()
        return DecisionBatch(
            replace(inputs, feature_midi=midi, feature_audio=audio, feature_meta=metadata), targets
        )
