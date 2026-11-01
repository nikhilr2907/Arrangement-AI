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


class _FeatureBank:
    def __init__(
        self, features: Mapping[FeatureIdentity, ClipFeatures], config: ModelConfig
    ) -> None:
        self.features = features
        self.config = config
        self.bins = next(iter(features.values())).spec.phase_bins if features else 16
        self.midi = [
            np.zeros((self.bins, config.feature_dims[0]), dtype=np.float32) for _ in range(2)
        ]
        self.audio = [
            np.zeros((self.bins, config.feature_dims[1]), dtype=np.float32) for _ in range(2)
        ]
        self.metadata = [[float(mode), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for mode in (2, 3)]
        self.references: dict[tuple[str, str, int], int] = {}

    def reference(self, channel: Channel, source: int | None) -> int:
        if source is None:
            return 0
        if not 0 <= source < len(channel.clip.grid.bars):
            return 1
        identity = (channel.clip.clip_id, channel.modality.value)
        key = (*identity, source)
        if key in self.references:
            return self.references[key]
        features = self.features[identity]
        mode = int(channel.modality.value == "audio")
        if (
            features.spec.feature_dim != self.config.feature_dims[mode]
            or features.spec.phase_bins != self.bins
        ):
            raise ValueError("Feature dimensions/preprocessing do not match the model")
        if len(features.values) != len(channel.clip.grid.bars):
            raise ValueError("Source bar count differs from feature cache")
        index = len(self.metadata)
        self.references[key] = index
        self.midi.append(features.values[source] if mode == 0 else np.zeros_like(self.midi[0]))
        self.audio.append(features.values[source] if mode == 1 else np.zeros_like(self.audio[0]))
        bar, count = channel.clip.grid.bars[source], len(channel.clip.grid.bars)
        self.metadata.append(
            [
                float(mode),
                float(channel.clip.role),
                source / count,
                (count - source) / 32,
                bar.meter.numerator / 16,
                bar.meter.denominator / 16,
                (bar.end - bar.start) / channel.clip.grid.ppq / 8,
                channel.clip.grid.tempos[0].bpm / 240,
            ]
        )
        return index

    def tensors(self) -> tuple[Tensor, Tensor, Tensor]:
        return (
            torch.from_numpy(np.stack(self.midi)),
            torch.from_numpy(np.stack(self.audio)),
            torch.tensor(self.metadata),
        )


def _allocate(
    batch: int, history: int, channels: int, choices: int, bank: _FeatureBank
) -> ModelInputs:
    midi, audio, metadata = bank.tensors()
    return ModelInputs(
        feature_midi=midi,
        feature_audio=audio,
        feature_meta=metadata,
        history_global=torch.zeros((batch, history, channels), dtype=torch.long),
        history_channel_valid=torch.zeros((batch, history, channels), dtype=torch.bool),
        history_target=torch.zeros((batch, history), dtype=torch.long),
        history_valid=torch.zeros((batch, history), dtype=torch.bool),
        current=torch.zeros((batch, max(channels - 1, 1)), dtype=torch.long),
        current_status=torch.zeros((batch, max(channels - 1, 1)), dtype=torch.long),
        current_valid=torch.zeros((batch, max(channels - 1, 1)), dtype=torch.bool),
        keep=torch.zeros(batch, dtype=torch.long),
        query_meta=torch.zeros((batch, 6)),
        destination=torch.zeros((batch, choices), dtype=torch.long),
        destination_meta=torch.zeros((batch, choices, 3)),
        destination_valid=torch.zeros((batch, choices), dtype=torch.bool),
        keep_valid=torch.zeros(batch, dtype=torch.bool),
    )


def _fill_history(
    inputs: ModelInputs, row: int, example: DecisionExample, bank: _FeatureBank
) -> None:
    for position, previous in enumerate(example.history[-inputs.history_valid.size(1) :]):
        inputs.history_valid[row, position] = True
        for index, item in enumerate(previous):
            inputs.history_global[row, position, index] = bank.reference(
                example.request.channels[index], item.source_bar
            )
            inputs.history_channel_valid[row, position, index] = True
        inputs.history_target[row, position] = inputs.history_global[
            row, position, example.channel_index
        ]


def _fill_current(
    inputs: ModelInputs, row: int, example: DecisionExample, bank: _FeatureBank
) -> None:
    machine = PlaybackStateMachine()
    committed = {item.channel_id: item for item in example.committed}
    others = [
        index for index in range(len(example.request.channels)) if index != example.channel_index
    ]
    for slot, index in enumerate(others):
        channel = example.request.channels[index]
        actual = committed.get(channel.channel_id)
        if actual is not None:
            reference = bank.reference(channel, actual.source_bar)
        elif machine.keep_is_valid(example.states[index], channel, example.request, example.bar):
            reference = bank.reference(channel, example.states[index].next_source_bar)
        else:
            reference = 1
        inputs.current[row, slot] = reference
        inputs.current_status[row, slot] = int(actual is not None)
        inputs.current_valid[row, slot] = True


def _fill_query(
    inputs: ModelInputs, row: int, example: DecisionExample, bank: _FeatureBank
) -> None:
    channel = example.request.channels[example.channel_index]
    state = example.states[example.channel_index]
    keep_valid = PlaybackStateMachine().keep_is_valid(state, channel, example.request, example.bar)
    inputs.keep_valid[row] = keep_valid
    inputs.keep[row] = bank.reference(channel, state.next_source_bar) if keep_valid else 1
    bar = example.request.grid.bars[example.bar]
    inputs.query_meta[row] = torch.tensor(
        [
            example.bar / len(example.request.grid.bars),
            bar.meter.numerator / 16,
            bar.meter.denominator / 16,
            float(state.next_source_bar is not None),
            float(keep_valid),
            channel.clip.role / 16,
        ]
    )


def _fill_candidates(
    inputs: ModelInputs,
    row: int,
    example: DecisionExample,
    candidates: Sequence[PlaybackAction],
    bank: _FeatureBank,
) -> None:
    channel = example.request.channels[example.channel_index]
    for index, action in enumerate(candidates):
        inputs.destination[row, index] = bank.reference(channel, action.source_bar)
        inputs.destination_meta[row, index] = torch.tensor(
            [
                float(action.kind == ActionKind.START),
                (action.source_bar or 0) / len(channel.clip.grid.bars),
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
        self.features = features
        self.config = config

    def __call__(self, examples: list[DecisionExample]) -> DecisionBatch:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        bank = _FeatureBank(self.features, self.config)
        candidate_sets = [
            build_candidates(
                e.states[e.channel_index], e.request.channels[e.channel_index], e.request, e.bar
            )
            for e in examples
        ]
        count = len(examples)
        choices = max(1, max(map(len, candidate_sets)))
        channels = max(len(example.request.channels) for example in examples)
        inputs = _allocate(count, self.config.history_bars, channels, choices, bank)
        targets = DecisionTargets(
            torch.zeros(count),
            torch.zeros(count, dtype=torch.bool),
            torch.zeros((count, choices), dtype=torch.bool),
            torch.zeros(count, dtype=torch.bool),
        )
        for row, (example, candidates) in enumerate(zip(examples, candidate_sets, strict=True)):
            _fill_query(inputs, row, example, bank)
            _fill_history(inputs, row, example, bank)
            _fill_current(inputs, row, example, bank)
            _fill_candidates(inputs, row, example, candidates, bank)
            _fill_target(targets, row, example.target, candidates, bool(inputs.keep_valid[row]))
        midi, audio, metadata = bank.tensors()
        return DecisionBatch(
            replace(inputs, feature_midi=midi, feature_audio=audio, feature_meta=metadata), targets
        )
