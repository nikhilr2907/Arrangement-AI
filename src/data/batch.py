from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass(frozen=True)
class ModelInputs:
    feature_midi: Tensor
    feature_audio: Tensor
    feature_meta: Tensor
    resolution: Tensor
    history_global: Tensor
    history_channel_valid: Tensor
    history_valid: Tensor
    history_meta: Tensor
    history_roles: Tensor
    playback_tokens: Tensor
    playback_destination: Tensor
    playback_source: Tensor
    playback_valid: Tensor
    source: Tensor
    source_valid: Tensor
    current: Tensor
    current_status: Tensor
    current_roles: Tensor
    current_valid: Tensor
    keep: Tensor
    query_meta: Tensor
    destination: Tensor
    destination_meta: Tensor
    destination_valid: Tensor
    keep_valid: Tensor

    def to(self, device: torch.device) -> ModelInputs:
        return ModelInputs(
            **{field.name: getattr(self, field.name).to(device) for field in fields(self)}
        )


@dataclass(frozen=True)
class DecisionTargets:
    change: Tensor
    gate_valid: Tensor
    destinations: Tensor
    destination_valid: Tensor

    def to(self, device: torch.device) -> DecisionTargets:
        return DecisionTargets(
            **{field.name: getattr(self, field.name).to(device) for field in fields(self)}
        )


@dataclass(frozen=True)
class DecisionBatch:
    inputs: ModelInputs
    targets: DecisionTargets

    def to(self, device: torch.device) -> DecisionBatch:
        return DecisionBatch(self.inputs.to(device), self.targets.to(device))
