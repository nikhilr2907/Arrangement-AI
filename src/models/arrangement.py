from typing import cast

import torch
from torch import Tensor, nn

from src.config import ModelConfig
from src.data.batch import ModelInputs
from src.models.decisions import DecisionLogits
from src.models.encoders import (
    ContextReadout,
    HistoryEncoder,
    RotaryEncoder,
    SetPool,
    SourceBarEncoder,
    TrackSetEncoder,
)


class DestinationScorer(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(model_dim * 4, model_dim), nn.GELU(), nn.Linear(model_dim, 1)
        )

    def forward(self, query: Tensor, candidates: Tensor) -> Tensor:
        context = query[:, None].expand(-1, candidates.size(1), -1)
        combined = torch.cat((context, candidates, context * candidates, context - candidates), -1)
        return cast(Tensor, self.score(combined).squeeze(-1))


class ArrangementModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.source = SourceBarEncoder(config)
        self.history = HistoryEncoder(config)
        self.source_sequence = RotaryEncoder(config)
        self.playback_sequence = RotaryEncoder(config)
        self.playback_kind = nn.Embedding(3, dim)
        self.playback_destination = nn.Linear(6, dim)
        self.playback_source = nn.Linear(5, dim)
        self.stream_type = nn.Embedding(4, dim)
        self.current_tracks = TrackSetEncoder(config)
        self.current_pool = SetPool(dim, config.heads)
        self.role = nn.Embedding(16, dim)
        self.status = nn.Embedding(2, dim)
        self.clip_pool = SetPool(dim, config.heads)
        self.query = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.query_meta = nn.Linear(15, dim)
        self.readouts = nn.ModuleList(ContextReadout(config) for _ in range(3))
        self.fusion = TrackSetEncoder(config)
        self.fusion_readout = ContextReadout(config)
        self.destination_type = nn.Linear(3, dim)
        self.change = nn.Linear(dim, 1)
        self.scorer = DestinationScorer(dim)

    def _playback_memory(self, inputs: ModelInputs, steps: int) -> Tensor:
        playback = (
            self.playback_kind(inputs.playback_tokens)
            + self.playback_destination(inputs.playback_destination)
            + self.playback_source(inputs.playback_source)
            + self.stream_type.weight[1]
        )
        playback_positions = (
            inputs.playback_destination[..., 0] * self.config.history_bars
            + inputs.playback_destination[..., 1]
        ) * steps
        return cast(
            Tensor,
            self.playback_sequence(
                playback, inputs.playback_valid, playback_positions, causal=True
            ),
        )

    def _source_memory(self, inputs: ModelInputs, bank: Tensor) -> Tensor:
        source = bank[inputs.source] + self.stream_type.weight[2]
        source_positions = torch.arange(source.size(1), device=source.device).expand(
            source.size(0), -1
        )
        return cast(Tensor, self.source_sequence(source, inputs.source_valid, source_positions))

    def _current_context(self, inputs: ModelInputs, bank: Tensor) -> Tensor:
        other_tracks = (
            bank[inputs.current]
            + self.status(inputs.current_status)
            + self.role(inputs.current_roles)
        )
        other_tracks = self.current_tracks(other_tracks, inputs.current_valid)
        return cast(
            Tensor,
            self.current_pool(other_tracks, inputs.current_valid) + self.stream_type.weight[3],
        )

    def forward(self, inputs: ModelInputs) -> DecisionLogits:
        steps = int(inputs.resolution.item())
        bank = self.source(inputs.feature_midi, inputs.feature_audio, inputs.feature_meta, steps)
        music, music_valid = self.history(
            bank[inputs.history_global],
            inputs.history_channel_valid,
            inputs.history_valid,
            inputs.history_meta,
            inputs.history_roles,
        )
        music = music + self.stream_type.weight[0]
        playback = self._playback_memory(inputs, steps)
        source = self._source_memory(inputs, bank)
        current = self._current_context(inputs, bank)
        identity = self.clip_pool(source, inputs.source_valid)
        query = self.query + bank[inputs.keep] + identity + self.query_meta(inputs.query_meta)
        contexts = torch.stack(
            (
                self.readouts[0](query, music, music_valid),
                self.readouts[1](query, playback, inputs.playback_valid),
                self.readouts[2](query, source, inputs.source_valid),
                current,
            ),
            1,
        )
        valid = torch.ones(contexts.shape[:2], dtype=torch.bool, device=contexts.device)
        contexts = self.fusion(contexts, valid)
        context = self.fusion_readout(query, contexts, valid)
        destinations = bank[inputs.destination] + self.destination_type(inputs.destination_meta)
        return DecisionLogits(self.change(context).squeeze(-1), self.scorer(context, destinations))
