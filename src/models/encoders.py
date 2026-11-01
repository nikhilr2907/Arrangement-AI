import math
from typing import cast

import torch
from torch import Tensor, nn

from src.config import ModelConfig


class SetPool(nn.Module):
    def __init__(self, model_dim: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        self.attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, members: Tensor, valid: Tensor) -> Tensor:
        query = self.query.expand(members.size(0), -1, -1)
        memory = torch.cat((query, members), dim=1)
        padding = torch.cat(
            (torch.zeros((members.size(0), 1), dtype=torch.bool, device=members.device), ~valid),
            dim=1,
        )
        output, _ = self.attention(
            query, memory, memory, key_padding_mask=padding, need_weights=False
        )
        return cast(Tensor, self.norm(output[:, 0]))


class RotaryBlock(nn.Module):
    """Position-aware attention; a zero-valued null key keeps padded rows finite."""

    frequencies: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.heads = config.heads
        self.head_dim = dim // config.heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even attention-head width")
        self.register_buffer(
            "frequencies",
            torch.exp(-math.log(10000) * torch.arange(0, self.head_dim, 2) / self.head_dim),
            persistent=False,
        )
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.dropout)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(config.dropout),
        )

    def rotate(self, values: Tensor, positions: Tensor) -> Tensor:
        angles = positions[:, None, :, None] * self.frequencies[None, None, None, :]
        cosine, sine = angles.cos().to(values.dtype), angles.sin().to(values.dtype)
        even, odd = values[..., 0::2], values[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)

    def forward(
        self, values: Tensor, valid: Tensor, positions: Tensor, causal: bool = False
    ) -> Tensor:
        batch, length, dim = values.shape
        query, key, content = (
            self.qkv(self.norm(values))
            .view(batch, length, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        query, key = self.rotate(query, positions), self.rotate(key, positions)
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        if causal:
            future = torch.ones((length, length), dtype=torch.bool, device=values.device).triu(1)
            scores = scores.masked_fill(future, -torch.inf)
        scores = torch.cat((scores, scores.new_zeros((batch, self.heads, length, 1))), -1)
        content = torch.cat((content, content.new_zeros((batch, self.heads, 1, self.head_dim))), -2)
        attended = self.dropout(scores.softmax(-1)) @ content
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        values = values + self.dropout(self.output(attended))
        values = values + self.feed_forward(values)
        return values * valid.unsqueeze(-1)


class RotaryEncoder(nn.Module):
    def __init__(self, config: ModelConfig, layers: int | None = None) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            RotaryBlock(config) for _ in range(config.layers if layers is None else layers)
        )

    def forward(
        self, values: Tensor, valid: Tensor, positions: Tensor, causal: bool = False
    ) -> Tensor:
        for block in self.blocks:
            values = block(values, valid, positions, causal)
        return values


class TrackSetEncoder(nn.Module):
    """Track order is arbitrary: all members have the same rotary coordinate."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = RotaryEncoder(config, 1)

    def forward(self, values: Tensor, valid: Tensor) -> Tensor:
        return cast(Tensor, self.encoder(values, valid, values.new_zeros(valid.shape)))


class SourceBarEncoder(nn.Module):
    """Frozen feature bins become selectable source steps, preserving the step axis."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.projections = nn.ModuleList(nn.Linear(width, dim) for width in config.feature_dims)
        self.modality = nn.Embedding(2, dim)
        self.role = nn.Embedding(16, dim)
        self.coordinates = nn.Linear(5, dim)
        self.timing = nn.Linear(4, dim)
        self.temporal = RotaryEncoder(config, 1)
        self.pool = SetPool(dim, config.heads)
        self.special = nn.Embedding(2, dim)

    def forward(self, midi: Tensor, audio: Tensor, metadata: Tensor, steps: int = 1) -> Tensor:
        if midi.size(1) % steps:
            raise ValueError("Feature phase bins must be a multiple of the decision resolution")
        if midi.size(0) == 2:
            return self.special.weight
        mode = metadata[2:, 0].long()
        projected = self.projections[0](midi[2:]) * (mode == 0)[:, None, None]
        projected = projected + self.projections[1](audio[2:]) * (mode == 1)[:, None, None]
        bars, bins, dim = projected.shape
        phase = torch.arange(bins, device=midi.device, dtype=midi.dtype) / bins
        coordinates = torch.stack(
            (
                metadata[2:, 2, None].expand(-1, bins),
                phase.expand(bars, -1),
                torch.sin(2 * math.pi * phase).expand(bars, -1),
                torch.cos(2 * math.pi * phase).expand(bars, -1),
                metadata[2:, 3, None].expand(-1, bins),
            ),
            -1,
        )
        projected = projected + self.coordinates(coordinates)
        projected = (
            projected
            + (
                self.modality(mode)
                + self.role(metadata[2:, 1].long())
                + self.timing(metadata[2:, 4:])
            )[:, None]
        )
        valid = torch.ones((bars, bins), dtype=torch.bool, device=midi.device)
        projected = self.temporal(projected, valid, phase.expand(bars, -1) * bins)
        grouped = projected.reshape(bars * steps, bins // steps, dim)
        grouped_valid = valid.reshape(bars * steps, bins // steps)
        return torch.cat((self.special.weight, self.pool(grouped, grouped_valid)), 0)


class HistoryEncoder(nn.Module):
    """Temporal attention per track, track interaction per step, then bar pooling."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.history_bars = config.history_bars
        self.destination = nn.Linear(6, dim)
        self.role = nn.Embedding(16, dim)
        self.temporal = RotaryEncoder(config, 1)
        self.tracks = TrackSetEncoder(config)
        self.track_pool = SetPool(dim, config.heads)
        self.bar_pool = SetPool(dim, config.heads)
        self.bos = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.encoder = RotaryEncoder(config)

    def forward(
        self,
        members: Tensor,
        channel_valid: Tensor,
        step_valid: Tensor,
        metadata: Tensor,
        roles: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, bars, steps, tracks, dim = members.shape
        values = members + self.destination(metadata)[:, :, :, None]
        values = values + self.role(roles)[:, None, None]
        track_values = values.permute(0, 1, 3, 2, 4).reshape(batch * bars * tracks, steps, dim)
        track_valid = channel_valid.permute(0, 1, 3, 2).reshape(batch * bars * tracks, steps)
        phase = metadata[..., 1][:, :, None].expand(-1, -1, tracks, -1)
        track_values = self.temporal(
            track_values, track_valid, phase.reshape(batch * bars * tracks, steps) * steps
        )
        values = track_values.reshape(batch, bars, tracks, steps, dim).permute(0, 1, 3, 2, 4)
        per_step = values.reshape(batch * bars * steps, tracks, dim)
        per_step_valid = channel_valid.reshape(batch * bars * steps, tracks)
        per_step = self.tracks(per_step, per_step_valid)
        pooled = self.track_pool(per_step, per_step_valid).reshape(batch * bars, steps, dim)
        summaries = self.bar_pool(pooled, step_valid.reshape(batch * bars, steps))
        memory = torch.cat((self.bos.expand(batch, -1, -1), summaries.reshape(batch, bars, dim)), 1)
        valid = torch.cat(
            (torch.ones((batch, 1), dtype=torch.bool, device=members.device), step_valid.any(-1)), 1
        )
        positions = torch.arange(-bars, 1, device=members.device).expand(batch, -1)
        return self.encoder(memory, valid, positions, causal=True), valid


class ContextReadout(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.null = nn.Parameter(torch.randn(1, 1, config.model_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            config.model_dim, config.heads, dropout=config.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(self, query: Tensor, memory: Tensor, valid: Tensor) -> Tensor:
        memory = torch.cat((self.null.expand(query.size(0), -1, -1), memory), 1)
        valid = torch.cat(
            (torch.ones((query.size(0), 1), dtype=torch.bool, device=query.device), valid), 1
        )
        output, _ = self.attention(
            query[:, None], memory, memory, key_padding_mask=~valid, need_weights=False
        )
        return cast(Tensor, self.norm(output[:, 0] + query))
