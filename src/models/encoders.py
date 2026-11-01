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

    # Input: members float[B,N,D], valid bool[B,N]; an internal null is always valid.
    # Output: permutation-invariant summary float[B,D].
    def forward(self, members: Tensor, valid: Tensor) -> Tensor:
        query = self.query.expand(members.size(0), -1, -1)
        memory = torch.cat((query, members), dim=1)
        padding = torch.cat((torch.zeros_like(valid[:, :1]), ~valid), dim=1)
        if valid.size(1) == 0:
            padding = torch.zeros((members.size(0), 1), dtype=torch.bool, device=members.device)
        output, _ = self.attention(
            query, memory, memory, key_padding_mask=padding, need_weights=False
        )
        return cast(Tensor, self.norm(output[:, 0]))


class SourceBarEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.projections = nn.ModuleList(nn.Linear(width, dim) for width in config.feature_dims)
        self.modality = nn.Embedding(4, dim)
        self.role = nn.Embedding(16, dim)
        self.coordinates = nn.Linear(6, dim)
        self.phase = nn.Linear(3, dim)
        self.pool = SetPool(dim, config.heads)

    # Input: raw MIDI/audio float[N,P,F_m], metadata float[N,8].
    # Output: source or explicit silence/unknown summaries float[N,D].
    def forward(self, midi: Tensor, audio: Tensor, metadata: Tensor) -> Tensor:
        mode = metadata[:, 0].long()
        projected = self.projections[0](midi) * (mode == 0)[:, None, None]
        projected = projected + self.projections[1](audio) * (mode == 1)[:, None, None]
        phase = torch.linspace(0, 1, midi.size(1) + 1, device=midi.device)[:-1]
        positions = torch.stack(
            (phase, torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)), -1
        )
        projected = projected + self.phase(positions).unsqueeze(0)
        projected = projected + (
            self.modality(mode)
            + self.role(metadata[:, 1].long())
            + self.coordinates(metadata[:, 2:])
        ).unsqueeze(1)
        valid = torch.ones(projected.shape[:2], dtype=torch.bool, device=projected.device)
        return cast(Tensor, self.pool(projected, valid))


class HistoryEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.bar_pool = SetPool(dim, config.heads)
        self.combine = nn.Linear(dim * 2, dim)
        self.position = nn.Embedding(config.history_bars + 1, dim)
        self.bos = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim,
            config.heads,
            dim * 4,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)

    # Input: actual channels float[B,H,C,D], target float[B,H,D], validity bool[B,H,C]/[B,H].
    # Output: causal memory float[B,H+1,D], valid bool[B,H+1], including BOS.
    def forward(
        self, channels: Tensor, target: Tensor, channel_valid: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch, history, count, dim = channels.shape
        pooled = self.bar_pool(
            channels.reshape(batch * history, count, dim),
            channel_valid.reshape(batch * history, count),
        )
        bars = self.combine(torch.cat((pooled.reshape(batch, history, dim), target), -1))
        memory = torch.cat((self.bos.expand(batch, -1, -1), bars), 1)
        memory = memory + self.position(torch.arange(history + 1, device=memory.device))[None]
        mask = torch.cat((torch.ones((batch, 1), dtype=torch.bool, device=valid.device), valid), 1)
        causal = torch.ones((history + 1, history + 1), dtype=torch.bool, device=valid.device).triu(
            1
        )
        return self.encoder(memory, mask=causal, src_key_padding_mask=~mask), mask
