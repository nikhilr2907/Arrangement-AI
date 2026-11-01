from typing import cast

import torch
from torch import Tensor, nn

from src.config import ModelConfig
from src.data.batch import ModelInputs
from src.models.decisions import DecisionLogits
from src.models.encoders import HistoryEncoder, SetPool, SourceBarEncoder


class DestinationScorer(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(model_dim * 4, model_dim), nn.GELU(), nn.Linear(model_dim, 1)
        )

    # Input: context float[B,D], candidates float[B,J,D].
    # Output: raw candidate logits float[B,J], with no clip-ID-specific parameters.
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
        self.current_pool = SetPool(dim, config.heads)
        self.clip_pool = SetPool(dim, config.heads)
        self.status = nn.Embedding(2, dim)
        self.query_meta = nn.Linear(6, dim)
        self.history_attention = nn.MultiheadAttention(dim, config.heads, batch_first=True)
        self.context = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.LayerNorm(dim))
        self.destination_type = nn.Linear(3, dim)
        self.change = nn.Linear(dim, 1)
        self.scorer = DestinationScorer(dim)

    # Input: ModelInputs containing cached features, causal history and current decision prefix.
    # Output: change float[B], destination float[B,J]; targets never enter this method.
    def forward(self, inputs: ModelInputs) -> DecisionLogits:
        bank = self.source(inputs.feature_midi, inputs.feature_audio, inputs.feature_meta)
        memory, valid = self.history(
            bank[inputs.history_global],
            bank[inputs.history_target],
            inputs.history_channel_valid,
            inputs.history_valid,
        )
        current = self.current_pool(
            bank[inputs.current] + self.status(inputs.current_status), inputs.current_valid
        )
        destinations = bank[inputs.destination] + self.destination_type(inputs.destination_meta)
        clip = self.clip_pool(
            bank[inputs.destination],
            inputs.destination_valid & (inputs.destination_meta[..., 0] > 0),
        )
        query = bank[inputs.keep] + clip + self.query_meta(inputs.query_meta)
        history, _ = self.history_attention(
            query[:, None], memory, memory, key_padding_mask=~valid, need_weights=False
        )
        context = self.context(torch.cat((query, history[:, 0], current), -1))
        return DecisionLogits(self.change(context).squeeze(-1), self.scorer(context, destinations))
