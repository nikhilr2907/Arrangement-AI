from __future__ import annotations

import random
from collections.abc import Mapping

import torch

from src.contracts import (
    ActionKind,
    ArrangementRequest,
    ArrangementResult,
    BarPlayback,
    PlaybackAction,
    PlaybackState,
)
from src.data.collate import DecisionCollator
from src.data.dataset import DecisionExample
from src.features.cache import ClipFeatures, FeatureIdentity
from src.inference.candidates import build_candidates
from src.inference.state import PlaybackStateMachine, coalesce
from src.models.arrangement import ArrangementModel
from src.models.decisions import action_log_probabilities


class ArrangementGenerator:
    def __init__(
        self, model: ArrangementModel, features: Mapping[FeatureIdentity, ClipFeatures]
    ) -> None:
        self.model = model
        self.collate = DecisionCollator(features, model.config)

    def generate(
        self, request: ArrangementRequest, sample: bool = False, temperature: float = 1.0
    ) -> ArrangementResult:
        if temperature <= 0:
            raise ValueError("Sampling temperature must be positive")
        device = next(self.model.parameters()).device
        generator = torch.Generator(device=device).manual_seed(request.seed)
        order_rng = random.Random(request.seed)
        machine = PlaybackStateMachine()
        states = [PlaybackState() for _ in request.channels]
        history: list[tuple[BarPlayback, ...]] = []
        self.model.eval()
        with torch.inference_mode():
            for bar in range(len(request.grid.bars)):
                order = list(range(len(request.channels)))
                order_rng.shuffle(order)
                committed: dict[int, BarPlayback] = {}
                before = tuple(states)
                for index in order:
                    example = DecisionExample(
                        request,
                        bar,
                        index,
                        before,
                        tuple(history[-self.model.config.history_bars :]),
                        tuple(committed.values()),
                    )
                    inputs = self.collate([example]).inputs.to(device)
                    logits = self.model(inputs)
                    log_probs = action_log_probabilities(
                        logits, inputs.keep_valid, inputs.destination_valid
                    )[0]
                    if sample:
                        distribution = torch.softmax(log_probs / temperature, -1)
                        choice = int(torch.multinomial(distribution, 1, generator=generator).item())
                    else:
                        choice = int(log_probs.argmax().item())
                    candidates = build_candidates(
                        before[index], request.channels[index], request, bar
                    )
                    action = (
                        PlaybackAction(ActionKind.KEEP) if choice == 0 else candidates[choice - 1]
                    )
                    states[index], committed[index] = machine.apply(
                        before[index],
                        action,
                        request.channels[index],
                        request,
                        bar,
                        float(log_probs[choice].exp().item()),
                    )
                history.append(tuple(committed[index] for index in range(len(request.channels))))
        playback = tuple(history)
        return ArrangementResult(request, playback, coalesce(playback))
