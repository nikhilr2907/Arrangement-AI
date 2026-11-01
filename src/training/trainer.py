from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import torch

from src.config import TrainingConfig
from src.data.batch import DecisionBatch
from src.data.serialization import mapping
from src.models.arrangement import ArrangementModel
from src.models.decisions import ArrangementLoss, action_log_probabilities
from src.training.checkpoints import restore_rng, save_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: ArrangementModel,
        config: TrainingConfig,
        provenance: dict[str, str] | None = None,
    ) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.device = torch.device(config.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.objective = ArrangementLoss(config.destination_weight)
        self.provenance = provenance or {}
        self.start_epoch = 0
        self.best_loss = float("inf")
        self.modalities: set[str] = set()

    def resume(self, payload: dict[str, object]) -> None:
        from typing import cast

        if payload["provenance"] != self.provenance:
            raise ValueError("Dataset/extractor provenance differs from the resumed run")
        previous = mapping(payload["training_config"]).copy()
        current = asdict(self.config)
        for option in ("epochs", "device"):
            previous.pop(option, None)
            current.pop(option, None)
        if previous != current:
            raise ValueError(
                "Resume requires the same training options; use initialise for a new run"
            )
        self.optimizer.load_state_dict(cast(dict[str, object], payload["optimizer"]))
        self.start_epoch = int(cast(int, payload["epoch"])) + 1
        self.best_loss = float(cast(float, payload["best_loss"]))
        self.modalities = set(cast(list[str], payload["modalities"]))
        restore_rng(payload)

    def run(self, batches: Iterable[DecisionBatch], training: bool) -> dict[str, float]:
        self.model.train(training)
        gate_sum = destination_sum = 0.0
        gate_count = destination_count = correct = decisions = 0
        with torch.set_grad_enabled(training):
            for batch in batches:
                batch = batch.to(self.device)
                inputs, targets = batch.inputs, batch.targets
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                loss = self.objective(
                    logits,
                    inputs.keep_valid,
                    inputs.destination_valid,
                    targets.change,
                    targets.gate_valid,
                    targets.destinations,
                    targets.destination_valid,
                )
                if not torch.isfinite(loss.total):
                    raise FloatingPointError("Nonfinite arrangement loss")
                if training:
                    loss.total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip, error_if_nonfinite=True
                    )
                    self.optimizer.step()
                    modes = inputs.feature_meta[:, 0]
                    self.modalities.update(
                        name
                        for index, name in enumerate(("midi", "audio"))
                        if bool((modes == index).any())
                    )
                gate_sum += float(loss.gate_sum.detach().item())
                destination_sum += float(loss.destination_sum.detach().item())
                gate_count += loss.gate_count
                destination_count += loss.destination_count
                probabilities = action_log_probabilities(
                    logits, inputs.keep_valid, inputs.destination_valid
                )
                predicted = probabilities.argmax(-1)
                accepted = torch.cat(((targets.change == 0)[:, None], targets.destinations), -1)
                correct += int(accepted.gather(1, predicted[:, None]).sum().item())
                decisions += len(predicted)
        if not decisions:
            raise ValueError("DataLoader yielded no arrangement decisions")
        return {
            "loss": gate_sum / max(gate_count, 1)
            + self.config.destination_weight * destination_sum / max(destination_count, 1),
            "gate_loss": gate_sum / max(gate_count, 1),
            "destination_loss": destination_sum / max(destination_count, 1),
            "accuracy": correct / decisions,
            "decisions": float(decisions),
        }

    def fit(
        self,
        train_batches: Iterable[DecisionBatch],
        validation_batches: Iterable[DecisionBatch],
        checkpoint_dir: Path,
    ) -> list[dict[str, float]]:
        from src.data.dataset import ArrangementDecisionDataset

        metrics: list[dict[str, float]] = []
        for epoch in range(self.start_epoch, self.config.epochs):
            dataset = getattr(train_batches, "dataset", None)
            if isinstance(dataset, ArrangementDecisionDataset):
                dataset.set_epoch(epoch)
            train = self.run(train_batches, True)
            validation = self.run(validation_batches, False)
            metrics.append({"epoch": float(epoch), "train_loss": train["loss"], **validation})
            logger.info(
                "epoch=%d train=%.4f validation=%.4f accuracy=%.3f",
                epoch,
                train["loss"],
                validation["loss"],
                validation["accuracy"],
            )
            improved = validation["loss"] < self.best_loss
            self.best_loss = min(self.best_loss, validation["loss"])
            for name in ["last.pt", "best.pt"] if improved else ["last.pt"]:
                save_checkpoint(
                    checkpoint_dir / name,
                    self.model,
                    self.optimizer,
                    self.config,
                    epoch,
                    self.best_loss,
                    self.modalities,
                    self.provenance,
                )
        return metrics
