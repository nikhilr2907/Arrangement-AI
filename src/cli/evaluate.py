import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import TrainingConfig
from src.data.collate import DecisionCollator
from src.data.dataset import ArrangementDecisionDataset
from src.data.serialization import read_episodes, write_json
from src.features.inventory import FeatureInventory
from src.inference.generator import ArrangementGenerator
from src.training.checkpoints import load_checkpoint, validate_extractors
from src.training.evaluation import loop_baseline, rollout_metrics
from src.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out decisions and generated rollouts"
    )
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--rollouts", type=int, default=4)
    args = parser.parse_args()
    model, payload = load_checkpoint(args.checkpoint, torch.device("cpu"))
    features = FeatureInventory(args.inventory)
    validate_extractors(payload, features.specs)
    episodes = [episode for episode in read_episodes(args.episodes) if episode.split == args.split]
    dataset = ArrangementDecisionDataset(episodes, model.config.history_bars)
    loader = DataLoader(dataset, batch_size=32, collate_fn=DecisionCollator(features, model.config))
    metrics = Trainer(model, TrainingConfig()).run(loader, False)
    generator = ArrangementGenerator(model, features)
    rollouts = [generator.generate(episode.request) for episode in episodes[: args.rollouts]]
    rollout_scores = [
        rollout_metrics(result, target) for result, target in zip(rollouts, episodes, strict=False)
    ]
    baseline_scores = [
        rollout_metrics(loop_baseline(target.request), target)
        for target in episodes[: args.rollouts]
    ]
    write_json(
        args.output,
        {
            "metrics": metrics,
            "rollout_metrics": rollout_scores,
            "loop_baseline_metrics": baseline_scores,
            "rollouts": rollouts,
        },
    )


if __name__ == "__main__":
    main()
