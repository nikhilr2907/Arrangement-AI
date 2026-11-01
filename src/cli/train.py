import argparse
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from torch.utils.data import DataLoader

from src.config import ModelConfig, TrainingConfig, load_config
from src.data.collate import DecisionCollator
from src.data.dataset import ArrangementDecisionDataset
from src.data.serialization import mapping, read_episodes, sequence, text, write_json
from src.features.inventory import FeatureInventory
from src.models.arrangement import ArrangementModel
from src.training.checkpoints import load_checkpoint, seed_everything, validate_extractors
from src.training.trainer import Trainer


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Train the per-channel playback policy")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--mix-modalities", action="store_true")
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume", type=Path)
    checkpoint.add_argument("--initialise", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    training = load_config(args.config, TrainingConfig) if args.config else TrainingConfig()
    seed_everything(training.seed)
    inventory = FeatureInventory(args.inventory)
    dims = tuple(
        inventory.specs[mode].feature_dim if mode in inventory.specs else 768
        for mode in ("midi", "audio")
    )
    config = (
        load_config(args.model_config, ModelConfig)
        if args.model_config
        else ModelConfig(feature_dims=(dims[0], dims[1]))
    )
    config = replace(config, feature_dims=(dims[0], dims[1]))
    model = ArrangementModel(config)
    payload = None
    if args.resume or args.initialise:
        model, payload = load_checkpoint(
            args.resume or args.initialise, torch.device(training.device)
        )
        if any(
            inventory.specs[mode].feature_dim != model.config.feature_dims[index]
            for index, mode in enumerate(("midi", "audio"))
            if mode in inventory.specs
        ):
            raise ValueError("Extractor widths differ from the pretrained arranger")
        validate_extractors(payload, inventory.specs)
    episodes = read_episodes(args.episodes)
    train = [episode for episode in episodes if episode.split == "train"]
    validation = [episode for episode in episodes if episode.split == "validation"]
    split_groups: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        split_groups[episode.composition_id].add(episode.split)
    if any(len(splits) > 1 for splits in split_groups.values()):
        raise ValueError("Composition leaks across dataset splits")
    available: dict[str, list[str]] = defaultdict(list)
    if args.mix_modalities:
        for clip_id, mode in inventory:
            available[clip_id].append(mode)
    choices = {clip_id: tuple(modes) for clip_id, modes in available.items()}
    training_dataset = ArrangementDecisionDataset(
        train, model.config.history_bars, training.seed, choices
    )
    validation_dataset = ArrangementDecisionDataset(
        validation, model.config.history_bars, training.seed, choices
    )
    collator = DecisionCollator(inventory, model.config)
    train_loader = DataLoader(
        training_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training.batch_size,
        collate_fn=collator,
        num_workers=args.workers,
    )
    extractor_specs: dict[str, object] = {
        mode: asdict(spec) for mode, spec in inventory.specs.items()
    }
    if payload is not None:
        previous = mapping(payload["provenance"])
        inherited = mapping(json.loads(text(previous["extractors"])))
        extractor_specs = {**inherited, **extractor_specs}
    provenance = {
        "episodes": hashlib.sha256(args.episodes.read_bytes()).hexdigest(),
        "inventory": hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        "extractors": json.dumps(extractor_specs, sort_keys=True),
    }
    trainer = Trainer(model, training, provenance)
    if args.initialise and payload is not None:
        trainer.modalities = {text(mode) for mode in sequence(payload["modalities"])}
    if args.resume and payload is not None:
        trainer.resume(payload)
    metrics = trainer.fit(train_loader, validation_loader, args.output)
    write_json(args.output / "metrics.json", metrics)


if __name__ == "__main__":
    main()
