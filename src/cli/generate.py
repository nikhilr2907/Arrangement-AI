import argparse
from pathlib import Path

import torch

from src.data.serialization import read_request, sequence, text, write_json
from src.features.inventory import FeatureInventory
from src.inference.generator import ArrangementGenerator
from src.rendering.audio import render_audio
from src.rendering.midi import export_midi
from src.training.checkpoints import load_checkpoint, validate_extractors


def main() -> None:
    parser = argparse.ArgumentParser(description="Arrange supplied clips without full-song inputs")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--render-audio", action="store_true")
    args = parser.parse_args()
    request = read_request(args.request)
    model, payload = load_checkpoint(args.checkpoint, torch.device(args.device))
    trained = {text(mode) for mode in sequence(payload["modalities"])}
    if any(channel.modality.value not in trained for channel in request.channels):
        raise ValueError("Checkpoint has not trained every requested modality")
    inventory = FeatureInventory(args.inventory)
    validate_extractors(payload, inventory.specs)
    result = ArrangementGenerator(model, inventory).generate(request, args.sample, args.temperature)
    write_json(
        args.output / "arrangement.json",
        {"schema_version": 1, "checkpoint": str(args.checkpoint.resolve()), "result": result},
    )
    export_midi(result, args.output / "arrangement.mid")
    if args.render_audio:
        render_audio(result, args.output / "audio")


if __name__ == "__main__":
    main()
