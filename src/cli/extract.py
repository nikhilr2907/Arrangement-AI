import argparse
from dataclasses import replace
from pathlib import Path

from src.contracts import Channel, Modality
from src.data.serialization import read_episodes, read_request
from src.features.cache import FrozenFeatureExtractor
from src.features.inventory import extract_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen musical features")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episodes", type=Path)
    source.add_argument("--request", type=Path)
    parser.add_argument(
        "--modality", choices=("midi", "audio", "both", "declared"), default="declared"
    )
    parser.add_argument("--midi-checkpoint")
    parser.add_argument("--audio-checkpoint", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase-bins", type=int, default=16)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requests = (
        [read_request(args.request)]
        if args.request
        else [e.request for e in read_episodes(args.episodes)]
    )
    channels: list[Channel] = []
    for request in requests:
        for channel in request.channels:
            modes = (
                ("midi", "audio")
                if args.modality == "both"
                else (channel.modality.value,)
                if args.modality == "declared"
                else (args.modality,)
            )
            channels.extend(replace(channel, modality=Modality(mode)) for mode in modes)
    used_modes = {channel.modality.value for channel in channels}
    extractors: dict[str, FrozenFeatureExtractor] = {}
    if "midi" in used_modes:
        if not args.midi_checkpoint:
            parser.error("--midi-checkpoint must point to a pretrained HF MusicBERT checkpoint")
        from src.features.musicbert import MusicBertExtractor

        extractors["midi"] = MusicBertExtractor(
            args.midi_checkpoint, args.revision, args.device, args.phase_bins
        )
    if "audio" in used_modes:
        from src.features.mert import MertExtractor

        extractors["audio"] = MertExtractor(
            args.audio_checkpoint, args.revision, args.device, args.phase_bins
        )
    extract_inventory(channels, extractors, args.cache, args.output)


if __name__ == "__main__":
    main()
