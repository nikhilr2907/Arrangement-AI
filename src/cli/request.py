import argparse
from dataclasses import replace
from pathlib import Path

import soundfile as sf

from src.contracts import ArrangementRequest, Channel, Clip, Meter, Modality
from src.data.midi import read_midi
from src.data.serialization import write_json
from src.timing import regular_grid, tick_to_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a clip-only inference request")
    parser.add_argument("--midi", type=Path, action="append", default=[])
    parser.add_argument("--audio", type=Path, action="append", default=[])
    parser.add_argument("--bpm", type=float, required=True)
    parser.add_argument("--bars", type=int, required=True)
    parser.add_argument("--numerator", type=int, default=4)
    parser.add_argument("--denominator", type=int, default=4)
    parser.add_argument("--audio-starts-on-downbeat", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meter = Meter(args.numerator, args.denominator)
    target = regular_grid(args.bars, args.bpm, meter)
    channels: list[Channel] = []
    for file_index, path in enumerate(args.midi):
        channels.extend(
            Channel(
                f"midi_{file_index}_{part_index}",
                replace(clip, clip_id=f"midi_{file_index}_{part_index}"),
            )
            for part_index, clip in enumerate(read_midi(path))
        )
    if args.audio and not args.audio_starts_on_downbeat:
        parser.error(
            "Audio timing must be declared with --audio-starts-on-downbeat or a custom manifest"
        )
    one_bar_seconds = tick_to_seconds(target.bars[0].end, target)
    for file_index, path in enumerate(args.audio):
        info = sf.info(path)
        bars = round(info.duration / one_bar_seconds)
        if bars < 1 or abs(info.duration - bars * one_bar_seconds) > 0.002:
            raise ValueError(f"{path} is not complete bars at the declared tempo/meter")
        clip = Clip(
            f"audio_{file_index}", regular_grid(bars, args.bpm, meter), audio_path=path.resolve()
        )
        channels.append(Channel(f"audio_{file_index}", clip, Modality.AUDIO))
    write_json(args.output, ArrangementRequest(tuple(channels), target, args.seed))


if __name__ == "__main__":
    main()
