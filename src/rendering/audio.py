from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from src.contracts import ArrangementResult, Modality
from src.timing import bar_seconds, source_audio_seconds, tick_to_seconds


def render_audio(
    result: ArrangementResult, output_dir: Path, rate: int = 44100, fade_ms: float = 3.0
) -> None:
    if rate < 1 or fade_ms < 0:
        raise ValueError("Invalid rendering rate or fade duration")
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = tick_to_seconds(result.request.grid.bars[-1].end, result.request.grid)
    mixture = np.zeros((round(duration * rate), 2), dtype=np.float32)
    channels_rendered = 0
    for index, channel in enumerate(result.request.channels):
        if channel.modality != Modality.AUDIO:
            continue
        asset = channel.clip.audio_path
        if asset is None:
            raise ValueError("Missing audio asset")
        source, source_rate = sf.read(asset, dtype="float32", always_2d=True)
        if source.shape[1] == 1:
            source = np.repeat(source, 2, axis=1)
        if source.shape[1] != 2 or not np.isfinite(source).all():
            raise ValueError("Only finite mono/stereo audio is supported")
        divisor = gcd(int(source_rate), rate)
        source = resample_poly(source, rate // divisor, int(source_rate) // divisor, axis=0).astype(
            np.float32
        )
        output = np.zeros_like(mixture)
        for placement in result.placements:
            if placement.channel_id != channel.channel_id:
                continue
            source_spans = [
                source_audio_seconds(channel.clip, source_bar)
                for source_bar in range(
                    placement.source_start_bar, placement.source_start_bar + placement.bar_count
                )
            ]
            destination, _ = bar_seconds(placement.destination_start_bar, result.request.grid)
            _, target_end = bar_seconds(
                placement.destination_start_bar + placement.bar_count - 1, result.request.grid
            )
            if abs(sum(b - a for a, b in source_spans) - (target_end - destination)) > 0.002:
                raise ValueError("Audio placement requires time stretching; no implicit adaptation")
            pieces = [source[round(a * rate) : round(b * rate)].copy() for a, b in source_spans]
            boundaries = (
                [True]
                + [
                    abs(previous[1] - current[0]) > 1 / rate
                    for previous, current in zip(source_spans, source_spans[1:], strict=False)
                ]
                + [True]
            )
            for part_index, piece in enumerate(pieces):
                fade = min(round(fade_ms * rate / 1000), len(piece) // 2)
                if fade:
                    ramp = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
                    if boundaries[part_index]:
                        piece[:fade] *= ramp
                    if boundaries[part_index + 1]:
                        piece[-fade:] *= ramp[::-1]
            clip = np.concatenate(pieces)
            if abs(len(clip) / rate - sum(b - a for a, b in source_spans)) > 0.002:
                raise ValueError("Audio ends before the selected crop")
            offset = round(destination * rate)
            count = min(len(clip), len(output) - offset)
            output[offset : offset + count] += clip[:count]
        sf.write(output_dir / f"channel_{index:03d}.wav", output, rate, subtype="FLOAT")
        mixture += output
        channels_rendered += 1
    if channels_rendered:
        sf.write(output_dir / "audio_mix.wav", mixture, rate, subtype="FLOAT")
