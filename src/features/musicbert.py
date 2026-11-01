from __future__ import annotations

from bisect import bisect_right
from math import log2
from typing import cast

import numpy as np
import torch
from torch import Tensor

from src.contracts import Channel, Modality, Note
from src.features.cache import ClipFeatures, ExtractorSpec, checkpoint_revision

FIELD_SIZES = (256, 128, 129, 256, 128, 32, 254, 49)
DURATION_BUCKETS = tuple(
    bucket
    for power in range(8)
    for bucket in range(power * 16, (power + 1) * 16)
    for _ in range(2**power)
)


def octuple_fields(
    note: Note, channel: Channel, bar_index: int, first_bar: int = 0
) -> tuple[int, ...]:
    clip = channel.clip
    bar = clip.grid.bars[bar_index]
    denominator_power = int(log2(bar.meter.denominator))
    if denominator_power > 6 or bar.meter.numerator > 2 * bar.meter.denominator:
        raise ValueError("Meter is outside the pretrained MusicBERT vocabulary")
    position = round((note.onset - bar.start) * 16 / clip.grid.ppq)
    duration_positions = max(0, round((note.offset - note.onset) * 16 / clip.grid.ppq))
    duration = DURATION_BUCKETS[min(duration_positions, len(DURATION_BUCKETS) - 1)]
    tempos = clip.grid.tempos
    tempo = tempos[bisect_right([t.tick for t in tempos], note.onset) - 1].bpm
    fields = (
        bar_index - first_bar,
        position,
        128 if clip.is_drum else clip.program,
        note.pitch + (128 if clip.is_drum else 0),
        duration,
        note.velocity // 4,
        2 * (2**denominator_power - 1) + bar.meter.numerator - 1,
        round(log2(min(256.0, max(16.0, tempo)) / 16) * 12),
    )
    if any(not 0 <= value < size for value, size in zip(fields, FIELD_SIZES, strict=True)):
        raise ValueError("Note outside pretrained OctupleMIDI vocabulary")
    return fields


def octuple_ids(fields: tuple[int, ...]) -> list[int]:
    offset = 4
    result: list[int] = []
    for value, size in zip(fields, FIELD_SIZES, strict=True):
        result.append(offset + value)
        offset += size
    return result


class MusicBertExtractor:
    def __init__(
        self, checkpoint: str, revision: str = "main", device: str = "cpu", phase_bins: int = 16
    ) -> None:
        from musicbert_hf import MusicBert

        self.model = MusicBert.from_pretrained(checkpoint, revision=revision)
        self.model.to(device).eval().requires_grad_(False)
        self.model.bert.embeddings.upsample = False
        if int(self.model.config.vocab_size) not in {4 + sum(FIELD_SIZES), 5 + sum(FIELD_SIZES)}:
            raise ValueError("Checkpoint does not use the original MusicBERT dictionary")
        self.device = torch.device(device)
        self.max_notes = min(1000, int(self.model.config.max_position_embeddings) - 4)
        identity = str(getattr(self.model.config, "_commit_hash", None) or revision)
        identity = checkpoint_revision(checkpoint, identity)
        self._spec = ExtractorSpec(
            "musicbert", checkpoint, identity, int(self.model.config.hidden_size), phase_bins
        )

    @property
    def spec(self) -> ExtractorSpec:
        return self._spec

    # Input: native MIDI notes with source timing; an entire supplied clip is available.
    # Output: float32[source_bars, phase_bins, feature_dim], pooled across sounding notes.
    def encode(self, channel: Channel) -> ClipFeatures:
        if channel.modality != Modality.MIDI:
            raise ValueError("MusicBERT expects a MIDI channel")
        clip = channel.clip
        values = np.zeros(
            (len(clip.grid.bars), self.spec.phase_bins, self.spec.feature_dim), dtype=np.float32
        )
        counts = np.zeros(values.shape[:2], dtype=np.float32)
        starts = [bar.start for bar in clip.grid.bars]
        notes = sorted(clip.notes, key=lambda note: (note.onset, note.pitch, note.offset))
        with torch.inference_mode():
            offset = 0
            while offset < len(notes):
                first_bar = bisect_right(starts, notes[offset].onset) - 1
                end = offset
                while end < min(offset + self.max_notes, len(notes)):
                    if bisect_right(starts, notes[end].onset) - 1 - first_bar >= 256:
                        break
                    end += 1
                segment = notes[offset:end]
                offset = end
                ids = [0] * 8
                for note in segment:
                    ids.extend(
                        octuple_ids(
                            octuple_fields(
                                note, channel, bisect_right(starts, note.onset) - 1, first_bar
                            )
                        )
                    )
                ids.extend([2] * 8)
                inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
                hidden = cast(
                    Tensor, self.model.bert(input_ids=inputs, return_dict=True).last_hidden_state
                )
                if hidden.shape != (1, len(segment) + 2, self.spec.feature_dim):
                    raise ValueError("Unexpected MusicBERT compound-token output shape")
                embeddings = hidden[0, 1:-1].float().cpu().numpy()
                for note, embedding in zip(segment, embeddings, strict=True):
                    first = bisect_right(starts, note.onset) - 1
                    last = bisect_right(starts, note.offset - 1) - 1
                    for bar_index in range(first, last + 1):
                        bar = clip.grid.bars[bar_index]
                        for phase in range(self.spec.phase_bins):
                            left = bar.start + (bar.end - bar.start) * phase / self.spec.phase_bins
                            right = (
                                bar.start
                                + (bar.end - bar.start) * (phase + 1) / self.spec.phase_bins
                            )
                            if note.onset < right and note.offset > left:
                                values[bar_index, phase] += embedding
                                counts[bar_index, phase] += 1
        values /= np.maximum(counts[..., None], 1)
        return ClipFeatures(values, self.spec)
