from __future__ import annotations

from math import gcd
from typing import cast

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import Tensor, nn

from src.contracts import Channel, Modality
from src.features.cache import ClipFeatures, ExtractorSpec, checkpoint_revision
from src.timing import source_audio_seconds


class MertExtractor:
    def __init__(
        self,
        checkpoint: str = "m-a-p/MERT-v1-95M",
        revision: str = "main",
        device: str = "cpu",
        phase_bins: int = 16,
        layer: int = -1,
        chunk_seconds: float = 5.0,
    ) -> None:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.model = AutoModel.from_pretrained(
            checkpoint, revision=revision, trust_remote_code=True
        )
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint, revision=revision)
        self.model.to(device).eval().requires_grad_(False)
        self.device = torch.device(device)
        self.rate = int(self.processor.sampling_rate)
        self.layer = layer
        self.chunk_samples = round(chunk_seconds * self.rate)
        if self.chunk_samples < self.rate or phase_bins < 1:
            raise ValueError("Invalid MERT chunk or phase resolution")
        resolved = str(getattr(self.model.config, "_commit_hash", None) or revision)
        resolved = checkpoint_revision(checkpoint, resolved)
        self._spec = ExtractorSpec(
            "mert",
            checkpoint,
            f"{resolved}:layer={layer}:chunk={chunk_seconds}",
            int(self.model.config.hidden_size),
            phase_bins,
        )

    @property
    def spec(self) -> ExtractorSpec:
        return self._spec

    # Input: one audio clip with explicit source bar boundaries.
    # Output: frozen float32[source_bars, phase_bins, feature_dim].
    def encode(self, channel: Channel) -> ClipFeatures:
        if channel.modality != Modality.AUDIO or channel.clip.audio_path is None:
            raise ValueError("MERT expects an audio channel")
        audio, rate = sf.read(channel.clip.audio_path, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if not np.isfinite(mono).all():
            raise ValueError("Audio contains nonfinite samples")
        divisor = gcd(int(rate), self.rate)
        mono = np.asarray(
            resample_poly(mono, self.rate // divisor, int(rate) // divisor), dtype=np.float32
        )
        values: list[np.ndarray] = []
        with torch.inference_mode():
            for index in range(len(channel.clip.grid.bars)):
                start, end = source_audio_seconds(channel.clip, index)
                segment = mono[round(start * self.rate) : round(end * self.rate)]
                if abs(len(segment) / self.rate - (end - start)) > 0.002:
                    raise ValueError("Audio ends before a complete source bar")
                frames: list[Tensor] = []
                for offset in range(0, len(segment), self.chunk_samples):
                    chunk = segment[offset : offset + self.chunk_samples]
                    valid_samples = len(chunk)
                    # Convolutional frontends need a minimum receptive field at the tail.
                    chunk = np.pad(chunk, (0, max(0, 1024 - len(chunk))))
                    inputs = self.processor(chunk, sampling_rate=self.rate, return_tensors="pt")
                    outputs = self.model(
                        **{key: value.to(self.device) for key, value in inputs.items()},
                        output_hidden_states=True,
                    )
                    hidden = cast(Tensor, outputs.hidden_states[self.layer])[0]
                    keep = max(1, round(len(hidden) * valid_samples / len(chunk)))
                    frames.append(hidden[:keep].cpu())
                hidden = torch.cat(frames).T.unsqueeze(0)
                pooled = nn.functional.adaptive_avg_pool1d(hidden, self.spec.phase_bins)[0].T
                values.append(pooled.float().numpy())
        return ClipFeatures(np.asarray(values, dtype=np.float32), self.spec)
