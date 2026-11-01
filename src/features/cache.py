from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.contracts import Channel

FloatArray = NDArray[np.float32]
FeatureIdentity = tuple[str, str]


def checkpoint_revision(checkpoint: str, fallback: str) -> str:
    path = Path(checkpoint)
    if not path.is_dir():
        return fallback
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.suffix in {".bin", ".safetensors", ".json", ".py"}:
            digest.update(file.relative_to(path).as_posix().encode())
            with file.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExtractorSpec:
    name: str
    checkpoint: str
    revision: str
    feature_dim: int
    phase_bins: int = 16
    schema: int = 1

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.revision
            or min(self.feature_dim, self.phase_bins, self.schema) < 1
        ):
            raise ValueError("Invalid feature extractor specification")


@dataclass(frozen=True)
class ClipFeatures:
    values: FloatArray
    spec: ExtractorSpec

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.values.shape[1:] != (
            self.spec.phase_bins,
            self.spec.feature_dim,
        ):
            raise ValueError("Features must have shape [source bars, phase bins, feature dim]")
        if self.values.dtype != np.float32 or not np.isfinite(self.values).all():
            raise ValueError("Features must be finite float32 values")


class FrozenFeatureExtractor(Protocol):
    @property
    def spec(self) -> ExtractorSpec: ...

    def encode(self, channel: Channel) -> ClipFeatures: ...


class FeatureStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def key(self, channel: Channel, spec: ExtractorSpec) -> str:
        clip = channel.clip
        asset_digest: str | None = None
        if channel.modality.value == "audio":
            if clip.audio_path is None:
                raise ValueError("Missing audio asset")
            digest = hashlib.sha256()
            with clip.audio_path.open("rb") as asset:
                while block := asset.read(1024 * 1024):
                    digest.update(block)
            asset_digest = digest.hexdigest()
        clip_data = asdict(clip)
        clip_data.pop("audio_path", None)
        payload = {
            "clip": clip_data,
            "modality": channel.modality.value,
            "extractor": asdict(spec),
            "audio_digest": asset_digest,
        }
        # Asset contents and musical coordinates determine the representation, not a filename.
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def load(self, key: str) -> ClipFeatures:
        with np.load(self.root / f"{key}.npz", allow_pickle=False) as data:
            spec = ExtractorSpec(**json.loads(str(data["spec"].item())))
            values = np.asarray(data["values"], dtype=np.float32)
        return ClipFeatures(values, spec)

    def save(self, key: str, features: ClipFeatures) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key}.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, values=features.values, spec=json.dumps(asdict(features.spec))
            )
        temporary.replace(path)

    def obtain(self, channel: Channel, extractor: FrozenFeatureExtractor) -> ClipFeatures:
        key = self.key(channel, extractor.spec)
        if (self.root / f"{key}.npz").is_file():
            return self.load(key)
        features = extractor.encode(channel)
        if len(features.values) != len(channel.clip.grid.bars):
            raise ValueError("Extractor returned a different source-bar count")
        self.save(key, features)
        return features
