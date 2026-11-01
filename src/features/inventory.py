from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from src.contracts import Channel
from src.data.serialization import integer, mapping, sequence, text, write_json
from src.features.cache import (
    ClipFeatures,
    ExtractorSpec,
    FeatureIdentity,
    FeatureStore,
    FrozenFeatureExtractor,
)


class FeatureInventory(Mapping[FeatureIdentity, ClipFeatures]):
    def __init__(self, path: Path, max_cached_clips: int = 8) -> None:
        self.path = path
        data = mapping(json.loads(path.read_text(encoding="utf-8")))
        self.store = FeatureStore((path.parent / text(data["cache_root"])).resolve())
        self.entries: dict[FeatureIdentity, str] = {}
        self.specs: dict[str, ExtractorSpec] = {}
        for raw in sequence(data["entries"]):
            entry = mapping(raw)
            identity = (text(entry["clip_id"]), text(entry["modality"]))
            if identity in self.entries:
                raise ValueError(f"Duplicate feature identity {identity}")
            self.entries[identity] = text(entry["key"])
            raw_spec = mapping(entry["spec"])
            spec = ExtractorSpec(
                text(raw_spec["name"]),
                text(raw_spec["checkpoint"]),
                text(raw_spec["revision"]),
                integer(raw_spec["feature_dim"]),
                integer(raw_spec["phase_bins"]),
                integer(raw_spec["schema"]),
            )
            if identity[1] in self.specs and self.specs[identity[1]] != spec:
                raise ValueError("Inventory mixes incompatible extractor revisions")
            self.specs[identity[1]] = spec
        self.max_cached_clips = max_cached_clips
        if max_cached_clips < 1:
            raise ValueError("Feature cache capacity must be positive")
        self._cache: OrderedDict[FeatureIdentity, ClipFeatures] = OrderedDict()

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[FeatureIdentity]:
        return iter(self.entries)

    def __contains__(self, key: object) -> bool:
        return key in self.entries

    def __getitem__(self, key: FeatureIdentity) -> ClipFeatures:
        if key not in self._cache:
            loaded = self.store.load(self.entries[key])
            if loaded.spec != self.specs[key[1]]:
                raise ValueError("Inventory mixes incompatible extractor revisions")
            self._cache[key] = loaded
            if len(self._cache) > self.max_cached_clips:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]


def extract_inventory(
    channels: Sequence[Channel],
    extractors: Mapping[str, FrozenFeatureExtractor],
    cache_root: Path,
    output: Path,
) -> None:
    store = FeatureStore(cache_root.resolve())
    entries: dict[FeatureIdentity, dict[str, object]] = {}
    for channel in channels:
        mode = channel.modality.value
        identity = (channel.clip.clip_id, mode)
        if identity in entries:
            continue
        extractor = extractors[mode]
        key = store.key(channel, extractor.spec)
        store.obtain(channel, extractor)
        entries[identity] = {
            "clip_id": identity[0],
            "modality": mode,
            "key": key,
            "spec": asdict(extractor.spec),
        }
    if not entries:
        raise ValueError("No clips selected for feature extraction")
    write_json(output, {"cache_root": str(cache_root.resolve()), "entries": list(entries.values())})
