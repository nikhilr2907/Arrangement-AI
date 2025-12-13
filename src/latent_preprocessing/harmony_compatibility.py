import numpy as np
import math
from itertools import combinations
from typing import Dict, Tuple
from src.latent_preprocessing.audio_feature_extraction import extract_stem_feature


HARMONY_WEIGHTS = {
    "tempo":              0.20,
    "chroma_stft":        0.15,
    "chroma_cqt":         0.15,
    "chroma_cens":        0.10,
    "key_signature":      0.20,
    "rmse":               0.05,
    "spectral_centroid":  0.05,
    "spectral_bandwidth": 0.03,
    "rolloff":            0.03,
    "onset_strength":     0.02,
    "zcr":                0.02,
}

FEATURE_RANGES = {
    "tempo":              (60, 180),
    "chroma_stft":        (0, 1),
    "chroma_cqt":         (0, 1),
    "chroma_cens":        (0, 1),
    "rmse":               (0, 0.5),
    "spectral_centroid":  (0, 8000),
    "spectral_bandwidth": (0, 4000),
    "rolloff":            (0, 10000),
    "onset_strength":     (0, 2),
    "zcr":                (0, 0.5),
}


def normalize_feature(value: float, feature_name: str) -> float:
    """Normalize feature value to [0, 1] range."""
    if feature_name == "key_signature":
        return value

    min_val, max_val = FEATURE_RANGES.get(feature_name, (0, 1))
    return np.clip((value - min_val) / (max_val - min_val), 0, 1)


def key_compatibility_score(key1: int, key2: int) -> float:
    """Calculate key compatibility based on music theory (0-11 semitones)."""
    if key1 == key2:
        return 1.0

    interval = abs(key1 - key2) % 12

    if interval in [5, 7]:  return 0.9
    if interval in [3, 4]:  return 0.8
    if interval in [1, 2]:  return 0.6
    if interval == 6:       return 0.4
    return 0.5


def extract_normalized_features(audio_clip,sr) -> Dict[str, float]:
    """Extract and normalize all features from audio clip."""
    raw = extract_stem_feature(audio_clip,sr)
    return {k: v if k == "key_signature" else normalize_feature(v, k)
            for k, v in raw.items()}


def weighted_harmony_similarity(features1: Dict[str, float],
                                 features2: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Calculate weighted similarity score for harmony compatibility."""
    contributions = {}

    for name, weight in HARMONY_WEIGHTS.items():
        if name == "key_signature":
            score = key_compatibility_score(features1.get(name, 0), features2.get(name, 0))
        else:
            score = 1.0 - abs(features1.get(name, 0) - features2.get(name, 0))
        contributions[name] = weight * score

    return sum(contributions.values()), contributions


def should_harmonize(audio_clip1, audio_clip2,
                     threshold: float = 0.70,
                     verbose: bool = False) -> Tuple[bool, float, Dict[str, float]]:
    """Determine if two clips should harmonize (score >= threshold)."""
    features1 = extract_normalized_features(audio_clip1)
    features2 = extract_normalized_features(audio_clip2)
    score, contributions = weighted_harmony_similarity(features1, features2)

    if verbose:
        print(f"\nHarmony Compatibility: {score:.3f} (threshold: {threshold:.3f})")
        print("=" * 50)
        for feat, contrib in sorted(contributions.items(), key=lambda x: x[1], reverse=True):
            pct = (contrib / HARMONY_WEIGHTS[feat]) * 100
            print(f"  {feat:20s}: {contrib:.4f} ({pct:.1f}%)")

    return score >= threshold, score, contributions


def batch_harmony_compatibility(audio_clips: list) -> np.ndarray:
    """Calculate NxN compatibility matrix for all clip pairs."""
    n = len(audio_clips)
    matrix = np.zeros((n, n))
    features = [extract_normalized_features(clip) for clip in audio_clips]

    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                score, _ = weighted_harmony_similarity(features[i], features[j])
                matrix[i, j] = matrix[j, i] = score

    return matrix


def calculate_group_compatibility(features_list: list, main_melody_feature: dict[str, float] = None,
                                 aggregation: str = 'mean', melody_weight: float = 0.7) -> Tuple[float, Dict[str, float]]:
    """Calculate compatibility for group of 2+ clips via pairwise and melody aggregation."""
    if len(features_list) < 2:
        raise ValueError("Need at least 2 clips")

    melody_scores = []
    pairwise_scores = []
    all_contributions = {k: [] for k in HARMONY_WEIGHTS}

    if main_melody_feature is not None:
        # Compare each harmony clip with main melody
        for harmony_feat in features_list:
            score, contribs = weighted_harmony_similarity(main_melody_feature, harmony_feat)
            melody_scores.append(score)
            for k, v in contribs.items():
                all_contributions[k].append(v)

        # Also compare harmony clips with each other
        if len(features_list) > 1:
            for i in range(len(features_list)):
                for j in range(i + 1, len(features_list)):
                    score, contribs = weighted_harmony_similarity(features_list[i], features_list[j])
                    pairwise_scores.append(score)
                    for k, v in contribs.items():
                        all_contributions[k].append(v)

        # Aggregate melody-to-harmony scores
        melody_arr = np.array(melody_scores)
        if aggregation == 'mean':
            melody_score = np.mean(melody_arr)
        elif aggregation == 'min':
            melody_score = np.min(melody_arr)
        elif aggregation == 'harmonic_mean':
            melody_score = len(melody_arr) / np.sum(1.0 / (melody_arr + 1e-10))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Aggregate harmony-to-harmony scores
        if pairwise_scores:
            pairwise_arr = np.array(pairwise_scores)
            if aggregation == 'mean':
                pairwise_score = np.mean(pairwise_arr)
            elif aggregation == 'min':
                pairwise_score = np.min(pairwise_arr)
            elif aggregation == 'harmonic_mean':
                pairwise_score = len(pairwise_arr) / np.sum(1.0 / (pairwise_arr + 1e-10))
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            # Weighted combination of both
            group_score = melody_weight * melody_score + (1 - melody_weight) * pairwise_score
        else:
            # Only one harmony clip, so only melody compatibility matters
            group_score = melody_score
    else:
        # Original behavior: all pairwise comparisons (no main melody)
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                score, contribs = weighted_harmony_similarity(features_list[i], features_list[j])
                pairwise_scores.append(score)
                for k, v in contribs.items():
                    all_contributions[k].append(v)

        scores_arr = np.array(pairwise_scores)
        if aggregation == 'mean':
            group_score = np.mean(scores_arr)
        elif aggregation == 'min':
            group_score = np.min(scores_arr)
        elif aggregation == 'harmonic_mean':
            group_score = len(scores_arr) / np.sum(1.0 / (scores_arr + 1e-10))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    avg_contribs = {k: np.mean(v) for k, v in all_contributions.items()}
    return group_score, avg_contribs


def find_best_harmony_groups(audio_clips: dict,
                             group_size: int = 2,
                             top_k: int = 5,
                             threshold: float = 0.70,
                             aggregation: str = 'mean',
                             main_melody_clip: tuple = None,
                             melody_weight: float = 0.7) -> list:
    """Find top-k harmony groups of specified size from clips."""
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    if group_size > len(audio_clips):
        raise ValueError(f"group_size {group_size} > clip count {len(audio_clips)}")

    print(f"Extracting features from {len(audio_clips)} clips...")

    # Extract features for all clips and maintain key mapping
    keys = list(audio_clips.keys())
    features = {}
    for key, clip_tuple in audio_clips.items():
        features[key] = extract_normalized_features(clip_tuple[0], clip_tuple[1])

    main_melody_feature = extract_normalized_features(main_melody_clip[0], main_melody_clip[1]) if main_melody_clip is not None else None
    n_combos = math.comb(len(audio_clips), group_size)

    print(f"Evaluating {n_combos} groups of size {group_size}...")

    groups = []
    for key_combo in combinations(keys, group_size):
        group_feats = [features[key] for key in key_combo]
        score, _ = calculate_group_compatibility(group_feats, main_melody_feature, aggregation, melody_weight)
        if score >= threshold:
            groups.append((key_combo, score))

    groups.sort(key=lambda x: x[1], reverse=True)
    return groups[:top_k]


def find_best_harmony_pairs(audio_clips: dict, top_k: int = 5, threshold: float = 0.70) -> list:
    """Find best harmony pairs (wrapper around find_best_harmony_groups)."""
    return find_best_harmony_groups(audio_clips, 2, top_k, threshold, 'mean')
