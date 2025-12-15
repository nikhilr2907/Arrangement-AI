from typing import Dict, List

import numpy as np


def find_melodic_candidates(
    BARS: Dict[str, np.ndarray], activity_threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Find melodic candidates based on activity across the track.

    Filters stems that are active (non-empty) for more than the threshold percentage.

    Args:
        BARS: Dictionary mapping filename to array of bars for that stem
        activity_threshold: Minimum percentage of non-empty bars (0-1)

    Returns:
        Dictionary mapping filename to bars for melodic candidate stems
    """
    melodic_candidates = {}

    for filename, bars in BARS.items():
        # Calculate sum of absolute values for each bar
        summed_bars = np.array([np.sum(np.abs(bar)) for bar in bars])

        # Calculate percentage of non-empty bars
        percentage_active = np.sum(summed_bars > 0.01) / len(summed_bars)

        # Keep stems that are active more than threshold percentage
        if percentage_active > activity_threshold:
            melodic_candidates[filename] = bars

    return melodic_candidates   

        





