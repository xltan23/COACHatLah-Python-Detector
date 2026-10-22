"""
Hourly flag-spike detection.

Takes a student's 24 hourly flag counts (index = hour of day, so index 15 is
15:00-15:59) and returns the hours that are anomalous spikes, as "HH:00" labels.

Uses scipy.signal.find_peaks with a prominence threshold: a bin only counts as
a spike if it rises at least `prominence` flags above the surrounding baseline,
so a student who is steadily noisy all day is not flagged for every local max.
"""

import os

import numpy as np
from scipy.signal import find_peaks

HOURS_PER_DAY = 24

# Minimum prominence (in flags) for a bin to count as a spike. Tunable without a
# code change: set SPIKE_PROMINENCE in the environment before starting the service.
DEFAULT_PROMINENCE = float(os.environ.get("SPIKE_PROMINENCE", "10"))


def detect_spikes(counts, prominence=DEFAULT_PROMINENCE):
    """Return the anomalous hours for one student as a list of "HH:00" strings.

    counts must have exactly 24 non-negative entries. find_peaks never reports
    the first or last sample as a peak, so spikes at 00:00 and 23:00 are not
    detected; the app is unavailable outside 07:00-22:00, so that is deliberate.
    """
    series = np.asarray(counts, dtype=float)
    if series.shape != (HOURS_PER_DAY,):
        raise ValueError(f"counts must have {HOURS_PER_DAY} entries, got shape {series.shape}")
    if (series < 0).any():
        raise ValueError("counts must be non-negative")

    peaks, _ = find_peaks(series, prominence=prominence)
    return [f"{hour:02d}:00" for hour in peaks]
