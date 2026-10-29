"""
Hourly flag-spike detection.

Takes a student's 24 hourly flag counts (index = hour of day, so index 15 is
15:00-15:59) and returns the hours that are anomalous spikes, as "HH:00" labels.

Uses scipy.signal.find_peaks with a prominence threshold: a bin only counts as
a spike if it rises at least `prominence` flags above the surrounding baseline.

Prominence is measured within a local window (`wlen` hours centred on the peak).
Without it, the day's highest hour is compared against the curfew zeros at the
start and end of the day, so any busy hour on a uniformly noisy day would count
as a spike. With the window, a steadily noisy student is not flagged; only an
hour that rises sharply above the hours around it is.
"""

import os
import warnings

import numpy as np
from scipy.signal import find_peaks

HOURS_PER_DAY = 24

# Minimum prominence (in flags) for a bin to count as a spike. Tunable without a
# code change: set SPIKE_PROMINENCE in the environment before starting the service.
DEFAULT_PROMINENCE = float(os.environ.get("SPIKE_PROMINENCE", "10"))

# Hours of context used to measure prominence (the peak +/- 3 hours by default).
# 7 is the smallest window that still catches a burst tailing off over 2-3 hours
# (e.g. 16, 11, 8); 5 misses it. Keep it odd; SciPy rounds an even value up.
# Set SPIKE_WINDOW_HOURS to override.
DEFAULT_WINDOW_HOURS = int(os.environ.get("SPIKE_WINDOW_HOURS", "7"))


def detect_spikes(counts, prominence=DEFAULT_PROMINENCE, window_hours=DEFAULT_WINDOW_HOURS):
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

    with warnings.catch_warnings():
        # Flat stretches inside the window have zero prominence; SciPy warns, but
        # they are simply not spikes, so the warning is noise in the service logs.
        warnings.filterwarnings("ignore", message="some peaks have a prominence of 0")
        peaks, _ = find_peaks(series, prominence=prominence, wlen=window_hours)
    return [f"{hour:02d}:00" for hour in peaks]
