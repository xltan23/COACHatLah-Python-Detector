import pytest

from detector import detect_spikes

# Sample day from SampleDetector.ipynb
SAMPLE = [5, 3, 3, 6, 10, 2, 2, 4, 5, 8, 4, 3, 3, 6, 16, 11, 8, 2, 2, 0, 0, 0, 1, 1]


def test_sample_day_detects_the_14_00_spike():
    assert detect_spikes(SAMPLE, prominence=10) == ["14:00"]


def test_flat_day_has_no_spikes():
    assert detect_spikes([2] * 24) == []


def test_all_zero_day_has_no_spikes():
    assert detect_spikes([0] * 24) == []


def test_single_isolated_burst_is_detected():
    counts = [0] * 24
    counts[19] = 12
    assert detect_spikes(counts) == ["19:00"]


def test_small_bump_below_prominence_is_ignored():
    counts = [0] * 24
    counts[10] = 3
    assert detect_spikes(counts, prominence=10) == []


def test_sustained_high_baseline_is_not_a_spike():
    # Steadily noisy student: many local maxima but none stand out from the baseline
    assert detect_spikes([8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9]) == []


def test_multiple_spikes_are_returned_in_hour_order():
    counts = [0] * 24
    counts[9] = 15
    counts[16] = 20
    assert detect_spikes(counts) == ["09:00", "16:00"]


def test_labels_are_zero_padded():
    counts = [0] * 24
    counts[7] = 12
    assert detect_spikes(counts) == ["07:00"]


@pytest.mark.parametrize("bad", [[1] * 23, [1] * 25, [], [-1] + [0] * 23])
def test_invalid_input_raises(bad):
    with pytest.raises(ValueError):
        detect_spikes(bad)
