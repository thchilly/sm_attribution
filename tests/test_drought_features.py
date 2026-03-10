"""
Unit tests for _runtheo_features_1d and the theory-of-runs drought
feature extraction.

Tests cover:
    - DRD uses last-min-to-end (not D − DDD)
    - Peak intensity = |min(SSI)|
    - TTM10/TTS15/TTE20 first-crossing semantics
    - Bridge default is 3 months
    - Inter-arrival (Ld) and return period (Rp)
    - Edge cases: single event, all-NaN, no events, repeated min
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sm_attribution.analysis.drought_features import _runtheo_features_1d


# ── helpers ──────────────────────────────────────────────────────────────

def _unpack(result):
    """Unpack the 12-tuple into a dict for readable assertions."""
    names = [
        "duration", "magnitude", "intensity", "peak_intensity",
        "ddd", "ttm10", "tts15", "tte20", "drd",
        "n_events", "interarrival", "return_period",
    ]
    return {n: float(v) for n, v in zip(names, result)}


def _isnan(val):
    return math.isnan(float(val))


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def simple_one_event():
    """
    Single 10-month drought that reaches -2.5 at month 4 (0-based=3)
    and again at month 7 (0-based=6).

    SSI:  0.5,  -0.3, -0.8, -2.5, -1.2, -0.9, -2.5, -1.0, -0.4,  0.3
    idx:   0      1    2     3     4      5      6     7     8     9

    After bridging & filtering:
      - The 0.5 at idx 0 and 0.3 at idx 9 are in [0,1] but are single
        months, so they get bridged (< 3).
      - The whole series becomes negative.  But min = -2.5 <= -1.0,
        so the run survives Rule B.
    """
    return np.array(
        [0.5, -0.3, -0.8, -2.5, -1.2, -0.9, -2.5, -1.0, -0.4, 0.3],
        dtype=np.float32,
    )


@pytest.fixture
def two_events():
    """
    Two well-separated drought events.

    Event 1: months 2-6 (5 months, min -1.5 at month 4)
    Event 2: months 12-16 (5 months, min -2.0 at month 14)
    Separated by 5 months of positive values > 1 (no bridging).
    """
    ssi = np.ones(20, dtype=np.float32) * 1.5  # all positive, > 1
    # Event 1
    ssi[2:7] = [-0.3, -0.8, -1.5, -0.9, -0.4]
    # Event 2
    ssi[12:17] = [-0.5, -1.0, -2.0, -1.2, -0.3]
    return ssi


# ── tests ────────────────────────────────────────────────────────────────

class TestReturnShape:
    """Basic structural tests."""

    def test_returns_12_values(self, simple_one_event):
        result = _runtheo_features_1d(simple_one_event)
        assert len(result) == 12

    def test_all_nan_input(self):
        """All-NaN input (ocean) → every feature including n_events is NaN."""
        ssi = np.full(50, np.nan, dtype=np.float32)
        r = _unpack(_runtheo_features_1d(ssi))
        assert _isnan(r["n_events"])
        assert _isnan(r["duration"])
        assert _isnan(r["interarrival"])

    def test_no_drought(self):
        ssi = np.ones(50, dtype=np.float32) * 2.0  # all positive
        r = _unpack(_runtheo_features_1d(ssi))
        assert r["n_events"] == 0.0
        assert _isnan(r["duration"])

    def test_weak_event_removed(self):
        """A negative run whose min > severity_threshold (-1) is discarded."""
        ssi = np.ones(20, dtype=np.float32) * 1.5
        ssi[5:10] = [-0.1, -0.2, -0.3, -0.2, -0.1]  # never reaches -1
        r = _unpack(_runtheo_features_1d(ssi))
        assert r["n_events"] == 0.0


class TestDRD:
    """DRD must use LAST occurrence of minimum, not D − DDD."""

    def test_drd_single_min(self, two_events):
        """When min occurs once, DRD = D - min_pos."""
        r = _unpack(_runtheo_features_1d(two_events))
        # Event 1: min -1.5 at position 2 (within event, 0-based),
        #   D=5, last_min=pos 2, DRD = 5 - (2+1) = 2
        # Event 2: min -2.0 at position 2 (within event, 0-based),
        #   D=5, last_min=pos 2, DRD = 5 - (2+1) = 2
        # Both drds = 2. Mean = 2
        assert r["drd"] == pytest.approx(2.0, abs=0.01)

    def test_drd_repeated_min(self):
        """
        Min SSI -2.0 appears at within-event positions 1 and 7 (0-based)
        in a 10-month event. DDD=2 (first+1), DRD=10-(7+1)=2.
        Old code would give DRD = 10-2 = 8 (wrong).
        """
        ssi = np.ones(20, dtype=np.float32) * 1.5
        # Single 10-month event: months 3-12
        ssi[3:13] = [-0.5, -2.0, -0.8, -1.2, -0.9, -0.6, -1.1, -2.0, -0.4, -0.3]
        #  within-event:  0     1     2     3     4     5     6     7     8     9
        # min = -2.0, first at pos 1, last at pos 7
        r = _unpack(_runtheo_features_1d(ssi))
        assert r["n_events"] == 1.0
        assert r["ddd"] == pytest.approx(2.0, abs=0.01)   # first min + 1
        assert r["drd"] == pytest.approx(2.0, abs=0.01)   # D - (last min + 1) = 10 - 8 = 2
        # Crucially: DDD + DRD = 4 ≠ D = 10.  That's correct.

    def test_drd_min_at_end(self):
        """When min is at the last step, DRD should be 0."""
        ssi = np.ones(15, dtype=np.float32) * 1.5
        ssi[3:8] = [-0.3, -0.5, -0.8, -1.0, -1.5]  # min at pos 4 (last)
        r = _unpack(_runtheo_features_1d(ssi))
        assert r["drd"] == pytest.approx(0.0, abs=0.01)


class TestPeakIntensity:
    """Peak intensity = |min(SSI)|."""

    def test_basic(self, two_events):
        r = _unpack(_runtheo_features_1d(two_events))
        # Event 1 peak = |-1.5| = 1.5, Event 2 peak = |-2.0| = 2.0
        # Mean = 1.75
        assert r["peak_intensity"] == pytest.approx(1.75, abs=0.01)

    def test_single_event(self):
        ssi = np.ones(15, dtype=np.float32) * 1.5
        ssi[3:8] = [-0.3, -0.5, -3.5, -1.0, -0.4]
        r = _unpack(_runtheo_features_1d(ssi))
        assert r["peak_intensity"] == pytest.approx(3.5, abs=0.01)


class TestDDDThresholds:
    """TTM10/TTS15/TTE20 = first crossing of the threshold."""

    def test_ttm10_tts15_tte20(self):
        ssi = np.ones(20, dtype=np.float32) * 1.5
        # 8-month event: gradual deepening
        ssi[3:11] = [-0.3, -0.8, -1.2, -1.7, -2.3, -1.5, -0.9, -0.4]
        # within-event pos: 0    1     2     3     4     5     6     7
        # SSI ≤ -1.0 first at pos 2 → ttm10 = 3
        # SSI ≤ -1.5 first at pos 3 → tts15 = 4
        # SSI ≤ -2.0 first at pos 4 → tte20 = 5
        r = _unpack(_runtheo_features_1d(ssi))
        assert r["ttm10"] == pytest.approx(3.0, abs=0.01)
        assert r["tts15"] == pytest.approx(4.0, abs=0.01)
        assert r["tte20"] == pytest.approx(5.0, abs=0.01)

    def test_threshold_never_crossed(self):
        ssi = np.ones(20, dtype=np.float32) * 1.5
        ssi[3:8] = [-0.3, -0.5, -1.2, -0.9, -0.4]  # min = -1.2
        r = _unpack(_runtheo_features_1d(ssi))
        assert r["ttm10"] == pytest.approx(3.0, abs=0.01)  # -1.2 ≤ -1.0
        assert _isnan(r["tts15"])  # never reaches -1.5
        assert _isnan(r["tte20"])  # never reaches -2.0


class TestBridge:
    """Bridge default is 3 months (not 6)."""

    def test_bridge_3_months_merges(self):
        """Two negative runs separated by 2-month gap [0, 0.5] → bridged."""
        ssi = np.ones(20, dtype=np.float32) * 1.5
        ssi[3:6] = [-0.5, -1.2, -0.3]           # event A
        ssi[6:8] = [0.0, 0.5]                     # gap: 2 months in [0,1]
        ssi[8:11] = [-0.4, -1.5, -0.2]           # event B
        r = _unpack(_runtheo_features_1d(ssi))
        # Gap is 2 months < bridge=3, so it is merged → 1 event of 8 months
        assert r["n_events"] == 1.0
        assert r["duration"] == pytest.approx(8.0, abs=0.01)

    def test_bridge_3_months_no_merge(self):
        """Two negative runs separated by 4-month gap → NOT bridged."""
        ssi = np.ones(20, dtype=np.float32) * 1.5
        ssi[2:5] = [-0.5, -1.2, -0.3]
        ssi[5:9] = [0.0, 0.5, 0.3, 0.8]          # gap: 4 months ≥ 3
        ssi[9:12] = [-0.4, -1.5, -0.2]
        r = _unpack(_runtheo_features_1d(ssi))
        # Gap is 4 months ≥ bridge=3, so two separate events
        assert r["n_events"] == 2.0

    def test_bridge_custom_6(self):
        """Explicit bridge_len_months=6 merges a 5-month gap."""
        ssi = np.ones(25, dtype=np.float32) * 1.5
        ssi[2:5] = [-0.5, -1.2, -0.3]
        ssi[5:10] = [0.0, 0.5, 0.3, 0.8, 0.2]    # 5-month gap
        ssi[10:13] = [-0.4, -1.5, -0.2]
        r = _unpack(_runtheo_features_1d(ssi, bridge_len_months=6))
        assert r["n_events"] == 1.0


class TestInterarrivalAndReturnPeriod:
    """Ld and Rp require ≥ 2 events."""

    def test_single_event_nan(self):
        ssi = np.ones(15, dtype=np.float32) * 1.5
        ssi[3:8] = [-0.3, -0.5, -1.5, -0.9, -0.4]
        r = _unpack(_runtheo_features_1d(ssi))
        assert r["n_events"] == 1.0
        assert _isnan(r["interarrival"])
        assert _isnan(r["return_period"])

    def test_two_events(self, two_events):
        r = _unpack(_runtheo_features_1d(two_events))
        assert r["n_events"] == 2.0
        # Event 1 starts at idx 2, Event 2 starts at idx 12
        # Ld = 12 - 2 = 10
        assert r["interarrival"] == pytest.approx(10.0, abs=0.01)
        # Event 1 ends at idx 6 (inclusive), Event 2 starts at idx 12
        # Rp = 12 - 6 = 6
        assert r["return_period"] == pytest.approx(6.0, abs=0.01)

    def test_three_events(self):
        """Three events: Ld and Rp are means of two pairs."""
        ssi = np.ones(30, dtype=np.float32) * 1.5
        ssi[2:5] = [-0.5, -1.2, -0.3]    # event 1: start=2, end=4
        ssi[10:13] = [-0.4, -1.5, -0.2]  # event 2: start=10, end=12
        ssi[20:23] = [-0.6, -1.8, -0.1]  # event 3: start=20, end=22
        r = _unpack(_runtheo_features_1d(ssi))
        assert r["n_events"] == 3.0
        # Ld = mean([10-2, 20-10]) = mean([8, 10]) = 9
        assert r["interarrival"] == pytest.approx(9.0, abs=0.01)
        # Rp = mean([10-4, 20-12]) = mean([6, 8]) = 7
        assert r["return_period"] == pytest.approx(7.0, abs=0.01)


class TestClassicFeatures:
    """Verify duration, magnitude, intensity are unchanged."""

    def test_classic_single_event(self):
        ssi = np.ones(15, dtype=np.float32) * 1.5
        ssi[3:8] = [-0.5, -1.0, -2.0, -0.8, -0.3]  # 5 months
        r = _unpack(_runtheo_features_1d(ssi))
        assert r["duration"] == pytest.approx(5.0, abs=0.01)
        # Magnitude = 0.5+1.0+2.0+0.8+0.3 = 4.6
        assert r["magnitude"] == pytest.approx(4.6, abs=0.05)
        # Intensity = 4.6/5 = 0.92
        assert r["intensity"] == pytest.approx(0.92, abs=0.02)

    def test_ddd_first_min(self):
        """DDD uses the FIRST occurrence of min."""
        ssi = np.ones(15, dtype=np.float32) * 1.5
        ssi[3:8] = [-0.5, -2.0, -0.8, -2.0, -0.3]
        # min=-2.0 at within-event pos 1 (first), pos 3 (last)
        r = _unpack(_runtheo_features_1d(ssi))
        assert r["ddd"] == pytest.approx(2.0, abs=0.01)    # first + 1
        assert r["drd"] == pytest.approx(1.0, abs=0.01)    # D - (last+1) = 5 - 4 = 1
