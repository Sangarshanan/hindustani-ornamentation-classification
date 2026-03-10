"""Tests for segmentation: static and adaptive windowers."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hcm_transcription.segmentation import (
    AdaptiveWindower,
    StaticWindower,
    segment_phrases,
)


# ---------------------------------------------------------------------------
# Static Windower
# ---------------------------------------------------------------------------

class TestStaticWindower:

    def test_default_frame_and_hop(self):
        """StaticWindower defaults to 56-sample frame, 5-sample hop."""
        w = StaticWindower()
        assert w.frame_samples == 56
        assert w.hop_samples == 5

    def test_segment_produces_correct_frame_size(self):
        """Each PCP should have exactly frame_samples samples."""
        w = StaticWindower(frame_samples=10, hop_samples=3)
        x = np.random.rand(50)
        t = np.arange(50) * 0.004444
        pcps = w.segment(x, t)
        for pcp in pcps:
            assert len(pcp.f0_samples) == 10

    def test_segment_count(self):
        """Number of segments should match expected frame count formula."""
        w = StaticWindower(frame_samples=10, hop_samples=3)
        x = np.random.rand(50)
        t = np.arange(50) * 0.004444
        pcps = w.segment(x, t)
        expected = 1 + (50 - 10) // 3
        assert len(pcps) == expected

    def test_instability_normalised(self):
        """Instability values should sum to approximately 1 (pre-smoothing)."""
        w = StaticWindower(frame_samples=10, hop_samples=2, ma_window=1)
        # Create a signal with some variation
        x = np.concatenate([np.ones(20), np.ones(20) * 2.0])
        instab = w.compute_instability(x)
        # With ma_window=1, no smoothing, so sum should be ~1
        assert len(instab) > 0
        assert abs(np.nansum(instab) - 1.0) < 0.01

    def test_empty_input(self):
        """Empty input should produce no segments."""
        w = StaticWindower()
        pcps = w.segment(np.array([]), np.array([]))
        assert len(pcps) == 0

    def test_input_shorter_than_frame(self):
        """Input shorter than frame should produce no segments."""
        w = StaticWindower(frame_samples=100)
        x = np.random.rand(50)
        t = np.arange(50) * 0.004444
        assert len(w.segment(x, t)) == 0
        assert len(w.compute_instability(x)) == 0


# ---------------------------------------------------------------------------
# Adaptive Windower
# ---------------------------------------------------------------------------

class TestAdaptiveWindower:

    def test_high_density_smaller_frames(self):
        """
        Adaptive windower should produce smaller frames on high-density
        input than on low-density input.
        """
        aw = AdaptiveWindower(
            base_frame_ms=250,
            min_frame_ms=100,
            max_frame_ms=500,
            alpha=2.0,
            sample_period_s=0.004444,
            density_window_s=0.5,
            density_smoothing_frames=2,
        )

        # Low density: constant pitch
        low_density_labels = np.array(["S3"] * 500)
        low_density_times = np.arange(500) * 0.004444

        # High density: rapidly alternating pitches
        high_density_labels = np.array(
            ["S3", "R3", "G3", "m3", "P3"] * 100
        )
        high_density_times = np.arange(500) * 0.004444

        low_frames = aw.get_frame_sizes_ms(low_density_labels, low_density_times)
        high_frames = aw.get_frame_sizes_ms(high_density_labels, high_density_times)

        # Mean frame size should be smaller for high-density input
        assert np.mean(high_frames) < np.mean(low_frames)

    def test_segment_non_empty(self):
        """AdaptiveWindower should produce segments for reasonable input."""
        aw = AdaptiveWindower(
            base_frame_ms=50,
            min_frame_ms=20,
            max_frame_ms=100,
            sample_period_s=0.004444,
            density_window_s=0.1,
        )
        x = np.random.rand(200)
        t = np.arange(200) * 0.004444
        pcps = aw.segment(x, t)
        assert len(pcps) > 0

    def test_empty_input(self):
        """Empty input should produce no segments."""
        aw = AdaptiveWindower()
        assert len(aw.segment(np.array([]), np.array([]))) == 0


# ---------------------------------------------------------------------------
# Phrase Segmentation
# ---------------------------------------------------------------------------

class TestPhraseSegmentation:

    def test_single_phrase_no_silence(self):
        """A continuous voiced signal should produce one phrase."""
        f0 = np.ones(100) * 440.0
        times = np.arange(100) * 0.004444
        q_hz = f0.copy()
        labels = np.array(["S3"] * 100)
        phrases = segment_phrases(f0, times, q_hz, labels, min_silence_duration_s=0.1)
        assert len(phrases) == 1

    def test_silence_splits_phrases(self):
        """A long silence gap should split into two phrases."""
        voiced = np.ones(50) * 440.0
        silence = np.zeros(200)  # ~0.9s silence at ~225 Hz
        f0 = np.concatenate([voiced, silence, voiced])
        times = np.arange(len(f0)) * 0.004444
        q_hz = f0.copy()
        labels = np.array(["S3"] * 50 + ["REST"] * 200 + ["S3"] * 50)
        phrases = segment_phrases(f0, times, q_hz, labels, min_silence_duration_s=0.5)
        assert len(phrases) == 2

    def test_short_silence_no_split(self):
        """A short silence should not split phrases."""
        voiced = np.ones(50) * 440.0
        silence = np.zeros(10)  # ~0.04s silence
        f0 = np.concatenate([voiced, silence, voiced])
        times = np.arange(len(f0)) * 0.004444
        q_hz = f0.copy()
        labels = np.array(["S3"] * 50 + ["REST"] * 10 + ["S3"] * 50)
        phrases = segment_phrases(f0, times, q_hz, labels, min_silence_duration_s=0.5)
        assert len(phrases) == 1
