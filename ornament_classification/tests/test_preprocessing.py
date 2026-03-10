"""Tests for preprocessing: tonic normalization and quantization."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hcm_transcription.preprocessing import TonicNormalizer
from hcm_transcription.utils import RATIO_3OCT, SWARA_NAMES_3OCT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_raga_dict():
    """Minimal raga dictionary for testing."""
    return {
        "ragas": {
            "yaman": [
                1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
            ],
        },
    }


@pytest.fixture
def normalizer(sample_raga_dict):
    return TonicNormalizer(
        tonic_hz=261.63,  # C4 as Sa
        raga_name="yaman",
        raga_dict=sample_raga_dict,
        shruti_threshold_cents=35.0,
    )


# ---------------------------------------------------------------------------
# Test: just-intonation ratios
# ---------------------------------------------------------------------------

class TestJustIntonationRatios:
    """Verify all 12 swara ratios match Table 1 of the paper."""

    EXPECTED_RATIOS = {
        "Sa": 1.0, "re": 16/15, "Re": 9/8, "ga": 6/5, "Ga": 5/4,
        "ma": 4/3, "Ma": 45/32, "Pa": 3/2, "dha": 8/5, "Dha": 27/16,
        "ni": 9/5, "Ni": 15/8,
    }

    def test_single_octave_ratios(self):
        """The 12 single-octave ratios in RATIO_3OCT[12:24] match Table 1."""
        mid_octave = RATIO_3OCT[12:24]
        expected = np.array([
            1.0, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 27/16, 9/5, 15/8
        ])
        np.testing.assert_allclose(mid_octave, expected, rtol=1e-10)

    def test_lower_octave_half(self):
        """Lower octave (idx 0-11) should be half of middle octave."""
        np.testing.assert_allclose(RATIO_3OCT[:12], RATIO_3OCT[12:24] * 0.5, rtol=1e-10)

    def test_upper_octave_double(self):
        """Upper octave (idx 24-35) should be double of middle octave."""
        np.testing.assert_allclose(RATIO_3OCT[24:36], RATIO_3OCT[12:24] * 2.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test: quantization
# ---------------------------------------------------------------------------

class TestQuantization:

    def test_exact_swara_frequency(self, normalizer):
        """A pitch exactly at a swara frequency should be assigned that swara."""
        tonic = 261.63
        # Pa in middle octave = tonic * 3/2
        pa_freq = tonic * 1.5
        f0 = np.array([pa_freq])
        labels, quant = normalizer.quantize(f0)
        assert labels[0] == "P3"
        assert abs(quant[0] - pa_freq) < 0.01

    def test_silence_mapped_to_rest(self, normalizer):
        """Zero f0 values should map to REST."""
        f0 = np.array([0.0, 0.0, 0.0])
        labels, quant = normalizer.quantize(f0)
        assert all(l == "REST" for l in labels)
        assert all(q == 0.0 for q in quant)

    def test_35_cent_threshold_snaps_to_previous(self, normalizer):
        """
        A pitch between two swaras (>35 cents from closest) should snap
        to the previous sample's swara.
        """
        tonic = 261.63
        sa_freq = tonic  # exact Sa
        # A frequency ~60 cents above Sa (between Sa and re)
        ambiguous_freq = tonic * (2 ** (60 / 1200))
        f0 = np.array([sa_freq, ambiguous_freq])

        labels, _ = normalizer.quantize(f0)
        # First sample: exact Sa → S3
        assert labels[0] == "S3"
        # Second sample: ambiguous → should snap to previous (S3)
        assert labels[1] == "S3"

    def test_reduced_sequence_collapses_duplicates(self, normalizer):
        """Consecutive identical swaras should collapse into one (octave-free)."""
        labels = np.array(["S3", "S3", "S3", "R3", "R3", "G3", "REST", "P3"])
        result = normalizer.to_reduced_sequence(labels)
        assert result == "SRGP"

    def test_reduced_label_list_collapses_duplicates(self, normalizer):
        """to_reduced_label_list preserves full labels with octave info."""
        labels = np.array(["S3", "S3", "S3", "R3", "R3", "G3", "REST", "P3"])
        result = normalizer.to_reduced_label_list(labels)
        assert result == ["S3", "R3", "G3", "P3"]

    def test_reduced_sequence_empty_for_all_rest(self, normalizer):
        """All REST should produce empty string."""
        labels = np.array(["REST", "REST", "REST"])
        assert normalizer.to_reduced_sequence(labels) == ""
