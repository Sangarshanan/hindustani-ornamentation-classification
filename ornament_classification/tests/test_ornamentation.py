"""Tests for ornamentation: classifier and label mapper."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hcm_transcription.ornamentation import (
    CLASSIFIER_LABELS,
    LabelMapper,
    OrnamentClassifier,
    OrnamentEvent,
)


# ---------------------------------------------------------------------------
# Label Mapper
# ---------------------------------------------------------------------------

class TestLabelMapper:

    def test_meend_start(self):
        canonical, etype = LabelMapper.parse_ohv_label("me_s")
        assert canonical == "meend"
        assert etype == "start"

    def test_meend_end(self):
        canonical, etype = LabelMapper.parse_ohv_label("me_e")
        assert canonical == "meend"
        assert etype == "end"

    def test_kan(self):
        canonical, etype = LabelMapper.parse_ohv_label("k_s")
        assert canonical == "kan"
        assert etype == "start"

    def test_murki(self):
        canonical, etype = LabelMapper.parse_ohv_label("mu_s")
        assert canonical == "murki"
        assert etype == "start"

    def test_andolan(self):
        canonical, etype = LabelMapper.parse_ohv_label("a_s")
        assert canonical == "andolan"
        assert etype == "start"

    def test_gamak(self):
        canonical, etype = LabelMapper.parse_ohv_label("g_s")
        assert canonical == "gamak"
        assert etype == "start"

    def test_khatka(self):
        canonical, etype = LabelMapper.parse_ohv_label("kh_s")
        assert canonical == "khatka"
        assert etype == "start"

    def test_zamzama(self):
        canonical, etype = LabelMapper.parse_ohv_label("z_s")
        assert canonical == "zamzama"
        assert etype == "start"

    def test_compound_label(self):
        """Compound labels like 'c_me_me_me_s' should map to meend."""
        canonical, etype = LabelMapper.parse_ohv_label("c_me_me_me_s")
        assert canonical == "meend"
        assert etype == "start"

    def test_compound_mixed(self):
        """Compound with mixed types: 'c_k_k_kh_s' → khatka (last recognised)."""
        canonical, etype = LabelMapper.parse_ohv_label("c_k_k_kh_s")
        assert canonical == "khatka"
        assert etype == "start"

    def test_none_label(self):
        canonical, etype = LabelMapper.parse_ohv_label("none")
        assert canonical == "none"
        assert etype == "point"

    def test_is_classifiable(self):
        assert LabelMapper.is_classifiable("kan") is True
        assert LabelMapper.is_classifiable("meend") is True
        assert LabelMapper.is_classifiable("gamak") is False
        assert LabelMapper.is_classifiable("khatka") is False


# ---------------------------------------------------------------------------
# Ornament Classifier
# ---------------------------------------------------------------------------

class TestOrnamentClassifier:

    def test_no_instability_no_ornaments(self):
        """Flat instability below threshold should produce no events."""
        clf = OrnamentClassifier(instability_threshold=0.1)
        instab = np.ones(50) * 0.01  # all below threshold
        quant = np.ones(250) * 440.0
        times = np.arange(250) * 0.004444
        events = clf.classify(instab, quant, times)
        assert len(events) == 0

    def test_single_frame_spike_is_kan(self):
        """A single unstable frame should be classified as KAN."""
        clf = OrnamentClassifier(
            instability_threshold=0.05,
            min_merge_gap=0,
            kan_max_frames=1,
        )
        instab = np.ones(20) * 0.01
        instab[10] = 0.5  # single spike
        quant = np.ones(100) * 440.0
        times = np.arange(100) * 0.004444

        events = clf.classify(instab, quant, times, hop_samples=5)
        assert len(events) >= 1
        assert events[0].label == "kan"

    def test_extended_unstable_region_is_meend(self):
        """Multiple consecutive unstable frames with few direction changes → MEEND."""
        clf = OrnamentClassifier(
            instability_threshold=0.05,
            min_merge_gap=3,
            kan_max_frames=1,
        )
        instab = np.ones(30) * 0.01
        instab[10:16] = 0.5  # 6 consecutive unstable frames
        # Create a monotone quantized segment (few direction changes)
        quant = np.linspace(440, 500, 150)
        times = np.arange(150) * 0.004444

        events = clf.classify(instab, quant, times, hop_samples=5)
        assert len(events) >= 1
        assert events[0].label == "meend"

    def test_empty_instability(self):
        """Empty instability should produce no events."""
        clf = OrnamentClassifier()
        events = clf.classify(np.array([]), np.array([]), np.array([]))
        assert len(events) == 0
