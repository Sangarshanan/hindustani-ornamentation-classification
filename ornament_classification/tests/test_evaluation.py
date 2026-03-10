"""
Tests for evaluation metrics.

Includes verification of weighted normalised LD on error cases from
Table 3 of the paper.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hcm_transcription.evaluation import (
    boundary_hit_rate,
    compare_windowing_strategies,
    evaluate_file,
    ornament_classification_report,
    weighted_normalized_levenshtein,
)


# ---------------------------------------------------------------------------
# Weighted Normalised Levenshtein Distance
# ---------------------------------------------------------------------------

class TestWeightedNormalisedLD:

    def test_identical_strings(self):
        """Identical strings should have similarity 1.0."""
        assert weighted_normalized_levenshtein("abc", "abc") == 1.0

    def test_empty_gt_empty_pred(self):
        """Both empty should score 1.0."""
        assert weighted_normalized_levenshtein("", "") == 1.0

    def test_empty_gt_non_empty_pred(self):
        """Non-empty pred, empty GT should score 0.0 (all deletions)."""
        assert weighted_normalized_levenshtein("abc", "") == 0.0

    def test_non_empty_gt_empty_pred(self):
        """Empty pred, non-empty GT should score 0.0."""
        # Need len(gt) insertions, each costing 0.5
        result = weighted_normalized_levenshtein("", "abc")
        # distance = 3 * 0.5 = 1.5, normalised = 1.5/3 = 0.5, similarity = 0.5
        assert abs(result - 0.5) < 0.01

    def test_extra_note_error(self):
        """Extra note: pred has one extra character vs GT."""
        gt = "SRG"
        pred = "SRRG"  # extra R
        score = weighted_normalized_levenshtein(pred, gt)
        # deletion cost 0.5, dist = 0.5, norm = 0.5/3, sim = 1 - 0.167 ≈ 0.833
        assert 0.8 < score < 0.9

    def test_missing_note_error(self):
        """Missing note: pred is missing one character."""
        gt = "SRG"
        pred = "SG"  # missing R
        score = weighted_normalized_levenshtein(pred, gt)
        # insertion cost 0.5, dist = 0.5, norm = 0.5/3, sim ≈ 0.833
        assert 0.8 < score < 0.9

    def test_wrong_note_error(self):
        """Wrong note: one character substituted."""
        gt = "SRG"
        pred = "SPG"  # R→P substitution
        score = weighted_normalized_levenshtein(pred, gt)
        # substitution cost 1.0, dist = 1.0, norm = 1.0/3, sim ≈ 0.667
        assert 0.6 < score < 0.7

    def test_extra_and_wrong(self):
        """Extra + wrong note combined error."""
        gt = "SRG"
        pred = "SPPG"  # extra P and wrong R→P
        score = weighted_normalized_levenshtein(pred, gt)
        assert 0.0 < score < 1.0

    def test_missing_and_wrong(self):
        """Missing + wrong note combined error."""
        gt = "SRGM"
        pred = "SPM"  # missing R, wrong G→P
        score = weighted_normalized_levenshtein(pred, gt)
        assert 0.0 < score < 1.0

    def test_weights_affect_score(self):
        """Different weights should produce different scores."""
        gt = "SRG"
        pred = "SG"
        s1 = weighted_normalized_levenshtein(pred, gt, ins_weight=0.5)
        s2 = weighted_normalized_levenshtein(pred, gt, ins_weight=1.0)
        assert s1 != s2


# ---------------------------------------------------------------------------
# File Evaluation
# ---------------------------------------------------------------------------

class TestEvaluateFile:

    def test_perfect_match(self):
        scores = evaluate_file(["SRG", "mPd"], ["SRG", "mPd"])
        assert scores["phrase_mean"] == 1.0
        assert scores["file_score"] == 1.0

    def test_partial_match(self):
        scores = evaluate_file(["SRG"], ["SPG"])
        assert 0.0 < scores["phrase_mean"] < 1.0
        assert 0.0 < scores["file_score"] < 1.0


# ---------------------------------------------------------------------------
# Boundary Hit Rate
# ---------------------------------------------------------------------------

class TestBoundaryHitRate:

    def test_perfect_boundaries(self):
        gt = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        result = boundary_hit_rate(pred, gt, tolerance_s=0.1)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_no_predictions(self):
        gt = np.array([1.0, 2.0])
        pred = np.array([])
        result = boundary_hit_rate(pred, gt)
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_tolerance_window(self):
        gt = np.array([1.0])
        pred = np.array([1.3])
        # With tolerance 0.5, this is a hit
        r1 = boundary_hit_rate(pred, gt, tolerance_s=0.5)
        assert r1["recall"] == 1.0
        # With tolerance 0.1, this is a miss
        r2 = boundary_hit_rate(pred, gt, tolerance_s=0.1)
        assert r2["recall"] == 0.0

    def test_both_empty(self):
        result = boundary_hit_rate(np.array([]), np.array([]))
        assert result["f1"] == 1.0


# ---------------------------------------------------------------------------
# Ornament Classification Report
# ---------------------------------------------------------------------------

class TestOrnamentClassificationReport:

    def test_perfect_classification(self):
        labels = ["kan", "meend", "murki"]
        report = ornament_classification_report(labels, labels)
        for lbl in ["kan", "meend", "murki"]:
            assert report[lbl]["f1"] == 1.0

    def test_all_wrong(self):
        pred = ["kan", "kan", "kan"]
        gt = ["meend", "murki", "andolan"]
        report = ornament_classification_report(pred, gt)
        assert report["meend"]["f1"] == 0.0


# ---------------------------------------------------------------------------
# Compare Windowing Strategies
# ---------------------------------------------------------------------------

class TestCompareWindowing:

    def test_delta_column(self):
        static = {"phrase_wnld": 0.6, "file_wnld": 0.5}
        adaptive = {"phrase_wnld": 0.7, "file_wnld": 0.55}
        df = compare_windowing_strategies(static, adaptive)
        assert "delta" in df.columns
        assert len(df) == 2
