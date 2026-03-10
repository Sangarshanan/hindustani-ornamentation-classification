"""
Evaluation metrics for swara transcription and ornamentation detection.

Swara evaluation: weighted normalised Levenshtein distance (Section 4.1).
Ornamentation evaluation: boundary hit rate (Section 4.2) and per-class F1.

References:
  Jain & Arthur (2023) DLfM. DOI: 10.1145/3625135.3625137
  Daniel et al. (2008) ISMIR — weighted LD weights.
  Turnbull et al. (2007) ISMIR — boundary hit rate metric.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===================================================================
# SWARA EVALUATION (Section 4.1)
# ===================================================================

def weighted_normalized_levenshtein(
    pred: str,
    gt: str,
    ins_weight: float = 0.5,
    del_weight: float = 0.5,
    sub_weight: float = 1.0,
) -> float:
    """
    Weighted normalised Levenshtein distance similarity score.

    Returns a value in ``[0, 1]`` where 1 = perfect match.
    Weights follow Daniel et al. (2008) as used in paper Section 4.1.

    Parameters
    ----------
    pred : str
        Predicted swara string.
    gt : str
        Ground-truth swara string.
    ins_weight, del_weight, sub_weight : float
        Operation costs.

    Returns
    -------
    float
        Similarity score in [0, 1].
    """
    m, n = len(pred), len(gt)
    if n == 0:
        return 1.0 if m == 0 else 0.0

    # DP matrix
    dp = np.zeros((m + 1, n + 1), dtype=np.float64)
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + del_weight
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + ins_weight

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == gt[j - 1]:
                cost = 0.0
            else:
                cost = sub_weight
            dp[i][j] = min(
                dp[i - 1][j] + del_weight,
                dp[i][j - 1] + ins_weight,
                dp[i - 1][j - 1] + cost,
            )

    dist = dp[m][n]
    normalised = dist / n
    return max(0.0, 1.0 - normalised)


def evaluate_file(
    pred_phrases: List[str],
    gt_phrases: List[str],
    ins_weight: float = 0.5,
    del_weight: float = 0.5,
    sub_weight: float = 1.0,
) -> Dict[str, float]:
    """
    Compute phrase-level and file-level weighted-norm LD.

    Parameters
    ----------
    pred_phrases : List[str]
        List of predicted swara strings (one per phrase).
    gt_phrases : List[str]
        List of ground-truth swara strings.
    ins_weight, del_weight, sub_weight : float
        LD operation costs.

    Returns
    -------
    dict
        Keys: ``phrase_mean``, ``phrase_std``, ``file_score``.
    """
    n = min(len(pred_phrases), len(gt_phrases))
    phrase_scores = []
    for i in range(n):
        s = weighted_normalized_levenshtein(
            pred_phrases[i], gt_phrases[i],
            ins_weight, del_weight, sub_weight,
        )
        phrase_scores.append(s)

    # File-level: concatenate all phrases
    pred_full = "".join(pred_phrases)
    gt_full = "".join(gt_phrases)
    file_score = weighted_normalized_levenshtein(
        pred_full, gt_full, ins_weight, del_weight, sub_weight,
    )

    return {
        "phrase_mean": float(np.mean(phrase_scores)) if phrase_scores else 0.0,
        "phrase_std": float(np.std(phrase_scores)) if phrase_scores else 0.0,
        "file_score": file_score,
    }


# ===================================================================
# ORNAMENTATION EVALUATION (Section 4.2)
# ===================================================================

def boundary_hit_rate(
    pred_boundaries: np.ndarray,
    gt_boundaries: np.ndarray,
    tolerance_s: float = 0.5,
) -> Dict[str, float]:
    """
    Boundary hit rate metric (Turnbull et al. 2007).

    A hit is counted when a reference boundary falls within
    ``tolerance_s`` seconds of a predicted boundary.

    Parameters
    ----------
    pred_boundaries : np.ndarray
        Predicted ornament boundary times in seconds.
    gt_boundaries : np.ndarray
        Ground-truth boundary times.
    tolerance_s : float
        Tolerance window for a hit.

    Returns
    -------
    dict
        Keys: ``precision``, ``recall``, ``f1``.
    """
    if len(pred_boundaries) == 0 and len(gt_boundaries) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(pred_boundaries) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if len(gt_boundaries) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Recall: fraction of GT boundaries matched by a prediction
    hits_recall = 0
    for gt_b in gt_boundaries:
        if np.any(np.abs(pred_boundaries - gt_b) <= tolerance_s):
            hits_recall += 1
    recall = hits_recall / len(gt_boundaries)

    # Precision: fraction of predicted boundaries matched by GT
    hits_prec = 0
    for pred_b in pred_boundaries:
        if np.any(np.abs(gt_boundaries - pred_b) <= tolerance_s):
            hits_prec += 1
    precision = hits_prec / len(pred_boundaries)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def ornament_classification_report(
    pred_labels: List[str],
    gt_labels: List[str],
    classifiable_labels: Optional[set] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Per-class precision, recall, F1 for classifier-output ornament types.

    Unknown / unclassifiable ornaments (gamak, khatka, zamzama, other) are
    excluded with a logged warning.

    Parameters
    ----------
    pred_labels : List[str]
        Predicted ornament labels.
    gt_labels : List[str]
        Ground-truth labels (canonical names).
    classifiable_labels : set, optional
        Set of labels the classifier can output. Defaults to
        ``{"kan", "meend", "andolan", "murki"}``.

    Returns
    -------
    dict
        Nested dict ``{label: {precision, recall, f1}}``.
    """
    if classifiable_labels is None:
        classifiable_labels = {"kan", "meend", "andolan", "murki"}

    # Filter to classifiable
    excluded_gt = [l for l in gt_labels if l not in classifiable_labels]
    if excluded_gt:
        logger.warning(
            "Excluding %d ground-truth ornaments not in classifier output: %s",
            len(excluded_gt),
            set(excluded_gt),
        )

    # Build per-class counts
    report: Dict[str, Dict[str, float]] = {}
    for label in sorted(classifiable_labels):
        tp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == label and g == label)
        fp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == label and g != label)
        fn = sum(1 for p, g in zip(pred_labels, gt_labels) if p != label and g == label)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        report[label] = {"precision": prec, "recall": rec, "f1": f1}

    return report


# ===================================================================
# ADAPTIVE vs STATIC ABLATION
# ===================================================================

def compare_windowing_strategies(
    results_static: Dict[str, float],
    results_adaptive: Dict[str, float],
) -> pd.DataFrame:
    """
    Produce a side-by-side comparison DataFrame of all metrics.

    Parameters
    ----------
    results_static : dict
        Metric name → value for static windowing.
    results_adaptive : dict
        Metric name → value for adaptive windowing.

    Returns
    -------
    pd.DataFrame
        Columns: ``metric``, ``static``, ``adaptive``, ``delta``.
    """
    metrics = sorted(set(results_static) | set(results_adaptive))
    rows = []
    for m in metrics:
        s = results_static.get(m, float("nan"))
        a = results_adaptive.get(m, float("nan"))
        rows.append({
            "metric": m,
            "static": s,
            "adaptive": a,
            "delta": a - s,
        })
    return pd.DataFrame(rows)
