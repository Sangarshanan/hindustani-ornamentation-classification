"""
Rule-based ornament detection and classification.

Implements the decision-tree classifier from Figure 5 of the paper:
  unstable region → direction changes → kan / meend / andolan / murki.

Also provides ``LabelMapper`` for canonicalising OHV annotation labels.

Reference: Jain & Arthur (2023) DLfM, Section 3.2. DOI: 10.1145/3625135.3625137
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from hcm_transcription.segmentation import BaseWindower

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ornament event dataclass
# ---------------------------------------------------------------------------

@dataclass
class OrnamentEvent:
    """A detected ornament region."""
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    label: str          # kan, meend, andolan, murki, or unknown
    direction_changes: int
    unique_pitches: int


# ---------------------------------------------------------------------------
# OHV Label Mapper
# ---------------------------------------------------------------------------

# Canonical mapping from OHV CSV label codes to ornament names
OHV_LABEL_MAP: Dict[str, str] = {
    "k":  "kan",
    "me": "meend",
    "a":  "andolan",
    "mu": "murki",
    "g":  "gamak",
    "kh": "khatka",
    "z":  "zamzama",
    "o":  "other",
}

# Labels that the f0-based classifier can produce
CLASSIFIER_LABELS = {"kan", "meend", "andolan", "murki"}

# Labels present in OHV but not distinguishable from f0 alone
UNCLASSIFIABLE_LABELS = {"gamak", "khatka", "zamzama", "other"}


class LabelMapper:
    """
    Canonicalise OHV annotation labels.

    OHV format uses paired start/end tags like ``k_s`` / ``k_e``,
    ``me_s`` / ``me_e``, etc.  Compound labels (e.g. ``c_k_k_kh``)
    are split and each component is mapped individually.
    """

    @staticmethod
    def parse_ohv_label(raw: str) -> Tuple[str, str]:
        """
        Parse a raw OHV label into (canonical_name, event_type).

        Parameters
        ----------
        raw : str
            Raw label from CSV, e.g. ``"me_s"``, ``"c_k_k_s"``, ``"none"``.

        Returns
        -------
        canonical : str
            One of the canonical ornament names, ``"none"``, or ``"unknown"``.
        event_type : str
            ``"start"``, ``"end"``, or ``"point"`` (for ``"none"``).
        """
        raw = raw.strip()
        if raw.lower() in ("none", ""):
            return "none", "point"

        # Determine if start or end
        if raw.endswith("_s"):
            event_type = "start"
            body = raw[:-2]
        elif raw.endswith("_e"):
            event_type = "end"
            body = raw[:-2]
        else:
            return "unknown", "point"

        # Handle compound labels (c_me_me_me_s → meend)
        # Strip leading "c_" prefix
        if body.startswith("c_"):
            body = body[2:]

        # Split remaining components and take the last recognized one
        parts = body.split("_")
        canonical = "unknown"
        for part in reversed(parts):
            if part in OHV_LABEL_MAP:
                canonical = OHV_LABEL_MAP[part]
                break

        return canonical, event_type

    @staticmethod
    def is_classifiable(label: str) -> bool:
        """Return True if the label is one the classifier can produce."""
        return label in CLASSIFIER_LABELS


# ---------------------------------------------------------------------------
# Ornament Classifier
# ---------------------------------------------------------------------------

class OrnamentClassifier:
    """
    Rule-based ornament classifier (Figure 5 of the paper).

    For each unstable region in the instability curve:
      1. Count direction changes (local minima + maxima).
      2. If direction_changes <= 2:
           - 1 unstable frame  → KAN
           - >1 unstable frames → MEEND
      3. If direction_changes > 2:
           - Count unique pitch regions
           - 2–3 unique pitches  → ANDOLAN
           - >3 unique pitches   → MURKI

    Parameters
    ----------
    instability_threshold : float
        Threshold above which a frame is considered unstable.
    min_merge_gap : int
        Merge consecutive unstable regions separated by fewer samples.
    kan_max_frames : int
        Maximum number of unstable frames for a KAN.
    andolan_max_pitches : int
        Upper bound on unique pitches for ANDOLAN classification.
    """

    def __init__(
        self,
        instability_threshold: float = 0.1,
        min_merge_gap: int = 3,
        kan_max_frames: int = 1,
        andolan_max_pitches: int = 3,
    ) -> None:
        self.instability_threshold = instability_threshold
        self.min_merge_gap = min_merge_gap
        self.kan_max_frames = kan_max_frames
        self.andolan_max_pitches = andolan_max_pitches

    @classmethod
    def from_config(cls, cfg: dict) -> "OrnamentClassifier":
        """Construct from a loaded YAML config."""
        orn = cfg["ornamentation"]
        sample_period = cfg["data"]["f0_sample_period_s"]
        kan_max_dur_ms = orn["kan_max_duration_ms"]
        base_frame_ms = cfg["adaptive_windowing"]["base_frame_ms"]
        # 1 frame ≈ base_frame_ms
        kan_max_frames = max(1, int(round(kan_max_dur_ms / base_frame_ms)))
        return cls(
            instability_threshold=orn["instability_threshold"],
            min_merge_gap=orn["min_merge_gap_samples"],
            kan_max_frames=kan_max_frames,
            andolan_max_pitches=orn["andolan_max_unique_pitches"],
        )

    def _find_unstable_regions(
        self, instability: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Identify contiguous unstable regions in the instability curve.

        Returns list of (start_idx, end_idx) pairs.
        """
        above = instability > self.instability_threshold
        # Handle NaN from moving average
        above = np.where(np.isnan(instability), False, above)

        regions: List[Tuple[int, int]] = []
        in_region = False
        start = 0
        for i in range(len(above)):
            if above[i] and not in_region:
                start = i
                in_region = True
            elif not above[i] and in_region:
                regions.append((start, i))
                in_region = False
        if in_region:
            regions.append((start, len(above)))

        # Merge close regions
        merged: List[Tuple[int, int]] = []
        for r in regions:
            if merged and (r[0] - merged[-1][1]) <= self.min_merge_gap:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)

        return merged

    @staticmethod
    def _count_direction_changes(segment: np.ndarray) -> int:
        """Count local extrema (direction changes) in a segment."""
        if len(segment) < 3:
            return 0
        diff = np.diff(segment)
        # Remove zeros
        diff = diff[diff != 0]
        if len(diff) < 2:
            return 0
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
        return int(sign_changes)

    @staticmethod
    def _count_unique_pitches(
        quantized_segment: np.ndarray,
    ) -> int:
        """Count unique pitch values in a quantized segment, ignoring zeros."""
        voiced = quantized_segment[quantized_segment != 0]
        if len(voiced) == 0:
            return 0
        return len(np.unique(np.round(voiced, decimals=4)))

    def classify(
        self,
        instability: np.ndarray,
        quantized_hz: np.ndarray,
        times: np.ndarray,
        hop_samples: int = 5,
    ) -> List[OrnamentEvent]:
        """
        Classify ornaments from the instability curve and quantized f0.

        Parameters
        ----------
        instability : np.ndarray
            Instability curve (from a windower).
        quantized_hz : np.ndarray
            Full quantized f0 for the phrase.
        times : np.ndarray
            Timestamps for the quantized f0.
        hop_samples : int
            Hop size used for instability computation (maps frame index
            back to sample index).

        Returns
        -------
        List[OrnamentEvent]
        """
        if len(instability) == 0:
            return []

        regions = self._find_unstable_regions(instability)
        events: List[OrnamentEvent] = []

        for start_frame, end_frame in regions:
            # Map frame indices back to sample indices
            sample_start = start_frame * hop_samples
            sample_end = min(end_frame * hop_samples, len(quantized_hz))
            if sample_end <= sample_start:
                continue

            segment = quantized_hz[sample_start:sample_end]
            n_frames = end_frame - start_frame

            dir_changes = self._count_direction_changes(segment)
            unique_p = self._count_unique_pitches(segment)

            # Decision tree (Figure 5)
            if dir_changes <= 2:
                if n_frames <= self.kan_max_frames:
                    label = "kan"
                else:
                    label = "meend"
            else:
                if unique_p <= self.andolan_max_pitches:
                    label = "andolan"
                else:
                    label = "murki"

            t_start = times[sample_start] if sample_start < len(times) else times[-1]
            t_end = times[min(sample_end - 1, len(times) - 1)]

            events.append(OrnamentEvent(
                start_idx=sample_start,
                end_idx=sample_end,
                start_time=t_start,
                end_time=t_end,
                label=label,
                direction_changes=dir_changes,
                unique_pitches=unique_p,
            ))

        return events
