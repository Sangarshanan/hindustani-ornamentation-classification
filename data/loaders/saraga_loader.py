"""
Saraga Hindustani dataset loader via compiam.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import compiam
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SaragaTrack:
    """Container for a single Saraga Hindustani track."""
    track_id: str
    f0_times: np.ndarray
    f0_freqs: np.ndarray
    tonic: float
    raga: str
    sections: Optional[List[Dict]] = None
    phrases: Optional[List[Dict]] = None



# Default data_home: repo root where saraga1.5_hindustani/ lives
_DEFAULT_DATA_HOME = str(Path(__file__).resolve().parents[2])


class SaragaHindustaniLoader:
    """
    Loader for the Saraga Hindustani dataset using compiam.

    Parameters
    ----------
    data_home : str or Path or None
        Root directory passed to ``compiam.load_dataset`` as *data_home*.
        If ``None``, uses the default Saraga download location.
    """

    def __init__(self, data_home: str | Path | None = None) -> None:
        self._dataset = compiam.load_dataset(
            "saraga_hindustani", data_home=str(data_home),
        )
        self._tracks = self._dataset.load_tracks()

    # ------------------------------------------------------------------
    # Track discovery
    # ------------------------------------------------------------------

    def get_all_track_ids(self) -> List[str]:
        """Return all track IDs available in the dataset."""
        return sorted(self._tracks.keys())

    def get_annotated_tracks(self) -> List[str]:
        """Return track IDs that have phrase (swara) ground-truth annotations."""
        annotated = []
        for tid, ct in self._tracks.items():
            phrases_path = getattr(ct, "phrases_path", None)
            if phrases_path and Path(phrases_path).exists():
                annotated.append(tid)
        return sorted(annotated)

    # ------------------------------------------------------------------
    # Track loading
    # ------------------------------------------------------------------

    def load_track(self, track_id: str) -> SaragaTrack:
        """
        Load a single track via compiam and return a :class:`SaragaTrack`.

        Parameters
        ----------
        track_id : str
            A valid track identifier from the dataset.

        Returns
        -------
        SaragaTrack
        """
        ct = self._tracks[track_id]

        # --- f0 pitch contour ---
        pitch_path = getattr(ct, "pitch_path", None)
        if pitch_path and Path(pitch_path).exists():
            data = np.loadtxt(pitch_path, delimiter="\t")
            f0_times = data[:, 0].astype(np.float64)
            f0_freqs = data[:, 1].astype(np.float64)
        else:
            f0_times = np.array([], dtype=np.float64)
            f0_freqs = np.array([], dtype=np.float64)

        # --- tonic ---
        tonic = 0.0
        ctonic_path = getattr(ct, "ctonic_path", None)
        if ctonic_path and Path(ctonic_path).exists():
            tonic = float(Path(ctonic_path).read_text().strip())

        # --- raga from metadata ---
        raga = ""
        metadata = getattr(ct, "metadata", None)
        if metadata and isinstance(metadata, dict):
            raaga_list = metadata.get("raags", metadata.get("raaga", []))
            if raaga_list and isinstance(raaga_list, list):
                raw = raaga_list[0]
                if isinstance(raw, dict):
                    raw = raw.get("common_name", raw.get("name", ""))
                raga = str(raw).lower().strip()

        # --- sections ---
        sections = self._load_sections(ct)

        # --- phrases (ground-truth swaras) ---
        phrases = self._load_phrases(ct)

        return SaragaTrack(
            track_id=track_id,
            f0_times=f0_times,
            f0_freqs=f0_freqs,
            tonic=tonic,
            raga=raga,
            sections=sections,
            phrases=phrases,
        )

    # ------------------------------------------------------------------
    # Section / phrase parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_sections(ct) -> Optional[List[Dict]]:
        """Extract sections from a compiam track object."""
        sections_path = getattr(ct, "sections_path", None)
        if not sections_path or not Path(sections_path).exists():
            return None
        sections: List[Dict] = []
        for line in Path(sections_path).read_text().strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            start = float(parts[0])
            duration = float(parts[2])
            label = parts[3]
            sections.append({"start": start, "end": start + duration, "label": label})
        return sections or None

    @staticmethod
    def _load_phrases(ct) -> Optional[List[Dict]]:
        """Extract phrase annotations from a compiam track object."""
        phrases_path = getattr(ct, "phrases_path", None)
        if not phrases_path or not Path(phrases_path).exists():
            return None
        phrases: List[Dict] = []
        for line in Path(phrases_path).read_text().strip().splitlines():
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            start = float(parts[0])
            duration = float(parts[2])
            swaras = parts[3].strip()
            phrases.append({
                "start": start,
                "end": start + duration,
                "swaras": swaras,
            })
        return phrases or None

    @staticmethod
    def _load_tempos(ct) -> Optional[List[Dict]]:
        """Extract tempo regions from a compiam track object."""
        tempo_path = getattr(ct, "tempo_path", None)
        if not tempo_path or not Path(tempo_path).exists():
            return None
        regions: List[Dict] = []
        for line in Path(tempo_path).read_text().strip().splitlines():
            raw = line.strip()
            if not raw:
                continue
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < 3:
                parts = [p.strip() for p in raw.split("\t")]
            if len(parts) < 3:
                continue
            label = parts[0]
            try:
                start = float(parts[1])
                end = float(parts[2])
            except ValueError:
                continue
            bpm = None
            try:
                bpm_val = float(label)
                if bpm_val > 0:
                    bpm = bpm_val
            except ValueError:
                bpm = None
            regions.append({"start": start, "end": end, "label": label, "bpm": bpm})
        return regions or None

    # ------------------------------------------------------------------
    # Section overrides for adaptive windower
    # ------------------------------------------------------------------

    def get_section_overrides(self, track: SaragaTrack) -> Dict:
        """
        Build section override dict for the AdaptiveWindower.

        Returns
        -------
        dict
            ``{(start_s, end_s): label}`` suitable for
            ``AdaptiveWindower.section_overrides``.
        """
        overrides: Dict = {}
        if track.sections:
            for sec in track.sections:
                overrides[(sec["start"], sec["end"])] = sec["label"]
        return overrides
