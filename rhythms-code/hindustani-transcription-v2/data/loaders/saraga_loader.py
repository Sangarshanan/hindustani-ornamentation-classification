"""
Saraga Hindustani dataset loader (local file-system).

Reads f0 contours, tonic, raga metadata, section boundaries, and
expert-annotated swara phrases from the Saraga Hindustani f0-annotations
directory structure.

Expected layout per track::

    <data_home>/<TrackFolder>/
        <TrackFolder>.pitch.txt          # tab: time  f0_hz
        <TrackFolder>.ctonic.txt         # single float (tonic Hz)
        <TrackFolder>.json               # metadata incl. raga
        <TrackFolder>.sections-manual-p.txt   # comma: start,num,dur,label
        <TrackFolder>.mphrases-manual.txt     # tab: start  num  dur  swaras

Reference:
  Srinivasamurthy et al. (2021) EMR. DOI: 10.18061/emr.v16i1.7492
  Melodia f0: Salamon & Gómez (2012) IEEE TASLP. DOI: 10.1109/TASL.2012.2188515
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Mapping from Saraga metadata raga names to raga_dict keys
_RAGA_NAME_ALIASES: Dict[str, str] = {
    "lalat": "lalit",
    "lalit": "lalit",
}


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


class SaragaHindustaniLoader:
    """
    Loader for the Saraga Hindustani dataset from local annotation files.

    Parameters
    ----------
    data_home : str or Path
        Root directory containing track sub-folders with annotation files.
    """

    def __init__(self, data_home: str | Path) -> None:
        self.data_home = Path(data_home).expanduser().resolve()
        if not self.data_home.exists():
            raise FileNotFoundError(f"Saraga data directory not found: {self.data_home}")

    # ------------------------------------------------------------------
    # Track discovery
    # ------------------------------------------------------------------

    def get_all_track_ids(self) -> List[str]:
        """Return all track IDs (subdirectory names) in the dataset."""
        return sorted([
            p.name for p in self.data_home.iterdir()
            if p.is_dir() and (p / f"{p.name}.pitch.txt").exists()
        ])

    def get_annotated_tracks(self) -> List[str]:
        """Return track IDs that have phrase (swara) ground-truth annotations."""
        annotated = []
        for tid in self.get_all_track_ids():
            phrases_path = self.data_home / tid / f"{tid}.mphrases-manual.txt"
            if phrases_path.exists():
                annotated.append(tid)
        return annotated

    # ------------------------------------------------------------------
    # Track loading
    # ------------------------------------------------------------------

    def load_track(self, track_id: str) -> SaragaTrack:
        """
        Load a single track from local files.

        Parameters
        ----------
        track_id : str
            Subdirectory name under ``data_home``.

        Returns
        -------
        SaragaTrack
        """
        track_dir = self.data_home / track_id

        # --- f0 pitch contour ---
        pitch_path = track_dir / f"{track_id}.pitch.txt"
        if pitch_path.exists():
            data = np.loadtxt(pitch_path, delimiter="\t")
            f0_times = data[:, 0].astype(np.float64)
            f0_freqs = data[:, 1].astype(np.float64)
        else:
            f0_times = np.array([], dtype=np.float64)
            f0_freqs = np.array([], dtype=np.float64)

        # --- tonic ---
        ctonic_path = track_dir / f"{track_id}.ctonic.txt"
        tonic = 0.0
        if ctonic_path.exists():
            tonic = float(ctonic_path.read_text().strip())

        # --- raga from JSON metadata ---
        raga = ""
        json_path = track_dir / f"{track_id}.json"
        if json_path.exists():
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            # Saraga uses "raags" (plural) as the key
            raaga_list = meta.get("raags", meta.get("raaga", []))
            if raaga_list and isinstance(raaga_list, list):
                raw = raaga_list[0]
                if isinstance(raw, dict):
                    raw = raw.get("common_name", raw.get("name", ""))
                raga = str(raw).lower().strip()
        # Normalise known aliases
        raga = _RAGA_NAME_ALIASES.get(raga, raga)

        # --- sections ---
        sections = self._load_sections(track_dir / f"{track_id}.sections-manual-p.txt")

        # --- phrases (ground-truth swaras) ---
        phrases = self._load_phrases(track_dir / f"{track_id}.mphrases-manual.txt")

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
    def _load_sections(path: Path) -> Optional[List[Dict]]:
        """Parse ``sections-manual-p.txt`` (comma-separated: start,num,dur,label)."""
        if not path.exists():
            return None
        sections: List[Dict] = []
        for line in path.read_text().strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            start = float(parts[0])
            duration = float(parts[2])
            label = parts[3]
            sections.append({"start": start, "end": start + duration, "label": label})
        return sections or None

    @staticmethod
    def _load_phrases(path: Path) -> Optional[List[Dict]]:
        """Parse ``mphrases-manual.txt`` (tab-separated: start, num, dur, swaras)."""
        if not path.exists():
            return None
        phrases: List[Dict] = []
        for line in path.read_text().strip().splitlines():
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
