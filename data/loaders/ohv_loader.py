"""
Loader for the Ornamentation in Hindustani Vocals (OHV) dataset.

The OHV dataset contains expert-annotated ornament boundaries for
Hindustani vocal renditions.

Dataset: Jain & Arthur (2023) DLfM. DOI: 10.1145/3625135.3625137

CSV format: each row is ``time, label`` where labels follow the OHV scheme
(``k_s``/``k_e``, ``me_s``/``me_e``, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

from hcm_transcription.ornamentation import LabelMapper

logger = logging.getLogger(__name__)


@dataclass
class OrnamentAnnotation:
    """A single ornament event from the OHV dataset."""
    onset: float
    offset: float
    label: str  # canonical name (kan, meend, andolan, murki, gamak, ...)


@dataclass
class OHVTrack:
    """Container for a single OHV annotated track."""
    track_id: str
    file_path: str
    annotations: List[OrnamentAnnotation] = field(default_factory=list)

    @property
    def boundaries(self) -> np.ndarray:
        """Return all onset and offset times as a flat sorted array."""
        times = []
        for a in self.annotations:
            times.extend([a.onset, a.offset])
        return np.array(sorted(set(times)))

    @property
    def labels(self) -> List[str]:
        """Return list of canonical labels."""
        return [a.label for a in self.annotations]


class OHVLoader:
    """
    Loader for the Ornamentation in Hindustani Vocals (OHV) dataset.

    Parameters
    ----------
    data_path : str or Path
        Root directory containing the OHV CSV files.
    """

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)

    def get_track_ids(self) -> List[str]:
        """Return track IDs (CSV file stems) in the dataset."""
        return sorted([
            p.stem for p in self.data_path.glob("*.csv")
        ])

    def load_track(self, track_id: str) -> OHVTrack:
        """
        Load a single OHV track by ID.

        Parameters
        ----------
        track_id : str
            File stem of the CSV file.

        Returns
        -------
        OHVTrack
        """
        csv_path = self.data_path / f"{track_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"OHV file not found: {csv_path}")

        df = pd.read_csv(csv_path, header=None, names=["time", "label"])
        annotations = self._parse_annotations(df)

        return OHVTrack(
            track_id=track_id,
            file_path=str(csv_path),
            annotations=annotations,
        )

    def iter_tracks(self) -> Iterator[OHVTrack]:
        """Iterate over all tracks in the dataset."""
        for tid in self.get_track_ids():
            yield self.load_track(tid)

    @staticmethod
    def _parse_annotations(df: pd.DataFrame) -> List[OrnamentAnnotation]:
        """
        Parse paired start/end labels from the OHV CSV into events.

        The CSV has rows like:
            time, me_s
            time, me_e
        Pair them into (onset, offset, canonical_label).
        """
        mapper = LabelMapper()
        annotations: List[OrnamentAnnotation] = []
        open_events: dict = {}  # canonical -> onset time

        for _, row in df.iterrows():
            if pd.isna(row["time"]) or pd.isna(row["label"]):
                continue
            time_val = float(row["time"])
            raw_label = str(row["label"]).strip()

            canonical, event_type = mapper.parse_ohv_label(raw_label)
            if canonical == "none" or canonical == "unknown":
                continue

            if event_type == "start":
                open_events[canonical] = time_val
            elif event_type == "end":
                if canonical in open_events:
                    annotations.append(OrnamentAnnotation(
                        onset=open_events.pop(canonical),
                        offset=time_val,
                        label=canonical,
                    ))
                else:
                    annotations.append(OrnamentAnnotation(
                        onset=time_val,
                        offset=time_val,
                        label=canonical,
                    ))
            elif event_type == "point":
                annotations.append(OrnamentAnnotation(
                    onset=time_val,
                    offset=time_val,
                    label=canonical,
                ))

        # Warn for unclosed events
        for label, onset in open_events.items():
            logger.warning(
                "Unclosed start for '%s' at time %.3f", label, onset
            )

        return annotations
