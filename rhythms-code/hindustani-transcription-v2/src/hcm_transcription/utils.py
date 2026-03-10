"""
Utility functions: raga dictionaries, just-intonation grid, and helpers.

Reference: Jain & Arthur (2023) DLfM. DOI: 10.1145/3625135.3625137
Saraga dataset: Srinivasamurthy et al. (2021) EMR. DOI: 10.18061/emr.v16i1.7492
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Just-intonation ratios for a single octave (Table 1 of the paper)
# ---------------------------------------------------------------------------
SWARA_RATIO_MAP: Dict[str, float] = {
    "Sa":  1.0,
    "re":  16 / 15,
    "Re":  9 / 8,
    "ga":  6 / 5,
    "Ga":  5 / 4,
    "ma":  4 / 3,
    "Ma":  45 / 32,
    "Pa":  3 / 2,
    "dha": 8 / 5,
    "Dha": 27 / 16,
    "ni":  9 / 5,
    "Ni":  15 / 8,
}

# 12-note chromatic order within one octave, using the paper's notation
CHROMATIC_ORDER: List[str] = [
    "S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"
]

# Full 3-octave swara name array matching the original code
SWARA_NAMES_3OCT: np.ndarray = np.array([
    "S2", "r2", "R2", "g2", "G2", "m2", "M2", "P2", "d2", "D2", "n2", "N2",
    "S3", "r3", "R3", "g3", "G3", "m3", "M3", "P3", "d3", "D3", "n3", "N3",
    "S4", "r4", "R4", "g4", "G4", "m4", "M4", "P4", "d4", "D4", "n4", "N4",
])

# Single-octave ratio array
_SINGLE_OCTAVE_RATIOS = np.array([
    1.0, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 27/16, 9/5, 15/8
])

# 3-octave ratio array: octave 2 (0.5x), octave 3 (1x), octave 4 (2x)
RATIO_3OCT: np.ndarray = np.concatenate([
    _SINGLE_OCTAVE_RATIOS * 0.5,
    _SINGLE_OCTAVE_RATIOS * 1.0,
    _SINGLE_OCTAVE_RATIOS * 2.0,
])


def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raga_dict(json_path: str | Path) -> dict:
    """Load the raga dictionary JSON file."""
    with open(json_path) as f:
        return json.load(f)


def get_allowed_swara_mask(raga_name: str, raga_dict: dict) -> np.ndarray:
    """
    Return boolean mask of allowed swaras for a given raga.

    Parameters
    ----------
    raga_name : str
        Key in the raga dictionary (e.g. ``"lalit"``).
    raga_dict : dict
        Loaded raga dictionary from ``data/raga_dict.json``.

    Returns
    -------
    np.ndarray
        Boolean array of length 36, True where the swara is allowed.
    """
    notes = raga_dict["ragas"][raga_name]
    return np.array(notes, dtype=bool)


def get_raga_frequencies(tonic_hz: float, raga_name: str,
                         raga_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the just-intonation frequencies for allowed swaras of a raga.

    Parameters
    ----------
    tonic_hz : float
        Frequency of the singer's Sa (tonic) in Hz.
    raga_name : str
        Raga name key in the dictionary.
    raga_dict : dict
        Full raga dictionary.

    Returns
    -------
    freqs_hz : np.ndarray
        Frequencies in Hz for each allowed swara.
    log_freqs : np.ndarray
        log2 of the frequencies (for distance computation).
    swara_labels : np.ndarray
        String labels for each allowed swara.
    """
    mask = get_allowed_swara_mask(raga_name, raga_dict)
    all_freqs = RATIO_3OCT * tonic_hz
    freqs_hz = all_freqs[mask]
    log_freqs = np.log2(freqs_hz)
    swara_labels = SWARA_NAMES_3OCT[mask]
    return freqs_hz, log_freqs, swara_labels


def hz_to_cents(f: np.ndarray, ref: float) -> np.ndarray:
    """Convert Hz to cents relative to a reference frequency."""
    with np.errstate(divide="ignore", invalid="ignore"):
        c = 1200.0 * np.log2(f / ref)
    c[~np.isfinite(c)] = 0.0
    return c


def cents_to_ratio(cents: float) -> float:
    """Convert a cent value to a frequency ratio."""
    return 2.0 ** (cents / 1200.0)
