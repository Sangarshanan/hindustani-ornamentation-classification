"""
Utility functions: raga dictionaries, just-intonation grid, and helpers.

Reference: Jain & Arthur (2023) DLfM. DOI: 10.1145/3625135.3625137
Saraga dataset: Srinivasamurthy et al. (2021) EMR. DOI: 10.18061/emr.v16i1.7492
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# Ornamentation annotation helpers
# ---------------------------------------------------------------------------
ORNAMENT_MAPPING = {
    "k": "Kan",
    "g": "Gamak",
    "mu": "Murki",
    "me": "Meend",
    "a": "Andolan",
    "kh": "Khatka",
    "z": "Zamzama",
    "o": "Other",
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def df_to_anno(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse an annotation CSV into a structured DataFrame with start/end times.
    """
    all_index = []
    df.columns = ["time", "label"]

    for index, label in enumerate(df.label):
        if label == "none" or label == "" or label == "Q":
            all_index.append(index)

    df_new = df.drop(all_index).reset_index(drop=True)

    time_s, time_e, labels = [], [], []

    for i in range(0, len(df_new) - 1, 2):
        s, e = i, i + 1
        label_s = str(df_new["label"].iloc[s])
        label_e = str(df_new["label"].iloc[e])

        if label_s.endswith("_s") and label_e.endswith("_e"):
            time_s.append(df_new["time"].iloc[s])
            time_e.append(df_new["time"].iloc[e])

            raw_label = label_s[:-2]
            if raw_label.startswith("c_"):
                parts = raw_label.split("_")[1:]
                mapped_parts = [ORNAMENT_MAPPING.get(p, p) for p in parts]
                labels.append(" + ".join(mapped_parts))
            else:
                labels.append(ORNAMENT_MAPPING.get(raw_label, raw_label))
        else:
            print(
                f"Warning: Unexpected pair at index {i}: {label_s}, {label_e}"
            )

    anno = pd.DataFrame({"time_s": time_s, "time_e": time_e, "label": labels})
    anno["duration"] = anno["time_e"] - anno["time_s"]
    return anno


def fetch_ornamentations(raga_name: str = "Aahir Bhairon", num_to_show: int = 5):
    """Load a Saraga Hindustani track and display ornamentation segments."""
    import compiam
    import librosa
    import matplotlib.pyplot as plt
    from IPython.display import Audio, display
    from hcm_transcription.mapping import MAPPING

    if raga_name not in MAPPING:
        print(f"Raga '{raga_name}' not found in mapping.py")
        return

    data = MAPPING[raga_name]
    track_id = data["track_id"]
    annotation_file = _PROJECT_ROOT / "Ornamentation-In-Hindustani-Vocals-Dataset" / list(data["annotators"].values())[0]

    data_home = str(_PROJECT_ROOT)

    print(f"Loading track {track_id}...")
    saraga_hindustani = compiam.load_dataset("saraga_hindustani", data_home=data_home)
    st = saraga_hindustani.load_tracks()
    track = st[track_id]
    audio_path = track.audio_path
    pitch_path = track.pitch_path

    df_raw = pd.read_csv(annotation_file, header=None)
    anno = df_to_anno(df_raw)

    df_pitch = pd.read_csv(pitch_path, sep="\t", header=None)
    df_pitch.columns = ["time", "f0"]
    df_pitch["log_f0"] = df_pitch["f0"].apply(
        lambda x: np.log2(x) if x > 0 else np.nan
    )

    y, sr = librosa.load(audio_path, sr=None)

    print(f"\nShowing first {num_to_show} ornamentations:\n")
    for i in range(min(num_to_show, len(anno))):
        row = anno.iloc[i]
        start_time = row["time_s"]
        end_time = row["time_e"]
        label = row["label"]

        plot_start = max(0, start_time - 0.5)
        plot_end = min(len(y) / sr, end_time + 0.5)

        print(
            f"{i+1}: {label} | Start: {start_time:.2f}s | End: {end_time:.2f}s | Duration: {row['duration']:.2f}s"
        )

        segment_pitch = df_pitch[
            (df_pitch["time"] >= plot_start) & (df_pitch["time"] <= plot_end)
        ]

        plt.figure(figsize=(12, 4))
        plt.plot(
            segment_pitch["time"],
            segment_pitch["log_f0"],
            color="blue",
            marker="o",
            markersize=2,
            linestyle="",
        )
        plt.axvspan(start_time, end_time, color="green", alpha=0.2, label="Ornamentation")
        plt.ylabel("Log2(F0)")
        plt.xlabel("Time (s)")
        plt.title(f"Pitch Contour - {label}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

        orn_start_sample = int(start_time * sr)
        orn_end_sample = int(end_time * sr)
        y_orn = y[orn_start_sample:orn_end_sample]

        display(Audio(y_orn, rate=sr))
        print("-" * 80)
