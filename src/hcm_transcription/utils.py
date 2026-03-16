"""
Utility functions: raga dictionaries, just-intonation grid, and helpers.

Reference: Jain & Arthur (2023) DLfM. DOI: 10.1145/3625135.3625137
Saraga dataset: Srinivasamurthy et al. (2021) EMR. DOI: 10.18061/emr.v16i1.7492
"""

from __future__ import annotations

from collections import Counter
import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

import compiam
from hcm_transcription.mapping import MAPPING


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


# Ornamentation annotation helpers
ORNAMENT_MAPPING = {
    # Standard abbreviations
    "k": "Kan",
    "g": "Gamak",
    "mu": "Murki",
    "me": "Meend",
    "a": "Andolan",
    "kh": "Khatka",
    "z": "Zamzama",
    "o": "Other",
    # Alternate abbreviations found in some annotations
    "m": "Meend",
    "kan": "Kan",
    "gamak": "Gamak",
    "murki": "Murki",
    "meend": "Meend",
    "andolan": "Andolan",
    "khatka": "Khatka",
    "zamzama": "Zamzama",
    # Other ornament types
    "thaan": "Thaan",
}

# What we will predict in the end
TARGET_LABELS = {
    "Kan", "Meend", "Murki", "Andolan"
}

# Collapse the paper's 7 annotated classes into the 4 prediction classes.
TARGET_LABEL_GROUPS = {
    "Kan": "Kan",
    "Khatka": "Kan",
    "Meend": "Meend",
    "Murki": "Murki",
    "Gamak": "Murki",
    "Zamzama": "Murki",
    "Andolan": "Andolan",
}

TARGET_LABEL_ALIASES = {
    "k": "Kan",
    "k_": "Kan",
    "kan": "Kan",
    "khatka": "Kan",
    "m": "Meend",
    "me": "Meend",
    "meend": "Meend",
    "mu": "Murki",
    "murki": "Murki",
    "g": "Murki",
    "gamak": "Murki",
    "z": "Murki",
    "zamzama": "Murki",
    "thaan": "Murki",
    "taan": "Murki",
    "than": "Murki",
    "a": "Andolan",
    "andolan": "Andolan",
    "kampan": "Andolan",
    "kampit": "Andolan",
    "vibrato": "Andolan",
    "soft_gamak": "Murki",
    "multiple_karn": "Kan",
}

# Based on label distribution.
TARGET_LABEL_PRIORITY = {
    "Andolan": 0,
    "Murki": 1,
    "Meend": 2,
    "Kan": 3,
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def normalize_target_label(label: str) -> str | None:
    """Map fine-grained or composite ornament labels to reduced target classes."""
    cleaned_label = label.strip()
    if not cleaned_label:
        return None

    direct_match = TARGET_LABEL_GROUPS.get(cleaned_label)
    if direct_match is not None:
        return direct_match

    normalized_parts = []
    for part in cleaned_label.split("+"):
        token = part.strip().strip("_")
        if not token:
            continue

        lowered_token = token.lower()

        mapped_label = TARGET_LABEL_GROUPS.get(token)
        if mapped_label is None:
            mapped_label = TARGET_LABEL_ALIASES.get(lowered_token)

        if mapped_label is None and lowered_token.startswith("o_"):
            mapped_label = TARGET_LABEL_ALIASES.get(lowered_token[2:])

        if mapped_label is not None:
            normalized_parts.append(mapped_label)

    if not normalized_parts:
        return None

    label_counts = Counter(normalized_parts)
    return min(
        label_counts.items(),
        key=lambda item: (-item[1], TARGET_LABEL_PRIORITY[item[0]]),
    )[0]


def df_to_anno(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Parse an annotation CSV into a structured DataFrame with start/end times.
    
    Uses a stack-based approach to match _s (start) markers with their corresponding
    _e (end) markers, handling nested/overlapping annotations and data quality issues.
    
    Args:
        df: DataFrame with columns [time, label]
        verbose: If True, print warnings for unexpected patterns
    
    Returns:
        DataFrame with columns [time_s, time_e, label, duration]
    """
    df = df.copy()
    df.columns = ["time", "label"]
    
    # Filter out invalid labels (none, empty, Q, NaN)
    df = df[~df["label"].isin(["none", "", "Q"])]
    df = df[df["label"].notna()]
    df = df[df["label"].apply(lambda x: str(x).lower() != "nan")]
    df = df.reset_index(drop=True)
    
    time_s, time_e, labels = [], [], []
    open_ornaments = {}  # base_label -> (start_time, original_label)
    
    for idx, row in df.iterrows():
        label = str(row["label"]).strip()
        time = row["time"]
        
        # Skip non-ornament markers
        if not (label.endswith("_s") or label.endswith("_e")):
            if verbose:
                print(f"Warning: Skipping non-ornament label at index {idx}: {label}")
            continue
        
        if label.endswith("_s"):
            # Start marker - push onto stack
            base_label = label[:-2]
            if base_label in open_ornaments and verbose:
                print(f"Warning: Duplicate start at index {idx}: {label} (previous not closed)")
            open_ornaments[base_label] = (time, label)
            
        elif label.endswith("_e"):
            # End marker - try to match with corresponding start
            base_label = label[:-2]
            
            if base_label in open_ornaments:
                start_time, start_label = open_ornaments.pop(base_label)
                time_s.append(start_time)
                time_e.append(time)
                
                # Map ornament label
                if base_label.startswith("c_"):
                    parts = base_label.split("_")[1:]
                    mapped_parts = [ORNAMENT_MAPPING.get(p, p) for p in parts]
                    labels.append(" + ".join(mapped_parts))
                else:
                    labels.append(ORNAMENT_MAPPING.get(base_label, base_label))
            else:
                if verbose:
                    print(f"Warning: Orphaned end marker at index {idx}: {label}")
    
    # Warn about unclosed ornaments
    if verbose and open_ornaments:
        for base_label, (start_time, start_label) in open_ornaments.items():
            print(f"Warning: Unclosed ornament: {start_label} at time {start_time:.2f}s")
    
    anno = pd.DataFrame({"time_s": time_s, "time_e": time_e, "label": labels})
    anno["duration"] = anno["time_e"] - anno["time_s"]
    return anno


def fetch_ornamentations(raga_name: str = "Aahir Bhairon", num_to_show: int = 5):
    """Load a Saraga Hindustani track and display ornamentation segments."""
    import librosa
    import matplotlib.pyplot as plt
    from IPython.display import Audio, display

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


def extract_ornament_segments(raga_names, data_home, annotation_base="Ornamentation-In-Hindustani-Vocals-Dataset"):

    segments = []
    skipped_labels = set()

    for raga_name in raga_names:
        if raga_name not in MAPPING:
            print(f"Skipping {raga_name} - not in MAPPING")
            continue

        data = MAPPING[raga_name]
        annotation_file = f"{annotation_base}/{list(data['annotators'].values())[0]}"

        # Build pitch path directly from audio_path — no compiam needed
        pitch_path = f"{data_home}/saraga1.5_hindustani/{data['audio_path']}.pitch.txt"

        print(f"Loading {raga_name}...")
        print(f"  Pitch: {pitch_path}")

        if not os.path.exists(pitch_path):
            print(f"  WARNING: pitch file not found, skipping")
            continue

        if not os.path.exists(annotation_file):
            print(f"  WARNING: annotation file not found, skipping")
            continue
        # Load pitch data
        df_pitch = pd.read_csv(pitch_path, sep="\t", header=None)
        df_pitch.columns = ["time", "f0"]
        df_pitch["log_f0"] = df_pitch["f0"].apply(lambda x: np.log2(x) if x > 0 else np.nan)

        # Load annotations
        df_raw = pd.read_csv(annotation_file, header=None)
        anno = df_to_anno(df_raw)
        skipped_labels.update(
            label for label in anno["label"].dropna().unique()
            if normalize_target_label(label) is None
        )

        # Collapse fine-grained annotations into the reduced target classes
        # used by the model, following the paper.
        anno = anno.assign(label=anno["label"].map(normalize_target_label))

        # Filter to target labels only
        anno_filtered = anno[anno['label'].isin(TARGET_LABELS)].reset_index(drop=True)
        print(f"  Found {len(anno_filtered)} target ornaments out of {len(anno)} total")

        for _, row in anno_filtered.iterrows():
            mask = (df_pitch["time"] >= row['time_s']) & (df_pitch["time"] <= row['time_e'])
            segment_pitch = df_pitch[mask]["log_f0"].values

            if len(segment_pitch) == 0 or np.isnan(segment_pitch).mean() > 0.3:
                continue

            segment_pitch = pd.Series(segment_pitch).interpolate().fillna(0).values

            segments.append({
                "raga": raga_name,
                "label": row['label'],
                "time_s": row['time_s'],
                "time_e": row['time_e'],
                "duration": row['duration'],
                "pitch_curve": segment_pitch
            })

    # Summary
    df_segments = pd.DataFrame([{k: v for k, v in s.items() if k != "pitch_curve"} 
                                  for s in segments])
    print(f"\n=== Extraction Summary ===")
    print(f"Total segments: {len(segments)}")
    print(f"Label distribution:\n{df_segments['label'].value_counts()}")
    print(f"Duration stats:\n{df_segments['duration'].describe()}")
    print(f"Skipped ornamentations: {sorted(skipped_labels)}")

    return segments