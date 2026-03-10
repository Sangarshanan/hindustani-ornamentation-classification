"""
Humdrum **bhatk formatter.

Exports transcribed swara sequences in the Humdrum **bhatk representation
for Hindustani music notation.

Reference: https://www.humdrum.org/Humdrum/representations/bhatk.rep.html
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# Mapping from internal swara labels to **bhatk token names.
# Octave 2 = lower (mandra), 3 = middle (madhya), 4 = upper (taar).
_SWARA_TO_BHATK = {
    "S": "Sa", "r": "re", "R": "Re", "g": "ga", "G": "Ga",
    "m": "ma", "M": "Ma", "P": "Pa", "d": "dha", "D": "Dha",
    "n": "ni", "N": "Ni",
}

_OCTAVE_PREFIX = {
    "2": "-",   # mandra (lower) → prefix dash
    "3": "",    # madhya (middle) → no prefix
    "4": "+",   # taar (upper)   → prefix plus
}


@dataclass
class BhatkNote:
    """A single note in **bhatk format."""
    token: str          # e.g. "Sa", "-Pa", "+Re"
    duration: float     # in seconds
    ornament: Optional[str] = None  # e.g. "kan", "meend"


def swara_label_to_bhatk(label: str) -> str:
    """
    Convert an internal swara label (e.g. ``"P3"``, ``"g2"``) to **bhatk token.

    Parameters
    ----------
    label : str
        Internal label with letter + octave digit.

    Returns
    -------
    str
        Bhatk-formatted token (e.g. ``"Pa"``, ``"-ga"``).
    """
    if label == "REST" or not label:
        return "."
    note_char = label[0]
    octave = label[1] if len(label) > 1 else "3"
    bhatk_name = _SWARA_TO_BHATK.get(note_char, note_char)
    prefix = _OCTAVE_PREFIX.get(octave, "")
    return prefix + bhatk_name


def format_bhatk_spine(
    notes: List[BhatkNote],
    title: Optional[str] = None,
    raga: Optional[str] = None,
) -> str:
    """
    Generate a complete **bhatk Humdrum file.

    Parameters
    ----------
    notes : List[BhatkNote]
        Sequence of transcribed notes.
    title : str, optional
        Track title for the reference record.
    raga : str, optional
        Raga name for the reference record.

    Returns
    -------
    str
        Full Humdrum-formatted string.
    """
    lines: List[str] = []

    # Header
    lines.append("**bhatk")
    if title:
        lines.append(f"!!!OTL: {title}")
    if raga:
        lines.append(f"!!!ARA: {raga}")

    # Data records
    for note in notes:
        token = note.token
        if note.ornament:
            token = f"{token}~{note.ornament}"
        lines.append(token)

    # Spine terminator
    lines.append("*-")
    return "\n".join(lines) + "\n"


def parse_bhatk_spine(text: str) -> List[BhatkNote]:
    """
    Parse a **bhatk Humdrum file back into a list of notes.

    Parameters
    ----------
    text : str
        Humdrum-formatted text.

    Returns
    -------
    List[BhatkNote]
    """
    notes: List[BhatkNote] = []
    for line in text.strip().splitlines():
        line = line.strip()
        # Skip headers, comments, tandem interpretations, and terminators
        if not line or line.startswith("*") or line.startswith("!"):
            continue
        if line == ".":
            notes.append(BhatkNote(token=".", duration=0.0))
            continue
        # Check for ornament annotation
        if "~" in line:
            token, ornament = line.split("~", 1)
            notes.append(BhatkNote(token=token, duration=0.0, ornament=ornament))
        else:
            notes.append(BhatkNote(token=line, duration=0.0))
    return notes


def export_bhatk(
    swara_labels: List[str],
    durations: List[float],
    output_path: str | Path,
    ornaments: Optional[List[Optional[str]]] = None,
    title: Optional[str] = None,
    raga: Optional[str] = None,
) -> Path:
    """
    Export a transcription to a **bhatk Humdrum file.

    Parameters
    ----------
    swara_labels : List[str]
        Sequence of swara labels (e.g. ``["S3", "R3", "G3"]``).
    durations : List[float]
        Duration of each note in seconds.
    output_path : str or Path
        Output file path.
    ornaments : List[Optional[str]], optional
        Ornament label per note (or None).
    title, raga : str, optional
        Metadata for the header.

    Returns
    -------
    Path
        Written file path.
    """
    if ornaments is None:
        ornaments = [None] * len(swara_labels)

    notes: List[BhatkNote] = []
    for label, dur, orn in zip(swara_labels, durations, ornaments):
        token = swara_label_to_bhatk(label)
        notes.append(BhatkNote(token=token, duration=dur, ornament=orn))

    text = format_bhatk_spine(notes, title=title, raga=raga)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out
