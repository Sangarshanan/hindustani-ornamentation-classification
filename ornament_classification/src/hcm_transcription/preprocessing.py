"""
Tonic normalization and pitch quantization.

Implements the pitch-to-swara mapping using a just-intonation grid anchored
at the singer's tonic (Sa), with a configurable shruti threshold for
ambiguous pitches (default 35 cents per Ganguli et al. 2016 ISMIR).

Reference: Jain & Arthur (2023) DLfM, Section 3.2. DOI: 10.1145/3625135.3625137
Threshold: Ganguli et al. (2016) ISMIR.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from hcm_transcription.utils import (
    RATIO_3OCT,
    SWARA_NAMES_3OCT,
    get_allowed_swara_mask,
)


class TonicNormalizer:
    """
    Quantize raw f0 values to swara labels on a 3-octave just-intonation grid.

    Parameters
    ----------
    tonic_hz : float
        Frequency of the singer's Sa (tonic) in Hz.
    raga_name : str
        Name of the raga (key in the raga dictionary).
    raga_dict : dict
        Loaded raga dictionary.
    shruti_threshold_cents : float
        Maximum distance in cents to snap an ambiguous pitch
        to the previous sample's swara (default 35).
    """

    def __init__(
        self,
        tonic_hz: float,
        raga_name: str,
        raga_dict: dict,
        shruti_threshold_cents: float = 35.0,
    ) -> None:
        self.tonic_hz = tonic_hz
        self.raga_name = raga_name
        self.shruti_threshold_cents = shruti_threshold_cents

        mask = get_allowed_swara_mask(raga_name, raga_dict)
        all_freqs = RATIO_3OCT * tonic_hz
        self._allowed_freqs_hz = all_freqs[mask]
        self._allowed_log_freqs = np.log2(self._allowed_freqs_hz)
        self._allowed_swara_labels = SWARA_NAMES_3OCT[mask]

    @property
    def allowed_freqs_hz(self) -> np.ndarray:
        return self._allowed_freqs_hz

    @property
    def log_freqs(self) -> np.ndarray:
        return self._allowed_log_freqs

    @property
    def swara_labels(self) -> np.ndarray:
        return self._allowed_swara_labels

    def quantize(
        self, f0_hz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Map raw f0 values to swara labels.

        Parameters
        ----------
        f0_hz : np.ndarray
            Raw fundamental frequency contour in Hz. Zero or NaN
            values are treated as silence / unvoiced.

        Returns
        -------
        swara_labels : np.ndarray
            String array of swara labels (``"REST"`` for silent frames).
        quantized_hz : np.ndarray
            Array of quantized frequency values (0.0 for rests).
        """
        n = len(f0_hz)
        swara_out = np.empty(n, dtype=object)
        quant_hz = np.zeros(n, dtype=np.float64)

        log_grid = self._allowed_log_freqs
        threshold_ratio = 2.0 ** (self.shruti_threshold_cents / 1200.0)

        prev_label: Optional[str] = None
        prev_idx: Optional[int] = None

        for i in range(n):
            freq = f0_hz[i]
            if freq <= 0 or not np.isfinite(freq):
                swara_out[i] = "REST"
                quant_hz[i] = 0.0
                prev_label = None
                prev_idx = None
                continue

            log_f = np.log2(freq)
            dists = np.abs(log_grid - log_f)
            best_idx = int(np.argmin(dists))
            best_cents = dists[best_idx] * 1200.0

            # If within threshold, assign the closest swara directly
            if best_cents <= self.shruti_threshold_cents:
                swara_out[i] = self._allowed_swara_labels[best_idx]
                quant_hz[i] = self._allowed_freqs_hz[best_idx]
                prev_label = swara_out[i]
                prev_idx = best_idx
            else:
                # Ambiguous pitch: snap to previous swara if available
                if prev_label is not None:
                    swara_out[i] = prev_label
                    quant_hz[i] = self._allowed_freqs_hz[prev_idx]
                else:
                    # No previous swara available; use closest anyway
                    swara_out[i] = self._allowed_swara_labels[best_idx]
                    quant_hz[i] = self._allowed_freqs_hz[best_idx]
                    prev_label = swara_out[i]
                    prev_idx = best_idx

        return swara_out, quant_hz

    @staticmethod
    def stability_filter(
        swara_labels: np.ndarray,
        frame_size: int = 56,
        hop_size: int = 5,
    ) -> np.ndarray:
        """
        Apply windowed stability filter to reduce micro-oscillations.

        Replicates the original algorithm's ``process_phrase()`` logic:
        a sliding window advances by *hop_size* samples; the note is updated
        **only** when every sample in the window carries the same swara label.
        Otherwise the previous stable note is carried forward.

        Parameters
        ----------
        swara_labels : np.ndarray
            Per-sample swara labels (may contain ``"REST"``).
        frame_size : int
            Window length in samples (default 56 ≈ 250 ms).
        hop_size : int
            Hop length in samples (default 5 ≈ 22 ms).

        Returns
        -------
        np.ndarray
            Stabilised swara label array (shorter than input).
        """
        # Filter out REST frames to match original behaviour
        voiced_mask = swara_labels != "REST"
        voiced = swara_labels[voiced_mask]
        n = len(voiced)
        if n < frame_size:
            return voiced.copy()

        frame_count = 1 + (n - frame_size) // hop_size
        stable = np.empty(frame_count, dtype=object)
        stable[0] = voiced[0]

        for i in range(1, frame_count):
            start = i * hop_size
            end = start + frame_size
            frame = voiced[start:end]
            # Update only when all samples in the window agree
            if np.all(frame == frame[0]):
                stable[i] = frame[0]
            else:
                stable[i] = stable[i - 1]

        return stable

    def to_reduced_sequence(
        self, swara_labels: np.ndarray
    ) -> str:
        """
        Collapse consecutive duplicate swaras into a compact string
        (octave-free, matching ground-truth notation).

        Suitable for evaluation against GT swara strings like ``"dS"``, ``"NMd"``.

        Parameters
        ----------
        swara_labels : np.ndarray
            Array from :meth:`quantize`.

        Returns
        -------
        str
            Collapsed swara string using single-character note names
            (e.g. ``"SRGmP"``), with ``"REST"`` segments removed.
        """
        parts: list[str] = []
        prev_char: str | None = None
        for s in swara_labels:
            if s == "REST":
                prev_char = None
                continue
            note_char = str(s)[0] if s else ""
            if note_char != prev_char:
                parts.append(note_char)
                prev_char = note_char
        return "".join(parts)

    def to_reduced_label_list(
        self, swara_labels: np.ndarray
    ) -> list[str]:
        """
        Collapse consecutive duplicate swaras into a list of full labels.

        Suitable for **bhatk export (includes octave information).

        Parameters
        ----------
        swara_labels : np.ndarray
            Array from :meth:`quantize`.

        Returns
        -------
        list[str]
            List of full swara labels (e.g. ``["S3", "R3", "G3"]``),
            with ``"REST"`` segments removed.
        """
        parts: list[str] = []
        prev = None
        for s in swara_labels:
            if s == "REST":
                prev = None
                continue
            if s != prev:
                parts.append(str(s))
                prev = s
        return parts
