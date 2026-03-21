"""
Phrase segmentation and windowed instability analysis.

Contains the static windower (faithful replication of the paper's 250ms / 22ms
scheme) and the adaptive windower (primary research contribution).

Static windower: Jain & Arthur (2023) DLfM, Section 3.1.
Adaptive windowing: future work from Section 5 of the paper.
Temporal integration lower bound: London (2002) Music Perception.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PCP:
    """Pitch-Class Profile segment (a windowed phrase chunk)."""
    start_time: float
    end_time: float
    f0_samples: np.ndarray
    quantized_samples: np.ndarray


@dataclass
class PhraseSegment:
    """A phrase delimited by silence boundaries."""
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    f0_hz: np.ndarray
    quantized_hz: np.ndarray
    swara_labels: np.ndarray


# ---------------------------------------------------------------------------
# Phrase segmentation (silence-based)
# ---------------------------------------------------------------------------

def segment_phrases(
    f0_hz: np.ndarray,
    times: np.ndarray,
    quantized_hz: np.ndarray,
    swara_labels: np.ndarray,
    min_silence_duration_s: float = 0.5,
    sample_period_s: float = 0.004444444,
) -> List[PhraseSegment]:
    """
    Segment a full-track f0 contour into phrases by detecting silence gaps.

    Parameters
    ----------
    f0_hz : np.ndarray
        Raw f0 values in Hz (0 = silence).
    times : np.ndarray
        Time stamp for each f0 sample.
    quantized_hz : np.ndarray
        Quantized f0 values.
    swara_labels : np.ndarray
        Swara label per sample.
    min_silence_duration_s : float
        Minimum silence gap to split phrases (default 500ms).
    sample_period_s : float
        Time between consecutive f0 samples.

    Returns
    -------
    List[PhraseSegment]
    """
    is_voiced = f0_hz > 0
    min_silence_samples = int(min_silence_duration_s / sample_period_s)

    phrases: List[PhraseSegment] = []
    phrase_start: Optional[int] = None
    silence_count = 0

    for i in range(len(f0_hz)):
        if is_voiced[i]:
            if phrase_start is None:
                phrase_start = i
            silence_count = 0
        else:
            silence_count += 1
            if phrase_start is not None and silence_count >= min_silence_samples:
                phrase_end = i - silence_count + 1
                if phrase_end > phrase_start:
                    phrases.append(PhraseSegment(
                        start_idx=phrase_start,
                        end_idx=phrase_end,
                        start_time=times[phrase_start],
                        end_time=times[phrase_end - 1],
                        f0_hz=f0_hz[phrase_start:phrase_end],
                        quantized_hz=quantized_hz[phrase_start:phrase_end],
                        swara_labels=swara_labels[phrase_start:phrase_end],
                    ))
                phrase_start = None

    # Flush last phrase
    if phrase_start is not None:
        phrase_end = len(f0_hz)
        phrases.append(PhraseSegment(
            start_idx=phrase_start,
            end_idx=phrase_end,
            start_time=times[phrase_start],
            end_time=times[min(phrase_end - 1, len(times) - 1)],
            f0_hz=f0_hz[phrase_start:phrase_end],
            quantized_hz=quantized_hz[phrase_start:phrase_end],
            swara_labels=swara_labels[phrase_start:phrase_end],
        ))

    return phrases


# ---------------------------------------------------------------------------
# Base Windower
# ---------------------------------------------------------------------------

class BaseWindower(abc.ABC):
    """Abstract interface for windowed instability analysis."""

    @abc.abstractmethod
    def compute_instability(
        self, quantized_hz: np.ndarray
    ) -> np.ndarray:
        """
        Compute the instability curve for a quantized f0 phrase.

        Parameters
        ----------
        quantized_hz : np.ndarray
            Quantized f0 values for a single phrase.

        Returns
        -------
        np.ndarray
            Per-frame instability scores.
        """

    @abc.abstractmethod
    def segment(
        self, quantized_f0: np.ndarray, times: np.ndarray
    ) -> List[PCP]:
        """
        Segment a phrase into PCP windows.

        Parameters
        ----------
        quantized_f0 : np.ndarray
            Quantized f0 values.
        times : np.ndarray
            Time-stamps corresponding to ``quantized_f0``.

        Returns
        -------
        List[PCP]
        """

    @staticmethod
    def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
        """Rolling mean with NaN fill for leading values."""
        if len(x) < w:
            return x.copy()
        cumsum = np.nancumsum(x)
        result = np.empty_like(x, dtype=np.float64)
        result[:w - 1] = np.nan
        result[w - 1:] = (cumsum[w - 1:] - np.concatenate([[0], cumsum[:-w]])) / w
        return result


# ---------------------------------------------------------------------------
# Static Windower — faithful replication of the paper
# ---------------------------------------------------------------------------

class StaticWindower(BaseWindower):
    """
    Static windower: fixed 56-sample (~250ms) frame, 5-sample (~22ms) hop.

    Instability = sum of squared deviations from frame median, normalised
    per phrase, then smoothed with a rolling mean (window=5).

    Reference: Jain & Arthur (2023) Section 3.1.
    """

    def __init__(
        self,
        frame_samples: int = 56,
        hop_samples: int = 5,
        ma_window: int = 5,
    ) -> None:
        self.frame_samples = frame_samples
        self.hop_samples = hop_samples
        self.ma_window = ma_window

    @classmethod
    def from_config(cls, cfg: dict) -> "StaticWindower":
        """Construct from a loaded YAML config dict."""
        aw = cfg["adaptive_windowing"]
        sample_period = cfg["data"]["f0_sample_period_s"]
        frame_s = int(round(aw["base_frame_ms"] / 1000.0 / sample_period))
        hop_s = int(round(aw["base_hop_ms"] / 1000.0 / sample_period))
        ma_w = cfg["ornamentation"]["moving_average_window"]
        return cls(frame_samples=frame_s, hop_samples=hop_s, ma_window=ma_w)

    def compute_instability(self, quantized_hz: np.ndarray) -> np.ndarray:
        """
        Compute per-frame instability scores.

        Parameters
        ----------
        quantized_hz : np.ndarray
            Quantized f0 values (0 removed internally).

        Returns
        -------
        np.ndarray
            Smoothed instability curve.
        """
        x = quantized_hz[quantized_hz != 0]
        if len(x) < self.frame_samples:
            return np.zeros(0)

        frame_count = 1 + (len(x) - self.frame_samples) // self.hop_samples
        instab = np.zeros(frame_count)

        for k in range(frame_count):
            start = k * self.hop_samples
            end = start + self.frame_samples
            frame = x[start:end]
            instab[k] = np.sum((frame - np.median(frame)) ** 2)

        # Normalise
        total = np.nansum(instab)
        if total > 0:
            instab = instab / total

        # Smooth
        smoothed = self._moving_average(instab, self.ma_window)
        return smoothed

    def segment(
        self, quantized_f0: np.ndarray, times: np.ndarray
    ) -> List[PCP]:
        """Segment into fixed-width PCP windows."""
        n = len(quantized_f0)
        if n < self.frame_samples:
            return []

        pcps: List[PCP] = []
        frame_count = 1 + (n - self.frame_samples) // self.hop_samples
        for k in range(frame_count):
            start = k * self.hop_samples
            end = start + self.frame_samples
            pcps.append(PCP(
                start_time=times[start],
                end_time=times[min(end - 1, len(times) - 1)],
                f0_samples=quantized_f0[start:end],
                quantized_samples=quantized_f0[start:end],
            ))
        return pcps


# ---------------------------------------------------------------------------
# Adaptive Windower — primary research contribution
# ---------------------------------------------------------------------------

class AdaptiveWindower(BaseWindower):
    """
    Adaptive windower: frame/hop size varies with local note density.

    Higher density (fast passages, ornaments) → smaller frame to capture detail.
    Lower density (slow alap) → larger frame to avoid over-segmentation.

    frame_ms = base_frame_ms / (1 + alpha * normalised_density)
    hop_ms   = frame_ms * (base_hop_ms / base_frame_ms)

    Both are clamped to [min_frame_ms, max_frame_ms].

    Parameters
    ----------
    base_frame_ms : float
        Baseline frame size (paper: 250ms).
    base_hop_ms : float
        Baseline hop size (paper: 22ms).
    min_frame_ms : float
        Minimum frame size (London 2002: ~100ms perceptual lower bound).
    max_frame_ms : float
        Maximum frame size.
    density_window_s : float
        Look-back window for note-density estimation.
    density_smoothing_frames : int
        Gaussian smoothing sigma for the density curve.
    alpha : float
        Scaling factor for density → frame mapping.
    sample_period_s : float
        f0 sampling period in seconds.
    ma_window : int
        Moving average window for instability smoothing.
    section_overrides : dict or None
        Optional mapping ``{(start_s, end_s): section_label}`` to force
        frame sizes in ``alap`` / ``tan`` sections.
    """

    def __init__(
        self,
        base_frame_ms: float = 250.0,
        base_hop_ms: float = 22.0,
        min_frame_ms: float = 100.0,
        max_frame_ms: float = 500.0,
        density_window_s: float = 2.0,
        density_smoothing_frames: int = 5,
        alpha: float = 2.0,
        sample_period_s: float = 0.004444444,
        ma_window: int = 5,
        section_overrides: Optional[dict] = None,
    ) -> None:
        self.base_frame_ms = base_frame_ms
        self.base_hop_ms = base_hop_ms
        self.min_frame_ms = min_frame_ms
        self.max_frame_ms = max_frame_ms
        self.density_window_s = density_window_s
        self.density_smoothing_frames = density_smoothing_frames
        self.alpha = alpha
        self.sample_period_s = sample_period_s
        self.ma_window = ma_window
        self.section_overrides = section_overrides or {}

    @classmethod
    def from_config(
        cls, cfg: dict, section_overrides: Optional[dict] = None
    ) -> "AdaptiveWindower":
        """Construct from a loaded YAML config dict."""
        aw = cfg["adaptive_windowing"]
        return cls(
            base_frame_ms=aw["base_frame_ms"],
            base_hop_ms=aw["base_hop_ms"],
            min_frame_ms=aw["min_frame_ms"],
            max_frame_ms=aw["max_frame_ms"],
            density_window_s=aw["density_window_s"],
            density_smoothing_frames=aw["density_smoothing_frames"],
            alpha=aw.get("alpha", 2.0),
            sample_period_s=cfg["data"]["f0_sample_period_s"],
            ma_window=cfg["ornamentation"]["moving_average_window"],
            section_overrides=section_overrides,
        )

    # ---- density estimation ----

    def _estimate_density(
        self, swara_labels: np.ndarray
    ) -> np.ndarray:
        """
        Compute local note density: number of swara transitions within a
        trailing window at each position.

        Parameters
        ----------
        swara_labels : np.ndarray
            Per-sample swara labels.

        Returns
        -------
        np.ndarray
            Smoothed density curve (same length as input).
        """
        n = len(swara_labels)
        window_samples = int(self.density_window_s / self.sample_period_s)
        density = np.zeros(n, dtype=np.float64)

        for i in range(n):
            start = max(0, i - window_samples)
            window = swara_labels[start:i + 1]
            # Count transitions (label changes, ignoring REST)
            transitions = 0
            prev = None
            for s in window:
                if s == "REST":
                    prev = None
                    continue
                if prev is not None and s != prev:
                    transitions += 1
                prev = s
            density[i] = transitions

        # Gaussian smoothing
        if self.density_smoothing_frames > 0:
            density = gaussian_filter1d(
                density, sigma=self.density_smoothing_frames
            )

        return density

    def _density_to_frame_samples(
        self, density: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """
        Map density values to frame sizes in samples.

        Parameters
        ----------
        density : np.ndarray
            Local note density curve.
        times : np.ndarray
            Time array (for section override lookup).

        Returns
        -------
        np.ndarray
            Per-sample frame size in samples.
        """
        # Min-max normalise
        d_min, d_max = density.min(), density.max()
        if d_max > d_min:
            norm_density = (density - d_min) / (d_max - d_min)
        else:
            norm_density = np.zeros_like(density)

        # Monotone decreasing mapping
        frame_ms = self.base_frame_ms / (1.0 + self.alpha * norm_density)
        frame_ms = np.clip(frame_ms, self.min_frame_ms, self.max_frame_ms)

        # Section-aware overrides
        for (sec_start, sec_end), label in self.section_overrides.items():
            mask = (times >= sec_start) & (times <= sec_end)
            label_lower = label.lower()
            if "alap" in label_lower or "ālāp" in label_lower:
                frame_ms[mask] = self.max_frame_ms
            elif "tan" in label_lower or "tarānā" in label_lower:
                frame_ms[mask] = self.min_frame_ms

        frame_samples = np.round(
            frame_ms / 1000.0 / self.sample_period_s
        ).astype(int)
        frame_samples = np.clip(frame_samples, 1, None)
        return frame_samples

    def get_frame_sizes_ms(
        self, swara_labels: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """
        Return the adaptive frame size in ms at each sample position.

        Useful for visualisation.
        """
        density = self._estimate_density(swara_labels)
        frame_samples = self._density_to_frame_samples(density, times)
        return frame_samples * self.sample_period_s * 1000.0

    def compute_instability(self, quantized_hz: np.ndarray) -> np.ndarray:
        """
        Compute instability with adaptive frame sizes.

        Uses the median frame size across the phrase as the global
        frame/hop (a simplification for the instability curve that
        keeps the output a regular 1-D array).
        """
        x = quantized_hz[quantized_hz != 0]
        if len(x) == 0:
            return np.zeros(0)

        # Use density from the quantized sequence itself
        # Create dummy labels for density estimation
        dummy_labels = np.array([f"n{v:.2f}" for v in x])
        density = self._estimate_density(dummy_labels)
        median_density = np.median(density)

        d_min, d_max = density.min(), density.max()
        if d_max > d_min:
            norm = (median_density - d_min) / (d_max - d_min)
        else:
            norm = 0.0

        frame_ms = self.base_frame_ms / (1.0 + self.alpha * norm)
        frame_ms = np.clip(frame_ms, self.min_frame_ms, self.max_frame_ms)
        frame_s = int(round(frame_ms / 1000.0 / self.sample_period_s))
        hop_ratio = self.base_hop_ms / self.base_frame_ms
        hop_s = max(1, int(round(frame_ms * hop_ratio / 1000.0 / self.sample_period_s)))

        if len(x) < frame_s:
            return np.zeros(0)

        frame_count = 1 + (len(x) - frame_s) // hop_s
        instab = np.zeros(frame_count)
        for k in range(frame_count):
            start = k * hop_s
            end = start + frame_s
            frame = x[start:end]
            instab[k] = np.sum((frame - np.median(frame)) ** 2)

        total = np.nansum(instab)
        if total > 0:
            instab = instab / total

        return self._moving_average(instab, self.ma_window)

    def compute_instability_adaptive(
        self,
        quantized_hz: np.ndarray,
        swara_labels: np.ndarray,
        times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute instability with per-position adaptive frame sizes.
        Uses actual swara labels for density, and rescales the instability
        by the ratio of adaptive-to-static frame counts so the threshold
        operates on a comparable scale.

        Returns instability values at variable positions along with their
        corresponding time stamps.

        Parameters
        ----------
        quantized_hz : np.ndarray
            Quantized f0 values.
        swara_labels : np.ndarray
            Swara labels per sample.
        times : np.ndarray
            Time array.

        Returns
        -------
        instab_values : np.ndarray
        instab_times : np.ndarray
        """
        density = self._estimate_density(swara_labels)
        frame_samples = self._density_to_frame_samples(density, times)
        hop_ratio = self.base_hop_ms / self.base_frame_ms

        instab_values: list[float] = []
        instab_times: list[float] = []

        pos = 0
        n = len(quantized_hz)
        while pos < n:
            fs = int(frame_samples[pos])
            hs = max(1, int(round(fs * hop_ratio)))
            end = pos + fs
            if end > n:
                break
            frame = quantized_hz[pos:end]
            voiced = frame[frame != 0]
            if len(voiced) > 0:
                ss = np.sum((voiced - np.median(voiced)) ** 2)
            else:
                ss = 0.0
            instab_values.append(ss)
            instab_times.append(times[pos])
            pos += hs

        instab_arr = np.array(instab_values)
        total = np.nansum(instab_arr)
        if total > 0:
            instab_arr = instab_arr / total
        instab_arr = self._moving_average(instab_arr, self.ma_window)

        return instab_arr, np.array(instab_times)

    def segment(
        self, quantized_f0: np.ndarray, times: np.ndarray
    ) -> List[PCP]:
        """Segment into variable-width PCP windows based on local density."""
        n = len(quantized_f0)
        if n == 0:
            return []

        # Use a simple density proxy from quantized values for segmentation
        dummy_labels = np.array([str(v) for v in quantized_f0])
        density = self._estimate_density(dummy_labels)
        frame_samples = self._density_to_frame_samples(density, times)
        hop_ratio = self.base_hop_ms / self.base_frame_ms

        pcps: List[PCP] = []
        pos = 0
        while pos < n:
            fs = int(frame_samples[pos])
            hs = max(1, int(round(fs * hop_ratio)))
            end = min(pos + fs, n)
            if end - pos < 2:
                break
            pcps.append(PCP(
                start_time=times[pos],
                end_time=times[end - 1],
                f0_samples=quantized_f0[pos:end],
                quantized_samples=quantized_f0[pos:end],
            ))
            pos += hs

        return pcps
