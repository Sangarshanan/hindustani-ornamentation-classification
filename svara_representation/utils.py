import os
import numpy as np
import pandas as pd
import pickle
from scipy.interpolate import UnivariateSpline

def save_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def smooth_pitch_curve(time_series, pitch_series, smoothing_factor=0.6, min_points=4):
    time_series = np.array(time_series, dtype=float)
    pitch_series = np.array(pitch_series, dtype=float)
    smoothed_pitch = np.full_like(pitch_series, np.nan)
    valid_mask = ~pd.isna(time_series) & ~pd.isna(pitch_series)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return smoothed_pitch

    contiguous_chunks = np.split(valid_indices, np.where(np.diff(valid_indices) > 1)[0] + 1)

    for chunk in contiguous_chunks:
        if len(chunk) >= min_points:
            time_chunk = time_series[chunk]
            pitch_chunk = pitch_series[chunk]
            pitch_min = np.min(pitch_chunk)
            pitch_max = np.max(pitch_chunk)
            normalized_pitch_chunk = (pitch_chunk - pitch_min) / (pitch_max - pitch_min)
            spline_func = UnivariateSpline(time_chunk, normalized_pitch_chunk, s=smoothing_factor)
            smoothed_normalized_pitch = spline_func(time_chunk)
            smoothed_pitch[chunk] = (smoothed_normalized_pitch * (pitch_max - pitch_min) + pitch_min)
        elif len(chunk) > 1:
            time_chunk = time_series[chunk]
            pitch_chunk = pitch_series[chunk]
            smoothed_pitch[chunk] = np.interp(time_chunk, time_chunk, pitch_chunk)

    return smoothed_pitch

def interpolate(arr, val, gap, indices=[]):
    s = np.copy(arr)
    indices = set(indices)
    if np.isnan(val):
        is_gap = np.isnan(s)
    else:
        is_gap = s == val
    in_gap = False
    gap_start = None
    gap_ranges = []

    for i, g in enumerate(is_gap):
        if g and not in_gap:
            in_gap = True
            gap_start = i
        elif not g and in_gap:
            in_gap = False
            gap_ranges.append((gap_start, i))

    if in_gap:
        gap_ranges.append((gap_start, len(s)))

    for start, end in gap_ranges:
        length = end - start
        if length > gap:
            continue
        if any(idx in indices for idx in range(start, end)):
            continue
        s[start:end] = np.nan

    series = pd.Series(s)
    interp = series.interpolate(method="linear").ffill().bfill().values

    return interp
