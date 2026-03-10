#!/usr/bin/env python
"""Diagnostic: compare raw vs stability-filtered evaluation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from hcm_transcription.utils import load_config, load_raga_dict
from hcm_transcription.preprocessing import TonicNormalizer
from hcm_transcription.evaluation import weighted_normalized_levenshtein
from data.loaders.saraga_loader import SaragaHindustaniLoader

cfg = load_config("configs/default.yaml")
raga_dict = load_raga_dict("data/raga_dict.json")

loader = SaragaHindustaniLoader(data_home=cfg["data"]["saraga_data_home"])
track = loader.load_track("Raag Lalit")

normalizer = TonicNormalizer(
    tonic_hz=track.tonic,
    raga_name=track.raga,
    raga_dict=raga_dict,
    shruti_threshold_cents=cfg["preprocessing"]["shruti_threshold_cents"],
)
swara_labels, quant_hz = normalizer.quantize(track.f0_freqs)

sample_period = cfg["data"]["f0_sample_period_s"]
frame_size = int(round(cfg["adaptive_windowing"]["base_frame_ms"] / 1000.0 / sample_period))
hop_size = int(round(cfg["adaptive_windowing"]["base_hop_ms"] / 1000.0 / sample_period))

print(f"Track: {track.track_id}, raga: {track.raga}")
print(f"GT phrases: {len(track.phrases)}")
print(f"Stability filter: frame={frame_size}, hop={hop_size}")
print()

def eval_phrase(phrase_labels, gt_swaras, use_stability):
    """Return (pred_str, score) for a phrase."""
    if use_stability:
        stable = TonicNormalizer.stability_filter(phrase_labels, frame_size, hop_size)
        pred = normalizer.to_reduced_sequence(stable)
    else:
        pred = normalizer.to_reduced_sequence(phrase_labels)
    score = weighted_normalized_levenshtein(pred, gt_swaras) if pred else 0.0
    return pred, score

# Per-phrase comparison (first 30)
print("=" * 80)
print(f"{'':3s} {'Time':>15s}  {'GT':>8s}  {'Raw pred':>30s}  {'Stable pred':>30s}  {'Raw':>5s} {'Stab':>5s}")
print("=" * 80)

for i, gt_p in enumerate(track.phrases[:30]):
    s_idx = max(0, int(gt_p["start"] / sample_period))
    e_idx = min(int(gt_p["end"] / sample_period), len(swara_labels))
    phrase_labels = swara_labels[s_idx:e_idx]
    gt_sw = gt_p["swaras"]

    raw_pred, raw_score = eval_phrase(phrase_labels, gt_sw, False)
    stab_pred, stab_score = eval_phrase(phrase_labels, gt_sw, True)

    print(f"{i:3d} {gt_p['start']:7.1f}-{gt_p['end']:6.1f}s "
          f"gt='{gt_sw:8s}' raw='{raw_pred[:25]:25s}' stab='{stab_pred[:25]:25s}' "
          f"{raw_score:.3f} {stab_score:.3f}")

# Full evaluation
raw_scores = []
stab_scores = []
for gt_p in track.phrases:
    s_idx = max(0, int(gt_p["start"] / sample_period))
    e_idx = min(int(gt_p["end"] / sample_period), len(swara_labels))
    pl = swara_labels[s_idx:e_idx]
    gt_sw = gt_p["swaras"]
    _, rs = eval_phrase(pl, gt_sw, False)
    _, ss = eval_phrase(pl, gt_sw, True)
    raw_scores.append(rs)
    stab_scores.append(ss)

print(f"\n{'='*60}")
print(f"{'Metric':30s} {'Raw':>10s} {'Stability':>10s}")
print(f"{'='*60}")
print(f"{'Mean phrase WN-LD':30s} {np.mean(raw_scores):10.4f} {np.mean(stab_scores):10.4f}")
print(f"{'Median phrase WN-LD':30s} {np.median(raw_scores):10.4f} {np.median(stab_scores):10.4f}")
print(f"{'Std phrase WN-LD':30s} {np.std(raw_scores):10.4f} {np.std(stab_scores):10.4f}")
print(f"{'Max phrase WN-LD':30s} {np.max(raw_scores):10.4f} {np.max(stab_scores):10.4f}")

# File-level
def file_level(use_stability):
    all_pred = []
    for p in track.phrases:
        s = max(0, int(p["start"] / sample_period))
        e = min(int(p["end"] / sample_period), len(swara_labels))
        pl = swara_labels[s:e]
        pred, _ = eval_phrase(pl, "", use_stability)
        all_pred.append(pred)
    return "".join(all_pred)

raw_all = file_level(False)
stab_all = file_level(True)
gt_all = "".join(p["swaras"] for p in track.phrases)

raw_file = weighted_normalized_levenshtein(raw_all, gt_all)
stab_file = weighted_normalized_levenshtein(stab_all, gt_all)
print(f"{'File WN-LD':30s} {raw_file:10.4f} {stab_file:10.4f}")
print(f"{'File pred length':30s} {len(raw_all):10d} {len(stab_all):10d}")
print(f"{'GT length':30s} {len(gt_all):10d} {len(gt_all):10d}")
