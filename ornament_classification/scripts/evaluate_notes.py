#!/usr/bin/env python
"""
Evaluate swara transcription accuracy against Saraga ground-truth annotations.

Usage:
    python scripts/evaluate_notes.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hcm_transcription.evaluation import evaluate_file, weighted_normalized_levenshtein
from hcm_transcription.preprocessing import TonicNormalizer
from hcm_transcription.utils import load_config, load_raga_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate swara transcription")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--raga-dict", type=str, default="data/raga_dict.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raga_dict = load_raga_dict(args.raga_dict)

    random.seed(42)
    np.random.seed(42)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.loaders.saraga_loader import SaragaHindustaniLoader
    loader = SaragaHindustaniLoader(data_home=cfg["data"]["saraga_data_home"])
    track_ids = loader.get_annotated_tracks()
    print(f"Evaluating {len(track_ids)} tracks")

    results_rows = []

    for tid in tqdm(track_ids, desc="Evaluating notes"):
        try:
            track = loader.load_track(tid)
            raga_name = track.raga.lower().replace(" ", "_")
            if raga_name not in raga_dict["ragas"]:
                continue

            normalizer = TonicNormalizer(
                tonic_hz=track.tonic,
                raga_name=raga_name,
                raga_dict=raga_dict,
                shruti_threshold_cents=cfg["preprocessing"]["shruti_threshold_cents"],
            )
            swara_labels, quant_hz = normalizer.quantize(track.f0_freqs)

            sample_period = cfg["data"]["f0_sample_period_s"]
            frame_size = int(round(cfg["adaptive_windowing"]["base_frame_ms"] / 1000.0 / sample_period))
            hop_size = int(round(cfg["adaptive_windowing"]["base_hop_ms"] / 1000.0 / sample_period))

            # Get ground-truth phrase strings
            gt_phrases = list(track.phrases) if track.phrases else []
            if not gt_phrases:
                continue

            # Time-aligned evaluation with stability filter:
            # For each GT phrase, extract quantized swaras from matching
            # time range, apply stability filter, then reduce.
            aligned_pred = []
            aligned_gt = []
            for gt_p in gt_phrases:
                gt_swaras = gt_p.get("swaras", "")
                if not gt_swaras:
                    continue

                s_idx = max(0, int(gt_p["start"] / sample_period))
                e_idx = min(int(gt_p["end"] / sample_period), len(swara_labels))
                phrase_labels = swara_labels[s_idx:e_idx]

                stable = normalizer.stability_filter(phrase_labels, frame_size, hop_size)
                pred_swaras = normalizer.to_reduced_sequence(stable)

                if pred_swaras:
                    aligned_pred.append(pred_swaras)
                    aligned_gt.append(gt_swaras)

            if not aligned_gt:
                continue

            scores = evaluate_file(
                aligned_pred, aligned_gt,
                ins_weight=cfg["evaluation"]["ld_insertion_weight"],
                del_weight=cfg["evaluation"]["ld_deletion_weight"],
                sub_weight=cfg["evaluation"]["ld_substitution_weight"],
            )

            results_rows.append({
                "track_id": tid,
                "raga": raga_name,
                "phrase_mean": scores["phrase_mean"],
                "phrase_std": scores["phrase_std"],
                "file_score": scores["file_score"],
                "n_phrases_pred": len(aligned_pred),
                "n_phrases_gt": len(gt_phrases),
                "n_aligned": len(aligned_gt),
            })

        except Exception as e:
            print(f"  Error on {tid}: {e}")

    df = pd.DataFrame(results_rows)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    df.to_csv(results_dir / "note_evaluation.csv", index=False)

    print("\n=== Swara Transcription Results ===")
    print(f"Tracks evaluated: {len(df)}")
    if len(df) > 0:
        print(f"Mean phrase WN-LD: {df['phrase_mean'].mean():.4f} ± {df['phrase_mean'].std():.4f}")
        print(f"Mean file WN-LD:   {df['file_score'].mean():.4f} ± {df['file_score'].std():.4f}")

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "config": cfg,
        "summary": {
            "n_tracks": len(df),
            "mean_phrase_wnld": float(df["phrase_mean"].mean()) if len(df) else 0,
            "mean_file_wnld": float(df["file_score"].mean()) if len(df) else 0,
        },
    }
    (results_dir / f"note_eval_snapshot_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(
        json.dumps(snapshot, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
