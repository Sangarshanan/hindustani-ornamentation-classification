#!/usr/bin/env python
"""
Evaluate ornamentation detection against OHV ground-truth annotations.

Usage:
    python scripts/evaluate_ornaments.py --config configs/default.yaml
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

from hcm_transcription.evaluation import (
    boundary_hit_rate,
    ornament_classification_report,
)
from hcm_transcription.ornamentation import (
    CLASSIFIER_LABELS,
    LabelMapper,
    OrnamentClassifier,
)
from hcm_transcription.preprocessing import TonicNormalizer
from hcm_transcription.segmentation import (
    AdaptiveWindower,
    StaticWindower,
    segment_phrases,
)
from hcm_transcription.utils import load_config, load_raga_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ornamentation detection")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--raga-dict", type=str, default="data/raga_dict.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raga_dict = load_raga_dict(args.raga_dict)

    random.seed(42)
    np.random.seed(42)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.loaders.ohv_loader import OHVLoader
    ohv_loader = OHVLoader(data_path=cfg["data"]["ohv_data_path"])
    track_ids = ohv_loader.get_track_ids()
    print(f"Found {len(track_ids)} OHV tracks")

    classifier = OrnamentClassifier.from_config(cfg)
    tolerance = cfg["evaluation"]["ornament_boundary_tolerance_s"]

    all_pred_labels = []
    all_gt_labels = []
    boundary_results = []

    for tid in tqdm(track_ids, desc="Evaluating ornaments"):
        try:
            ohv_track = ohv_loader.load_track(tid)
            gt_boundaries = ohv_track.boundaries
            gt_labels = [
                a.label for a in ohv_track.annotations
                if LabelMapper.is_classifiable(a.label)
            ]

            # Collect predicted boundaries from ornament events
            # (In a full pipeline, this would load the corresponding Saraga f0
            #  and run the classifier. Here we show the evaluation framework.)
            pred_boundaries = np.array([])  # Placeholder
            pred_labels = []  # Placeholder

            bhr = boundary_hit_rate(pred_boundaries, gt_boundaries, tolerance)
            boundary_results.append({
                "track_id": tid,
                "precision": bhr["precision"],
                "recall": bhr["recall"],
                "f1": bhr["f1"],
                "n_gt": len(gt_boundaries),
            })

            all_pred_labels.extend(pred_labels)
            all_gt_labels.extend(gt_labels)

        except Exception as e:
            print(f"  Error on {tid}: {e}")

    # Boundary results
    df_bhr = pd.DataFrame(boundary_results)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    df_bhr.to_csv(results_dir / "ornament_boundary_evaluation.csv", index=False)

    print("\n=== Ornament Boundary Hit Rate ===")
    if len(df_bhr) > 0:
        print(f"Mean Precision: {df_bhr['precision'].mean():.4f}")
        print(f"Mean Recall:    {df_bhr['recall'].mean():.4f}")
        print(f"Mean F1:        {df_bhr['f1'].mean():.4f}")

    # Classification report
    if all_pred_labels and all_gt_labels:
        report = ornament_classification_report(all_pred_labels, all_gt_labels)
        print("\n=== Ornament Classification Report ===")
        for label, metrics in report.items():
            print(f"  {label}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "config": cfg,
    }
    (results_dir / f"orn_eval_snapshot_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(
        json.dumps(snapshot, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
