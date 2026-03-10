#!/usr/bin/env python
"""
Transcribe all Saraga Hindustani tracks and export to **bhatk files.

Usage:
    python scripts/run_transcription.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hcm_transcription.preprocessing import TonicNormalizer
from hcm_transcription.segmentation import (
    AdaptiveWindower,
    StaticWindower,
    segment_phrases,
)
from hcm_transcription.ornamentation import OrnamentClassifier
from hcm_transcription.humdrum_export import export_bhatk, swara_label_to_bhatk
from hcm_transcription.utils import load_config, load_raga_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe Saraga Hindustani tracks")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--raga-dict", type=str, default="data/raga_dict.json")
    parser.add_argument("--output-dir", type=str, default="outputs/humdrum")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raga_dict = load_raga_dict(args.raga_dict)

    # Reproducibility
    random.seed(42)
    np.random.seed(42)

    # Save config snapshot
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "config": cfg,
    }
    snapshot_path = results_dir / f"config_snapshot_{datetime.now():%Y%m%d_%H%M%S}.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, default=str))

    # Load dataset
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.loaders.saraga_loader import SaragaHindustaniLoader
    loader = SaragaHindustaniLoader(data_home=cfg["data"]["saraga_data_home"])
    track_ids = loader.get_annotated_tracks()
    print(f"Found {len(track_ids)} annotated tracks")

    # Select windower
    if cfg["adaptive_windowing"]["enabled"]:
        print("Using ADAPTIVE windowing")
    else:
        print("Using STATIC windowing")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orn_cfg = cfg["ornamentation"]
    classifier = OrnamentClassifier.from_config(cfg)

    for tid in tqdm(track_ids, desc="Transcribing"):
        try:
            track = loader.load_track(tid)

            # Resolve raga name to dictionary key
            raga_name = track.raga.lower().replace(" ", "_")
            if raga_name not in raga_dict["ragas"]:
                print(f"  Skipping {tid}: raga '{raga_name}' not in dictionary")
                continue

            # Preprocessing
            normalizer = TonicNormalizer(
                tonic_hz=track.tonic,
                raga_name=raga_name,
                raga_dict=raga_dict,
                shruti_threshold_cents=cfg["preprocessing"]["shruti_threshold_cents"],
            )
            swara_labels, quant_hz = normalizer.quantize(track.f0_freqs)

            # Phrase segmentation
            phrases = segment_phrases(
                f0_hz=track.f0_freqs,
                times=track.f0_times,
                quantized_hz=quant_hz,
                swara_labels=swara_labels,
                min_silence_duration_s=cfg["segmentation"]["min_silence_duration_ms"] / 1000.0,
                sample_period_s=cfg["data"]["f0_sample_period_s"],
            )

            # Per-phrase transcription
            all_swaras: list[str] = []
            all_durations: list[float] = []
            all_ornaments: list[str | None] = []

            for phrase in phrases:
                # Select windower
                if cfg["adaptive_windowing"]["enabled"]:
                    section_overrides = loader.get_section_overrides(track)
                    windower = AdaptiveWindower.from_config(cfg, section_overrides)
                    instab = windower.compute_instability(phrase.quantized_hz)
                else:
                    windower = StaticWindower.from_config(cfg)
                    instab = windower.compute_instability(phrase.quantized_hz)

                # Ornament detection
                events = classifier.classify(
                    instab, phrase.quantized_hz, 
                    phrase.f0_hz,  # times proxy
                    hop_samples=windower.hop_samples if hasattr(windower, 'hop_samples') else 5,
                )

                # Collect reduced swara sequence (stability-filtered)
                stable = normalizer.stability_filter(
                    phrase.swara_labels,
                    frame_size=windower.frame_samples if hasattr(windower, 'frame_samples') else 56,
                    hop_size=windower.hop_samples if hasattr(windower, 'hop_samples') else 5,
                )
                reduced = normalizer.to_reduced_label_list(stable)
                for label in reduced:
                    all_swaras.append(label)
                    all_durations.append(0.0)
                    all_ornaments.append(None)

            if all_swaras:
                export_bhatk(
                    swara_labels=all_swaras,
                    durations=all_durations,
                    output_path=output_dir / f"{tid}.bhatk",
                    ornaments=all_ornaments,
                    title=tid,
                    raga=raga_name,
                )

        except Exception as e:
            print(f"  Error processing {tid}: {e}")

    print(f"Transcription complete. Output in {output_dir}/")


if __name__ == "__main__":
    main()
