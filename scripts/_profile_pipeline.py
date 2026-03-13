#!/usr/bin/env python
"""Quick profiling script for the pipeline on real data."""
import time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from hcm_transcription.utils import load_config, load_raga_dict
from hcm_transcription.preprocessing import TonicNormalizer
from hcm_transcription.segmentation import segment_phrases, StaticWindower, AdaptiveWindower
from hcm_transcription.ornamentation import OrnamentClassifier
from hcm_transcription.humdrum_export import export_bhatk

cfg = load_config("configs/default.yaml")
raga_dict = load_raga_dict("data/raga_dict.json")

from data.loaders.saraga_loader import SaragaHindustaniLoader
loader = SaragaHindustaniLoader(data_home=cfg["data"]["saraga_data_home"])
tids = loader.get_annotated_tracks()
print(f"Annotated tracks: {tids}")

track = loader.load_track(tids[0])
print(f"Track: {track.track_id}, raga: {track.raga}, tonic: {track.tonic:.2f} Hz")
print(f"Samples: {len(track.f0_freqs)}, Duration: {track.f0_times[-1]:.1f}s")
print(f"Sections: {track.sections}")
print(f"Phrases (GT): {len(track.phrases) if track.phrases else 0}")

normalizer = TonicNormalizer(
    tonic_hz=track.tonic,
    raga_name=track.raga,
    raga_dict=raga_dict,
    shruti_threshold_cents=cfg["preprocessing"]["shruti_threshold_cents"],
)
print(f"Allowed swaras ({len(normalizer.swara_labels)}): {list(normalizer.swara_labels)}")

t0 = time.time()
swara_labels, quant_hz = normalizer.quantize(track.f0_freqs)
t1 = time.time()
print(f"\nQuantization: {t1-t0:.2f}s")

n_voiced = int(np.sum(swara_labels != "REST"))
print(f"Voiced frames: {n_voiced}/{len(swara_labels)} ({100*n_voiced/len(swara_labels):.1f}%)")
unique = sorted(set(str(s) for s in swara_labels if s != "REST"))
print(f"Unique swaras ({len(unique)}): {unique}")

# Phrase segmentation
t2 = time.time()
phrases = segment_phrases(
    f0_hz=track.f0_freqs,
    times=track.f0_times,
    quantized_hz=quant_hz,
    swara_labels=swara_labels,
    min_silence_duration_s=cfg["segmentation"]["min_silence_duration_ms"] / 1000.0,
    sample_period_s=cfg["data"]["f0_sample_period_s"],
)
t3 = time.time()
print(f"\nPhrase segmentation: {t3-t2:.2f}s")
print(f"Phrases found: {len(phrases)}")
for i, p in enumerate(phrases[:5]):
    reduced = normalizer.to_reduced_sequence(p.swara_labels)
    print(f"  Phrase {i}: {p.start_time:.1f}s-{p.end_time:.1f}s, "
          f"samples={len(p.f0_hz)}, reduced='{reduced[:60]}{'...' if len(reduced) > 60 else ''}'")
if len(phrases) > 5:
    print(f"  ... ({len(phrases) - 5} more)")

# Instability + ornament detection on first non-trivial phrase
windower = StaticWindower.from_config(cfg)
classifier = OrnamentClassifier.from_config(cfg)
t4 = time.time()
all_swaras = []
all_durations = []
all_ornaments = []
total_orn_events = 0
for p in phrases:
    instab = windower.compute_instability(p.quantized_hz)
    events = classifier.classify(instab, p.quantized_hz, p.f0_hz, hop_samples=windower.hop_samples)
    total_orn_events += len(events)
    reduced_labels = normalizer.to_reduced_label_list(p.swara_labels)
    for label in reduced_labels:
        all_swaras.append(label)
        all_durations.append(0.0)
        all_ornaments.append(None)
t5 = time.time()
print(f"\nInstability + ornament detection: {t5-t4:.2f}s")
print(f"Total ornament events detected: {total_orn_events}")

# Export
outdir = Path("outputs/humdrum")
outdir.mkdir(parents=True, exist_ok=True)
if all_swaras:
    out = export_bhatk(
        swara_labels=all_swaras,
        durations=all_durations,
        output_path=outdir / f"{track.track_id}.bhatk",
        ornaments=all_ornaments,
        title=track.track_id,
        raga=track.raga,
    )
    print(f"\nExported to: {out}")
    # Show first 20 lines
    lines = out.read_text().splitlines()
    print(f"File has {len(lines)} lines. First 20:")
    for line in lines[:20]:
        print(f"  {line}")

# GT comparison
print(f"\n=== Ground Truth Phrases ===")
if track.phrases:
    for i, p in enumerate(track.phrases[:10]):
        print(f"  GT {i}: {p['start']:.1f}s-{p['end']:.1f}s swaras='{p['swaras']}'")
    if len(track.phrases) > 10:
        print(f"  ... ({len(track.phrases) - 10} more)")

print(f"\nTotal pipeline time: {t5-t0:.2f}s")
