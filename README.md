# Hindustani Ornamentation Classification

Algorithmic ornamentation classification of Hindustani classical vocals with **adaptive windowing**.

Extends the approach of [Jain & Arthur (2023)](https://doi.org/10.1145/3625135.3625137) presented at DLfM, introducing a density-driven adaptive windowing strategy that adjusts temporal resolution based on local melodic activity.

## Overview

The system converts continuous f0 (pitch) contours into discrete swara sequences and ornament annotations using:

1. **Tonic normalisation** — maps Hz to just-intonation ratios across 3 octaves (36 notes)
2. **Quantization** — snaps to nearest allowed swara within a 35-cent shruti threshold (Ganguli et al. 2016)
3. **Windowing & instability** — segments f0 into frames, computes instability (sum of squared deviations from median)
4. **Ornament classification** — rule-based decision tree: direction changes → kan/meend/andolan/murki
5. **Evaluation** — weighted normalised Levenshtein distance (Daniel et al. 2008) + boundary hit rate (Turnbull et al. 2007)

### Key Contribution: Adaptive Windowing

```
frame_ms = base_frame_ms / (1 + α × normalised_density)
```

Instead of a fixed 250ms (~56 sample) frame, the adaptive windower estimates local swara transition density and shrinks/expands the frame accordingly, clamped to [100ms, 500ms]. Section-aware overrides further tune the frame size for alap (large) vs. taan (small) sections when Saraga section annotations are available.

## Project Structure

```
hindustani-transcription-v2/
├── configs/default.yaml          # All hyperparameters
├── data/
│   ├── raga_dict.json            # 37 ragas with allowed notes & ratios
│   └── loaders/                  # mirdata (Saraga) and OHV dataset loaders
├── src/hcm_transcription/        # Core library
│   ├── utils.py                  # Constants, config loading, frequency helpers
│   ├── preprocessing.py          # Tonic normalisation & quantization
│   ├── segmentation.py           # Static + Adaptive windowers
│   ├── ornamentation.py          # Ornament classifier
│   ├── humdrum_export.py         # **bhatk Humdrum encoding
│   └── evaluation.py             # WN-LD, boundary hit rate, classification report
├── scripts/                      # CLI entry points
│   ├── run_transcription.py      # Full pipeline
│   ├── evaluate_notes.py         # Swara evaluation
│   ├── evaluate_ornaments.py     # Ornament evaluation
│   └── ablation_window_size.py   # Frame size ablation study
├── tests/                        # pytest suite (synthetic inputs, no data required)
├── notebooks/demo.ipynb          # Interactive demo with all visualisations
├── results/                      # Experiment outputs (CSV, plots, config snapshots)
└── outputs/humdrum/              # Generated **bhatk files
```

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate hcm-transcription

# 2. Install package in editable mode
pip install -e .

# 3. Run tests (no dataset required)
pytest tests/ -v

# 4. Download Saraga Hindustani dataset (via mirdata)
python -c "import mirdata; mirdata.initialize('saraga_hindustani')"
```

## Exact Reproduction of Paper Results

### Static Baseline (Tables 3 & 4 of Jain & Arthur 2023)

```bash
# Transcribe all annotated Saraga tracks with static 250ms windowing
python scripts/run_transcription.py \
    --config configs/default.yaml \
    --windowing static

# Evaluate swara transcription (Table 3)
python scripts/evaluate_notes.py --config configs/default.yaml
# Expected: phrase WN-LD ≈ 0.62, file WN-LD ≈ 0.56

# Evaluate ornament detection (Table 4)
python scripts/evaluate_ornaments.py --config configs/default.yaml
# Expected: boundary F1 ≈ 0.44, kan F1 ≈ 0.11, meend F1 ≈ 0.37
```

### Adaptive Windowing (Follow-up Contribution)

```bash
# Transcribe with adaptive windowing enabled
python scripts/run_transcription.py \
    --config configs/default.yaml \
    --windowing adaptive

# Evaluate and compare
python scripts/evaluate_notes.py --config configs/default.yaml
python scripts/evaluate_ornaments.py --config configs/default.yaml

# Run the full ablation study (static sweep + adaptive)
python scripts/ablation_window_size.py --config configs/default.yaml
# Outputs: results/ablation_windowing.csv, results/ablation_windowing.png
```

### Interactive Demo

```bash
jupyter notebook notebooks/demo.ipynb
```

## Datasets

- **[Saraga Hindustani](https://mtg.github.io/saraga/)** — f0 contours, tonic annotations, section boundaries, raga labels. Loaded via [mirdata](https://mirdata.readthedocs.io/).
  - Srinivasamurthy et al. (2021) EMR. DOI: [10.18061/emr.v16i1.7492](https://doi.org/10.18061/emr.v16i1.7492)

- **[OHV Dataset](https://zenodo.org/records/14632599)** — Ornamentation-in-Hindustani-Vocals annotations with per-ornament start/end boundaries and type labels.

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Shruti threshold | 35 cents | Ganguli et al. (2016) ISMIR |
| Base frame size | 250ms (~56 samples) | London (2002) *Music Perception* |
| Hop size | ~22ms (~5 samples) | Jain & Arthur (2023) Section 3.3 |
| Adaptive α | 2.0 | Tuned on validation set |
| Frame clamp | [100ms, 500ms] | Empirical |
| WN-LD weights | ins=0.5, del=0.5, sub=1.0 | Daniel et al. (2008) ISMIR |
| Boundary tolerance | 0.5s | Turnbull et al. (2007) ISMIR |

## References

- Jain & Arthur (2023). *An Algorithmic Approach to Automated Symbolic Transcription of Hindustani Vocals*. DLfM. DOI: [10.1145/3625135.3625137](https://doi.org/10.1145/3625135.3625137)
- Salamon & Gómez (2012). Melodia f0 estimation. IEEE TASLP. DOI: [10.1109/TASL.2012.2188515](https://doi.org/10.1109/TASL.2012.2188515)
- London (2002). Temporal integration. *Music Perception*. DOI: [10.1525/mp.2002.19.4.529](https://doi.org/10.1525/mp.2002.19.4.529)
- Ganguli et al. (2016). 35-cent threshold. ISMIR.
- Turnbull et al. (2007). Boundary hit rate. ISMIR.
- Daniel et al. (2008). Weighted Levenshtein Distance. ISMIR.

## License

Research use only. See individual dataset licenses for data redistribution terms.
