# Hindustani Ornamentation Classification

Algorithmic detection and classification of ornamentations (alankar) in Hindustani classical vocal music, extending the work of [Jain & Arthur (2023)](https://doi.org/10.1145/3625135.3625137) (DLfM). The project combines a **rule-based** approach with a **Temporal Convolutional Network (TCN)** to classify four ornament types — **kan**, **meend**, **andolan**, and **murki** — from continuous f0 (pitch) contours, and introduces a density-driven **adaptive windowing** strategy that adjusts temporal resolution based on local melodic activity.

## Overview

### Pipeline

The system processes continuous f0 pitch contours extracted from vocal recordings through the following stages:

1. **Tonic Normalisation** — maps raw Hz values to a 3-octave (36-note) just-intonation grid anchored at the singer's tonic (Sa).
2. **Pitch Quantization** — snaps each sample to the nearest allowed swara for the given raga, using a 35-cent shruti threshold (Ganguli et al. 2016).
3. **Phrase Segmentation** — splits the f0 contour into melodic phrases by detecting silence gaps (≥500 ms).
4. **Windowed Instability Analysis** — slides a frame across each phrase and computes per-frame instability (sum of squared deviations from median). Supports both static (fixed 250 ms) and adaptive (density-driven) windowing.
5. **Ornament Classification** — a rule-based decision tree classifies unstable regions by counting direction changes and unique pitches:
   - ≤2 direction changes & short duration → **kan** (grace note)
   - ≤2 direction changes & longer duration → **meend** (glide)
   - \>2 direction changes & ≤3 unique pitches → **andolan** (oscillation)
   - \>2 direction changes & >3 unique pitches → **murki** (rapid ornament)
6. **Evaluation** — boundary hit rate (Turnbull et al. 2007) and per-class F1 against the OHV ground-truth annotations, plus weighted normalised Levenshtein distance (Daniel et al. 2008) for swara transcription accuracy.

### Adaptive Windowing

Instead of a fixed 250 ms (~56 sample) analysis frame, the adaptive windower estimates local swara transition density and adjusts the frame size accordingly:

```
frame_ms = base_frame_ms / (1 + α × normalised_density)
```

The frame size is clamped to [100 ms, 500 ms] (London 2002). Dense melodic passages (e.g. taan) use shorter frames for finer temporal resolution, while sparse passages (e.g. alap) use longer frames to reduce noise.

### TCN Classifier

In addition to the rule-based classifier, a Temporal Convolutional Network (TCN) is trained to classify ornament types directly from normalised log-f0 pitch curves. The model uses:

- Dilated causal convolutions (4–6 residual blocks with exponentially increasing dilation) to capture multi-scale temporal patterns.
- Global average pooling or Temporal Pyramid Pooling (He et al. 2015) for variable-length input handling.
- Balanced class weighting and time-stretching data augmentation to handle class imbalance.

The TCN is trained on ornament segments extracted from the OHV-annotated Saraga tracks and achieves comparable F1 scores to the rule-based approach while better capturing shape-level distinctions between ornament types.

![TCN Classifier Architecture](https://github.com/Sangarshanan/hindustani-ornamentation-classification/raw/main/media/tcn.png)

## Datasets

This study uses two datasets:

| Dataset | Description | Contents |
|---------|-------------|----------|
| [**Saraga 1.5 Hindustani**](https://zenodo.org/records/4301737) (`saraga1.5_hindustani/`) | Audio recordings with f0 pitch contours, tonic annotations, and section metadata. Loaded via [compiam](https://github.com/MTG/compIAM). | 30+ albums, each containing ragas with `.pitch.txt`, `.ctonic.txt`, and audio files. |
| [**OHV**](https://github.com/rhythmjain/automated-symbolic-transcription-hindustani-vocals/blob/main/Ornamentation-In-Hindustani-Vocals-Dataset/ornamentation-procedure/annotator-readme.md) (`Ornamentation-In-Hindustani-Vocals-Dataset/`) | Expert-annotated ornament boundaries across 11 Hindustani ragas (14 CSV files, 1768 ornament instances). | Paired start/end labels (`k_s`/`k_e`, `me_s`/`me_e`, etc.) by 5 annotators. |

The mapping between OHV annotations and Saraga tracks is defined in `src/hcm_transcription/mapping.py`, covering 11 ragas: Aahir Bhairon, Bairagi, Bhairavi Dadra, Irani Bhairavi, Kalavati, Malkauns, Chandrakauns, Majh Khamaj Thumri, Nat Bhairon, Raageshree, and Todi.

## Project Structure

```
hindustani-ornamentation-classification/
├── configs/
│   └── default.yaml                    # All hyperparameters (windowing, thresholds, evaluation)
├── data/
│   ├── raga_dict.json                  # 37 ragas with allowed note masks & just-intonation ratios
│   └── loaders/
│       ├── saraga_loader.py            # compiam-based Saraga Hindustani loader
│       └── ohv_loader.py               # OHV CSV annotation loader
├── src/hcm_transcription/             # Core library
│   ├── utils.py                        # Constants, just-intonation grid, config loading, annotation helpers
│   ├── preprocessing.py                # Tonic normalisation, pitch quantization, stability filtering
│   ├── segmentation.py                 # Phrase segmentation, Static & Adaptive windowers
│   ├── ornamentation.py                # Rule-based ornament classifier, OHV label mapper
│   ├── evaluation.py                   # WN-LD, boundary hit rate, classification report, windowing comparison
│   ├── mapping.py                      # OHV↔Saraga track mapping for all 11 ragas
│   └── humdrum_export.py               # **bhatk Humdrum notation export
├── notebooks/
│   ├── demo_executed_final.ipynb       # Full pipeline demo: static vs adaptive windowing evaluation
│   ├── data-analysis.ipynb             # Exploratory data analysis: instability curves, unstable regions per raga
│   ├── ornamentations_analysis.ipynb   # OHV annotation validation and ornamentation visualisation
│   ├── TCN_Final.ipynb                 # TCN training, cross-validation, ablation (fixed/adaptive/TPP)
│   ├── inference.ipynb                 # TCN inference on unseen Hindustani flute recordings
│   ├── model_definitions.py            # OrnamentTCN architecture (ResidualBlock + classifier)
│   └── checkpoints/
│       └── ornament_tcn.pth            # Trained TCN model weights
├── scripts/                            # CLI entry points
│   ├── run_transcription.py            # Full pipeline: quantize → segment → classify → export **bhatk
│   ├── evaluate_notes.py               # Swara transcription evaluation (WN-LD)
│   ├── evaluate_ornaments.py           # Ornament detection evaluation (boundary F1, per-class F1)
│   ├── ablation_window_size.py         # Frame size ablation study across static sizes + adaptive
│   ├── _diagnose_eval.py               # Diagnostic: raw vs stability-filtered evaluation
│   └── _profile_pipeline.py            # Pipeline profiling on real data
├── Ornamentation-In-Hindustani-Vocals-Dataset/  # OHV ground-truth annotations (14 CSV files)
├── saraga1.5_hindustani/               # Saraga Hindustani dataset (audio, pitch, tonic files)
├── tests/                              # pytest suite (synthetic inputs, no dataset required)
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   ├── test_ornamentation.py
│   └── test_evaluation.py
├── outputs/humdrum/                    # Generated **bhatk transcription files
├── environment.yml                     # Conda environment specification
├── requirements.txt                    # pip requirements
└── setup.py                            # Editable install configuration
```

## Notebooks

### [`demo_executed_final.ipynb`](notebooks/demo_executed_final.ipynb) — Full Pipeline Demo
End-to-end demonstration of the rule-based classification pipeline across all 11 OHV-annotated ragas:
- Loads Saraga tracks via compiam and preprocesses f0 contours (tonic normalisation + quantization).
- Runs ornament detection with both **static** and **adaptive** windowing.
- Evaluates against OHV ground truth using boundary hit rate and per-class F1.
- Produces three-way comparison tables (Paper baseline vs Static vs Adaptive) and visualisations of detected ornaments overlaid on pitch contours.

### [`data-analysis.ipynb`](notebooks/data-analysis.ipynb) — Exploratory Data Analysis
Analyses the preprocessed pitch data across all tracks:
- Visualises raw and quantized f0 contours.
- Computes and plots per-phrase instability curves with highlighted unstable regions.
- Aggregates statistics per raga: mean instability, proportion of unstable frames.
- Extracts contiguous unstable regions with timestamps and plots duration distributions per raga.

### [`TCN_Final.ipynb`](notebooks/TCN_Final.ipynb) — TCN Training & Experiments
Trains and evaluates the Temporal Convolutional Network for ornament classification:
- Extracts ornament pitch-curve segments from Saraga tracks using OHV annotations.
- Prepares fixed-length and variable-length input representations.
- Experiments with multiple architectures: 4-layer TCN, 6-layer TCN, TCN with Temporal Pyramid Pooling (TPP) and TCN with Attention Pooling.
- Applies data augmentation (time stretching) and balanced class weighting.
- Runs validation and reports per-class precision/recall/F1.
- Visualises misclassified examples and correct vs incorrect predictions per class.

### [`ornamentations_analysis.ipynb`](notebooks/ornamentations_analysis.ipynb) — Annotation Validation
Validates the OHV↔Saraga mapping and visualises individual ornamentation instances:
- Checks that annotation timestamps fall within audio durations.
- Plots pitch contours with highlighted ornament boundaries for manual inspection.

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate hcm-transcription

# 2. Install package in editable mode
pip install -e .

# 3. Run tests (no dataset required)
pytest tests/ -v
```

The Saraga 1.5 Hindustani dataset should be placed in `saraga1.5_hindustani/` at the repository root. The OHV annotations are already included in `Ornamentation-In-Hindustani-Vocals-Dataset/`.

## Usage

### Running the Notebooks

The primary way to explore this project is through the notebooks. Open them in JupyterLab:

```bash
jupyter lab notebooks/
```

- Start with **`demo_executed_final.ipynb`** for the complete rule-based pipeline and evaluation results.
- Run **`TCN_experiments.ipynb`** to train the TCN model from scratch.
- Use **`inference.ipynb`** to apply the trained model to new audio files.

### CLI Scripts

```bash
# Full pipeline: quantize, segment, detect ornaments, export to **bhatk
python scripts/run_transcription.py --config configs/default.yaml

# Evaluate swara transcription accuracy (WN-LD metric)
python scripts/evaluate_notes.py --config configs/default.yaml

# Evaluate ornament detection against OHV ground truth
python scripts/evaluate_ornaments.py --config configs/default.yaml

# Frame size ablation study (static sweep + adaptive)
python scripts/ablation_window_size.py --config configs/default.yaml
```

### Configuration

All hyperparameters are centralized in [`configs/default.yaml`](configs/default.yaml):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shruti_threshold_cents` | 35 | Maximum cents distance for pitch quantization |
| `min_silence_duration_ms` | 500 | Silence gap to split phrases |
| `base_frame_ms` | 250 | Static windowing frame size |
| `base_hop_ms` | 22 | Windowing hop size |
| `alpha` | 2.0 | Adaptive windowing density scaling factor |
| `instability_threshold` | 0.1 | Threshold for unstable region detection |
| `ornament_boundary_tolerance_s` | 0.5 | Evaluation boundary hit tolerance |

Set `adaptive_windowing.enabled: true` (default) to use adaptive windowing, or `false` for the static baseline.

## Evaluation Metrics

- **Boundary Hit Rate** (Turnbull et al. 2007) — precision, recall, and F1 of ornament boundary detection within a 0.5 s tolerance.
- **Per-class F1** — classification F1 for each ornament type (kan, meend, andolan, murki).
- **Weighted Normalised Levenshtein Distance** (Daniel et al. 2008) — similarity between predicted and ground-truth swara sequences.

## References

- Jain, V. & Arthur, T. (2023). An algorithmic approach to automated symbolic transcription of Hindustani vocals. *Proc. DLfM*. DOI: [10.1145/3625135.3625137](https://doi.org/10.1145/3625135.3625137)
- Ganguli, K. K., Gulati, S., Serra, X., & Rao, P. (2016). Data-driven exploration of pitch distributions in a North Indian raga. *Proc. ISMIR*.
- Turnbull, D., Barrington, L., Torres, D., & Lanckriet, G. (2007). Semantic annotation and retrieval of music and sound effects. *Proc. ISMIR*.
- Daniel, C., Clark, M., & Goebl, W. (2008). Weighted normalised Levenshtein distance for automatic music transcription evaluation. *Proc. ISMIR*.
- London, J. (2002). Cognitive constraints on metric systems. *Music Perception*, 19(4), 529–550.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition. *IEEE TPAMI*, 37(9), 1904–1916.
