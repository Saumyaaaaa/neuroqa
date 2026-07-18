# NeuroQA — Explainable EEG Artifact Detection Platform

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch)
![MNE](https://img.shields.io/badge/MNE--Python-1.6+-brightgreen)
![Tests](https://img.shields.io/badge/Tests-Passing-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

A modular, research-grade deep learning framework for automatic detection,
segmentation, and clinical interpretation of artifacts in multi-channel
EEG (Electroencephalogram) recordings.

NeuroQA combines multi-band digital signal processing, a spatial-temporal
Vision Transformer, and an attention rollout explainability layer to translate
black-box neural network decisions into human-readable clinical time-window
reports — making it suitable for both research and clinical trust applications.

---

## The Problem This Solves

EEG recordings are contaminated by artifacts — eye blinks, muscle movement,
electrode noise, and power line interference — that corrupt downstream
neurological analysis. Manual artifact rejection is slow, subjective, and
inconsistent across clinicians. Existing automated methods are either too
simple (threshold rules) or completely uninterpretable (black-box classifiers).

NeuroQA solves both problems: it detects artifacts automatically AND explains
exactly which time windows and signal patterns triggered the detection, in
seconds and channel names a clinician can understand.

---

## Architecture Overview

Data flows sequentially through six functional layers:

```text
[Raw .edf File]
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  loader.py — Ingestion & Validation                 │
│  Reads .edf files via MNE-Python                    │
│  Validates 19 standard 10-20 channels               │
│  Resamples to 256 Hz if needed                      │
│  Raises ChannelNotFoundError on missing channels    │
└─────────────────────────┬───────────────────────────┘
                          │ (n_channels, n_timepoints)
                          ▼
┌─────────────────────────────────────────────────────┐
│  filters.py — Digital Signal Processing             │
│  Butterworth bandpass filter: 1–40 Hz               │
│  IIR notch filter: 50 Hz power line removal         │
│  Zero-phase filtering (no temporal distortion)      │
└─────────────────────────┬───────────────────────────┘
                          │ (n_channels, n_timepoints)
                          ▼
┌─────────────────────────────────────────────────────┐
│  segmenter.py — Sliding Window Extraction           │
│  2.0-second windows at 256 Hz = 512 samples         │
│  50% overlap between consecutive windows            │
│  Output: (n_segments, n_channels, 512)              │
└─────────────────────────┬───────────────────────────┘
                          │ (n_segments, 19, 512)
                          ▼
┌─────────────────────────────────────────────────────┐
│  dataset.py — PyTorch Data Pipeline                 │
│  EEGArtifactDataset wraps segments as tensors       │
│  WeightedRandomSampler handles class imbalance      │
│  Graceful skip on corrupted or missing files        │
└─────────────────────────┬───────────────────────────┘
                          │ batched tensors
                          ▼
┌─────────────────────────────────────────────────────┐
│  transformer.py — EEGTransformer Model              │
│  Patch embedding: 32-sample patches → d_model       │
│  Learnable CLS token for classification             │
│  4-layer Transformer Encoder (batch_first=True)     │
│  Sinusoidal positional encoding                     │
│  Output: logits (batch, 2) — artifact / clean       │
└─────────────────────────┬───────────────────────────┘
                          │ logits + attention weights
                          ▼
┌─────────────────────────────────────────────────────┐
│  attention_rollout.py — Explainability Layer        │
│  Implements Abnar & Zuidema (2020) rollout          │
│  Maps attention weights → time windows in seconds   │
│  Returns top-3 suspicious time windows per segment  │
│  Output: clinical diagnostic report (dict)          │
└─────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

| Module | Layer | Core Function |
|---|---|---|
| `loader.py` | Ingestion | Reads `.edf` files, validates channels, resamples |
| `filters.py` | Signal Processing | Bandpass 1–40 Hz + 50 Hz notch filter |
| `segmenter.py` | Windowing | 2s overlapping sliding windows |
| `dataset.py` | Data Pipeline | PyTorch Dataset with weighted class balancing |
| `transformer.py` | Model | Vision Transformer ~600k params, CLS classification |
| `attention_rollout.py` | Explainability | Attention rollout → clinical time-window report |

---

## Project Structure

```text
neuroqa/
│
├── configs/
│   └── default.yaml              # All hyperparameters and channel config
│
├── data/
│   └── raw/                      # Place .edf subject files here
│
├── src/
│   └── neuroqa/
│       ├── __init__.py
│       │
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── loader.py         # EDF ingestion and validation
│       │   ├── filters.py        # Bandpass and notch filtering
│       │   ├── segmenter.py      # Sliding window segmentation
│       │   └── dataset.py        # PyTorch Dataset and DataLoader factory
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── transformer.py    # EEGTransformer architecture
│       │   └── registry.py       # Model factory from config
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── attention_rollout.py  # Explainability layer
│       │
│       └── training/
│           ├── __init__.py
│           └── trainer.py        # Training loop and validation
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Shared pytest fixtures
│   ├── test_preprocessing.py     # Preprocessing pipeline tests
│   ├── test_model_shapes.py      # Model output shape and speed tests
│   └── test_adversarial.py       # Robustness under edge-case inputs
│
├── notebooks/
│   └── exploration.ipynb         # Exploratory analysis only
│
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions — pytest on every push
│
├── REPRODUCIBILITY.md            # Exact steps to reproduce all results
├── FAILURE_MODES.md              # Documented model failure cases
├── requirements.txt              # Pinned dependencies
├── run_pipeline.py               # End-to-end integration script
└── README.md
```

---

## Quickstart

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourusername/neuroqa.git
cd neuroqa
pip install -r requirements.txt
```

### 2. Set Python path

```bash
# Windows PowerShell
$env:PYTHONPATH="src"

# Linux / macOS
export PYTHONPATH="src"
```

### 3. Download sample EEG data

Download any subject file from the
[EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
on PhysioNet (free, no registration required).

Place `.edf` files in `data/raw/`.

### 4. Run the full pipeline

```bash
python run_pipeline.py
```

This runs a complete end-to-end pass: data loading → filtering → segmentation
→ model inference → explainability report. No training required to see output.

### 5. Run the test suite

```bash
python -m pytest tests/ -v --tb=short
```

All tests run without requiring real EDF files — the test suite uses mocked
data and synthetic signals throughout.

---

## The QA Engineering Difference

Most AI research repositories contain untested, non-reproducible code that
only works on the author's machine. NeuroQA is built differently:

- **Formal test suite** with pytest covering preprocessing invariants,
  model output shapes, inference speed, gradient flow, and adversarial inputs
- **CI pipeline** via GitHub Actions running the full test suite on every push
- **Reproducibility guarantee** — pinned dependencies, fixed random seeds,
  and a dedicated `REPRODUCIBILITY.md` documenting exact steps
- **Failure mode documentation** — `FAILURE_MODES.md` catalogues known
  failure cases, edge conditions, and planned mitigations
- **Pure functions throughout** — no global state, no side effects,
  same input always produces same output
- **Adversarial robustness testing** — model tested against zero inputs,
  extreme amplitude signals, and edge-case window sizes

This reflects a software quality assurance mindset applied to AI research —
a combination rarely seen in academic ML codebases.

---

## Explainability Example

Given a 2-second EEG segment flagged as an artifact, NeuroQA produces:

```python
{
    "prediction": "artifact",
    "confidence": 0.923,
    "top_time_windows": [
        (0.0, 0.25, 0.412),   # start_sec, end_sec, importance
        (0.5, 0.75, 0.287),
        (1.0, 1.25, 0.198)
    ],
    "channel_names": ["Fp1", "Fp2", "F3", ...],
    "raw_logits": [-2.14, 2.89]
}
```

The `top_time_windows` field tells a clinician: the model flagged this segment
primarily because of signal patterns between 0.0s and 0.25s — a time range
they can visually inspect in the raw recording.

---

## Research Foundation

- **Transformer architecture**: Vaswani et al., *Attention Is All You Need* (2017)
- **Attention Rollout**: Abnar & Zuidema, *Quantifying Attention Flow in Transformers* (2020)
- **EEG artifact detection**: Jiang et al., *Removal of Artifacts from EEG Signals* (2019)
- **Dataset**: Goldberger et al., PhysioNet EEG Motor Movement/Imagery Dataset

---

## Roadmap

- [ ] Real EDF file integration with PhysioNet dataset
- [ ] Streamlit demo — upload EEG segment, get artifact report in browser
- [ ] FastAPI inference endpoint with request validation
- [ ] Cross-dataset domain adaptation experiment
- [ ] Comparison against threshold-based baseline

---

## License

MIT License — see `LICENSE` for details.