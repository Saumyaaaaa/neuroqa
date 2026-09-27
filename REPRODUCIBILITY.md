# REPRODUCIBILITY — NeuroQA

This document provides exact steps to reproduce every result in this
repository from scratch. Anyone following these steps on any machine
should get identical outputs.

---

## Environment

| Component | Version |
|---|---|
| Python | 3.12 |
| Operating System | Windows 11 / Ubuntu 22.04 / macOS 13+ |
| PyTorch | 2.1.0+ |
| MNE-Python | 1.6.0+ |
| ONNX | 1.15.0+ |
| onnxruntime | 1.17.0+ |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/Saumyaaaaa/neuroqa.git
cd neuroqa
git checkout feature/data-pipeline
```

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

All dependencies are pinned to exact versions in requirements.txt.
Do not upgrade packages individually as this may break compatibility.

---

## Step 3 — Set Python Path

Windows PowerShell:
```bash
$env:PYTHONPATH="D:\neuroqa\src"
```

Linux / macOS:
```bash
export PYTHONPATH="$(pwd)/src"
```

---

## Step 4 — Verify All Imports Work

```bash
python -c "
from neuroqa.preprocessing import load_edf, apply_standard_filters, segment_signal
from neuroqa.models import EEGTransformer, create_model
from neuroqa.evaluation import generate_artifact_report
print('All imports OK')
"
```

Expected output:
```
All imports OK
```

---

## Step 5 — Verify Model Builds from Config

```bash
python -c "
import sys, yaml, torch
sys.path.insert(0, 'src')
from neuroqa.models import create_model
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)
model = create_model(config)
print(f'Parameters: {model.get_param_count():,}')
x = torch.randn(4, 19, 512)
out = model(x)
print(f'Output shape: {out.shape}')
print('Model OK')
"
```

Expected output:
```
Parameters: [under 700,000]
Output shape: torch.Size([4, 2])
Model OK
```

---

## Step 6 — Run the Full Test Suite

```bash
python -m pytest tests/ -v --tb=short
```

Expected output:
```
tests/test_preprocessing.py ....... PASSED
tests/test_model_shapes.py ........ PASSED
tests/test_adversarial.py ......... PASSED
tests/test_api.py ................. PASSED
======================== ALL TESTS PASSED ========================
```

All tests must pass before any other step is run.
If any test fails, do not proceed — fix the failure first.

---

## Step 7 — Export Model to ONNX

```bash
python scripts/export_model.py --benchmark --n-runs 50
```

Expected output:
```
NeuroQA ONNX Export Results
┌────────────────────────┬────────────────────────┐
│ Output path │ checkpoints\model.onnx │
│ Model size │ 0.379 MB │
│ PyTorch match │ ✓ YES │
│ Output shape │ (1, 2) │
│ Opset version │ 14 │
└────────────────────────┴────────────────────────┘

CPU Inference Benchmark (50 runs)
┌────────────────────────┬──────────────────┐
│ Mean latency │ ~4.5 ms │
│ Throughput │ ~220 /sec │
└────────────────────────┴──────────────────┘
```

The ONNX file is saved to checkpoints/model.onnx.
Model size must be under 1 MB.
PyTorch match must show ✓ YES.

---

## Step 8 — Run the FastAPI Server

```bash
uvicorn neuroqa.api.app:app --host 0.0.0.0 --port 8000
```

Verify with:
```bash
curl http://localhost:8000/health
```

Expected output:
```json
{"status": "ok", "model_loaded": true, "version": "1.0.0"}
```

Test the predict endpoint:
```bash
python -c "
import httpx, numpy as np, json
signal = np.random.randn(19, 512).astype('float32')
response = httpx.post(
    'http://localhost:8000/predict',
    json={'eeg_segment': signal.tolist()},
    timeout=30
)
print(json.dumps(response.json(), indent=2))
"
```

Expected output shape:
```json
{
  "prediction": "clean",
  "confidence": 0.858,
  "top_time_windows": [[0.0, 0.125, 0.0625], ...],
  "processing_time_ms": 7.6,
  "model_version": "1.0.0"
}
```

---

## Step 9 — Run the Streamlit Demo

In a second terminal while FastAPI is running:

```bash
streamlit run streamlit_app/demo.py
```

Open browser at http://localhost:8501

Expected behaviour:
- Page loads with EEG signal chart
- Select "Generate demo signal" in sidebar
- Click "Run Detection"
- Result shows prediction with confidence score
- Annotated chart shows highlighted time windows

---

## Random Seeds Used

| Location | Seed | Purpose |
|---|---|---|
| tests/conftest.py | 42 | Sample EEG fixture data |
| tests/conftest.py | 42 | Model weight initialisation |
| streamlit_app/demo.py | 42 | Demo signal generation |
| scripts/export_model.py | None | Random weights (untrained) |

All seeds are fixed to guarantee identical outputs across machines
and runs. If you change any seed, outputs will differ from those
documented here.

---

## Known Non-Reproducibility

**Model predictions on real EEG data:**
The model has not been trained. All predictions currently reflect
random weight initialisation. Once trained on the Temple University
EEG Artifact Corpus, a trained checkpoint will be added to
checkpoints/trained_model.pt and this document will be updated
with training reproducibility steps including dataset version,
training duration, final validation metrics, and hardware used.

**Latency benchmarks:**
Inference latency varies by CPU. The documented ~4.5ms was measured
on an Intel Core i5 Windows 11 machine. Your hardware will produce
different latency values. This does not affect prediction correctness.

---

## File Checksums

After running Step 7, verify your ONNX file exists:

Windows PowerShell:
```bash
Get-FileHash checkpoints\model.onnx -Algorithm MD5
```

Linux / macOS:
```bash
md5sum checkpoints/model.onnx
```

Note your hash value here after first export:

MD5: [run the command and paste your hash here]


Re-exporting from the same random weights should produce
the same hash, confirming deterministic export.

---

## Troubleshooting

**Import error: cannot import name X**
Check that PYTHONPATH is set to the src/ directory.

**ONNX export fails with encoder_layer_fwd error**
This is a PyTorch opset compatibility issue. Ensure
opset_version=14 is used and the sdp_kernel context
manager is wrapping the torch.onnx.export call in
src/neuroqa/models/exporter.py.

**FastAPI returns 422 on /predict**
Your input signal contains NaN or Inf values, or the
shape is not exactly (19, 512). Check your input array
with numpy before sending.

**Streamlit cannot connect to API**
Ensure the FastAPI server is running in a separate
terminal before starting Streamlit. Check the API URL
in the sidebar matches your server address.

---

*Last updated: September 2026*
*Repository: github.com/Saumyaaaaa/neuroqa*
*Python: 3.12 | PyTorch: 2.1+ | ONNX: 1.15+*
