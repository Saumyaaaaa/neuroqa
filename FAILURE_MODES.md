# FAILURE MODES — NeuroQA

This document catalogues known failure cases, edge conditions, and 
planned mitigations for the NeuroQA EEG artifact detection pipeline.
Maintained in the spirit of honest AI engineering.

---

## FM-001 — Model Operates on Random Weights

**Status:** Known / Unresolved  
**Severity:** Critical for production use  

**Description:**  
The EEGTransformer model has never been trained on real labelled EEG 
data. All inference results currently reflect random weight 
initialisation, not learned artifact patterns. The model will return 
confident-looking predictions (e.g. 85% confidence) that have no 
clinical validity.

**Observed behaviour:**  
The model consistently predicts "clean" with ~85% confidence regardless 
of whether the input contains real artifacts or not.

**Root cause:**  
No labelled training dataset has been integrated. The pipeline 
architecture is complete but the training phase has not been executed.

**Planned mitigation:**  
Download the Temple University EEG Artifact Corpus (TUH EAR) from 
isip.piconepress.com and the PhysioNet EEG Motor Movement dataset. 
Train the model using trainer.py with real artifact and clean labels. 
Re-export to ONNX after training.

---

## FM-002 — Attention Rollout Time Windows Are Uniform Placeholders

**Status:** Known / Partial  
**Severity:** Medium  

**Description:**  
The top_time_windows returned by the FastAPI /predict endpoint are 
computed by dividing the 512-sample window into 16 patches of 32 
samples each and assigning uniform importance scores of 0.0625 to 
all patches. This is a placeholder implementation.

**Observed behaviour:**  
All three top time windows always show importance score of 0.0625. 
The windows always cover 0.000s–0.125s, 0.125s–0.250s, 0.250s–0.375s 
regardless of input signal content.

**Root cause:**  
The ONNX inference session does not expose internal attention weights. 
Real attention rollout (Abnar and Zuidema 2020) requires access to 
per-layer attention matrices which are only available in the PyTorch 
model during a forward pass.

**Planned mitigation:**  
Add a separate PyTorch-based explainability endpoint to the FastAPI 
server that runs attention_rollout.py when real attention weights are 
needed. Use ONNX for fast classification and PyTorch for explainability 
on demand.

---

## FM-003 — No Cross-Dataset Validation

**Status:** Known / Unresolved  
**Severity:** High for research claims  

**Description:**  
The model has not been evaluated on any external dataset. Generalisation 
performance across different EEG recording setups, electrode placements, 
amplifier hardware, and clinical populations is completely unknown.

**Planned mitigation:**  
After initial training on TUH EAR, evaluate on the PhysioNet EEG Motor 
Movement dataset as a held-out cross-dataset test. Report per-class F1 
scores and confusion matrices.

---

## FM-004 — Signal Too Short Error on Short Recordings

**Status:** Known / Handled  
**Severity:** Low  

**Description:**  
If an uploaded .npy file contains fewer than 512 time samples after 
transposition, the pipeline raises a handled error and stops processing.

**Observed behaviour:**  
Streamlit shows: "Signal too short: N samples. Need at least 512."

**Root cause:**  
The model requires exactly 512 samples (2 seconds at 256Hz) per window. 
Recordings shorter than 2 seconds cannot be processed.

**Current mitigation:**  
Graceful error with clear message. No crash.

**Planned mitigation:**  
Add zero-padding for signals between 256 and 512 samples with a warning 
that results may be less reliable.

---

## FM-005 — Channel Name Mismatch on Upload

**Status:** Known / Partial  
**Severity:** Medium  

**Description:**  
When a user uploads a .npy file, the pipeline assumes the 19 channels 
are in standard 10-20 order (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, 
O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz). If the uploaded file uses 
a different channel ordering or contains non-standard channels, the 
model processes incorrect channel assignments silently.

**Planned mitigation:**  
Add a channel mapping UI in the Streamlit sidebar allowing users to 
specify their channel order. Add a metadata sidecar .json file option.

---

## FM-006 — NaN Propagation from Aggressive Filtering

**Status:** Known / Handled  
**Severity:** Medium  

**Description:**  
If input signal amplitude is extremely large (e.g. electrode 
disconnection producing rail values), the Butterworth filter can 
produce NaN or Inf values that propagate through the pipeline.

**Current mitigation:**  
The FastAPI /predict endpoint checks for NaN and Inf values before 
inference and returns HTTP 422 with message "NaN or Inf detected in 
input signal."

**Planned mitigation:**  
Add amplitude clipping at ±500 microvolts before filtering as a 
pre-processing guard in filters.py.

---

## FM-007 — Free Tier Memory Limits on Deployment

**Status:** Anticipated / Not yet deployed  
**Severity:** Low (mitigated by ONNX)  

**Description:**  
Free hosting tiers on Render and HuggingFace Spaces typically provide 
512MB RAM. The original PyTorch model requires approximately 400MB to 
load. Deploying PyTorch directly would exhaust free tier memory.

**Current mitigation:**  
ONNX export reduces model to 0.379 MB. The onnxruntime inference 
session uses approximately 80MB total memory. Well within free tier 
limits.

---

## FM-008 — Demo Signal Always Static

**Status:** Known / By design  
**Severity:** Informational  

**Description:**  
The "Generate demo signal" option in the Streamlit demo always produces 
the same synthetic signal using np.random.seed(42). This is intentional 
for reproducibility but means every visitor sees identical demo output.

**Current behaviour:**  
Deterministic output by design. The injected artifact on channel Fp1 
(samples 100-150, amplitude x20) is always identical.

**Note:**  
The model does not currently detect this synthetic artifact because it 
has not been trained. After training this demo will demonstrate real 
artifact detection on the injected signal.

---

## Summary Table

| ID | Description | Severity | Status |
|---|---|---|---|
| FM-001 | Model on random weights | Critical | Unresolved |
| FM-002 | Uniform attention placeholders | Medium | Partial |
| FM-003 | No cross-dataset validation | High | Unresolved |
| FM-004 | Short signal error | Low | Handled |
| FM-005 | Channel name mismatch | Medium | Partial |
| FM-006 | NaN from aggressive filtering | Medium | Handled |
| FM-007 | Free tier memory limits | Low | Mitigated |
| FM-008 | Static demo signal | Info | By design |

---

*Last updated: September 2026*  
*Maintained by: Saumya*  
*Repository: github.com/Saumyaaaaa/neuroqa*
