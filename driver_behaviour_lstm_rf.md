# Driver Behavior Profiling — LSTM + Random Forest Pipeline

## Overview
The Driver Behavior Profiling module has been redesigned from a single-stage **K-Means clustering** approach to a **two-stage hybrid pipeline**:

1. **LSTM (Labelling Stage)** — consumes a raw time-series window of driving telemetry and outputs a cluster-like pseudo-label representing the underlying driving pattern in that window (replacing the role K-Means previously played).
2. **Random Forest (Classification Stage)** — takes the engineered statistical features (same features used in the original K-Means version: RPM std, RPM MAD, speed MAD, acceleration std/range, throttle std) **plus** the LSTM's pseudo-label as an additional input feature, and classifies the window into the final driver behavior category: **Economical / Moderate / Harsh**.

This replaces the old single-model flow (`raw window → engineered features → K-Means → cluster label`) with a two-model flow where the LSTM contributes a learned, sequence-aware signal that the RF then combines with the existing hand-crafted features to make the final call.

---

## Pipeline Diagram

```
┌────────────────────┐
│  Raw OBD-II Window  │
│  (RPM, Speed,       │
│   Throttle values)  │
└─────────┬───────────┘
          │
          ├─────────────────────────────┐
          ▼                             ▼
┌───────────────────┐         ┌───────────────────────┐
│   LSTM Model        │         │  Feature Engineering  │
│   (sequence input)  │         │  (existing formulas:  │
│                      │         │  RPM_std, RPM_mad,    │
│  Outputs a           │        │  speed_mad,            │
│  pseudo cluster       │       │  accel_std/range,      │
│  label per window      │      │  throttle_std)          │
└─────────┬────────────┘         └───────────┬────────────┘
          │                                   │
          │           pseudo_label            │  engineered_features
          └────────────────┬──────────────────┘
                            ▼
                ┌───────────────────────┐
                │   Random Forest         │
                │   Classifier             │
                │                           │
                │   Input: engineered       │
                │   features + LSTM         │
                │   pseudo_label             │
                │                             │
                │   Output: Economical /      │
                │   Moderate / Harsh            │
                └───────────────────────────────┘
```

---

## Stage 1: LSTM (Labelling)

**Purpose:** Replace K-Means as the source of the underlying driving-pattern label. Instead of clustering purely on hand-crafted statistical features, the LSTM reads the raw sequential telemetry directly, so it can pick up on temporal patterns (e.g. how acceleration/braking evolves *within* a window) that summary statistics alone lose.

**Input:** Raw time-series window — same underlying signals as before (RPM, Speed, Throttle) at their native per-timestep resolution, not the summarized features.

**Output:** A pseudo-label per window — analogous to a K-Means cluster ID, but produced by the LSTM. This label is *not* the final behavior class; it's an intermediate, learned representation of "which driving pattern this window resembles."

**Role in the pipeline:** Functions as a drop-in replacement for the K-Means step, but instead of directly mapping to Economical/Moderate/Harsh via a fixed `CLUSTER_MAP`, its output is passed forward as one more input to the Random Forest.

---

## Stage 2: Random Forest (Classification)

**Purpose:** Produce the final Economical / Moderate / Harsh classification.

**Input features:**
- All the original engineered features from the K-Means-era pipeline:
  - `Engine RPM [RPM]_std`
  - `Engine RPM [RPM]_mad`
  - `Vehicle Speed Sensor [km/h]_mad`
  - `acceleration_std`
  - `acceleration_range`
  - `Absolute Throttle Position [%]_std`
- Plus the **LSTM pseudo-label** for the same window, added as an extra feature column.

**Output:** Final classification — `Economical`, `Moderate`, or `Harsh` — along with (typically) class probabilities/confidence, consistent with how the existing `health_classifier.py` RF module reports `probabilities` and `confidence`.

**Why RF on top of LSTM instead of LSTM → label directly:** The RF acts as a second, interpretable decision layer that can weigh the LSTM's sequence-derived signal against the existing, already-validated statistical features — rather than fully trusting the LSTM's raw cluster assignment as the final answer.

---

## Comparison to Previous (K-Means-only) Approach

| Aspect | Old: K-Means | New: LSTM + RF |
| --- | --- | --- |
| Input to clustering/labelling | Engineered features only | Raw sequential window |
| Captures temporal/sequential patterns | No (summary stats only) | Yes (LSTM reads the sequence) |
| Final classification step | Direct cluster → class map (`CLUSTER_MAP`) | RF trained classifier using engineered features + LSTM label |
| Number of models | 1 (KMeans + scaler) | 2 (LSTM + RF), each with its own preprocessing |
| Output | Cluster ID → mapped class | Class + confidence/probabilities (RF-native) |

---

## Expected Artifacts (for backend integration)

Following the existing pattern in `app/ml/` (`driver_behaviour.py`, `health_classifier.py`, `anomaly_lstm.py`), this pipeline will need model artifacts saved to `models/`, e.g.:

```
models/lstm_driver_behavior.keras         # LSTM labelling model
models/lstm_driver_behavior_scaler.joblib # scaler for raw window inputs to LSTM
models/rf_driver_behavior.joblib          # Random Forest classifier
models/rf_driver_behavior_scaler.joblib   # scaler for RF's engineered-feature + label input
```

*(Exact filenames should be adjusted to match whatever the training script actually saves — these are illustrative, following the naming convention already used elsewhere in the project.)*

## Open Items
- Confirm the exact output type of the LSTM stage (integer pseudo-cluster ID vs. embedding vector) — this doc assumes a single pseudo-label ID, consistent with "cluster-like pseudo-labels."
- Confirm RF training used the LSTM label as a *categorical* feature (may need one-hot encoding) vs. a numeric feature, since this affects the RF's expected input shape at inference time.
- Update `app/ml/driver_behaviour.py` (or add a new module) once training scripts and artifacts are finalized, following the same load-model/scaler-at-import pattern used in `health_classifier.py`.
