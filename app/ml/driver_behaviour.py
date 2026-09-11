"""
Driver Behaviour Classifier — XGBoost (replaces old KMeans model).

Model artefacts (from DriverBehavior_XGBoost_012.ipynb, Cell 10):
  models/driver_model.pkl       — trained XGBoost / RF / SVM classifier
  models/driver_scaler.pkl      — StandardScaler (used only when needs_scaling=True)
  models/driver_metadata.json   — feature list, label names, TTS messages, thresholds

Input  : raw OBD window arrays (rpm, speed, throttle/pedal) — same API shape as before,
         plus an optional `window_index` parameter (defaults to 0).
Output : { label, confidence, tts_message, feature_values }
         label is one of: "Economical" | "Moderate" | "Aggressive"
"""

import json
import numpy as np
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent.parent / "models"

# ---------- Load artefacts once at import time ----------
_model    = joblib.load(MODEL_DIR / "driver_model.pkl")
_scaler   = joblib.load(MODEL_DIR / "driver_scaler.pkl")
with open(MODEL_DIR / "driver_metadata.json") as _f:
    _meta = json.load(_f)

FEATURE_COLS   = _meta["feature_cols"]       # e.g. ['window','avg_speed','vs_dev',...]
LABEL_NAMES    = _meta["label_names"]        # ['Aggressive','Economical','Moderate']
TTS_MESSAGES   = _meta["tts_messages"]
NEEDS_SCALING  = _meta["needs_scaling"]      # True only for SVM/AdaBoost

# Thresholds from training (used to compute window features)
_thresh = _meta["thresholds"]
HARSH_ACCEL_G        = _thresh["HARSH_ACCEL_G"]
HARSH_BRAKE_G        = _thresh["HARSH_BRAKE_G"]
HIGH_RPM_THRESHOLD   = _thresh["HIGH_RPM_THRESHOLD"]
HARSH_PEDAL_JUMP     = _thresh["HARSH_PEDAL_JUMP"]


# ---------- Feature engineering (mirrors notebook Cell 5 / Cell 7) ----------

def _compute_window_features(
    rpm: list[float],
    speed: list[float],
    pedal: list[float],
    window_index: int = 0,
) -> dict:
    """
    Convert raw OBD arrays into the 9 window-level features the model expects.
    Matches the feature engineering in the training notebook exactly.
    """
    rpm_arr   = np.array(rpm,   dtype=float)
    spd_arr   = np.array(speed, dtype=float)
    ped_arr   = np.array(pedal, dtype=float)

    # Acceleration in g (from speed in km/h → m/s, assume 1 Hz → dt=1s)
    speed_ms  = spd_arr / 3.6
    smoothed  = np.convolve(speed_ms, np.ones(min(10, len(speed_ms))) / min(10, len(speed_ms)), mode="same")
    dt        = np.ones(len(smoothed))          # 1 s per tick (simulator default)
    accel_g   = np.clip(np.gradient(smoothed) / (dt * 9.81), -3, 3)

    return {
        "window":      float(window_index),
        "avg_speed":   float(np.mean(spd_arr)),
        "vs_dev":      float(np.std(spd_arr)),
        "mean_rpm":    float(np.mean(rpm_arr)),
        "rpm_std":     float(np.std(rpm_arr)),
        "mean_pedal":  float(np.mean(ped_arr)),
        "pedal_std":   float(np.std(ped_arr)),
        "max_speed":   float(np.max(spd_arr)),
        "accel_std":   float(np.std(accel_g)),
    }


# ---------- Public inference function ----------

def predict_from_raw_window(
    rpm_values:     list[float],
    speed_values:   list[float],
    throttle_values: list[float],
    window_index:   int = 0,
) -> dict:
    """
    Classifies driver behaviour from a rolling window of OBD readings.

    Parameters
    ----------
    rpm_values      : engine RPM time-series for the window
    speed_values    : vehicle speed (km/h) time-series
    throttle_values : throttle / accelerator pedal (%) time-series
                      (used as the pedal proxy — same column the model was trained on)
    window_index    : optional monotonic window counter (default 0)

    Returns
    -------
    {
        "label":          "Economical" | "Moderate" | "Aggressive",
        "confidence":     0.0–1.0,
        "tts_message":    str,          # ready-to-speak TTS string
        "feature_values": { ... }       # the 9 engineered features (for debugging)
    }
    """
    feat_dict = _compute_window_features(rpm_values, speed_values, throttle_values, window_index)
    X = np.array([[feat_dict[c] for c in FEATURE_COLS]])

    if NEEDS_SCALING:
        X = _scaler.transform(X)

    y_pred = int(_model.predict(X)[0])
    label  = LABEL_NAMES[y_pred]

    if hasattr(_model, "predict_proba"):
        proba      = _model.predict_proba(X)[0]
        confidence = float(proba[y_pred])
    else:
        confidence = 1.0

    return {
        "label":          label,
        "confidence":     round(confidence, 4),
        "tts_message":    TTS_MESSAGES.get(label, label),
        "feature_values": feat_dict,
    }