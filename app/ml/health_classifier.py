"""
Vehicle Health Anomaly Detector — LSTM Autoencoder (replaces old Random Forest).

Model artefacts (from LSTM_Autoencoder_Train_Eval__2_.ipynb):
  models/lstm_autoencoder.pt      — PyTorch state_dict
  models/health_scaler.pkl        — MinMaxScaler fitted on normal training data
  models/health_model_config.pkl  — seq_len, n_features, hidden_dim, latent_dim, feature_cols, threshold
  models/optimized_thresholds.pkl — per-feature thresholds (chosen on tune split)

Input  : list of N dicts (N >= seq_len=24), each dict having the feature keys
         (will use the last `seq_len` ticks automatically).
Output : { is_anomaly, anomaly_score, feature_errors, triggered_features }
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Limit PyTorch to 1 thread (prevents 100% CPU spikes for small batch inference)
torch.set_num_threads(1)

MODEL_DIR = Path(__file__).parent.parent.parent / "models"


# ---------- Reproduce the model architecture (must match training exactly) ----------

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim=64, latent_dim=16, seq_len=24):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.encoder_fc   = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc   = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_fc    = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        latent       = self.encoder_fc(h_n[-1])
        dec_in       = self.decoder_fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _   = self.decoder_lstm(dec_in)
        return self.output_fc(decoded)


# ---------- Load artefacts once at import time ----------

with open(MODEL_DIR / "health_model_config.pkl", "rb") as _f:
    _config = pickle.load(_f)

with open(MODEL_DIR / "health_scaler.pkl", "rb") as _f:
    _scaler = pickle.load(_f)

with open(MODEL_DIR / "optimized_thresholds.pkl", "rb") as _f:
    _per_feature_thresholds: dict = pickle.load(_f)

FEATURE_COLS = _config["feature_cols"]
SEQ_LEN      = _config["seq_len"]       # 24
N_FEATURES   = _config["n_features"]
HIDDEN_DIM   = _config.get("hidden_dim", 64)
LATENT_DIM   = _config.get("latent_dim", 16)
# Global threshold kept as fallback; per-feature thresholds are preferred
_GLOBAL_THRESHOLD = _config.get("threshold", None)

_device = torch.device("cpu")   # CPU is fine for inference; avoids GPU dependency at runtime
_lstm_model = LSTMAutoencoder(N_FEATURES, HIDDEN_DIM, LATENT_DIM, SEQ_LEN).to(_device)
_lstm_model.load_state_dict(
    torch.load(MODEL_DIR / "lstm_autoencoder.pt", map_location=_device)
)
_lstm_model.eval()


# ---------- Public inference function ----------

def _get_short_key(col_name: str) -> str:
    c = col_name.lower()
    if "coolant" in c: return "coolant_temp"
    if "manifold" in c or "pressure" in c: return "map_kpa"
    if "rpm" in c: return "rpm"
    if "speed" in c: return "vss"
    if "intake air" in c: return "intake_air_temp"
    if "mass flow" in c: return "maf"
    if "throttle" in c: return "throttle_pos"
    if "ambient" in c: return "ambient_temp"
    if "pedal position d" in c: return "pedal_d"
    if "pedal position e" in c: return "pedal_e"
    return col_name

def classify_vehicle_health(tick_window: list[dict]) -> dict:
    """
    Runs LSTM Autoencoder anomaly detection on a sliding window of OBD ticks.

    Parameters
    ----------
    tick_window : list of dicts, each dict must contain all keys in FEATURE_COLS.
                  Must have at least SEQ_LEN (24) entries — if more are supplied,
                  only the most recent SEQ_LEN ticks are used.

    Returns
    -------
    {
        "is_anomaly":         bool,
        "anomaly_score":      float,   # max single-feature reconstruction error
        "feature_errors":     { feature_name: float, ... },
        "triggered_features": [ feature_names that exceeded their threshold ],
        "status":             "Normal" | "Anomaly"   # for drop-in compat with old API
    }
    """
    if len(tick_window) < SEQ_LEN:
        raise ValueError(
            f"Need at least {SEQ_LEN} ticks for LSTM inference; got {len(tick_window)}."
        )

    # Take the last SEQ_LEN ticks
    window = tick_window[-SEQ_LEN:]

    # Build numpy array (SEQ_LEN, N_FEATURES)
    raw = np.array(
        [[tick.get(_get_short_key(c), 0.0) for c in FEATURE_COLS] for tick in window],
        dtype=np.float32,
    )

    # Scale with the training MinMaxScaler
    scaled = _scaler.transform(raw)                          # (SEQ_LEN, N_FEATURES)
    X = torch.tensor(scaled[np.newaxis, ...], dtype=torch.float32).to(_device)  # (1, SEQ_LEN, N_FEATURES)

    with torch.no_grad():
        reconstructed = _lstm_model(X)                       # (1, SEQ_LEN, N_FEATURES)
        # Per-feature MAX error across the time dimension (matches training evaluation logic)
        per_feature_error = (
            (reconstructed - X) ** 2
        ).max(dim=1).values.cpu().numpy().flatten()          # (N_FEATURES,)

    feature_errors = {col: float(per_feature_error[i]) for i, col in enumerate(FEATURE_COLS)}

    # Relax the ultra-strict training thresholds to prevent false positives on normal data
    SENSITIVITY_MULTIPLIER = 2.0 

    # Anomaly flag: any feature exceeds its individual threshold
    triggered = [
        col for i, col in enumerate(FEATURE_COLS)
        if per_feature_error[i] > (_per_feature_thresholds.get(col, float("inf")) * SENSITIVITY_MULTIPLIER)
    ]
    is_anomaly = len(triggered) > 0

    return {
        "is_anomaly":         is_anomaly,
        "status":             "Anomaly" if is_anomaly else "Normal",
        "anomaly_score":      round(float(per_feature_error.max()), 6),
        "feature_errors":     {k: round(v, 6) for k, v in feature_errors.items()},
        "triggered_features": triggered,
    }
