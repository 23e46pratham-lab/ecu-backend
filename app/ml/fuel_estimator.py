"""
Fuel Consumption Rate Estimator — BiLSTM + Self-Attention (TensorFlow/Keras).

Model artefacts (from AutoVue_Fuel_LSTM__3_.ipynb):
  models/fuel_lstm_model.keras    — full Keras SavedModel
  models/fuel_feature_scaler.pkl  — MinMaxScaler fitted on training data

Input  : sliding window of 20 OBD ticks, each with:
           rpm, vss, maf, throttle_pos, map_kpa, pedal_d, pedal_e
         (accel, throttle_rate, engine_load are derived here from the window)
Output : { fcr_gs, mileage_kmpl }
         fcr_gs       — fuel consumption rate in g/s
         mileage_kmpl — instantaneous mileage in km/L (None if stationary)
"""

import pickle
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent.parent / "models"

# ---------- Custom Keras layer (must be defined before model load) ----------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Limit TensorFlow to 1 thread (prevents 100% CPU spikes for small batch inference)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

class ScaledDotProductAttention(layers.Layer):
    """Single-head scaled dot-product self-attention — matches training architecture."""

    def build(self, input_shape):
        F = input_shape[-1]
        self.W_q    = self.add_weight(shape=(F, F), initializer="glorot_uniform", trainable=True, name="W_q")
        self.W_k    = self.add_weight(shape=(F, F), initializer="glorot_uniform", trainable=True, name="W_k")
        self.W_v    = self.add_weight(shape=(F, F), initializer="glorot_uniform", trainable=True, name="W_v")
        self.scale  = tf.sqrt(tf.cast(F, tf.float32))
        super().build(input_shape)

    def call(self, x):
        Q       = tf.matmul(x, self.W_q)
        K       = tf.matmul(x, self.W_k)
        V       = tf.matmul(x, self.W_v)
        scores  = tf.matmul(Q, K, transpose_b=True) / self.scale
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, V)
        return tf.reduce_mean(context, axis=1)

    def get_config(self):
        return super().get_config()


# ---------- Constants (must match training notebook exactly) ----------
FEATURE_COLS = [
    "rpm", "vss", "maf", "throttle_pos", "map_kpa",
    "pedal_d", "pedal_e",
    "accel",          # derived: Δvss per tick
    "throttle_rate",  # derived: Δthrottle_pos per tick
    "engine_load",    # derived: rpm / (vss + 1)
]
WINDOW_SIZE       = 20
FUEL_DENSITY      = 730.0   # g/L  (Indian BS6 E10 petrol)
MIN_SPEED_KMPH    = 2.0     # below this, mileage is undefined


# ---------- Load artefacts once at import time ----------
_fuel_model = keras.models.load_model(
    MODEL_DIR / "fuel_lstm_model.keras",
    custom_objects={"ScaledDotProductAttention": ScaledDotProductAttention},
)

with open(MODEL_DIR / "fuel_feature_scaler.pkl", "rb") as _f:
    _fuel_scaler = pickle.load(_f)


# ---------- Feature engineering ----------

def _engineer_features(tick_window: list[dict]) -> np.ndarray:
    """
    Adds derived features (accel, throttle_rate, engine_load) to the raw ticks
    and returns a (WINDOW_SIZE, 10) float32 array ready for the model.
    """
    raw_keys = ["rpm", "vss", "maf", "throttle_pos", "map_kpa", "pedal_d", "pedal_e"]

    # Base sensor matrix (window, 7)
    base = np.array([[t.get(k, 0.0) for k in raw_keys] for t in tick_window], dtype=np.float64)

    vss         = base[:, 1]
    throttle    = base[:, 3]
    rpm         = base[:, 0]

    accel          = np.diff(vss,    prepend=vss[0])
    throttle_rate  = np.diff(throttle, prepend=throttle[0])
    engine_load    = rpm / (vss + 1.0)

    accel         = np.clip(accel,         -20,   20)
    throttle_rate = np.clip(throttle_rate, -50,   50)
    engine_load   = np.clip(engine_load,     0, 5000)

    derived = np.column_stack([accel, throttle_rate, engine_load])  # (window, 3)
    full    = np.hstack([base, derived])                              # (window, 10)

    return full.astype(np.float32)


# ---------- Public inference function ----------

def estimate_fuel(tick_window: list[dict]) -> dict:
    """
    Estimates real-time fuel consumption rate and mileage.

    Parameters
    ----------
    tick_window : list of dicts, each with keys:
                  rpm, vss, maf, throttle_pos, map_kpa, pedal_d, pedal_e
                  Must have at least WINDOW_SIZE (20) entries.
                  Only the most recent 20 ticks are used.

    Returns
    -------
    {
        "fcr_gs":       float | None,   # fuel consumption rate in g/s
        "mileage_kmpl": float | None,   # km/L (None if vss < MIN_SPEED_KMPH)
        "vss_kmph":     float           # speed of the last tick used
    }
    """
    if len(tick_window) < WINDOW_SIZE:
        raise ValueError(
            f"Need at least {WINDOW_SIZE} ticks for fuel inference; got {len(tick_window)}."
        )

    window = tick_window[-WINDOW_SIZE:]
    features = _engineer_features(window)                        # (20, 10)
    scaled   = _fuel_scaler.transform(features)                  # (20, 10)
    X        = scaled[np.newaxis, ...]                           # (1, 20, 10)

    pred_fcr = float(_fuel_model.predict(X, verbose=0)[0][0])
    pred_fcr = max(0.0, pred_fcr)                                # FCR cannot be negative

    last_vss = float(window[-1].get("vss", 0.0))

    if last_vss > MIN_SPEED_KMPH:
        vol_flow_ls = pred_fcr / FUEL_DENSITY                   # L/s
        mileage     = (last_vss / 3600.0) / (vol_flow_ls + 1e-8)  # km/L
        mileage     = min(float(mileage), 50.0)                  # cap at 50 km/L
    else:
        mileage = None

    return {
        "fcr_gs":       round(pred_fcr, 4),
        "mileage_kmpl": round(mileage, 2) if mileage is not None else None,
        "vss_kmph":     round(last_vss, 1),
    }
