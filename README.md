# AutoVue Backend

**ML-powered vehicle diagnostics API + OBD-II data simulator, in one deployable FastAPI service.**

Part of the *AutoVue — Smart Vehicle ECU Monitoring and Predictive Maintenance* major project (VTU, Dept. of ICBS, St Joseph Engineering College).

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-337AB7.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/license-Academic--Project-lightgrey.svg)](#license)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [ML Models](#ml-models)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
  - [Machine Learning Endpoints](#machine-learning-endpoints)
  - [Simulator — Dataset Management](#simulator--dataset-management)
  - [Simulator — Playback Control](#simulator--playback-control)
  - [Simulator — Live Data & ML](#simulator--live-data--ml)
  - [Simulator — WebSockets](#simulator--websockets)
  - [System](#system)
- [Standard Sensor Field Reference](#standard-sensor-field-reference)
- [Model Files Reference](#model-files-reference)
- [Testing POST Endpoints](#testing-post-endpoints)
- [Deployment](#deployment)
- [Extending the System](#extending-the-system)
- [Roadmap](#roadmap)

---

## Overview

This service exposes three things behind a single base URL:

1. **Machine learning inference** — three specialised models for driver
   behaviour classification, vehicle health anomaly detection, and fuel
   consumption estimation, all running on live OBD-II data.
2. **An OBD-II simulator** — plays back a recorded vehicle dataset row by
   row, in real time, over REST and WebSockets, so a frontend or ML
   pipeline can be built and demoed without a physical vehicle or adapter.
3. **Real-time ML integration** — the simulator automatically runs all
   three ML models on every tick and pushes inference results alongside
   raw sensor data over WebSocket.

Both the ML API and the simulator are independently useful and independently
testable, but ship as one process so the whole project has a single
deployment and a single URL.

## Architecture

```
                    ┌──────────────────────────────────┐
                    │          FastAPI App v2.0.0       │
                    │          (app/main.py)            │
                    └───────────────┬──────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                            │
    ┌─────────▼──────────┐                      ┌──────────▼──────────┐
    │       ML API        │                      │   OBD-II Simulator   │
    │     (app/ml/)       │                      │  (app/simulator/)    │
    │                     │                      │                      │
    │ • XGBoost driver    │    ML functions are   │ • Dataset loader     │
    │   behaviour         │◄───imported by the───►│ • Playback engine    │
    │ • LSTM Autoencoder  │    simulator to run   │ • REST control API   │
    │   health anomaly    │    inference on each   │ • WebSocket stream   │
    │ • Physics fuel      │    tick automatically  │ • Rolling ML buffer  │
    │   estimator (3-tier)│                      │                      │
    └─────────────────────┘                      └──────────────────────┘
```

The simulator imports the three ML inference functions and runs them
automatically on a rolling buffer of ticks. Results are attached to
every WebSocket message under the `ml` key, so frontends get raw sensor
data **and** ML predictions in a single push with zero extra API calls.

## ML Models

| Model | Architecture | Input | Output | Training Notebook |
|---|---|---|---|---|
| **Driver Behaviour** | XGBoost (Random Forest) | Rolling window of RPM, speed, throttle arrays (≥5 points) | `Economical` / `Moderate` / `Aggressive` + confidence + TTS message | `DriverBehavior_XGBoost_012.ipynb` |
| **Vehicle Health** | LSTM Autoencoder (PyTorch) | Sequence of 24 OBD-II ticks (10 sensor features each) | Anomaly flag + per-feature reconstruction errors + triggered features | `LSTM_Autoencoder_Train_Eval.ipynb` |
| **Fuel Estimation** | Physics-based (MAF→MAP+RPM→throttle heuristic) | Last 3 ticks averaged (RPM, VSS, MAF, MAP, IAT, throttle) | FCR (g/s) + mileage (km/L) + tier used | No training required |

## Project Structure

```
ecu-backend/
├── app/
│   ├── main.py                      # Entry point — mounts ML + simulator, warning filters
│   ├── ml/
│   │   ├── driver_behaviour.py      # XGBoost driver behaviour classifier
│   │   ├── health_classifier.py     # LSTM Autoencoder vehicle health anomaly detector
│   │   └── fuel_estimator.py        # BiLSTM + Attention fuel consumption estimator
│   ├── simulator/
│   │   ├── config.py                # Simulator settings (env-var overridable)
│   │   ├── services/
│   │   │   ├── dataset_manager.py   # Loads/cleans CSV & XLSX OBD-II datasets
│   │   │   └── simulator.py         # Playback engine + rolling ML buffer + inference hooks
│   │   └── api/
│   │       ├── routes.py            # REST endpoints for datasets + playback + ML polling
│   │       └── websocket.py         # WebSocket connection manager
│   └── static/
│       └── dashboard.html           # Built-in control dashboard (single file, no build step)
├── datasets/                        # Bundled sample OBD-II datasets
├── uploads/                         # User-uploaded datasets (gitignored)
├── models/                          # Trained model artifacts (see Model Files Reference)
├── requirements.txt
├── Dockerfile
├── run.py                           # Local dev entrypoint
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.11+
- pip

### Installation
```bash
git clone <your-repo-url>
cd ecu-backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run
```bash
python run.py
# equivalently: uvicorn app.main:app --reload --port 8000
```

### Explore
| What | Where |
|---|---|
| Simulator control dashboard | http://localhost:8000 |
| Interactive API docs (Swagger) | http://localhost:8000/docs |
| Alternative API docs (ReDoc) | http://localhost:8000/redoc |
| Health check | http://localhost:8000/health |

A sample dataset is bundled in `datasets/` and loads automatically at
startup — the simulator works out of the box with zero configuration.

---

## API Reference

Base URL (local): `http://localhost:8000`
Base URL (deployed): `https://<your-app>.onrender.com`

### Machine Learning Endpoints

#### `POST /api/driver/predict`
Classifies driver behaviour from a rolling window of OBD-II readings using a pre-trained XGBoost model. The model evaluates **driving smoothness** — variance in speed, RPM, pedal position, and acceleration — rather than absolute sensor values.

**Request body**
```json
{
  "rpm_values": [800, 850, 900, 1200, 1500, 1400, 1300],
  "speed_values": [0, 0, 5, 20, 35, 40, 38],
  "throttle_values": [10, 12, 15, 30, 45, 40, 35],
  "window_index": 0
}
```
- `rpm_values`, `speed_values`, `throttle_values` — time-series arrays, minimum 5 data points, all same length.
- `window_index` — optional monotonic counter (default `0`), used as one of the 9 engineered features.

**Response `200`**
```json
{
  "label": "Moderate",
  "confidence": 0.5035,
  "tts_message": "Your driving is acceptable. Watch your speed variations and try to accelerate more gently.",
  "feature_values": {
    "window": 0,
    "avg_speed": 19.71,
    "vs_dev": 17.51,
    "mean_rpm": 1135.71,
    "rpm_std": 262.83,
    "mean_pedal": 26.71,
    "pedal_std": 13.22,
    "max_speed": 40,
    "accel_std": 0.183
  }
}
```
- `label` — one of `"Economical"`, `"Moderate"`, `"Aggressive"`.
- `confidence` — model probability for the predicted class (0.0–1.0).
- `tts_message` — human-readable sentence suitable for text-to-speech.
- `feature_values` — the 9 engineered features sent to the model (useful for debugging).

**Errors**
| Code | Cause |
|---|---|
| `400` | Fewer than 5 data points supplied |
| `500` | Internal model/feature-extraction error |

---

#### `POST /api/health/predict`
Detects anomalies in vehicle health using an LSTM Autoencoder. The model learns the **normal correlations** between all 10 sensor features — when the relationship between sensors breaks (e.g. high RPM but low airflow), it flags an anomaly even if no single sensor is out of range.

**Request body**
```json
{
  "ticks": [
    {
      "rpm": 798, "vss": 0, "maf": 8.74, "throttle_pos": 83.1,
      "map_kpa": 97, "coolant_temp": 33, "intake_air_temp": 32,
      "ambient_temp": 24, "pedal_d": 14.1, "pedal_e": 14.5
    }
  ]
}
```
- `ticks` — array of sensor snapshots. **Must have ≥ 24 items** (the LSTM sequence length). If more are provided, only the most recent 24 are used.
- All fields default to `0.0` if omitted.

**Response `200` — Normal**
```json
{
  "is_anomaly": false,
  "status": "Normal",
  "anomaly_score": 0.000012,
  "feature_errors": {
    "Engine Coolant Temperature [°C]": 0.000003,
    "Intake Manifold Absolute Pressure [kPa]": 0.000008,
    "Engine RPM [RPM]": 0.000005,
    "Vehicle Speed Sensor [km/h]": 0.000002,
    "Intake Air Temperature [°C]": 0.000004,
    "Air Flow Rate from Mass Flow Sensor [g/s]": 0.000007,
    "Absolute Throttle Position [%]": 0.000001,
    "Ambient Air Temperature [°C]": 0.000003,
    "Accelerator Pedal Position D [%]": 0.000012,
    "Accelerator Pedal Position E [%]": 0.000009
  },
  "triggered_features": []
}
```

**Response `200` — Anomaly detected**
```json
{
  "is_anomaly": true,
  "status": "Anomaly",
  "anomaly_score": 3.945678,
  "feature_errors": {
    "Engine RPM [RPM]": 3.945678,
    "Air Flow Rate from Mass Flow Sensor [g/s]": 0.86078,
    "Vehicle Speed Sensor [km/h]": 0.231513
  },
  "triggered_features": [
    "Engine RPM [RPM]",
    "Air Flow Rate from Mass Flow Sensor [g/s]",
    "Vehicle Speed Sensor [km/h]"
  ]
}
```
- `is_anomaly` — `true` if any feature's reconstruction error exceeds its per-feature threshold.
- `anomaly_score` — the highest single-feature error (useful for severity ranking).
- `feature_errors` — per-feature reconstruction error (higher = more anomalous).
- `triggered_features` — list of sensor names that exceeded their individual thresholds.

**Errors**
| Code | Cause |
|---|---|
| `400` | Fewer than 24 ticks supplied |
| `500` | Internal model error |

---

#### `POST /api/fuel/predict`
Three-tier physics engine. Tier 1: MAF-based (FCR = MAF/14.7).
Tier 2: Speed-density equation (MAP + RPM + IAT). Tier 3: throttle+RPM
heuristic. AFR=14.7, fuel density=0.730 g/mL (BS6 E10), displacement=1.598L
(Seat Leon 1.6 FSI). Mileage capped at 50 km/L.

| Tier | Method | Condition |
|---|---|---|
| 1 | MAF-based | MAF > 0.5 g/s AND RPM > 200 |
| 2 | Speed-density | MAP present AND RPM > 200 |
| 3 | Throttle heuristic | RPM > 200 (fallback) |
| 0 | Engine off | RPM ≤ 200 |

**Request body**
```json
{
  "ticks": [
    {
      "rpm": 1500, "vss": 45, "maf": 12.5, "throttle_pos": 30,
      "map_kpa": 97, "pedal_d": 20, "pedal_e": 18
    }
  ]
}
```
- `ticks` — array of sensor snapshots. **Must have ≥ 20 items** (the model's window size). If more are provided, only the most recent 20 are used.
- All fields default to `0.0` if omitted.

**Response `200`**
```json
{
  "fcr_gs": 1.2345,
  "mileage_kmpl": 14.2,
  "vss_kmph": 45.0,
  "method": "maf",
  "tier": 1
}
```
- `fcr_gs` — fuel consumption rate in grams per second.
- `mileage_kmpl` — instantaneous mileage in km/L. Returns `null` if the vehicle speed is below 2 km/h (stationary/crawling).
- `vss_kmph` — the speed of the last tick used (for reference).
- `method` — the tier method used (`maf`, `map_rpm`, `throttle_heuristic`, `stopped`).
- `tier` — the integer tier level (0-3).

**Errors**
| Code | Cause |
|---|---|
| `400` | Fewer than 20 ticks supplied |
| `500` | Internal model error |

---

### Simulator — Dataset Management

#### `GET /api/datasets`
Lists every dataset available on disk (bundled + uploaded), with cleaning metadata.

**Response `200`**
```json
{
  "datasets": [
    {
      "dataset_id": "575a02c5-501c-44d0-b487-be107fce2957",
      "filename": "2017-07-05_Seat_Leon_S_KA_Normal.csv",
      "row_count": 30817,
      "duration_seconds": 3080.3,
      "missing_value_report": {
        "coolant_temp": 30668,
        "rpm": 1,
        "vss": 2
      }
    }
  ]
}
```

#### `POST /api/upload`
Uploads a new dataset. Accepts `.csv`, `.xlsx`, `.xls`. `multipart/form-data`, field name `file`.

```bash
curl -X POST http://localhost:8000/api/upload -F "file=@my_dataset.csv"
```

**Response `201`**
```json
{
  "dataset_id": "a1b2c3d4-...",
  "filename": "my_dataset.csv",
  "row_count": 5000,
  "duration_seconds": 500.0,
  "missing_value_report": { "coolant_temp": 12 }
}
```
**Errors:** `400` if the file extension is unsupported or the file can't be parsed.

#### `DELETE /api/datasets/{dataset_id}`
Deletes an uploaded dataset. Bundled sample datasets are protected — they're removed from
the in-memory cache but never deleted from disk.

**Response `200`:** `{"deleted": true}` · **Errors:** `404` if `dataset_id` doesn't exist.

#### `PATCH /api/datasets/{dataset_id}/rename`
```json
{ "dataset_id": "a1b2c3d4-...", "new_name": "highway_run_2" }
```
**Response `200`:** `{"renamed": true}` · **Errors:** `404` if not found or rename fails.

#### `POST /api/change-dataset`
Switches the active dataset. If playback is currently running, it restarts from row 0 on the new dataset.
```json
{ "dataset_id": "a1b2c3d4-..." }
```
**Response `200`:** `{"active_dataset_id": "a1b2c3d4-..."}`

---

### Simulator — Playback Control

All endpoints below return the same **status object** shape:
```json
{
  "state": "running",
  "dataset_id": "575a02c5-...",
  "dataset_name": "2017-07-05_Seat_Leon_S_KA_Normal.csv",
  "current_row": 105,
  "total_rows": 30817,
  "speed": 10.0,
  "loop": true,
  "elapsed_playback_seconds": 1049.0,
  "dataset_duration_seconds": 3080.3,
  "playback_percent": 0.34
}
```
`state` is one of: `stopped` · `running` · `paused` · `finished`

| Endpoint | Body | Notes |
|---|---|---|
| `POST /api/start` | `{"dataset_id"?, "speed"?, "loop"?}` (all optional) | Starts streaming from row 0. Loads the default dataset if none is active. `400` if no dataset is available. |
| `POST /api/pause` | — | No-op unless currently `running`. |
| `POST /api/resume` | — | No-op unless currently `paused`. |
| `POST /api/stop` | — | Cancels the playback task entirely. Clears the ML buffer. |
| `POST /api/reset` | — | Stops and rewinds to row 0. |
| `POST /api/speed` | `{"speed": 5}` | Must be one of `0.5, 1, 2, 5, 10`. `400` otherwise. |
| `POST /api/loop` | `{"loop": true}` | Toggle looping when the dataset ends. |
| `GET /api/status` | — | Current status object (also polled by the dashboard every 2s as a WebSocket backstop). |

---

### Simulator — Live Data & ML

#### `GET /api/live-data`
Returns the most recent telemetry tick, including ML inference results.

**Response `200`**
```json
{
  "row_index": 105,
  "total_rows": 30817,
  "elapsed_seconds": 1049.0,
  "playback_percent": 0.34,
  "data": {
    "coolant_temp": 33.0,
    "map_kpa": 97.0,
    "rpm": 798.0,
    "vss": 0.0,
    "intake_air_temp": 32.0,
    "maf": 8.74,
    "throttle_pos": 83.1,
    "ambient_temp": 24.0,
    "pedal_d": 14.1,
    "pedal_e": 14.5
  },
  "ml": {
    "driver_behaviour": { "..." },
    "health": { "..." },
    "fuel": { "..." }
  }
}
```
**Errors:** `404` if the simulator hasn't been started yet.

#### `GET /api/ml/latest`
Returns only the ML inference results from the most recent simulator tick — useful for polling without needing a WebSocket.

**Response `200`**
```json
{
  "driver_behaviour": {
    "label": "Economical",
    "confidence": 0.87,
    "tts_message": "Great driving! You are being smooth and fuel-efficient. Keep it up.",
    "feature_values": { "..." }
  },
  "health": {
    "is_anomaly": false,
    "status": "Normal",
    "anomaly_score": 0.000012,
    "feature_errors": { "..." },
    "triggered_features": []
  },
  "fuel": {
    "fcr_gs": 1.2345,
    "mileage_kmpl": 14.2,
    "vss_kmph": 45.0,
    "method": "maf",
    "tier": 1
  }
}
```
Returns `{}` if the simulator hasn't accumulated enough ticks for the ML models yet (first ~30 ticks).

**Errors:** `404` if the simulator hasn't been started yet.

#### `GET /api/history?limit=100`
Returns the most recent `limit` ticks (default 100, ring buffer capped at 500 — see `HISTORY_BUFFER_SIZE`). Each entry includes the `ml` key. Useful for backfilling a chart on page load before the WebSocket starts pushing new points.

---

### Simulator — WebSockets

#### `WS /api/ws/live`
Pushes a telemetry tick every time the simulator advances a row. After the ML buffer fills (~30 ticks), each message includes a top-level `ml` key with real-time inference results from all three models.

**Message shape (after buffer fills)**
```json
{
  "row_index": 105,
  "total_rows": 30817,
  "elapsed_seconds": 1049.0,
  "playback_percent": 0.34,
  "data": {
    "coolant_temp": 33.0,
    "map_kpa": 97.0,
    "rpm": 798.0,
    "vss": 0.0,
    "intake_air_temp": 32.0,
    "maf": 8.74,
    "throttle_pos": 83.1,
    "ambient_temp": 24.0,
    "pedal_d": 14.1,
    "pedal_e": 14.5
  },
  "ml": {
    "driver_behaviour": {
      "label": "Economical",
      "confidence": 0.87,
      "tts_message": "Great driving! You are being smooth and fuel-efficient. Keep it up.",
      "feature_values": {
        "window": 42,
        "avg_speed": 5.53,
        "vs_dev": 2.82,
        "mean_rpm": 1022.06,
        "rpm_std": 98.04,
        "mean_pedal": 83.36,
        "pedal_std": 0.18,
        "max_speed": 10,
        "accel_std": 0.011
      }
    },
    "health": {
      "is_anomaly": false,
      "status": "Normal",
      "anomaly_score": 0.000012,
      "feature_errors": {
        "Engine Coolant Temperature [°C]": 0.000003,
        "Intake Manifold Absolute Pressure [kPa]": 0.000008,
        "Engine RPM [RPM]": 0.000005,
        "Vehicle Speed Sensor [km/h]": 0.000002,
        "Intake Air Temperature [°C]": 0.000004,
        "Air Flow Rate from Mass Flow Sensor [g/s]": 0.000007,
        "Absolute Throttle Position [%]": 0.000001,
        "Ambient Air Temperature [°C]": 0.000003,
        "Accelerator Pedal Position D [%]": 0.000012,
        "Accelerator Pedal Position E [%]": 0.000009
      },
      "triggered_features": []
    },
    "fuel": {
      "fcr_gs": 1.2345,
      "mileage_kmpl": 14.2,
      "vss_kmph": 45.0,
      "method": "maf",
      "tier": 1
    }
  }
}
```

**Buffer warm-up:** The `ml` key is `{}` for the first ~30 ticks while the rolling buffer fills. Each model activates at its minimum buffer size:
- **Fuel estimation** — tick 20+
- **Health anomaly detection** — tick 24+
- **Driver behaviour** — tick 30+

**Example usage**
```js
const ws = new WebSocket("wss://<your-app>/api/ws/live");
ws.onmessage = (event) => {
  const tick = JSON.parse(event.data);
  console.log("RPM:", tick.data.rpm);

  if (tick.ml?.health) {
    console.log("Health:", tick.ml.health.status);
  }
  if (tick.ml?.fuel) {
    console.log("Mileage:", tick.ml.fuel.mileage_kmpl, "km/L");
  }
  if (tick.ml?.driver_behaviour) {
    console.log("Driver:", tick.ml.driver_behaviour.label);
  }
};
```

#### `WS /api/ws/logs`
Pushes simulator log events (dataset loaded, streaming started/paused, warnings, errors) as they happen.
```json
{ "level": "info", "message": "Streaming started (speed=10.0x, loop=true)", "timestamp": 1751900000.123 }
```

---

### System

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check (used by hosting platforms to detect a live instance) |
| `GET /` | Serves the simulator control dashboard |
| `GET /docs` | Swagger UI — interactive, supports "Try it out" for every endpoint |
| `GET /redoc` | ReDoc — clean read-only API reference |

---

## Standard Sensor Field Reference

Every simulator tick — from the WebSocket, `/api/live-data`, and
`/api/history` — uses these exact keys, regardless of what the original
dataset's column headers were called:

| Field | Meaning | Unit |
|---|---|---|
| `coolant_temp` | Engine coolant temperature | °C |
| `map_kpa` | Intake manifold absolute pressure | kPa |
| `rpm` | Engine RPM | RPM |
| `vss` | Vehicle speed | km/h |
| `intake_air_temp` | Intake air temperature | °C |
| `maf` | Mass air flow | g/s |
| `throttle_pos` | Absolute throttle position | % |
| `ambient_temp` | Ambient air temperature | °C |
| `pedal_d` | Accelerator pedal position D | % |
| `pedal_e` | Accelerator pedal position E | % |

Keep any ML inference or dashboard code pointed at these names — it's the
contract that lets you swap datasets, and later swap in real OBD-II
hardware, without touching downstream code.

---

## Model Files Reference

All model artifacts live in the `models/` directory:

| File | Purpose | Source |
|---|---|---|
| `driver_model.pkl` | XGBoost/RF driver behaviour classifier | `DriverBehavior_XGBoost_012.ipynb` (Cell 10) |
| `driver_scaler.pkl` | StandardScaler for driver features | `DriverBehavior_XGBoost_012.ipynb` (Cell 10) |
| `driver_metadata.json` | Feature list, label names, TTS messages, thresholds | `DriverBehavior_XGBoost_012.ipynb` (Cell 10) |
| `lstm_autoencoder.pt` | PyTorch LSTM Autoencoder weights | `LSTM_Autoencoder_Train_Eval.ipynb` (Cell 14) |
| `health_scaler.pkl` | MinMaxScaler for health features | `LSTM_Autoencoder_Train_Eval.ipynb` |
| `health_model_config.pkl` | seq_len, n_features, hidden_dim, latent_dim, feature_cols | `LSTM_Autoencoder_Train_Eval.ipynb` |
| `optimized_thresholds.pkl` | Per-feature anomaly thresholds | `LSTM_Autoencoder_Train_Eval.ipynb` (Cell 13) |
| `fuel_lstm_model.keras` | (archived, not loaded at runtime) | `AutoVue_Fuel_LSTM.ipynb` (Cell 11) |
| `fuel_feature_scaler.pkl` | (archived, not loaded at runtime) | `AutoVue_Fuel_LSTM.ipynb` |

> **Note (fuel):** `fuel_lstm_model.keras` and `fuel_feature_scaler.pkl` are retained for archival purposes. Runtime uses `app/ml/fuel_estimator.py` (physics-based). BiLSTM implementation preserved in `app/ml/fuel_estimator_bilstm.py`.

> **Note:** The health and fuel notebooks both save files called `scaler.pkl` / `feature_scaler.pkl`. They are renamed to `health_scaler.pkl` and `fuel_feature_scaler.pkl` respectively to avoid silent overwrites in the shared `models/` directory.

---

## Testing POST Endpoints

A browser address bar can only ever send `GET` — typing a URL and hitting
Enter will never trigger a `POST`, even if the endpoint exists and works
correctly. To exercise POST endpoints:

- **Swagger UI** — visit `/docs`, expand an endpoint, click "Try it out"
- **curl**
  ```bash
  # Driver behaviour
  curl -X POST http://localhost:8000/api/driver/predict \
    -H "Content-Type: application/json" \
    -d '{"rpm_values":[800,850,900,1200,1500,1400,1300],"speed_values":[0,0,5,20,35,40,38],"throttle_values":[10,12,15,30,45,40,35]}'

  # Health (needs 24+ ticks — abbreviated here)
  curl -X POST http://localhost:8000/api/health/predict \
    -H "Content-Type: application/json" \
    -d '{"ticks":[{"rpm":798,"vss":0,"maf":8.74,"throttle_pos":83.1,"map_kpa":97,"coolant_temp":33,"intake_air_temp":32,"ambient_temp":24,"pedal_d":14.1,"pedal_e":14.5}, ...]}'

  # Fuel (needs 20+ ticks — abbreviated here)
  curl -X POST http://localhost:8000/api/fuel/predict \
    -H "Content-Type: application/json" \
    -d '{"ticks":[{"rpm":1500,"vss":45,"maf":12.5,"throttle_pos":30,"map_kpa":97,"pedal_d":20,"pedal_e":18}, ...]}'
  ```
- **Postman / Insomnia** — import the OpenAPI schema from `/openapi.json`
- **The dashboard itself** — all its buttons already issue correct `fetch(..., {method: "POST"})` calls

---

## Deployment

**Recommended: [Render](https://render.com)** — free tier, no credit card, Dockerfile deploys work out of the box.

1. Push this repository to GitHub.
2. Render → **New +** → **Web Service** → connect the repo.
3. Render auto-detects the `Dockerfile`. Leave build/start commands blank.
4. Instance type: **Free**.
5. **Create Web Service** — first deploy takes 5–10 minutes (TensorFlow + PyTorch are large).
6. You'll get a URL like `https://your-app-name.onrender.com`.

**Free tier caveats:**
- Services sleep after ~15 minutes of inactivity and take 30–60s to wake
  on the next request. Open the URL a few minutes before a live demo to
  warm it up.
- The filesystem is not persistent across restarts/redeploys on the free
  tier — uploaded datasets won't survive a redeploy. Bundled datasets in
  `datasets/` (committed to the repo) always come back since they're part
  of the image.

**For a live demo with zero cold-start risk**, run locally and tunnel it:
```bash
uvicorn app.main:app --port 8000
ngrok http 8000        # or: npx localtunnel --port 8000
```

## Extending the System

### Adding a real OBD-II adapter
Only `app/simulator/services/simulator.py` needs a sibling, not a rewrite.
Create e.g. `app/simulator/services/live_adapter.py` implementing the same
tick-producing shape (using [`python-OBD`](https://python-obd.readthedocs.io/)
against a real ELM327 adapter instead of reading DataFrame rows), then
swap which one `app/main.py` instantiates behind a config flag. The API
layer, dashboard, and any ML code built against the
[standard field names](#standard-sensor-field-reference) keep working unchanged.

### Splitting the simulator into its own service
Deploy `app/simulator/` (plus `app/static/`, `datasets/`, `uploads/`) as
its own Render service, and point the frontend at two base URLs instead of
one. No internal rewrite required — this is exactly why it stayed a
self-contained package instead of being fused directly into `main.py`.

## Roadmap

- [x] ~~Wire simulator ticks directly into ML models so inference runs automatically on live data~~
- [x] ~~Anomaly detection (LSTM Autoencoder)~~
- [x] ~~Fuel efficiency estimation (BiLSTM + Attention)~~
- [ ] Composite Vehicle Health Score (0–100, weighted aggregation)
- [ ] Predictive maintenance / Remaining Useful Life forecasting
- [ ] DTC (Diagnostic Trouble Code) retrieval and decoding
- [ ] Persistent storage (PostgreSQL) and auth (JWT + bcrypt)
- [ ] Real-time mobile push notifications for critical anomalies

## License

Academic project — St Joseph Engineering College, VTU Belagavi. Not licensed for commercial redistribution.
