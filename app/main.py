"""
Merged entrypoint: runs the ECU Guardian ML API (driver behaviour + health
classification + fuel estimation) and the OBD-II simulator (dataset playback
+ streaming) in a single FastAPI app / single process / single Render service.

Why merged (see chat discussion): avoids double cold-starts on Render's
free tier and keeps one URL for the frontend. The simulator lives under
app/simulator/ as a self-contained package with its own router - splitting
it back out into its own service later is just deploying that package
separately and pointing the frontend at two URLs instead of one; no
internal rewrite needed.

Route map (no collisions):
  /health                     - shared health check
  /api/driver/predict         - ML: driver behaviour (XGBoost)
  /api/health/predict         - ML: ECU anomaly detection (LSTM Autoencoder, sequence input)
  /api/fuel/predict           - ML: fuel consumption rate (BiLSTM)
  /api/datasets, /api/upload,
  /api/start, /api/status,
  /api/live-data, /api/ws/live, ...  - simulator (see app/simulator/api/routes.py)
  /                            - simulator control dashboard
"""
import asyncio
import logging
import warnings
from typing import List

# Suppress specific harmless ML warnings to keep logs clean and save log buffer space
warnings.filterwarnings("ignore", message=".*sklearn\.utils\.parallel\.delayed.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*", category=UserWarning)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.simulator.api.routes import router as simulator_router
from app.simulator.api.websocket import telemetry_manager, log_manager
from app.simulator.services.simulator import simulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ecu_guardian")

app = FastAPI(title="ECU Guardian API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before real production use
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Pydantic request models ----------------

# ── Driver behaviour ─────────────────────────────────────────────────────────
class WindowRequest(BaseModel):
    rpm_values:      List[float]
    speed_values:    List[float]
    throttle_values: List[float]   # also used as pedal proxy
    window_index:    int = 0       # optional monotonic window counter


# ── Health anomaly detection ─────────────────────────────────────────────────
class HealthTick(BaseModel):
    # Include every field the LSTM was trained on.
    # Fields not recognised by the model are ignored; missing fields default to 0.0.
    rpm:             float = 0.0
    vss:             float = 0.0
    maf:             float = 0.0
    throttle_pos:    float = 0.0
    map_kpa:         float = 0.0
    coolant_temp:    float = 0.0
    intake_air_temp: float = 0.0
    ambient_temp:    float = 0.0
    pedal_d:         float = 0.0
    pedal_e:         float = 0.0

class HealthSequenceRequest(BaseModel):
    ticks: List[HealthTick]   # must be >= 24 items


# ── Fuel estimation ──────────────────────────────────────────────────────────
class FuelTick(BaseModel):
    rpm:          float = 0.0
    vss:          float = 0.0
    maf:          float = 0.0
    throttle_pos: float = 0.0
    map_kpa:      float = 0.0
    pedal_d:      float = 0.0
    pedal_e:      float = 0.0

class FuelWindowRequest(BaseModel):
    ticks: List[FuelTick]    # must be >= 20 items


# ---------------- ML endpoints ----------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/driver/predict")
def predict_driver(body: WindowRequest):
    from app.ml.driver_behaviour import predict_from_raw_window
    try:
        if len(body.rpm_values) < 5:
            raise HTTPException(status_code=400, detail="Need at least 5 data points")
        return predict_from_raw_window(
            body.rpm_values,
            body.speed_values,
            body.throttle_values,
            body.window_index,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/health/predict")
def predict_health(body: HealthSequenceRequest):
    from app.ml.health_classifier import classify_vehicle_health
    try:
        tick_dicts = [t.dict() for t in body.ticks]
        return classify_vehicle_health(tick_dicts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fuel/predict")
def predict_fuel(body: FuelWindowRequest):
    from app.ml.fuel_estimator import estimate_fuel
    try:
        tick_dicts = [t.dict() for t in body.ticks]
        return estimate_fuel(tick_dicts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- Simulator (dataset playback + streaming) ----------------

app.include_router(simulator_router)


async def _on_telemetry(payload: dict):
    await telemetry_manager.broadcast_json(payload)


def _on_log(entry: dict):
    coro = log_manager.broadcast_json(entry)
    try:
        asyncio.get_running_loop()
        asyncio.create_task(coro)
    except RuntimeError:
        coro.close()


simulator.subscribe(_on_telemetry)
simulator.subscribe_logs(_on_log)


@app.on_event("startup")
async def on_startup():
    def _preload_ml():
        try:
            import app.ml.driver_behaviour
            import app.ml.health_classifier
            import app.ml.fuel_estimator
            logger.info("Background ML models preloaded.")
        except Exception as e:
            logger.error("Error preloading ML models: %s", e)

    # Defer heavy framework loading to startup event so port can bind, loading in background
    asyncio.get_running_loop().run_in_executor(None, _preload_ml)
    
    try:
        simulator.load_dataset()
        logger.info("Simulator: default dataset pre-loaded. POST /api/start to begin streaming.")
    except ValueError as e:
        logger.warning("Simulator: no dataset available at startup: %s", e)


@app.get("/")
def serve_dashboard():
    return FileResponse("app/static/dashboard.html")


app.mount("/static", StaticFiles(directory="app/static"), name="static")
