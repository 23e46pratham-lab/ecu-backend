"""
REST API - dataset management + simulator control, matching the
endpoints requested in the brief.
"""
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.simulator.services.dataset_manager import dataset_manager
from app.simulator.services.simulator import simulator, SimState
from app.simulator.api.websocket import telemetry_manager, log_manager

logger = logging.getLogger("obd_simulator.api")
router = APIRouter(prefix="/api", tags=["simulator"])


# ---------- request bodies ----------

class StartRequest(BaseModel):
    dataset_id: str | None = None
    speed: float | None = None
    loop: bool | None = None


class SpeedRequest(BaseModel):
    speed: float


class LoopRequest(BaseModel):
    loop: bool


class ChangeDatasetRequest(BaseModel):
    dataset_id: str


class RenameRequest(BaseModel):
    dataset_id: str
    new_name: str


class DialSetRequest(BaseModel):
    field: str
    value: float


class DtcRequest(BaseModel):
    codes: list[str]


# ---------- dataset endpoints ----------

@router.get("/datasets")
def list_datasets():
    datasets = dataset_manager.list_datasets()
    return {"datasets": datasets}


@router.post("/upload", status_code=201)
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(400, "Only .csv, .xlsx, .xls files are supported")
    content = await file.read()
    try:
        ds = dataset_manager.save_upload(file.filename, content)
    except Exception as e:
        raise HTTPException(400, f"Could not parse uploaded file: {e}")
    return {
        "dataset_id": ds.dataset_id,
        "filename": ds.name,
        "row_count": ds.row_count,
        "duration_seconds": round(ds.duration_seconds, 1),
        "missing_value_report": ds.missing_value_report,
    }


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    ok = dataset_manager.delete_dataset(dataset_id)
    if not ok:
        raise HTTPException(404, "Dataset not found")
    return {"deleted": True}


@router.patch("/datasets/{dataset_id}/rename")
def rename_dataset(dataset_id: str, body: RenameRequest):
    ok = dataset_manager.rename_dataset(dataset_id, body.new_name)
    if not ok:
        raise HTTPException(404, "Dataset not found or rename failed")
    return {"renamed": True}


@router.post("/change-dataset")
async def change_dataset(body: ChangeDatasetRequest):
    """Switches the active dataset. Restarts playback from row 0 if currently running."""
    was_running = simulator.status.state == SimState.RUNNING
    simulator.stop()
    simulator.load_dataset(body.dataset_id)
    if was_running:
        simulator.start(dataset_id=body.dataset_id)
    return {"active_dataset_id": body.dataset_id}


# ---------- simulator control endpoints ----------
# NOTE: /start (and change-dataset above) are `async def` deliberately -
# FastAPI runs sync `def` routes in a worker thread pool, where
# asyncio.create_task() (used inside Simulator.start() to launch the
# playback loop) has no running event loop to attach to and raises a
# RuntimeError. Declaring the route `async def` makes FastAPI run it
# directly on the event loop instead, so create_task() works correctly.

@router.post("/start")
async def start_simulation(body: StartRequest):
    try:
        status = simulator.start(dataset_id=body.dataset_id, speed=body.speed, loop=body.loop)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _status_dict(status)


@router.post("/pause")
def pause_simulation():
    return _status_dict(simulator.pause())


@router.post("/resume")
def resume_simulation():
    return _status_dict(simulator.resume())


@router.post("/stop")
def stop_simulation():
    return _status_dict(simulator.stop())


@router.post("/reset")
def reset_simulation():
    return _status_dict(simulator.reset())


@router.post("/speed")
def set_speed(body: SpeedRequest):
    try:
        return _status_dict(simulator.set_speed(body.speed))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/loop")
def set_loop(body: LoopRequest):
    return _status_dict(simulator.set_loop(body.loop))


@router.get("/status")
def get_status():
    return _status_dict(simulator.status)


@router.get("/live-data")
def get_live_data():
    if not simulator.latest_row:
        raise HTTPException(404, "No data yet - start the simulator first")
    return simulator.latest_row


@router.get("/history")
def get_history(limit: int = 100):
    return {"history": simulator.get_history(limit)}


@router.get("/ml/latest")
def get_latest_ml():
    """Returns the ML inference results attached to the most recent simulator tick."""
    if not simulator.latest_row:
        raise HTTPException(404, "No data yet — start the simulator first")
    return simulator.latest_row.get("ml", {})


# ---------- GPS / route endpoints ----------

@router.get("/route")
def get_route():
    """
    Returns the full GPS polyline for the currently-loaded dataset as a list
    of [lat, lon] pairs.  The frontend calls this once on load to draw the
    full route on the map before playback starts.

    404 if no dataset is loaded or the dataset has no GPS data.
    """
    if not simulator._dataset:
        raise HTTPException(404, "No dataset loaded")
    polyline = simulator._dataset.get_route_polyline()
    if polyline is None:
        raise HTTPException(404, "Current dataset has no GPS data")
    return {
        "has_gps": True,
        "point_count": len(polyline),
        "polyline": polyline,
        "summary": simulator._dataset.gps_summary,
    }


@router.get("/trip-summary")
def get_trip_summary():
    """
    Aggregated OBD + GPS statistics for the currently-loaded dataset.
    GPS fields (distance, point count, route bounding box) are included
    only when the dataset has GPS columns; OBD stats are always present.
    Safe to call at any point — does not require the simulator to be running.
    """
    import math

    if not simulator._dataset:
        raise HTTPException(404, "No dataset loaded")

    df = simulator._dataset.df
    stats: dict = {}

    # —— Always-available OBD stats ——————————————————————————————
    if "rpm" in df.columns:
        stats["avg_rpm"] = round(float(df["rpm"].mean()), 0)
        stats["max_rpm"] = round(float(df["rpm"].max()), 0)
    if "vss" in df.columns:
        stats["avg_speed_kmh"] = round(float(df["vss"].mean()), 1)
        stats["max_speed_kmh"] = round(float(df["vss"].max()), 1)
    if "throttle_pos" in df.columns:
        stats["avg_throttle"] = round(float(df["throttle_pos"].mean()), 1)

    # —— GPS-only stats —————————————————————————————————
    if simulator.has_gps:
        def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            """Return great-circle distance in metres."""
            R = 6_371_000
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlam = math.radians(lon2 - lon1)
            a = (math.sin(dphi / 2) ** 2
                 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        pts = df[["lat", "lon"]].dropna().values.tolist()
        dist_m = sum(
            haversine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
            for i in range(len(pts) - 1)
        )
        stats["distance_km"] = round(dist_m / 1000, 2)
        stats["gps_points"] = len(pts)
        stats["route_summary"] = simulator._dataset.gps_summary

    return {"trip_stats": stats, "has_gps": simulator.has_gps}



def _status_dict(status) -> dict:
    return {
        "state": status.state.value,
        "dataset_id": status.dataset_id,
        "dataset_name": status.dataset_name,
        "current_row": status.current_row,
        "total_rows": status.total_rows,
        "speed": status.speed,
        "loop": status.loop,
        "elapsed_playback_seconds": round(status.elapsed_playback_seconds, 1),
        "dataset_duration_seconds": round(status.dataset_duration_seconds, 1),
        "playback_percent": round(100 * status.current_row / status.total_rows, 2) if status.total_rows else 0,
    }


# ---------- sensor dial override endpoints ----------

@router.get("/dials")
def get_dials():
    """Returns all currently active dial overrides as {field: value}."""
    return {"overrides": simulator.get_dials()}


@router.post("/dials")
def set_dial(body: DialSetRequest):
    """
    Override a single sensor field with a fixed value.
    The override takes effect on the very next simulator tick and persists
    until explicitly released. The dataset playback continues normally—
    only the value broadcast to clients is replaced.
    """
    try:
        overrides = simulator.set_dial(body.field, body.value)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"overrides": overrides}


@router.delete("/dials")
def release_all_dials():
    """Releases all active dial overrides. All fields snap back to dataset values."""
    return {"overrides": simulator.release_all_dials()}


@router.delete("/dials/{field}")
def release_dial(field: str):
    """Releases the override for a single field; it resumes reading from the dataset."""
    try:
        overrides = simulator.release_dial(field)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"overrides": overrides}



# ---------- DTC (Diagnostic Trouble Code) endpoints ----------

@router.get("/dtc")
def get_dtcs():
    """Returns the currently active DTC fault code list."""
    return {"dtcs": simulator.get_dtcs()}


@router.post("/dtc")
def set_dtcs(body: DtcRequest):
    """
    Replace the active DTC list with the supplied codes.
    Codes are normalised to uppercase (e.g. ``P0300``).
    Pass an empty list to clear all active codes.
    When at least one code is active the simulator broadcast payload includes
    a ``dtcs`` key; when the list is empty the key is omitted entirely so
    existing clients are unaffected.
    """
    codes = simulator.set_dtcs(body.codes)
    return {"dtcs": codes}


@router.delete("/dtc")
def clear_all_dtcs():
    """Clears all active DTC fault codes."""
    return {"dtcs": simulator.clear_dtcs()}


@router.delete("/dtc/{code}")
def clear_single_dtc(code: str):
    """Removes a single DTC fault code. Silently succeeds if the code is not active."""
    return {"dtcs": simulator.clear_dtc(code)}


# ---------- WebSocket endpoints ----------

@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await telemetry_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        telemetry_manager.disconnect(websocket)


@router.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await log_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        log_manager.disconnect(websocket)
