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


# ---------- dataset endpoints ----------

@router.get("/datasets")
def list_datasets():
    return {"datasets": dataset_manager.list_datasets()}


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
