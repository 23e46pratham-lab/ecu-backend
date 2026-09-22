"""
Simulator engine: plays back a loaded dataset row-by-row in real time,
mimicking how a real ELM327/OBD-II adapter streams live PIDs.

Design note on future hardware swap: this class exposes exactly the same
shape of state/output that a real adapter reader would (a dict of standard
sensor fields + a timestamp), regardless of where the numbers come from.
To later swap in a real Bluetooth/WiFi ELM327 adapter or CAN bus reader,
you only need to write a new class with the same `current_state` property
and `subscribe()` mechanism - nothing in api/ or the frontend needs to
change. That's the abstraction boundary requested in the brief.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import pandas as pd

from app.simulator import config
from app.simulator.services.dataset_manager import LoadedDataset, dataset_manager

logger = logging.getLogger("obd_simulator.simulator")


class SimState(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"


@dataclass
class SimulatorStatus:
    state: SimState = SimState.STOPPED
    dataset_id: str | None = None
    dataset_name: str | None = None
    current_row: int = 0
    total_rows: int = 0
    speed: float = 1.0
    loop: bool = True
    elapsed_playback_seconds: float = 0.0
    dataset_duration_seconds: float = 0.0
    started_at: float | None = None


class Simulator:
    """
    Single global simulator instance (one active "virtual vehicle" at a
    time - matches the brief's use case of one demo vehicle streaming to
    one dashboard). Runs its playback loop as an asyncio background task.
    """

    def __init__(self):
        self._dataset: LoadedDataset | None = None
        self._status = SimulatorStatus()
        self._task: asyncio.Task | None = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # not paused by default
        self._subscribers: list[Callable] = []
        self._log_subscribers: list[Callable] = []
        self._latest_row: dict = {}
        self._history: list[dict] = []

        # Rolling buffer for ML inference
        self._tick_buffer: list[dict] = []
        self._HEALTH_SEQ_LEN    = 24          # must match LSTM training
        self._FUEL_WINDOW_SIZE  = 20          # must match Fuel LSTM training
        self._DRIVER_WINDOW_MIN = 30          # min ticks before driver inference runs
        self._driver_window_index = 0         # monotonic counter

        # Dial override state — maps field name -> override float value.
        # When a field is present here, its value replaces the dataset reading
        # in the tick payload. When absent, the dataset reading is used as-is.
        # This dict is mutated only via the set_dial / release_dial / release_all
        # methods, which are called from the API layer.
        self._dial_overrides: dict[str, float] = {}

    # ---------- subscription (used by the WebSocket layer) ----------

    def subscribe(self, callback: Callable):
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable):
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def subscribe_logs(self, callback: Callable):
        self._log_subscribers.append(callback)

    def unsubscribe_logs(self, callback: Callable):
        if callback in self._log_subscribers:
            self._log_subscribers.remove(callback)

    def _emit_log(self, level: str, message: str):
        entry = {"level": level, "message": message, "timestamp": time.time()}
        logger.log(getattr(logging, level.upper(), logging.INFO), message)
        for cb in list(self._log_subscribers):
            try:
                cb(entry)
            except Exception:
                logger.exception("Log subscriber callback failed")

    async def _broadcast(self, payload: dict):
        for cb in list(self._subscribers):
            try:
                result = cb(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Subscriber callback failed")

    # ---------- dataset selection ----------

    def load_dataset(self, dataset_id: str | None = None):
        if dataset_id:
            ds = dataset_manager.get_by_id(dataset_id)
            if not ds:
                raise ValueError(f"Unknown dataset_id: {dataset_id}")
        else:
            ds = dataset_manager.get_default()
            if not ds:
                raise ValueError("No datasets available. Upload one first.")

        self._dataset = ds
        self._status.dataset_id = ds.dataset_id
        self._status.dataset_name = ds.name
        self._status.total_rows = ds.row_count
        self._status.dataset_duration_seconds = ds.duration_seconds
        self._status.current_row = 0
        self._status.elapsed_playback_seconds = 0.0
        self._tick_buffer.clear()
        self._driver_window_index = 0
        self._emit_log("info", f"Dataset loaded: {ds.name} ({ds.row_count} rows, "
                                f"~{ds.duration_seconds:.0f}s)")

    # ---------- playback controls ----------

    def start(self, dataset_id: str | None = None, speed: float | None = None, loop: bool | None = None):
        if self._dataset is None or dataset_id:
            self.load_dataset(dataset_id)
        if speed is not None:
            self.set_speed(speed)
        if loop is not None:
            self._status.loop = loop

        if self._task and not self._task.done():
            self._emit_log("warning", "Start requested but simulator already running")
            return self._status

        self._status.current_row = 0
        self._status.state = SimState.RUNNING
        self._status.started_at = time.time()
        self._pause_event.set()
        self._task = asyncio.create_task(self._run_loop())
        self._emit_log("info", f"Streaming started (speed={self._status.speed}x, loop={self._status.loop})")
        return self._status

    def pause(self):
        if self._status.state != SimState.RUNNING:
            return self._status
        self._pause_event.clear()
        self._status.state = SimState.PAUSED
        self._emit_log("info", "Streaming paused")
        return self._status

    def resume(self):
        if self._status.state != SimState.PAUSED:
            return self._status
        self._pause_event.set()
        self._status.state = SimState.RUNNING
        self._emit_log("info", "Streaming resumed")
        return self._status

    def stop(self):
        if self._task:
            self._task.cancel()
        self._status.state = SimState.STOPPED
        self._pause_event.set()
        self._tick_buffer.clear()
        self._driver_window_index = 0
        self._emit_log("info", "Streaming stopped")
        return self._status

    def reset(self):
        self.stop()
        self._status.current_row = 0
        self._status.elapsed_playback_seconds = 0.0
        self._status.state = SimState.STOPPED
        self._emit_log("info", "Simulator reset to start of dataset")
        return self._status

    def set_speed(self, speed: float):
        if speed not in config.ALLOWED_SPEEDS:
            raise ValueError(f"Speed must be one of {config.ALLOWED_SPEEDS}")
        self._status.speed = speed
        self._emit_log("info", f"Playback speed set to {speed}x")
        return self._status

    def set_loop(self, loop: bool):
        self._status.loop = loop
        return self._status

    # ---------- dial overrides ----------

    # Valid sensor field names — the contract between the frontend and the
    # dataset_manager's STANDARD_COLUMNS list.
    _VALID_DIAL_FIELDS = frozenset([
        "coolant_temp", "map_kpa", "rpm", "vss", "intake_air_temp",
        "maf", "throttle_pos", "ambient_temp", "pedal_d", "pedal_e",
    ])

    def set_dial(self, field: str, value: float) -> dict:
        """Override a sensor field with a fixed value. Affects next tick onward."""
        if field not in self._VALID_DIAL_FIELDS:
            raise ValueError(f"Unknown sensor field: '{field}'. "
                             f"Valid fields: {sorted(self._VALID_DIAL_FIELDS)}")
        self._dial_overrides[field] = float(value)
        self._emit_log("info", f"Dial override set: {field} = {value}")
        return dict(self._dial_overrides)

    def release_dial(self, field: str) -> dict:
        """Release override for one field; it resumes reading from the dataset."""
        if field not in self._VALID_DIAL_FIELDS:
            raise ValueError(f"Unknown sensor field: '{field}'")
        self._dial_overrides.pop(field, None)
        self._emit_log("info", f"Dial override released: {field}")
        return dict(self._dial_overrides)

    def release_all_dials(self) -> dict:
        """Release all active overrides at once."""
        self._dial_overrides.clear()
        self._emit_log("info", "All dial overrides released")
        return {}

    def get_dials(self) -> dict:
        """Return the current override values (field -> value)."""
        return dict(self._dial_overrides)

    # ---------- state accessors ----------

    @property
    def status(self) -> SimulatorStatus:
        return self._status

    @property
    def has_gps(self) -> bool:
        """True when the currently-loaded dataset has GPS columns merged in."""
        return self._dataset.has_gps if self._dataset else False

    @property
    def latest_row(self) -> dict:
        return self._latest_row

    def get_history(self, limit: int = 100) -> list[dict]:
        return self._history[-limit:]

    # ---------- the playback loop itself ----------

    async def _run_loop(self):
        ds = self._dataset
        df = ds.df
        n = len(df)

        try:
            while self._status.current_row < n:
                await self._pause_event.wait()  # blocks here while paused

                row = df.iloc[self._status.current_row]

                # Build the raw data dict from the dataset row first.
                raw_data = {
                    "coolant_temp": _safe_float(row.get("coolant_temp")),
                    "map_kpa": _safe_float(row.get("map_kpa")),
                    "rpm": _safe_float(row.get("rpm")),
                    "vss": _safe_float(row.get("vss")),
                    "intake_air_temp": _safe_float(row.get("intake_air_temp")),
                    "maf": _safe_float(row.get("maf")),
                    "throttle_pos": _safe_float(row.get("throttle_pos")),
                    "ambient_temp": _safe_float(row.get("ambient_temp")),
                    "pedal_d": _safe_float(row.get("pedal_d")),
                    "pedal_e": _safe_float(row.get("pedal_e")),
                }

                # Append GPS fields when the dataset has them and this row has a
                # valid fix. NaN rows (interpolation gaps) are omitted entirely
                # so the frontend never sees a null position mid-route.
                if ds.has_gps:
                    _GPS_FIELDS = [
                        "lat", "lon", "elevation_m",
                        "gps_bearing", "gps_speed_ms", "gps_fix",
                    ]
                    for field in _GPS_FIELDS:
                        if field in row.index and pd.notna(row[field]):
                            precision = 7 if field in ("lat", "lon") else 3
                            raw_data[field] = round(float(row[field]), precision)

                # Merge dial overrides on top (snapshot to avoid race conditions
                # if the API layer updates overrides mid-tick).
                active_overrides = dict(self._dial_overrides)
                effective_data = {**raw_data, **active_overrides}

                payload = {
                    "row_index": int(self._status.current_row),
                    "total_rows": n,
                    "elapsed_seconds": float(row["elapsed_seconds"]),
                    "playback_percent": round(100 * (self._status.current_row + 1) / n, 2),
                    # `data` contains the effective values (dataset + any overrides merged)
                    "data": effective_data,
                    # `dataset_data` is always the raw dataset reading — lets the
                    # frontend keep tracking the true dataset position for dials
                    # that are not currently overridden.
                    "dataset_data": raw_data,
                    # `overrides` tells the frontend exactly which fields are
                    # currently manually overridden (to drive dial highlight/animation).
                    "overrides": list(active_overrides.keys()),
                }

                # ── Accumulate tick in the rolling buffer ──────────────────────
                self._tick_buffer.append(payload["data"])
                max_buf = max(self._HEALTH_SEQ_LEN, self._FUEL_WINDOW_SIZE, self._DRIVER_WINDOW_MIN)
                if len(self._tick_buffer) > max_buf:
                    self._tick_buffer.pop(0)

                # ── Run ML inference (errors swallowed so simulator never crashes) ──
                ml_results = {}

                # Driver behaviour — runs once we have enough ticks
                if len(self._tick_buffer) >= self._DRIVER_WINDOW_MIN:
                    buf = self._tick_buffer
                    rpm_vals   = [t.get("rpm",          0.0) for t in buf]
                    speed_vals = [t.get("vss",          0.0) for t in buf]
                    pedal_vals = [t.get("throttle_pos", 0.0) for t in buf]
                    try:
                        from app.ml.driver_behaviour import predict_from_raw_window
                        ml_results["driver_behaviour"] = predict_from_raw_window(
                            rpm_vals, speed_vals, pedal_vals, self._driver_window_index
                        )
                        self._driver_window_index += 1
                    except Exception as exc:
                        ml_results["driver_behaviour"] = {"error": str(exc)}

                # Health anomaly detection — needs 24 ticks
                if len(self._tick_buffer) >= self._HEALTH_SEQ_LEN:
                    try:
                        from app.ml.health_classifier import classify_vehicle_health
                        ml_results["health"] = classify_vehicle_health(self._tick_buffer)
                    except Exception as exc:
                        ml_results["health"] = {"error": str(exc)}

                # Fuel estimation — needs 20 ticks
                if len(self._tick_buffer) >= self._FUEL_WINDOW_SIZE:
                    try:
                        # Physics-based: Tier 1=MAF, Tier 2=MAP+RPM, Tier 3=throttle heuristic
                        from app.ml.fuel_estimator import estimate_fuel
                        ml_results["fuel"] = estimate_fuel(self._tick_buffer)
                    except Exception as exc:
                        ml_results["fuel"] = {"error": str(exc)}

                # ── Attach ML results to the tick payload ─────────────────────
                payload["ml"] = ml_results

                self._latest_row = payload
                self._history.append(payload)
                if len(self._history) > config.HISTORY_BUFFER_SIZE:
                    self._history.pop(0)

                await self._broadcast(payload)

                self._status.current_row += 1
                self._status.elapsed_playback_seconds = payload["elapsed_seconds"]

                tick = float(row.get("tick_interval", config.DEFAULT_TICK_SECONDS))
                sleep_for = max(tick / max(self._status.speed, 0.01), 0.0)
                await asyncio.sleep(sleep_for)

            # dataset exhausted
            if self._status.loop:
                self._emit_log("info", "Dataset finished - looping back to start")
                self._status.current_row = 0
                self._task = asyncio.create_task(self._run_loop())
            else:
                self._status.state = SimState.FINISHED
                self._emit_log("info", "Dataset finished - loop disabled, stopping")

        except asyncio.CancelledError:
            logger.info("Playback loop cancelled")
            raise
        except Exception:
            logger.exception("Playback loop crashed")
            self._status.state = SimState.STOPPED
            self._emit_log("error", "Playback loop crashed - see server logs")


def _safe_float(val):
    """Converts NaN to None so it serializes as JSON null, not the invalid 'NaN' token."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # f != f is True only for NaN


simulator = Simulator()
