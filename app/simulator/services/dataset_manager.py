"""
Dataset management: discovering, loading, cleaning, and validating
OBD-II datasets from datasets/ (bundled) and uploads/ (user-provided).

Handles the real-world messiness described in the brief:
  - mis-encoded degree signs ("Â°C" from a Windows-1252 file read as UTF-8)
  - blank/NULL/zero values, especially at the start of a recording
  - inconsistent column name formatting between dataset exports
"""
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.simulator import config

logger = logging.getLogger("obd_simulator.dataset_manager")

# Internal standard field names used everywhere downstream (API responses,
# simulator ticks, future ML inference). Keep these stable - this is the
# contract the frontend and any future ML service will rely on.
STANDARD_COLUMNS = [
    "coolant_temp", "map_kpa", "rpm", "vss", "intake_air_temp",
    "maf", "throttle_pos", "ambient_temp", "pedal_d", "pedal_e",
]

# GPS columns added by merge_gps_obd.py. Only "lat" and "lon" are required;
# the optional columns are included when present.
GPS_REQUIRED = {"lat", "lon"}
GPS_OPTIONAL = {"elevation_m", "gps_bearing", "gps_speed_ms", "gps_fix"}
GPS_ALL = GPS_REQUIRED | GPS_OPTIONAL

# Keyword-based matching instead of exact string matching, because dataset
# exports vary in exact punctuation/encoding of the header row (e.g. the
# degree sign). Matching on a stable keyword is far more robust than
# matching the full header string.
COLUMN_KEYWORDS = {
    "coolant_temp": ["coolant"],
    "map_kpa": ["manifold absolute pressure", "map"],
    "rpm": ["rpm"],
    "vss": ["vehicle speed", "speed sensor"],
    "intake_air_temp": ["intake air temperature"],
    "maf": ["mass flow", "air flow rate"],
    "throttle_pos": ["throttle"],
    "ambient_temp": ["ambient"],
    "pedal_d": ["pedal position d"],
    "pedal_e": ["pedal position e"],
}
TIME_KEYWORDS = ["time"]


def _match_column(header: str) -> str | None:
    h = header.lower()
    for std_name, keywords in COLUMN_KEYWORDS.items():
        if any(k in h for k in keywords):
            return std_name
    if any(k in h for k in TIME_KEYWORDS):
        return "time_raw"
    return None


def _read_any_encoding(path: Path) -> pd.DataFrame:
    """Try common encodings in order until one parses cleanly."""
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            if path.suffix.lower() in (".xlsx", ".xls"):
                return pd.read_excel(path)
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    raise ValueError(f"Could not decode {path.name} with any known encoding: {last_err}")


def _parse_time_to_seconds(raw: pd.Series) -> pd.Series:
    """
    Dataset 'Time' column looks like 'MM:SS.s' (minutes:seconds.tenths),
    e.g. '17:19.4'. Convert to elapsed seconds since the start of the file.

    Real-world quirk observed in the actual KIT sample file: the Time
    column is only populated for the first ~150 rows and then goes
    completely blank for the rest of the recording (a logging artifact
    of the recording app). So this can't just ffill/bfill - that would
    flatline elapsed time. Instead: once real timestamps run out,
    extrapolate forward at DEFAULT_TICK_SECONDS per row from the last
    known real timestamp. Any leading gap (no valid timestamp seen yet)
    falls back to a synthetic index * DEFAULT_TICK_SECONDS, retroactively
    corrected once the first real timestamp is found.
    """
    pattern = re.compile(r"^(\d+):(\d+(?:\.\d+)?)$")

    def parse_one(val):
        if pd.isna(val):
            return None
        m = pattern.match(str(val).strip())
        if not m:
            return None
        minutes, seconds = m.groups()
        return int(minutes) * 60 + float(seconds)

    parsed = raw.map(parse_one)

    if parsed.isna().all():
        logger.warning("Time column unparseable; falling back to synthetic even spacing")
        return pd.Series(range(len(raw)), dtype=float) * config.DEFAULT_TICK_SECONDS

    n_missing = int(parsed.isna().sum())
    if n_missing:
        logger.info("Time column has %d/%d missing timestamps; extrapolating "
                     "at %.2fs/row from the last known timestamp",
                     n_missing, len(raw), config.DEFAULT_TICK_SECONDS)

    values = parsed.tolist()
    result = [0.0] * len(values)
    last_valid_val = None
    last_valid_idx = -1
    for i, v in enumerate(values):
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            result[i] = v
            last_valid_val = v
            last_valid_idx = i
        elif last_valid_val is not None:
            result[i] = last_valid_val + (i - last_valid_idx) * config.DEFAULT_TICK_SECONDS
        else:
            result[i] = i * config.DEFAULT_TICK_SECONDS

    first_valid_idx = next((i for i, v in enumerate(values)
                             if v is not None and not (isinstance(v, float) and pd.isna(v))), None)
    if first_valid_idx is not None and first_valid_idx > 0:
        first_valid = values[first_valid_idx]
        offset = first_valid - (first_valid_idx * config.DEFAULT_TICK_SECONDS)
        for i in range(first_valid_idx):
            result[i] += offset

    return pd.Series(result, index=raw.index, dtype=float)


@dataclass
class LoadedDataset:
    dataset_id: str
    name: str
    path: Path
    df: pd.DataFrame              # cleaned, standardized columns + elapsed_seconds + tick_interval
    row_count: int
    duration_seconds: float
    missing_value_report: dict = field(default_factory=dict)
    has_gps: bool = False

    @property
    def gps_summary(self) -> dict | None:
        """Return bounding-box + centre stats for the GPS track, or None if no GPS."""
        if not self.has_gps:
            return None
        lats = self.df["lat"].dropna()
        lons = self.df["lon"].dropna()
        return {
            "has_gps": True,
            "point_count": len(lats),
            "bbox": {
                "min_lat": float(lats.min()),
                "max_lat": float(lats.max()),
                "min_lon": float(lons.min()),
                "max_lon": float(lons.max()),
            },
            "center": {
                "lat": float(lats.mean()),
                "lon": float(lons.mean()),
            },
        }

    def get_route_polyline(self) -> list[list[float]] | None:
        """
        Return the full GPS track as [[lat, lon], ...] — one entry per dataset
        row that has a valid fix. Returns None if the dataset has no GPS columns.
        Called by GET /api/route so the frontend can draw the full path on load.
        """
        if not self.has_gps:
            return None
        return (
            self.df[["lat", "lon"]]
            .dropna()
            .apply(lambda r: [round(r["lat"], 7), round(r["lon"], 7)], axis=1)
            .tolist()
        )


def _clean_dataframe(raw_df: pd.DataFrame, source_name: str) -> LoadedDataset:
    rename_map = {}
    for col in raw_df.columns:
        std = _match_column(col)
        if std:
            rename_map[col] = std
    df = raw_df.rename(columns=rename_map)

    # Detect GPS columns before filtering — GPS cols have exact lowercase names
    # added by merge_gps_obd.py, so no keyword matching is needed.
    cols_lower_set = {c.lower() for c in df.columns}
    has_gps = GPS_REQUIRED.issubset(cols_lower_set)
    gps_present = [c for c in df.columns if c.lower() in GPS_ALL] if has_gps else []
    if has_gps:
        logger.info("Dataset '%s': GPS columns detected: %s", source_name, gps_present)

    keep_cols = [c for c in STANDARD_COLUMNS + ["time_raw"] if c in df.columns]
    # Append GPS columns so they survive into the cleaned DataFrame.
    keep_cols += [c for c in gps_present if c not in keep_cols]
    df = df[keep_cols].copy()

    numeric_cols = [c for c in STANDARD_COLUMNS if c in df.columns]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Coerce GPS columns to numeric as well (handles any stray string tokens).
    for c in gps_present:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    missing_report = {c: int(df[c].isna().sum()) for c in numeric_cols}
    for c, n in missing_report.items():
        if n > 0:
            logger.info("Dataset '%s': column '%s' has %d/%d missing values",
                        source_name, c, n, len(df))

    if config.MISSING_VALUE_STRATEGY == "ffill":
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
    elif config.MISSING_VALUE_STRATEGY == "zero":
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
    # "keep_null" -> leave NaN as-is; API layer converts NaN -> null in JSON
    # GPS columns intentionally keep NaN for rows without a fix — the tick
    # builder skips NaN GPS values rather than broadcasting stale coordinates.

    if "time_raw" in df.columns:
        elapsed = _parse_time_to_seconds(df["time_raw"])
    else:
        elapsed = pd.Series(range(len(df)), dtype=float) * config.DEFAULT_TICK_SECONDS

    df["elapsed_seconds"] = elapsed
    deltas = df["elapsed_seconds"].diff().fillna(config.DEFAULT_TICK_SECONDS)
    df["tick_interval"] = deltas.clip(lower=0.01, upper=config.MAX_TICK_SECONDS)

    duration = float(df["elapsed_seconds"].iloc[-1] - df["elapsed_seconds"].iloc[0]) if len(df) else 0.0

    return LoadedDataset(
        dataset_id=str(uuid.uuid4()),
        name=source_name,
        path=Path(source_name),
        df=df,
        row_count=len(df),
        duration_seconds=max(duration, 0.0),
        missing_value_report=missing_report,
        has_gps=has_gps,
    )


class DatasetManager:
    """
    Discovers dataset files on disk and caches loaded/cleaned DataFrames
    in memory by dataset_id. A single instance is shared across the app
    (see app/main.py).
    """

    def __init__(self):
        self._loaded: dict[str, LoadedDataset] = {}
        self._id_by_filename: dict[str, str] = {}

    def _all_files(self) -> list[Path]:
        files = []
        for d in (config.DATASETS_DIR, config.UPLOADS_DIR):
            files.extend(sorted(d.glob("*.csv")))
            files.extend(sorted(d.glob("*.xlsx")))
            files.extend(sorted(d.glob("*.xls")))
        return files

    def list_datasets(self) -> list[dict]:
        result = []
        for path in self._all_files():
            ds = self._get_or_load(path)
            result.append({
                "dataset_id": ds.dataset_id,
                "filename": path.name,
                "row_count": ds.row_count,
                "duration_seconds": round(ds.duration_seconds, 1),
                "missing_value_report": ds.missing_value_report,
            })
        return result

    def _get_or_load(self, path: Path) -> LoadedDataset:
        filename = path.name
        if filename in self._id_by_filename:
            return self._loaded[self._id_by_filename[filename]]

        raw_df = _read_any_encoding(path)
        ds = _clean_dataframe(raw_df, source_name=filename)
        ds.path = path
        self._loaded[ds.dataset_id] = ds
        self._id_by_filename[filename] = ds.dataset_id
        logger.info("Loaded dataset '%s': %d rows, ~%.1fs duration",
                     filename, ds.row_count, ds.duration_seconds)
        return ds

    def get_by_id(self, dataset_id: str) -> LoadedDataset | None:
        return self._loaded.get(dataset_id)

    def get_default(self) -> LoadedDataset | None:
        files = self._all_files()
        if not files:
            return None
        return self._get_or_load(files[0])

    def save_upload(self, filename: str, content: bytes) -> LoadedDataset:
        safe_name = Path(filename).name
        dest = config.UPLOADS_DIR / safe_name
        dest.write_bytes(content)
        self._id_by_filename.pop(safe_name, None)
        return self._get_or_load(dest)

    def delete_dataset(self, dataset_id: str) -> bool:
        ds = self._loaded.get(dataset_id)
        if not ds:
            return False
        try:
            if ds.path.exists() and ds.path.is_relative_to(config.UPLOADS_DIR):
                ds.path.unlink()
            else:
                logger.warning("Refusing to delete bundled sample dataset '%s' from disk; "
                                "removing from memory cache only.", ds.name)
        except Exception as e:
            logger.error("Failed to delete file %s: %s", ds.path, e)
        self._loaded.pop(dataset_id, None)
        self._id_by_filename.pop(ds.name, None)
        return True

    def rename_dataset(self, dataset_id: str, new_name: str) -> bool:
        ds = self._loaded.get(dataset_id)
        if not ds:
            return False
        new_name = Path(new_name).stem + ds.path.suffix
        new_path = ds.path.with_name(new_name)
        try:
            ds.path.rename(new_path)
        except Exception as e:
            logger.error("Rename failed: %s", e)
            return False
        self._id_by_filename.pop(ds.name, None)
        ds.name = new_name
        ds.path = new_path
        self._id_by_filename[new_name] = ds.dataset_id
        return True


dataset_manager = DatasetManager()
