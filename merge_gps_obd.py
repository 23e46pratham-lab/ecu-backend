#!/usr/bin/env python3
"""
merge_gps_obd.py — AutoVue GPS + OBD-II Dataset Merger
========================================================
Merges a GPS log (GPX or CSV from GPSLogger/track_points) with an OBD-II
CSV dataset by aligning on RELATIVE time from trip start, then interpolating
GPS coordinates linearly for every OBD row.

Since the GPS log and OBD log are separate recordings, this is a synthetic
merge purely for presentation — the GPS track is "replayed" alongside the
OBD data. GPS coordinates are interpolated so every OBD row gets a
smooth lat/lon/elevation value.

Usage
-----
# Using a GPSLogger CSV (recommended — ~14 s interval):
python merge_gps_obd.py --obd datasets/your_obd.csv --gps 20260917131011.csv

# Using the GPX file directly:
python merge_gps_obd.py --obd datasets/your_obd.csv --gps 20260917.gpx

# Using the track_points.csv (converted from GPX):
python merge_gps_obd.py --obd datasets/your_obd.csv --gps track_points.csv

# Output path (default: <obd_stem>_with_gps.csv):
python merge_gps_obd.py --obd datasets/normal.csv --gps 20260917131011.csv -o datasets/normal_gps.csv

# Loop GPS track if OBD is longer than GPS:
python merge_gps_obd.py --obd datasets/normal.csv --gps 20260917131011.csv --loop

Output columns (appended to every OBD row)
-------------------------------------------
  lat          Latitude (decimal degrees)
  lon          Longitude (decimal degrees)
  elevation    Elevation (metres, may be negative = uncalibrated GPS)
  gps_bearing  Heading 0-360°
  gps_speed_ms Speed from GPS (m/s)
  gps_fix      1 = real GPS fix, 0 = interpolated
"""

import argparse
import csv
import math
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

try:
    from dateutil import parser as dtparser
except ImportError:
    sys.exit("pip install python-dateutil")


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_utc(s: str) -> Optional[datetime]:
    """Parse any reasonable ISO-8601 string → UTC datetime."""
    try:
        dt = dtparser.parse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _lerp(a, b, t):
    """Linear interpolation between a and b at fraction t ∈ [0,1]."""
    return a + (b - a) * t


def _bearing_lerp(b0, b1, t):
    """Interpolate bearing with wrap-around (0/360 boundary)."""
    diff = ((b1 - b0 + 540) % 360) - 180
    return (b0 + diff * t) % 360


# ── GPS loaders ──────────────────────────────────────────────────────────────

def load_gps_gpslogger_csv(path: str) -> List[Dict]:
    """Load GPSLogger CSV (20260917131011.csv format)."""
    points = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = _parse_utc(row.get("time", ""))
            if t is None:
                continue
            try:
                points.append({
                    "utc": t,
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "ele": float(row.get("elevation") or 0),
                    "bearing": float(row.get("bearing") or 0),
                    "speed_ms": float(row.get("speed") or 0),
                })
            except (ValueError, KeyError):
                continue
    return points


def load_gps_track_points_csv(path: str) -> List[Dict]:
    """Load track_points.csv (GPX→CSV online-converter format, X=lon, Y=lat)."""
    points = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = _parse_utc(row.get("time", ""))
            if t is None:
                continue
            try:
                points.append({
                    "utc": t,
                    "lat": float(row["Y"]),      # Y = latitude in this format
                    "lon": float(row["X"]),      # X = longitude
                    "ele": float(row.get("ele") or 0),
                    "bearing": float(row.get("course") or 0),
                    "speed_ms": float(row.get("speed") or 0),
                })
            except (ValueError, KeyError):
                continue
    return points


def load_gps_gpx(path: str) -> List[Dict]:
    """Load .gpx file directly (handles namespace)."""
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"g": root.tag.split("}")[0].lstrip("{")} if "}" in root.tag else {"g": ""}
    tag = lambda name: f"{{{ns['g']}}}{name}" if ns["g"] else name

    points = []
    for trkpt in root.iter(tag("trkpt")):
        t = _parse_utc(trkpt.findtext(tag("time")) or "")
        if t is None:
            continue
        try:
            points.append({
                "utc": t,
                "lat": float(trkpt.attrib["lat"]),
                "lon": float(trkpt.attrib["lon"]),
                "ele": float(trkpt.findtext(tag("ele")) or 0),
                "bearing": float(trkpt.findtext(tag("course")) or 0),
                "speed_ms": float(trkpt.findtext(tag("speed")) or 0),
            })
        except (ValueError, KeyError):
            continue
    return points


def detect_and_load_gps(path: str) -> List[Dict]:
    """Auto-detect GPS file type and load."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".gpx":
        print(f"[GPS] Detected GPX file: {p.name}")
        return load_gps_gpx(path)
    if ext == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f))
        header_set = set(h.strip().lower() for h in header)
        if "lat" in header_set and "lon" in header_set:
            print(f"[GPS] Detected GPSLogger CSV: {p.name}")
            return load_gps_gpslogger_csv(path)
        if "x" in header_set and "y" in header_set:
            print(f"[GPS] Detected track_points CSV (GPX-converted): {p.name}")
            return load_gps_track_points_csv(path)
    sys.exit(f"[ERROR] Cannot detect GPS format for: {path}")


# ── OBD loader ────────────────────────────────────────────────────────────────

# Column aliases: map various OBD CSV header names → standard internal key
_OBD_COL_MAP = {
    "time":                                  "obd_time",
    "device time":                           "obd_time",
    "engine coolant temp":                   "coolant_temp",
    "engine coolant temperature":            "coolant_temp",
    "coolant_temp":                          "coolant_temp",
    "intake manifold abs":                   "map_kpa",
    "intake manifold absolute pressure":     "map_kpa",
    "map_kpa":                               "map_kpa",
    "engine rpm [rpm]":                      "rpm",
    "engine rpm":                            "rpm",
    "rpm":                                   "rpm",
    "vehicle speed sensor":                  "vss",
    "vehicle speed sens":                    "vss",
    "vss":                                   "vss",
    "intake air temperature":                "intake_air_temp",
    "intake air temp":                       "intake_air_temp",
    "intake_air_temp":                       "intake_air_temp",
    "air flow rate from m":                  "maf",
    "air flow rate from maf":                "maf",
    "maf":                                   "maf",
    "absolute throttle po":                  "throttle_pos",
    "absolute throttle position":            "throttle_pos",
    "throttle_pos":                          "throttle_pos",
    "ambient air temperature":               "ambient_temp",
    "ambient air temper":                    "ambient_temp",
    "ambient_temp":                          "ambient_temp",
    "accelerator pedal p":                   "pedal_d",
    "accelerator pedal position d":         "pedal_d",
    "pedal_d":                               "pedal_d",
    "accelerator pedal p.1":                 "pedal_e",
    "accelerator pedal position e":         "pedal_e",
    "pedal_e":                               "pedal_e",
}

def load_obd_csv(path: str):
    """Load OBD CSV, normalising column names. Returns (headers_original, rows_dicts)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        raw_rows = list(csv.DictReader(f))

    if not raw_rows:
        sys.exit(f"[ERROR] OBD CSV is empty: {path}")

    print(f"[OBD] Loaded {len(raw_rows)} rows from {Path(path).name}")
    print(f"[OBD] Columns: {list(raw_rows[0].keys())}")
    return raw_rows


def obd_relative_seconds(rows) -> List[float]:
    """
    Extract relative seconds (from row 0 = t=0) for each OBD row.
    Tries 'time'/'device time' column first (HH:MM:SS), then uses row index * 1 s.
    """
    time_key = None
    for k in rows[0].keys():
        if k.strip().lower() in ("time", "device time"):
            time_key = k
            break

    if time_key is None:
        print("[OBD] No time column found — assuming 1 s/row")
        return [float(i) for i in range(len(rows))]

    def _hms(s):
        """'19:55:06' → seconds since midnight."""
        try:
            parts = s.strip().split(":")
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
        except Exception:
            pass
        return None

    anchor = None
    rel = []
    for row in rows:
        val = _hms(row.get(time_key, ""))
        if val is None:
            rel.append(float(len(rel)))
            continue
        if anchor is None:
            anchor = val
        elapsed = val - anchor
        # Handle midnight wrap-around
        if elapsed < -1:
            elapsed += 86400
        rel.append(elapsed)
    return rel


# ── interpolator ──────────────────────────────────────────────────────────────

def build_gps_interpolator(gps_points: List[Dict], loop: bool = False):
    """
    Returns a function: rel_seconds → {lat, lon, ele, bearing, speed_ms, gps_fix}
    'rel_seconds' is seconds from the start of the GPS track.
    If loop=True, the GPS track loops indefinitely.
    """
    if not gps_points:
        sys.exit("[ERROR] GPS file has no valid points.")

    # Convert to relative seconds
    t0 = gps_points[0]["utc"]
    rel_times = [(p["utc"] - t0).total_seconds() for p in gps_points]
    total_dur = rel_times[-1]

    print(f"[GPS] {len(gps_points)} points, duration={total_dur:.0f}s "
          f"({total_dur/60:.1f} min), avg interval="
          f"{total_dur/(max(len(gps_points)-1,1)):.1f}s")

    def interpolate(t_rel: float) -> Dict:
        nonlocal total_dur

        if loop and total_dur > 0:
            t_rel = t_rel % total_dur

        # Clamp to range
        t_rel = max(0.0, min(t_rel, total_dur))

        # Binary search for bracket
        lo, hi = 0, len(rel_times) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if rel_times[mid] <= t_rel:
                lo = mid
            else:
                hi = mid

        if lo == hi or rel_times[hi] == rel_times[lo]:
            p = gps_points[lo]
            return {**p, "gps_fix": 1}

        frac = (t_rel - rel_times[lo]) / (rel_times[hi] - rel_times[lo])
        a, b = gps_points[lo], gps_points[hi]
        return {
            "lat":       _lerp(a["lat"],      b["lat"],      frac),
            "lon":       _lerp(a["lon"],      b["lon"],      frac),
            "ele":       _lerp(a["ele"],      b["ele"],      frac),
            "bearing":   _bearing_lerp(a["bearing"], b["bearing"], frac),
            "speed_ms":  _lerp(a["speed_ms"], b["speed_ms"], frac),
            "gps_fix":   0,   # 0 = interpolated, 1 = exact GPS fix
        }

    return interpolate, total_dur


# ── main merge ────────────────────────────────────────────────────────────────

def merge(obd_path: str, gps_path: str, output_path: str, loop: bool):
    gps_points = detect_and_load_gps(gps_path)
    obd_rows = load_obd_csv(obd_path)
    obd_rel = obd_relative_seconds(obd_rows)

    interpolate, gps_dur = build_gps_interpolator(gps_points, loop=loop)

    obd_dur = obd_rel[-1] if obd_rel else 0
    print(f"[OBD] Duration: {obd_dur:.0f}s ({obd_dur/60:.1f} min)")

    if not loop and obd_dur > gps_dur:
        pct = int(100 * gps_dur / obd_dur)
        print(f"[WARN] OBD is {obd_dur:.0f}s but GPS is only {gps_dur:.0f}s "
              f"({pct}% coverage). Last GPS fix will be held for remaining rows.")
        print(f"       Re-run with --loop to cycle the GPS track instead.")

    # Build output rows
    gps_cols = ["lat", "lon", "elevation_m", "gps_bearing", "gps_speed_ms", "gps_fix"]
    out_rows = []
    for i, row in enumerate(obd_rows):
        gps = interpolate(obd_rel[i])
        out_row = dict(row)
        out_row["lat"]          = round(gps["lat"],      7)
        out_row["lon"]          = round(gps["lon"],      7)
        out_row["elevation_m"]  = round(gps["ele"],      1)
        out_row["gps_bearing"]  = round(gps["bearing"],  1)
        out_row["gps_speed_ms"] = round(gps["speed_ms"], 3)
        out_row["gps_fix"]      = gps["gps_fix"]
        out_rows.append(out_row)

    # Write output
    all_cols = list(obd_rows[0].keys()) + gps_cols
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\n[OK] Merged CSV written → {output_path}")
    print(f"     {len(out_rows)} rows, {len(all_cols)} columns")
    print(f"     GPS columns added: {gps_cols}")
    print(f"\nDrop this file into datasets/ and upload via the simulator.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Merge OBD-II CSV with GPS log for AutoVue presentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--obd",  required=True, help="Path to OBD-II CSV (any column format)")
    ap.add_argument("--gps",  required=True, help="Path to GPS file (.gpx, GPSLogger CSV, track_points CSV)")
    ap.add_argument("-o",     "--output",    help="Output path (default: <obd>_with_gps.csv)")
    ap.add_argument("--loop", action="store_true",
                    help="Loop GPS track if OBD drive is longer than GPS recording")
    args = ap.parse_args()

    out = args.output or str(Path(args.obd).with_stem(Path(args.obd).stem + "_with_gps"))
    merge(args.obd, args.gps, out, loop=args.loop)
