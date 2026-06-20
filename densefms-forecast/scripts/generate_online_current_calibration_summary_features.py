"""Generate leakage-safe calibration-only summary features from prediction CSVs.

The output format matches ``load_calibration_residual_features`` so it can be
fed into the existing calibration_residual_adapter path.  Features use only the
first ``calibration_steps`` rows of each source DenseFMS CSV.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from densefms_forecast.data import read_csv_robust
from densefms_forecast.utils import ensure_dir


MOTION_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
]


def _slope(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if int(finite.sum()) < 2:
        return 0.0
    y = values[finite].astype(np.float64)
    x = np.arange(values.size, dtype=np.float64)[finite]
    x = x - float(x.mean())
    denom = float(np.sum(x * x))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x * (y - float(y.mean()))) / denom)


def _stats(prefix: str, values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)].astype(np.float64)
    if finite.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_first": 0.0,
            f"{prefix}_last": 0.0,
            f"{prefix}_delta": 0.0,
            f"{prefix}_slope": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_first": float(finite[0]),
        f"{prefix}_last": float(finite[-1]),
        f"{prefix}_delta": float(finite[-1] - finite[0]),
        f"{prefix}_slope": _slope(values),
    }


def _resolve_source_path(source_file: str, source_root: Path) -> Optional[Path]:
    text = str(source_file or "").strip()
    if not text:
        return None
    normalized = Path(text.replace("\\", "/"))
    candidates = [normalized]
    if not normalized.is_absolute():
        candidates.extend([source_root / normalized, source_root / normalized.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _feature_dict(path: Path, calibration_steps: int) -> Dict[str, float]:
    raw, _meta = read_csv_robust(path)
    n = max(1, int(calibration_steps))
    cal = raw.iloc[:n].copy()
    fms = pd.to_numeric(cal.get("fms", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float64)
    row = _stats("calib_fms", fms)
    finite_fms = fms[np.isfinite(fms)]
    row["calib_fms_frac_low2"] = float(np.mean(finite_fms < 2.0)) if finite_fms.size else 0.0
    row["calib_fms_frac_low5"] = float(np.mean(finite_fms < 5.0)) if finite_fms.size else 0.0
    row["calib_fms_frac_high8"] = float(np.mean(finite_fms >= 8.0)) if finite_fms.size else 0.0
    row["calib_fms_frac_high12"] = float(np.mean(finite_fms >= 12.0)) if finite_fms.size else 0.0
    for window in (20, 60, 120):
        tail = fms[-min(window, fms.size) :]
        finite_tail = tail[np.isfinite(tail)]
        row[f"calib_fms_last{window}_mean"] = float(np.mean(finite_tail)) if finite_tail.size else 0.0
        row[f"calib_fms_last{window}_std"] = float(np.std(finite_tail)) if finite_tail.size else 0.0

    motion_arrays: List[np.ndarray] = []
    for col in MOTION_COLUMNS:
        if col in cal.columns:
            values = pd.to_numeric(cal[col], errors="coerce").to_numpy(dtype=np.float64)
            row.update(_stats(f"motion_calib_{col}", values))
            motion_arrays.append(values)
    if motion_arrays:
        motion = np.vstack(motion_arrays).T
        acc = motion[:, :3] if motion.shape[1] >= 3 else motion
        gyro = motion[:, 3:6] if motion.shape[1] >= 6 else motion[:, :0]
        row.update(_stats("motion_calib_acc_mag", np.linalg.norm(np.nan_to_num(acc, nan=0.0), axis=1)))
        if gyro.size:
            row.update(_stats("motion_calib_gyro_mag", np.linalg.norm(np.nan_to_num(gyro, nan=0.0), axis=1)))
        if motion.shape[0] > 1:
            row.update(_stats("motion_calib_diff_mag", np.linalg.norm(np.diff(np.nan_to_num(motion, nan=0.0), axis=0), axis=1)))
    return {key: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for key, value in row.items()}


def _split_path_arg(value: str) -> Tuple[str, Path]:
    if "=" in value:
        split, path = value.split("=", 1)
        return split, Path(path)
    path = Path(value)
    return path.parent.name, path


def _session_rows(frame: pd.DataFrame, split: str, source_root: Path) -> List[Dict[str, Any]]:
    required = {"source_file", "calibration_steps"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {sorted(missing)}")
    keys = ["source_file", "participant_id", "session_id", "calibration_steps"]
    rows: List[Dict[str, Any]] = []
    cache: Dict[Tuple[str, int], Dict[str, float]] = {}
    for item in frame.loc[:, [c for c in keys if c in frame.columns]].drop_duplicates().to_dict(orient="records"):
        source_file = str(item.get("source_file") or "")
        steps = int(float(item.get("calibration_steps") or 0))
        cache_key = (source_file, steps)
        if cache_key not in cache:
            path = _resolve_source_path(source_file, source_root)
            if path is None:
                raise FileNotFoundError(f"Could not resolve source_file={source_file!r}")
            cache[cache_key] = _feature_dict(path, steps)
        feature_dict = cache[cache_key]
        source_path = Path(source_file)
        rows.append(
            {
                "split": split,
                "session_key": source_file or str(item.get("session_id") or ""),
                "participant_id": item.get("participant_id"),
                "session_id": item.get("session_id"),
                "source_file": source_file,
                "source_file_name": source_path.name,
                "source_file_stem": source_path.stem,
                "feature_dict": feature_dict,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_csvs", nargs="+", required=True, help="split=path entries or raw CSV paths.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_root", default=".")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    session_rows: List[Dict[str, Any]] = []
    for spec in args.prediction_csvs:
        split, path = _split_path_arg(spec)
        frame = pd.read_csv(path)
        session_rows.extend(_session_rows(frame, split, source_root))

    feature_names = sorted({name for row in session_rows for name in row["feature_dict"].keys()})
    output_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    for row in session_rows:
        features = [float(row["feature_dict"].get(name, 0.0)) for name in feature_names]
        base = {key: value for key, value in row.items() if key != "feature_dict"}
        output_rows.append({**base, "feature_names": feature_names, "features": features})
        csv_row = dict(base)
        csv_row.update({f"feature__{name}": value for name, value in zip(feature_names, features)})
        csv_rows.append(csv_row)

    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    payload = {
        "feature_names": feature_names,
        "features": output_rows,
        "metadata": {
            "prediction_csvs": list(args.prediction_csvs),
            "source_root": str(source_root),
            "session_count": len(output_rows),
            "feature_count": len(feature_names),
            "leakage_policy": "Features use only the first calibration_steps rows of each DenseFMS source CSV.",
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(out_path.with_suffix(".csv"), index=False)
    print(json.dumps({"output": str(out_path), "sessions": len(output_rows), "features": len(feature_names)}, indent=2))


if __name__ == "__main__":
    main()
