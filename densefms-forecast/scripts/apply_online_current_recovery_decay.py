"""Apply high-anchor recovery decay to online-current prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


def _metrics(frame: pd.DataFrame) -> Dict[str, float]:
    target = pd.to_numeric(frame["target_fms_now"], errors="coerce").to_numpy(dtype=np.float64)
    pred = pd.to_numeric(frame["predicted_fms_now"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(target) & np.isfinite(pred)
    target = target[mask]
    pred = pred[mask]
    err = pred - target
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    original_low = target < 2.0
    return {
        "n": float(target.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "original_low_signed_bias": float(np.mean(err[original_low])) if np.any(original_low) else float("nan"),
        "original_low_mae": float(np.mean(np.abs(err[original_low]))) if np.any(original_low) else float("nan"),
    }


def apply_decay(
    frame: pd.DataFrame,
    anchor_threshold: float,
    start_seconds: float,
    duration_seconds: float,
    strength: float,
    floor: float,
    calibration_seconds: float,
) -> pd.DataFrame:
    out = frame.copy()
    pred = pd.to_numeric(out["predicted_fms_now"], errors="coerce").to_numpy(dtype=np.float64)
    anchor = pd.to_numeric(out["anchor_fms"], errors="coerce").to_numpy(dtype=np.float64)
    current_time = pd.to_numeric(out["current_time"], errors="coerce").to_numpy(dtype=np.float64)
    elapsed = current_time - float(calibration_seconds)
    progress = np.clip((elapsed - float(start_seconds)) / max(float(duration_seconds), 1e-6), 0.0, 1.0)
    high_anchor = np.isfinite(anchor) & (anchor >= float(anchor_threshold))
    available_drop = np.maximum(anchor - float(floor), 0.0)
    correction = np.where(high_anchor, float(strength) * available_drop * progress, 0.0)
    corrected = np.clip(pred - correction, 0.0, 20.0)
    out["base_predicted_fms_now"] = pred
    out["recovery_decay_correction"] = correction
    out["recovery_decay_applied"] = correction > 0.0
    out["predicted_fms_now"] = corrected
    out["fms_absolute_error"] = np.abs(corrected - pd.to_numeric(out["target_fms_now"], errors="coerce").to_numpy(dtype=np.float64))
    if "alarm_caution" in out.columns:
        out["alarm_caution"] = out["predicted_fms_now"] >= 8.0
    if "alarm_warning_high_fms" in out.columns:
        out["alarm_warning_high_fms"] = out["predicted_fms_now"] >= 12.0
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--label", default=None)
    parser.add_argument("--anchor_thresholds", nargs="+", type=float, default=[5.0, 8.0, 10.0, 12.0])
    parser.add_argument("--start_seconds", nargs="+", type=float, default=[0.0, 15.0, 30.0])
    parser.add_argument("--duration_seconds", nargs="+", type=float, default=[30.0, 60.0, 90.0])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--floors", nargs="+", type=float, default=[0.0, 2.0, 5.0])
    parser.add_argument("--calibration_seconds", type=float, default=120.0)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None
    best_frame: pd.DataFrame | None = None
    for anchor_threshold in args.anchor_thresholds:
        for start in args.start_seconds:
            for duration in args.duration_seconds:
                for strength in args.strengths:
                    for floor in args.floors:
                        candidate = apply_decay(
                            frame,
                            anchor_threshold=anchor_threshold,
                            start_seconds=start,
                            duration_seconds=duration,
                            strength=strength,
                            floor=floor,
                            calibration_seconds=args.calibration_seconds,
                        )
                        metrics = _metrics(candidate)
                        row: Dict[str, Any] = {
                            "anchor_threshold": float(anchor_threshold),
                            "start_seconds": float(start),
                            "duration_seconds": float(duration),
                            "strength": float(strength),
                            "floor": float(floor),
                            "decay_rate": float(candidate["recovery_decay_applied"].mean()),
                            **metrics,
                        }
                        rows.append(row)
                        if best_row is None or (float(row["mae"]), float(row["rmse"])) < (
                            float(best_row["mae"]),
                            float(best_row["rmse"]),
                        ):
                            best_row = row
                            best_frame = candidate
    if best_row is None or best_frame is None:
        raise ValueError("No recovery decay candidates evaluated.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "recovery_decay_grid.csv", rows)
    best_frame = best_frame.copy()
    best_frame["run_name"] = args.label or out_dir.name
    best_frame["split"] = args.split
    best_frame.to_csv(out_dir / f"{args.split}_predictions.csv", index=False)
    payload = {"selected": best_row, "input_csv": args.input_csv}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
