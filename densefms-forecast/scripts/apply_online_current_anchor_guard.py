"""Apply a calibration-anchor guard to online-current prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


def _metrics(
    frame: pd.DataFrame,
    low_bin_min: float = 0.0,
    low_bin_max: float = 2.0,
    low_bin_upper_exclusive: bool = False,
) -> Dict[str, float]:
    target = pd.to_numeric(frame["target_fms_now"], errors="coerce").to_numpy(dtype=np.float64)
    pred = pd.to_numeric(frame["predicted_fms_now"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(target) & np.isfinite(pred)
    target = target[mask]
    pred = pred[mask]
    err = pred - target
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    low = target >= float(low_bin_min)
    if bool(low_bin_upper_exclusive):
        low = low & (target < float(low_bin_max))
    else:
        low = low & (target <= float(low_bin_max))
    row: Dict[str, float] = {
        "n": float(target.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "low_signed_bias": float(np.mean(err[low])) if np.any(low) else float("nan"),
        "low_mae": float(np.mean(np.abs(err[low]))) if np.any(low) else float("nan"),
    }
    for threshold in (8.0, 12.0):
        y = target >= threshold
        p = pred >= threshold
        tp = float(np.sum(y & p))
        fp = float(np.sum(~y & p))
        fn = float(np.sum(y & ~p))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        row[f"high{threshold:g}_precision"] = precision
        row[f"high{threshold:g}_recall"] = recall
        row[f"high{threshold:g}_f1"] = f1
    return row


def apply_guard(
    frame: pd.DataFrame,
    anchor_threshold: float,
    margin: float,
    strength: float,
    anchor_column: str = "anchor_fms",
) -> pd.DataFrame:
    out = frame.copy()
    pred = pd.to_numeric(out["predicted_fms_now"], errors="coerce").to_numpy(dtype=np.float64)
    anchor = pd.to_numeric(out[anchor_column], errors="coerce").to_numpy(dtype=np.float64)
    cap = anchor + float(margin)
    guard = np.isfinite(pred) & np.isfinite(anchor) & (anchor <= float(anchor_threshold)) & (pred > cap)
    corrected = pred.copy()
    corrected[guard] = pred[guard] - float(strength) * (pred[guard] - cap[guard])
    corrected = np.clip(corrected, 0.0, 20.0)
    out["base_predicted_fms_now"] = pred
    out["anchor_guard_applied"] = guard
    out["anchor_guard_cap"] = cap
    out["predicted_fms_now"] = corrected
    out["fms_absolute_error"] = np.abs(corrected - pd.to_numeric(out["target_fms_now"], errors="coerce").to_numpy(dtype=np.float64))
    if "alarm_caution" in out.columns:
        out["alarm_caution"] = out["predicted_fms_now"] >= 8.0
    if "alarm_warning_high_fms" in out.columns:
        out["alarm_warning_high_fms"] = out["predicted_fms_now"] >= 12.0
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
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
    parser.add_argument("--anchor_column", default="anchor_fms")
    parser.add_argument("--anchor_thresholds", nargs="+", type=float, default=[1.0, 2.0, 3.0, 5.0])
    parser.add_argument("--margins", nargs="+", type=float, default=[0.0, 1.0, 2.0, 3.0, 4.0])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--select_low_bias_max", type=float, default=2.5)
    parser.add_argument("--low_bin_min", type=float, default=0.0)
    parser.add_argument("--low_bin_max", type=float, default=2.0)
    parser.add_argument("--low_bin_upper_exclusive", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None
    best_frame: pd.DataFrame | None = None
    for anchor_threshold in args.anchor_thresholds:
        for margin in args.margins:
            for strength in args.strengths:
                corrected = apply_guard(
                    frame,
                    anchor_threshold=anchor_threshold,
                    margin=margin,
                    strength=strength,
                    anchor_column=args.anchor_column,
                )
                metrics = _metrics(
                    corrected,
                    low_bin_min=float(args.low_bin_min),
                    low_bin_max=float(args.low_bin_max),
                    low_bin_upper_exclusive=bool(args.low_bin_upper_exclusive),
                )
                row: Dict[str, Any] = {
                    "anchor_threshold": float(anchor_threshold),
                    "margin": float(margin),
                    "strength": float(strength),
                    "guard_rate": float(corrected["anchor_guard_applied"].mean()),
                    **metrics,
                }
                rows.append(row)
                feasible = float(row["low_signed_bias"]) <= float(args.select_low_bias_max)
                if best_row is None:
                    best_row = row
                    best_frame = corrected
                else:
                    best_feasible = float(best_row["low_signed_bias"]) <= float(args.select_low_bias_max)
                    if (feasible and not best_feasible) or (
                        feasible == best_feasible
                        and (float(row["mae"]), abs(float(row["low_signed_bias"]))) < (float(best_row["mae"]), abs(float(best_row["low_signed_bias"])))
                    ):
                        best_row = row
                        best_frame = corrected

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "anchor_guard_grid.csv", rows)
    if best_frame is None or best_row is None:
        raise ValueError("No anchor guard candidate was evaluated.")
    best_frame["run_name"] = args.label or out_dir.name
    best_frame["split"] = args.split
    best_frame.to_csv(out_dir / f"{args.split}_predictions.csv", index=False)
    payload = {
        "selected": best_row,
        "input_csv": args.input_csv,
        "low_bin": {
            "min": float(args.low_bin_min),
            "max": float(args.low_bin_max),
            "upper_exclusive": bool(args.low_bin_upper_exclusive),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
