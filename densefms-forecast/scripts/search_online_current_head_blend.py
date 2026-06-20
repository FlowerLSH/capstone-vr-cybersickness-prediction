"""Validation-only blend search for saved regression/ordinal current-FMS heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


DEFAULT_ORDINAL_BINS = [0, 2, 4, 6, 8, 10, 12, 15, 20]


def _prediction_path(path: Path, split: str) -> Path:
    if path.is_file():
        return path
    direct = path / f"{split}_predictions.csv"
    if direct.exists():
        return direct
    nested = path / f"eval_{split}" / f"{split}_predictions.csv"
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Missing {split} prediction CSV under {path}")


def _ordinal_bins(values: np.ndarray, bins: Sequence[float]) -> np.ndarray:
    edges = np.asarray(list(bins), dtype=np.float64)
    return np.digitize(values, edges[1:-1], right=False).astype(int)


def _metrics(frame: pd.DataFrame) -> Dict[str, float]:
    y = frame["target_fms_now"].to_numpy(dtype=np.float64)
    pred = frame["predicted_fms_now"].to_numpy(dtype=np.float64)
    err = pred - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "prediction_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(y)),
        "prediction_std": float(np.std(pred)),
        "target_std": float(np.std(y)),
        "n": float(len(frame)),
    }


def _search_weight(frame: pd.DataFrame, weights: np.ndarray, metric: str) -> Dict[str, float]:
    y = frame["target_fms_now"].to_numpy(dtype=np.float64)
    reg = frame["predicted_fms_regression"].to_numpy(dtype=np.float64)
    ordinal = frame["predicted_fms_ordinal"].to_numpy(dtype=np.float64)
    rows = []
    for weight in weights:
        pred = (1.0 - float(weight)) * reg + float(weight) * ordinal
        err = pred - y
        rows.append(
            {
                "weight": float(weight),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
            }
        )
    key = "rmse" if str(metric).lower() == "rmse" else "mae"
    return min(rows, key=lambda row: (float(row[key]), float(row["mae"]), float(row["rmse"])))


def _search_conditional_weights(
    frame: pd.DataFrame,
    bins: np.ndarray,
    weights: np.ndarray,
    metric: str,
) -> Dict[str, object]:
    y = frame["target_fms_now"].to_numpy(dtype=np.float64)
    reg = frame["predicted_fms_regression"].to_numpy(dtype=np.float64)
    ordinal = frame["predicted_fms_ordinal"].to_numpy(dtype=np.float64)
    chosen = np.zeros(int(np.max(bins)) + 1, dtype=np.float64)
    pred = np.empty_like(y)
    per_bin = []
    for bin_id in range(int(np.max(bins)) + 1):
        mask = bins == bin_id
        if not np.any(mask):
            per_bin.append({"bin": bin_id, "weight": 0.0, "mae": float("nan"), "rmse": float("nan"), "n": 0})
            continue
        sub = frame.loc[mask, :]
        best = _search_weight(sub, weights, metric)
        chosen[bin_id] = float(best["weight"])
        pred[mask] = (1.0 - chosen[bin_id]) * reg[mask] + chosen[bin_id] * ordinal[mask]
        per_bin.append(
            {
                "bin": bin_id,
                "weight": float(chosen[bin_id]),
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "n": int(np.sum(mask)),
            }
        )
    err = pred - y
    return {
        "weights": [float(v) for v in chosen],
        "per_bin": per_bin,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }


def _write_output(
    frame: pd.DataFrame,
    out_dir: Path,
    split: str,
    label: str,
    weight: float | np.ndarray,
    source_path: Path,
    ordinal_bins: Sequence[float],
    condition_payload: Dict[str, object] | None = None,
) -> Dict[str, float]:
    out = frame.copy()
    reg = out["predicted_fms_regression"].to_numpy(dtype=np.float64)
    ordinal = out["predicted_fms_ordinal"].to_numpy(dtype=np.float64)
    if isinstance(weight, np.ndarray):
        ordinal_weight = weight.astype(np.float64)
        pred = (1.0 - ordinal_weight) * reg + ordinal_weight * ordinal
    else:
        ordinal_weight = np.full_like(reg, float(weight), dtype=np.float64)
        pred = (1.0 - float(weight)) * reg + float(weight) * ordinal
    out["run_name"] = label
    out["model_name"] = "online_current_head_blend"
    out["split"] = split
    out["predicted_fms_now"] = pred
    out["fms_absolute_error"] = np.abs(pred - out["target_fms_now"].to_numpy(dtype=np.float64))
    if "ordinal_bin_pred" in out.columns:
        out["ordinal_bin_pred"] = _ordinal_bins(pred, ordinal_bins)
    if "alarm_caution" in out.columns:
        out["alarm_caution"] = out["predicted_fms_now"] >= 8.0
    if "alarm_warning_high_fms" in out.columns:
        out["alarm_warning_high_fms"] = out["predicted_fms_now"] >= 12.0
    if "final_warning" in out.columns:
        rapid = out["alarm_warning_rapid_rise"] if "alarm_warning_rapid_rise" in out.columns else False
        out["final_warning"] = out.get("alarm_warning_high_fms", False) | rapid
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"{split}_predictions.csv", index=False)
    metrics = _metrics(out)
    payload = {
        "task": {
            "head_blend": True,
            "test_eval_skipped": split != "test",
        },
        "split": split,
        "source": str(source_path),
        "regression_weight": float(1.0 - float(weight)) if not isinstance(weight, np.ndarray) else None,
        "ordinal_weight": float(weight) if not isinstance(weight, np.ndarray) else None,
        "condition": condition_payload,
        "metrics": metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Search or apply regression/ordinal current-head blend weights.")
    parser.add_argument("--input", required=True, help="Run dir or prediction CSV containing saved regression/ordinal heads.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--label", default=None)
    parser.add_argument("--weight", type=float, default=None, help="Ordinal weight. If omitted, a validation grid search is used.")
    parser.add_argument("--metric", choices=["mae", "rmse"], default="mae")
    parser.add_argument("--grid_step", type=float, default=0.01)
    parser.add_argument("--condition_column", default=None, help="Optional column used for per-bin ordinal weights.")
    parser.add_argument("--condition_bins", nargs="*", type=float, default=None)
    parser.add_argument("--condition_column2", default=None, help="Optional second condition column for 2D bins.")
    parser.add_argument("--condition_bins2", nargs="*", type=float, default=None)
    parser.add_argument("--condition_column3", default=None, help="Optional third condition column for 3D bins.")
    parser.add_argument("--condition_bins3", nargs="*", type=float, default=None)
    parser.add_argument("--ordinal_bins", nargs="*", type=float, default=DEFAULT_ORDINAL_BINS)
    args = parser.parse_args()

    source_path = _prediction_path(Path(args.input), args.split)
    frame = pd.read_csv(source_path)
    required = {"target_fms_now", "predicted_fms_regression", "predicted_fms_ordinal"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source_path} is missing required columns: {sorted(missing)}")

    condition_payload = None
    if args.condition_column:
        if args.weight is not None:
            raise ValueError("--weight cannot be combined with --condition_column.")
        if args.condition_column not in frame.columns:
            raise ValueError(f"Missing condition column: {args.condition_column}")
        if args.condition_column2 and args.condition_column2 not in frame.columns:
            raise ValueError(f"Missing second condition column: {args.condition_column2}")
        if args.condition_column3 and not args.condition_column2:
            raise ValueError("--condition_column3 requires --condition_column2.")
        if args.condition_column3 and args.condition_column3 not in frame.columns:
            raise ValueError(f"Missing third condition column: {args.condition_column3}")
        step = max(float(args.grid_step), 1e-6)
        weight_grid = np.arange(0.0, 1.0 + 0.5 * step, step, dtype=np.float64).clip(0.0, 1.0)
        condition_bins = [float(v) for v in (args.condition_bins or [])]
        condition_values = pd.to_numeric(frame[args.condition_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        bin_ids = np.digitize(condition_values, np.asarray(condition_bins, dtype=np.float64), right=False).astype(int)
        condition_bins2 = None
        if args.condition_column2:
            condition_bins2 = [float(v) for v in (args.condition_bins2 or [])]
            condition_values2 = (
                pd.to_numeric(frame[args.condition_column2], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            )
            bin_ids2 = np.digitize(condition_values2, np.asarray(condition_bins2, dtype=np.float64), right=False).astype(int)
            bin_ids = bin_ids * (len(condition_bins2) + 1) + bin_ids2
        condition_bins3 = None
        if args.condition_column3:
            condition_bins3 = [float(v) for v in (args.condition_bins3 or [])]
            condition_values3 = (
                pd.to_numeric(frame[args.condition_column3], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            )
            bin_ids3 = np.digitize(condition_values3, np.asarray(condition_bins3, dtype=np.float64), right=False).astype(int)
            bin_ids = bin_ids * (len(condition_bins3) + 1) + bin_ids3
        best = _search_conditional_weights(frame, bin_ids, weight_grid, args.metric)
        selected_weights = np.asarray(best["weights"], dtype=np.float64)
        weight = selected_weights[bin_ids]
        condition_payload = {
            "column": args.condition_column,
            "bins": condition_bins,
            "column2": args.condition_column2,
            "bins2": condition_bins2,
            "column3": args.condition_column3,
            "bins3": condition_bins3,
            "ordinal_weights": [float(v) for v in selected_weights],
            "per_bin": best["per_bin"],
        }
    elif args.weight is None:
        step = max(float(args.grid_step), 1e-6)
        weights = np.arange(0.0, 1.0 + 0.5 * step, step, dtype=np.float64)
        best = _search_weight(frame, weights.clip(0.0, 1.0), args.metric)
        weight = float(best["weight"])
    else:
        weight = max(0.0, min(1.0, float(args.weight)))
        best = _search_weight(frame, np.array([weight], dtype=np.float64), args.metric)

    out_dir = Path(args.out_dir)
    label = args.label or out_dir.name
    metrics = _write_output(frame, out_dir, args.split, label, weight, source_path, args.ordinal_bins, condition_payload)
    payload = {"selected": best, "metrics": metrics, "out_dir": str(out_dir)}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
