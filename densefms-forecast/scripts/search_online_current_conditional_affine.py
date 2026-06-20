"""Fit condition-binned affine corrections for online current-FMS predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

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
    target = frame["target_fms_now"].to_numpy(dtype=np.float64)
    pred = frame["predicted_fms_now"].to_numpy(dtype=np.float64)
    err = pred - target
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "prediction_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(target)),
        "prediction_std": float(np.std(pred)),
        "target_std": float(np.std(target)),
        "n": float(len(frame)),
    }


def _bin_ids(frame: pd.DataFrame, specs: Sequence[tuple[str, Sequence[float]]]) -> np.ndarray:
    ids = np.zeros(len(frame), dtype=np.int64)
    for column, bins in specs:
        if column not in frame.columns:
            raise ValueError(f"Missing condition column: {column}")
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        edges = np.asarray(list(bins), dtype=np.float64)
        current = np.digitize(values, edges, right=False).astype(np.int64)
        ids = ids * (len(edges) + 1) + current
    return ids


def _fit_affine(pred: np.ndarray, target: np.ndarray, ridge: float) -> tuple[float, float]:
    x = np.column_stack([pred.astype(np.float64), np.ones(len(pred), dtype=np.float64)])
    y = target.astype(np.float64)
    if ridge > 0.0:
        xtx = x.T @ x
        penalty = np.asarray([[float(ridge), 0.0], [0.0, 0.0]], dtype=np.float64)
        scale, bias = np.linalg.solve(xtx + penalty, x.T @ y)
    else:
        scale, bias = np.linalg.lstsq(x, y, rcond=None)[0]
    return float(scale), float(bias)


def _fit_conditional_affine(
    frame: pd.DataFrame,
    bin_ids: np.ndarray,
    min_bin_count: int,
    ridge: float,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, List[Dict[str, float]]]:
    target = frame["target_fms_now"].to_numpy(dtype=np.float64)
    raw = frame["predicted_fms_now"].to_numpy(dtype=np.float64)
    global_scale, global_bias = _fit_affine(raw, target, ridge)
    corrected = np.empty_like(raw)
    per_bin: List[Dict[str, float]] = []

    for bin_id in sorted(int(v) for v in np.unique(bin_ids)):
        mask = bin_ids == bin_id
        n = int(np.sum(mask))
        if n >= int(min_bin_count):
            scale, bias = _fit_affine(raw[mask], target[mask], ridge)
            fallback = False
        else:
            scale, bias = global_scale, global_bias
            fallback = True
        pred = np.clip(scale * raw[mask] + bias, clip_min, clip_max)
        corrected[mask] = pred
        err = pred - target[mask]
        per_bin.append(
            {
                "bin": float(bin_id),
                "scale": float(scale),
                "bias": float(bias),
                "fallback_global": float(fallback),
                "n": float(n),
                "mae": float(np.mean(np.abs(err))) if n > 0 else float("nan"),
                "rmse": float(np.sqrt(np.mean(err**2))) if n > 0 else float("nan"),
            }
        )
    return corrected, per_bin


def _write_output(
    frame: pd.DataFrame,
    out_dir: Path,
    split: str,
    label: str,
    corrected: np.ndarray,
    source_path: Path,
    condition_payload: Dict[str, object],
    ordinal_bins: Sequence[float],
) -> Dict[str, float]:
    out = frame.copy()
    out["base_predicted_fms_now"] = out["predicted_fms_now"].astype(float)
    out["run_name"] = label
    out["model_name"] = "online_current_conditional_affine"
    out["split"] = split
    out["predicted_fms_now"] = corrected
    out["fms_absolute_error"] = np.abs(
        out["predicted_fms_now"].to_numpy(dtype=np.float64) - out["target_fms_now"].to_numpy(dtype=np.float64)
    )
    if "ordinal_bin_pred" in out.columns:
        out["ordinal_bin_pred"] = _ordinal_bins(out["predicted_fms_now"].to_numpy(dtype=np.float64), ordinal_bins)
    if "alarm_caution" in out.columns:
        out["alarm_caution"] = out["predicted_fms_now"] >= 8.0
    if "alarm_warning_high_fms" in out.columns:
        out["alarm_warning_high_fms"] = out["predicted_fms_now"] >= 12.0
    if "final_warning" in out.columns and "alarm_warning_high_fms" in out.columns:
        rapid = out["alarm_warning_rapid_rise"] if "alarm_warning_rapid_rise" in out.columns else False
        out["final_warning"] = out["alarm_warning_high_fms"] | rapid

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"{split}_predictions.csv", index=False)
    metrics = _metrics(out)
    payload = {
        "task": {
            "conditional_affine": True,
            "validation_fitted_postprocess": split == "val",
            "test_eval_skipped": split != "test",
        },
        "split": split,
        "source": str(source_path),
        "condition": condition_payload,
        "metrics": metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit condition-binned affine corrections on saved predictions.")
    parser.add_argument("--input", required=True, help="Run dir or prediction CSV.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--label", default=None)
    parser.add_argument("--condition_column", required=True)
    parser.add_argument("--condition_bins", nargs="*", type=float, default=[])
    parser.add_argument("--condition_column2", default=None)
    parser.add_argument("--condition_bins2", nargs="*", type=float, default=[])
    parser.add_argument("--condition_column3", default=None)
    parser.add_argument("--condition_bins3", nargs="*", type=float, default=[])
    parser.add_argument("--min_bin_count", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--clip_min", type=float, default=0.0)
    parser.add_argument("--clip_max", type=float, default=20.0)
    parser.add_argument("--ordinal_bins", nargs="*", type=float, default=DEFAULT_ORDINAL_BINS)
    args = parser.parse_args()

    source_path = _prediction_path(Path(args.input), args.split)
    frame = pd.read_csv(source_path)
    required = {"target_fms_now", "predicted_fms_now"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source_path} is missing required columns: {sorted(missing)}")

    specs: List[tuple[str, Sequence[float]]] = [(args.condition_column, args.condition_bins)]
    if args.condition_column2:
        specs.append((args.condition_column2, args.condition_bins2))
    if args.condition_column3:
        if not args.condition_column2:
            raise ValueError("--condition_column3 requires --condition_column2.")
        specs.append((args.condition_column3, args.condition_bins3))

    ids = _bin_ids(frame, specs)
    corrected, per_bin = _fit_conditional_affine(
        frame,
        ids,
        min_bin_count=args.min_bin_count,
        ridge=max(0.0, float(args.ridge)),
        clip_min=float(args.clip_min),
        clip_max=float(args.clip_max),
    )
    condition_payload = {
        "columns": [column for column, _bins in specs],
        "bins": [[float(v) for v in bins] for _column, bins in specs],
        "min_bin_count": int(args.min_bin_count),
        "ridge": float(max(0.0, args.ridge)),
        "clip_min": float(args.clip_min),
        "clip_max": float(args.clip_max),
        "per_bin": per_bin,
    }
    out_dir = Path(args.out_dir)
    label = args.label or out_dir.name
    metrics = _write_output(
        frame,
        out_dir,
        args.split,
        label,
        corrected,
        source_path,
        condition_payload,
        args.ordinal_bins,
    )
    print(json.dumps({"metrics": metrics, "out_dir": str(out_dir), "condition": condition_payload}, indent=2))


if __name__ == "__main__":
    main()
