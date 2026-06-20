"""Train-only residual-bin correction for online-current prediction CSVs.

This script fits residual lookup tables on train predictions only, searches
correction hyperparameters on validation predictions, and optionally applies the
selected validation policy to a final test prediction CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _edges(values: str) -> np.ndarray:
    if values == "anchor":
        return np.asarray([-np.inf, 0.5, 2.0, 5.0, 8.0, 12.0, np.inf], dtype=np.float64)
    if values == "pred":
        return np.asarray([-np.inf, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, np.inf], dtype=np.float64)
    return np.asarray([float(v) for v in values.split(",")], dtype=np.float64)


def _base_prediction(frame: pd.DataFrame, base_col: str) -> np.ndarray:
    if base_col == "equal4":
        member_cols = sorted(col for col in frame.columns if col.startswith("member_pred_"))
        if not member_cols:
            raise ValueError("base_col=equal4 requires member_pred_* columns.")
        return frame[member_cols].astype(float).to_numpy(dtype=np.float64).mean(axis=1)
    if base_col not in frame.columns:
        raise ValueError(f"Missing base prediction column: {base_col}")
    return pd.to_numeric(frame[base_col], errors="coerce").to_numpy(dtype=np.float64)


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges[1:-1], right=False).astype(np.int64)


def _fit_table(
    train: pd.DataFrame,
    base_col: str,
    anchor_edges: np.ndarray,
    pred_edges: np.ndarray,
    min_count: int,
    shrinkage: float,
    clip_abs: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    y = pd.to_numeric(train["target_fms_now"], errors="coerce").to_numpy(dtype=np.float64)
    pred = _base_prediction(train, base_col)
    anchor = pd.to_numeric(train["anchor_fms"], errors="coerce").to_numpy(dtype=np.float64)
    residual = y - pred
    valid = np.isfinite(y) & np.isfinite(pred) & np.isfinite(anchor)
    residual = residual[valid]
    anchor = anchor[valid]
    pred = pred[valid]
    global_mean = float(np.mean(residual)) if residual.size else 0.0
    a_idx = _bin_indices(anchor, anchor_edges)
    p_idx = _bin_indices(pred, pred_edges)
    shape = (len(anchor_edges) - 1, len(pred_edges) - 1)
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.float64)
    for ai, pi, r in zip(a_idx, p_idx, residual):
        sums[ai, pi] += float(r)
        counts[ai, pi] += 1.0
    means = np.divide(sums, counts, out=np.full(shape, global_mean, dtype=np.float64), where=counts > 0)
    shrink = counts / (counts + float(shrinkage))
    table = global_mean * (1.0 - shrink) + means * shrink
    table = np.where(counts >= float(min_count), table, global_mean)
    table = np.clip(table, -float(clip_abs), float(clip_abs))
    return table, counts, global_mean


def _apply_table(
    frame: pd.DataFrame,
    base_col: str,
    table: np.ndarray,
    anchor_edges: np.ndarray,
    pred_edges: np.ndarray,
    strength: float,
    low_anchor_max: float,
    pred_max: float,
    correction_floor: float,
) -> Tuple[np.ndarray, float, float]:
    pred = _base_prediction(frame, base_col)
    anchor = pd.to_numeric(frame["anchor_fms"], errors="coerce").to_numpy(dtype=np.float64)
    a_idx = _bin_indices(anchor, anchor_edges)
    p_idx = _bin_indices(pred, pred_edges)
    correction = table[a_idx, p_idx] * float(strength)
    active = np.isfinite(pred) & np.isfinite(anchor)
    if np.isfinite(low_anchor_max):
        active = active & (anchor <= float(low_anchor_max))
    if np.isfinite(pred_max):
        active = active & (pred <= float(pred_max))
    if np.isfinite(correction_floor):
        active = active & (correction <= float(correction_floor))
    correction = np.where(active, correction, 0.0)
    out = np.clip(pred + correction, 0.0, 20.0)
    return out, float(np.mean(active)) if active.size else 0.0, float(np.mean(correction)) if correction.size else 0.0


def _high_metrics(y: np.ndarray, pred: np.ndarray, threshold: float) -> Dict[str, float]:
    true = y >= float(threshold)
    hit = pred >= float(threshold)
    tp = float(np.sum(true & hit))
    fp = float(np.sum(~true & hit))
    fn = float(np.sum(true & ~hit))
    tn = float(np.sum(~true & ~hit))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        f"high{threshold:g}_precision": precision,
        f"high{threshold:g}_recall": recall,
        f"high{threshold:g}_f1": f1,
        f"high{threshold:g}_fpr": fp / (fp + tn) if fp + tn > 0 else 0.0,
        f"high{threshold:g}_fnr": fn / (tp + fn) if tp + fn > 0 else 0.0,
    }


def _metrics(frame: pd.DataFrame, pred: np.ndarray) -> Dict[str, float]:
    y = pd.to_numeric(frame["target_fms_now"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    err = pred - y
    low = (y >= 0.0) & (y < 2.0)
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    row = {
        "n": int(y.size),
        "mae": float(np.mean(np.abs(err))) if y.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(err * err))) if y.size else float("nan"),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"),
        "original_low_0_2_n": int(np.sum(low)),
        "original_low_0_2_bias": float(np.mean(err[low])) if np.any(low) else float("nan"),
        "original_low_0_2_mae": float(np.mean(np.abs(err[low]))) if np.any(low) else float("nan"),
        "prediction_mean": float(np.mean(pred)) if y.size else float("nan"),
        "target_mean": float(np.mean(y)) if y.size else float("nan"),
        "prediction_std": float(np.std(pred)) if y.size else float("nan"),
        "target_std": float(np.std(y)) if y.size else float("nan"),
    }
    row.update(_high_metrics(y, pred, 8.0))
    row.update(_high_metrics(y, pred, 12.0))
    row["goal_composite_strict120"] = (
        float(row["mae"])
        + 0.25 * max(0.0, float(row["original_low_0_2_bias"]) - 2.5)
        + 2.0 * max(0.0, 0.70 - float(row["r2"]))
        + 0.5 * max(0.0, 0.76 - float(row["high12_f1"]))
    )
    row["goal_pass_count"] = int(row["mae"] <= 1.8) + int(row["r2"] >= 0.75) + int(row["original_low_0_2_bias"] <= 2.5)
    return row


def _score(row: Mapping[str, Any], mode: str) -> Tuple[float, ...]:
    if mode == "score_only":
        return (
            float(row["goal_composite_strict120"]),
            float(row["mae"]),
            max(0.0, float(row["original_low_0_2_bias"]) - 2.5),
            -float(row["r2"]),
        )
    if mode == "low_then_score":
        return (
            max(0.0, float(row["original_low_0_2_bias"]) - 2.5),
            float(row["goal_composite_strict120"]),
            float(row["mae"]),
            -float(row["r2"]),
        )
    raise ValueError("selection_mode must be score_only or low_then_score.")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_predictions(frame: pd.DataFrame, pred: np.ndarray, out_path: Path, label: str, split: str) -> None:
    out = frame.copy()
    out["base_predicted_fms_now"] = _base_prediction(frame, "predicted_fms_now") if "predicted_fms_now" in frame.columns else np.nan
    out["run_name"] = label
    out["model_name"] = "online_current_train_residual_bins"
    out["split"] = split
    out["predicted_fms_now"] = pred
    out["fms_absolute_error"] = np.abs(pred - pd.to_numeric(out["target_fms_now"], errors="coerce").to_numpy(dtype=np.float64))
    if "alarm_caution" in out.columns:
        out["alarm_caution"] = out["predicted_fms_now"] >= 8.0
    if "alarm_warning_high_fms" in out.columns:
        out["alarm_warning_high_fms"] = out["predicted_fms_now"] >= 12.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def _grid(values: Sequence[float]) -> Iterable[float]:
    for value in values:
        yield float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_col", default="predicted_fms_now")
    parser.add_argument("--anchor_edges", default="anchor")
    parser.add_argument("--pred_edges", default="pred")
    parser.add_argument("--min_counts", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--shrinkages", nargs="+", type=float, default=[20.0, 50.0, 100.0, 200.0])
    parser.add_argument("--clip_abs_values", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--low_anchor_maxs", nargs="+", type=float, default=[2.0, 5.0, 8.0, float("inf")])
    parser.add_argument("--pred_maxs", nargs="+", type=float, default=[4.0, 5.0, 6.0, 8.0, float("inf")])
    parser.add_argument("--correction_floors", nargs="+", type=float, default=[0.0, float("inf")])
    parser.add_argument("--selection_mode", choices=["score_only", "low_then_score"], default="score_only")
    parser.add_argument("--label", default="train_residual_bins")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv).sort_values(["session_id", "current_index"]).reset_index(drop=True)
    val = pd.read_csv(args.val_csv).sort_values(["session_id", "current_index"]).reset_index(drop=True)
    anchor_edges = _edges(args.anchor_edges)
    pred_edges = _edges(args.pred_edges)
    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_table: Optional[np.ndarray] = None
    for min_count in args.min_counts:
        for shrinkage in args.shrinkages:
            for clip_abs in args.clip_abs_values:
                table, counts, global_mean = _fit_table(
                    train,
                    args.base_col,
                    anchor_edges,
                    pred_edges,
                    int(min_count),
                    float(shrinkage),
                    float(clip_abs),
                )
                for strength in _grid(args.strengths):
                    for low_anchor_max in _grid(args.low_anchor_maxs):
                        for pred_max in _grid(args.pred_maxs):
                            for correction_floor in _grid(args.correction_floors):
                                pred, active_rate, correction_mean = _apply_table(
                                    val,
                                    args.base_col,
                                    table,
                                    anchor_edges,
                                    pred_edges,
                                    strength,
                                    low_anchor_max,
                                    pred_max,
                                    correction_floor,
                                )
                                row: Dict[str, Any] = {
                                    "base_col": args.base_col,
                                    "min_count": int(min_count),
                                    "shrinkage": float(shrinkage),
                                    "clip_abs": float(clip_abs),
                                    "strength": float(strength),
                                    "low_anchor_max": float(low_anchor_max),
                                    "pred_max": float(pred_max),
                                    "correction_floor": float(correction_floor),
                                    "active_rate": active_rate,
                                    "correction_mean": correction_mean,
                                    "global_residual_mean": global_mean,
                                }
                                row.update(_metrics(val, pred))
                                rows.append(row)
                                if best is None or _score(row, args.selection_mode) < _score(best, args.selection_mode):
                                    best = row
                                    best_table = table.copy()
    assert best is not None and best_table is not None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "validation_residual_bin_grid.csv", rows)
    val_pred, _, _ = _apply_table(
        val,
        args.base_col,
        best_table,
        anchor_edges,
        pred_edges,
        float(best["strength"]),
        float(best["low_anchor_max"]),
        float(best["pred_max"]),
        float(best["correction_floor"]),
    )
    _save_predictions(val, val_pred, out_dir / "val_predictions.csv", args.label, "val")
    result: Dict[str, Any] = {
        "selection_mode": args.selection_mode,
        "label": args.label,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "selected_validation": best,
        "validation_metrics": _metrics(val, val_pred),
        "anchor_edges": anchor_edges.tolist(),
        "pred_edges": pred_edges.tolist(),
        "residual_table": best_table.tolist(),
    }
    if args.test_csv:
        test = pd.read_csv(args.test_csv).sort_values(["session_id", "current_index"]).reset_index(drop=True)
        test_pred, _, _ = _apply_table(
            test,
            args.base_col,
            best_table,
            anchor_edges,
            pred_edges,
            float(best["strength"]),
            float(best["low_anchor_max"]),
            float(best["pred_max"]),
            float(best["correction_floor"]),
        )
        _save_predictions(test, test_pred, out_dir / "test_predictions.csv", args.label, "test")
        result["test_metrics"] = _metrics(test, test_pred)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("selected_validation", "validation_metrics")}, indent=2))


if __name__ == "__main__":
    main()
