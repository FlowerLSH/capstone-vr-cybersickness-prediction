"""Validation-only strict-goal search over 4-candidate online-current ensembles.

The input CSV should contain member prediction columns, e.g.
``member_pred_selected_risk035``.  The script searches simplex weights and an
optional anchor guard on validation only.  A selected policy can then be applied
to test exactly once, without using test metrics for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_MEMBERS = [
    "member_pred_selected_risk035",
    "member_pred_risk045",
    "member_pred_zero_anchor",
    "member_pred_range_scaled",
]


def _simplex_grid(n: int, step: float) -> Iterable[np.ndarray]:
    scale = int(round(1.0 / float(step)))
    if scale <= 0 or abs(scale * float(step) - 1.0) > 1e-8:
        raise ValueError("step must divide 1.0 exactly, e.g. 0.05 or 0.025.")
    if n <= 1:
        yield np.ones(1, dtype=np.float64)
        return
    for counts in product(range(scale + 1), repeat=n - 1):
        used = int(sum(counts))
        if used > scale:
            continue
        last = scale - used
        yield np.asarray(list(counts) + [last], dtype=np.float64) / float(scale)


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


def _metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
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


def _apply_anchor_guard(
    pred: np.ndarray,
    anchor: np.ndarray,
    anchor_threshold: float,
    margin: float,
    strength: float,
    pred_max: float = float("inf"),
) -> Tuple[np.ndarray, float]:
    corrected = pred.copy()
    if float(strength) <= 0.0 or not np.isfinite(anchor_threshold):
        return np.clip(corrected, 0.0, 20.0), 0.0
    cap = anchor + float(margin)
    active = np.isfinite(corrected) & np.isfinite(anchor) & (anchor <= float(anchor_threshold)) & (corrected > cap)
    if np.isfinite(pred_max):
        active = active & (corrected <= float(pred_max))
    corrected[active] = corrected[active] - float(strength) * (corrected[active] - cap[active])
    return np.clip(corrected, 0.0, 20.0), float(np.mean(active)) if active.size else 0.0


def _member_columns(frame: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    columns = [col for col in requested if col in frame.columns]
    if not columns:
        columns = sorted(col for col in frame.columns if col.startswith("member_pred_"))
    if len(columns) < 2:
        raise ValueError("At least two member prediction columns are required.")
    return columns


def _score_row(row: Mapping[str, Any], mode: str) -> Tuple[float, ...]:
    if mode == "score_only":
        return (
            float(row["goal_composite_strict120"]),
            float(row["mae"]),
            max(0.0, float(row["original_low_0_2_bias"]) - 2.5),
            -float(row["r2"]),
            -float(row["high12_f1"]),
        )
    if mode == "composite":
        return (
            -float(row["goal_pass_count"]),
            float(row["goal_composite_strict120"]),
            float(row["mae"]),
            max(0.0, float(row["original_low_0_2_bias"]) - 2.5),
            -float(row["high12_f1"]),
        )
    if mode == "low_then_composite":
        low_penalty = max(0.0, float(row["original_low_0_2_bias"]) - 2.5)
        return (
            low_penalty,
            float(row["goal_composite_strict120"]),
            float(row["mae"]),
            -float(row["r2"]),
            -float(row["high12_f1"]),
        )
    raise ValueError("selection_mode must be score_only, composite, or low_then_composite.")


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


def _prediction(frame: pd.DataFrame, member_cols: Sequence[str], weights: Sequence[float]) -> np.ndarray:
    matrix = frame.loc[:, list(member_cols)].to_numpy(dtype=np.float64)
    return np.asarray(weights, dtype=np.float64) @ matrix.T


def _evaluate_policy(frame: pd.DataFrame, member_cols: Sequence[str], policy: Mapping[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    pred = _prediction(frame, member_cols, [float(v) for v in str(policy["weights"]).split(",")])
    anchor = pd.to_numeric(frame["anchor_fms"], errors="coerce").to_numpy(dtype=np.float64)
    pred, guard_rate = _apply_anchor_guard(
        pred,
        anchor,
        float(policy["anchor_threshold"]),
        float(policy["margin"]),
        float(policy["strength"]),
        float(policy.get("guard_pred_max", float("inf"))),
    )
    out = frame.copy()
    out["base_predicted_fms_now"] = out["predicted_fms_now"]
    out["predicted_fms_now"] = pred
    out["strict_goal_anchor_guard_rate"] = guard_rate
    out["strict_goal_anchor_threshold"] = float(policy["anchor_threshold"])
    out["strict_goal_anchor_margin"] = float(policy["margin"])
    out["strict_goal_anchor_strength"] = float(policy["strength"])
    out["strict_goal_guard_pred_max"] = float(policy.get("guard_pred_max", float("inf")))
    out["fms_absolute_error"] = np.abs(pred - pd.to_numeric(out["target_fms_now"], errors="coerce").to_numpy(dtype=np.float64))
    for col, weight in zip(member_cols, [float(v) for v in str(policy["weights"]).split(",")]):
        out[f"strict_goal_weight_{col}"] = weight
    if "alarm_caution" in out.columns:
        out["alarm_caution"] = out["predicted_fms_now"] >= 8.0
    if "alarm_warning_high_fms" in out.columns:
        out["alarm_warning_high_fms"] = out["predicted_fms_now"] >= 12.0
    metrics = _metrics(
        pd.to_numeric(out["target_fms_now"], errors="coerce").to_numpy(dtype=np.float64),
        out["predicted_fms_now"].to_numpy(dtype=np.float64),
    )
    return out, metrics


def _format_weights(weights: Sequence[float]) -> str:
    return ",".join(f"{float(v):.8f}" for v in weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--member_cols", nargs="*", default=DEFAULT_MEMBERS)
    parser.add_argument("--weight_step", type=float, default=0.05)
    parser.add_argument("--anchor_thresholds", nargs="+", type=float, default=[float("inf")])
    parser.add_argument("--margins", nargs="+", type=float, default=[0.0])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.0])
    parser.add_argument("--guard_pred_maxs", nargs="+", type=float, default=[float("inf")])
    parser.add_argument("--selection_mode", choices=["score_only", "composite", "low_then_composite"], default="composite")
    parser.add_argument("--label", default="strict_goal_ensemble")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(args.val_csv)
    member_cols = _member_columns(val, args.member_cols)
    y_val = pd.to_numeric(val["target_fms_now"], errors="coerce").to_numpy(dtype=np.float64)
    anchor_val = pd.to_numeric(val["anchor_fms"], errors="coerce").to_numpy(dtype=np.float64)
    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for weights in _simplex_grid(len(member_cols), float(args.weight_step)):
        base_pred = _prediction(val, member_cols, weights)
        for anchor_threshold in args.anchor_thresholds:
            for margin in args.margins:
                for strength in args.strengths:
                    for guard_pred_max in args.guard_pred_maxs:
                        pred, guard_rate = _apply_anchor_guard(
                            base_pred,
                            anchor_val,
                            anchor_threshold,
                            margin,
                            strength,
                            guard_pred_max,
                        )
                        row: Dict[str, Any] = {
                            "weights": _format_weights(weights),
                            "member_cols": ",".join(member_cols),
                            "anchor_threshold": float(anchor_threshold),
                            "margin": float(margin),
                            "strength": float(strength),
                            "guard_pred_max": float(guard_pred_max),
                            "guard_rate": guard_rate,
                        }
                        row.update(_metrics(y_val, pred))
                        rows.append(row)
                        if best is None or _score_row(row, args.selection_mode) < _score_row(best, args.selection_mode):
                            best = row
    assert best is not None
    _write_csv(out_dir / "validation_strict_goal_ensemble_grid.csv", rows)
    best_payload = {
        "selection_mode": args.selection_mode,
        "label": args.label,
        "member_cols": member_cols,
        "selected_validation": best,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
    }
    (out_dir / "selected_strict_goal_ensemble.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    val_out, val_metrics = _evaluate_policy(val, member_cols, best)
    val_out["run_name"] = args.label
    val_out["model_name"] = "online_current_strict_goal_ensemble"
    val_out["split"] = "val"
    val_out.to_csv(out_dir / "val_predictions.csv", index=False)
    result = {"validation_metrics": val_metrics}
    if args.test_csv:
        test = pd.read_csv(args.test_csv)
        missing = [col for col in member_cols if col not in test.columns]
        if missing:
            raise ValueError(f"Test CSV missing selected member columns: {missing}")
        test_out, test_metrics = _evaluate_policy(test, member_cols, best)
        test_out["run_name"] = args.label
        test_out["model_name"] = "online_current_strict_goal_ensemble"
        test_out["split"] = "test"
        test_out.to_csv(out_dir / "test_predictions.csv", index=False)
        result["test_metrics"] = test_metrics
    result["selected_validation"] = best
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
