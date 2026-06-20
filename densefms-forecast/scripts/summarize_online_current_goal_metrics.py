"""Summarize online-current FMS prediction CSVs for the current goal audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def _parse_input(spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path)
    path = Path(spec)
    return path.parent.name, path


def _classification_metrics(target: np.ndarray, pred: np.ndarray, threshold: float) -> Dict[str, float]:
    y = target >= float(threshold)
    p = pred >= float(threshold)
    tp = float(np.sum(y & p))
    fp = float(np.sum(~y & p))
    fn = float(np.sum(y & ~p))
    tn = float(np.sum(~y & ~p))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        f"high{threshold:g}_precision": precision,
        f"high{threshold:g}_recall": recall,
        f"high{threshold:g}_f1": f1,
        f"high{threshold:g}_false_positive_rate": fp / (fp + tn) if fp + tn > 0 else 0.0,
        f"high{threshold:g}_false_negative_rate": fn / (tp + fn) if tp + fn > 0 else 0.0,
    }


def _row(label: str, path: Path, pred_column: str, low_bin_max: float, thresholds: Sequence[float]) -> Dict[str, Any]:
    frame = pd.read_csv(path)
    if "target_fms_now" not in frame.columns or pred_column not in frame.columns:
        raise ValueError(f"{path} must contain target_fms_now and {pred_column}")
    target = pd.to_numeric(frame["target_fms_now"], errors="coerce").to_numpy(dtype=np.float64)
    pred = pd.to_numeric(frame[pred_column], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(target) & np.isfinite(pred)
    target = target[mask]
    pred = pred[mask]
    err = pred - target
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    low = target <= float(low_bin_max)
    strict_low = (target >= 0.0) & (target < float(low_bin_max))
    row: Dict[str, Any] = {
        "label": label,
        "path": str(path),
        "pred_column": pred_column,
        "n": int(target.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "target_mean": float(np.mean(target)),
        "pred_mean": float(np.mean(pred)),
        "low_bin": f"0-{low_bin_max:g}",
        "low_n": int(np.sum(low)),
        "low_target_mean": float(np.mean(target[low])) if np.any(low) else float("nan"),
        "low_pred_mean": float(np.mean(pred[low])) if np.any(low) else float("nan"),
        "low_signed_bias": float(np.mean(err[low])) if np.any(low) else float("nan"),
        "low_mae": float(np.mean(np.abs(err[low]))) if np.any(low) else float("nan"),
        "strict_low_bin": f"0-{low_bin_max:g}_exclusive_upper",
        "strict_low_n": int(np.sum(strict_low)),
        "strict_low_target_mean": float(np.mean(target[strict_low])) if np.any(strict_low) else float("nan"),
        "strict_low_pred_mean": float(np.mean(pred[strict_low])) if np.any(strict_low) else float("nan"),
        "strict_low_signed_bias": float(np.mean(err[strict_low])) if np.any(strict_low) else float("nan"),
        "strict_low_mae": float(np.mean(np.abs(err[strict_low]))) if np.any(strict_low) else float("nan"),
    }
    for threshold in thresholds:
        row.update(_classification_metrics(target, pred, float(threshold)))
    return row


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "nan"
    return f"{f:.{digits}f}"


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


def _write_markdown(path: Path, rows: Sequence[Mapping[str, Any]], thresholds: Sequence[float]) -> None:
    metric_cols = [
        "label",
        "mae",
        "rmse",
        "r2",
        "strict_low_signed_bias",
        "strict_low_mae",
        "low_signed_bias",
        "low_mae",
    ]
    for threshold in thresholds:
        metric_cols.extend([f"high{threshold:g}_precision", f"high{threshold:g}_recall", f"high{threshold:g}_f1"])
    lines = [
        "# Online Current Goal Metrics",
        "",
        "| " + " | ".join(metric_cols) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(metric_cols) - 1)) + " |",
    ]
    for row in rows:
        values = [str(row["label"])]
        values.extend(_fmt(row[col]) for col in metric_cols[1:])
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Inputs as label=prediction_csv.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pred_column", default="predicted_fms_now")
    parser.add_argument("--low_bin_max", type=float, default=2.0)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[8.0, 12.0])
    args = parser.parse_args()

    rows = [
        _row(label, path, pred_column=str(args.pred_column), low_bin_max=float(args.low_bin_max), thresholds=args.thresholds)
        for label, path in [_parse_input(spec) for spec in args.inputs]
    ]
    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "goal_metrics.csv", rows)
    _write_markdown(out_dir / "goal_metrics.md", rows, args.thresholds)
    (out_dir / "goal_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
