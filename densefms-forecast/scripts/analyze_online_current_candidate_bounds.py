"""Candidate-envelope diagnostics for strict online-current FMS goals.

This is not a model-selection script.  It quantifies how far the four saved
candidate predictions can go under simple non-parametric upper bounds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_MEMBERS = [
    "member_pred_selected_risk035",
    "member_pred_risk045",
    "member_pred_zero_anchor",
    "member_pred_range_scaled",
]


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


def _metrics(split: str, label: str, y: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    err = pred - y
    low = (y >= 0.0) & (y < 2.0)
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    row: Dict[str, object] = {
        "split": split,
        "label": label,
        "n": int(y.size),
        "mae": float(np.mean(np.abs(err))) if y.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(err * err))) if y.size else float("nan"),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"),
        "original_low_0_2_n": int(np.sum(low)),
        "original_low_0_2_target_mean": float(np.mean(y[low])) if np.any(low) else float("nan"),
        "original_low_0_2_pred_mean": float(np.mean(pred[low])) if np.any(low) else float("nan"),
        "original_low_0_2_bias": float(np.mean(err[low])) if np.any(low) else float("nan"),
        "prediction_mean": float(np.mean(pred)) if y.size else float("nan"),
        "target_mean": float(np.mean(y)) if y.size else float("nan"),
    }
    row.update(_high_metrics(y, pred, 8.0))
    row.update(_high_metrics(y, pred, 12.0))
    row["pass_mae_le_1p8"] = bool(float(row["mae"]) <= 1.8)
    row["pass_r2_ge_0p75"] = bool(float(row["r2"]) >= 0.75)
    row["pass_low_bias_le_2p5"] = bool(float(row["original_low_0_2_bias"]) <= 2.5)
    row["pass_count_c1_c3"] = int(row["pass_mae_le_1p8"]) + int(row["pass_r2_ge_0p75"]) + int(row["pass_low_bias_le_2p5"])
    return row


def _prediction_variants(frame: pd.DataFrame, members: Sequence[str]) -> Dict[str, np.ndarray]:
    y = frame["target_fms_now"].to_numpy(dtype=np.float64)
    matrix = frame.loc[:, list(members)].to_numpy(dtype=np.float64)
    low = (y >= 0.0) & (y < 2.0)
    closest = matrix[np.arange(len(y)), np.argmin(np.abs(matrix - y[:, None]), axis=1)]
    equal4 = np.mean(matrix, axis=1)
    return {
        "equal4": equal4,
        "min4": np.min(matrix, axis=1),
        "median4": np.median(matrix, axis=1),
        "max4": np.max(matrix, axis=1),
        "oracle_closest_member": closest,
        "oracle_low_target_else_equal4": np.where(low, y, equal4),
        "oracle_low_target_else_closest": np.where(low, y, closest),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 4) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "nan"
    return f"{f:.{digits}f}"


def _write_report(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# Candidate Envelope Bounds",
        "",
        "4개 후보 prediction이 낼 수 있는 단순 envelope와 oracle upper-bound를 계산한 진단이다. `oracle_*` 행은 target을 사용하므로 배포 가능한 모델이 아니며, 목표 달성 가능성의 상한을 보기 위한 용도다.",
        "",
        "| split | label | MAE | R2 | original 0_2 bias | high8 F1 | high12 F1 | C1-C3 pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["split"]),
                    str(row["label"]),
                    _fmt(row["mae"]),
                    _fmt(row["r2"]),
                    _fmt(row["original_low_0_2_bias"]),
                    _fmt(row["high8_f1"]),
                    _fmt(row["high12_f1"]),
                    str(row["pass_count_c1_c3"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csvs", nargs="+", required=True, help="split=path entries.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--member_cols", nargs="*", default=DEFAULT_MEMBERS)
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for item in args.csvs:
        if "=" not in item:
            raise ValueError("--csvs entries must be split=path.")
        split, path = item.split("=", 1)
        frame = pd.read_csv(path)
        missing = [col for col in args.member_cols if col not in frame.columns]
        if missing:
            raise ValueError(f"{path} missing member columns: {missing}")
        y = frame["target_fms_now"].to_numpy(dtype=np.float64)
        for label, pred in _prediction_variants(frame, args.member_cols).items():
            rows.append(_metrics(split, label, y, pred))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "candidate_envelope_metrics.csv", rows)
    (out_dir / "candidate_envelope_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_report(out_dir / "candidate_envelope_report.md", rows)
    print(f"wrote candidate envelope diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
