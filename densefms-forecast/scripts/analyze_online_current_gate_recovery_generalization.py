"""Analyze whether a learned gate identifies low/recovery FMS states.

This script is diagnostic only: it consumes existing prediction CSVs and does
not train or select a model.  It is useful for checking whether validation gate
behavior transfers to held-out test predictions.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
FMS_BINS: Sequence[Tuple[float, float, str]] = (
    (0.0, 2.0, "0_2"),
    (2.0, 5.0, "2_5"),
    (5.0, 10.0, "5_10"),
    (10.0, 15.0, "10_15"),
    (15.0, 20.000001, "15_20"),
)
ANCHOR_BINS: Sequence[Tuple[float, float, str]] = (
    (0.0, 2.0, "anchor_0_2"),
    (2.0, 5.0, "anchor_2_5"),
    (5.0, 8.0, "anchor_5_8"),
    (8.0, 12.0, "anchor_8_12"),
    (12.0, 20.000001, "anchor_12_20"),
)
GATE_CANDIDATES = (
    "current_calib_prior_gate",
    "current_low_suppressor_gate",
    "current_residual_adapter_gate",
    "current_anchor_gate",
)


def _parse_input(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Input must be label=path, got {spec!r}")
    label, path = spec.split("=", 1)
    return label, Path(path)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        starts = np.cumsum(np.r_[0, counts[:-1]])
        ranks = (starts + (counts - 1) / 2.0)[inverse]
    return ranks + 1.0


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    labels = labels[mask].astype(bool)
    scores = scores[mask].astype(float)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    pos_rank_sum = float(ranks[labels].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    labels = labels[mask].astype(bool)
    scores = scores[mask].astype(float)
    if labels.size == 0 or int(labels.sum()) == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order].astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(float(labels.sum()), 1.0)
    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _prf(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    mask = np.isfinite(scores)
    labels = labels[mask].astype(bool)
    scores = scores[mask].astype(float)
    pred = scores >= float(threshold)
    tp = float(np.sum(pred & labels))
    fp = float(np.sum(pred & ~labels))
    fn = float(np.sum(~pred & labels))
    tn = float(np.sum(~pred & ~labels))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_rate": float(np.mean(pred)) if pred.size else float("nan"),
        "false_positive_rate": fp / (fp + tn) if fp + tn > 0 else 0.0,
        "false_negative_rate": fn / (tp + fn) if tp + fn > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _select_gate_column(df: pd.DataFrame, requested: Optional[str]) -> str:
    if requested:
        if requested not in df.columns:
            raise KeyError(f"Requested gate column {requested!r} not found.")
        return requested
    for column in GATE_CANDIDATES:
        if column in df.columns:
            return column
    raise KeyError(f"No known gate column found. Tried {list(GATE_CANDIDATES)}")


def _safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def _safe_quantile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q)) if len(values) else float("nan")


def _label_masks(df: pd.DataFrame, low_threshold: float, anchor_threshold: float, recovery_delta: float) -> Dict[str, np.ndarray]:
    target = df["target_fms_now"].astype(float).to_numpy()
    anchor = df["anchor_fms"].astype(float).to_numpy()
    finite = np.isfinite(target) & np.isfinite(anchor)
    low = finite & (target < float(low_threshold))
    recovery_low = low & (anchor >= float(anchor_threshold))
    anchor_drop_low = low & ((anchor - target) >= float(recovery_delta))
    return {
        "low": low,
        "recovery_low": recovery_low,
        "anchor_drop_low": anchor_drop_low,
    }


def _summarize_label(
    run_label: str,
    split: str,
    df: pd.DataFrame,
    gate_col: str,
    label_name: str,
    labels: np.ndarray,
    thresholds: Sequence[float],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    scores = df[gate_col].astype(float).to_numpy()
    finite = np.isfinite(scores)
    labels = labels[finite].astype(bool)
    scores = scores[finite]
    summary = {
        "run_label": run_label,
        "split": split,
        "gate_col": gate_col,
        "label": label_name,
        "n": int(labels.size),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()) if labels.size else float("nan"),
        "score_mean": float(np.mean(scores)) if scores.size else float("nan"),
        "score_p50": float(np.quantile(scores, 0.5)) if scores.size else float("nan"),
        "score_p90": float(np.quantile(scores, 0.9)) if scores.size else float("nan"),
        "positive_score_mean": float(np.mean(scores[labels])) if labels.any() else float("nan"),
        "negative_score_mean": float(np.mean(scores[~labels])) if (~labels).any() else float("nan"),
        "auroc": _auroc(labels, scores),
        "auprc": _auprc(labels, scores),
    }
    threshold_rows: List[Dict[str, object]] = []
    for threshold in thresholds:
        threshold_rows.append(
            {
                "run_label": run_label,
                "split": split,
                "gate_col": gate_col,
                "label": label_name,
                **_prf(labels, scores, threshold),
            }
        )
    return [summary], threshold_rows


def _bin_rows(run_label: str, split: str, df: pd.DataFrame, gate_col: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    target = df["target_fms_now"].astype(float)
    anchor = df["anchor_fms"].astype(float)
    for lo, hi, label in FMS_BINS:
        g = df[(target >= lo) & (target < hi)]
        rows.append(
            {
                "run_label": run_label,
                "split": split,
                "group_type": "target_fms_bin",
                "group": label,
                "n": int(len(g)),
                "target_mean": _safe_mean(g["target_fms_now"]),
                "anchor_mean": _safe_mean(g["anchor_fms"]),
                "pred_mean": _safe_mean(g["predicted_fms_now"]),
                "bias": _safe_mean(g["predicted_fms_now"] - g["target_fms_now"]),
                "gate_mean": _safe_mean(g[gate_col]),
                "gate_p90": _safe_quantile(g[gate_col], 0.9),
                "gate_gt_0.5_rate": float((g[gate_col] > 0.5).mean()) if len(g) else float("nan"),
            }
        )
    for lo, hi, label in ANCHOR_BINS:
        g = df[(anchor >= lo) & (anchor < hi)]
        rows.append(
            {
                "run_label": run_label,
                "split": split,
                "group_type": "anchor_fms_bin",
                "group": label,
                "n": int(len(g)),
                "target_mean": _safe_mean(g["target_fms_now"]),
                "anchor_mean": _safe_mean(g["anchor_fms"]),
                "pred_mean": _safe_mean(g["predicted_fms_now"]),
                "bias": _safe_mean(g["predicted_fms_now"] - g["target_fms_now"]),
                "gate_mean": _safe_mean(g[gate_col]),
                "gate_p90": _safe_quantile(g[gate_col], 0.9),
                "gate_gt_0.5_rate": float((g[gate_col] > 0.5).mean()) if len(g) else float("nan"),
            }
        )
    return rows


def _participant_rows(run_label: str, split: str, df: pd.DataFrame, gate_col: str, labels: np.ndarray) -> List[Dict[str, object]]:
    tmp = df.copy()
    tmp["_recovery_low"] = labels.astype(bool)
    rows: List[Dict[str, object]] = []
    for participant_id, g in tmp.groupby("participant_id"):
        low = g["_recovery_low"].to_numpy(dtype=bool)
        rows.append(
            {
                "run_label": run_label,
                "split": split,
                "participant_id": participant_id,
                "n": int(len(g)),
                "recovery_low_count": int(low.sum()),
                "recovery_low_rate": float(low.mean()) if len(low) else float("nan"),
                "gate_mean": _safe_mean(g[gate_col]),
                "gate_p90": _safe_quantile(g[gate_col], 0.9),
                "target_mean": _safe_mean(g["target_fms_now"]),
                "pred_mean": _safe_mean(g["predicted_fms_now"]),
                "bias": _safe_mean(g["predicted_fms_now"] - g["target_fms_now"]),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 4) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(out):
        return "nan"
    return f"{out:.{digits}f}"


def _write_report(path: Path, summary_rows: Sequence[Mapping[str, object]], threshold_rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# Gate Recovery Generalization Diagnostic",
        "",
        "Prediction CSV 기반 사후 진단이다. 새 모델 선택이나 test hyperparameter search에는 사용하지 않는다.",
        "",
        "## Ranking Metrics",
        "",
        "| run | split | gate | label | positive rate | positive score | negative score | AUROC | AUPRC |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run_label"]),
                    str(row["split"]),
                    str(row["gate_col"]),
                    str(row["label"]),
                    _fmt(row["positive_rate"]),
                    _fmt(row["positive_score_mean"]),
                    _fmt(row["negative_score_mean"]),
                    _fmt(row["auroc"]),
                    _fmt(row["auprc"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Threshold Metrics For Recovery-Low",
            "",
            "| run | split | thr | precision | recall | F1 | pred rate | FPR | FNR |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in threshold_rows:
        if str(row["label"]) != "recovery_low":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run_label"]),
                    str(row["split"]),
                    _fmt(row["threshold"], 2),
                    _fmt(row["precision"]),
                    _fmt(row["recall"]),
                    _fmt(row["f1"]),
                    _fmt(row["pred_rate"]),
                    _fmt(row["false_positive_rate"]),
                    _fmt(row["false_negative_rate"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="label=prediction_csv")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--gate_col", default=None)
    parser.add_argument("--low_threshold", type=float, default=2.0)
    parser.add_argument("--anchor_threshold", type=float, default=5.0)
    parser.add_argument("--recovery_delta", type=float, default=4.0)
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    args = parser.parse_args()

    summary_rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []
    bin_rows: List[Dict[str, object]] = []
    participant_rows: List[Dict[str, object]] = []
    for input_spec in args.inputs:
        run_label, path = _parse_input(input_spec)
        df = pd.read_csv(path)
        split = str(df["split"].iloc[0]) if "split" in df.columns and len(df) else path.parent.name
        gate_col = _select_gate_column(df, args.gate_col)
        masks = _label_masks(df, args.low_threshold, args.anchor_threshold, args.recovery_delta)
        for label_name, labels in masks.items():
            summary, thresholds = _summarize_label(run_label, split, df, gate_col, label_name, labels, args.thresholds)
            summary_rows.extend(summary)
            threshold_rows.extend(thresholds)
        bin_rows.extend(_bin_rows(run_label, split, df, gate_col))
        participant_rows.extend(_participant_rows(run_label, split, df, gate_col, masks["recovery_low"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "gate_label_summary.csv", summary_rows)
    _write_csv(out_dir / "gate_threshold_metrics.csv", threshold_rows)
    _write_csv(out_dir / "gate_by_bin.csv", bin_rows)
    _write_csv(out_dir / "participant_gate_summary.csv", participant_rows)
    _write_report(out_dir / "gate_recovery_generalization_report.md", summary_rows, threshold_rows)
    print(f"wrote gate recovery diagnostic to {out_dir}")


if __name__ == "__main__":
    main()
