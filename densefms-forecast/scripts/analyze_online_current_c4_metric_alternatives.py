"""Compare concrete C4 metric interpretations on existing goal metrics.

This script is report-only. It reads existing prediction/metric CSV files and
does not train, evaluate checkpoints, or touch the original test split beyond
using already generated prediction CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_VAL_METRICS = "reports/overnight_current_fms_goal_0514_120s/post_cutoff_strict_metric_refresh_val/goal_metrics.csv"
DEFAULT_TEST_METRICS = "reports/overnight_current_fms_goal_0514_120s/post_cutoff_strict_metric_refresh_test/goal_metrics.csv"
DEFAULT_OUT_DIR = "reports/overnight_current_fms_goal_0514_120s/c4_metric_alternatives"

INCREASE_METRICS = ("precision", "recall", "f1", "auprc")
DECREASE_METRICS = ("false_positive_rate", "false_negative_rate")
THRESHOLDS = (8, 12)


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z %z")


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any, digits: int = 4) -> str:
    f = _float(value)
    if not math.isfinite(f):
        return "nan"
    return f"{f:.{digits}f}"


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


def _read_rows(path: Path, split: str) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        row["split"] = split
    return rows


def _by_label(rows: Sequence[Mapping[str, Any]], label: str) -> Mapping[str, Any]:
    for row in rows:
        if str(row.get("label")) == label:
            return row
    labels = ", ".join(str(row.get("label")) for row in rows)
    raise SystemExit(f"Missing baseline label {label!r}. Available labels: {labels}")


def _prediction_scores(row: Mapping[str, Any]) -> Optional[pd.DataFrame]:
    path = Path(str(row.get("path", "")))
    pred_col = str(row.get("pred_column", "predicted_fms_now"))
    if not path.exists() or pred_col not in pd.read_csv(path, nrows=0).columns:
        return None
    return pd.read_csv(path, usecols=["target_fms_now", pred_col]).rename(columns={pred_col: "_score"})


def _append_rank_metrics(row: Dict[str, Any]) -> None:
    df = _prediction_scores(row)
    if df is None:
        for threshold in THRESHOLDS:
            row[f"high{threshold}_auprc"] = float("nan")
            row[f"high{threshold}_auroc"] = float("nan")
        return
    y = df["target_fms_now"].astype(float).to_numpy()
    scores = df["_score"].astype(float).to_numpy()
    for threshold in THRESHOLDS:
        labels = np.isfinite(y) & (y >= float(threshold))
        row[f"high{threshold}_auprc"] = _auprc(labels, scores)
        row[f"high{threshold}_auroc"] = _auroc(labels, scores)


def _metric_value(row: Mapping[str, Any], threshold: int, metric: str) -> float:
    return _float(row.get(f"high{threshold}_{metric}"))


def _compare_metric(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    threshold: int,
    metric: str,
    improvement_factor: float,
) -> Dict[str, Any]:
    base = _metric_value(baseline, threshold, metric)
    value = _metric_value(candidate, threshold, metric)
    higher_is_better = metric in INCREASE_METRICS
    if higher_is_better:
        target = base * improvement_factor
        feasible = target <= 1.0
        passed = bool(feasible and value >= target)
        relative = (value / base - 1.0) if base > 0 else float("nan")
    else:
        target = base * (2.0 - improvement_factor)
        feasible = target >= 0.0
        passed = bool(feasible and value <= target)
        relative = (1.0 - value / base) if base > 0 else float("nan")
    return {
        "threshold": threshold,
        "metric": metric,
        "baseline_value": base,
        "candidate_value": value,
        "target_value": target,
        "higher_is_better": higher_is_better,
        "feasible": bool(feasible),
        "pass": passed,
        "relative_improvement": relative,
    }


def _candidate_summary(
    split: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    improvement_factor: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for metric in (*INCREASE_METRICS, *DECREASE_METRICS):
        checks = [
            _compare_metric(
                baseline,
                candidate,
                threshold=threshold,
                metric=metric,
                improvement_factor=improvement_factor,
            )
            for threshold in THRESHOLDS
        ]
        for check in checks:
            detail_rows.append(
                {
                    "split": split,
                    "baseline_label": baseline.get("label"),
                    "candidate_label": candidate.get("label"),
                    **check,
                }
            )
        feasible_both = all(row["feasible"] for row in checks)
        pass_both = all(row["pass"] for row in checks)
        mean_relative = float(np.nanmean([row["relative_improvement"] for row in checks]))
        summary_rows.append(
            {
                "split": split,
                "baseline_label": baseline.get("label"),
                "candidate_label": candidate.get("label"),
                "metric_family": metric,
                "feasible_both_high8_high12": feasible_both,
                "pass_both_high8_high12": pass_both,
                "mean_relative_improvement": mean_relative,
                "high8_relative_improvement": checks[0]["relative_improvement"],
                "high12_relative_improvement": checks[1]["relative_improvement"],
                "high8_candidate_value": checks[0]["candidate_value"],
                "high12_candidate_value": checks[1]["candidate_value"],
                "high8_target_value": checks[0]["target_value"],
                "high12_target_value": checks[1]["target_value"],
            }
        )
    return detail_rows, summary_rows


def analyze(
    val_metrics: Path,
    test_metrics: Path,
    *,
    val_baseline_label: str,
    test_baseline_label: str,
    improvement_factor: float,
) -> Dict[str, Any]:
    split_rows = {
        "validation": _read_rows(val_metrics, "validation"),
        "test_existing": _read_rows(test_metrics, "test_existing"),
    }
    for rows in split_rows.values():
        for row in rows:
            _append_rank_metrics(row)

    baselines = {
        "validation": _by_label(split_rows["validation"], val_baseline_label),
        "test_existing": _by_label(split_rows["test_existing"], test_baseline_label),
    }
    details: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for split, rows in split_rows.items():
        baseline = baselines[split]
        for candidate in rows:
            detail, summary = _candidate_summary(
                split,
                baseline,
                candidate,
                improvement_factor=improvement_factor,
            )
            details.extend(detail)
            summaries.extend(summary)
    return {
        "created_at": _now(),
        "improvement_factor": improvement_factor,
        "val_metrics": str(val_metrics),
        "test_metrics": str(test_metrics),
        "val_baseline_label": val_baseline_label,
        "test_baseline_label": test_baseline_label,
        "details": details,
        "summaries": summaries,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    summaries = list(result["summaries"])
    lines = [
        "# C4 Metric Alternatives",
        "",
        f"작성일: {result['created_at']}",
        "",
        "기존 prediction CSV와 post-cutoff goal metric CSV만 읽은 분석이다. 새 학습이나 original test evaluation은 수행하지 않았다.",
        "",
        "## Baselines",
        "",
        f"- validation baseline: `{result['val_baseline_label']}`",
        f"- existing test baseline: `{result['test_baseline_label']}`",
        f"- improvement factor: `{result['improvement_factor']}`",
        "",
        "## Both-Threshold Pass Summary",
        "",
        "| split | candidate | metric family | feasible | pass high8+high12 | mean relative improvement | high8 relative | high12 relative |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {split} | {candidate} | {metric} | {feasible} | {passed} | {mean} | {h8} | {h12} |".format(
                split=row["split"],
                candidate=row["candidate_label"],
                metric=row["metric_family"],
                feasible=bool(row["feasible_both_high8_high12"]),
                passed=bool(row["pass_both_high8_high12"]),
                mean=_fmt(row["mean_relative_improvement"]),
                h8=_fmt(row["high8_relative_improvement"]),
                h12=_fmt(row["high12_relative_improvement"]),
            )
        )
    passed_rows = [row for row in summaries if row["pass_both_high8_high12"]]
    lines.extend(["", "## Interpretation", ""])
    if passed_rows:
        lines.append("At least one existing row satisfies a concrete both-threshold C4 interpretation:")
        for row in passed_rows:
            lines.append(
                f"- `{row['split']}` / `{row['candidate_label']}` / `{row['metric_family']}`"
            )
    else:
        lines.append("No existing candidate satisfies a +25% improvement simultaneously for high8 and high12 under the tested metric families.")
    lines.extend(
        [
            "",
            "F1/precision/recall are bounded above by 1.0, so high8 can be mathematically infeasible when the baseline is already high. FPR/FNR reduction is the most deployable current-FMS interpretation because it remains bounded and directly targets false high-state or missed high-state errors.",
            "",
            "## Recommended C4 Handling",
            "",
            "Treat the original C4 as unresolved until the user fixes the metric. For the next strict 120s current-FMS experiment, prefer:",
            "",
            "1. high8/high12 false-positive-rate 25% reduction with recall not worse than baseline, or",
            "2. high8/high12 false-negative-rate 25% reduction with precision not worse than baseline.",
            "",
            "Do not use F1 +25% for high8; it is infeasible for the current baseline.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val_metrics_csv", default=DEFAULT_VAL_METRICS)
    parser.add_argument("--test_metrics_csv", default=DEFAULT_TEST_METRICS)
    parser.add_argument("--val_baseline_label", default="range_original")
    parser.add_argument("--test_baseline_label", default="final_equal4_anchor_guard")
    parser.add_argument("--improvement_factor", type=float, default=1.25)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    result = analyze(
        Path(args.val_metrics_csv),
        Path(args.test_metrics_csv),
        val_baseline_label=str(args.val_baseline_label),
        test_baseline_label=str(args.test_baseline_label),
        improvement_factor=float(args.improvement_factor),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "c4_metric_alternatives.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_csv(out_dir / "c4_metric_alternative_summary.csv", result["summaries"])
    _write_csv(out_dir / "c4_metric_alternative_details.csv", result["details"])
    _write_report(out_dir / "c4_metric_alternatives.md", result)
    print(json.dumps({k: result[k] for k in ["created_at", "improvement_factor", "val_baseline_label", "test_baseline_label"]}, indent=2))
    print(f"wrote C4 metric alternatives to {out_dir}")


if __name__ == "__main__":
    main()
