"""Audit feasible interpretations of the high8/high12 +25% objective."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


DEFAULT_METRICS_CSV = "reports/overnight_current_fms_goal_0514_120s/post_cutoff_strict_metric_refresh_test/goal_metrics.csv"
DEFAULT_OUT_DIR = "reports/overnight_current_fms_goal_0514_120s/c4_feasibility_audit"

INCREASE_METRICS = ["precision", "recall", "f1"]
DECREASE_METRICS = ["false_positive_rate", "false_negative_rate"]


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z %z")


def _float(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "nan"
    return f"{f:.{digits}f}"


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_by_label(rows: Sequence[Mapping[str, str]], label: str) -> Mapping[str, str]:
    for row in rows:
        if row.get("label") == label:
            return row
    labels = ", ".join(row.get("label", "") for row in rows)
    raise SystemExit(f"Could not find label {label!r}. Available: {labels}")


def audit_metric(
    baseline: Mapping[str, str],
    candidate: Mapping[str, str],
    *,
    threshold: int,
    metric: str,
    improvement_factor: float,
) -> Dict[str, Any]:
    key = f"high{threshold}_{metric}"
    base = _float(baseline, key)
    value = _float(candidate, key)
    higher_is_better = metric in INCREASE_METRICS
    if higher_is_better:
        target = base * float(improvement_factor)
        feasible = target <= 1.0
        passed = feasible and value >= target
        improvement = (value / base - 1.0) if base > 0 else float("nan")
    else:
        target = base * (2.0 - float(improvement_factor))
        feasible = target >= 0.0
        passed = feasible and value <= target
        improvement = (1.0 - value / base) if base > 0 else float("nan")
    return {
        "threshold": threshold,
        "metric": metric,
        "baseline": base,
        "candidate": value,
        "target": target,
        "higher_is_better": higher_is_better,
        "feasible": bool(feasible),
        "pass": bool(passed),
        "relative_improvement": improvement,
    }


def audit(
    rows: Sequence[Mapping[str, str]],
    *,
    baseline_label: str,
    candidate_label: str,
    improvement_factor: float,
) -> Dict[str, Any]:
    baseline = _row_by_label(rows, baseline_label)
    candidate = _row_by_label(rows, candidate_label)
    checks: List[Dict[str, Any]] = []
    for threshold in (8, 12):
        for metric in [*INCREASE_METRICS, *DECREASE_METRICS]:
            checks.append(
                audit_metric(
                    baseline,
                    candidate,
                    threshold=threshold,
                    metric=metric,
                    improvement_factor=improvement_factor,
                )
            )
    both_threshold_pass = {
        metric: all(row["pass"] for row in checks if row["metric"] == metric)
        for metric in [*INCREASE_METRICS, *DECREASE_METRICS]
    }
    both_threshold_feasible = {
        metric: all(row["feasible"] for row in checks if row["metric"] == metric)
        for metric in [*INCREASE_METRICS, *DECREASE_METRICS]
    }
    return {
        "created_at": _now(),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "improvement_factor": float(improvement_factor),
        "checks": checks,
        "both_threshold_feasible": both_threshold_feasible,
        "both_threshold_pass": both_threshold_pass,
        "recommended_c4_metric_family": (
            "F1 is not feasible for high8 because +25% exceeds 1.0. "
            "Use false-positive/false-negative reduction, AUPRC, or event-level warning metrics instead."
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# C4 Feasibility Audit",
        "",
        f"작성일: {result.get('created_at', '')}",
        "",
        f"- baseline: `{result.get('baseline_label')}`",
        f"- candidate: `{result.get('candidate_label')}`",
        f"- improvement factor: `{result.get('improvement_factor')}`",
        "",
        "## Feasibility / Pass By Metric Family",
        "",
        "| metric | feasible for both high8/high12 | current candidate pass |",
        "|---|---|---|",
    ]
    feasible = result.get("both_threshold_feasible", {})
    passed = result.get("both_threshold_pass", {})
    for metric in [*INCREASE_METRICS, *DECREASE_METRICS]:
        lines.append(f"| {metric} | {bool(feasible.get(metric))} | {bool(passed.get(metric))} |")
    lines.extend(
        [
            "",
            "## Detailed Checks",
            "",
            "| threshold | metric | baseline | candidate | target | feasible | pass | relative improvement |",
            "|---:|---|---:|---:|---:|---|---|---:|",
        ]
    )
    for row in result.get("checks", []):
        lines.append(
            "| {thr} | {metric} | {base} | {cand} | {target} | {feasible} | {passed} | {imp} |".format(
                thr=row["threshold"],
                metric=row["metric"],
                base=_fmt(row["baseline"]),
                cand=_fmt(row["candidate"]),
                target=_fmt(row["target"]),
                feasible=bool(row["feasible"]),
                passed=bool(row["pass"]),
                imp=_fmt(row["relative_improvement"]),
            )
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            str(result.get("recommended_c4_metric_family", "")),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_csv", default=DEFAULT_METRICS_CSV)
    parser.add_argument("--baseline_label", default="final_equal4_anchor_guard")
    parser.add_argument("--candidate_label", default="final_equal4_anchor_guard")
    parser.add_argument("--improvement_factor", type=float, default=1.25)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    result = audit(
        _read_rows(Path(args.metrics_csv)),
        baseline_label=str(args.baseline_label),
        candidate_label=str(args.candidate_label),
        improvement_factor=float(args.improvement_factor),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "c4_feasibility_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_csv(out_dir / "c4_feasibility_checks.csv", result["checks"])
    _write_report(out_dir / "c4_feasibility_audit.md", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
