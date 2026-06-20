"""Audit the strict 120s online-current goal status from metric CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


DEFAULT_METRICS_CSV = "reports/overnight_current_fms_goal_0514_120s/post_cutoff_strict_metric_refresh_test/goal_metrics.csv"
DEFAULT_OUT_DIR = "reports/overnight_current_fms_goal_0514_120s/goal_status_audit"


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


def audit_row(
    row: Mapping[str, Any],
    *,
    mae_max: float = 1.8,
    r2_min: float = 0.75,
    strict_low_bias_max: float = 2.5,
    high8_baseline_f1: float = 0.872008960390999,
    high12_baseline_f1: float = 0.6697650400206558,
    improvement_factor: float = 1.25,
    required_pass_count: int = 3,
) -> Dict[str, Any]:
    mae = _float(row, "mae")
    r2 = _float(row, "r2")
    strict_low_bias = _float(row, "strict_low_signed_bias")
    high8_f1 = _float(row, "high8_f1")
    high12_f1 = _float(row, "high12_f1")
    high8_target = float(high8_baseline_f1) * float(improvement_factor)
    high12_target = float(high12_baseline_f1) * float(improvement_factor)
    c4_possible = high8_target <= 1.0 and high12_target <= 1.0
    c4_pass = c4_possible and high8_f1 >= high8_target and high12_f1 >= high12_target
    criteria = {
        "C1_MAE": {
            "value": mae,
            "target": f"<= {mae_max:g}",
            "pass": bool(mae <= float(mae_max)),
        },
        "C2_R2": {
            "value": r2,
            "target": f">= {r2_min:g}",
            "pass": bool(r2 >= float(r2_min)),
        },
        "C3_STRICT_LOW_BIAS": {
            "value": strict_low_bias,
            "target": f"<= {strict_low_bias_max:g}",
            "pass": bool(strict_low_bias <= float(strict_low_bias_max)),
        },
        "C4_HIGH8_HIGH12_F1_RELATIVE": {
            "high8_value": high8_f1,
            "high8_target": high8_target,
            "high12_value": high12_f1,
            "high12_target": high12_target,
            "possible_under_f1": bool(c4_possible),
            "pass": bool(c4_pass),
        },
    }
    pass_count = int(sum(1 for item in criteria.values() if item["pass"]))
    return {
        "label": row.get("label", ""),
        "path": row.get("path", ""),
        "required_pass_count": int(required_pass_count),
        "pass_count": pass_count,
        "goal_complete": bool(pass_count >= int(required_pass_count)),
        "criteria": criteria,
    }


def _read_row(path: Path, label: str) -> Dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("label") == label]
    if not matches:
        labels = ", ".join(row.get("label", "") for row in rows)
        raise SystemExit(f"Could not find label {label!r} in {path}. Available labels: {labels}")
    return matches[0]


def _write_report(path: Path, audit: Mapping[str, Any], args: argparse.Namespace) -> None:
    criteria = audit["criteria"]
    c4 = criteria["C4_HIGH8_HIGH12_F1_RELATIVE"]
    lines = [
        "# Strict 120s Goal Status Audit",
        "",
        f"작성일: {_now()}",
        "",
        "This audit reads metric CSV values and checks the 4 objective criteria directly.",
        "",
        "## Inputs",
        "",
        f"- metrics csv: `{args.metrics_csv}`",
        f"- label: `{args.label}`",
        f"- required pass count: `{audit['required_pass_count']}`",
        "",
        "## Result",
        "",
        f"- pass count: `{audit['pass_count']}/{len(criteria)}`",
        f"- goal complete: `{audit['goal_complete']}`",
        "",
        "## Criteria",
        "",
        "| id | value | target | pass |",
        "|---|---:|---:|---|",
        "| C1 MAE | {value} | {target} | {passed} |".format(
            value=_fmt(criteria["C1_MAE"]["value"]),
            target=criteria["C1_MAE"]["target"],
            passed=criteria["C1_MAE"]["pass"],
        ),
        "| C2 R2 | {value} | {target} | {passed} |".format(
            value=_fmt(criteria["C2_R2"]["value"]),
            target=criteria["C2_R2"]["target"],
            passed=criteria["C2_R2"]["pass"],
        ),
        "| C3 strict `0<=FMS<2` signed bias | {value} | {target} | {passed} |".format(
            value=_fmt(criteria["C3_STRICT_LOW_BIAS"]["value"]),
            target=criteria["C3_STRICT_LOW_BIAS"]["target"],
            passed=criteria["C3_STRICT_LOW_BIAS"]["pass"],
        ),
        "| C4 high8/high12 F1 +25% | high8 {h8} / high12 {h12} | high8 {h8t} / high12 {h12t} | {passed} |".format(
            h8=_fmt(c4["high8_value"]),
            h12=_fmt(c4["high12_value"]),
            h8t=_fmt(c4["high8_target"]),
            h12t=_fmt(c4["high12_target"]),
            passed=c4["pass"],
        ),
        "",
        "## C4 Note",
        "",
        f"- high8 F1 target under +25% is `{_fmt(c4['high8_target'])}`.",
        "- Because F1 is capped at 1.0, C4 is impossible under F1 if the required high8 target exceeds 1.0.",
        f"- possible under F1: `{c4['possible_under_f1']}`",
        "",
        "## Decision",
        "",
        "Do not mark the active goal complete unless goal_complete is true and the underlying metric definition matches the user's intent.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_csv", default=DEFAULT_METRICS_CSV)
    parser.add_argument("--label", default="final_equal4_anchor_guard")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--mae_max", type=float, default=1.8)
    parser.add_argument("--r2_min", type=float, default=0.75)
    parser.add_argument("--strict_low_bias_max", type=float, default=2.5)
    parser.add_argument("--high8_baseline_f1", type=float, default=0.872008960390999)
    parser.add_argument("--high12_baseline_f1", type=float, default=0.6697650400206558)
    parser.add_argument("--improvement_factor", type=float, default=1.25)
    parser.add_argument("--required_pass_count", type=int, default=3)
    parser.add_argument("--fail_if_incomplete", action="store_true")
    args = parser.parse_args()

    row = _read_row(Path(args.metrics_csv), str(args.label))
    audit = audit_row(
        row,
        mae_max=float(args.mae_max),
        r2_min=float(args.r2_min),
        strict_low_bias_max=float(args.strict_low_bias_max),
        high8_baseline_f1=float(args.high8_baseline_f1),
        high12_baseline_f1=float(args.high12_baseline_f1),
        improvement_factor=float(args.improvement_factor),
        required_pass_count=int(args.required_pass_count),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "goal_status_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    _write_report(out_dir / "goal_status_audit.md", audit, args)
    print(json.dumps(audit, indent=2))
    if args.fail_if_incomplete and not audit["goal_complete"]:
        raise SystemExit("Goal is incomplete.")


if __name__ == "__main__":
    main()
