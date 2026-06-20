"""Verify that strict 120s goal artifacts are based on four candidate models.

This script reads existing meta-calibrator prediction CSVs only. It does not
train, run validation, or evaluate the original test set.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


DEFAULT_OUT_DIR = "reports/overnight_current_fms_goal_0514_120s/four_candidate_evidence"
DEFAULT_PREDICTION_CSVS = [
    "runs/overnight_current_fms_goal_0514_120s/meta_calibrator_input_train/train_predictions.csv",
    "runs/overnight_current_fms_goal_0514_120s/meta_calibrator_input_val/val_predictions.csv",
    "runs/overnight_current_fms_goal_0514_120s/meta_calibrator_input_test/test_predictions.csv",
]
REQUIRED_MEMBER_COLUMNS = [
    "member_pred_selected_risk035",
    "member_pred_risk045",
    "member_pred_zero_anchor",
    "member_pred_range_scaled",
]


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z %z")


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def verify_prediction_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "ok": False,
            "row_count": 0,
            "split_values": [],
            "checks": {"exists": False},
        }
    row_count = 0
    split_values = set()
    calibration_values = set()
    nonfinite_counts = {col: 0 for col in REQUIRED_MEMBER_COLUMNS}
    missing_counts = {col: 0 for col in REQUIRED_MEMBER_COLUMNS}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        has_required_columns = all(col in fieldnames for col in REQUIRED_MEMBER_COLUMNS)
        for row in reader:
            row_count += 1
            if row.get("split"):
                split_values.add(str(row.get("split")))
            value = _float_or_none(row.get("calibration_seconds"))
            if value is not None:
                calibration_values.add(value)
            for col in REQUIRED_MEMBER_COLUMNS:
                if col not in row or row.get(col, "") == "":
                    missing_counts[col] += 1
                    continue
                if _float_or_none(row.get(col)) is None:
                    nonfinite_counts[col] += 1
    checks = {
        "exists": True,
        "has_rows": row_count > 0,
        "has_required_member_columns": has_required_columns,
        "all_member_values_present": all(count == 0 for count in missing_counts.values()),
        "all_member_values_numeric": all(count == 0 for count in nonfinite_counts.values()),
        "calibration_seconds_120": calibration_values == {120.0},
    }
    return {
        "path": str(path),
        "exists": True,
        "ok": all(bool(value) for value in checks.values()),
        "row_count": row_count,
        "split_values": sorted(split_values),
        "calibration_seconds_values": sorted(calibration_values),
        "required_member_columns": REQUIRED_MEMBER_COLUMNS,
        "missing_counts": missing_counts,
        "nonfinite_counts": nonfinite_counts,
        "checks": checks,
    }


def verify(args: argparse.Namespace) -> Dict[str, Any]:
    rows = [verify_prediction_csv(Path(path)) for path in args.prediction_csvs]
    all_split_values = sorted({split for row in rows for split in row.get("split_values", [])})
    checks = {
        "three_split_csvs_checked": len(rows) == 3,
        "train_val_test_present": set(all_split_values) == {"test", "train", "val"},
        "all_csvs_ok": bool(rows) and all(bool(row.get("ok")) for row in rows),
    }
    return {
        "created_at": _now(),
        "ok": all(bool(value) for value in checks.values()),
        "checks": checks,
        "required_member_columns": REQUIRED_MEMBER_COLUMNS,
        "csvs": rows,
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Four-Candidate Evidence Verification",
        "",
        f"작성일: {result.get('created_at', '')}",
        "",
        f"- overall ok: `{result.get('ok')}`",
        "",
        "## Required Member Columns",
        "",
    ]
    for col in result.get("required_member_columns", []):
        lines.append(f"- `{col}`")
    lines.extend(["", "## CSV Checks", "", "| csv | rows | splits | calibration seconds | ok |", "|---|---:|---|---|---|"])
    for row in result.get("csvs", []):
        lines.append(
            "| {path} | {rows} | {splits} | {calib} | {ok} |".format(
                path=row.get("path", ""),
                rows=int(row.get("row_count", 0)),
                splits=", ".join(row.get("split_values", [])),
                calib=", ".join(str(v) for v in row.get("calibration_seconds_values", [])),
                ok=bool(row.get("ok")),
            )
        )
    lines.extend(["", "## Checks", "", "| check | pass |", "|---|---|"])
    for key, value in result.get("checks", {}).items():
        lines.append(f"| {key} | {bool(value)} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This verifier confirms that the train/val/test meta-calibrator inputs contain four numeric member prediction columns under the strict 120s calibration condition.",
            "It does not imply the research goal is complete.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_csvs", nargs="+", default=DEFAULT_PREDICTION_CSVS)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--fail_on_error", action="store_true")
    args = parser.parse_args()
    result = verify(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "four_candidate_evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_report(out_dir / "four_candidate_evidence.md", result)
    print(json.dumps(result, indent=2))
    if args.fail_on_error and not result["ok"]:
        raise SystemExit("Four-candidate evidence verification failed.")


if __name__ == "__main__":
    main()
