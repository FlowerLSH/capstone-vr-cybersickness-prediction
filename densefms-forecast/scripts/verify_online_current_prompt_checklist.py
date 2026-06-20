"""Verify the strict 120s prompt-to-artifact checklist.

This script reads existing reports only. It does not train, run validation, or
evaluate the original test set.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


DEFAULT_BASE_DIR = Path("reports/overnight_current_fms_goal_0514_120s")
DEFAULT_CHECKLIST_CSV = DEFAULT_BASE_DIR / "prompt_to_artifact_checklist/checklist.csv"
DEFAULT_CHECKLIST_JSON = DEFAULT_BASE_DIR / "prompt_to_artifact_checklist/checklist.json"
DEFAULT_COMPLETION_BLOCKER_JSON = Path("reports/overnight_current_fms_goal_0514_120s_completion_blocker.json")
DEFAULT_GOAL_STATUS_JSON = DEFAULT_BASE_DIR / "goal_status_audit/goal_status_audit.json"
DEFAULT_OUT_DIR = DEFAULT_BASE_DIR / "prompt_to_artifact_checklist"

REQUIRED_REQUIREMENTS = [
    "4 candidate models based",
    "120s calibration",
    "achieve 3 of 4 test criteria",
    "C1 test MAE <= 1.8",
    "C2 test R2 >= 0.75",
    "C3 strict original FMS 0-2 signed bias <= +2.5",
    "C4 high8/high12 >=25% improvement",
    "full training allowed",
    "validation-based selection",
    "test final-report-only",
    "run time checked and cutoff honored",
    "final report written",
    "do not mark goal complete",
    "next experiment plan prepared",
]


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z %z")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _evidence_exists(raw: str) -> bool:
    raw = str(raw).strip()
    if not raw:
        return False
    if "*" in raw:
        return bool(glob.glob(raw))
    return Path(raw).exists()


def _flatten_evidence(rows: Iterable[Mapping[str, Any]]) -> List[str]:
    evidence: List[str] = []
    for row in rows:
        raw_values = row.get("evidence", [])
        if isinstance(raw_values, str):
            parts = [part.strip() for part in raw_values.split(";")]
        else:
            parts = [str(part).strip() for part in raw_values]
        evidence.extend(part for part in parts if part)
    return evidence


def verify(args: argparse.Namespace) -> Dict[str, Any]:
    checklist_csv = Path(args.checklist_csv)
    checklist_json = Path(args.checklist_json)
    blocker_json = Path(args.completion_blocker_json)
    goal_status_json = Path(args.goal_status_json)

    csv_rows = _read_csv(checklist_csv)
    checklist = _read_json(checklist_json)
    blocker = _read_json(blocker_json)
    goal_status = _read_json(goal_status_json)
    json_rows = checklist.get("rows", []) if isinstance(checklist, dict) else []
    requirement_names = {str(row.get("requirement", "")) for row in json_rows}
    missing_requirements = [name for name in REQUIRED_REQUIREMENTS if name not in requirement_names]
    evidence = _flatten_evidence(json_rows)
    missing_evidence = [path for path in evidence if not _evidence_exists(path)]

    status_by_requirement = {str(row.get("requirement", "")): str(row.get("status", "")) for row in json_rows}
    checks = {
        "checklist_csv_exists": checklist_csv.exists(),
        "checklist_json_exists": checklist_json.exists(),
        "csv_json_row_count_match": len(csv_rows) == len(json_rows),
        "required_requirements_present": not missing_requirements,
        "all_evidence_paths_exist": not missing_evidence,
        "checklist_goal_incomplete": checklist.get("goal_complete") is False,
        "checklist_pass_count_zero": checklist.get("pass_count") == 0,
        "checklist_required_pass_count_three": checklist.get("required_pass_count") == 3,
        "matches_completion_blocker": (
            checklist.get("goal_complete") == blocker.get("goal_complete")
            and checklist.get("pass_count") == blocker.get("pass_count")
            and checklist.get("required_pass_count") == blocker.get("required_pass_count")
            and blocker.get("update_goal_allowed") is False
        ),
        "matches_goal_status_audit": (
            checklist.get("goal_complete") == goal_status.get("goal_complete")
            and checklist.get("pass_count") == goal_status.get("pass_count")
        ),
        "criteria_statuses_fail_or_block": (
            status_by_requirement.get("achieve 3 of 4 test criteria") == "not_achieved"
            and status_by_requirement.get("C1 test MAE <= 1.8") == "failed"
            and status_by_requirement.get("C2 test R2 >= 0.75") == "failed"
            and status_by_requirement.get("C3 strict original FMS 0-2 signed bias <= +2.5") == "failed"
            and status_by_requirement.get("C4 high8/high12 >=25% improvement") == "failed_or_metric_blocked"
            and status_by_requirement.get("do not mark goal complete") == "blocked"
        ),
    }
    return {
        "created_at": _now(),
        "ok": all(bool(value) for value in checks.values()),
        "checks": checks,
        "observed": {
            "csv_row_count": len(csv_rows),
            "json_row_count": len(json_rows),
            "missing_requirements": missing_requirements,
            "evidence_count": len(evidence),
            "missing_evidence": missing_evidence,
            "goal_complete": checklist.get("goal_complete"),
            "pass_count": checklist.get("pass_count"),
            "required_pass_count": checklist.get("required_pass_count"),
            "update_goal_allowed": blocker.get("update_goal_allowed"),
        },
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Prompt-To-Artifact Checklist Verification",
        "",
        f"작성일: {result.get('created_at', '')}",
        "",
        f"- overall ok: `{result.get('ok')}`",
        "",
        "## Observed",
        "",
    ]
    for key, value in result.get("observed", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks", "", "| check | pass |", "|---|---|"])
    for key, value in result.get("checks", {}).items():
        lines.append(f"| {key} | {bool(value)} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This verifier confirms that the checklist maps the active goal to existing evidence artifacts.",
            "It does not imply the research goal is complete.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checklist_csv", default=str(DEFAULT_CHECKLIST_CSV))
    parser.add_argument("--checklist_json", default=str(DEFAULT_CHECKLIST_JSON))
    parser.add_argument("--completion_blocker_json", default=str(DEFAULT_COMPLETION_BLOCKER_JSON))
    parser.add_argument("--goal_status_json", default=str(DEFAULT_GOAL_STATUS_JSON))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--fail_on_error", action="store_true")
    args = parser.parse_args()

    result = verify(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_report(out_dir / "verification.md", result)
    print(json.dumps(result, indent=2))
    if args.fail_on_error and not result["ok"]:
        raise SystemExit("Prompt-to-artifact checklist verification failed.")


if __name__ == "__main__":
    main()
