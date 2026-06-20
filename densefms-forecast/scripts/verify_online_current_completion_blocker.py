"""Verify the strict 120s completion blocker.

This script reads existing reports only. It does not train, run validation, or
evaluate the original test set.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping


DEFAULT_BASE_DIR = Path("reports/overnight_current_fms_goal_0514_120s")
DEFAULT_BLOCKER_JSON = Path("reports/overnight_current_fms_goal_0514_120s_completion_blocker.json")
DEFAULT_BLOCKER_MD = Path("reports/overnight_current_fms_goal_0514_120s_completion_blocker.md")
DEFAULT_GOAL_STATUS_JSON = DEFAULT_BASE_DIR / "goal_status_audit/goal_status_audit.json"
DEFAULT_GATE_CSV = DEFAULT_BASE_DIR / "calsummary_earlyfusion_gate_eval/validation_gate_metrics.csv"
DEFAULT_PROMOTION_CSV = DEFAULT_BASE_DIR / "test_promotion_commands/test_promotion_commands.csv"
DEFAULT_CUTOFF_LOCK_JSON = Path("reports/overnight_current_fms_goal_0514_120s_cutoff_lock.json")
DEFAULT_PROCESS_AUDIT_JSON = DEFAULT_BASE_DIR / "post_cutoff_process_audit/process_audit.json"
DEFAULT_OUT_DIR = DEFAULT_BASE_DIR / "completion_blocker_verification"


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


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def _close(value: object, expected: float, tolerance: float = 1e-3) -> bool:
    try:
        return abs(float(value) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def verify(args: argparse.Namespace) -> Dict[str, Any]:
    blocker_path = Path(args.blocker_json)
    blocker_md_path = Path(args.blocker_md)
    goal_path = Path(args.goal_status_json)
    gate_path = Path(args.gate_csv)
    promotion_path = Path(args.promotion_csv)
    cutoff_path = Path(args.cutoff_lock_json)
    process_path = Path(args.process_audit_json)

    blocker = _read_json(blocker_path)
    goal = _read_json(goal_path)
    gate_rows = _read_csv(gate_path)
    promotion_rows = _read_csv(promotion_path)
    cutoff = _read_json(cutoff_path)
    process = _read_json(process_path)
    blocker_text = blocker_md_path.read_text(encoding="utf-8") if blocker_md_path.exists() else ""
    criteria = blocker.get("criteria", {}) if isinstance(blocker, dict) else {}
    c4 = criteria.get("high8_high12_plus_25_percent", {}) if isinstance(criteria, dict) else {}
    goal_criteria = goal.get("criteria", {}) if isinstance(goal, dict) else {}
    gate_candidate_count = sum(1 for row in gate_rows if _truthy(row.get("test_candidate", False)))
    promotion_command_count = sum(1 for row in promotion_rows if str(row.get("command", "")).strip())

    checks = {
        "blocker_json_exists": blocker_path.exists(),
        "blocker_md_exists": blocker_md_path.exists(),
        "blocker_marks_incomplete": blocker.get("goal_complete") is False,
        "blocker_disallows_update_goal": blocker.get("update_goal_allowed") is False,
        "blocker_pass_count_zero": blocker.get("pass_count") == 0,
        "blocker_required_pass_count_three": blocker.get("required_pass_count") == 3,
        "blocker_current_best_matches_goal_audit": blocker.get("current_strict_best") == goal.get("label"),
        "blocker_goal_status_match": (
            blocker.get("goal_complete") == goal.get("goal_complete")
            and blocker.get("pass_count") == goal.get("pass_count")
            and blocker.get("required_pass_count") == goal.get("required_pass_count")
        ),
        "blocker_c1_matches_goal_audit": _close(
            criteria.get("test_mae_le_1_8", {}).get("value"),
            goal_criteria.get("C1_MAE", {}).get("value"),
        ),
        "blocker_c2_matches_goal_audit": _close(
            criteria.get("test_r2_ge_0_75", {}).get("value"),
            goal_criteria.get("C2_R2", {}).get("value"),
        ),
        "blocker_c3_matches_goal_audit": _close(
            criteria.get("strict_original_low_fms_0_2_signed_bias_le_2_5", {}).get("value"),
            goal_criteria.get("C3_STRICT_LOW_BIAS", {}).get("value"),
        ),
        "blocker_c4_matches_goal_audit": (
            _close(c4.get("high8_f1"), goal_criteria.get("C4_HIGH8_HIGH12_F1_RELATIVE", {}).get("high8_value"))
            and _close(c4.get("high8_f1_target"), goal_criteria.get("C4_HIGH8_HIGH12_F1_RELATIVE", {}).get("high8_target"))
            and _close(c4.get("high12_f1"), goal_criteria.get("C4_HIGH8_HIGH12_F1_RELATIVE", {}).get("high12_value"))
            and _close(c4.get("high12_f1_target"), goal_criteria.get("C4_HIGH8_HIGH12_F1_RELATIVE", {}).get("high12_target"))
            and c4.get("possible_under_high8_f1") is False
        ),
        "all_blocker_criteria_false": all(
            row.get("pass") is False for row in criteria.values() if isinstance(row, dict) and "pass" in row
        ),
        "validation_gate_counts_match": (
            blocker.get("validation_gate", {}).get("validation_gated_test_candidates") == gate_candidate_count
            and blocker.get("validation_gate", {}).get("test_promotion_commands") == promotion_command_count
        ),
        "cutoff_lock_counts_match": (
            blocker.get("cutoff", {}).get("cutoff_passed") == cutoff.get("cutoff", {}).get("cutoff_passed")
            and blocker.get("cutoff", {}).get("new_full_training_allowed") == cutoff.get("cutoff", {}).get("new_full_training_allowed")
            and blocker.get("cutoff", {}).get("original_test_allowed") == cutoff.get("cutoff", {}).get("original_test_allowed")
        ),
        "process_audit_blocks_running_training": process.get("ok") is True,
        "blocker_text_says_do_not_update_goal": "Do not call `update_goal`" in blocker_text,
    }
    return {
        "created_at": _now(),
        "ok": all(bool(value) for value in checks.values()),
        "checks": checks,
        "observed": {
            "goal_complete": blocker.get("goal_complete"),
            "update_goal_allowed": blocker.get("update_goal_allowed"),
            "pass_count": blocker.get("pass_count"),
            "required_pass_count": blocker.get("required_pass_count"),
            "current_strict_best": blocker.get("current_strict_best"),
            "validation_gated_test_candidates": gate_candidate_count,
            "test_promotion_commands": promotion_command_count,
            "new_full_training_allowed": blocker.get("cutoff", {}).get("new_full_training_allowed"),
            "original_test_allowed": blocker.get("cutoff", {}).get("original_test_allowed"),
        },
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Completion Blocker Verification",
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
            "This verifier confirms that the completion blocker matches the current goal audit and cutoff state.",
            "It does not imply the research goal is complete.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocker_json", default=str(DEFAULT_BLOCKER_JSON))
    parser.add_argument("--blocker_md", default=str(DEFAULT_BLOCKER_MD))
    parser.add_argument("--goal_status_json", default=str(DEFAULT_GOAL_STATUS_JSON))
    parser.add_argument("--gate_csv", default=str(DEFAULT_GATE_CSV))
    parser.add_argument("--promotion_csv", default=str(DEFAULT_PROMOTION_CSV))
    parser.add_argument("--cutoff_lock_json", default=str(DEFAULT_CUTOFF_LOCK_JSON))
    parser.add_argument("--process_audit_json", default=str(DEFAULT_PROCESS_AUDIT_JSON))
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
        raise SystemExit("Completion blocker verification failed.")


if __name__ == "__main__":
    main()
