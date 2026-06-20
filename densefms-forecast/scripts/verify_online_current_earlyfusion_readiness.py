"""Verify readiness of prepared early-fusion validation-only commands.

This verifier reads generated command CSVs and checks file prerequisites and
test-safety flags. It does not train or evaluate.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


DEFAULT_COMMAND_CSV = "reports/overnight_current_fms_goal_0514_120s/calsummary_earlyfusion_commands/commands.csv"
DEFAULT_COMMAND_MD = "reports/overnight_current_fms_goal_0514_120s/calsummary_earlyfusion_commands/commands.md"
DEFAULT_OUT_DIR = "reports/overnight_current_fms_goal_0514_120s/earlyfusion_readiness"


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z %z")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _arg_value(parts: Sequence[str], name: str) -> str:
    if name not in parts:
        return ""
    idx = parts.index(name)
    if idx + 1 >= len(parts):
        return ""
    return str(parts[idx + 1])


def _has_flag(parts: Sequence[str], name: str) -> bool:
    return name in parts


def _path_exists(value: str, *, expect_dir: bool = False) -> bool:
    if not value:
        return False
    path = Path(value)
    return path.is_dir() if expect_dir else path.exists()


def _config_calibration_seconds(value: str) -> float | None:
    if not value:
        return None
    path = Path(value)
    if not path.exists():
        return None
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    try:
        return float(config.get("data", {}).get("calibration_seconds"))
    except (TypeError, ValueError):
        return None


def _read_command_report_time_status(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "command_report_exists": False,
            "time_check_present": False,
            "checked_at": "",
            "cutoff_passed": None,
            "execute_requested": None,
            "execute_allowed_by_time": None,
        }
    text = path.read_text(encoding="utf-8")

    def _line_value(label: str) -> str:
        prefix = f"- {label}: `"
        for line in text.splitlines():
            if line.startswith(prefix) and line.endswith("`"):
                return line[len(prefix) : -1]
        return ""

    def _bool_value(label: str) -> Any:
        value = _line_value(label)
        if value == "":
            return None
        return value.strip().lower() == "true"

    checked_at = _line_value("checked at")
    return {
        "command_report_exists": True,
        "time_check_present": "## Time Check" in text and bool(checked_at),
        "checked_at": checked_at,
        "cutoff_passed": _bool_value("cutoff passed"),
        "execute_requested": _bool_value("execute requested"),
        "execute_allowed_by_time": _bool_value("execute allowed by time"),
    }


def verify_command(row: Mapping[str, str]) -> Dict[str, Any]:
    command = row.get("command", "")
    parts = shlex.split(command)
    data_dir = _arg_value(parts, "--data_dir")
    config = _arg_value(parts, "--config")
    checkpoint = _arg_value(parts, "--init_checkpoint")
    features = _arg_value(parts, "--calibration_residual_features_path")
    max_points = _arg_value(parts, "--max_session_points")
    selection_metric = _arg_value(parts, "--selection_metric")
    config_calibration_seconds = _config_calibration_seconds(config)
    checks = {
        "command_parses": bool(parts),
        "data_dir_exists": _path_exists(data_dir, expect_dir=True),
        "config_exists": _path_exists(config),
        "config_calibration_seconds_120": config_calibration_seconds == 120.0,
        "init_checkpoint_exists": _path_exists(checkpoint),
        "calibration_summary_features_exists": _path_exists(features),
        "uses_no_test_eval": _has_flag(parts, "--no_test_eval"),
        "does_not_request_test_split": "--split" not in parts and "--split_file" not in parts,
        "requires_calibration_residual_features": _has_flag(parts, "--require_calibration_residual_features"),
        "enables_calibration_summary_fusion": _has_flag(parts, "--calibration_summary_fusion_enabled"),
        "max_session_points_420": str(max_points) == "420",
        "selection_metric_strict120": str(selection_metric) == "goal_composite.strict120",
    }
    return {
        "run_name": row.get("run_name", ""),
        "description": row.get("description", ""),
        "ok": all(bool(value) for value in checks.values()),
        "checks": checks,
        "paths": {
            "data_dir": data_dir,
            "config": config,
            "init_checkpoint": checkpoint,
            "calibration_summary_features": features,
            "config_calibration_seconds": config_calibration_seconds,
        },
    }


def verify(args: argparse.Namespace) -> Dict[str, Any]:
    rows = _read_csv(Path(args.command_csv))
    command_results = [verify_command(row) for row in rows]
    time_status = _read_command_report_time_status(Path(args.command_md))
    return {
        "created_at": _now(),
        "command_csv": str(args.command_csv),
        "command_md": str(args.command_md),
        "command_count": len(rows),
        "time_status": time_status,
        "ok": (
            bool(command_results)
            and all(bool(row["ok"]) for row in command_results)
            and bool(time_status.get("time_check_present"))
        ),
        "commands": command_results,
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Early-Fusion Readiness Verification",
        "",
        f"작성일: {result.get('created_at', '')}",
        "",
        f"- command csv: `{result.get('command_csv')}`",
        f"- command report: `{result.get('command_md')}`",
        f"- command count: `{result.get('command_count')}`",
        f"- overall ok: `{result.get('ok')}`",
        "",
        "## Time Check",
        "",
        f"- time check present: `{result.get('time_status', {}).get('time_check_present')}`",
        f"- checked at: `{result.get('time_status', {}).get('checked_at')}`",
        f"- cutoff passed: `{result.get('time_status', {}).get('cutoff_passed')}`",
        f"- execute requested: `{result.get('time_status', {}).get('execute_requested')}`",
        f"- execute allowed by time: `{result.get('time_status', {}).get('execute_allowed_by_time')}`",
        "",
        "## Commands",
        "",
        "| run | ok | failed checks |",
        "|---|---|---|",
    ]
    for row in result.get("commands", []):
        failed = [name for name, passed in row.get("checks", {}).items() if not passed]
        lines.append(
            "| {run} | {ok} | {failed} |".format(
                run=row.get("run_name", ""),
                ok=bool(row.get("ok")),
                failed=", ".join(failed) if failed else "",
            )
        )
    lines.extend(
        [
            "",
            "## Rule",
            "",
            "This verifier only checks readiness and test-safety flags. It does not execute training or test evaluation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command_csv", default=DEFAULT_COMMAND_CSV)
    parser.add_argument("--command_md", default=DEFAULT_COMMAND_MD)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--fail_on_error", action="store_true")
    args = parser.parse_args()
    result = verify(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "earlyfusion_readiness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_report(out_dir / "earlyfusion_readiness.md", result)
    print(json.dumps(result, indent=2))
    if args.fail_on_error and not result["ok"]:
        raise SystemExit("Early-fusion readiness verification failed.")


if __name__ == "__main__":
    main()
