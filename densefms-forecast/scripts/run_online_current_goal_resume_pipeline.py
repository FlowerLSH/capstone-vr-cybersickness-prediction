"""Resume pipeline for the strict 120s online-current goal.

Default behavior is dry-run only. Safe non-training steps can be run with
--run_safe_steps. Validation training requires --execute_validation and still
uses the early-fusion launcher's cutoff guard. This script never executes the
original test set; it only prepares final-test promotion commands.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = "reports/overnight_current_fms_goal_0514_120s/resume_pipeline"


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z %z")


def is_past_cutoff(now: datetime, cutoff_hour: int) -> bool:
    return now.hour >= int(cutoff_hour)


def build_cutoff_status(args: argparse.Namespace, now: datetime | None = None) -> Dict[str, Any]:
    current = now.astimezone() if now is not None else datetime.now().astimezone()
    cutoff_passed = is_past_cutoff(current, int(args.cutoff_hour))
    training_blocked = bool(args.execute_validation) and cutoff_passed and not bool(args.disable_cutoff_guard)
    return {
        "checked_at": current.strftime("%Y-%m-%d %H:%M:%S %Z %z"),
        "cutoff_hour": int(args.cutoff_hour),
        "cutoff_passed": bool(cutoff_passed),
        "cutoff_guard_disabled": bool(args.disable_cutoff_guard),
        "execute_validation_requested": bool(args.execute_validation),
        "validation_training_blocked_by_cutoff": bool(training_blocked),
        "validation_training_allowed_by_time": bool(bool(args.execute_validation) and (not cutoff_passed or bool(args.disable_cutoff_guard))),
    }


def build_step_commands(args: argparse.Namespace) -> List[Dict[str, Any]]:
    validation = [
        args.python,
        "scripts/run_online_current_calsummary_earlyfusion_experiments.py",
        "--cutoff_hour",
        str(int(args.cutoff_hour)),
    ]
    if args.disable_cutoff_guard:
        validation.append("--disable_cutoff_guard")
    if args.execute_validation:
        validation.append("--execute")

    return [
        {
            "name": "validation_earlyfusion",
            "kind": "training" if args.execute_validation else "dry_run_command_generation",
            "may_train": bool(args.execute_validation),
            "runs_test": False,
            "command": validation,
        },
        {
            "name": "validation_gate",
            "kind": "safe_validation_report",
            "may_train": False,
            "runs_test": False,
            "command": [args.python, "scripts/evaluate_online_current_calsummary_earlyfusion_gates.py"],
        },
        {
            "name": "prepare_test_promotion",
            "kind": "safe_test_command_dry_run",
            "may_train": False,
            "runs_test": False,
            "command": [args.python, "scripts/prepare_online_current_test_promotion.py"],
        },
        {
            "name": "goal_status_audit",
            "kind": "safe_metric_audit",
            "may_train": False,
            "runs_test": False,
            "command": [args.python, "scripts/audit_online_current_goal_status.py"],
        },
    ]


def _write_report(
    path: Path,
    args: argparse.Namespace,
    steps: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
    time_status: Mapping[str, Any],
) -> None:
    lines = [
        "# Strict 120s Goal Resume Pipeline",
        "",
        f"작성일: {_now()}",
        "",
        "This pipeline coordinates the next validation-only resume sequence. It does not execute original test.",
        "",
        "## Mode",
        "",
        f"- run safe steps: `{bool(args.run_safe_steps)}`",
        f"- execute validation training: `{bool(args.execute_validation)}`",
        f"- cutoff hour: `{args.cutoff_hour}`",
        f"- cutoff guard disabled: `{bool(args.disable_cutoff_guard)}`",
        "",
        "## Time Check",
        "",
        f"- checked at: `{time_status.get('checked_at')}`",
        f"- cutoff passed: `{time_status.get('cutoff_passed')}`",
        f"- validation training blocked by cutoff: `{time_status.get('validation_training_blocked_by_cutoff')}`",
        f"- validation training allowed by time: `{time_status.get('validation_training_allowed_by_time')}`",
        "",
        "## Steps",
        "",
        "| step | kind | may train | runs test | command |",
        "|---|---|---|---|---|",
    ]
    for step in steps:
        lines.append(
            "| {name} | {kind} | {train} | {test} | `{cmd}` |".format(
                name=step["name"],
                kind=step["kind"],
                train=bool(step["may_train"]),
                test=bool(step["runs_test"]),
                cmd=shlex.join([str(part) for part in step["command"]]),
            )
        )
    lines.extend(["", "## Results", "", "| step | status | return code |", "|---|---|---:|"])
    if not results:
        lines.append("| not_run | dry-run only |  |")
    else:
        for row in results:
            lines.append(f"| {row['name']} | {row['status']} | {row.get('returncode', '')} |")
    lines.extend(
        [
            "",
            "## Rules",
            "",
            "- Original test is never executed by this pipeline.",
            "- Use `--execute_validation` only when full training is explicitly open and the cutoff has not passed.",
            "- After safe steps, inspect `test_promotion_commands.md`; run original test only if exactly one validation-gated command is present.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_steps(steps: Sequence[Dict[str, Any]], *, run_safe_steps: bool) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for step in steps:
        if step["may_train"] or run_safe_steps:
            proc = subprocess.run([str(part) for part in step["command"]], cwd=ROOT)
            results.append(
                {
                    "name": step["name"],
                    "status": "ok" if proc.returncode == 0 else "failed",
                    "returncode": int(proc.returncode),
                }
            )
            if proc.returncode != 0:
                break
        else:
            results.append({"name": step["name"], "status": "planned_only", "returncode": ""})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--report_dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--run_safe_steps", action="store_true")
    parser.add_argument("--execute_validation", action="store_true")
    parser.add_argument("--cutoff_hour", type=int, default=12)
    parser.add_argument("--disable_cutoff_guard", action="store_true")
    args = parser.parse_args()

    steps = build_step_commands(args)
    if any(step["runs_test"] for step in steps):
        raise SystemExit("Internal error: resume pipeline must not execute original test.")
    time_status = build_cutoff_status(args)
    results = _run_steps(steps, run_safe_steps=bool(args.run_safe_steps))
    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": _now(), "time_status": time_status, "steps": steps, "results": results}
    (out_dir / "resume_pipeline.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(out_dir / "resume_pipeline.md", args, steps, results, time_status)
    print(f"wrote resume pipeline report to {out_dir}")


if __name__ == "__main__":
    main()
