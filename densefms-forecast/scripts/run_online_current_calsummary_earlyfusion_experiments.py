"""Generate or execute calibration-summary early-fusion validation experiments."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

BASE_CONFIG = "runs/head_redesign_ablation_0513/range_scaled_delta2_120_seed42/config_snapshot.json"
BASE_CHECKPOINT = "runs/head_redesign_ablation_0513/range_scaled_delta2_120_seed42/best.pt"
SUMMARY_FEATURES = "reports/overnight_current_fms_goal_0514_120s/calibration_summary_features_train_val.json"

TRAINABLE_PATTERNS = [
    "calibration_summary_fusion",
    "current_range_",
    "current_reg_head",
    "ordinal_head",
]

RECIPES: Sequence[Mapping[str, Any]] = [
    {
        "run_name": "range_calsummary_earlyfusion_add_goalcomp_seed42",
        "description": "Identity-initialized additive-gated calibration summary fusion before current/future heads.",
        "extra_args": [
            "--calibration_summary_fusion_mode",
            "additive_gated",
            "--calibration_summary_fusion_strength",
            "1.0",
        ],
    },
    {
        "run_name": "range_calsummary_earlyfusion_add_low002_goalcomp_seed42",
        "description": "Additive-gated early fusion with weak low-overprediction penalty.",
        "extra_args": [
            "--calibration_summary_fusion_mode",
            "additive_gated",
            "--calibration_summary_fusion_strength",
            "1.0",
            "--low_overprediction_weight",
            "0.02",
            "--high_underprediction_weight",
            "0.01",
        ],
    },
]


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z %z")


def _is_past_cutoff(now: datetime, cutoff_hour: int) -> bool:
    if cutoff_hour < 0 or cutoff_hour > 23:
        raise ValueError("cutoff_hour must be in [0, 23].")
    cutoff = now.replace(hour=int(cutoff_hour), minute=0, second=0, microsecond=0)
    return now >= cutoff


def _time_status(args: argparse.Namespace, now: datetime | None = None) -> Dict[str, Any]:
    current = now.astimezone() if now is not None else datetime.now().astimezone()
    cutoff_passed = _is_past_cutoff(current, int(args.cutoff_hour))
    return {
        "checked_at": current.strftime("%Y-%m-%d %H:%M:%S %Z %z"),
        "cutoff_hour": int(args.cutoff_hour),
        "cutoff_passed": bool(cutoff_passed),
        "cutoff_guard_disabled": bool(args.disable_cutoff_guard),
        "execute_requested": bool(args.execute),
        "execute_allowed_by_time": bool(not cutoff_passed or bool(args.disable_cutoff_guard)),
    }


def _check_cutoff(args: argparse.Namespace) -> None:
    if args.disable_cutoff_guard:
        return
    now = datetime.now().astimezone()
    if _is_past_cutoff(now, int(args.cutoff_hour)):
        raise SystemExit(
            "Refusing to start a full run because the local cutoff has passed: "
            f"now={now.strftime('%Y-%m-%d %H:%M:%S %Z %z')}, cutoff_hour={args.cutoff_hour}. "
            "Use --disable_cutoff_guard only if the user explicitly reopens training."
        )


def _command(args: argparse.Namespace, recipe: Mapping[str, Any]) -> List[str]:
    cmd = [
        args.python,
        "-m",
        "src.densefms_forecast.train",
        "--data_dir",
        args.data_dir,
        "--config",
        args.base_config,
        "--model",
        "online_fms_risk_tracker",
        "--runs_dir",
        args.runs_dir,
        "--run_name",
        str(recipe["run_name"]),
        "--init_checkpoint",
        args.init_checkpoint,
        "--freeze_loaded_parameters",
        "--trainable_parameter_patterns",
        *TRAINABLE_PATTERNS,
        "--calibration_residual_features_path",
        args.calibration_summary_features,
        "--require_calibration_residual_features",
        "--calibration_summary_fusion_enabled",
        "--selection_metric",
        args.selection_metric,
        "--selection_mode",
        "min",
        "--learning_rate",
        str(args.learning_rate),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--max_session_points",
        str(args.max_session_points),
        "--no_test_eval",
        "--save_predictions",
        "--save_plots",
        "--skip_existing",
    ]
    if args.split_file:
        cmd.extend(["--split_file", args.split_file])
    if args.smoke:
        cmd.extend(
            [
                "--limit_sessions",
                str(args.smoke_limit_sessions),
                "--max_train_batches",
                str(args.smoke_max_train_batches),
                "--max_eval_batches",
                str(args.smoke_max_eval_batches),
                "--epochs",
                str(args.smoke_epochs),
                "--patience",
                str(args.smoke_patience),
            ]
        )
    cmd.extend(str(value) for value in recipe.get("extra_args", []))
    return cmd


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_name", "description", "command"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_summary(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
    time_status: Mapping[str, Any],
) -> None:
    lines = [
        "# Calibration-Summary Early-Fusion Experiment Commands",
        "",
        f"작성일: {_now()}",
        "",
        "이 파일은 validation-only 후보 실행 command를 기록한다. Original test는 여기서 실행하지 않는다.",
        "",
        "## Shared Settings",
        "",
        f"- base config: `{args.base_config}`",
        f"- init checkpoint: `{args.init_checkpoint}`",
        f"- calibration summary features: `{args.calibration_summary_features}`",
        f"- runs dir: `{args.runs_dir}`",
        f"- selection metric: `{args.selection_metric}`",
        "- test: `--no_test_eval`",
        f"- execute cutoff guard: cutoff_hour={args.cutoff_hour}, disabled={args.disable_cutoff_guard}",
        "",
        "## Time Check",
        "",
        f"- checked at: `{time_status.get('checked_at')}`",
        f"- cutoff passed: `{time_status.get('cutoff_passed')}`",
        f"- execute requested: `{time_status.get('execute_requested')}`",
        f"- execute allowed by time: `{time_status.get('execute_allowed_by_time')}`",
        "",
        "## Commands",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['run_name']}",
                "",
                str(row["description"]),
                "",
                "```bash",
                str(row["command"]),
                "```",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe")
    parser.add_argument("--data_dir", default="DenseFMS/Dataset")
    parser.add_argument("--base_config", default=BASE_CONFIG)
    parser.add_argument("--init_checkpoint", default=BASE_CHECKPOINT)
    parser.add_argument("--calibration_summary_features", default=SUMMARY_FEATURES)
    parser.add_argument("--runs_dir", default="runs/overnight_current_fms_goal_0514_120s")
    parser.add_argument("--report_dir", default="reports/overnight_current_fms_goal_0514_120s/calsummary_earlyfusion_commands")
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--selection_metric", default="goal_composite.strict120")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_session_points", type=int, default=420)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_limit_sessions", type=int, default=12)
    parser.add_argument("--smoke_max_train_batches", type=int, default=2)
    parser.add_argument("--smoke_max_eval_batches", type=int, default=1)
    parser.add_argument("--smoke_epochs", type=int, default=1)
    parser.add_argument("--smoke_patience", type=int, default=1)
    parser.add_argument("--cutoff_hour", type=int, default=12)
    parser.add_argument("--disable_cutoff_guard", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default only writes dry-run outputs.")
    args = parser.parse_args()
    time_status = _time_status(args)

    rows: List[Dict[str, str]] = []
    commands: List[tuple[Mapping[str, Any], List[str]]] = []
    for recipe in RECIPES:
        cmd = _command(args, recipe)
        commands.append((recipe, cmd))
        rows.append(
            {
                "run_name": str(recipe["run_name"]),
                "description": str(recipe["description"]),
                "command": shlex.join(cmd),
            }
        )
    report_dir = Path(args.report_dir)
    _write_csv(report_dir / "commands.csv", rows)
    _write_summary(report_dir / "commands.md", rows, args, time_status)
    if args.execute:
        for recipe, cmd in commands:
            _check_cutoff(args)
            print(f"[{_now()}] running {recipe['run_name']}", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"wrote {len(rows)} command(s) to {report_dir}")


if __name__ == "__main__":
    main()
