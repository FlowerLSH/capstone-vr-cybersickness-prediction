"""Generate or smoke-test DenseFMS static feature ablation commands."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


CONDITION_CONFIGS = {
    ("no_static", "level_only"): "configs/coff_lstm_no_static_level.yaml",
    ("no_static", "level_trend_raw"): "configs/coff_lstm_no_static_trend.yaml",
    ("age_gender", "level_only"): "configs/coff_lstm_static_level.yaml",
    ("age_gender", "level_trend_raw"): "configs/coff_lstm_static_trend.yaml",
    ("full_static", "level_only"): "configs/coff_lstm_static_full_level.yaml",
    ("full_static", "level_trend_raw"): "configs/coff_lstm_static_full_trend.yaml",
}

STATIC_ARGS = {
    "no_static": [],
    "age_gender": ["--use_static", "--static_features", "age", "gender"],
    "full_static": ["--use_static", "--static_features", "age", "gender", "mssq"],
}


def run_name(condition: str, loss_mode: str, trend_weight: float, smoke: bool = False) -> str:
    prefix = "smoke_" if smoke else ""
    loss_tag = "trend" if loss_mode == "level_trend_raw" else "level"
    if loss_mode == "level_trend_raw":
        return f"{prefix}coff_lstm_{condition}_{loss_tag}_w{trend_weight:g}"
    return f"{prefix}coff_lstm_{condition}_{loss_tag}"


def train_command(args: argparse.Namespace, condition: str) -> List[str]:
    trend_weight = float(args.trend_weight if args.trend_weight is not None else (0.1 if args.loss_mode == "level_trend_raw" else 0.0))
    cmd = [
        sys.executable,
        "-m",
        "src.densefms_forecast.train",
        "--data_dir",
        args.data_dir,
        "--config",
        CONDITION_CONFIGS[(condition, args.loss_mode)],
        "--model",
        "coff_lstm",
        "--loss_mode",
        args.loss_mode,
        "--trend_weight",
        f"{trend_weight:g}",
        "--split_file",
        args.split_file,
        "--run_name",
        run_name(condition, args.loss_mode, trend_weight, smoke=args.smoke_test),
    ]
    cmd.extend(STATIC_ARGS[condition])
    if args.allow_missing_static:
        cmd.append("--allow_missing_static")
    if args.smoke_test:
        cmd.extend(["--epochs", str(args.smoke_epochs), "--limit_sessions", str(args.limit_sessions)])
    return cmd


def evaluate_command(train_cmd: List[str], data_dir: str) -> List[str]:
    run = train_cmd[train_cmd.index("--run_name") + 1]
    return [
        sys.executable,
        "-m",
        "src.densefms_forecast.evaluate",
        "--checkpoint",
        str(Path("runs") / run / "best.pt"),
        "--data_dir",
        data_dir,
        "--split",
        "test",
    ]


def format_cmd(cmd: List[str]) -> str:
    return " ".join(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or print DenseFMS static feature ablation commands.")
    parser.add_argument("--data_dir", default="./DenseFMS/Dataset")
    parser.add_argument("--split_file", default="./artifacts/densefms_split_seed42.json")
    parser.add_argument("--conditions", nargs="+", default=["no_static", "age_gender", "full_static"], choices=["no_static", "age_gender", "full_static"])
    parser.add_argument("--loss_mode", default="level_only", choices=["level_only", "level_trend_raw"])
    parser.add_argument("--trend_weight", type=float, default=None)
    parser.add_argument("--allow_missing_static", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke_epochs", type=int, default=1)
    parser.add_argument("--limit_sessions", type=int, default=8)
    args = parser.parse_args()

    if args.dry_run and args.smoke_test:
        raise ValueError("Choose either --dry_run or --smoke_test, not both.")

    commands: Dict[str, List[str]] = {condition: train_command(args, condition) for condition in args.conditions}
    for condition, cmd in commands.items():
        print(f"\n[{condition}] train")
        print(format_cmd(cmd))
        print(f"[{condition}] evaluate")
        print(format_cmd(evaluate_command(cmd, args.data_dir)))

    if args.dry_run:
        return

    if args.smoke_test:
        for condition, cmd in commands.items():
            print(f"\nRunning smoke test for {condition}...")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
