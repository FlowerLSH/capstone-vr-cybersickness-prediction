"""Run or dry-run warning/future-extension experiments for final online-current candidates."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]


RECIPES: Dict[str, Dict[str, Any]] = {
    "selected_risk035": {
        "config": "configs/online_current/selected_deeptcn_risk035_static4.yaml",
        "checkpoint": "results/selected_deeptcn_risk035_static4/checkpoints/best.pt",
        "description": "Original selected DeepTCN risk035 static4 baseline.",
    },
    "risk045_smooth005": {
        "config": "runs/calibration_branch_revision_0513/cbr_bestbase_risk045_smooth005_seed42/config_snapshot.json",
        "checkpoint": "runs/calibration_branch_revision_0513/cbr_bestbase_risk045_smooth005_seed42/best.pt",
        "description": "Risk0.45 + smooth0.005 strengthened baseline.",
    },
    "zero_anchor_highgate_delta2": {
        "config": "runs/calibration_branch_revision_0513/cbr_zero_anchor_highgate_t12_w030_pos4_delta2_seed42/config_snapshot.json",
        "checkpoint": "runs/calibration_branch_revision_0513/cbr_zero_anchor_highgate_t12_w030_pos4_delta2_seed42/best.pt",
        "description": "MAE-best zero-anchor high-gate delta2 candidate.",
    },
    "range_scaled_delta2": {
        "config": "runs/head_redesign_ablation_0513/range_scaled_delta2_120_seed42/config_snapshot.json",
        "checkpoint": "runs/head_redesign_ablation_0513/range_scaled_delta2_120_seed42/best.pt",
        "description": "Plot/range-preserving range-scaled delta2 candidate.",
    },
}


TRAINABLE_HEAD_PATTERNS = [
    "uncertainty_head",
    "future_aux_head",
    "event_aux_head",
    "risk_head",
    "fall_risk_head",
    "high_risk_head",
]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _command(args: argparse.Namespace, recipe_name: str) -> List[str]:
    recipe = RECIPES[recipe_name]
    run_kind = "smoke" if args.smoke else "full"
    run_name = f"{recipe_name}_warnext_stage1_{run_kind}_seed{args.seed}"
    cmd = [
        args.python,
        "-m",
        "src.densefms_forecast.train",
        "--data_dir",
        args.data_dir,
        "--config",
        str(recipe["config"]),
        "--model",
        "online_fms_risk_tracker",
        "--run_name",
        run_name,
        "--runs_dir",
        args.runs_dir,
        "--split_file",
        args.split_file,
        "--task_mode",
        "online_current_risk",
        "--init_checkpoint",
        str(recipe["checkpoint"]),
        "--freeze_loaded_parameters",
        "--trainable_parameter_patterns",
        *TRAINABLE_HEAD_PATTERNS,
        "--uncertainty_head_enabled",
        "--uncertainty_loss_weight",
        str(args.uncertainty_loss_weight),
        "--future_aux_horizon_seconds",
        "5.0",
        "10.0",
        "20.0",
        "30.0",
        "--future_aux_loss_weight",
        str(args.future_aux_loss_weight),
        "--delta_aux_loss_weight",
        str(args.delta_aux_loss_weight),
        "--event_aux_loss_weight",
        str(args.event_aux_loss_weight),
        "--event_delta_threshold",
        str(args.event_delta_threshold),
        "--rise_horizon_seconds",
        "10.0",
        "20.0",
        "--rise_thresholds",
        "2.0",
        "3.0",
        "--fall_risk_head_enabled",
        "--fall_horizon_seconds",
        "10.0",
        "20.0",
        "--fall_thresholds",
        "2.0",
        "3.0",
        "--risk_loss_weight",
        str(args.risk_loss_weight),
        "--fall_loss_weight",
        str(args.fall_loss_weight),
        "--high_risk_head_enabled",
        "--high_risk_horizon_seconds",
        "20.0",
        "--high_risk_thresholds",
        "8.0",
        "12.0",
        "--high_risk_loss_weight",
        str(args.high_risk_loss_weight),
        "--high_fms_caution_threshold",
        "8.0",
        "--high_fms_warning_threshold",
        "12.0",
        "--rapid_rise_probability_threshold",
        str(args.probability_threshold),
        "--rapid_drop_probability_threshold",
        str(args.probability_threshold),
        "--final_warning_mode",
        "high_or_rapid",
        "--selection_metric",
        args.selection_metric,
        "--selection_mode",
        args.selection_mode,
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
    return cmd


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["recipe", "description", "config", "checkpoint", "run_name", "command"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online-current warning-extension candidate experiments.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data_dir", default="DenseFMS/Dataset")
    parser.add_argument("--split_file", default="results/selected_deeptcn_risk035_static4/split.json")
    parser.add_argument("--runs_dir", default="runs/online_current_warning_extension_0514")
    parser.add_argument("--report_dir", default="reports/online_current_warning_extension_0514")
    parser.add_argument("--recipes", nargs="+", default=list(RECIPES))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_session_points", type=int, default=420)
    parser.add_argument("--selection_metric", default="mae")
    parser.add_argument("--selection_mode", choices=["min", "max"], default="min")
    parser.add_argument("--uncertainty_loss_weight", type=float, default=0.005)
    parser.add_argument("--future_aux_loss_weight", type=float, default=0.2)
    parser.add_argument("--delta_aux_loss_weight", type=float, default=0.2)
    parser.add_argument("--event_aux_loss_weight", type=float, default=0.05)
    parser.add_argument("--event_delta_threshold", type=float, default=2.0)
    parser.add_argument("--risk_loss_weight", type=float, default=0.45)
    parser.add_argument("--fall_loss_weight", type=float, default=0.30)
    parser.add_argument("--high_risk_loss_weight", type=float, default=0.50)
    parser.add_argument("--probability_threshold", type=float, default=0.5)
    parser.add_argument("--smoke_limit_sessions", type=int, default=24)
    parser.add_argument("--smoke_max_train_batches", type=int, default=2)
    parser.add_argument("--smoke_max_eval_batches", type=int, default=2)
    parser.add_argument("--smoke_epochs", type=int, default=1)
    parser.add_argument("--smoke_patience", type=int, default=1)
    args = parser.parse_args()

    unknown = [recipe for recipe in args.recipes if recipe not in RECIPES]
    if unknown:
        raise ValueError(f"Unknown recipe(s): {unknown}. Known: {sorted(RECIPES)}")
    rows: List[Dict[str, Any]] = []
    commands: List[List[str]] = []
    for recipe_name in args.recipes:
        recipe = RECIPES[recipe_name]
        config_path = ROOT / str(recipe["config"])
        checkpoint_path = ROOT / str(recipe["checkpoint"])
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config for {recipe_name}: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {recipe_name}: {checkpoint_path}")
        cmd = _command(args, recipe_name)
        commands.append(cmd)
        rows.append(
            {
                "recipe": recipe_name,
                "description": recipe["description"],
                "config": recipe["config"],
                "checkpoint": recipe["checkpoint"],
                "run_name": f"{recipe_name}_warnext_stage1_{'smoke' if args.smoke else 'full'}_seed{args.seed}",
                "command": " ".join(cmd),
            }
        )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report_dir / ("smoke_commands.csv" if args.smoke else "stage1_commands.csv"), rows)
    (report_dir / "stage1_manifest.json").write_text(
        json.dumps(
            {
                "created_at": _now(),
                "smoke": bool(args.smoke),
                "run": bool(args.run),
                "recipes": rows,
                "trainable_head_patterns": TRAINABLE_HEAD_PATTERNS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"commands": len(commands), "run": bool(args.run), "smoke": bool(args.smoke)}, indent=2))
    if not args.run:
        for cmd in commands:
            print(" ".join(cmd))
        return
    for cmd in commands:
        print(f"[{_now()}] RUN {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
