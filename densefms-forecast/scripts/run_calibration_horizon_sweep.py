"""Run or print DenseFMS calibration/horizon ablation commands."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


SUMMARY_COLUMNS = [
    "run_name",
    "calibration_seconds",
    "horizon_seconds",
    "recent_window_seconds",
    "loss_mode",
    "use_static",
    "best_epoch",
    "val_MAE",
    "test_MAE",
    "test_RMSE",
    "test_R2",
    "test_sMAPE",
    "derivative_mae_all",
    "derivative_mae_stationary_eps0.5",
    "derivative_mae_moving_eps0.5",
    "trend_macro_f1_2s_eps0.5",
    "trend_macro_f1_5s_eps0.5",
    "high_fms_false_positive_rate",
    "common_test_MAE",
    "common_test_RMSE",
    "common_derivative_mae_all",
    "common_trend_macro_f1_2s_eps0.5",
    "checkpoint_path",
    "metrics_path",
    "plot_dir",
]

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


def tag(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p")


def run_name(prefix: str, condition: str, loss_mode: str, calibration: float, horizon: float, recent: float) -> str:
    loss_tag = "trend" if loss_mode == "level_trend_raw" else "level"
    return f"{prefix}_{condition}_{loss_tag}_calib{tag(calibration)}_h{tag(horizon)}_rw{tag(recent)}"


def common_window_spec(calibrations: Sequence[float], horizons: Sequence[float]) -> Dict[str, float | None]:
    max_calib = max(float(v) for v in calibrations)
    max_horizon = max(float(v) for v in horizons)
    target_start = max_calib + max_horizon if len(set(float(v) for v in horizons)) == 1 else None
    return {
        "current_start": max_calib,
        "max_horizon_seconds": max_horizon,
        "target_start": target_start,
    }


def common_args(calibrations: Sequence[float], horizons: Sequence[float]) -> List[str]:
    spec = common_window_spec(calibrations, horizons)
    args = [
        "--common_eval_current_start",
        f"{float(spec['current_start']):g}",
        "--common_eval_max_horizon_seconds",
        f"{float(spec['max_horizon_seconds']):g}",
    ]
    if spec["target_start"] is not None:
        args.extend(["--common_eval_target_start", f"{float(spec['target_start']):g}"])
    return args


def train_command(args: argparse.Namespace, calibration: float, horizon: float) -> List[str]:
    trend_weight = float(args.trend_weight if args.trend_weight is not None else (0.1 if args.loss_mode == "level_trend_raw" else 0.0))
    name = run_name(args.run_prefix, args.condition, args.loss_mode, calibration, horizon, args.recent_window_seconds)
    config = CONDITION_CONFIGS[(args.condition, args.loss_mode)]
    cmd = [
        sys.executable,
        "-m",
        "src.densefms_forecast.train",
        "--data_dir",
        args.data_dir,
        "--config",
        config,
        "--model",
        "coff_lstm",
        "--loss_mode",
        args.loss_mode,
        "--trend_weight",
        f"{trend_weight:g}",
        "--split_file",
        args.split_file,
        "--run_name",
        name,
        "--calibration_seconds",
        f"{float(calibration):g}",
        "--horizon_seconds",
        f"{float(horizon):g}",
        "--recent_window_seconds",
        f"{float(args.recent_window_seconds):g}",
    ]
    cmd.extend(STATIC_ARGS[args.condition])
    cmd.extend(common_args(args.calibration_seconds, args.horizon_seconds))
    if args.smoke_test:
        cmd.extend(["--epochs", str(args.smoke_epochs), "--limit_sessions", str(args.limit_sessions)])
    return cmd


def eval_command(args: argparse.Namespace, train_cmd: List[str], calibration: float, horizon: float) -> List[str]:
    name = train_cmd[train_cmd.index("--run_name") + 1]
    cmd = [
        sys.executable,
        "-m",
        "src.densefms_forecast.evaluate",
        "--checkpoint",
        str(Path("runs") / name / "best.pt"),
        "--data_dir",
        args.data_dir,
        "--split",
        "test",
        "--split_file",
        args.split_file,
        "--calibration_seconds",
        f"{float(calibration):g}",
        "--horizon_seconds",
        f"{float(horizon):g}",
        "--recent_window_seconds",
        f"{float(args.recent_window_seconds):g}",
    ]
    cmd.extend(common_args(args.calibration_seconds, args.horizon_seconds))
    return cmd


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def metric(metrics: Dict[str, Any], key: str) -> Any:
    return metrics.get(key)


def _tag_value(name: str, prefix: str) -> str:
    match = re.search(rf"(?:^|_){re.escape(prefix)}([0-9]+(?:p[0-9]+)?)(?:_|$)", name)
    return match.group(1).replace("p", ".") if match else ""


def build_summary(run_names: Sequence[str], args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in run_names:
        train_metrics_path = Path("runs") / name / "metrics.json"
        eval_metrics_path = Path("runs") / name / "eval_test" / "metrics.json"
        if not train_metrics_path.exists():
            print(f"WARNING: missing metrics for {name}: {train_metrics_path}")
            continue
        payload = load_json(train_metrics_path)
        metrics_block = payload.get("metrics", payload)
        test = metrics_block.get("test_metrics", {})
        metrics_path = train_metrics_path
        if eval_metrics_path.exists():
            eval_payload = load_json(eval_metrics_path)
            test = eval_payload.get("metrics", test)
            metrics_path = eval_metrics_path
        best_val = metrics_block.get("best_val_metrics", {})
        row = {
            "run_name": name,
            "calibration_seconds": _tag_value(name, "calib"),
            "horizon_seconds": _tag_value(name, "h"),
            "recent_window_seconds": _tag_value(name, "rw"),
            "loss_mode": payload.get("loss", {}).get("mode", args.loss_mode),
            "use_static": bool(payload.get("model_kwargs", {}).get("use_static", args.condition != "no_static")),
            "best_epoch": metrics_block.get("best_epoch"),
            "val_MAE": best_val.get("mae"),
            "test_MAE": metric(test, "mae"),
            "test_RMSE": metric(test, "rmse"),
            "test_R2": metric(test, "r2"),
            "test_sMAPE": metric(test, "smape"),
            "derivative_mae_all": metric(test, "derivative_mae_all"),
            "derivative_mae_stationary_eps0.5": metric(test, "derivative_mae_stationary_eps0.5"),
            "derivative_mae_moving_eps0.5": metric(test, "derivative_mae_moving_eps0.5"),
            "trend_macro_f1_2s_eps0.5": metric(test, "trend_macro_f1_2s_eps0.5"),
            "trend_macro_f1_5s_eps0.5": metric(test, "trend_macro_f1_5s_eps0.5"),
            "high_fms_false_positive_rate": metric(test, "high_fms_false_positive_rate"),
            "common_test_MAE": metric(test, "common_mae"),
            "common_test_RMSE": metric(test, "common_rmse"),
            "common_derivative_mae_all": metric(test, "common_derivative_mae_all"),
            "common_trend_macro_f1_2s_eps0.5": metric(test, "common_trend_macro_f1_2s_eps0.5"),
            "checkpoint_path": str(Path("runs") / name / "best.pt"),
            "metrics_path": str(metrics_path),
            "plot_dir": str(Path("runs") / name / "plots"),
        }
        rows.append(row)
    return rows


def write_summary(rows: Sequence[Dict[str, Any]]) -> None:
    out_csv = Path("runs") / "calibration_horizon_sweep_summary.csv"
    out_md = Path("runs") / "calibration_horizon_sweep_summary.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    md: List[str] = ["# Calibration/Horizon Sweep Summary", ""]
    md.append("| " + " | ".join(SUMMARY_COLUMNS) + " |")
    md.append("| " + " | ".join(["---"] * len(SUMMARY_COLUMNS)) + " |")
    for row in rows:
        md.append("| " + " | ".join(str(row.get(col, "")) for col in SUMMARY_COLUMNS) + " |")
    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Natural metrics use all valid predictions for each run.",
            "- Common-window metrics filter predictions to a shared current-time range based on the largest calibration and horizon values in the sweep.",
            "- Check whether increasing calibration_seconds improves both natural and common MAE before interpreting it as useful; longer calibration changes the available evaluation window.",
            "- Check whether MAE/RMSE rise as horizon_seconds increases, and where that degradation becomes large.",
            "- If longer calibration does not improve common-window metrics, it may indicate limited benefit from the calibration branch or limits of head-only online forecasting in this setup.",
            "- If short horizons are also weak, that should be interpreted descriptively as difficulty under the current real-time input constraints, not as a causal conclusion.",
        ]
    )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Saved summary to {out_csv} and {out_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DenseFMS calibration/horizon ablations.")
    parser.add_argument("--data_dir", default="./DenseFMS/Dataset")
    parser.add_argument("--split_file", default="./artifacts/densefms_split_seed42.json")
    parser.add_argument("--calibration_seconds", nargs="+", type=float, default=[30.0])
    parser.add_argument("--horizon_seconds", nargs="+", type=float, default=[5.0])
    parser.add_argument("--recent_window_seconds", type=float, default=10.0)
    parser.add_argument("--condition", choices=["no_static", "age_gender", "full_static"], default="no_static")
    parser.add_argument("--loss_mode", choices=["level_only", "level_trend_raw"], default="level_only")
    parser.add_argument("--trend_weight", type=float, default=None)
    parser.add_argument("--run_prefix", default="calib_horizon_sweep")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--smoke_epochs", type=int, default=1)
    parser.add_argument("--limit_sessions", type=int, default=8)
    args = parser.parse_args()

    run_names: List[str] = []
    commands: List[List[str]] = []
    for calibration in args.calibration_seconds:
        for horizon in args.horizon_seconds:
            cmd = train_command(args, calibration, horizon)
            commands.append(cmd)
            run_names.append(cmd[cmd.index("--run_name") + 1])
            print("\n[train]")
            print(" ".join(cmd))
            print("[evaluate]")
            print(" ".join(eval_command(args, cmd, calibration, horizon)))

    if args.dry_run:
        return

    for cmd in commands:
        name = cmd[cmd.index("--run_name") + 1]
        checkpoint = Path("runs") / name / "best.pt"
        if args.skip_existing and checkpoint.exists():
            print(f"Skipping existing run {name}")
            continue
        subprocess.run(cmd, check=True)
        if not args.smoke_test:
            calibration = float(cmd[cmd.index("--calibration_seconds") + 1])
            horizon = float(cmd[cmd.index("--horizon_seconds") + 1])
            subprocess.run(eval_command(args, cmd, calibration, horizon), check=True)

    if not args.smoke_test:
        write_summary(build_summary(run_names, args))


if __name__ == "__main__":
    main()
