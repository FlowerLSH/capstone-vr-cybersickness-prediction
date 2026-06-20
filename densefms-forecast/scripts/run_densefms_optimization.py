"""Manifest-driven DenseFMS optimization runner with durable per-run logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


CONFIGS = {
    ("no_static", "level_only"): "configs/coff_lstm_no_static_level.yaml",
    ("no_static", "level_trend_raw"): "configs/coff_lstm_no_static_trend.yaml",
    ("full_static", "level_only"): "configs/coff_lstm_static_full_level.yaml",
    ("full_static", "level_trend_raw"): "configs/coff_lstm_static_full_trend.yaml",
}

STATIC_ARGS = {
    "no_static": [],
    "full_static": ["--use_static", "--static_features", "age", "gender", "mssq"],
}

SUMMARY_COLUMNS = [
    "run_name",
    "status",
    "stage",
    "condition",
    "recent_encoder",
    "calibration_seconds",
    "horizon_seconds",
    "recent_window_seconds",
    "loss_mode",
    "last_completed_epoch",
    "best_epoch_so_far",
    "best_val_MAE_so_far",
    "best_epoch",
    "val_MAE",
    "test_MAE",
    "test_RMSE",
    "common_test_MAE",
    "common_test_RMSE",
    "derivative_mae_all",
    "trend_macro_f1_2s_eps0.5",
    "high_fms_false_positive_rate",
    "checkpoint_path",
    "metrics_path",
    "run_dir",
]

EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+).*?val_mae=(?P<val_mae>[-+0-9.eE]+|nan).*?val_rmse=(?P<val_rmse>[-+0-9.eE]+|nan)",
    re.IGNORECASE,
)


def tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def now() -> float:
    return time.time()


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def common_window_args(runs: Sequence[Mapping[str, Any]]) -> List[str]:
    max_calib = max(float(run["calibration_seconds"]) for run in runs)
    max_horizon = max(float(run["horizon_seconds"]) for run in runs)
    return [
        "--common_eval_current_start",
        f"{max_calib:g}",
        "--common_eval_max_horizon_seconds",
        f"{max_horizon:g}",
    ]


def run_name(run: Mapping[str, Any]) -> str:
    loss_tag = "trend" if run["loss_mode"] == "level_trend_raw" else "level"
    return (
        f"{run['prefix']}_{run['stage']}_{run['recent_encoder']}_{run['condition']}_{loss_tag}_"
        f"calib{tag(run['calibration_seconds'])}_h{tag(run['horizon_seconds'])}_rw{tag(run['recent_window_seconds'])}"
    )


def make_run(
    stage: str,
    prefix: str,
    condition: str,
    recent_encoder: str,
    calibration: float,
    horizon: float,
    recent_window: float,
    loss_mode: str = "level_only",
    trend_weight: Optional[float] = None,
) -> Dict[str, Any]:
    run = {
        "stage": stage,
        "prefix": prefix,
        "condition": condition,
        "recent_encoder": recent_encoder,
        "calibration_seconds": float(calibration),
        "horizon_seconds": float(horizon),
        "recent_window_seconds": float(recent_window),
        "loss_mode": loss_mode,
        "trend_weight": float(trend_weight if trend_weight is not None else (0.1 if loss_mode == "level_trend_raw" else 0.0)),
    }
    run["run_name"] = run_name(run)
    return run


def unique_runs(runs: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for run in runs:
        key = (
            run["condition"],
            run["recent_encoder"],
            float(run["calibration_seconds"]),
            float(run["horizon_seconds"]),
            float(run["recent_window_seconds"]),
            run["loss_mode"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(run))
    return out


def build_plan(args: argparse.Namespace) -> List[Dict[str, Any]]:
    stages = set(args.stages)
    if "all" in stages:
        stages = {"stage1", "stage2", "stage3", "stage4", "stage5"}
    runs: List[Dict[str, Any]] = []
    conditions = args.conditions

    if "stage1" in stages:
        for condition in conditions:
            runs.append(make_run("stage1", args.run_prefix, condition, "tcn", 90, 5, 10))

    if "stage2" in stages:
        for recent_encoder in ("tcn", "transformer"):
            for condition in conditions:
                runs.append(make_run("stage2", args.run_prefix, condition, recent_encoder, 90, 5, 10))

    if "stage3" in stages:
        for recent_encoder in ("tcn", "transformer"):
            for recent_window in (10, 30):
                for condition in conditions:
                    runs.append(make_run("stage3", args.run_prefix, condition, recent_encoder, 90, 5, recent_window))

    if "stage4" in stages:
        for horizon in (0.5, 2.5, 5.0, 10.0):
            for condition in conditions:
                runs.append(make_run("stage4", args.run_prefix, condition, args.best_recent_encoder, 90, horizon, args.best_recent_window))

    if "stage5" in stages:
        for calibration in (90, 120):
            for condition in conditions:
                runs.append(make_run("stage5", args.run_prefix, condition, args.best_recent_encoder, calibration, 5, args.best_recent_window))

    runs = unique_runs(runs)
    if args.recent_encoders:
        allowed = set(args.recent_encoders)
        runs = [run for run in runs if run["recent_encoder"] in allowed]
    if args.max_runs is not None:
        runs = runs[: int(args.max_runs)]
    common_args = common_window_args(runs) if runs else []
    for run in runs:
        run["common_args"] = common_args
    return runs


def train_command(args: argparse.Namespace, run: Mapping[str, Any]) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.densefms_forecast.train",
        "--data_dir",
        args.data_dir,
        "--config",
        CONFIGS[(run["condition"], run["loss_mode"])],
        "--model",
        "coff_lstm",
        "--loss_mode",
        run["loss_mode"],
        "--trend_weight",
        f"{float(run['trend_weight']):g}",
        "--split_file",
        args.split_file,
        "--run_name",
        run["run_name"],
        "--calibration_seconds",
        f"{float(run['calibration_seconds']):g}",
        "--horizon_seconds",
        f"{float(run['horizon_seconds']):g}",
        "--recent_window_seconds",
        f"{float(run['recent_window_seconds']):g}",
        "--recent_encoder",
        run["recent_encoder"],
    ]
    cmd.extend(STATIC_ARGS[run["condition"]])
    cmd.extend(run.get("common_args", []))
    if args.smoke_test:
        cmd.extend(["--epochs", str(args.smoke_epochs), "--limit_sessions", str(args.limit_sessions)])
    return cmd


def eval_command(args: argparse.Namespace, run: Mapping[str, Any]) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.densefms_forecast.evaluate",
        "--checkpoint",
        str(Path(args.runs_dir) / run["run_name"] / "best.pt"),
        "--data_dir",
        args.data_dir,
        "--split",
        "test",
        "--split_file",
        args.split_file,
        "--calibration_seconds",
        f"{float(run['calibration_seconds']):g}",
        "--horizon_seconds",
        f"{float(run['horizon_seconds']):g}",
        "--recent_window_seconds",
        f"{float(run['recent_window_seconds']):g}",
    ]
    cmd.extend(run.get("common_args", []))
    return cmd


def append_event(run_dir: Path, event: Mapping[str, Any]) -> None:
    payload = {"time": now(), **event}
    with open(run_dir / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_status(run_dir: Path, status: Mapping[str, Any]) -> None:
    save_json(run_dir / "status.json", status)


def initial_status(run: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "status": "pending",
        "run_name": run["run_name"],
        "stage": run["stage"],
        "condition": run["condition"],
        "recent_encoder": run["recent_encoder"],
        "last_completed_epoch": 0,
        "best_epoch_so_far": None,
        "best_val_MAE_so_far": None,
        "started_at": None,
        "finished_at": None,
        "returncode": None,
    }


def update_status_from_line(run_dir: Path, status: Dict[str, Any], line: str) -> None:
    match = EPOCH_RE.search(line)
    if not match:
        return
    epoch = int(match.group("epoch"))
    val_mae = float(match.group("val_mae"))
    status["last_completed_epoch"] = max(int(status.get("last_completed_epoch") or 0), epoch)
    best = status.get("best_val_MAE_so_far")
    if best is None or (math.isfinite(val_mae) and val_mae < float(best)):
        status["best_val_MAE_so_far"] = val_mae
        status["best_epoch_so_far"] = epoch
    write_status(run_dir, status)
    append_event(run_dir, {"event": "epoch", "epoch": epoch, "val_mae": val_mae})


def stream_pipe(pipe: Any, log_path: Path, kind: str, run_dir: Path, status: Dict[str, Any], lock: threading.Lock) -> None:
    with open(log_path, "a", encoding="utf-8", buffering=1) as f:
        for line in iter(pipe.readline, ""):
            f.write(line)
            if kind == "stdout":
                with lock:
                    update_status_from_line(run_dir, status, line)
    pipe.close()


def run_process(cmd: Sequence[str], run_dir: Path, status: Dict[str, Any], phase: str) -> int:
    append_event(run_dir, {"event": f"{phase}_started", "command": list(cmd)})
    with open(run_dir / "stdout.log", "a", encoding="utf-8", buffering=1) as out:
        out.write(f"\n===== {phase.upper()} START =====\n{' '.join(cmd)}\n")
    with open(run_dir / "stderr.log", "a", encoding="utf-8", buffering=1) as err:
        err.write(f"\n===== {phase.upper()} START =====\n")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    lock = threading.Lock()
    threads = [
        threading.Thread(target=stream_pipe, args=(proc.stdout, run_dir / "stdout.log", "stdout", run_dir, status, lock), daemon=True),
        threading.Thread(target=stream_pipe, args=(proc.stderr, run_dir / "stderr.log", "stderr", run_dir, status, lock), daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        status["status"] = "interrupted"
        status["finished_at"] = now()
        write_status(run_dir, status)
        append_event(run_dir, {"event": "interrupted", "phase": phase})
        raise
    for thread in threads:
        thread.join(timeout=5)
    append_event(run_dir, {"event": f"{phase}_finished", "returncode": returncode})
    return returncode


def prepare_run_dir(args: argparse.Namespace, run: Mapping[str, Any], train_cmd: Sequence[str], eval_cmd: Sequence[str]) -> Path:
    run_dir = Path(args.runs_dir) / run["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "run_config.json", run)
    save_json(run_dir / "command.json", {"train": list(train_cmd), "evaluate": list(eval_cmd)})
    with open(run_dir / "command.txt", "w", encoding="utf-8") as f:
        f.write("[train]\n" + " ".join(train_cmd) + "\n\n[evaluate]\n" + " ".join(eval_cmd) + "\n")
    with open(run_dir / "git_commit.txt", "w", encoding="utf-8") as f:
        f.write(git_commit() + "\n")
    status = initial_status(run)
    write_status(run_dir, status)
    append_event(run_dir, {"event": "prepared"})
    return run_dir


def run_one(args: argparse.Namespace, run: Mapping[str, Any]) -> None:
    train_cmd = train_command(args, run)
    evaluate_cmd = eval_command(args, run)
    run_dir = prepare_run_dir(args, run, train_cmd, evaluate_cmd)
    status = load_json(run_dir / "status.json")
    checkpoint = run_dir / "best.pt"
    if args.skip_existing and checkpoint.exists() and (run_dir / "metrics.json").exists():
        status["status"] = "completed"
        write_status(run_dir, status)
        append_event(run_dir, {"event": "skipped_existing"})
        return
    status.update({"status": "running", "started_at": now(), "finished_at": None, "returncode": None})
    write_status(run_dir, status)
    train_rc = run_process(train_cmd, run_dir, status, "train")
    if train_rc != 0:
        status.update({"status": "failed", "finished_at": now(), "returncode": train_rc})
        write_status(run_dir, status)
        return
    eval_rc = 0
    if not args.smoke_test:
        eval_rc = run_process(evaluate_cmd, run_dir, status, "evaluate")
    status.update(
        {
            "status": "completed" if eval_rc == 0 else "failed",
            "finished_at": now(),
            "returncode": eval_rc,
        }
    )
    write_status(run_dir, status)


def metric(metrics: Mapping[str, Any], key: str) -> Any:
    return metrics.get(key)


def summarize_run(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    status = load_json(run_dir / "status.json") if (run_dir / "status.json").exists() else {}
    run_cfg = load_json(run_dir / "run_config.json") if (run_dir / "run_config.json").exists() else {}
    metrics_path = run_dir / "eval_test" / "metrics.json"
    train_metrics_path = run_dir / "metrics.json"
    best_val: Mapping[str, Any] = {}
    test: Mapping[str, Any] = {}
    best_epoch = None
    if train_metrics_path.exists():
        payload = load_json(train_metrics_path)
        block = payload.get("metrics", payload)
        best_val = block.get("best_val_metrics", {})
        test = block.get("test_metrics", {})
        best_epoch = block.get("best_epoch")
        metrics_path = train_metrics_path
    if (run_dir / "eval_test" / "metrics.json").exists():
        payload = load_json(run_dir / "eval_test" / "metrics.json")
        test = payload.get("metrics", test)
        metrics_path = run_dir / "eval_test" / "metrics.json"
    return {
        "run_name": run_dir.name,
        "status": status.get("status", "unknown"),
        "stage": run_cfg.get("stage"),
        "condition": run_cfg.get("condition"),
        "recent_encoder": run_cfg.get("recent_encoder"),
        "calibration_seconds": run_cfg.get("calibration_seconds"),
        "horizon_seconds": run_cfg.get("horizon_seconds"),
        "recent_window_seconds": run_cfg.get("recent_window_seconds"),
        "loss_mode": run_cfg.get("loss_mode"),
        "last_completed_epoch": status.get("last_completed_epoch"),
        "best_epoch_so_far": status.get("best_epoch_so_far"),
        "best_val_MAE_so_far": status.get("best_val_MAE_so_far"),
        "best_epoch": best_epoch,
        "val_MAE": best_val.get("mae"),
        "test_MAE": metric(test, "mae"),
        "test_RMSE": metric(test, "rmse"),
        "common_test_MAE": metric(test, "common_mae"),
        "common_test_RMSE": metric(test, "common_rmse"),
        "derivative_mae_all": metric(test, "derivative_mae_all"),
        "trend_macro_f1_2s_eps0.5": metric(test, "trend_macro_f1_2s_eps0.5"),
        "high_fms_false_positive_rate": metric(test, "high_fms_false_positive_rate"),
        "checkpoint_path": str(run_dir / "best.pt") if (run_dir / "best.pt").exists() else "",
        "metrics_path": str(metrics_path) if metrics_path.exists() else "",
        "run_dir": str(run_dir),
    }


def write_summary(rows: Sequence[Mapping[str, Any]], runs_dir: str | Path) -> None:
    runs_dir = Path(runs_dir)
    csv_path = runs_dir / "performance_optimization_summary.csv"
    md_path = runs_dir / "performance_optimization_summary.md"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# DenseFMS Performance Optimization Summary", ""]
    lines.append("| " + " | ".join(SUMMARY_COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(SUMMARY_COLUMNS)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in SUMMARY_COLUMNS) + " |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Model selection should use validation MAE.",
            "- Test/common-window metrics are for final reporting and analysis.",
            "- Rows with `running`, `failed`, or `interrupted` status may contain partial epoch information only.",
        ]
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved summary to {csv_path} and {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DenseFMS adaptive performance optimization experiments.")
    parser.add_argument("--data_dir", default="./DenseFMS/Dataset")
    parser.add_argument("--split_file", default="./artifacts/densefms_split_seed42.json")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--run_prefix", default="opt")
    parser.add_argument("--stages", nargs="+", default=["stage1", "stage2"], choices=["stage1", "stage2", "stage3", "stage4", "stage5", "all"])
    parser.add_argument("--conditions", nargs="+", default=["no_static", "full_static"], choices=["no_static", "full_static"])
    parser.add_argument("--recent_encoders", nargs="+", default=None, choices=["tcn", "transformer"])
    parser.add_argument("--best_recent_encoder", default="transformer", choices=["tcn", "transformer"])
    parser.add_argument("--best_recent_window", type=float, default=10.0)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke_epochs", type=int, default=1)
    parser.add_argument("--limit_sessions", type=int, default=8)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--max_runs", type=int, default=None)
    args = parser.parse_args()

    if args.summary_only:
        rows = [summarize_run(path) for path in Path(args.runs_dir).glob(f"{args.run_prefix}_*") if path.is_dir()]
        write_summary(rows, args.runs_dir)
        return

    runs = build_plan(args)
    manifest_path = Path(args.runs_dir) / "performance_optimization_manifest.json"
    save_json(manifest_path, {"created_at": now(), "git_commit": git_commit(), "runs": runs})
    print(f"Planned {len(runs)} runs. Manifest: {manifest_path}")
    for run in runs:
        train_cmd = train_command(args, run)
        eval_cmd = eval_command(args, run)
        print(f"\n[{run['run_name']}]")
        print("train:", " ".join(train_cmd))
        print("eval: ", " ".join(eval_cmd))

    if args.dry_run:
        return

    try:
        for run in runs:
            run_one(args, run)
            rows = [summarize_run(Path(args.runs_dir) / planned["run_name"]) for planned in runs]
            write_summary(rows, args.runs_dir)
    except KeyboardInterrupt:
        rows = [summarize_run(Path(args.runs_dir) / planned["run_name"]) for planned in runs]
        write_summary(rows, args.runs_dir)
        raise


if __name__ == "__main__":
    main()
