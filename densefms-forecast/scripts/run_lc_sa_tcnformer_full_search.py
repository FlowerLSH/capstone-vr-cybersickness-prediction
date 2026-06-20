#!/usr/bin/env python
"""Bounded validation-based LC-SA-TCNFormer full-search runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.densefms_forecast.data import apply_saved_split, load_raw_sessions
from src.densefms_forecast.utils import compute_regression_metrics, load_json, save_json, seconds_to_steps


LEADERBOARD_COLUMNS = [
    "rank",
    "run_name",
    "stage",
    "calibration_seconds",
    "recent_window_seconds",
    "horizon_seconds",
    "fms_context_mode",
    "anchor_mode",
    "anchor_interval_seconds",
    "use_static",
    "predict_delta_from_anchor",
    "d_model",
    "transformer_layers",
    "pooling",
    "recent_dilations",
    "recent_rf_seconds",
    "val_MAE",
    "val_RMSE",
    "val_R2",
    "val_sMAPE",
    "common_val_MAE",
    "common_val_RMSE",
    "best_epoch",
    "parameter_count",
    "checkpoint_path",
    "metrics_path",
    "prediction_csv_path",
]

FMS_CONTEXT_MODES = {"none", "start_only", "calibration_history", "sparse_anchor"}


def now() -> float:
    return time.time()


def tag(value: Any) -> str:
    return f"{float(value):g}".replace(".", "p")


def git_status() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
    except Exception as exc:
        return f"git status unavailable: {exc}"


def run_key(spec: Mapping[str, Any]) -> tuple:
    return (
        float(spec["calibration_seconds"]),
        float(spec["recent_window_seconds"]),
        float(spec["horizon_seconds"]),
        str(spec.get("fms_context_mode", "start_only")),
        spec["anchor_mode"],
        float(spec.get("anchor_interval_seconds", 60.0)),
        bool(spec.get("use_static", False)),
        bool(spec.get("predict_delta_from_anchor", False)),
        int(spec.get("d_model", 64)),
        int(spec.get("transformer_layers", 1)),
        str(spec.get("pooling", "mean")),
        str(spec.get("loss_mode", "level_only")),
    )


def make_run_name(spec: Mapping[str, Any]) -> str:
    static = "static" if spec.get("use_static") else "no_static"
    delta = "_delta" if spec.get("predict_delta_from_anchor") else ""
    loss_mode = str(spec.get("loss_mode", "level_only"))
    loss = "" if loss_mode == "level_only" else f"_{loss_mode}"
    context = f"_fms{str(spec.get('fms_context_mode', 'start_only')).replace('_', '')}"
    interval = ""
    if spec["anchor_mode"] == "sparse_observed":
        interval = f"_ai{tag(spec.get('anchor_interval_seconds', 60.0))}"
    return (
        f"{spec['stage']}_c{tag(spec['calibration_seconds'])}_w{tag(spec['recent_window_seconds'])}_"
        f"h{tag(spec['horizon_seconds'])}{context}_{spec['anchor_mode']}{interval}_{static}_"
        f"d{spec.get('d_model', 64)}_l{spec.get('transformer_layers', 1)}_{spec.get('pooling', 'mean')}{delta}{loss}"
    )


def spec(
    stage: str,
    calibration: float,
    recent: float,
    horizon: float,
    anchor: str,
    use_static: bool = False,
    anchor_interval: float = 60.0,
    d_model: int = 64,
    transformer_layers: int = 1,
    pooling: str = "mean",
    predict_delta: bool = False,
    loss_mode: str = "level_only",
    fms_context_mode: Optional[str] = None,
) -> Dict[str, Any]:
    mode = str(fms_context_mode or ("sparse_anchor" if anchor == "sparse_observed" else "none" if anchor == "none" else "start_only"))
    if mode not in FMS_CONTEXT_MODES:
        raise ValueError(f"fms_context_mode must be one of {sorted(FMS_CONTEXT_MODES)}, got {mode!r}")
    if mode == "sparse_anchor":
        anchor = "sparse_observed"
        anchor_interval = float(anchor_interval or 60.0)
    else:
        anchor = "none"
        anchor_interval = 0.0
        predict_delta = False
    use_static = False
    item = {
        "stage": stage,
        "model": "lc_sa_tcnformer",
        "calibration_seconds": float(calibration),
        "recent_window_seconds": float(recent),
        "horizon_seconds": float(horizon),
        "fms_context_mode": mode,
        "anchor_mode": anchor,
        "anchor_interval_seconds": float(anchor_interval),
        "use_static": bool(use_static),
        "predict_delta_from_anchor": bool(predict_delta),
        "d_model": int(d_model),
        "kernel_size": 3,
        "dropout": 0.1,
        "transformer_layers": int(transformer_layers),
        "transformer_heads": 4,
        "transformer_ff_dim": 128,
        "pooling": pooling,
        "loss_type": "smooth_l1",
        "loss_mode": loss_mode,
    }
    item["run_name"] = make_run_name(item)
    return item


def complete_spec_item(raw: Mapping[str, Any]) -> Dict[str, Any]:
    item = dict(raw)
    mode = str(item.get("fms_context_mode", "start_only"))
    if mode not in FMS_CONTEXT_MODES:
        raise ValueError(f"fms_context_mode must be one of {sorted(FMS_CONTEXT_MODES)}, got {mode!r}")
    item["fms_context_mode"] = mode
    if mode == "sparse_anchor":
        item["anchor_mode"] = "sparse_observed"
        item["anchor_interval_seconds"] = float(item.get("anchor_interval_seconds") or 60.0)
        item["predict_delta_from_anchor"] = bool(item.get("predict_delta_from_anchor", True))
    else:
        item["anchor_mode"] = "none"
        item["anchor_interval_seconds"] = 0.0
        item["predict_delta_from_anchor"] = False
    item["use_static"] = False
    item["run_name"] = make_run_name(item)
    return item


def unique_specs(items: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for raw in items:
        item = complete_spec_item(raw)
        key = run_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def build_stage1(max_runs: Optional[int] = None) -> List[Dict[str, Any]]:
    runs = [
        spec("stage1", 90, 30, 5, "calibration_end"),
        spec("stage1", 90, 30, 10, "calibration_end"),
        spec("stage1", 90, 30, 15, "calibration_end"),
        spec("stage1", 60, 30, 10, "calibration_end"),
        spec("stage1", 90, 30, 10, "calibration_end"),
        spec("stage1", 120, 30, 10, "calibration_end"),
        spec("stage1", 90, 10, 10, "calibration_end"),
        spec("stage1", 90, 60, 10, "calibration_end"),
        spec("stage1", 90, 30, 10, "none"),
        spec("stage1", 90, 30, 10, "sparse_observed", anchor_interval=60),
        spec("stage1", 90, 30, 10, "calibration_end", use_static=True),
        spec("stage1", 90, 30, 10, "sparse_observed", use_static=True, anchor_interval=60),
        spec("stage1", 90, 30, 20, "calibration_end"),
        spec("stage1", 90, 30, 30, "calibration_end"),
        spec("stage1", 120, 60, 10, "calibration_end", use_static=True),
        spec("stage1", 120, 30, 15, "sparse_observed", use_static=True, anchor_interval=60),
        spec("stage1", 90, 60, 15, "sparse_observed", use_static=True, anchor_interval=60),
    ]
    runs = unique_specs(runs)
    return runs[: int(max_runs)] if max_runs is not None else runs


def build_stage2(top_rows: Sequence[Mapping[str, Any]], max_runs: int) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for row in top_rows[:3]:
        base = row["spec"]
        runs.append({**base, "stage": "stage2", "pooling": "attention"})
        runs.append({**base, "stage": "stage2", "transformer_layers": 2})
        runs.append({**base, "stage": "stage2", "d_model": 128})
        if base.get("anchor_mode") != "none":
            runs.append({**base, "stage": "stage2", "predict_delta_from_anchor": True})
    fixed = []
    for item in runs:
        item = dict(item)
        item["run_name"] = make_run_name(item)
        fixed.append(item)
    return unique_specs(fixed)[: int(max_runs)]


def build_adaptive(best: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], max_runs: int) -> List[Dict[str, Any]]:
    base = dict(best["spec"])
    base["stage"] = "adaptive"
    candidates: List[Dict[str, Any]] = []
    if base.get("anchor_mode") != "none":
        candidates.append({**base, "predict_delta_from_anchor": True})
    candidates.append({**base, "loss_mode": "level_plus_trend"})
    candidates.append({**base, "pooling": "attention"})
    candidates.append({**base, "transformer_layers": 2})
    for interval in (30.0, 90.0, 120.0):
        candidates.append({**base, "fms_context_mode": "sparse_anchor", "anchor_mode": "sparse_observed", "anchor_interval_seconds": interval})
    for horizon in (1.0, 2.5, 5.0):
        candidates.append({**base, "horizon_seconds": horizon})
    fixed = []
    for item in candidates:
        if item.get("predict_delta_from_anchor") and item.get("anchor_mode") == "none":
            continue
        item = dict(item)
        item["run_name"] = make_run_name(item)
        fixed.append(item)
    return unique_specs(fixed)[: int(max_runs)]


def common_args(runs: Sequence[Mapping[str, Any]]) -> List[str]:
    max_calib = max(float(run["calibration_seconds"]) for run in runs)
    max_horizon = max(float(run["horizon_seconds"]) for run in runs)
    return [
        "--common_eval_current_start",
        f"{max_calib:g}",
        "--common_eval_max_horizon_seconds",
        f"{max_horizon:g}",
    ]


def train_cmd(args: argparse.Namespace, item: Mapping[str, Any], epochs: int, patience: int, all_runs: Sequence[Mapping[str, Any]]) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.densefms_forecast.train",
        "--data_dir",
        args.data_dir,
        "--config",
        args.config,
        "--runs_dir",
        args.output_dir,
        "--model",
        "lc_sa_tcnformer",
        "--run_name",
        item["run_name"],
        "--split_file",
        args.split_file,
        "--seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        f"{args.learning_rate:g}",
        "--weight_decay",
        f"{args.weight_decay:g}",
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--num_workers",
        str(args.num_workers),
        "--loss_type",
        item.get("loss_type", "smooth_l1"),
        "--loss_mode",
        item.get("loss_mode", "level_only"),
        "--high_fms_threshold",
        "10.0",
        "--calibration_seconds",
        f"{float(item['calibration_seconds']):g}",
        "--recent_window_seconds",
        f"{float(item['recent_window_seconds']):g}",
        "--horizon_seconds",
        f"{float(item['horizon_seconds']):g}",
        "--anchor_mode",
        item["anchor_mode"],
        "--anchor_interval_seconds",
        f"{float(item.get('anchor_interval_seconds', 60.0)):g}",
        "--fms_context_mode",
        str(item.get("fms_context_mode", "start_only")),
        "--d_model",
        str(int(item.get("d_model", 64))),
        "--kernel_size",
        str(int(item.get("kernel_size", 3))),
        "--dropout",
        f"{float(item.get('dropout', 0.1)):g}",
        "--transformer_layers",
        str(int(item.get("transformer_layers", 1))),
        "--transformer_heads",
        str(int(item.get("transformer_heads", 4))),
        "--transformer_ff_dim",
        str(int(item.get("transformer_ff_dim", 128))),
        "--pooling",
        str(item.get("pooling", "mean")),
        "--no_test_eval",
        "--no-save_plots",
    ]
    if item.get("use_static"):
        cmd.extend(["--use_static", "--static_features", "age", "gender", "mssq"])
    else:
        cmd.append("--no_static")
    if item.get("predict_delta_from_anchor"):
        cmd.append("--predict_delta_from_anchor")
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.device:
        cmd.extend(["--device", args.device])
    cmd.extend(common_args(all_runs))
    return cmd


def eval_cmd(
    args: argparse.Namespace,
    item: Mapping[str, Any],
    split: str = "test",
    common_runs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "src.densefms_forecast.evaluate",
        "--checkpoint",
        str(Path(args.output_dir) / item["run_name"] / "best.pt"),
        "--data_dir",
        args.data_dir,
        "--split",
        split,
        "--split_file",
        args.split_file,
        "--batch_size",
        str(args.batch_size),
        "--calibration_seconds",
        f"{float(item['calibration_seconds']):g}",
        "--recent_window_seconds",
        f"{float(item['recent_window_seconds']):g}",
        "--horizon_seconds",
        f"{float(item['horizon_seconds']):g}",
        *common_args(common_runs or [item]),
    ]


def write_command_files(run_dir: Path, train: Sequence[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "command.json", {"train": list(train)})
    (run_dir / "command.txt").write_text(" ".join(train) + "\n", encoding="utf-8")


def run_process(cmd: Sequence[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(cmd), flush=True)
    with open(log_path, "a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n===== START {time.ctime()} =====\n{' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        rc = proc.wait()
        log.write(f"===== END rc={rc} {time.ctime()} =====\n")
    return int(rc)


def run_train(args: argparse.Namespace, item: Mapping[str, Any], epochs: int, patience: int, all_runs: Sequence[Mapping[str, Any]]) -> None:
    run_dir = Path(args.output_dir) / item["run_name"]
    metrics = run_dir / "metrics.json"
    checkpoint = run_dir / "best.pt"
    if args.skip_existing and metrics.exists() and checkpoint.exists():
        print(f"skip existing: {run_dir}")
        return
    cmd = train_cmd(args, item, epochs, patience, all_runs)
    write_command_files(run_dir, cmd)
    save_json(run_dir / "run_spec.json", item)
    rc = run_process(cmd, ROOT, run_dir / "train.log")
    save_json(run_dir / "status.json", {"status": "completed" if rc == 0 else "failed", "returncode": rc, "finished_at": now()})
    if rc != 0:
        raise RuntimeError(f"Training failed for {item['run_name']} with rc={rc}")


def summarize_run(args: argparse.Namespace, item: Mapping[str, Any]) -> Dict[str, Any]:
    run_dir = Path(args.output_dir) / item["run_name"]
    metrics_path = run_dir / "metrics.json"
    row: Dict[str, Any] = {**item, "status": "missing", "spec": dict(item)}
    row.update(
        {
            "checkpoint_path": str(run_dir / "best.pt"),
            "metrics_path": str(metrics_path),
            "prediction_csv_path": str(run_dir / "val_predictions.csv"),
        }
    )
    if not metrics_path.exists():
        return row
    payload = load_json(metrics_path)
    block = payload.get("metrics", {})
    best = block.get("best_val_metrics", {})
    row.update(
        {
            "status": "completed",
            "val_MAE": best.get("mae"),
            "val_RMSE": best.get("rmse"),
            "val_R2": best.get("r2"),
            "val_sMAPE": best.get("smape"),
            "common_val_MAE": best.get("common_mae"),
            "common_val_RMSE": best.get("common_rmse"),
            "best_epoch": block.get("best_epoch"),
            "parameter_count": block.get("parameter_count"),
            "recent_rf_seconds": block.get("recent_rf_seconds"),
            "recent_dilations": payload.get("metrics", {}).get("recent_dilations", item.get("recent_dilations", "auto")),
        }
    )
    return row


def completed_sorted(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        try:
            mae = float(row.get("val_MAE"))
        except Exception:
            continue
        if math.isfinite(mae):
            out.append(dict(row))
    return sorted(out, key=lambda r: (float(r["val_MAE"]), float(r.get("val_RMSE", math.inf) or math.inf)))


def write_leaderboard(rows: Sequence[Mapping[str, Any]], output_dir: str | Path) -> List[Dict[str, Any]]:
    ranked = completed_sorted(rows)
    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(ranked, 1):
        out = {col: row.get(col, "") for col in LEADERBOARD_COLUMNS}
        out["rank"] = idx
        out_rows.append(out)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "leaderboard_val.csv"
    md_path = out_dir / "leaderboard_val.md"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)
    lines = ["# LC-SA-TCNFormer Validation Leaderboard", ""]
    lines.append("| " + " | ".join(LEADERBOARD_COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(LEADERBOARD_COLUMNS)) + " |")
    for row in out_rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in LEADERBOARD_COLUMNS) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved leaderboard: {csv_path}")
    return ranked


def inspect_hardware(output_dir: str | Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for name, cmd in {
        "nvidia_smi": ["nvidia-smi"],
        "free_h": ["free", "-h"],
        "df_h": ["df", "-h", "."],
    }.items():
        try:
            payload[name] = subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=20)
        except Exception as exc:
            payload[name] = f"unavailable: {exc}"
    payload["git_status_short"] = git_status()
    save_json(Path(output_dir) / "hardware_summary.json", payload)
    return payload


def run_sanity_tests(args: argparse.Namespace) -> None:
    cmd = [sys.executable, "scripts/run_densefms_sanity_tests.py"]
    if args.dry_run:
        print("dry-run sanity:", " ".join(cmd))
        return
    rc = run_process(cmd, ROOT, Path(args.output_dir) / "sanity_tests.log")
    if rc != 0:
        raise RuntimeError("Sanity tests failed; stopping before full training.")


def baseline_samples(
    sessions: Sequence[Any],
    calibration_steps: int,
    recent_steps: int,
    horizon_steps: int,
    anchor_mode: str,
    anchor_interval_steps: int,
    cap: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    targets: List[float] = []
    global_features: List[List[float]] = []
    calib_anchor: List[float] = []
    sparse_anchor: List[float] = []
    for sess in sessions:
        start = max(calibration_steps, recent_steps - 1)
        end = sess.length - horizon_steps
        for t in range(start, end):
            target_idx = t + horizon_steps
            y = float(sess.fms[target_idx])
            if not np.isfinite(y):
                continue
            calib = sess.fms[:calibration_steps].astype(np.float64)
            recent = sess.head[t - recent_steps + 1 : t + 1].astype(np.float64)
            slope = float((calib[-1] - calib[0]) / max(calibration_steps - 1, 1))
            feat = [float(np.nanmean(calib)), float(calib[-1]), slope]
            feat.extend(np.nanmean(recent, axis=0).tolist())
            feat.extend(np.nanstd(recent, axis=0).tolist())
            feat.append(float(horizon_steps))
            targets.append(y)
            global_features.append(feat)
            calib_anchor.append(float(sess.fms[calibration_steps - 1]))
            idx_sparse = max(calibration_steps - 1, (t // anchor_interval_steps) * anchor_interval_steps)
            idx_sparse = min(idx_sparse, t)
            sparse_anchor.append(float(sess.fms[idx_sparse]))
            if cap is not None and len(targets) >= cap:
                return {
                    "y": np.asarray(targets, dtype=np.float64),
                    "x": np.asarray(global_features, dtype=np.float64),
                    "calibration_end": np.asarray(calib_anchor, dtype=np.float64),
                    "sparse_observed": np.asarray(sparse_anchor, dtype=np.float64),
                }
    return {
        "y": np.asarray(targets, dtype=np.float64),
        "x": np.asarray(global_features, dtype=np.float64),
        "calibration_end": np.asarray(calib_anchor, dtype=np.float64),
        "sparse_observed": np.asarray(sparse_anchor, dtype=np.float64),
    }


def compute_baselines(args: argparse.Namespace, row: Mapping[str, Any]) -> Dict[str, Any]:
    item = row["spec"]
    metrics_payload = load_json(Path(row["metrics_path"]))
    mapping = metrics_payload.get("inferred_columns")
    config = metrics_payload.get("data_info", {})
    sampling = float(config.get("sampling_interval", 0.5))
    c_steps = seconds_to_steps(float(item["calibration_seconds"]), sampling, name="calibration_seconds", warn=False)
    w_steps = seconds_to_steps(float(item["recent_window_seconds"]), sampling, name="recent_window_seconds", warn=False)
    h_steps = seconds_to_steps(float(item["horizon_seconds"]), sampling, name="horizon_seconds", warn=False)
    anchor_seconds = float(item.get("anchor_interval_seconds") or 60.0)
    anchor_interval_steps = seconds_to_steps(anchor_seconds if anchor_seconds > 0 else 60.0, sampling, name="anchor_interval_seconds", warn=False)
    raw_sessions, _, _ = load_raw_sessions(
        args.data_dir,
        mapping=mapping,
        calibration_seconds=float(item["calibration_seconds"]),
        horizon_seconds=float(item["horizon_seconds"]),
        default_sampling_interval=sampling,
    )
    split_info = load_json(args.split_file) if Path(args.split_file).exists() else metrics_payload.get("split_info")
    split = apply_saved_split(raw_sessions, split_info)
    train = baseline_samples(split["train"], c_steps, w_steps, h_steps, item["anchor_mode"], anchor_interval_steps, cap=20000)
    val = baseline_samples(split["val"], c_steps, w_steps, h_steps, item["anchor_mode"], anchor_interval_steps)
    y_val = val["y"]
    train_mean = float(np.nanmean(train["y"])) if train["y"].size else float("nan")
    out: Dict[str, Any] = {
        "global_train_mean": compute_regression_metrics(y_val, np.full_like(y_val, train_mean)),
        "calibration_end_anchor": compute_regression_metrics(y_val, val["calibration_end"]),
        "sparse_observed_anchor": compute_regression_metrics(y_val, val["sparse_observed"]),
    }
    if train["x"].size and val["x"].size:
        x_train = train["x"]
        x_val = val["x"]
        mean = np.nanmean(x_train, axis=0)
        std = np.nanstd(x_train, axis=0)
        std[std < 1e-8] = 1.0
        xt = np.nan_to_num((x_train - mean) / std)
        xv = np.nan_to_num((x_val - mean) / std)
        xt = np.concatenate([np.ones((xt.shape[0], 1)), xt], axis=1)
        xv = np.concatenate([np.ones((xv.shape[0], 1)), xv], axis=1)
        alpha = 1.0
        eye = np.eye(xt.shape[1])
        eye[0, 0] = 0.0
        coef = np.linalg.solve(xt.T @ xt + alpha * eye, xt.T @ train["y"])
        out["ridge_small_feature"] = compute_regression_metrics(y_val, xv @ coef)
    maes = {name: vals.get("mae", math.inf) for name, vals in out.items()}
    out["strongest_baseline"] = min(maes, key=lambda key: float(maes[key]))
    out["strongest_baseline_mae"] = float(maes[out["strongest_baseline"]])
    return out


def diagnose_mediocre(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]], output_dir: str | Path) -> Dict[str, Any]:
    ranked = completed_sorted(rows)
    if not ranked:
        return {"triggered": True, "reasons": ["No completed validation runs."], "baselines": {}}
    best = ranked[0]
    baselines = compute_baselines(args, best)
    pred_path = Path(best["prediction_csv_path"])
    pred_df = pd.read_csv(pred_path) if pred_path.exists() and pred_path.stat().st_size else pd.DataFrame()
    pred_var = float(pred_df["predicted_fms"].var()) if not pred_df.empty else float("nan")
    target_var = float(pred_df["target_fms"].var()) if not pred_df.empty else float("nan")
    reasons: List[str] = []
    best_mae = float(best["val_MAE"])
    strongest = float(baselines.get("strongest_baseline_mae", math.inf))
    if math.isfinite(strongest) and best_mae > 0.90 * strongest:
        reasons.append(f"Best val MAE improvement over strongest baseline is <10% ({best_mae:.4f} vs {strongest:.4f}).")
    if float(best.get("val_R2", math.nan)) <= 0.10:
        reasons.append(f"Best val R2 <= 0.10 ({float(best.get('val_R2', math.nan)):.4f}).")
    if math.isfinite(pred_var) and math.isfinite(target_var) and target_var > 0 and pred_var < 0.25 * target_var:
        reasons.append(f"Prediction variance is <25% of target variance ({pred_var:.4f} vs {target_var:.4f}).")
    no_anchor = [row for row in ranked if row.get("anchor_mode") == "none"]
    anchored = [row for row in ranked if row.get("anchor_mode") in {"calibration_end", "sparse_observed"}]
    if no_anchor and anchored and float(no_anchor[0]["val_MAE"]) > 1.15 * float(anchored[0]["val_MAE"]):
        reasons.append("Best no-anchor model is much worse than anchor-based models.")
    diagnosis = {
        "triggered": bool(reasons),
        "reasons": reasons,
        "best_run": best["run_name"],
        "best_val_MAE": best.get("val_MAE"),
        "best_val_R2": best.get("val_R2"),
        "prediction_variance": pred_var,
        "target_variance": target_var,
        "baselines": baselines,
    }
    lines = [
        "# Adaptive Diagnosis",
        "",
        f"- triggered: {diagnosis['triggered']}",
        f"- best_run: {diagnosis['best_run']}",
        f"- best_val_MAE: {diagnosis['best_val_MAE']}",
        f"- best_val_R2: {diagnosis['best_val_R2']}",
        f"- prediction_variance: {pred_var}",
        f"- target_variance: {target_var}",
        "",
        "## Reasons",
        *(f"- {reason}" for reason in reasons),
        "",
        "## Baselines",
    ]
    for name, vals in baselines.items():
        if isinstance(vals, Mapping):
            lines.append(f"- {name}: MAE={vals.get('mae')} RMSE={vals.get('rmse')} R2={vals.get('r2')}")
        else:
            lines.append(f"- {name}: {vals}")
    Path(output_dir, "adaptive_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(Path(output_dir) / "adaptive_diagnosis.json", diagnosis)
    return diagnosis


def make_final_spec(selected: Mapping[str, Any]) -> Dict[str, Any]:
    item = dict(selected["spec"])
    item["stage"] = "final"
    item["run_name"] = "final_" + make_run_name(item)
    return item


def run_final_evaluation(args: argparse.Namespace, best: Mapping[str, Any], common_runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if args.no_test_eval:
        return {}
    for split in ("val", "test"):
        rc = run_process(
            eval_cmd(args, best["spec"], split=split, common_runs=common_runs),
            ROOT,
            Path(args.output_dir) / best["run_name"] / f"eval_{split}.log",
        )
        if rc != 0:
            raise RuntimeError(f"Final {split} evaluation failed for {best['run_name']}")
    test_metrics_path = Path(args.output_dir) / best["run_name"] / "eval_test" / "metrics.json"
    payload = load_json(test_metrics_path) if test_metrics_path.exists() else {}
    metrics = payload.get("metrics", {})
    csv_path = Path(args.output_dir) / "final_test_metrics.csv"
    fieldnames = ["run_name", "test_MAE", "test_RMSE", "test_R2", "test_sMAPE", "common_test_MAE", "common_test_RMSE"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "run_name": best["run_name"],
                "test_MAE": metrics.get("mae"),
                "test_RMSE": metrics.get("rmse"),
                "test_R2": metrics.get("r2"),
                "test_sMAPE": metrics.get("smape"),
                "common_test_MAE": metrics.get("common_mae"),
                "common_test_RMSE": metrics.get("common_rmse"),
            }
        )
    return metrics


def generate_final_plots(args: argparse.Namespace, best: Mapping[str, Any], ranked: Sequence[Mapping[str, Any]]) -> List[str]:
    out: List[str] = []
    plot_dir = Path(args.output_dir) / best["run_name"] / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if ranked:
        top = list(ranked[: min(15, len(ranked))])
        plt.figure(figsize=(10, 5))
        plt.barh([row["run_name"] for row in reversed(top)], [float(row["val_MAE"]) for row in reversed(top)])
        plt.xlabel("Validation MAE")
        plt.tight_layout()
        path = plot_dir / "validation_leaderboard_bar.png"
        plt.savefig(path, dpi=150)
        plt.close()
        out.append(str(path))
    for split in ("val", "test"):
        pred_path = Path(args.output_dir) / best["run_name"] / f"eval_{split}" / f"{split}_predictions.csv"
        if not pred_path.exists():
            pred_path = Path(args.output_dir) / best["run_name"] / f"{split}_predictions.csv"
        if not pred_path.exists() or pred_path.stat().st_size == 0:
            continue
        df = pd.read_csv(pred_path)
        if df.empty:
            continue
        plt.figure(figsize=(5, 5))
        plt.scatter(df["target_fms"], df["predicted_fms"], s=5, alpha=0.35)
        lo = float(min(df["target_fms"].min(), df["predicted_fms"].min()))
        hi = float(max(df["target_fms"].max(), df["predicted_fms"].max()))
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        plt.xlabel("Target FMS")
        plt.ylabel("Predicted FMS")
        plt.tight_layout()
        path = plot_dir / f"{split}_predicted_vs_target.png"
        plt.savefig(path, dpi=150)
        plt.close()
        out.append(str(path))

        plt.figure(figsize=(7, 4))
        residual = df["predicted_fms"] - df["target_fms"]
        plt.hist(residual, bins=40)
        plt.xlabel("Prediction residual")
        plt.ylabel("Count")
        plt.tight_layout()
        path = plot_dir / f"{split}_residual_histogram.png"
        plt.savefig(path, dpi=150)
        plt.close()
        out.append(str(path))
    return out


def write_adaptive_report(output_dir: str | Path, diagnosis: Mapping[str, Any], adaptive_rows: Sequence[Mapping[str, Any]], final_metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Adaptive Report",
        "",
        "## Why Triggered",
        *(f"- {reason}" for reason in diagnosis.get("reasons", [])),
        "",
        "## Adaptive Budget Used",
        f"- runs: {len(adaptive_rows)}",
        "",
        "## Adaptive Validation Results",
    ]
    for row in completed_sorted(adaptive_rows):
        lines.append(f"- {row['run_name']}: val_MAE={row.get('val_MAE')} val_R2={row.get('val_R2')}")
    lines.extend(
        [
            "",
            "## Final Test Metrics After Validation Selection",
            f"- test_MAE: {final_metrics.get('mae')}",
            f"- test_RMSE: {final_metrics.get('rmse')}",
            f"- test_R2: {final_metrics.get('r2')}",
            "",
            "## Remaining Limitations",
            "- Head/motion-only forecasting may be insufficient at longer horizons.",
            "- Sparse FMS anchors are reported only as anchor-assisted diagnostics, not as window-start-FMS performance.",
            "- Short-horizon adaptive runs should be compared descriptively against longer-horizon runs, not as identical forecasting tasks.",
        ]
    )
    Path(output_dir, "adaptive_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(
    args: argparse.Namespace,
    ranked: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    plots: Sequence[str],
    hardware: Mapping[str, Any],
    start_time: float,
    adaptive_used: int,
) -> None:
    top_lines = []
    for row in ranked[:10]:
        top_lines.append(
            f"| {row.get('run_name')} | {row.get('val_MAE')} | {row.get('val_RMSE')} | {row.get('val_R2')} | {row.get('fms_context_mode')} | {row.get('anchor_mode')} |"
        )
    lines = [
        "# LC-SA-TCNFormer Final Report",
        "",
        "## 구현 내용",
        "- LC-SA-TCNFormer 모델, validation-only search runner, 앵커 메타데이터 포함 prediction CSV, validation leaderboard를 생성했습니다.",
        "",
        "## 실제 예산",
        f"- Stage/adaptive completed runs in leaderboard: {len(ranked)}",
        f"- Adaptive runs used: {adaptive_used}",
        f"- Elapsed seconds for this invocation: {time.time() - start_time:.1f}",
        "",
        "## Validation Leaderboard",
        "| run_name | val_MAE | val_RMSE | val_R2 | fms_context_mode | anchor_mode |",
        "| --- | --- | --- | --- | --- | --- |",
        *top_lines,
        "",
        "## 선택 구성",
        f"- run_name: {selected.get('run_name')}",
        f"- calibration_seconds: {selected.get('calibration_seconds')}",
        f"- recent_window_seconds: {selected.get('recent_window_seconds')}",
        f"- horizon_seconds: {selected.get('horizon_seconds')}",
        f"- fms_context_mode: {selected.get('fms_context_mode')}",
        f"- anchor_mode: {selected.get('anchor_mode')}",
        f"- use_static: {selected.get('use_static')}",
        f"- d_model: {selected.get('d_model')}",
        f"- transformer_layers: {selected.get('transformer_layers')}",
        f"- pooling: {selected.get('pooling')}",
        f"- loss_mode: {selected.get('loss_mode')}",
        "",
        "## Final Test Metrics",
        f"- test_MAE: {test_metrics.get('mae')}",
        f"- test_RMSE: {test_metrics.get('rmse')}",
        f"- test_R2: {test_metrics.get('r2')}",
        f"- test_sMAPE: {test_metrics.get('smape')}",
        f"- common_test_MAE: {test_metrics.get('common_mae')}",
        f"- common_test_RMSE: {test_metrics.get('common_rmse')}",
        "",
        "## Plots",
        *(f"- {path}" for path in plots),
        "",
        "## Hardware",
        f"- GPU: {hardware.get('gpu')}",
        f"- CUDA available: {hardware.get('cuda_available')}",
        "",
        "## Reproducibility",
        f"- command: {' '.join(sys.argv)}",
        f"- seed: {args.seed}",
        f"- git status short: {hardware.get('git_status_short')}",
        "",
        "## 주의",
        "- 모델 선택은 validation MAE 기준으로만 수행했습니다.",
        "- test metrics는 최종 선택 이후에만 계산했습니다.",
        "- sparse_anchor 결과는 anchor-assisted diagnostic으로만 해석하고 window-start-FMS 결과와 섞지 않습니다.",
        "- 최종 선택은 adaptive short-horizon 후보(horizon_seconds=1.0)입니다. 5/10/15초 이상 horizon 결과와는 같은 난이도의 task로 직접 비교하지 말고 horizon별 성능으로 해석해야 합니다.",
    ]
    Path(args.output_dir, "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded LC-SA-TCNFormer validation-based full search.")
    parser.add_argument("--data_dir", default="./DenseFMS/Dataset")
    parser.add_argument("--split_file", default="./artifacts/densefms_split_seed42.json")
    parser.add_argument("--output_dir", default="./runs/lc_sa_tcnformer_full_search")
    parser.add_argument("--config", default="configs/lc_sa_tcnformer.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--max_stage1_runs", type=int, default=24)
    parser.add_argument("--max_stage2_runs", type=int, default=8)
    parser.add_argument("--max_adaptive_runs", type=int, default=8)
    parser.add_argument("--max_epochs_stage1", type=int, default=40)
    parser.add_argument("--max_epochs_stage2", type=int, default=80)
    parser.add_argument("--max_epochs_final", type=int, default=100)
    parser.add_argument("--patience_stage1", type=int, default=6)
    parser.add_argument("--patience_stage2", type=int, default=10)
    parser.add_argument("--patience_final", type=int, default=12)
    parser.add_argument("--allow_test_eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_test_eval", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()
    if not args.allow_test_eval:
        args.no_test_eval = True

    start_time = time.time()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    hardware = inspect_hardware(args.output_dir)
    stage1 = build_stage1(args.max_stage1_runs)
    save_json(Path(args.output_dir) / "stage1_manifest.json", {"runs": stage1, "hardware": hardware})
    for item in stage1:
        print("[stage1]", " ".join(train_cmd(args, item, args.max_epochs_stage1, args.patience_stage1, stage1)))
    if args.dry_run:
        write_leaderboard([summarize_run(args, item) for item in stage1], args.output_dir)
        return
    if not args.summary_only:
        run_sanity_tests(args)
        for item in stage1:
            run_train(args, item, args.max_epochs_stage1, args.patience_stage1, stage1)
    rows = [summarize_run(args, item) for item in stage1]
    ranked = write_leaderboard(rows, args.output_dir)
    if not ranked:
        raise RuntimeError("No completed Stage 1 runs.")

    stage2 = build_stage2(ranked[:3], args.max_stage2_runs)
    save_json(Path(args.output_dir) / "stage2_manifest.json", {"runs": stage2})
    if not args.summary_only:
        for item in stage2:
            run_train(args, item, args.max_epochs_stage2, args.patience_stage2, stage1 + stage2)
    all_specs = stage1 + stage2
    rows = [summarize_run(args, item) for item in all_specs]
    ranked = write_leaderboard(rows, args.output_dir)
    diagnosis = diagnose_mediocre(args, ranked, args.output_dir)

    adaptive: List[Dict[str, Any]] = []
    adaptive_rows: List[Dict[str, Any]] = []
    if diagnosis.get("triggered") and not args.summary_only:
        adaptive = build_adaptive(ranked[0], ranked, args.max_adaptive_runs)
        save_json(Path(args.output_dir) / "adaptive_manifest.json", {"runs": adaptive, "diagnosis": diagnosis})
        for item in adaptive:
            run_train(args, item, min(args.max_epochs_stage2, 60), min(args.patience_stage2, 8), all_specs + adaptive)
        adaptive_rows = [summarize_run(args, item) for item in adaptive]
        all_specs += adaptive
        rows = [summarize_run(args, item) for item in all_specs]
        ranked = write_leaderboard(rows, args.output_dir)

    if not ranked:
        raise RuntimeError("No completed runs after search.")
    main_candidates = [row for row in ranked if str(row.get("fms_context_mode", "start_only")) in {"none", "start_only"}]
    selected = main_candidates[0] if main_candidates else ranked[0]
    save_json(Path(args.output_dir) / "selected_config.json", selected["spec"])
    final_item = make_final_spec(selected)
    save_json(Path(args.output_dir) / "final_training_spec.json", final_item)
    if not args.summary_only:
        run_train(args, final_item, args.max_epochs_final, args.patience_final, all_specs + [final_item])
    final_row = summarize_run(args, final_item)
    if final_row.get("status") == "completed":
        selected = final_row
        rows = [summarize_run(args, item) for item in all_specs + [final_item]]
        ranked = write_leaderboard(rows, args.output_dir)
    test_metrics = run_final_evaluation(args, selected, all_specs + [final_item]) if not args.summary_only else {}
    plots = generate_final_plots(args, selected, ranked)
    if diagnosis.get("triggered"):
        write_adaptive_report(args.output_dir, diagnosis, adaptive_rows, test_metrics)
    write_final_report(args, ranked, selected, test_metrics, plots, hardware, start_time, len(adaptive_rows))


if __name__ == "__main__":
    main()
