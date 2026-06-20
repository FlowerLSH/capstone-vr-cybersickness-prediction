#!/usr/bin/env python
"""Long target-driven DenseFMS model search runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


HORIZONS = [1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0]
CORE_HORIZONS = [1.0, 2.5, 5.0, 10.0, 15.0]
LEADERBOARD_COLUMNS = [
    "rank",
    "run_name",
    "status",
    "stage",
    "track",
    "model_family",
    "model",
    "horizon_seconds",
    "anchor_mode",
    "anchor_interval_seconds",
    "use_static",
    "predict_delta_from_anchor",
    "multi_horizon",
    "horizon_set",
    "fms_context_mode",
    "recent_window_seconds",
    "calibration_seconds",
    "learning_rate",
    "weight_decay",
    "dropout",
    "hidden_dim",
    "d_model",
    "val_MAE",
    "val_RMSE",
    "val_R2",
    "common_val_MAE",
    "best_epoch",
    "parameter_count",
    "deployment_realistic",
    "upper_bound",
    "target_tier",
    "metrics_path",
    "checkpoint_path",
    "prediction_csv_path",
]
BASELINE_COLUMNS = [
    "baseline_name",
    "horizon_seconds",
    "fms_context_mode",
    "anchor_mode",
    "anchor_interval_seconds",
    "track",
    "val_MAE",
    "val_RMSE",
    "val_R2",
    "val_sMAPE",
    "n",
    "target_tier",
    "prediction_csv_path",
]


def tag(value: Any) -> str:
    return f"{float(value):g}".replace(".", "p")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def git_status() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
    except Exception as exc:
        return f"git status unavailable: {exc}"


def target_tier(horizon: float, mae: float) -> str:
    if not math.isfinite(mae):
        return "none"
    h = float(horizon)
    if h == 1.0:
        if mae <= 1.0:
            return "stretch"
        if mae <= 1.25:
            return "strong"
        if mae <= 1.50:
            return "acceptable"
    if h in {2.5, 5.0}:
        if mae <= 1.25:
            return "stretch"
        if mae <= 1.50:
            return "strong"
        if mae <= 1.75:
            return "acceptable"
    if h in {10.0, 15.0}:
        if mae <= 1.50:
            return "stretch"
        if mae <= 1.75:
            return "strong"
        if mae <= 2.00:
            return "acceptable"
    return "stress" if h in {20.0, 30.0} else "none"


FMS_CONTEXT_MODES = {"none", "start_only", "calibration_history", "sparse_anchor"}


def is_deployment(spec: Mapping[str, Any]) -> bool:
    """Compatibility flag for sparse-anchor diagnostic candidates.

    These candidates are anchor-assisted diagnostics, not deployment-realistic
    window-start-FMS results.
    """
    return (
        float(spec.get("horizon_seconds", 0.0)) in {5.0, 10.0, 15.0}
        and str(spec.get("fms_context_mode", "")) == "sparse_anchor"
        and spec.get("anchor_mode") == "sparse_observed"
        and float(spec.get("anchor_interval_seconds", 0.0)) >= 60.0
        and not bool(spec.get("upper_bound", False))
    )


def run_name(spec: Mapping[str, Any]) -> str:
    model = str(spec["model"])
    stage = str(spec["stage"])
    fam = str(spec.get("model_family", model))
    context = f"_fms{str(spec.get('fms_context_mode', 'start_only')).replace('_', '')}"
    interval = ""
    if spec.get("anchor_mode") == "sparse_observed":
        interval = f"_ai{tag(spec.get('anchor_interval_seconds', 60.0))}"
    static = "static" if spec.get("use_static") else "no_static"
    delta = "_delta" if spec.get("predict_delta_from_anchor") else ""
    mh = "_mh" if spec.get("multi_horizon") else ""
    extras = []
    for key in (
        "hidden_dim",
        "d_model",
        "gru_layers",
        "branch_dropout",
        "anchor_dropout",
        "dropout",
        "loss_type",
        "learning_rate",
        "weight_decay",
        "seed",
    ):
        if key in spec and spec[key] not in (None, ""):
            extras.append(f"{key[:3]}{tag(spec[key]) if isinstance(spec[key], (int, float)) else spec[key]}")
    extra = "_" + "_".join(extras) if extras else ""
    return (
        f"{stage}_{fam}_c{tag(spec.get('calibration_seconds', 120))}_w{tag(spec.get('recent_window_seconds', 30))}_"
        f"h{tag(spec.get('horizon_seconds', 1))}{context}_{spec.get('anchor_mode', 'none')}{interval}_{static}_{model}{delta}{mh}{extra}"
    )


def with_name(spec: Mapping[str, Any]) -> Dict[str, Any]:
    item = dict(spec)
    item.setdefault("fms_context_mode", "start_only")
    mode = str(item.get("fms_context_mode", "start_only")).lower()
    if mode not in FMS_CONTEXT_MODES:
        raise ValueError(f"fms_context_mode must be one of {sorted(FMS_CONTEXT_MODES)}, got {mode!r}")
    item["fms_context_mode"] = mode
    item.setdefault("anchor_interval_seconds", 60.0)
    item.setdefault("predict_delta_from_anchor", False)
    item.setdefault("use_static", False)
    if mode == "sparse_anchor":
        item["anchor_mode"] = "sparse_observed"
        item["anchor_interval_seconds"] = float(item.get("anchor_interval_seconds") or 60.0)
        item["predict_delta_from_anchor"] = bool(item.get("predict_delta_from_anchor", True))
        item["track"] = "sparse_anchor_diagnostic"
    else:
        item["anchor_mode"] = "none"
        item["anchor_interval_seconds"] = 0.0
        item["predict_delta_from_anchor"] = False
        if mode == "none":
            item["track"] = "motion_only"
        elif mode == "start_only":
            item["track"] = "start_fms_only"
        else:
            item["track"] = "calibration_history_diagnostic"
    item["run_name"] = run_name(item)
    item["upper_bound"] = False
    item["deployment_realistic"] = is_deployment(item)
    return item


def unique_specs(items: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        item = with_name(item)
        key = (
            item.get("stage"),
            item.get("model"),
            item.get("model_family"),
            item.get("fms_context_mode"),
            float(item.get("calibration_seconds", 120)),
            float(item.get("recent_window_seconds", 30)),
            float(item.get("horizon_seconds", 1)),
            item.get("anchor_mode"),
            float(item.get("anchor_interval_seconds", 60)),
            bool(item.get("use_static", False)),
            bool(item.get("predict_delta_from_anchor", False)),
            bool(item.get("multi_horizon", False)),
            tuple(float(v) for v in item.get("horizon_set", []) or []),
            item.get("hidden_dim"),
            item.get("d_model"),
            item.get("gru_layers"),
            item.get("branch_dropout"),
            item.get("dropout"),
            item.get("loss_type"),
            item.get("learning_rate"),
            item.get("weight_decay"),
            item.get("seed"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def base_spec(stage: str, model: str, family: str, horizon: float, interval: float = 60.0) -> Dict[str, Any]:
    return {
        "stage": stage,
        "track": "best_score",
        "model_family": family,
        "model": model,
        "calibration_seconds": 120.0,
        "recent_window_seconds": 30.0,
        "horizon_seconds": float(horizon),
        "fms_context_mode": "start_only",
        "anchor_mode": "none",
        "anchor_interval_seconds": float(interval),
        "use_static": False,
        "predict_delta_from_anchor": False,
        "loss_type": "smooth_l1",
        "loss_mode": "level_only",
        "learning_rate": None,
        "weight_decay": None,
        "dropout": 0.1,
        "hidden_dim": 128,
        "d_model": 64,
        "multi_horizon": False,
        "horizon_set": None,
    }


def context_ablation_specs(horizon: float) -> List[Dict[str, Any]]:
    base = base_spec("stage1_context", "lc_sa_tcnformer", "lc_sa_tcnformer", horizon, 0.0)
    return [
        {**base, "stage": "motion_only", "fms_context_mode": "none"},
        {**base, "stage": "start_fms_only", "fms_context_mode": "start_only"},
        {**base, "stage": "calibration_history_diag", "fms_context_mode": "calibration_history"},
        {
            **base,
            "stage": "sparse_anchor_60s_diag",
            "fms_context_mode": "sparse_anchor",
            "anchor_mode": "sparse_observed",
            "anchor_interval_seconds": 60.0,
            "predict_delta_from_anchor": True,
        },
    ]


def build_stage1(reduced: bool = False) -> List[Dict[str, Any]]:
    horizons = CORE_HORIZONS if reduced else HORIZONS
    specs: List[Dict[str, Any]] = []
    for h in horizons:
        specs.extend(context_ablation_specs(h))
    return unique_specs(specs)


def build_stage2(reduced: bool = False) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    intervals = [0.0]
    for h in CORE_HORIZONS:
        for interval in intervals:
            specs.append(base_spec("stage2", "anchor_delta_mlp", "anchor_delta_mlp", h, interval))
    for h in [1.0, 5.0, 10.0, 15.0]:
        for interval in [30.0, 60.0]:
            for hidden in ([64] if reduced else [64, 128]):
                specs.append({**base_spec("stage2", "anchor_delta_gru", "anchor_delta_gru", h, interval), "hidden_dim": hidden, "gru_layers": 1})
    for h in CORE_HORIZONS:
        for interval in [30.0, 60.0]:
            for w in ([30.0] if reduced else [10.0, 30.0, 60.0]):
                specs.append({**base_spec("stage2", "recent_tcn_summary_calib", "recent_tcn_summary_calib", h, interval), "recent_window_seconds": w})
    for h in [1.0, 5.0, 10.0, 15.0]:
        for interval in [30.0, 60.0]:
            for branch_dropout in ([0.1] if reduced else [0.0, 0.1]):
                specs.append({**base_spec("stage2", "gated_fusion", "gated_fusion", h, interval), "branch_dropout": branch_dropout})
    mh_set = [1.0, 2.5, 5.0, 10.0, 15.0, 30.0]
    for model, family in [
        ("lc_sa_tcnformer", "multi_horizon_lc"),
        ("anchor_delta_mlp", "multi_horizon_mlp"),
        ("recent_tcn_summary_calib", "multi_horizon_summary_tcn"),
    ]:
        specs.append(
            {
                **base_spec("stage2", model, family, 1.0, 60.0),
                "multi_horizon": True,
                "horizon_set": mh_set,
            }
        )
    specs = unique_specs(specs)
    if not reduced:
        return specs
    mandatory = [s for s in specs if s.get("upper_bound") or s.get("multi_horizon")]
    regular = [s for s in specs if not (s.get("upper_bound") or s.get("multi_horizon"))]
    max_runs = 42
    return unique_specs(regular[: max(0, max_runs - len(mandatory))] + mandatory)


def build_classical_specs(reduced: bool = False) -> List[Dict[str, Any]]:
    models = ["ridge", "elasticnet", "hist_gradient_boosting", "gradient_boosting", "random_forest"]
    if reduced:
        models = ["ridge", "hist_gradient_boosting"]
    specs = []
    for h in CORE_HORIZONS:
        for model in models:
            interval = 60.0
            specs.append(
                with_name(
                    {
                        **base_spec("classical", model, "classical", h, interval),
                        "model": model,
                        "model_family": "classical",
                        "track": "classical",
                    }
                )
            )
    return specs[:20]


def build_refinement_specs(rows: Sequence[Mapping[str, Any]], reduced: bool = False) -> List[Dict[str, Any]]:
    ranked = completed_sorted(rows)
    seeds: List[Mapping[str, Any]] = []
    for h in CORE_HORIZONS:
        candidates = [
            row
            for row in ranked
            if abs(float(row.get("horizon_seconds", -999)) - h) < 1e-6
            and not row.get("upper_bound")
            and str(row.get("fms_context_mode", "start_only")) in {"none", "start_only"}
        ]
        if candidates:
            seeds.append(candidates[0]["spec"])
    deploy = [row for row in ranked if str(row.get("fms_context_mode", "")) == "sparse_anchor"]
    if deploy:
        seeds.append(deploy[0]["spec"])
    specs: List[Dict[str, Any]] = []
    for seed in seeds[: (4 if reduced else 8)]:
        for lr in ([3e-4, 1e-3] if reduced else [3e-4, 1e-3, 3e-3]):
            specs.append({**seed, "stage": "refine", "learning_rate": lr})
        for dropout in ([0.0, 0.2] if reduced else [0.0, 0.2, 0.3]):
            specs.append({**seed, "stage": "refine", "dropout": dropout})
        specs.append({**seed, "stage": "refine", "loss_type": "l1"})
    return unique_specs(specs)[: (12 if reduced else 40)]


def build_multiseed_specs(rows: Sequence[Mapping[str, Any]], reduced: bool = False) -> List[Dict[str, Any]]:
    ranked = completed_sorted(rows)
    selected: List[Mapping[str, Any]] = []
    for h in [1.0, 2.5, 5.0, 10.0, 15.0]:
        candidates = [
            row
            for row in ranked
            if abs(float(row.get("horizon_seconds", -999)) - h) < 1e-6
            and not row.get("upper_bound")
            and str(row.get("fms_context_mode", "start_only")) in {"none", "start_only"}
        ]
        if candidates:
            selected.append(candidates[0]["spec"])
    deploy = [row for row in ranked if str(row.get("fms_context_mode", "")) == "sparse_anchor"]
    if deploy:
        selected.append(deploy[0]["spec"])
    seeds = [42, 43] if reduced else [42, 43, 44]
    specs = []
    seen_base = set()
    for item in selected:
        key = item.get("run_name")
        if key in seen_base:
            continue
        seen_base.add(key)
        for seed in seeds:
            specs.append({**item, "stage": "multiseed", "seed": int(seed)})
    return unique_specs(specs)


def common_args(runs: Sequence[Mapping[str, Any]]) -> List[str]:
    max_calib = max(float(run.get("calibration_seconds", 120.0)) for run in runs)
    max_horizon = max(
        max([float(v) for v in (run.get("horizon_set") or [])] or [float(run.get("horizon_seconds", 1.0))])
        for run in runs
    )
    return ["--common_eval_current_start", f"{max_calib:g}", "--common_eval_max_horizon_seconds", f"{max_horizon:g}"]


def train_cmd(args: argparse.Namespace, item: Mapping[str, Any], epochs: int, patience: int, all_runs: Sequence[Mapping[str, Any]]) -> List[str]:
    lr = float(item.get("learning_rate") or args.learning_rate)
    wd = float(item.get("weight_decay") or args.weight_decay)
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
        str(item["model"]),
        "--run_name",
        str(item["run_name"]),
        "--split_file",
        args.split_file,
        "--seed",
        str(int(item.get("seed", args.seed))),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        f"{lr:g}",
        "--weight_decay",
        f"{wd:g}",
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--num_workers",
        str(args.num_workers),
        "--loss_type",
        str(item.get("loss_type", "smooth_l1")),
        "--loss_mode",
        str(item.get("loss_mode", "level_only")),
        "--high_fms_threshold",
        "10.0",
        "--calibration_seconds",
        f"{float(item.get('calibration_seconds', 120.0)):g}",
        "--recent_window_seconds",
        f"{float(item.get('recent_window_seconds', 30.0)):g}",
        "--horizon_seconds",
        f"{float(item.get('horizon_seconds', 1.0)):g}",
        "--anchor_mode",
        str(item.get("anchor_mode", "none")),
        "--anchor_interval_seconds",
        f"{float(item.get('anchor_interval_seconds', 60.0)):g}",
        "--fms_context_mode",
        str(item.get("fms_context_mode", "start_only")),
        "--d_model",
        str(int(item.get("d_model", 64))),
        "--hidden_dim",
        str(int(item.get("hidden_dim", item.get("d_model", 128)))),
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
        "--branch_dropout",
        f"{float(item.get('branch_dropout', 0.0)):g}",
        "--anchor_dropout",
        f"{float(item.get('anchor_dropout', 0.0)):g}",
        "--delta_scale",
        f"{float(item.get('delta_scale', 0.5)):g}",
        "--no_test_eval",
        "--no-save_plots",
    ]
    if "gru_layers" in item:
        cmd.extend(["--gru_layers", str(int(item["gru_layers"]))])
    if item.get("mlp_layers"):
        cmd.append("--mlp_layers")
        cmd.extend(str(int(v)) for v in item["mlp_layers"])
    if item.get("use_static"):
        cmd.extend(["--use_static", "--static_features", "age", "gender", "mssq"])
    else:
        cmd.append("--no_static")
    if item.get("predict_delta_from_anchor"):
        cmd.append("--predict_delta_from_anchor")
    if item.get("multi_horizon"):
        cmd.append("--multi_horizon")
        cmd.append("--horizon_set")
        cmd.extend(f"{float(v):g}" for v in item.get("horizon_set", []))
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.device:
        cmd.extend(["--device", args.device])
    cmd.extend(common_args(all_runs))
    return cmd


def eval_cmd(args: argparse.Namespace, item: Mapping[str, Any], split: str, common_runs: Sequence[Mapping[str, Any]]) -> List[str]:
    cmd = [
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
        f"{float(item.get('calibration_seconds', 120.0)):g}",
        "--recent_window_seconds",
        f"{float(item.get('recent_window_seconds', 30.0)):g}",
        "--horizon_seconds",
        f"{float(item.get('horizon_seconds', 1.0)):g}",
    ]
    cmd.extend(common_args(common_runs))
    return cmd


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


def inspect_hardware(output_dir: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_status_short": git_status(),
    }
    for name, cmd in {"nvidia_smi": ["nvidia-smi"], "free_h": ["free", "-h"], "df_h": ["df", "-h", "."]}.items():
        try:
            payload[name] = subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=20)
        except Exception as exc:
            payload[name] = f"unavailable: {exc}"
    save_json(output_dir / "hardware_summary.json", payload)
    return payload


def run_sanity(args: argparse.Namespace) -> None:
    cmd = [sys.executable, "scripts/run_densefms_sanity_tests.py"]
    if args.dry_run:
        return
    rc = run_process(cmd, ROOT, Path(args.output_dir) / "sanity_tests.log")
    if rc != 0:
        raise RuntimeError("Sanity tests failed; stopping before training.")


def neural_run(args: argparse.Namespace, item: Mapping[str, Any], epochs: int, patience: int, all_runs: Sequence[Mapping[str, Any]]) -> None:
    run_dir = Path(args.output_dir) / item["run_name"]
    if args.skip_existing and (run_dir / "metrics.json").exists() and (run_dir / "best.pt").exists():
        print(f"skip existing: {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = train_cmd(args, item, epochs, patience, all_runs)
    save_json(run_dir / "run_spec.json", item)
    save_json(run_dir / "command.json", {"train": cmd})
    (run_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    rc = run_process(cmd, ROOT, run_dir / "train.log")
    save_json(run_dir / "status.json", {"status": "completed" if rc == 0 else "failed", "returncode": rc, "finished_at": time.time()})
    if rc != 0:
        raise RuntimeError(f"Training failed for {item['run_name']}")


def summarize_neural(args: argparse.Namespace, item: Mapping[str, Any]) -> Dict[str, Any]:
    run_dir = Path(args.output_dir) / item["run_name"]
    metrics_path = run_dir / "metrics.json"
    row = {**item, "status": "missing", "spec": dict(item), "metrics_path": str(metrics_path), "checkpoint_path": str(run_dir / "best.pt"), "prediction_csv_path": str(run_dir / "val_predictions.csv")}
    if not metrics_path.exists():
        return row
    payload = load_json(metrics_path)
    metrics = payload.get("metrics", {})
    best = metrics.get("best_val_metrics", {})
    row.update(
        {
            "status": "completed",
            "val_MAE": best.get("mae"),
            "val_RMSE": best.get("rmse"),
            "val_R2": best.get("r2"),
            "common_val_MAE": best.get("common_mae"),
            "best_epoch": metrics.get("best_epoch"),
            "parameter_count": metrics.get("parameter_count"),
            "target_tier": target_tier(float(item.get("horizon_seconds", 1.0)), float(best.get("mae", math.nan))),
        }
    )
    return row


def recent_stats_np(head: np.ndarray) -> np.ndarray:
    first = head[:, :3]
    second = head[:, 3:6] if head.shape[1] >= 6 else head[:, :3]
    first_mag = np.linalg.norm(first, axis=1)
    second_mag = np.linalg.norm(second, axis=1)
    jerk = np.diff(first, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=1) if len(jerk) else np.asarray([0.0])
    return np.concatenate(
        [
            np.nanmean(head, axis=0),
            np.nanstd(head, axis=0),
            np.nanmin(head, axis=0),
            np.nanmax(head, axis=0),
            [np.nanmean(first_mag), np.nanstd(first_mag), np.nanmax(first_mag)],
            [np.nanmean(second_mag), np.nanstd(second_mag), np.nanmax(second_mag)],
            [np.nanmean(jerk_mag), np.nanstd(jerk_mag), np.nanmax(jerk_mag)],
        ]
    )


def static_np(sess: Any) -> List[float]:
    gender = str(sess.gender or "unknown")
    return [
        float(sess.age) if sess.age is not None and np.isfinite(sess.age) else 0.0,
        float(sess.mssq) if sess.mssq is not None and np.isfinite(sess.mssq) else 0.0,
        1.0 if gender == "male" else 0.0,
        1.0 if gender == "female" else 0.0,
        1.0 if gender not in {"male", "female"} else 0.0,
    ]


def feature_samples(
    sessions: Sequence[Any],
    calibration_steps: int,
    recent_steps: int,
    horizon_steps: int,
    anchor_interval_steps: int,
    horizon_seconds: float,
    fms_context_mode: str = "start_only",
    use_static: bool = False,
    cap: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    x: List[np.ndarray] = []
    y: List[float] = []
    recs: List[Dict[str, Any]] = []
    mode = str(fms_context_mode).lower()
    if mode not in FMS_CONTEXT_MODES:
        raise ValueError(f"fms_context_mode must be one of {sorted(FMS_CONTEXT_MODES)}, got {mode!r}")
    for sess in sessions:
        start = max(calibration_steps, recent_steps - 1)
        end = sess.length - horizon_steps
        calib = sess.fms[:calibration_steps].astype(np.float64)
        if mode == "none":
            context_calib = np.zeros_like(calib)
        else:
            context_calib = calib
        calib_summary = np.asarray(
            [
                context_calib[0],
                context_calib[-1],
                np.nanmean(context_calib),
                np.nanstd(context_calib),
                np.nanmax(context_calib),
                np.nanmin(context_calib),
                context_calib[-1] - context_calib[0],
                (context_calib[-1] - context_calib[0]) / max(calibration_steps - 1, 1),
            ],
            dtype=np.float64,
        )
        for t in range(start, end):
            target_idx = t + horizon_steps
            target = float(sess.fms[target_idx])
            if not np.isfinite(target):
                continue
            if mode == "sparse_anchor":
                anchor_idx = max(calibration_steps - 1, (t // anchor_interval_steps) * anchor_interval_steps)
                anchor_idx = min(anchor_idx, t)
                if not np.isfinite(sess.fms[anchor_idx]):
                    finite = np.where(np.isfinite(sess.fms[: anchor_idx + 1]))[0]
                    if len(finite):
                        anchor_idx = int(finite[-1])
                anchor_fms = float(sess.fms[anchor_idx])
                time_since = float(t - anchor_idx) * 0.5
                anchor_mode = "sparse_observed"
            elif mode == "start_only":
                anchor_idx = max(0, t - recent_steps + 1)
                if not np.isfinite(sess.fms[anchor_idx]):
                    finite = np.where(np.isfinite(sess.fms[: anchor_idx + 1]))[0]
                    if len(finite):
                        anchor_idx = int(finite[-1])
                anchor_fms = float(sess.fms[anchor_idx]) if np.isfinite(sess.fms[anchor_idx]) else 0.0
                time_since = float(t - anchor_idx) * 0.5
                anchor_mode = "none"
            else:
                anchor_idx = -1
                anchor_fms = 0.0
                time_since = 0.0
                anchor_mode = "none"
            recent = sess.head[t - recent_steps + 1 : t + 1].astype(np.float64)
            parts = [
                np.asarray([anchor_fms, time_since / 120.0, float(horizon_seconds) / 60.0], dtype=np.float64),
                calib_summary,
                recent_stats_np(recent),
            ]
            if use_static:
                parts.append(np.asarray(static_np(sess), dtype=np.float64))
            feat = np.concatenate(parts)
            x.append(feat)
            y.append(target)
            recs.append(
                {
                    "participant_id": sess.participant_id,
                    "session_id": sess.session_id,
                    "source_file": sess.source_file,
                    "current_index": t,
                    "target_index": target_idx,
                    "current_time": float(sess.time[t]),
                    "target_time": float(sess.time[target_idx]) if target_idx < len(sess.time) else float(sess.time[t] + horizon_seconds),
                    "fms_context_mode": mode,
                    "anchor_index": anchor_idx if anchor_idx >= 0 else "",
                    "anchor_mode": anchor_mode,
                    "anchor_fms": anchor_fms,
                    "target_fms": target,
                }
            )
            if cap is not None and len(y) >= cap:
                return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), recs
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), recs


def _baseline_anchor_index(
    fms: np.ndarray,
    t: int,
    calibration_steps: int,
    recent_steps: int,
    anchor_interval_steps: int,
    anchor_mode: str,
) -> Optional[int]:
    if anchor_mode == "start_only":
        anchor_idx = t - recent_steps + 1
    elif anchor_mode == "calibration_end":
        anchor_idx = calibration_steps - 1
    elif anchor_mode == "recent_start_observed":
        anchor_idx = t - recent_steps + 1
    elif anchor_mode == "sparse_observed":
        anchor_idx = max(calibration_steps - 1, (t // anchor_interval_steps) * anchor_interval_steps)
        anchor_idx = min(anchor_idx, t)
    else:
        return None
    anchor_idx = max(0, min(int(anchor_idx), min(t, len(fms) - 1)))
    if np.isfinite(fms[anchor_idx]):
        return anchor_idx
    finite = np.where(np.isfinite(fms[: anchor_idx + 1]))[0]
    return int(finite[-1]) if len(finite) else None


def _target_values(
    sessions: Sequence[Any],
    calibration_steps: int,
    recent_steps: int,
    horizon_steps: int,
) -> np.ndarray:
    values: List[float] = []
    for sess in sessions:
        start = max(calibration_steps, recent_steps - 1)
        end = sess.length - horizon_steps
        for t in range(start, end):
            target = float(sess.fms[t + horizon_steps])
            if np.isfinite(target):
                values.append(target)
    return np.asarray(values, dtype=np.float64)


def _anchor_baseline_predictions(
    sessions: Sequence[Any],
    calibration_steps: int,
    recent_steps: int,
    horizon_steps: int,
    anchor_interval_steps: int,
    horizon_seconds: float,
    anchor_mode: str,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    y_true: List[float] = []
    y_pred: List[float] = []
    records: List[Dict[str, Any]] = []
    for sess in sessions:
        start = max(calibration_steps, recent_steps - 1)
        end = sess.length - horizon_steps
        for t in range(start, end):
            target_idx = t + horizon_steps
            target = float(sess.fms[target_idx])
            anchor_idx = _baseline_anchor_index(sess.fms, t, calibration_steps, recent_steps, anchor_interval_steps, anchor_mode)
            if anchor_idx is None or not np.isfinite(target):
                continue
            pred = float(sess.fms[anchor_idx])
            y_true.append(target)
            y_pred.append(pred)
            records.append(
                {
                    "participant_id": sess.participant_id,
                    "session_id": sess.session_id,
                    "source_file": sess.source_file,
                    "split": "val",
                    "horizon_seconds": float(horizon_seconds),
                    "fms_context_mode": "sparse_anchor" if anchor_mode == "sparse_observed" else "start_only" if anchor_mode == "start_only" else "calibration_history",
                    "current_index": int(t),
                    "target_index": int(target_idx),
                    "current_time": float(sess.time[t]),
                    "target_time": float(sess.time[target_idx]) if target_idx < len(sess.time) else float(sess.time[t] + horizon_seconds),
                    "anchor_index": int(anchor_idx),
                    "anchor_mode": anchor_mode,
                    "anchor_fms": pred,
                    "target_fms": target,
                    "predicted_fms": pred,
                    "absolute_error": abs(pred - target),
                    "squared_error": float((pred - target) ** 2),
                }
            )
    return np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64), records


def _constant_baseline_predictions(
    sessions: Sequence[Any],
    calibration_steps: int,
    recent_steps: int,
    horizon_steps: int,
    horizon_seconds: float,
    pred_value: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    y_true: List[float] = []
    y_pred: List[float] = []
    records: List[Dict[str, Any]] = []
    for sess in sessions:
        start = max(calibration_steps, recent_steps - 1)
        end = sess.length - horizon_steps
        for t in range(start, end):
            target_idx = t + horizon_steps
            target = float(sess.fms[target_idx])
            if not np.isfinite(target):
                continue
            y_true.append(target)
            y_pred.append(float(pred_value))
            records.append(
                {
                    "participant_id": sess.participant_id,
                    "session_id": sess.session_id,
                    "source_file": sess.source_file,
                    "split": "val",
                    "horizon_seconds": float(horizon_seconds),
                    "fms_context_mode": "none",
                    "current_index": int(t),
                    "target_index": int(target_idx),
                    "current_time": float(sess.time[t]),
                    "target_time": float(sess.time[target_idx]) if target_idx < len(sess.time) else float(sess.time[t] + horizon_seconds),
                    "anchor_index": "",
                    "anchor_mode": "none",
                    "anchor_fms": "",
                    "target_fms": target,
                    "predicted_fms": float(pred_value),
                    "absolute_error": abs(float(pred_value) - target),
                    "squared_error": float((float(pred_value) - target) ** 2),
                }
            )
    return np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64), records


def compute_stage0_baselines(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pred_dir = output_dir / "stage0_baseline_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for horizon in HORIZONS:
        raw_sessions, mapping, info = load_raw_sessions(
            args.data_dir,
            calibration_seconds=120.0,
            horizon_seconds=float(horizon),
            default_sampling_interval=0.5,
        )
        split = apply_saved_split(raw_sessions, load_json(args.split_file))
        sampling = float(info["sampling_interval"])
        c_steps = seconds_to_steps(120.0, sampling, name="calibration_seconds", warn=False)
        w_steps = seconds_to_steps(30.0, sampling, name="recent_window_seconds", warn=False)
        h_steps = seconds_to_steps(float(horizon), sampling, name="horizon_seconds", warn=False)
        ai_steps = seconds_to_steps(60.0, sampling, name="anchor_interval_seconds", warn=False)
        train_targets = _target_values(split["train"], c_steps, w_steps, h_steps)
        if train_targets.size:
            pred_value = float(np.nanmean(train_targets))
            y_true, y_pred, records = _constant_baseline_predictions(split["val"], c_steps, w_steps, h_steps, float(horizon), pred_value)
            pred_path = pred_dir / f"global_train_mean_h{tag(horizon)}.csv"
            if records:
                write_table(pred_path, records, list(records[0].keys()))
            metrics = compute_regression_metrics(y_true, y_pred)
            rows.append(
                {
                    "baseline_name": "global_train_mean",
                    "horizon_seconds": float(horizon),
                    "fms_context_mode": "none",
                    "anchor_mode": "none",
                    "anchor_interval_seconds": "",
                    "track": "baseline",
                    "val_MAE": metrics.get("mae"),
                    "val_RMSE": metrics.get("rmse"),
                    "val_R2": metrics.get("r2"),
                    "val_sMAPE": metrics.get("smape"),
                    "n": metrics.get("n"),
                    "target_tier": target_tier(float(horizon), float(metrics.get("mae", math.nan))),
                    "prediction_csv_path": str(pred_path),
                }
            )
        for anchor_mode, fms_context_mode, track in [
            ("start_only", "start_only", "start_fms_only"),
            ("calibration_end", "calibration_history", "calibration_history_diagnostic"),
            ("sparse_observed", "sparse_anchor", "sparse_anchor_diagnostic"),
        ]:
            y_true, y_pred, records = _anchor_baseline_predictions(
                split["val"], c_steps, w_steps, h_steps, ai_steps, float(horizon), anchor_mode
            )
            pred_path = pred_dir / f"{anchor_mode}_h{tag(horizon)}.csv"
            if records:
                write_table(pred_path, records, list(records[0].keys()))
            metrics = compute_regression_metrics(y_true, y_pred)
            rows.append(
                {
                    "baseline_name": anchor_mode,
                    "horizon_seconds": float(horizon),
                    "fms_context_mode": fms_context_mode,
                    "anchor_mode": anchor_mode,
                    "anchor_interval_seconds": 60.0 if anchor_mode == "sparse_observed" else "",
                    "track": track,
                    "val_MAE": metrics.get("mae"),
                    "val_RMSE": metrics.get("rmse"),
                    "val_R2": metrics.get("r2"),
                    "val_sMAPE": metrics.get("smape"),
                    "n": metrics.get("n"),
                    "target_tier": target_tier(float(horizon), float(metrics.get("mae", math.nan))),
                    "prediction_csv_path": str(pred_path),
                }
            )
        try:
            from sklearn.linear_model import Ridge

            x_train, y_train, _ = feature_samples(
                split["train"],
                c_steps,
                w_steps,
                h_steps,
                ai_steps,
                float(horizon),
                fms_context_mode="sparse_anchor",
                use_static=False,
                cap=args.classical_train_cap,
            )
            x_val, y_val, records = feature_samples(
                split["val"],
                c_steps,
                w_steps,
                h_steps,
                ai_steps,
                float(horizon),
                fms_context_mode="sparse_anchor",
                use_static=False,
            )
            if x_train.size and x_val.size:
                mean = np.nanmean(x_train, axis=0)
                std = np.nanstd(x_train, axis=0)
                std[std < 1e-8] = 1.0
                model = Ridge(alpha=1.0)
                model.fit(np.nan_to_num((x_train - mean) / std), y_train)
                pred = np.clip(np.asarray(model.predict(np.nan_to_num((x_val - mean) / std)), dtype=np.float64), 0.0, 20.0)
                for rec, p in zip(records, pred.tolist()):
                    rec.update(
                        {
                            "split": "val",
                            "horizon_seconds": float(horizon),
                            "fms_context_mode": "sparse_anchor",
                            "anchor_mode": "sparse_observed",
                            "anchor_interval_seconds": 60.0,
                            "predicted_fms": float(p),
                            "absolute_error": abs(float(p) - float(rec["target_fms"])),
                            "squared_error": float((float(p) - float(rec["target_fms"])) ** 2),
                        }
                    )
                pred_path = pred_dir / f"ridge_feature_h{tag(horizon)}.csv"
                if records:
                    write_table(pred_path, records, list(records[0].keys()))
                metrics = compute_regression_metrics(y_val, pred)
                rows.append(
                    {
                        "baseline_name": "ridge_feature",
                        "horizon_seconds": float(horizon),
                        "fms_context_mode": "sparse_anchor",
                        "anchor_mode": "sparse_observed",
                        "anchor_interval_seconds": 60.0,
                        "track": "baseline",
                        "val_MAE": metrics.get("mae"),
                        "val_RMSE": metrics.get("rmse"),
                        "val_R2": metrics.get("r2"),
                        "val_sMAPE": metrics.get("smape"),
                        "n": metrics.get("n"),
                        "target_tier": target_tier(float(horizon), float(metrics.get("mae", math.nan))),
                        "prediction_csv_path": str(pred_path),
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "baseline_name": "ridge_feature",
                    "horizon_seconds": float(horizon),
                    "fms_context_mode": "sparse_anchor",
                    "anchor_mode": "sparse_observed",
                    "anchor_interval_seconds": 60.0,
                    "track": "baseline",
                    "val_MAE": math.nan,
                    "val_RMSE": math.nan,
                    "val_R2": math.nan,
                    "val_sMAPE": math.nan,
                    "n": 0,
                    "target_tier": f"failed: {exc}",
                    "prediction_csv_path": "",
                }
            )
        _ = mapping
    rows = sorted(rows, key=lambda r: (float(r["horizon_seconds"]), float(r["val_MAE"]) if math.isfinite(float(r["val_MAE"])) else math.inf))
    write_table(output_dir / "baseline_results.csv", rows, BASELINE_COLUMNS)
    lines = ["# Stage 0 Baseline Results", ""]
    lines.append("| " + " | ".join(BASELINE_COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(BASELINE_COLUMNS)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in BASELINE_COLUMNS) + " |")
    (output_dir / "baseline_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def sklearn_model(name: str, seed: int) -> Any:
    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0)
    if name == "elasticnet":
        from sklearn.linear_model import ElasticNet

        return ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=5000, random_state=seed)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=120, max_depth=16, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    if name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, l2_regularization=0.01, random_state=seed)
    if name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(n_estimators=160, learning_rate=0.05, max_depth=3, random_state=seed)
    raise ValueError(f"Unknown classical model {name}")


def classical_run(args: argparse.Namespace, item: Mapping[str, Any]) -> None:
    run_dir = Path(args.output_dir) / item["run_name"]
    metrics_path = run_dir / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        print(f"skip existing: {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_sessions, mapping, info = load_raw_sessions(
        args.data_dir,
        calibration_seconds=float(item["calibration_seconds"]),
        horizon_seconds=float(item["horizon_seconds"]),
        default_sampling_interval=0.5,
    )
    split = apply_saved_split(raw_sessions, load_json(args.split_file))
    sampling = float(info["sampling_interval"])
    c_steps = seconds_to_steps(float(item["calibration_seconds"]), sampling, name="calibration_seconds", warn=False)
    w_steps = seconds_to_steps(float(item["recent_window_seconds"]), sampling, name="recent_window_seconds", warn=False)
    h_steps = seconds_to_steps(float(item["horizon_seconds"]), sampling, name="horizon_seconds", warn=False)
    ai_seconds = float(item.get("anchor_interval_seconds") or 60.0)
    ai_steps = seconds_to_steps(ai_seconds if ai_seconds > 0 else 60.0, sampling, name="anchor_interval_seconds", warn=False)
    x_train, y_train, _ = feature_samples(
        split["train"],
        c_steps,
        w_steps,
        h_steps,
        ai_steps,
        float(item["horizon_seconds"]),
        fms_context_mode=str(item.get("fms_context_mode", "start_only")),
        use_static=bool(item.get("use_static", False)),
        cap=args.classical_train_cap,
    )
    x_val, y_val, val_records = feature_samples(
        split["val"],
        c_steps,
        w_steps,
        h_steps,
        ai_steps,
        float(item["horizon_seconds"]),
        fms_context_mode=str(item.get("fms_context_mode", "start_only")),
        use_static=bool(item.get("use_static", False)),
    )
    if x_train.size == 0 or x_val.size == 0:
        raise RuntimeError(f"No classical samples for {item['run_name']}")
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std[std < 1e-8] = 1.0
    xt = np.nan_to_num((x_train - mean) / std)
    xv = np.nan_to_num((x_val - mean) / std)
    model = sklearn_model(str(item["model"]), int(args.seed))
    model.fit(xt, y_train)
    pred = np.asarray(model.predict(xv), dtype=np.float64)
    pred = np.clip(pred, 0.0, 20.0)
    metrics = compute_regression_metrics(y_val, pred)
    for rec, p in zip(val_records, pred.tolist()):
        rec.update(
            {
                "run_name": item["run_name"],
                "model_name": item["model"],
                "split": "val",
                "horizon_seconds": float(item["horizon_seconds"]),
                "fms_context_mode": item.get("fms_context_mode", "start_only"),
                "anchor_mode": item["anchor_mode"],
                "anchor_interval_seconds": float(item["anchor_interval_seconds"]),
                "predicted_fms": float(p),
                "absolute_error": abs(float(p) - float(rec["target_fms"])),
                "squared_error": float((float(p) - float(rec["target_fms"])) ** 2),
            }
        )
    pred_path = run_dir / "val_predictions.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(val_records[0].keys()))
        writer.writeheader()
        writer.writerows(val_records)
    with open(run_dir / "model.pkl", "wb") as f:
        pickle.dump({"model": model, "mean": mean, "std": std, "mapping": mapping, "info": info, "spec": dict(item)}, f)
    save_json(run_dir / "run_spec.json", item)
    save_json(
        metrics_path,
        {
            "run_dir": str(run_dir),
            "model": item["model"],
            "metrics": {"best_val_metrics": metrics, "best_epoch": 0, "parameter_count": 0, "val_metrics": metrics},
            "data_info": info,
            "inferred_columns": mapping,
        },
    )
    save_json(run_dir / "status.json", {"status": "completed", "finished_at": time.time()})


def summarize_classical(args: argparse.Namespace, item: Mapping[str, Any]) -> Dict[str, Any]:
    row = summarize_neural(args, item)
    row["checkpoint_path"] = str(Path(args.output_dir) / item["run_name"] / "model.pkl")
    return row


def summarize(args: argparse.Namespace, item: Mapping[str, Any]) -> Dict[str, Any]:
    return summarize_classical(args, item) if item.get("model_family") == "classical" else summarize_neural(args, item)


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


def is_main_context(row: Mapping[str, Any]) -> bool:
    return str(row.get("fms_context_mode", "start_only")) in {"none", "start_only"}


def is_sparse_anchor_diagnostic(row: Mapping[str, Any]) -> bool:
    return str(row.get("fms_context_mode", "")) == "sparse_anchor"


def write_leaderboard(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> List[Dict[str, Any]]:
    ranked = completed_sorted(rows)
    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(ranked, 1):
        row["target_tier"] = target_tier(float(row.get("horizon_seconds", 1.0)), float(row.get("val_MAE", math.nan)))
        out = {col: row.get(col, "") for col in LEADERBOARD_COLUMNS}
        out["rank"] = idx
        out_rows.append(out)
    with open(output_dir / "leaderboard_val.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)
    lines = ["# DenseFMS Long Target Validation Leaderboard", ""]
    lines.append("| " + " | ".join(LEADERBOARD_COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(LEADERBOARD_COLUMNS)) + " |")
    for row in out_rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in LEADERBOARD_COLUMNS) + " |")
    (output_dir / "leaderboard_val.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ranked


def append_progress(args: argparse.Namespace, row: Mapping[str, Any], ranked: Sequence[Mapping[str, Any]], next_action: str, start_time: float) -> None:
    out_dir = Path(args.output_dir)
    best = ranked[0] if ranked else {}
    payload = {
        "timestamp": now_iso(),
        "elapsed_seconds": time.time() - start_time,
        "run_name": row.get("run_name"),
        "model_family": row.get("model_family"),
        "horizon_seconds": row.get("horizon_seconds"),
        "anchor_mode": row.get("anchor_mode"),
        "anchor_interval_seconds": row.get("anchor_interval_seconds"),
        "fms_context_mode": row.get("fms_context_mode"),
        "val_MAE": row.get("val_MAE"),
        "val_RMSE": row.get("val_RMSE"),
        "val_R2": row.get("val_R2"),
        "best_so_far": best.get("run_name"),
        "best_val_MAE": best.get("val_MAE"),
        "target_tier": row.get("target_tier"),
        "next_planned_action": next_action,
    }
    with open(out_dir / "progress_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    with open(out_dir / "progress_log.md", "a", encoding="utf-8") as f:
        f.write(
            f"- {payload['timestamp']} | {payload['run_name']} | family={payload['model_family']} "
            f"| H={payload['horizon_seconds']} | val_MAE={payload['val_MAE']} | best={payload['best_val_MAE']} | next={next_action}\n"
        )


def write_checkpoint_summary(args: argparse.Namespace, ranked: Sequence[Mapping[str, Any]], start_time: float, next_action: str) -> None:
    output_dir = Path(args.output_dir)
    elapsed_hours = (time.time() - start_time) / 3600.0
    remaining_hard = max(0.0, float(args.wall_clock_hard_cap_hours) - elapsed_hours)
    deploy = next((r for r in ranked if str(r.get("fms_context_mode", "")) == "sparse_anchor"), None)
    multi = next((r for r in ranked if r.get("multi_horizon") and not r.get("upper_bound") and str(r.get("fms_context_mode", "start_only")) in {"none", "start_only"}), None)
    best_h = best_by_horizon(ranked, include_upper=False)
    lines = [
        "# DenseFMS Long Target Checkpoint Summary",
        "",
        f"- timestamp: {now_iso()}",
        f"- elapsed_hours: {elapsed_hours:.2f}",
        f"- remaining_hard_cap_hours: {remaining_hard:.2f}",
        f"- completed_validation_runs: {len(ranked)}",
        f"- next_step_decision: {next_action}",
        "",
        "## Best By Horizon",
    ]
    for h in ["1", "2.5", "5", "10", "15"]:
        row = best_h.get(h)
        if row:
            lines.append(f"- H={h}s: val_MAE={row.get('val_MAE')} run={row.get('run_name')}")
        else:
            lines.append(f"- H={h}s: pending")
    lines.extend(["", "## Best Sparse-Anchor Diagnostic"])
    if deploy:
        lines.append(f"- {deploy.get('run_name')}: H={deploy.get('horizon_seconds')} val_MAE={deploy.get('val_MAE')}")
    else:
        lines.append("- pending")
    lines.extend(["", "## Best Multi-Horizon"])
    if multi:
        lines.append(f"- {multi.get('run_name')}: aggregate val_MAE={multi.get('val_MAE')}")
    else:
        lines.append("- pending")
    (output_dir / "checkpoint_summary_latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(output_dir / "progress_log.md", "a", encoding="utf-8") as f:
        f.write(
            f"\n## Checkpoint {now_iso()}\n\n"
            f"- elapsed_hours={elapsed_hours:.2f}, completed_validation_runs={len(ranked)}, "
            f"remaining_hard_cap_hours={remaining_hard:.2f}, next={next_action}\n"
        )


def write_table(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_analysis_tables(output_dir: Path, ranked: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "run_name",
        "stage",
        "track",
        "model_family",
        "model",
        "horizon_seconds",
        "fms_context_mode",
        "anchor_mode",
        "anchor_interval_seconds",
        "val_MAE",
        "val_RMSE",
        "val_R2",
        "deployment_realistic",
        "upper_bound",
    ]
    write_table(output_dir / "horizon_sweep.csv", [r for r in ranked if r.get("stage") == "stage1"], fieldnames)
    write_table(output_dir / "anchor_interval_sweep.csv", [r for r in ranked if r.get("stage") == "stage1"], fieldnames)
    write_table(output_dir / "model_family_comparison.csv", ranked, fieldnames)
    write_table(output_dir / "refinement_results.csv", [r for r in ranked if r.get("stage") == "refine"], fieldnames)
    write_table(output_dir / "multiseed_results.csv", [r for r in ranked if r.get("stage") == "multiseed"], fieldnames)
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in ranked:
        if row.get("stage") != "multiseed":
            continue
        key = (
            row.get("model_family"),
            row.get("model"),
            row.get("horizon_seconds"),
            row.get("fms_context_mode"),
            row.get("anchor_mode"),
            row.get("anchor_interval_seconds"),
            row.get("multi_horizon"),
            row.get("deployment_realistic"),
        )
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for key, vals in groups.items():
        mae = np.asarray([float(v.get("val_MAE")) for v in vals if math.isfinite(float(v.get("val_MAE", math.nan)))], dtype=np.float64)
        rmse = np.asarray([float(v.get("val_RMSE")) for v in vals if math.isfinite(float(v.get("val_RMSE", math.nan)))], dtype=np.float64)
        epochs = [float(v.get("best_epoch")) for v in vals if v.get("best_epoch") not in (None, "")]
        summary_rows.append(
            {
                "model_family": key[0],
                "model": key[1],
                "horizon_seconds": key[2],
                "fms_context_mode": key[3],
                "anchor_mode": key[4],
                "anchor_interval_seconds": key[5],
                "multi_horizon": key[6],
                "deployment_realistic": key[7],
                "runs": len(vals),
                "val_MAE_mean": float(np.mean(mae)) if mae.size else math.nan,
                "val_MAE_std": float(np.std(mae, ddof=1)) if mae.size > 1 else 0.0 if mae.size == 1 else math.nan,
                "val_RMSE_mean": float(np.mean(rmse)) if rmse.size else math.nan,
                "val_RMSE_std": float(np.std(rmse, ddof=1)) if rmse.size > 1 else 0.0 if rmse.size == 1 else math.nan,
                "best_epoch_mean": float(np.mean(epochs)) if epochs else math.nan,
                "best_epoch_min": float(np.min(epochs)) if epochs else math.nan,
                "best_epoch_max": float(np.max(epochs)) if epochs else math.nan,
            }
        )
    write_table(
        output_dir / "multiseed_summary.csv",
        summary_rows,
        [
            "model_family",
            "model",
            "horizon_seconds",
            "fms_context_mode",
            "anchor_mode",
            "anchor_interval_seconds",
            "multi_horizon",
            "deployment_realistic",
            "runs",
            "val_MAE_mean",
            "val_MAE_std",
            "val_RMSE_mean",
            "val_RMSE_std",
            "best_epoch_mean",
            "best_epoch_min",
            "best_epoch_max",
        ],
    )


def best_by_horizon(ranked: Sequence[Mapping[str, Any]], include_upper: bool = False) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for h in HORIZONS:
        candidates = [
            r
            for r in ranked
            if abs(float(r.get("horizon_seconds", -999)) - h) < 1e-6
            and (include_upper or not r.get("upper_bound"))
            and is_main_context(r)
        ]
        if candidates:
            out[f"{h:g}"] = dict(candidates[0])
    return out


def should_continue_after_target(args: argparse.Namespace, ranked: Sequence[Mapping[str, Any]], recent_rows: Sequence[Mapping[str, Any]], start_time: float) -> bool:
    elapsed_hours = (time.time() - start_time) / 3600.0
    if elapsed_hours >= float(args.wall_clock_hard_cap_hours):
        return False
    if elapsed_hours < float(args.wall_clock_target_hours):
        return True
    recent = completed_sorted(recent_rows[-10:])
    if not recent:
        return False
    overall_best = float(completed_sorted(ranked)[0]["val_MAE"]) if completed_sorted(ranked) else math.inf
    recent_best = float(recent[0]["val_MAE"])
    deploy_recent = any(str(r.get("fms_context_mode", "")) == "sparse_anchor" for r in recent)
    return recent_best <= overall_best + 0.03 or deploy_recent


def run_spec(args: argparse.Namespace, item: Mapping[str, Any], epochs: int, patience: int, all_runs: Sequence[Mapping[str, Any]], rows: List[Dict[str, Any]], start_time: float) -> None:
    if args.dry_run:
        print("DRY", item["run_name"])
        return
    if item.get("model_family") == "classical":
        classical_run(args, item)
    else:
        neural_run(args, item, epochs, patience, all_runs)
    row = summarize(args, item)
    rows.append(row)
    ranked = write_leaderboard(rows, Path(args.output_dir))
    write_analysis_tables(Path(args.output_dir), ranked)
    append_progress(args, row, ranked, "continue search", start_time)
    write_checkpoint_summary(args, ranked, start_time, "continue search")


def evaluate_classical_test(args: argparse.Namespace, row: Mapping[str, Any]) -> Dict[str, Any]:
    item = row["spec"]
    run_dir = Path(args.output_dir) / item["run_name"]
    with open(run_dir / "model.pkl", "rb") as f:
        payload = pickle.load(f)
    raw_sessions, _, _ = load_raw_sessions(
        args.data_dir,
        mapping=payload.get("mapping"),
        calibration_seconds=float(item["calibration_seconds"]),
        horizon_seconds=float(item["horizon_seconds"]),
        default_sampling_interval=0.5,
    )
    split = apply_saved_split(raw_sessions, load_json(args.split_file))
    sampling = float(payload["info"]["sampling_interval"])
    c_steps = seconds_to_steps(float(item["calibration_seconds"]), sampling, name="calibration_seconds", warn=False)
    w_steps = seconds_to_steps(float(item["recent_window_seconds"]), sampling, name="recent_window_seconds", warn=False)
    h_steps = seconds_to_steps(float(item["horizon_seconds"]), sampling, name="horizon_seconds", warn=False)
    ai_seconds = float(item.get("anchor_interval_seconds") or 60.0)
    ai_steps = seconds_to_steps(ai_seconds if ai_seconds > 0 else 60.0, sampling, name="anchor_interval_seconds", warn=False)
    x_test, y_test, recs = feature_samples(
        split["test"],
        c_steps,
        w_steps,
        h_steps,
        ai_steps,
        float(item["horizon_seconds"]),
        fms_context_mode=str(item.get("fms_context_mode", "start_only")),
        use_static=bool(item.get("use_static", False)),
    )
    x = np.nan_to_num((x_test - payload["mean"]) / payload["std"])
    pred = np.clip(np.asarray(payload["model"].predict(x), dtype=np.float64), 0.0, 20.0)
    metrics = compute_regression_metrics(y_test, pred)
    for rec, p in zip(recs, pred.tolist()):
        rec.update({"run_name": item["run_name"], "model_name": item["model"], "split": "test", "predicted_fms": float(p)})
    with open(run_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        writer.writeheader()
        writer.writerows(recs)
    save_json(run_dir / "eval_test_metrics.json", {"metrics": metrics})
    return metrics


def final_select(ranked: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    short = [r for r in ranked if float(r.get("horizon_seconds", 99)) in {1.0, 2.5} and not r.get("upper_bound") and is_main_context(r)]
    deploy = [r for r in ranked if is_sparse_anchor_diagnostic(r)]
    mh = [r for r in ranked if r.get("multi_horizon") and not r.get("upper_bound") and is_main_context(r)]
    upper = [r for r in ranked if r.get("upper_bound")]
    for pool in (short, deploy, mh, upper):
        if pool and pool[0]["run_name"] not in {r["run_name"] for r in selected}:
            selected.append(dict(pool[0]))
    return selected


def run_final_tests(args: argparse.Namespace, selected: Sequence[Mapping[str, Any]], all_runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if args.no_test_eval or not args.allow_final_test_eval:
        return []
    rows: List[Dict[str, Any]] = []
    for row in selected:
        item = row["spec"]
        if row.get("model_family") == "classical":
            metrics = evaluate_classical_test(args, row)
        else:
            for split in ("val", "test"):
                rc = run_process(eval_cmd(args, item, split, all_runs), ROOT, Path(args.output_dir) / item["run_name"] / f"eval_{split}.log")
                if rc != 0:
                    raise RuntimeError(f"Final {split} evaluation failed for {item['run_name']}")
            metrics_path = Path(args.output_dir) / item["run_name"] / "eval_test" / "metrics.json"
            metrics = load_json(metrics_path).get("metrics", {})
        rows.append(
            {
                "run_name": row["run_name"],
                "selection_role": "upper_bound" if row.get("upper_bound") else "sparse_anchor_diagnostic" if row.get("deployment_realistic") else "short_or_multihorizon",
                "model_family": row.get("model_family"),
                "horizon_seconds": row.get("horizon_seconds"),
                "test_MAE": metrics.get("mae"),
                "test_RMSE": metrics.get("rmse"),
                "test_R2": metrics.get("r2"),
                "test_sMAPE": metrics.get("smape"),
                "common_test_MAE": metrics.get("common_mae"),
                "common_test_RMSE": metrics.get("common_rmse"),
            }
        )
    write_table(
        Path(args.output_dir) / "final_test_metrics.csv",
        rows,
        ["run_name", "selection_role", "model_family", "horizon_seconds", "test_MAE", "test_RMSE", "test_R2", "test_sMAPE", "common_test_MAE", "common_test_RMSE"],
    )
    return rows


def plot_outputs(output_dir: Path, ranked: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ranked)
    if not df.empty:
        for path, x_col, title in [
            ("horizon_mae_curve.png", "horizon_seconds", "Horizon vs validation MAE"),
            ("anchor_interval_curve.png", "anchor_interval_seconds", "Anchor interval vs validation MAE"),
        ]:
            plt.figure(figsize=(7, 4))
            for fam, sub in df.groupby("model_family"):
                sub = sub.sort_values(x_col)
                plt.plot(sub[x_col].astype(float), sub["val_MAE"].astype(float), marker="o", linestyle="-", label=str(fam)[:20], alpha=0.7)
            plt.xlabel(x_col)
            plt.ylabel("Validation MAE")
            plt.title(title)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(plot_dir / path, dpi=150)
            plt.close()
        best_h = best_by_horizon(ranked, include_upper=False)
        plt.figure(figsize=(7, 4))
        plt.bar(list(best_h.keys()), [float(v["val_MAE"]) for v in best_h.values()])
        plt.xlabel("Horizon seconds")
        plt.ylabel("Best validation MAE")
        plt.tight_layout()
        plt.savefig(plot_dir / "best_by_horizon.png", dpi=150)
        plt.close()
        fam = df.groupby("model_family")["val_MAE"].min().sort_values()
        plt.figure(figsize=(8, 4))
        fam.plot(kind="bar")
        plt.ylabel("Best validation MAE")
        plt.tight_layout()
        plt.savefig(plot_dir / "model_family_comparison.png", dpi=150)
        plt.close()
    progress = output_dir / "progress_log.jsonl"
    if progress.exists():
        vals = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines() if line.strip()]
        if vals:
            best = []
            cur = math.inf
            for row in vals:
                mae = row.get("val_MAE")
                if mae is not None:
                    cur = min(cur, float(mae))
                best.append(cur)
            plt.figure(figsize=(8, 4))
            plt.plot(best)
            plt.ylabel("Best validation MAE so far")
            plt.xlabel("Completed run index")
            plt.tight_layout()
            plt.savefig(plot_dir / "progress_best_mae_over_time.png", dpi=150)
            plt.close()
    for role, row in [
        ("best_h1", next((r for r in ranked if abs(float(r.get("horizon_seconds", -1)) - 1.0) < 1e-6 and is_main_context(r)), None)),
        ("best_sparse_anchor_diag", next((r for r in ranked if is_sparse_anchor_diagnostic(r)), None)),
    ]:
        if row is None:
            continue
        pred_path = Path(row.get("prediction_csv_path", ""))
        if pred_path.exists():
            dfp = pd.read_csv(pred_path)
            if not dfp.empty and {"target_fms", "predicted_fms"}.issubset(dfp.columns):
                plt.figure(figsize=(5, 5))
                plt.scatter(dfp["target_fms"], dfp["predicted_fms"], s=4, alpha=0.3)
                lo = min(float(dfp["target_fms"].min()), float(dfp["predicted_fms"].min()))
                hi = max(float(dfp["target_fms"].max()), float(dfp["predicted_fms"].max()))
                plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
                plt.xlabel("Target FMS")
                plt.ylabel("Predicted FMS")
                plt.tight_layout()
                plt.savefig(plot_dir / f"val_predicted_vs_target_{role}.png", dpi=150)
                plt.close()
    if selected:
        row = selected[0]
        pred_path = Path(args_output := output_dir) / row["run_name"] / "eval_test" / "test_predictions.csv"
        if not pred_path.exists():
            pred_path = args_output / row["run_name"] / "test_predictions.csv"
        if pred_path.exists():
            dfp = pd.read_csv(pred_path)
            if not dfp.empty and {"target_fms", "predicted_fms"}.issubset(dfp.columns):
                plt.figure(figsize=(5, 5))
                plt.scatter(dfp["target_fms"], dfp["predicted_fms"], s=4, alpha=0.3)
                lo = min(float(dfp["target_fms"].min()), float(dfp["predicted_fms"].min()))
                hi = max(float(dfp["target_fms"].max()), float(dfp["predicted_fms"].max()))
                plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
                plt.tight_layout()
                plt.savefig(plot_dir / "test_predicted_vs_target_selected.png", dpi=150)
                plt.close()
                residual = dfp["predicted_fms"] - dfp["target_fms"]
                plt.figure(figsize=(7, 4))
                plt.hist(residual, bins=50)
                plt.tight_layout()
                plt.savefig(plot_dir / "residual_histogram_selected.png", dpi=150)
                plt.close()
    mh = [r for r in ranked if r.get("multi_horizon")]
    if mh:
        row = mh[0]
        pred_path = Path(row.get("prediction_csv_path", ""))
        if pred_path.exists():
            dfp = pd.read_csv(pred_path)
            if "horizon_seconds" in dfp.columns:
                curve = dfp.groupby("horizon_seconds")["absolute_error"].mean()
                plt.figure(figsize=(7, 4))
                curve.plot(marker="o")
                plt.ylabel("Validation MAE")
                plt.tight_layout()
                plt.savefig(plot_dir / "multi_horizon_curve.png", dpi=150)
                plt.close()


def write_final_report(args: argparse.Namespace, ranked: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]], final_tests: Sequence[Mapping[str, Any]], hardware: Mapping[str, Any], start_time: float) -> None:
    output_dir = Path(args.output_dir)
    main_ranked = [r for r in ranked if is_main_context(r)]
    sparse_ranked = [r for r in ranked if is_sparse_anchor_diagnostic(r)]
    calibration_history_ranked = [r for r in ranked if str(r.get("fms_context_mode", "")) == "calibration_history"]
    best_h = best_by_horizon(ranked, include_upper=False)
    reached_1 = any(float(r.get("val_MAE", math.inf)) <= 1.0 for r in main_ranked if abs(float(r.get("horizon_seconds", -1)) - 1.0) < 1e-6)
    low_1 = any(float(r.get("val_MAE", math.inf)) <= 1.5 for r in main_ranked if float(r.get("horizon_seconds", 99)) in {1.0, 2.5, 5.0})
    best_short = next((r for r in selected if float(r.get("horizon_seconds", 99)) in {1.0, 2.5} and not r.get("upper_bound")), None)
    best_sparse_anchor = next((r for r in selected if r.get("deployment_realistic")), None)
    best_multi = next((r for r in selected if r.get("multi_horizon") and not r.get("upper_bound")), None)
    best_upper = next((r for r in selected if r.get("upper_bound")), None)
    baseline_rows: List[Dict[str, Any]] = []
    baseline_path = output_dir / "baseline_results.csv"
    if baseline_path.exists():
        try:
            baseline_rows = pd.read_csv(baseline_path).to_dict("records")
        except Exception:
            baseline_rows = []
    fam_best: Dict[Any, Mapping[str, Any]] = {}
    for row in main_ranked:
        fam = row.get("model_family")
        if fam and fam not in fam_best:
            fam_best[fam] = row

    def mae_text(row: Optional[Mapping[str, Any]]) -> str:
        if not row:
            return "pending"
        try:
            return f"{float(row.get('val_MAE')):.4f}"
        except Exception:
            return str(row.get("val_MAE"))

    def family_mae(name: str) -> float:
        row = fam_best.get(name)
        try:
            return float(row.get("val_MAE")) if row else math.inf
        except Exception:
            return math.inf

    mlp_mae = family_mae("anchor_delta_mlp")
    gru_mae = family_mae("anchor_delta_gru")
    multi_candidates = [r for r in ranked if r.get("multi_horizon") and not r.get("upper_bound") and is_main_context(r)]
    best_multi_mae = float(multi_candidates[0].get("val_MAE", math.inf)) if multi_candidates else math.inf

    lines = [
        "# DenseFMS Long Target Search Final Report",
        "",
        "## 1. Previous result recap",
        "- Previous best final model was LC-SA-TCNFormer H=1s sparse_observed/static/delta with validation MAE 1.7570 and test MAE 1.7076.",
        "- Previous 15s sparse-anchor-assisted diagnostic candidate validation MAE was about 2.10.",
        "",
        "## 2. Target definition",
        "- Stretch target: validation MAE <= 1.0 for short-horizon forecasting if achievable.",
        "- H=1, H=2.5, H=5, H=10, H=15, H=20, and H=30 are reported separately.",
        "- Test metrics are final-report-only after validation-based selection.",
        "",
        "## 3. Search budget actually used",
        f"- Completed validation runs: {len(ranked)}",
        f"- Elapsed hours: {(time.time() - start_time) / 3600.0:.2f}",
        f"- Wall-clock target/soft/hard caps: {args.wall_clock_target_hours}/{args.wall_clock_soft_cap_hours}/{args.wall_clock_hard_cap_hours} hours",
        "",
        "## 4. Hardware summary",
        f"- GPU: {hardware.get('gpu')}",
        f"- CUDA available: {hardware.get('cuda_available')}",
        f"- CUDA device count: {hardware.get('cuda_device_count')}",
        "",
        "## 5. Sanity test results",
        "- Sanity tests were run before training; see `sanity_tests.log`.",
        "",
        "## 6. Baseline results",
        "- Stage 0 baselines are in `baseline_results.csv` and `baseline_results.md`.",
    ]
    for h in HORIZONS:
        candidates = [
            r
            for r in baseline_rows
            if abs(float(r.get("horizon_seconds", -999)) - float(h)) < 1e-6
            and math.isfinite(float(r.get("val_MAE", math.nan)))
        ]
        if candidates:
            best_base = sorted(candidates, key=lambda r: float(r["val_MAE"]))[0]
            lines.append(f"- H={h:g}s best baseline: {best_base.get('baseline_name')} val_MAE={float(best_base.get('val_MAE')):.4f}")
    lines.extend(
        [
            "",
            "## 7. Controlled horizon/anchor sweep",
            "- See `horizon_sweep.csv` and `anchor_interval_sweep.csv`; these are validation-only stage1 rows.",
            "",
            "## 8. Model family comparison",
        ]
    )
    for fam, row in fam_best.items():
        lines.append(f"- {fam}: best val_MAE={row.get('val_MAE')} run={row.get('run_name')}")
    lines.extend(
        [
            "",
            "## 9. Progressive refinement results",
            "- See `refinement_results.csv`.",
            "",
            "## 10. Multi-seed confirmation",
            "- See `multiseed_results.csv` and `multiseed_summary.csv` for validation MAE/RMSE mean and std.",
            "",
            "## 11. Best validation models by horizon",
            "- This table excludes `calibration_history` and `sparse_anchor` diagnostic rows.",
        ]
    )
    for h, row in best_h.items():
        lines.append(f"- H={h}s: val_MAE={row.get('val_MAE')} run={row.get('run_name')}")
    lines.extend(
        [
            "",
            "## 12. Whether MAE <= 1.0 was reached",
            f"- reached: {bool(reached_1)}",
            "",
            "## 13. Whether low-1 MAE was reached",
            f"- reached: {bool(low_1)}",
            "",
            "## 14. Best short-horizon model",
        ]
    )
    if best_short:
        lines.append(f"- {best_short.get('run_name')}: H={best_short.get('horizon_seconds')} val_MAE={best_short.get('val_MAE')}")
    else:
        lines.append("- No validation-selected short-horizon model was available.")
    lines.extend(["", "## 15. Sparse-anchor-assisted diagnostic"])
    lines.append("- This role uses sparse observed FMS anchors and must be interpreted separately from window-start-FMS results.")
    if best_sparse_anchor:
        lines.append(f"- selected: {best_sparse_anchor.get('run_name')}: H={best_sparse_anchor.get('horizon_seconds')} val_MAE={best_sparse_anchor.get('val_MAE')}")
    else:
        lines.append("- No sparse-anchor-assisted diagnostic model was available.")
    for row in sparse_ranked[:5]:
        lines.append(f"- diagnostic rank: H={row.get('horizon_seconds')} val_MAE={row.get('val_MAE')} run={row.get('run_name')}")
    lines.extend(["", "## 15b. Calibration-history diagnostic"])
    if calibration_history_ranked:
        for row in calibration_history_ranked[:5]:
            lines.append(f"- H={row.get('horizon_seconds')} val_MAE={row.get('val_MAE')} run={row.get('run_name')}")
    else:
        lines.append("- No calibration-history diagnostic model was available.")
    lines.extend(["", "## 16. Best multi-horizon model"])
    if best_multi:
        lines.append(f"- {best_multi.get('run_name')}: aggregate val_MAE={best_multi.get('val_MAE')}")
    else:
        lines.append("- No competitive multi-horizon model was selected.")
    if best_upper:
        lines.append(f"- Upper-bound reference: {best_upper.get('run_name')} val_MAE={best_upper.get('val_MAE')}")
    lines.extend(["", "## 17. Final test metrics"])
    for row in final_tests:
        lines.append(f"- {row.get('run_name')}: test_MAE={row.get('test_MAE')} test_RMSE={row.get('test_RMSE')} test_R2={row.get('test_R2')}")
    if not final_tests:
        lines.append("- Final test evaluation was skipped or no selected model was available.")
    lines.extend(
        [
            "",
            "## 18. Interpretation",
            f"- Did simple AnchorDeltaMLP beat complex models? AnchorDeltaMLP best={mae_text(fam_best.get('anchor_delta_mlp'))}, LC-SA-TCNFormer best={mae_text(fam_best.get('lc_sa_tcnformer'))}.",
            f"- Did GRU help? AnchorDeltaGRU best={mae_text(fam_best.get('anchor_delta_gru'))}; improvement over MLP: {bool(math.isfinite(gru_mae) and gru_mae < mlp_mae)}.",
            f"- Did recent motion help? RecentTCN+SummaryCalib best={mae_text(fam_best.get('recent_tcn_summary_calib'))}; compare within the same `fms_context_mode`.",
            "- Did window-start FMS help? Compare `motion_only` and `start_fms_only` rows; calibration-history-only and sparse-anchor rows are diagnostic only.",
            "- Did static help? Main search keeps static features disabled unless explicitly enabled, so static effect is not claimed here.",
            f"- Did multi-horizon help? Best multi-horizon aggregate val_MAE={best_multi_mae if math.isfinite(best_multi_mae) else 'pending'}.",
            f"- Is performance anchor-dominated? Compare sparse_anchor diagnostic rows with the best window-start-FMS validation MAE {mae_text(main_ranked[0] if main_ranked else None)}.",
            "- Sparse-anchor-assisted diagnostic and window-start-FMS tracks are kept separate by `fms_context_mode`.",
            "- H=1 performance is not claimed as long-horizon forecasting success.",
            "",
            "## 19. Limitations",
            "- Head/motion-only inputs may be insufficient at longer horizons.",
            "- FMS is subjective; calibration FMS, sparse FMS prompts, age/gender, and MSSQ create user burden when enabled.",
            "- H=1 performance is distinct from H=5/H=10/H=15 long-horizon performance.",
            "- Single-dataset validation limits generalization claims.",
            "",
            "## 20. Next recommended research step",
            "- Focus on window-start-FMS models for H=5/H=10/H=15 and keep sparse-anchor-assisted diagnostics as a separate diagnostic reference.",
        ]
    )
    (output_dir / "final_long_target_search_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run long target-driven DenseFMS search.")
    p.add_argument("--data_dir", default="./DenseFMS/Dataset")
    p.add_argument("--split_file", default="./artifacts/densefms_split_seed42.json")
    p.add_argument("--output_dir", default="./runs/densefms_long_target_search")
    p.add_argument("--config", default="configs/lc_sa_tcnformer.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--wall_clock_target_hours", type=float, default=7.0)
    p.add_argument("--wall_clock_soft_cap_hours", type=float, default=10.0)
    p.add_argument("--wall_clock_hard_cap_hours", type=float, default=12.0)
    p.add_argument("--reduced_budget", action="store_true")
    p.add_argument("--aggressive_budget", action="store_true")
    p.add_argument("--allow_final_test_eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no_test_eval", action="store_true")
    p.add_argument("--classical_train_cap", type=int, default=120000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    hardware = inspect_hardware(output_dir)
    progress_md = output_dir / "progress_log.md"
    if progress_md.exists():
        with open(progress_md, "a", encoding="utf-8") as f:
            f.write(f"\n## Runner restart / wall-clock reset {now_iso()}\n\n")
    else:
        progress_md.write_text(f"# DenseFMS long target search progress\n\nStarted: {now_iso()}\n\n", encoding="utf-8")
    if args.dry_run:
        print("dry run")
    run_sanity(args)
    previous = Path("runs/lc_sa_tcnformer_full_search/leaderboard_val.csv")
    if previous.exists():
        (output_dir / "previous_lc_sa_leaderboard_snapshot.csv").write_text(previous.read_text(encoding="utf-8"), encoding="utf-8")

    stage1 = build_stage1(args.reduced_budget)
    stage2 = build_stage2(args.reduced_budget)
    classical = build_classical_specs(args.reduced_budget)
    all_planned: List[Dict[str, Any]] = stage1 + classical + stage2
    save_json(output_dir / "planned_manifest.json", {"stage1": stage1, "classical": classical, "stage2": stage2})
    rows: List[Dict[str, Any]] = []
    if args.dry_run:
        for item in all_planned:
            print(item["run_name"])
        write_leaderboard([summarize(args, item) for item in all_planned], output_dir)
        return

    compute_stage0_baselines(args, output_dir)

    for item in stage1:
        run_spec(args, item, args.max_epochs, args.patience, all_planned, rows, start_time)
    for item in classical:
        run_spec(args, item, 0, 0, all_planned, rows, start_time)
    for item in stage2:
        if not should_continue_after_target(args, completed_sorted(rows), rows, start_time):
            break
        run_spec(args, item, args.max_epochs, args.patience, all_planned, rows, start_time)

    ranked = write_leaderboard(rows, output_dir)
    refine = build_refinement_specs(ranked, args.reduced_budget)
    save_json(output_dir / "refinement_manifest.json", {"runs": refine})
    all_planned += refine
    for item in refine:
        if not should_continue_after_target(args, completed_sorted(rows), rows, start_time):
            break
        run_spec(args, item, 90 if not args.reduced_budget else args.max_epochs, 12 if not args.reduced_budget else args.patience, all_planned, rows, start_time)

    ranked = write_leaderboard(rows, output_dir)
    multiseed = build_multiseed_specs(ranked, args.reduced_budget)
    save_json(output_dir / "multiseed_manifest.json", {"runs": multiseed})
    all_planned += multiseed
    for item in multiseed:
        if (time.time() - start_time) / 3600.0 >= float(args.wall_clock_hard_cap_hours):
            break
        run_spec(args, item, 120 if not args.reduced_budget else args.max_epochs, 15 if not args.reduced_budget else args.patience, all_planned, rows, start_time)

    ranked = write_leaderboard(rows, output_dir)
    write_analysis_tables(output_dir, ranked)
    selected = final_select(ranked)
    save_json(output_dir / "final_selected_models.json", {"models": selected})
    final_tests = run_final_tests(args, selected, all_planned)
    plot_outputs(output_dir, ranked, selected)
    write_final_report(args, ranked, selected, final_tests, hardware, start_time)
    write_checkpoint_summary(args, ranked, start_time, "final report complete")


if __name__ == "__main__":
    main()
