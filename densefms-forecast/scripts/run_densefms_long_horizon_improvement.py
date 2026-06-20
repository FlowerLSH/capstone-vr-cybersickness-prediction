#!/usr/bin/env python
"""Interrupt-safe long-horizon DenseFMS improvement runner.

This runner intentionally keeps model selection validation-only. Test
evaluation is deferred until final validation-selected roles are fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import signal
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

from src.densefms_forecast.utils import compute_regression_metrics, load_json, save_json, seconds_to_steps


RUN_ROOT = "runs/densefms_long_horizon_improvement_20260503"
DEFAULT_BASELINE_DIR = "runs/densefms_long_target_search"
DEFAULT_SPLIT = "artifacts/densefms_split_seed42.json"
DEFAULT_CONFIG = "configs/lc_sa_tcnformer.yaml"
SAMPLING_INTERVAL = 0.5
PRIMARY_HORIZONS = [5.0, 10.0, 15.0]
SECONDARY_HORIZONS = [2.5]
ALL_TARGET_HORIZONS = [2.5, 5.0, 10.0, 15.0]
BEST_SCORE_BASELINE = {2.5: 1.0492, 5.0: 1.2870, 10.0: 1.6261, 15.0: 1.8481}
DEPLOYMENT_BASELINE = {5.0: 1.7735, 10.0: 1.9199, 15.0: 2.0376}
STRETCH_TARGETS = {2.5: 1.0000, 5.0: 1.1500, 10.0: 1.4500, 15.0: 1.6500}
DEPLOYMENT_TARGETS = {5.0: 1.5500, 10.0: 1.7500, 15.0: 1.9000}

LEADERBOARD_COLUMNS = [
    "rank",
    "run_name",
    "status",
    "event",
    "stage",
    "track",
    "model_family",
    "model",
    "horizon_seconds",
    "horizon_steps",
    "horizon_set",
    "fms_context_mode",
    "calibration_seconds",
    "recent_window_seconds",
    "anchor_policy",
    "anchor_interval_seconds",
    "use_static",
    "loss_type",
    "loss_mode",
    "trend_weight",
    "horizon_loss_weights",
    "change_weight",
    "high_target_weight",
    "delta_prediction",
    "multi_horizon",
    "per_horizon_heads",
    "seed",
    "best_epoch",
    "val_MAE",
    "val_RMSE",
    "val_R2",
    "common_val_MAE",
    "runtime_seconds",
    "parameter_count",
    "deployment_realistic",
    "upper_bound",
    "improvement_vs_baseline_pct",
    "metrics_path",
    "checkpoint_path",
    "prediction_csv_path",
    "output_dir",
]

FINAL_TEST_COLUMNS = [
    "role_name",
    "run_name",
    "model_family",
    "model",
    "horizon_seconds",
    "horizon_set",
    "track",
    "selection_rationale",
    "val_MAE",
    "val_RMSE",
    "test_MAE",
    "test_RMSE",
    "test_R2",
    "test_sMAPE",
    "common_test_MAE",
    "common_test_RMSE",
    "checkpoint_path",
    "prediction_csv_path",
]

CURRENT_RUN: Optional[str] = None
STOP_REQUESTED = False


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def tag(value: Any) -> str:
    return f"{float(value):g}".replace(".", "p")


def horizon_steps(horizon_seconds: float) -> int:
    return seconds_to_steps(float(horizon_seconds), SAMPLING_INTERVAL, name="horizon_seconds", warn=False)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def command_text(cmd: Sequence[str]) -> str:
    return " ".join(str(part) for part in cmd)


def git_text(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def is_complete_run(run_dir: Path) -> bool:
    return (run_dir / "metrics.json").exists() and (run_dir / "best.pt").exists()


def is_deployment(spec: Mapping[str, Any]) -> bool:
    """Compatibility flag for sparse-anchor diagnostic candidates.

    These candidates are anchor-assisted diagnostics, not deployment-realistic
    window-start-FMS results.
    """
    return (
        float(spec.get("horizon_seconds", 0.0)) in {5.0, 10.0, 15.0}
        and str(spec.get("fms_context_mode", "")) == "sparse_anchor"
        and str(spec.get("anchor_mode")) == "sparse_observed"
        and float(spec.get("anchor_interval_seconds", 0.0)) >= 60.0
        and not bool(spec.get("upper_bound", False))
    )


def improvement_pct(horizon: float, val_mae: Any, deployment: bool = False) -> float:
    baseline = DEPLOYMENT_BASELINE.get(float(horizon)) if deployment else BEST_SCORE_BASELINE.get(float(horizon))
    mae = safe_float(val_mae)
    if baseline is None or not math.isfinite(mae):
        return math.nan
    return (float(baseline) - mae) / float(baseline) * 100.0


def run_name(spec: Mapping[str, Any]) -> str:
    context = f"_fms{str(spec.get('fms_context_mode', 'calibration_history')).replace('_', '')}"
    interval = ""
    if spec.get("anchor_mode") == "sparse_observed":
        interval = f"_ai{tag(spec.get('anchor_interval_seconds', 60.0))}"
    static = "static" if spec.get("use_static") else "no_static"
    delta = "_delta" if spec.get("predict_delta_from_anchor") else ""
    mh = "_mh" if spec.get("multi_horizon") else ""
    ph = "_ph" if spec.get("per_horizon_heads") else ""
    loss = str(spec.get("loss_type", "smooth_l1")).replace("_", "")
    extras = [
        f"d{tag(spec.get('d_model', 64))}",
        f"hid{tag(spec.get('hidden_dim', 128))}",
        f"drop{tag(spec.get('dropout', 0.1))}",
        f"loss{loss}",
        f"seed{int(spec.get('seed', 42))}",
    ]
    if safe_float(spec.get("trend_weight"), 0.0) > 0:
        extras.append(f"tw{tag(spec.get('trend_weight'))}")
    if safe_float(spec.get("change_weight"), 0.0) > 0:
        extras.append(f"cw{tag(spec.get('change_weight'))}")
    if safe_float(spec.get("high_target_weight"), 0.0) > 0:
        extras.append(f"hw{tag(spec.get('high_target_weight'))}")
    return (
        f"{spec.get('stage')}_{spec.get('model_family')}_c{tag(spec.get('calibration_seconds', 120))}"
        f"_w{tag(spec.get('recent_window_seconds', 30))}_h{tag(spec.get('horizon_seconds', 5))}"
        f"{context}_{spec.get('anchor_mode', 'none')}{interval}_{static}_{spec.get('model')}{delta}{mh}{ph}_"
        + "_".join(extras)
    )


def complete_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    item = dict(spec)
    item.setdefault("stage", "single")
    item.setdefault("track", "best_score")
    item.setdefault("model_family", item.get("model", "unknown"))
    item.setdefault("calibration_seconds", 120.0)
    item.setdefault("recent_window_seconds", 30.0)
    item.setdefault("horizon_seconds", 5.0)
    item.setdefault("fms_context_mode", "start_only")
    item.setdefault("anchor_mode", "none")
    item.setdefault("anchor_interval_seconds", 60.0)
    item.setdefault("use_static", False)
    item.setdefault("predict_delta_from_anchor", False)
    item.setdefault("loss_type", "smooth_l1")
    item.setdefault("loss_mode", "level_only")
    item.setdefault("trend_weight", 0.0)
    item.setdefault("horizon_loss_weights", None)
    item.setdefault("change_weight", 0.0)
    item.setdefault("high_target_weight", 0.0)
    item.setdefault("high_target_threshold", 0.5)
    item.setdefault("dropout", 0.1)
    item.setdefault("hidden_dim", 128)
    item.setdefault("d_model", 64)
    item.setdefault("learning_rate", None)
    item.setdefault("weight_decay", None)
    item.setdefault("kernel_size", 3)
    item.setdefault("transformer_layers", 1)
    item.setdefault("transformer_heads", 4)
    item.setdefault("transformer_ff_dim", 128)
    item.setdefault("pooling", "mean")
    item.setdefault("branch_dropout", 0.0)
    item.setdefault("anchor_dropout", 0.0)
    item.setdefault("delta_scale", 0.5)
    item.setdefault("multi_horizon", False)
    item.setdefault("horizon_set", None)
    item.setdefault("per_horizon_heads", False)
    item.setdefault("seed", 42)
    mode = str(item.get("fms_context_mode", "start_only"))
    if mode == "sparse_anchor":
        item["anchor_mode"] = "sparse_observed"
        item["anchor_interval_seconds"] = float(item.get("anchor_interval_seconds") or 60.0)
        item["predict_delta_from_anchor"] = bool(item.get("predict_delta_from_anchor", True))
        item["track"] = "sparse_anchor_diagnostic"
    else:
        if mode == "none":
            item["track"] = "motion_only"
        elif mode == "start_only":
            item["track"] = "start_fms_only"
        elif mode == "calibration_history":
            item["track"] = "calibration_history_diagnostic"
        item["anchor_mode"] = "none"
        item["anchor_interval_seconds"] = 0.0
        item["predict_delta_from_anchor"] = False
    item["horizon_seconds"] = float(item["horizon_seconds"])
    item["horizon_steps"] = horizon_steps(float(item["horizon_seconds"]))
    if item.get("horizon_set"):
        item["horizon_set"] = [float(v) for v in item["horizon_set"]]
        item["horizon_set_steps"] = [horizon_steps(v) for v in item["horizon_set"]]
    item["upper_bound"] = False
    item["deployment_realistic"] = is_deployment(item)
    if str(item.get("fms_context_mode")) == "sparse_anchor":
        item["track"] = "sparse_anchor_diagnostic"
    if item.get("multi_horizon"):
        item["track"] = item.get("track") or "multi_horizon_diagnostic"
    item.setdefault("run_name", run_name(item))
    return item


def unique_specs(items: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for raw in items:
        item = complete_spec(raw)
        key = (
            item.get("stage"),
            item.get("model"),
            item.get("model_family"),
            item.get("horizon_seconds"),
            tuple(item.get("horizon_set") or []),
            item.get("fms_context_mode"),
            item.get("calibration_seconds"),
            item.get("recent_window_seconds"),
            item.get("anchor_mode"),
            item.get("anchor_interval_seconds"),
            item.get("use_static"),
            item.get("predict_delta_from_anchor"),
            item.get("loss_type"),
            item.get("loss_mode"),
            item.get("trend_weight"),
            tuple(item.get("horizon_loss_weights") or []),
            item.get("change_weight"),
            item.get("high_target_weight"),
            item.get("dropout"),
            item.get("hidden_dim"),
            item.get("d_model"),
            item.get("learning_rate"),
            item.get("weight_decay"),
            item.get("multi_horizon"),
            item.get("per_horizon_heads"),
            item.get("seed"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def single_base(model: str, family: str, horizon: float, anchor_interval: float, seed: int) -> Dict[str, Any]:
    return {
        "stage": "single",
        "model": model,
        "model_family": family,
        "horizon_seconds": float(horizon),
        "calibration_seconds": 120.0,
        "recent_window_seconds": 30.0,
        "fms_context_mode": "start_only",
        "anchor_mode": "none",
        "anchor_interval_seconds": float(anchor_interval),
        "use_static": False,
        "predict_delta_from_anchor": False,
        "loss_type": "l1",
        "loss_mode": "level_only",
        "dropout": 0.1,
        "hidden_dim": 128,
        "d_model": 64,
        "seed": int(seed),
    }


def context_ablation_specs(model: str, family: str, horizon: float, seed: int) -> List[Dict[str, Any]]:
    base = single_base(model, family, horizon, 0.0, seed)
    base.update({"calibration_seconds": 120.0, "recent_window_seconds": 30.0, "loss_type": "l1"})
    return [
        {**base, "stage": "context_motion_only", "fms_context_mode": "none"},
        {**base, "stage": "context_start_fms_only", "fms_context_mode": "start_only"},
        {**base, "stage": "context_calibration_history", "fms_context_mode": "calibration_history"},
        {
            **base,
            "stage": "context_sparse_anchor_60s",
            "fms_context_mode": "sparse_anchor",
            "anchor_mode": "sparse_observed",
            "anchor_interval_seconds": 60.0,
            "predict_delta_from_anchor": True,
        },
    ]


def ordered_horizons(horizons: Sequence[float], primary: Sequence[float]) -> List[float]:
    values = [float(v) for v in horizons]
    out = [float(v) for v in primary if float(v) in values]
    out.extend(float(v) for v in values if float(v) not in out and float(v) != 1.0)
    if 1.0 in values:
        out.append(1.0)
    return out


def build_initial_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    seed = int(args.seeds[0] if args.seeds else 42)
    horizons = ordered_horizons(args.horizons, args.primary_horizons)
    specs: List[Dict[str, Any]] = []
    for h in horizons:
        primary = float(h) in {float(v) for v in args.primary_horizons}
        specs.extend(context_ablation_specs("lc_sa_tcnformer", "lc_sa_tcnformer", h, seed))
        lc_variants = [
            {"calibration_seconds": 120.0, "recent_window_seconds": 30.0, "anchor_interval_seconds": 10.0, "loss_type": "l1"},
            {"calibration_seconds": 180.0, "recent_window_seconds": 30.0, "anchor_interval_seconds": 10.0, "loss_type": "l1"},
            {"calibration_seconds": 120.0, "recent_window_seconds": 60.0, "anchor_interval_seconds": 10.0, "loss_type": "l1"},
            {"calibration_seconds": 120.0, "recent_window_seconds": 30.0, "anchor_interval_seconds": 30.0, "loss_type": "l1"},
            {"calibration_seconds": 120.0, "recent_window_seconds": 30.0, "anchor_interval_seconds": 10.0, "loss_type": "smooth_l1", "loss_mode": "level_trend_raw", "trend_weight": 0.05},
            {"calibration_seconds": 120.0, "recent_window_seconds": 30.0, "anchor_interval_seconds": 10.0, "loss_type": "l1", "change_weight": 2.0, "high_target_weight": 0.25},
            {"calibration_seconds": 120.0, "recent_window_seconds": 30.0, "anchor_interval_seconds": 10.0, "loss_type": "l1", "d_model": 128, "dropout": 0.05},
        ]
        if primary:
            lc_variants.extend(
                [
                    {"calibration_seconds": 120.0, "recent_window_seconds": 60.0, "anchor_interval_seconds": 60.0, "loss_type": "l1"},
                    {"calibration_seconds": 180.0, "recent_window_seconds": 60.0, "anchor_interval_seconds": 60.0, "loss_type": "smooth_l1"},
                    {"calibration_seconds": 180.0, "recent_window_seconds": 30.0, "anchor_interval_seconds": 90.0, "loss_type": "l1"},
                ]
            )
        for idx, variant in enumerate(lc_variants):
            specs.append({**single_base("lc_sa_tcnformer", "lc_sa_tcnformer", h, variant["anchor_interval_seconds"], seed), **variant, "stage": "single_lc"})

        summary_variants = [
            {"calibration_seconds": 120.0, "recent_window_seconds": 30.0, "anchor_interval_seconds": 60.0, "loss_type": "l1", "d_model": 64, "dropout": 0.1},
            {"calibration_seconds": 180.0, "recent_window_seconds": 60.0, "anchor_interval_seconds": 60.0, "loss_type": "smooth_l1", "d_model": 64, "dropout": 0.2},
            {"calibration_seconds": 120.0, "recent_window_seconds": 60.0, "anchor_interval_seconds": 30.0, "loss_type": "l1", "d_model": 128, "dropout": 0.1},
            {"calibration_seconds": 180.0, "recent_window_seconds": 30.0, "anchor_interval_seconds": 90.0, "loss_type": "smooth_l1", "d_model": 128, "dropout": 0.2},
            {"calibration_seconds": 120.0, "recent_window_seconds": 30.0, "anchor_interval_seconds": 60.0, "loss_type": "smooth_l1", "loss_mode": "level_trend_raw", "trend_weight": 0.05},
            {"calibration_seconds": 120.0, "recent_window_seconds": 60.0, "anchor_interval_seconds": 60.0, "loss_type": "l1", "change_weight": 2.0, "high_target_weight": 0.25},
        ]
        if not primary:
            summary_variants = summary_variants[:3]
        for variant in summary_variants:
            specs.append({**single_base("recent_tcn_summary_calib", "recent_tcn_summary_calib", h, variant["anchor_interval_seconds"], seed), **variant, "stage": "single_summary"})

        if primary:
            specs.append(
                {
                    **single_base("gated_fusion", "gated_fusion", h, 60.0, seed),
                    "stage": "single_gated",
                    "calibration_seconds": 120.0,
                    "recent_window_seconds": 60.0,
                    "branch_dropout": 0.05,
                    "anchor_dropout": 0.05,
                }
            )
    mh_set = [2.5, 5.0, 10.0, 15.0]
    mh_weights = [0.5, 1.0, 1.5, 2.0]
    for model, family, intervals in [
        ("lc_sa_tcnformer", "multi_horizon_lc", [10.0, 60.0]),
        ("recent_tcn_summary_calib", "multi_horizon_summary_tcn", [30.0, 60.0]),
    ]:
        for interval in intervals:
            base = single_base(model, family, 5.0, interval, seed)
            base.update(
                {
                    "stage": "multi_weighted",
                    "track": "multi_horizon_diagnostic",
                    "multi_horizon": True,
                    "horizon_set": mh_set,
                    "horizon_loss_weights": mh_weights,
                    "loss_type": "smooth_l1",
                    "loss_mode": "level_only",
                    "recent_window_seconds": 60.0,
                    "calibration_seconds": 120.0,
                }
            )
            specs.append(base)
            if model == "lc_sa_tcnformer":
                specs.append({**base, "stage": "multi_per_head", "per_horizon_heads": True})
    return unique_specs(specs)


def build_confirmation_specs(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    seeds = [int(s) for s in args.seeds if int(s) != int(args.seeds[0])]
    if not seeds:
        return []
    specs: List[Dict[str, Any]] = []
    for h in [2.5, 5.0, 10.0, 15.0]:
        candidates = [
            r
            for r in sorted_completed(rows)
            if not as_bool(r.get("multi_horizon"))
            and not as_bool(r.get("upper_bound"))
            and str(r.get("fms_context_mode", "start_only")) in {"none", "start_only"}
            and abs(float(r.get("horizon_seconds", -999)) - h) < 1e-6
        ]
        for row in candidates[:2]:
            spec = dict(row.get("spec") or {})
            for seed in seeds:
                specs.append({**spec, "stage": "multiseed_confirm", "seed": seed, "run_name": None})
    for h in [5.0, 10.0, 15.0]:
        candidates = [
            r
            for r in sorted_completed(rows)
            if str(r.get("fms_context_mode", "")) == "sparse_anchor" and abs(float(r.get("horizon_seconds", -999)) - h) < 1e-6
        ]
        if candidates:
            spec = dict(candidates[0].get("spec") or {})
            for seed in seeds:
                specs.append({**spec, "stage": "sparse_anchor_multiseed", "seed": seed, "run_name": None})
    for spec in specs:
        if spec.get("run_name") is None:
            spec.pop("run_name", None)
    return unique_specs(specs)


def common_eval_args(specs: Sequence[Mapping[str, Any]]) -> List[str]:
    max_calib = max(float(s.get("calibration_seconds", 120.0)) for s in specs) if specs else 120.0
    max_h = max(
        max([float(v) for v in (s.get("horizon_set") or [])] or [float(s.get("horizon_seconds", 5.0))])
        for s in specs
    ) if specs else 15.0
    return ["--common_eval_current_start", f"{max_calib:g}", "--common_eval_max_horizon_seconds", f"{max_h:g}"]


def train_cmd(args: argparse.Namespace, spec: Mapping[str, Any], epochs: int, patience: int, all_specs: Sequence[Mapping[str, Any]]) -> List[str]:
    lr = float(spec.get("learning_rate") or args.learning_rate)
    wd = float(spec.get("weight_decay") or args.weight_decay)
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
        args.run_root,
        "--model",
        str(spec["model"]),
        "--run_name",
        str(spec["run_name"]),
        "--split_file",
        args.split_file,
        "--seed",
        str(int(spec.get("seed", args.seeds[0]))),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        f"{lr:g}",
        "--weight_decay",
        f"{wd:g}",
        "--epochs",
        str(int(epochs)),
        "--patience",
        str(int(patience)),
        "--num_workers",
        str(args.num_workers),
        "--loss_type",
        str(spec.get("loss_type", "smooth_l1")),
        "--loss_mode",
        str(spec.get("loss_mode", "level_only")),
        "--trend_weight",
        f"{float(spec.get('trend_weight', 0.0)):g}",
        "--high_fms_threshold",
        "10.0",
        "--calibration_seconds",
        f"{float(spec.get('calibration_seconds', 120.0)):g}",
        "--recent_window_seconds",
        f"{float(spec.get('recent_window_seconds', 30.0)):g}",
        "--horizon_seconds",
        f"{float(spec.get('horizon_seconds', 5.0)):g}",
        "--anchor_mode",
        str(spec.get("anchor_mode", "sparse_observed")),
        "--anchor_interval_seconds",
        f"{float(spec.get('anchor_interval_seconds', 60.0)):g}",
        "--fms_context_mode",
        str(spec.get("fms_context_mode", "start_only")),
        "--d_model",
        str(int(spec.get("d_model", 64))),
        "--hidden_dim",
        str(int(spec.get("hidden_dim", spec.get("d_model", 128)))),
        "--kernel_size",
        str(int(spec.get("kernel_size", 3))),
        "--dropout",
        f"{float(spec.get('dropout', 0.1)):g}",
        "--transformer_layers",
        str(int(spec.get("transformer_layers", 1))),
        "--transformer_heads",
        str(int(spec.get("transformer_heads", 4))),
        "--transformer_ff_dim",
        str(int(spec.get("transformer_ff_dim", 128))),
        "--pooling",
        str(spec.get("pooling", "mean")),
        "--branch_dropout",
        f"{float(spec.get('branch_dropout', 0.0)):g}",
        "--anchor_dropout",
        f"{float(spec.get('anchor_dropout', 0.0)):g}",
        "--delta_scale",
        f"{float(spec.get('delta_scale', 0.5)):g}",
        "--no_test_eval",
        "--no-save_plots",
    ]
    if spec.get("use_static"):
        cmd.extend(["--use_static", "--static_features", "age", "gender", "mssq"])
    else:
        cmd.append("--no_static")
    if spec.get("predict_delta_from_anchor"):
        cmd.append("--predict_delta_from_anchor")
    if spec.get("multi_horizon"):
        cmd.append("--multi_horizon")
        cmd.append("--horizon_set")
        cmd.extend(f"{float(v):g}" for v in spec.get("horizon_set", []))
    if spec.get("per_horizon_heads"):
        cmd.append("--per_horizon_heads")
    if spec.get("horizon_loss_weights"):
        cmd.append("--horizon_loss_weights")
        cmd.extend(f"{float(v):g}" for v in spec["horizon_loss_weights"])
    if safe_float(spec.get("change_weight"), 0.0) > 0:
        cmd.extend(["--change_weight", f"{float(spec.get('change_weight')):g}"])
    if safe_float(spec.get("high_target_weight"), 0.0) > 0:
        cmd.extend(["--high_target_weight", f"{float(spec.get('high_target_weight')):g}", "--high_target_threshold", f"{float(spec.get('high_target_threshold', 0.5)):g}"])
    if spec.get("gru_layers"):
        cmd.extend(["--gru_layers", str(int(spec["gru_layers"]))])
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.device:
        cmd.extend(["--device", args.device])
    if args.smoke_test:
        cmd.extend(["--max_train_batches", str(args.smoke_train_batches), "--max_eval_batches", str(args.smoke_eval_batches)])
    cmd.extend(common_eval_args(all_specs))
    return cmd


def eval_cmd(args: argparse.Namespace, spec: Mapping[str, Any], split: str, all_specs: Sequence[Mapping[str, Any]]) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.densefms_forecast.evaluate",
        "--checkpoint",
        str(Path(args.run_root) / str(spec["run_name"]) / "best.pt"),
        "--data_dir",
        args.data_dir,
        "--split",
        split,
        "--split_file",
        args.split_file,
        "--batch_size",
        str(args.batch_size),
        "--calibration_seconds",
        f"{float(spec.get('calibration_seconds', 120.0)):g}",
        "--recent_window_seconds",
        f"{float(spec.get('recent_window_seconds', 30.0)):g}",
        "--horizon_seconds",
        f"{float(spec.get('horizon_seconds', 5.0)):g}",
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    cmd.extend(common_eval_args(all_specs))
    return cmd


def write_table(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def inspect_hardware(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = Path(args.run_root)
    payload: Dict[str, Any] = {
        "timestamp": now_iso(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_status_short": git_text(["status", "--short"]),
        "data_dir": args.data_dir,
        "data_dir_exists": Path(args.data_dir).exists(),
        "baseline_dir": args.baseline_dir,
        "baseline_dir_exists": Path(args.baseline_dir).exists(),
        "baseline_artifacts": {
            "leaderboard_val.csv": (Path(args.baseline_dir) / "leaderboard_val.csv").exists(),
            "final_selected_models.json": (Path(args.baseline_dir) / "final_selected_models.json").exists(),
            "progress_log.jsonl": (Path(args.baseline_dir) / "progress_log.jsonl").exists(),
        },
    }
    for name, cmd in {"nvidia_smi": ["nvidia-smi"], "free_h": ["free", "-h"], "df_h": ["df", "-h", "."]}.items():
        try:
            payload[name] = subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=20)
        except Exception as exc:
            payload[name] = f"unavailable: {exc}"
    save_json(run_root / "hardware_summary.json", payload)
    return payload


def append_event(run_root: Path, event: str, payload: Mapping[str, Any]) -> None:
    record = {"timestamp": now_iso(), "event": event, **dict(payload)}
    with open(run_root / "progress_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(run_root / "progress_log.md", "a", encoding="utf-8") as f:
        f.write(
            f"- {record['timestamp']} | {event} | {record.get('run_name', '')} | "
            f"H={record.get('horizon_seconds', '')} | model={record.get('model', '')} | "
            f"val_MAE={record.get('val_MAE', '')} | status={record.get('status', '')}\n"
        )


def sorted_completed(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        mae = safe_float(row.get("val_MAE"))
        if math.isfinite(mae) and str(row.get("status", "completed")) in {"completed", "ensemble"}:
            out.append(dict(row))
    return sorted(out, key=lambda r: (safe_float(r.get("val_MAE")), safe_float(r.get("val_RMSE"))))


def summarize_run(args: argparse.Namespace, spec: Mapping[str, Any], event: str = "run_end", runtime_seconds: float = math.nan) -> Dict[str, Any]:
    run_dir = Path(args.run_root) / str(spec["run_name"])
    metrics_path = run_dir / "metrics.json"
    row: Dict[str, Any] = {
        **dict(spec),
        "status": "missing",
        "event": event,
        "spec": dict(spec),
        "anchor_policy": spec.get("anchor_mode"),
        "fms_context_mode": spec.get("fms_context_mode"),
        "delta_prediction": spec.get("predict_delta_from_anchor"),
        "metrics_path": str(metrics_path),
        "checkpoint_path": str(run_dir / "best.pt"),
        "prediction_csv_path": str(run_dir / "val_predictions.csv"),
        "output_dir": str(run_dir),
        "runtime_seconds": runtime_seconds,
    }
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
            "by_horizon": best.get("by_horizon"),
        }
    )
    row["improvement_vs_baseline_pct"] = improvement_pct(
        float(row.get("horizon_seconds", math.nan)),
        row.get("val_MAE"),
        deployment=as_bool(row.get("deployment_realistic")),
    )
    return row


def best_by_horizon(rows: Sequence[Mapping[str, Any]], deployment: bool = False) -> Dict[float, Dict[str, Any]]:
    out: Dict[float, Dict[str, Any]] = {}
    for h in ALL_TARGET_HORIZONS:
        candidates = [
            r
            for r in sorted_completed(rows)
            if not as_bool(r.get("multi_horizon"))
            and not as_bool(r.get("upper_bound"))
            and abs(float(r.get("horizon_seconds", -999)) - h) < 1e-6
            and (
                str(r.get("fms_context_mode", "")) == "sparse_anchor"
                if deployment
                else str(r.get("fms_context_mode", "start_only")) in {"none", "start_only"}
            )
        ]
        if candidates:
            out[h] = dict(candidates[0])
    return out


def write_live_leaderboard(run_root: Path, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted_completed(rows)
    out_rows = []
    for idx, row in enumerate(ranked, 1):
        out = {col: row.get(col, "") for col in LEADERBOARD_COLUMNS}
        out["rank"] = idx
        if isinstance(out.get("horizon_set"), (list, tuple)):
            out["horizon_set"] = " ".join(f"{float(v):g}" for v in out["horizon_set"])
        if isinstance(out.get("horizon_loss_weights"), (list, tuple)):
            out["horizon_loss_weights"] = " ".join(f"{float(v):g}" for v in out["horizon_loss_weights"])
        out_rows.append(out)
    write_table(run_root / "leaderboard_live.csv", out_rows, LEADERBOARD_COLUMNS)
    write_table(run_root / "leaderboard_val.csv", out_rows, LEADERBOARD_COLUMNS)
    lines = ["# DenseFMS Long Horizon Improvement Live Leaderboard", ""]
    lines.append("| " + " | ".join(LEADERBOARD_COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(LEADERBOARD_COLUMNS)) + " |")
    for row in out_rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in LEADERBOARD_COLUMNS) + " |")
    text = "\n".join(lines) + "\n"
    (run_root / "leaderboard_live.md").write_text(text, encoding="utf-8")
    (run_root / "leaderboard_val.md").write_text(text, encoding="utf-8")
    return ranked


def progress_rows(run_root: Path) -> List[Dict[str, Any]]:
    path = run_root / "progress_log.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_partial_summary(
    run_root: Path,
    rows: Sequence[Mapping[str, Any]],
    planned: Sequence[Mapping[str, Any]],
    start_time: float,
    current_run: Optional[str] = None,
    interrupted: bool = False,
) -> None:
    events = progress_rows(run_root)
    completed = sorted_completed(rows)
    failed = [e for e in events if e.get("event") == "run_failed"]
    skipped = [e for e in events if e.get("event") == "run_skipped"]
    completed_names = {r.get("run_name") for r in completed}
    remaining = [p for p in planned if p.get("run_name") not in completed_names]
    best = best_by_horizon(rows, deployment=False)
    deploy = best_by_horizon(rows, deployment=True)
    lines = [
        "# DenseFMS Long Horizon Improvement Partial Summary",
        "",
        f"- timestamp: {now_iso()}",
        f"- elapsed_hours: {(time.time() - start_time) / 3600.0:.3f}",
        f"- interrupted: {interrupted}",
        f"- currently_running_run: {current_run or ''}",
        f"- planned_runs: {len(planned)}",
        f"- completed_runs: {len(completed)}",
        f"- failed_runs: {len(failed)}",
        f"- skipped_existing_runs: {len(skipped)}",
        "",
        "## Best-Score By Horizon",
    ]
    for h in ALL_TARGET_HORIZONS:
        row = best.get(h)
        baseline = BEST_SCORE_BASELINE.get(h)
        if row:
            lines.append(
                f"- H={h:g}: val_MAE={safe_float(row.get('val_MAE')):.4f}, "
                f"baseline={baseline:.4f}, improvement={improvement_pct(h, row.get('val_MAE')):.2f}%, run={row.get('run_name')}"
            )
        else:
            lines.append(f"- H={h:g}: pending, baseline={baseline:.4f}")
    lines.extend(["", "## Sparse-Anchor Diagnostic By Horizon"])
    for h in PRIMARY_HORIZONS:
        row = deploy.get(h)
        baseline = DEPLOYMENT_BASELINE.get(h)
        if row:
            lines.append(
                f"- H={h:g}: val_MAE={safe_float(row.get('val_MAE')):.4f}, "
                f"baseline={baseline:.4f}, improvement={improvement_pct(h, row.get('val_MAE'), deployment=True):.2f}%, run={row.get('run_name')}"
            )
        else:
            lines.append(f"- H={h:g}: pending, baseline={baseline:.4f}")
    lines.extend(["", "## Resume"])
    lines.append(f"- remaining_commands_not_yet_completed: {len(remaining)}")
    for item in remaining[:20]:
        lines.append(f"- {item.get('run_name')}: {command_text(item.get('command', []))}")
    if len(remaining) > 20:
        lines.append(f"- ... {len(remaining) - 20} more")
    text = "\n".join(lines) + "\n"
    (run_root / "partial_summary.md").write_text(text, encoding="utf-8")
    interrupt_path = run_root / "interrupt_summary.md"
    if interrupted or not interrupt_path.exists() or interrupt_path.stat().st_size == 0:
        interrupt_path.write_text(text, encoding="utf-8")


def run_process(cmd: Sequence[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n===== START {time.ctime()} =====\n{command_text(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        rc = proc.wait()
        log.write(f"===== END rc={rc} {time.ctime()} =====\n")
    return int(rc)


def resolve_incomplete_existing(args: argparse.Namespace, spec: Mapping[str, Any]) -> Dict[str, Any]:
    item = dict(spec)
    run_root = Path(args.run_root)
    run_dir = run_root / str(item["run_name"])
    if not run_dir.exists() or is_complete_run(run_dir):
        return item
    idx = 1
    base = str(item["run_name"])
    while (run_root / f"{base}_rerun{idx}").exists():
        idx += 1
    item["run_name"] = f"{base}_rerun{idx}"
    item["rerun_of_incomplete"] = base
    return complete_spec(item)


def execute_spec(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    epochs: int,
    patience: int,
    all_specs: Sequence[Mapping[str, Any]],
    rows: List[Dict[str, Any]],
    planned: Sequence[Mapping[str, Any]],
    start_time: float,
) -> Optional[Dict[str, Any]]:
    global CURRENT_RUN, STOP_REQUESTED
    run_root = Path(args.run_root)
    item = resolve_incomplete_existing(args, spec) if args.skip_existing else dict(spec)
    run_dir = run_root / str(item["run_name"])
    if args.skip_existing and is_complete_run(run_dir):
        row = summarize_run(args, item, event="run_skipped", runtime_seconds=0.0)
        row["event"] = "run_skipped"
        rows.append(row)
        append_event(run_root, "run_skipped", {"run_name": item["run_name"], "status": "completed_existing", "horizon_seconds": item.get("horizon_seconds"), "model": item.get("model")})
        write_live_leaderboard(run_root, rows)
        write_partial_summary(run_root, rows, planned, start_time, CURRENT_RUN)
        return row

    cmd = train_cmd(args, item, epochs, patience, all_specs)
    item["command"] = cmd
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "run_spec.json", item)
    save_json(run_dir / "command.json", {"train": cmd})
    (run_dir / "command.txt").write_text(command_text(cmd) + "\n", encoding="utf-8")
    CURRENT_RUN = str(item["run_name"])
    append_event(
        run_root,
        "run_start",
        {
            "run_name": item["run_name"],
            "command": cmd,
            "horizon_seconds": item.get("horizon_seconds"),
            "horizon_steps": item.get("horizon_steps"),
            "model": item.get("model"),
            "model_family": item.get("model_family"),
            "seed": item.get("seed"),
            "output_dir": str(run_dir),
            "status": "started",
        },
    )
    write_partial_summary(run_root, rows, planned, start_time, CURRENT_RUN)
    t0 = time.time()
    try:
        rc = run_process(cmd, ROOT, run_dir / "train.log")
        runtime = time.time() - t0
        if rc != 0:
            save_json(run_dir / "status.json", {"status": "failed", "returncode": rc, "finished_at": time.time()})
            append_event(
                run_root,
                "run_failed",
                {
                    "run_name": item["run_name"],
                    "command": cmd,
                    "return_code": rc,
                    "runtime_seconds": runtime,
                    "output_dir": str(run_dir),
                    "status": "failed",
                    "horizon_seconds": item.get("horizon_seconds"),
                    "model": item.get("model"),
                },
            )
            write_partial_summary(run_root, rows, planned, start_time, None)
            return None
        row = summarize_run(args, item, event="run_end", runtime_seconds=runtime)
        rows.append(row)
        append_event(
            run_root,
            "run_end",
            {
                "run_name": item["run_name"],
                "horizon_seconds": item.get("horizon_seconds"),
                "horizon_steps": item.get("horizon_steps"),
                "model": item.get("model"),
                "seed": item.get("seed"),
                "best_epoch": row.get("best_epoch"),
                "val_MAE": row.get("val_MAE"),
                "val_RMSE": row.get("val_RMSE"),
                "runtime_seconds": runtime,
                "checkpoint_path": row.get("checkpoint_path"),
                "metrics_path": row.get("metrics_path"),
                "prediction_path": row.get("prediction_csv_path"),
                "status": row.get("status"),
            },
        )
        write_live_leaderboard(run_root, rows)
        write_partial_summary(run_root, rows, planned, start_time, None)
        return row
    except Exception as exc:
        runtime = time.time() - t0
        append_event(
            run_root,
            "run_failed",
            {
                "run_name": item["run_name"],
                "command": cmd,
                "exception": repr(exc),
                "runtime_seconds": runtime,
                "output_dir": str(run_dir),
                "status": "failed",
                "horizon_seconds": item.get("horizon_seconds"),
                "model": item.get("model"),
            },
        )
        write_partial_summary(run_root, rows, planned, start_time, None)
        return None
    finally:
        CURRENT_RUN = None
        if STOP_REQUESTED:
            write_partial_summary(run_root, rows, planned, start_time, None, interrupted=True)


def ensemble_key_columns(df: pd.DataFrame) -> List[str]:
    candidates = ["split", "session_id", "source_file", "current_index", "target_index", "horizon_seconds"]
    return [c for c in candidates if c in df.columns]


def make_ensemble_for_horizon(args: argparse.Namespace, horizon: float, rows: Sequence[Mapping[str, Any]], run_root: Path) -> Optional[Dict[str, Any]]:
    candidates = [
        r
        for r in sorted_completed(rows)
        if not as_bool(r.get("multi_horizon"))
        and not as_bool(r.get("upper_bound"))
        and abs(float(r.get("horizon_seconds", -999)) - float(horizon)) < 1e-6
        and Path(str(r.get("prediction_csv_path", ""))).exists()
    ][:3]
    if len(candidates) < 2:
        return None
    frames = []
    for idx, row in enumerate(candidates):
        df = pd.read_csv(row["prediction_csv_path"])
        if df.empty:
            return None
        keys = ensemble_key_columns(df)
        keep = keys + ["target_fms", "predicted_fms"]
        frames.append(df[keep].rename(columns={"predicted_fms": f"pred_{idx}", "target_fms": f"target_{idx}"}))
    merged = frames[0]
    keys = ensemble_key_columns(merged)
    for frame in frames[1:]:
        merged = merged.merge(frame, on=keys, how="inner")
    if merged.empty:
        return None
    pred_cols = [c for c in merged.columns if c.startswith("pred_")]
    target_cols = [c for c in merged.columns if c.startswith("target_")]
    merged["target_fms"] = merged[target_cols[0]]
    merged["predicted_fms"] = merged[pred_cols].mean(axis=1)
    merged["absolute_error"] = (merged["predicted_fms"] - merged["target_fms"]).abs()
    merged["squared_error"] = (merged["predicted_fms"] - merged["target_fms"]) ** 2
    metrics = compute_regression_metrics(merged["target_fms"].to_numpy(), merged["predicted_fms"].to_numpy())
    best_single = safe_float(candidates[0].get("val_MAE"))
    run_name_value = f"ensemble_h{tag(horizon)}_top{len(candidates)}_validation_selected"
    out_dir = run_root / run_name_value
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "val_predictions.csv"
    merged.to_csv(pred_path, index=False)
    save_json(
        out_dir / "metrics.json",
        {
            "metrics": {"best_val_metrics": metrics, "best_epoch": 0, "parameter_count": 0, "val_metrics": metrics},
            "members": [r.get("run_name") for r in candidates],
            "selection": "top validation MAE members only; no test metrics used",
        },
    )
    row: Dict[str, Any] = {
        "run_name": run_name_value,
        "status": "ensemble",
        "event": "ensemble_val",
        "stage": "ensemble",
        "track": "ensemble_diagnostic",
        "model_family": "validation_ensemble",
        "model": "mean_ensemble",
        "horizon_seconds": float(horizon),
        "horizon_steps": horizon_steps(float(horizon)),
        "horizon_set": "",
        "calibration_seconds": "",
        "recent_window_seconds": "",
        "anchor_policy": "member_specific",
        "anchor_mode": "member_specific",
        "anchor_interval_seconds": "",
        "use_static": "",
        "loss_type": "",
        "loss_mode": "",
        "trend_weight": "",
        "horizon_loss_weights": "",
        "change_weight": "",
        "high_target_weight": "",
        "delta_prediction": "",
        "multi_horizon": False,
        "per_horizon_heads": False,
        "seed": "",
        "best_epoch": 0,
        "val_MAE": metrics.get("mae"),
        "val_RMSE": metrics.get("rmse"),
        "val_R2": metrics.get("r2"),
        "common_val_MAE": "",
        "runtime_seconds": 0.0,
        "parameter_count": 0,
        "deployment_realistic": False,
        "upper_bound": False,
        "improvement_vs_baseline_pct": improvement_pct(float(horizon), metrics.get("mae")),
        "metrics_path": str(out_dir / "metrics.json"),
        "checkpoint_path": "",
        "prediction_csv_path": str(pred_path),
        "output_dir": str(out_dir),
        "members": [dict(r) for r in candidates],
        "ensemble_improved_best_single": safe_float(metrics.get("mae")) < best_single,
    }
    return row


def create_validation_ensembles(args: argparse.Namespace, rows: List[Dict[str, Any]], planned: Sequence[Mapping[str, Any]], start_time: float) -> List[Dict[str, Any]]:
    run_root = Path(args.run_root)
    ensemble_rows = []
    for h in PRIMARY_HORIZONS:
        row = make_ensemble_for_horizon(args, h, rows, run_root)
        if not row:
            continue
        rows.append(row)
        ensemble_rows.append(row)
        append_event(
            run_root,
            "ensemble_val",
            {
                "run_name": row["run_name"],
                "horizon_seconds": h,
                "val_MAE": row.get("val_MAE"),
                "val_RMSE": row.get("val_RMSE"),
                "status": "ensemble",
                "members": [m.get("run_name") for m in row.get("members", [])],
            },
        )
    write_live_leaderboard(run_root, rows)
    write_partial_summary(run_root, rows, planned, start_time, None)
    return ensemble_rows


def select_final_roles(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    used_role = set()

    def add(role: str, row: Optional[Mapping[str, Any]], rationale: str) -> None:
        if row is None:
            return
        item = dict(row)
        item["selection_role"] = role
        item["selection_rationale"] = rationale
        selected.append(item)
        used_role.add(role)

    for h in ALL_TARGET_HORIZONS:
        candidates = [
            r
            for r in sorted_completed(rows)
            if not as_bool(r.get("multi_horizon"))
            and not as_bool(r.get("upper_bound"))
            and str(r.get("model_family")) != "validation_ensemble"
            and str(r.get("fms_context_mode", "start_only")) in {"none", "start_only"}
            and abs(float(r.get("horizon_seconds", -999)) - h) < 1e-6
        ]
        add(f"best_score_h{tag(h)}", candidates[0] if candidates else None, f"lowest validation MAE for H={h:g}")
    for h in PRIMARY_HORIZONS:
        candidates = [
            r
            for r in sorted_completed(rows)
            if str(r.get("fms_context_mode", "")) == "sparse_anchor" and abs(float(r.get("horizon_seconds", -999)) - h) < 1e-6
        ]
        add(f"sparse_anchor_diag_h{tag(h)}", candidates[0] if candidates else None, f"lowest sparse-anchor diagnostic validation MAE for H={h:g}")
    mh = [
        r
        for r in sorted_completed(rows)
        if as_bool(r.get("multi_horizon"))
        and not as_bool(r.get("upper_bound"))
        and str(r.get("fms_context_mode", "start_only")) in {"none", "start_only"}
    ]
    add("multi_horizon_diagnostic", mh[0] if mh else None, "lowest validation aggregate MAE among multi-horizon candidates")
    ens = [r for r in sorted_completed(rows) if str(r.get("model_family")) == "validation_ensemble"]
    improved = [r for r in ens if as_bool(r.get("ensemble_improved_best_single"))]
    add("ensemble_diagnostic", (improved or ens)[0] if ens else None, "validation-selected mean ensemble diagnostic")
    return selected


def run_final_test_for_model(args: argparse.Namespace, row: Mapping[str, Any], all_specs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    spec = row.get("spec") or row
    run_name_value = str(row["run_name"])
    rc = run_process(eval_cmd(args, spec, "test", all_specs), ROOT, Path(args.run_root) / run_name_value / "eval_test.log")
    if rc != 0:
        raise RuntimeError(f"Final test evaluation failed for {run_name_value}")
    metrics_path = Path(args.run_root) / run_name_value / "eval_test" / "metrics.json"
    metrics = load_json(metrics_path).get("metrics", {})
    return {
        "test_metrics": metrics,
        "test_prediction_csv": str(Path(args.run_root) / run_name_value / "eval_test" / "test_predictions.csv"),
    }


def run_final_test_for_ensemble(args: argparse.Namespace, row: Mapping[str, Any], all_specs: Sequence[Mapping[str, Any]], cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    members = row.get("members") or []
    frames = []
    for idx, member in enumerate(members):
        member_name = str(member["run_name"])
        if member_name not in cache:
            cache[member_name] = run_final_test_for_model(args, member, all_specs)
        pred_path = Path(cache[member_name]["test_prediction_csv"])
        df = pd.read_csv(pred_path)
        keys = ensemble_key_columns(df)
        keep = keys + ["target_fms", "predicted_fms"]
        frames.append(df[keep].rename(columns={"predicted_fms": f"pred_{idx}", "target_fms": f"target_{idx}"}))
    merged = frames[0]
    keys = ensemble_key_columns(merged)
    for frame in frames[1:]:
        merged = merged.merge(frame, on=keys, how="inner")
    pred_cols = [c for c in merged.columns if c.startswith("pred_")]
    target_cols = [c for c in merged.columns if c.startswith("target_")]
    merged["target_fms"] = merged[target_cols[0]]
    merged["predicted_fms"] = merged[pred_cols].mean(axis=1)
    merged["absolute_error"] = (merged["predicted_fms"] - merged["target_fms"]).abs()
    merged["squared_error"] = (merged["predicted_fms"] - merged["target_fms"]) ** 2
    metrics = compute_regression_metrics(merged["target_fms"].to_numpy(), merged["predicted_fms"].to_numpy())
    out_dir = Path(args.run_root) / str(row["run_name"]) / "eval_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "test_predictions.csv"
    merged.to_csv(pred_path, index=False)
    save_json(out_dir / "metrics.json", {"metrics": metrics, "members": [m.get("run_name") for m in members]})
    return {"test_metrics": metrics, "test_prediction_csv": str(pred_path)}


def run_final_tests(args: argparse.Namespace, selected: Sequence[Mapping[str, Any]], all_specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if args.no_test_eval:
        return []
    run_root = Path(args.run_root)
    cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    for row in selected:
        role = str(row.get("selection_role"))
        run_name_value = str(row["run_name"])
        if str(row.get("model_family")) == "validation_ensemble":
            result = run_final_test_for_ensemble(args, row, all_specs, cache)
        else:
            if run_name_value not in cache:
                cache[run_name_value] = run_final_test_for_model(args, row, all_specs)
            result = cache[run_name_value]
        metrics = result["test_metrics"]
        rows.append(
            {
                "role_name": role,
                "run_name": run_name_value,
                "model_family": row.get("model_family"),
                "model": row.get("model"),
                "horizon_seconds": row.get("horizon_seconds"),
                "horizon_set": " ".join(f"{float(v):g}" for v in row.get("horizon_set", []) or []),
                "track": row.get("track"),
                "selection_rationale": row.get("selection_rationale"),
                "val_MAE": row.get("val_MAE"),
                "val_RMSE": row.get("val_RMSE"),
                "test_MAE": metrics.get("mae"),
                "test_RMSE": metrics.get("rmse"),
                "test_R2": metrics.get("r2"),
                "test_sMAPE": metrics.get("smape"),
                "common_test_MAE": metrics.get("common_mae"),
                "common_test_RMSE": metrics.get("common_rmse"),
                "checkpoint_path": row.get("checkpoint_path"),
                "prediction_csv_path": result["test_prediction_csv"],
            }
        )
        append_event(run_root, "final_test", {"run_name": run_name_value, "role_name": role, "test_MAE": metrics.get("mae"), "test_RMSE": metrics.get("rmse"), "status": "completed"})
    write_table(run_root / "final_test_metrics.csv", rows, FINAL_TEST_COLUMNS)
    return rows


def write_plots(run_root: Path, rows: Sequence[Mapping[str, Any]]) -> List[str]:
    plot_dir = run_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    completed = sorted_completed(rows)
    df = pd.DataFrame(completed)
    made: List[str] = []

    def save_current(name: str) -> None:
        plt.tight_layout()
        path = plot_dir / name
        plt.savefig(path, dpi=150)
        plt.close()
        made.append(str(path))

    best = best_by_horizon(rows, deployment=False)
    hs = [h for h in ALL_TARGET_HORIZONS if h in best]
    if hs:
        plt.figure(figsize=(7, 4))
        new = [safe_float(best[h].get("val_MAE")) for h in hs]
        base = [BEST_SCORE_BASELINE[h] for h in hs]
        x = np.arange(len(hs))
        plt.bar(x - 0.18, base, width=0.36, label="previous")
        plt.bar(x + 0.18, new, width=0.36, label="new")
        plt.xticks(x, [f"H={h:g}" for h in hs])
        plt.ylabel("Validation MAE")
        plt.legend()
        save_current("best_by_horizon_improvement.png")

        plt.figure(figsize=(7, 4))
        plt.plot(hs, base, marker="o", label="previous")
        plt.plot(hs, new, marker="o", label="new")
        plt.xlabel("Horizon seconds")
        plt.ylabel("Validation MAE")
        plt.legend()
        save_current("horizon_mae_curve_previous_vs_new.png")

    deploy = best_by_horizon(rows, deployment=True)
    dhs = [h for h in PRIMARY_HORIZONS if h in deploy]
    if dhs:
        plt.figure(figsize=(7, 4))
        plt.plot(dhs, [DEPLOYMENT_BASELINE[h] for h in dhs], marker="o", label="previous sparse-anchor diagnostic")
        plt.plot(dhs, [safe_float(deploy[h].get("val_MAE")) for h in dhs], marker="o", label="new sparse-anchor diagnostic")
        plt.xlabel("Horizon seconds")
        plt.ylabel("Validation MAE")
        plt.legend()
        save_current("sparse_anchor_diagnostic_horizon_mae_curve_previous_vs_new.png")

    events = progress_rows(run_root)
    vals = []
    cur = math.inf
    for event in events:
        if event.get("event") not in {"run_end", "run_skipped", "ensemble_val"}:
            continue
        mae = safe_float(event.get("val_MAE"))
        if math.isfinite(mae):
            cur = min(cur, mae)
        if math.isfinite(cur):
            vals.append(cur)
    if vals:
        plt.figure(figsize=(7, 4))
        plt.plot(np.arange(1, len(vals) + 1), vals)
        plt.xlabel("Completed validation results")
        plt.ylabel("Best validation MAE so far")
        save_current("progress_best_mae_over_time.png")

    if not df.empty and {"model_family", "val_MAE", "horizon_seconds"}.issubset(df.columns):
        sub = df[df["horizon_seconds"].astype(float).isin(PRIMARY_HORIZONS)].copy()
        if not sub.empty:
            pivot = sub.groupby(["model_family", "horizon_seconds"])["val_MAE"].min().unstack()
            pivot.plot(kind="bar", figsize=(9, 4))
            plt.ylabel("Best validation MAE")
            save_current("model_family_comparison_h5_h10_h15.png")

    for h in ALL_TARGET_HORIZONS:
        row = best.get(h)
        if not row:
            continue
        pred_path = Path(str(row.get("prediction_csv_path", "")))
        if pred_path.exists():
            dfp = pd.read_csv(pred_path)
            if not dfp.empty and {"target_fms", "predicted_fms"}.issubset(dfp.columns):
                plt.figure(figsize=(5, 5))
                plt.scatter(dfp["target_fms"], dfp["predicted_fms"], s=4, alpha=0.25)
                lo = min(float(dfp["target_fms"].min()), float(dfp["predicted_fms"].min()))
                hi = max(float(dfp["target_fms"].max()), float(dfp["predicted_fms"].max()))
                plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
                plt.xlabel("Target FMS")
                plt.ylabel("Predicted FMS")
                save_current(f"val_predicted_vs_target_h{tag(h)}.png")
                if h in PRIMARY_HORIZONS:
                    residual = dfp["predicted_fms"] - dfp["target_fms"]
                    plt.figure(figsize=(7, 4))
                    plt.hist(residual, bins=50)
                    plt.xlabel("Prediction residual")
                    plt.ylabel("Count")
                    save_current(f"residual_histogram_h{tag(h)}.png")
    return made


def write_final_report(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    final_tests: Sequence[Mapping[str, Any]],
    plots: Sequence[str],
    hardware: Mapping[str, Any],
    start_time: float,
) -> None:
    run_root = Path(args.run_root)
    completed = sorted_completed(rows)
    failed = [e for e in progress_rows(run_root) if e.get("event") == "run_failed"]
    best = best_by_horizon(rows, deployment=False)
    deploy = best_by_horizon(rows, deployment=True)
    status = git_text(["status", "--short"])
    lines = [
        "# DenseFMS 장기 Horizon 개선 최종 보고서",
        "",
        "## 1. 작업 요약",
        f"- 새 run root: `{args.run_root}`",
        "- 모델 선택은 validation MAE/RMSE만 사용했고 test는 final selected role 확정 후에만 평가했다.",
        "- H=1은 optimization target에 포함하지 않았다.",
        "",
        "## 2. 사용한 baseline",
        f"- baseline run root: `{args.baseline_dir}`",
        "- best-score baseline: H=2.5 1.0492, H=5 1.2870, H=10 1.6261, H=15 1.8481",
        "- sparse-anchor-assisted diagnostic baseline: H=5 1.7735, H=10 1.9199, H=15 2.0376",
        "",
        "## 3. 변경/추가한 파일",
        "- `scripts/run_densefms_long_horizon_improvement.py` 추가",
        "- `src/densefms_forecast/losses.py` multi-horizon 가중 손실 및 change-aware weighting 옵션 추가",
        "- `src/densefms_forecast/train.py` 새 loss/model/FMS-context CLI 옵션 연결",
        "- `src/densefms_forecast/model.py` LC-SA-TCNFormer multi-horizon per-horizon head 및 FMS context mode 옵션 추가",
        "",
        "## 4. 새 CLI/config 옵션",
        "- runner: `--run_root`, `--baseline_dir`, `--horizons`, `--primary_horizons`, `--include_h1_diagnostic`, `--skip_existing`, `--dry_run`, `--smoke_test`, `--max_runs`, `--seeds`",
        "- train: `--fms_context_mode`, `--horizon_loss_weights`, `--change_weight`, `--high_target_weight`, `--high_target_threshold`, `--per_horizon_heads`",
        "",
        "## 5. Dataset/windowing 변경",
        "- 기존 dataset split/windowing 로직을 유지했다.",
        "- target은 기존 `FMS[t + horizon_steps]` shift를 사용한다.",
        "- 메인 track은 `fms_context_mode=start_only`로 각 recent window 시작 FMS scalar 1개를 사용한다.",
        "- `calibration_history`는 calibration FMS만 쓰는 diagnostic, `sparse_anchor`는 추가 anchor-assisted diagnostic으로 분리한다.",
        "- sparse anchor index는 diagnostic track에서만 current index 이하로 clamp된다.",
        "",
        "## 6. Model 변경",
        "- LC-SA-TCNFormer: multi-horizon에서 optional per-horizon prediction head 지원.",
        "- RecentTCN+SummaryCalib/LC-SA-TCNFormer를 우선 탐색했다.",
        "- delta/residual target은 `sparse_anchor` diagnostic 후보에서만 사용했다.",
        "",
        "## 7. Anchor/static/multi-horizon 지원 상태",
        "- main track은 static feature를 기본으로 사용하지 않는다.",
        "- sparse observed anchor는 `fms_context_mode=sparse_anchor`, 60s anchor-assisted diagnostic으로만 분리했다.",
        "- weighted multi-horizon은 H=2.5/5/10/15 및 weights 0.5/1.0/1.5/2.0으로 수행했다.",
        "- per-horizon heads는 LC-SA multi-horizon diagnostic에서 수행했다.",
        "- trend/change auxiliary 및 change-aware weighting 후보를 수행했다.",
        "",
        "## 8. Sanity test 결과",
        "- `sanity_tests.log` 참조. dry run/smoke/full 순서로 실행했다.",
        "",
        "## 9. Search budget 실제 사용량",
        f"- elapsed_hours: {(time.time() - start_time) / 3600.0:.3f}",
        f"- completed validation results: {len(completed)}",
        f"- failed runs: {len(failed)}",
        f"- wall-clock target/soft/hard: {args.wall_clock_target_hours}/{args.wall_clock_soft_cap_hours}/{args.wall_clock_hard_cap_hours} hours",
        f"- GPU: {hardware.get('gpu')}, CUDA available: {hardware.get('cuda_available')}",
        "",
        "## 10. Validation leaderboard",
        "- 전체 live leaderboard: `leaderboard_live.csv`, `leaderboard_live.md`",
    ]
    for h in ALL_TARGET_HORIZONS:
        row = best.get(h)
        baseline = BEST_SCORE_BASELINE[h]
        if row:
            mae = safe_float(row.get("val_MAE"))
            lines.append(f"- H={h:g}: best val_MAE={mae:.4f}, baseline={baseline:.4f}, relative_improvement={improvement_pct(h, mae):.2f}%, run={row.get('run_name')}")
        else:
            lines.append(f"- H={h:g}: no completed candidate, baseline={baseline:.4f}")
    lines.extend(["", "## 11. Sparse-anchor-assisted diagnostic leaderboard"])
    lines.append("- 이 track은 sparse FMS anchor를 추가로 쓰는 진단 조건이며 window-start-FMS 결과와 섞어 해석하지 않는다.")
    for h in PRIMARY_HORIZONS:
        row = deploy.get(h)
        baseline = DEPLOYMENT_BASELINE[h]
        if row:
            mae = safe_float(row.get("val_MAE"))
            lines.append(f"- H={h:g}: sparse-anchor diagnostic val_MAE={mae:.4f}, baseline={baseline:.4f}, relative_improvement={improvement_pct(h, mae, deployment=True):.2f}%, run={row.get('run_name')}")
        else:
            lines.append(f"- H={h:g}: no completed sparse-anchor diagnostic candidate, baseline={baseline:.4f}")
    lines.extend(["", "## 12. Stretch target 달성 여부"])
    for h, target in STRETCH_TARGETS.items():
        row = best.get(h)
        mae = safe_float(row.get("val_MAE")) if row else math.nan
        lines.append(f"- H={h:g}: target <= {target:.4f}, best={mae if math.isfinite(mae) else 'pending'}, reached={bool(math.isfinite(mae) and mae <= target)}")
    for h, target in DEPLOYMENT_TARGETS.items():
        row = deploy.get(h)
        mae = safe_float(row.get("val_MAE")) if row else math.nan
        lines.append(f"- sparse-anchor diagnostic H={h:g}: target <= {target:.4f}, best={mae if math.isfinite(mae) else 'pending'}, reached={bool(math.isfinite(mae) and mae <= target)}")
    lines.extend(["", "## 13. Multi-horizon 결과"])
    mh = [r for r in sorted_completed(rows) if as_bool(r.get("multi_horizon"))]
    if mh:
        for row in mh[:8]:
            lines.append(f"- {row.get('run_name')}: aggregate val_MAE={row.get('val_MAE')} horizon_set={row.get('horizon_set')} per_head={row.get('per_horizon_heads')}")
    else:
        lines.append("- completed multi-horizon result 없음")
    lines.extend(["", "## 14. Ensemble 결과"])
    ens = [r for r in sorted_completed(rows) if str(r.get("model_family")) == "validation_ensemble"]
    if ens:
        for row in ens:
            lines.append(f"- {row.get('run_name')}: val_MAE={row.get('val_MAE')} improved_best_single={row.get('ensemble_improved_best_single')}")
    else:
        lines.append("- validation ensemble 생성 결과 없음")
    lines.extend(["", "## 15. Final selected configuration"])
    for row in selected:
        lines.append(f"- {row.get('selection_role')}: {row.get('run_name')} | val_MAE={row.get('val_MAE')} | rationale={row.get('selection_rationale')}")
    lines.extend(["", "## 16. Final test-set metrics"])
    if final_tests:
        for row in final_tests:
            lines.append(f"- {row.get('role_name')}: test_MAE={row.get('test_MAE')} test_RMSE={row.get('test_RMSE')} test_R2={row.get('test_R2')}")
    else:
        lines.append("- final test evaluation skipped or unavailable")
    lines.extend(["", "## 17. Generated plots/tables"])
    for path in plots:
        lines.append(f"- `{path}`")
    for path in ["planned_manifest.json", "progress_log.jsonl", "progress_log.md", "leaderboard_live.csv", "leaderboard_live.md", "partial_summary.md", "interrupt_summary.md", "final_selected_models.json", "final_test_metrics.csv"]:
        lines.append(f"- `{run_root / path}`")
    lines.extend(
        [
            "",
            "## 18. 해석",
            "- validation 개선과 final test 결과를 분리해서 해석해야 한다.",
            "- H=1 결과는 장기 horizon 성공 근거로 사용하지 않았다.",
            "- window-start-FMS track과 sparse-anchor-assisted diagnostic track은 분리해서 선택했다.",
            "- sparse-anchor-assisted diagnostic track은 추가 sparse FMS anchor를 요구하므로 window-start-FMS 또는 passive motion-only 결과로 해석하지 않는다.",
            "- smoke-test metric은 실제 성능으로 해석하지 않는다.",
            "",
            "## 19. 남은 이슈 또는 경고",
            "- stretch target 미달 항목은 위 target 표에 그대로 표시했다.",
            "- ensemble test는 validation-selected ensemble role 산출을 위해 member prediction을 생성하며, member 선택에는 test를 사용하지 않는다.",
            "",
            "## 20. git status --short",
        ]
    )
    lines.extend(status.splitlines() or ["(clean)"])
    (run_root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(args: argparse.Namespace, specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    run_root = Path(args.run_root)
    complete = []
    for spec in specs:
        item = dict(spec)
        item["output_dir"] = str(run_root / str(item["run_name"]))
        item["command"] = train_cmd(args, item, args.max_epochs, args.patience, specs)
        item["horizon_steps"] = horizon_steps(float(item.get("horizon_seconds", 5.0)))
        complete.append(item)
    payload = {
        "created_at": now_iso(),
        "run_root": args.run_root,
        "baseline_dir": args.baseline_dir,
        "sampling_interval": SAMPLING_INTERVAL,
        "wall_clock_target_hours": args.wall_clock_target_hours,
        "wall_clock_soft_cap_hours": args.wall_clock_soft_cap_hours,
        "wall_clock_hard_cap_hours": args.wall_clock_hard_cap_hours,
        "runs": complete,
    }
    save_json(run_root / "planned_manifest.json", payload)
    return complete


def handle_signal(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    try:
        run_root = Path(RUN_ROOT)
        append_event(run_root, "interrupt", {"signal": signum, "currently_running_run": CURRENT_RUN, "status": "interrupted"})
    except Exception:
        pass


def initialize_run_root(args: argparse.Namespace) -> Mapping[str, Any]:
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    if not (run_root / "progress_log.md").exists():
        (run_root / "progress_log.md").write_text(f"# DenseFMS long-horizon improvement progress\n\nStarted: {now_iso()}\n\n", encoding="utf-8")
    else:
        with open(run_root / "progress_log.md", "a", encoding="utf-8") as f:
            f.write(f"\n## Runner restart {now_iso()}\n\n")
    for name in ["progress_log.jsonl", "leaderboard_live.csv", "leaderboard_live.md", "partial_summary.md", "interrupt_summary.md"]:
        path = run_root / name
        if not path.exists():
            path.write_text("", encoding="utf-8")
    hardware = inspect_hardware(args)
    append_event(run_root, "runner_start", {"run_root": args.run_root, "baseline_dir": args.baseline_dir, "status": "started"})
    return hardware


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DenseFMS long-horizon improvement search.")
    p.add_argument("--data_dir", default="./DenseFMS/Dataset")
    p.add_argument("--run_root", default=RUN_ROOT)
    p.add_argument("--baseline_dir", default=DEFAULT_BASELINE_DIR)
    p.add_argument("--split_file", default=DEFAULT_SPLIT)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--horizons", nargs="+", type=float, default=[2.5, 5.0, 10.0, 15.0])
    p.add_argument("--primary_horizons", nargs="+", type=float, default=[5.0, 10.0, 15.0])
    p.add_argument("--include_h1_diagnostic", action="store_true")
    p.add_argument("--wall_clock_target_hours", type=float, default=8.0)
    p.add_argument("--wall_clock_soft_cap_hours", type=float, default=11.0)
    p.add_argument("--wall_clock_hard_cap_hours", type=float, default=12.0)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--max_runs", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=70)
    p.add_argument("--patience", type=int, default=9)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--smoke_epochs", type=int, default=2)
    p.add_argument("--smoke_patience", type=int, default=1)
    p.add_argument("--smoke_train_batches", type=int, default=2)
    p.add_argument("--smoke_eval_batches", type=int, default=2)
    p.add_argument("--no_test_eval", action="store_true")
    args = p.parse_args()
    if args.include_h1_diagnostic and 1.0 not in args.horizons:
        args.horizons.append(1.0)
    if args.smoke_test and args.max_runs is None:
        args.max_runs = 2
    return args


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    start_time = time.time()
    hardware = initialize_run_root(args)
    run_root = Path(args.run_root)
    specs = build_initial_specs(args)
    if args.smoke_test:
        smoke_specs = []
        for spec in specs:
            item = dict(spec)
            item["stage"] = f"smoke_{item.get('stage', 'run')}"
            item.pop("run_name", None)
            smoke_specs.append(complete_spec(item))
        specs = smoke_specs
    if args.max_runs is not None:
        specs = specs[: int(args.max_runs)]
    planned = write_manifest(args, specs)
    write_live_leaderboard(run_root, [])
    write_partial_summary(run_root, [], planned, start_time, None)
    if args.dry_run:
        append_event(run_root, "dry_run", {"planned_runs": len(planned), "status": "completed"})
        print(f"Dry run wrote manifest with {len(planned)} candidates to {run_root / 'planned_manifest.json'}")
        return

    rows: List[Dict[str, Any]] = []
    epochs = args.smoke_epochs if args.smoke_test else args.max_epochs
    patience = args.smoke_patience if args.smoke_test else args.patience
    for spec in planned:
        if STOP_REQUESTED:
            break
        if (time.time() - start_time) / 3600.0 >= float(args.wall_clock_hard_cap_hours):
            append_event(run_root, "hard_cap_stop", {"elapsed_hours": (time.time() - start_time) / 3600.0, "status": "stopped"})
            break
        execute_spec(args, spec, epochs, patience, planned, rows, planned, start_time)

    if not args.smoke_test and not STOP_REQUESTED:
        confirm = build_confirmation_specs(rows, args)
        if confirm:
            confirm_planned = write_manifest(args, [*planned, *confirm])
            for spec in confirm:
                if STOP_REQUESTED or (time.time() - start_time) / 3600.0 >= float(args.wall_clock_hard_cap_hours):
                    break
                execute_spec(args, spec, min(args.max_epochs + 30, 120), max(args.patience, 12), [*planned, *confirm], rows, confirm_planned, start_time)
            planned = confirm_planned

    ensemble_rows: List[Dict[str, Any]] = []
    if not args.smoke_test and not STOP_REQUESTED:
        ensemble_rows = create_validation_ensembles(args, rows, planned, start_time)

    ranked = write_live_leaderboard(run_root, rows)
    selected = select_final_roles(ranked)
    save_json(run_root / "final_selected_models.json", {"selected": selected, "selection_rule": "validation metrics only; test held until after selection"})
    final_tests: List[Dict[str, Any]] = []
    if not args.smoke_test and selected and not STOP_REQUESTED:
        final_tests = run_final_tests(args, selected, planned)
    plots = write_plots(run_root, rows)
    write_partial_summary(run_root, rows, planned, start_time, None, interrupted=STOP_REQUESTED)
    write_final_report(args, rows, selected, final_tests, plots, hardware, start_time)
    append_event(
        run_root,
        "runner_end",
        {
            "completed_results": len(sorted_completed(rows)),
            "ensemble_results": len(ensemble_rows),
            "final_test_roles": len(final_tests),
            "status": "interrupted" if STOP_REQUESTED else "completed",
        },
    )
    print(f"Final report: {run_root / 'final_report.md'}")


if __name__ == "__main__":
    main()
