#!/usr/bin/env python
"""Goal-specific DenseFMS MAE search runner.

This script keeps normal search runs validation-only. Test evaluation is only
performed when --final_test is supplied after validation-based selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "runs/goal_mae_search"
PRIMARY_HORIZONS = (5.0, 10.0, 15.0)

MANIFEST_FIELDS = [
    "run_name",
    "status",
    "command",
    "config_path",
    "checkpoint_path",
    "metrics_path",
    "prediction_csv_path",
    "train_prediction_csv_path",
    "plot_dir",
    "start_time",
    "end_time",
    "best_epoch",
    "best_val_mae",
    "failure_reason",
    "interrupt_reason",
    "resume_action",
]

EXPERIMENT_FIELDS = [
    "run_name",
    "family",
    "status",
    "hypothesis",
    "selected_reason",
    "model_type",
    "fms_context_mode",
    "anchor_mode",
    "anchor_interval_seconds",
    "use_static",
    "static_feature_set",
    "recent_start_observed",
    "sparse_observed",
    "predict_delta_from_anchor",
    "calibration_seconds",
    "recent_window_seconds",
    "horizon_seconds",
    "loss_type",
    "loss_mode",
    "learning_rate",
    "weight_decay",
    "dropout",
    "d_model",
    "seed",
    "split_file",
    "best_epoch",
    "val_mae",
    "val_rmse",
    "val_n",
    "common_val_mae",
    "checkpoint_path",
    "metrics_path",
    "prediction_csv_path",
]

LEADERBOARD_FIELDS = [
    "rank",
    "family",
    "status",
    "model_type",
    "fms_context_mode",
    "anchor_mode",
    "anchor_interval_seconds",
    "use_static",
    "static_feature_set",
    "recent_start_observed",
    "sparse_observed",
    "predict_delta_from_anchor",
    "calibration_seconds",
    "recent_window_seconds",
    "horizon_seconds",
    "loss_type",
    "loss_mode",
    "seed",
    "split_file",
    "h5_val_mae",
    "h10_val_mae",
    "h15_val_mae",
    "mean_val_mae_h5_h10_h15",
    "h2p5_val_mae",
    "h1_diagnostic_val_mae",
    "member_runs",
    "test_metric_final_only",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def tag(value: Any) -> str:
    return f"{float(value):g}".replace(".", "p")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_csv(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    if not rows:
        return "| empty |\n|---|\n"
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def load_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    return read_json(path)


def completed(run_dir: Path) -> bool:
    return (run_dir / "metrics.json").exists() and (run_dir / "best.pt").exists()


def make_spec(
    family: str,
    horizon: float,
    *,
    use_static: bool,
    hypothesis: str,
    selected_reason: str,
    loss_type: str = "smooth_l1",
    loss_mode: str = "level_only",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    d_model: int = 64,
    calibration_seconds: float = 90.0,
    recent_window_seconds: float = 30.0,
    seed: int = 42,
    multi_horizon: bool = False,
    horizon_set: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    static_tag = "static" if use_static else "nostatic"
    h_tag = "mh" if multi_horizon else f"h{tag(horizon)}"
    run_name = (
        f"goal_{family}_{h_tag}_startonly_none_{static_tag}_"
        f"lr{tag(learning_rate)}_wd{tag(weight_decay)}_drop{tag(dropout)}_d{d_model}_seed{seed}"
    )
    return {
        "family": family,
        "run_name": run_name,
        "model": "lc_sa_tcnformer",
        "fms_context_mode": "start_only",
        "anchor_mode": "none",
        "anchor_interval_seconds": 0.0,
        "use_static": bool(use_static),
        "static_feature_set": "age+gender+mssq" if use_static else "none",
        "recent_start_observed": False,
        "sparse_observed": False,
        "predict_delta_from_anchor": False,
        "calibration_seconds": float(calibration_seconds),
        "recent_window_seconds": float(recent_window_seconds),
        "horizon_seconds": float(horizon),
        "loss_type": loss_type,
        "loss_mode": loss_mode,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "dropout": float(dropout),
        "d_model": int(d_model),
        "seed": int(seed),
        "multi_horizon": bool(multi_horizon),
        "horizon_set": [float(v) for v in horizon_set] if horizon_set else None,
        "hypothesis": hypothesis,
        "selected_reason": selected_reason,
    }


def build_plan() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    early_families = [
        (
            "baseline_nostatic",
            False,
            "Reproduce patched start_only no-static baseline under the target h=5/10/15 policy.",
            "Phase 2 baseline reproduction.",
            {},
        ),
        (
            "baseline_static",
            True,
            "Allowed Age/Gender/MSSQ static covariates may reduce subject-level susceptibility error.",
            "Cheap improvement: static personalization with the same main-track inputs.",
            {},
        ),
        (
            "nostatic_l1",
            False,
            "No-static L1 checks whether gains are from loss choice rather than static covariates.",
            "Static baseline underperformed no-static, so no-static loss ablation has higher priority.",
            {"loss_type": "l1"},
        ),
    ]
    for family, use_static, hypothesis, reason, overrides in early_families:
        for horizon in PRIMARY_HORIZONS:
            specs.append(make_spec(family, horizon, use_static=use_static, hypothesis=hypothesis, selected_reason=reason, **overrides))
    specs.append(
        make_spec(
            "multi_nostatic",
            5.0,
            use_static=False,
            hypothesis="No-static multi-horizon checks whether shared horizon learning helps without user covariates.",
            selected_reason="Static baseline underperformed; no-static multi-horizon is the next compact primary-metric-aligned test.",
            multi_horizon=True,
            horizon_set=PRIMARY_HORIZONS,
            learning_rate=5e-4,
        )
    )
    specs.append(
        make_spec(
            "multi_nostatic_lr3e4",
            5.0,
            use_static=False,
            hypothesis="The first multi-horizon refinement lowers LR because the winning run improved late and may benefit from smoother optimization.",
            selected_reason="multi_nostatic was the only family to improve primary mean validation MAE.",
            multi_horizon=True,
            horizon_set=PRIMARY_HORIZONS,
            learning_rate=3e-4,
        )
    )
    specs.append(
        make_spec(
            "multi_nostatic_drop005",
            5.0,
            use_static=False,
            hypothesis="Lower dropout may improve underfit within the strict 10-epoch budget while keeping inputs unchanged.",
            selected_reason="multi_nostatic showed small gains but may be underfit at 10 epochs.",
            multi_horizon=True,
            horizon_set=PRIMARY_HORIZONS,
            learning_rate=5e-4,
            dropout=0.05,
        )
    )
    specs.append(
        make_spec(
            "multi_nostatic_d128",
            5.0,
            use_static=False,
            hypothesis="Increasing d_model may help shared h=5/10/15 representation without adding prohibited inputs.",
            selected_reason="Only the no-static multi-horizon family improved, so a small capacity ablation is justified.",
            multi_horizon=True,
            horizon_set=PRIMARY_HORIZONS,
            learning_rate=5e-4,
            d_model=128,
        )
    )
    diagnostic_families = [
        (
            "diagnostic_h2p5_bestshape",
            2.5,
            "Record the requested h=2.5 short-horizon forecasting result using the best validation family shape; excluded from primary selection.",
            "Post-selection validation-only diagnostic specified by the goal, not a model-selection signal.",
        ),
        (
            "diagnostic_h1_bestshape",
            1.0,
            "Record the requested h=1 diagnostic lower-bound/sanity result using the best validation family shape; excluded from primary selection.",
            "Post-selection validation-only diagnostic specified by the goal, not a model-selection signal.",
        ),
    ]
    for family, horizon, hypothesis, reason in diagnostic_families:
        specs.append(
            make_spec(
                family,
                horizon,
                use_static=False,
                hypothesis=hypothesis,
                selected_reason=reason,
                learning_rate=5e-4,
                dropout=0.05,
            )
        )
    return specs


def train_command(args: argparse.Namespace, spec: Mapping[str, Any]) -> List[str]:
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
        str(args.output_dir),
        "--model",
        str(spec["model"]),
        "--run_name",
        str(spec["run_name"]),
        "--split_file",
        args.split_file,
        "--seed",
        str(int(spec["seed"])),
        "--epochs",
        str(int(args.epochs)),
        "--patience",
        str(int(args.patience)),
        "--batch_size",
        str(int(args.batch_size)),
        "--num_workers",
        str(int(args.num_workers)),
        "--learning_rate",
        f"{float(spec['learning_rate']):g}",
        "--weight_decay",
        f"{float(spec['weight_decay']):g}",
        "--loss_type",
        str(spec["loss_type"]),
        "--loss_mode",
        str(spec["loss_mode"]),
        "--calibration_seconds",
        f"{float(spec['calibration_seconds']):g}",
        "--recent_window_seconds",
        f"{float(spec['recent_window_seconds']):g}",
        "--horizon_seconds",
        f"{float(spec['horizon_seconds']):g}",
        "--anchor_mode",
        "none",
        "--anchor_interval_seconds",
        "0",
        "--fms_context_mode",
        "start_only",
        "--d_model",
        str(int(spec["d_model"])),
        "--dropout",
        f"{float(spec['dropout']):g}",
        "--high_fms_threshold",
        "10.0",
        "--no_test_eval",
        "--skip_existing",
    ]
    if spec.get("use_static"):
        cmd.extend(["--use_static", "--static_features", "age", "gender", "mssq"])
    else:
        cmd.append("--no_static")
    if spec.get("multi_horizon"):
        cmd.append("--multi_horizon")
        cmd.append("--horizon_set")
        cmd.extend(f"{float(v):g}" for v in spec.get("horizon_set") or [])
    if args.device:
        cmd.extend(["--device", args.device])
    return cmd


def eval_command(args: argparse.Namespace, spec: Mapping[str, Any], split: str) -> List[str]:
    run_dir = Path(args.output_dir) / str(spec["run_name"])
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.densefms_forecast.evaluate",
        "--checkpoint",
        str(run_dir / "best.pt"),
        "--data_dir",
        args.data_dir,
        "--split",
        split,
        "--split_file",
        args.split_file,
        "--batch_size",
        str(int(args.batch_size)),
        "--calibration_seconds",
        f"{float(spec['calibration_seconds']):g}",
        "--recent_window_seconds",
        f"{float(spec['recent_window_seconds']):g}",
        "--horizon_seconds",
        f"{float(spec['horizon_seconds']):g}",
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    return cmd


def init_docs(args: argparse.Namespace, specs: Sequence[Mapping[str, Any]]) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "input_contract.md").write_text(
        "\n".join(
            [
                "# Input Contract",
                "",
                f"- Policy file: `{rel(args.policy_file)}`",
                "- FULL_TRAINING_ALLOWED: true",
                "- Main track: fms_context_mode=start_only, anchor_mode=none, anchor_interval_seconds=0.",
                "- Post-calibration FMS input: only one start_fms scalar at recent window start.",
                "- Backward-compatible CSV aliases: anchor_index/anchor_time/anchor_fms.",
                "- Added explicit CSV metadata: start_fms_index/start_fms_time/start_fms_value, nominal_start_index, nominal_start_time, anchor_is_fallback.",
                "- Forbidden inputs in main track: target FMS, current FMS, future FMS, dense FMS sequence in recent window, sparse_observed, recent_start_observed, calibration_end anchor, sparse_anchor.",
                "- Allowed static covariates: Age, Gender, MSSQ. Direct identity fields are not model inputs.",
                "- Sampling interval: 0.5s. Default calibration=90s, recent window=30s.",
                "- Primary selection metric: mean validation MAE over h=5/10/15.",
                "- Normal run test policy: train/val only; test is final-only.",
                "",
                "## Code Inventory",
                "",
                "- Dataset/windowing: `src/densefms_forecast/data.py`",
                "- Models: `src/densefms_forecast/model.py`",
                "- Training/metrics/prediction CSV: `src/densefms_forecast/train.py`",
                "- Evaluation: `src/densefms_forecast/evaluate.py`",
                "- Losses: `src/densefms_forecast/losses.py`",
                "- Plotting: `src/densefms_forecast/plot_compare.py` and train-time plots",
                "- Config: `configs/lc_sa_tcnformer.yaml`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "leakage_audit.md").write_text(
        "\n".join(
            [
                "# Leakage Audit",
                "",
                "Status: initialized before search.",
                "",
                "- Subject/session split: `split_sessions` uses participant_id parsed from filename when no participant column exists; `run_data_sanity_checks` checks disjoint train/val/test groups.",
                "- Target shift: `future_sequence_targets` indexes target as current position + horizon_steps.",
                "- Calibration FMS: `calibration_context_fms` returns only the first calibration_steps for start_only.",
                "- Recent motion: LC-SA prediction positions start at max(calibration_steps, recent_steps-1); recent encoder indexes windows ending at current position.",
                "- start_only FMS: LC-SA uses anchor_mode=none plus fms_context_mode=start_only to gather positions - recent_steps + 1 only.",
                "- Fallback: `_gather_anchor_fms` and prediction CSV metadata use latest finite FMS at or before nominal start index.",
                "- Static scaler/imputation: Age/MSSQ scaler is fit on train sessions only; val/test use transform only.",
                "- Test usage: normal train commands include `--no_test_eval`.",
                "- Sanity test coverage: see `sanity_tests.log` after initialization.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "baseline_summary.md").write_text(
        "\n".join(
            [
                "# Baseline Summary",
                "",
                "Baseline reproduction is planned as `baseline_nostatic` and `baseline_static` families at h=5/10/15.",
                "",
                f"Planned run count: {len(specs)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for path, fields in [
        (out / "resume_manifest.csv", MANIFEST_FIELDS),
        (out / "experiment_log.csv", EXPERIMENT_FIELDS),
        (out / "leaderboard.csv", LEADERBOARD_FIELDS),
    ]:
        if not path.exists():
            write_csv(path, [], fields)
    for path, text in [
        (out / "experiment_log.md", "# Experiment Log\n\nNo completed runs yet.\n"),
        (out / "leaderboard.md", "# Leaderboard\n\nNo completed run families yet.\n"),
        (out / "best_model_summary.md", "# Best Model Summary\n\nNo validation-selected candidate yet.\n"),
        (out / "RUN_STATE.md", "# RUN_STATE\n\nStatus: initialized.\n"),
    ]:
        if not path.exists():
            path.write_text(text, encoding="utf-8")


def update_manifest(args: argparse.Namespace, spec: Mapping[str, Any], status: str, **extra: Any) -> None:
    out = Path(args.output_dir)
    rows = read_csv(out / "resume_manifest.csv")
    run_name = str(spec["run_name"])
    existing = {row["run_name"]: row for row in rows if row.get("run_name")}
    row = existing.get(run_name, {})
    run_dir = out / run_name
    row.update(
        {
            "run_name": run_name,
            "status": status,
            "command": " ".join(train_command(args, spec)),
            "config_path": args.config,
            "checkpoint_path": rel(run_dir / "best.pt"),
            "metrics_path": rel(run_dir / "metrics.json"),
            "prediction_csv_path": rel(run_dir / "val_predictions.csv"),
            "train_prediction_csv_path": rel(run_dir / "eval_train/train_predictions.csv"),
            "plot_dir": rel(run_dir / "plots"),
            "resume_action": "skip_completed" if status == "completed" else row.get("resume_action", ""),
        }
    )
    row.update({k: v for k, v in extra.items() if v is not None})
    existing[run_name] = row
    write_csv(out / "resume_manifest.csv", list(existing.values()), MANIFEST_FIELDS)


def run_subprocess(cmd: Sequence[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
        proc = subprocess.run(list(cmd), cwd=str(cwd), stdout=out_f, stderr=err_f, text=True)
    return int(proc.returncode)


def metric_for_run(run_dir: Path, spec: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = load_metrics(run_dir)
    if not metrics:
        return {}
    val = metrics.get("metrics", {}).get("best_val_metrics", {})
    if spec.get("multi_horizon"):
        by_h = val.get("by_horizon", {})
        h_values = {float(k): as_float(v.get("mae")) for k, v in by_h.items() if isinstance(v, Mapping)}
        vals = [h_values.get(h, math.nan) for h in PRIMARY_HORIZONS]
        primary = sum(vals) / len(vals) if all(math.isfinite(v) for v in vals) else math.nan
        return {
            "best_epoch": metrics.get("metrics", {}).get("best_epoch"),
            "val_mae": primary,
            "val_rmse": val.get("rmse"),
            "val_n": val.get("n"),
            "common_val_mae": val.get("common_mae"),
            "by_horizon": h_values,
        }
    return {
        "best_epoch": metrics.get("metrics", {}).get("best_epoch"),
        "val_mae": val.get("mae"),
        "val_rmse": val.get("rmse"),
        "val_n": val.get("n"),
        "common_val_mae": val.get("common_mae"),
        "by_horizon": {float(spec["horizon_seconds"]): as_float(val.get("mae"))},
    }


def run_one(args: argparse.Namespace, spec: Mapping[str, Any]) -> None:
    out = Path(args.output_dir)
    run_dir = out / str(spec["run_name"])
    if completed(run_dir):
        m = metric_for_run(run_dir, spec)
        update_manifest(args, spec, "completed", best_epoch=m.get("best_epoch"), best_val_mae=m.get("val_mae"))
        return
    update_manifest(args, spec, "started", start_time=now_iso())
    update_manifest(args, spec, "running")
    logs = out / "logs"
    train_code = run_subprocess(
        train_command(args, spec),
        ROOT,
        logs / f"{spec['run_name']}.stdout.log",
        logs / f"{spec['run_name']}.stderr.log",
    )
    if train_code != 0:
        update_manifest(args, spec, "failed", end_time=now_iso(), failure_reason=f"train_exit_{train_code}", resume_action="inspect_logs")
        return
    train_eval_code = run_subprocess(
        eval_command(args, spec, "train"),
        ROOT,
        logs / f"{spec['run_name']}.eval_train.stdout.log",
        logs / f"{spec['run_name']}.eval_train.stderr.log",
    )
    if train_eval_code != 0:
        update_manifest(args, spec, "failed", end_time=now_iso(), failure_reason=f"train_eval_exit_{train_eval_code}", resume_action="inspect_logs")
        return
    m = metric_for_run(run_dir, spec)
    update_manifest(args, spec, "completed", end_time=now_iso(), best_epoch=m.get("best_epoch"), best_val_mae=m.get("val_mae"))


def experiment_rows(args: argparse.Namespace, specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    manifest = {row["run_name"]: row for row in read_csv(Path(args.output_dir) / "resume_manifest.csv")}
    rows = []
    for spec in specs:
        run_dir = Path(args.output_dir) / str(spec["run_name"])
        metrics = metric_for_run(run_dir, spec)
        rows.append(
            {
                "run_name": spec["run_name"],
                "family": spec["family"],
                "status": manifest.get(str(spec["run_name"]), {}).get("status", "pending"),
                "hypothesis": spec["hypothesis"],
                "selected_reason": spec["selected_reason"],
                "model_type": spec["model"],
                "fms_context_mode": spec["fms_context_mode"],
                "anchor_mode": spec["anchor_mode"],
                "anchor_interval_seconds": spec["anchor_interval_seconds"],
                "use_static": spec["use_static"],
                "static_feature_set": spec["static_feature_set"],
                "recent_start_observed": spec["recent_start_observed"],
                "sparse_observed": spec["sparse_observed"],
                "predict_delta_from_anchor": spec["predict_delta_from_anchor"],
                "calibration_seconds": spec["calibration_seconds"],
                "recent_window_seconds": spec["recent_window_seconds"],
                "horizon_seconds": "multi" if spec.get("multi_horizon") else spec["horizon_seconds"],
                "loss_type": spec["loss_type"],
                "loss_mode": spec["loss_mode"],
                "learning_rate": spec["learning_rate"],
                "weight_decay": spec["weight_decay"],
                "dropout": spec["dropout"],
                "d_model": spec["d_model"],
                "seed": spec["seed"],
                "split_file": args.split_file,
                "best_epoch": metrics.get("best_epoch", ""),
                "val_mae": metrics.get("val_mae", ""),
                "val_rmse": metrics.get("val_rmse", ""),
                "val_n": metrics.get("val_n", ""),
                "common_val_mae": metrics.get("common_val_mae", ""),
                "checkpoint_path": rel(run_dir / "best.pt") if (run_dir / "best.pt").exists() else "",
                "metrics_path": rel(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else "",
                "prediction_csv_path": rel(run_dir / "val_predictions.csv") if (run_dir / "val_predictions.csv").exists() else "",
            }
        )
    return rows


def leaderboard_rows(args: argparse.Namespace, specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_family: Dict[str, List[Mapping[str, Any]]] = {}
    for spec in specs:
        by_family.setdefault(str(spec["family"]), []).append(spec)
    final_by_family = {row.get("family"): row for row in read_csv(Path(args.output_dir) / "final_test_metrics.csv")}
    rows: List[Dict[str, Any]] = []
    for family, items in by_family.items():
        h_scores: Dict[float, float] = {}
        members: List[str] = []
        representative = items[0]
        complete_count = 0
        for spec in items:
            run_dir = Path(args.output_dir) / str(spec["run_name"])
            metrics = metric_for_run(run_dir, spec)
            if metrics:
                complete_count += 1
                members.append(str(spec["run_name"]))
                h_scores.update(metrics.get("by_horizon", {}))
        vals = [h_scores.get(h, math.nan) for h in PRIMARY_HORIZONS]
        mean = sum(vals) / len(vals) if all(math.isfinite(v) for v in vals) else math.nan
        is_diagnostic = family.startswith("diagnostic_")
        if all(math.isfinite(v) for v in vals):
            status = "completed"
        elif is_diagnostic and complete_count == len(items) and h_scores:
            status = "diagnostic_completed"
        else:
            status = f"partial_{complete_count}/{len(items)}"
        horizon_label = "5/10/15"
        if is_diagnostic:
            horizon_label = f"{float(representative['horizon_seconds']):g}"
        rows.append(
            {
                "family": family,
                "status": status,
                "model_type": representative["model"],
                "fms_context_mode": representative["fms_context_mode"],
                "anchor_mode": representative["anchor_mode"],
                "anchor_interval_seconds": representative["anchor_interval_seconds"],
                "use_static": representative["use_static"],
                "static_feature_set": representative["static_feature_set"],
                "recent_start_observed": representative["recent_start_observed"],
                "sparse_observed": representative["sparse_observed"],
                "predict_delta_from_anchor": representative["predict_delta_from_anchor"],
                "calibration_seconds": representative["calibration_seconds"],
                "recent_window_seconds": representative["recent_window_seconds"],
                "horizon_seconds": horizon_label,
                "loss_type": representative["loss_type"],
                "loss_mode": representative["loss_mode"],
                "seed": representative["seed"],
                "split_file": args.split_file,
                "h5_val_mae": h_scores.get(5.0, ""),
                "h10_val_mae": h_scores.get(10.0, ""),
                "h15_val_mae": h_scores.get(15.0, ""),
                "mean_val_mae_h5_h10_h15": mean if math.isfinite(mean) else "",
                "h2p5_val_mae": h_scores.get(2.5, ""),
                "h1_diagnostic_val_mae": h_scores.get(1.0, ""),
                "member_runs": ";".join(members),
                "test_metric_final_only": (
                    f"test_mae={final_by_family[family].get('test_mae')}"
                    if family in final_by_family
                    else ""
                ),
            }
        )
    ranked = sorted(rows, key=lambda r: as_float(r.get("mean_val_mae_h5_h10_h15"), math.inf))
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx if math.isfinite(as_float(row.get("mean_val_mae_h5_h10_h15"), math.inf)) else ""
    return ranked


def write_summaries(args: argparse.Namespace, specs: Sequence[Mapping[str, Any]], start_time: float) -> None:
    out = Path(args.output_dir)
    exp = experiment_rows(args, specs)
    board = leaderboard_rows(args, specs)
    write_csv(out / "experiment_log.csv", exp, EXPERIMENT_FIELDS)
    write_csv(out / "leaderboard.csv", board, LEADERBOARD_FIELDS)
    (out / "experiment_log.md").write_text("# Experiment Log\n\n" + markdown_table(exp, EXPERIMENT_FIELDS), encoding="utf-8")
    (out / "leaderboard.md").write_text("# Leaderboard\n\n" + markdown_table(board, LEADERBOARD_FIELDS), encoding="utf-8")
    completed_rows = [r for r in exp if r.get("status") == "completed"]
    best = next((r for r in board if r.get("rank") == 1), None)
    best_lines = ["# Best Model Summary", ""]
    if best:
        best_lines.append(f"- Primary selection family: {best.get('family')}")
        best_lines.append(f"- mean validation MAE h=5/10/15: {best.get('mean_val_mae_h5_h10_h15')}")
        best_lines.append(f"- h=5 validation MAE: {best.get('h5_val_mae')}")
        best_lines.append(f"- h=10 validation MAE: {best.get('h10_val_mae')}")
        best_lines.append(f"- h=15 validation MAE: {best.get('h15_val_mae')}")
        best_lines.append(f"- member runs: {best.get('member_runs')}")
        best_lines.append("- Selection basis: validation only; h=1 diagnostic and final test are excluded from model selection.")
    else:
        best_lines.append("- No complete h=5/10/15 family yet.")
    diag_2p5 = next((r for r in board if r.get("family") == "diagnostic_h2p5_bestshape"), None)
    diag_1 = next((r for r in board if r.get("family") == "diagnostic_h1_bestshape"), None)
    if diag_2p5 or diag_1:
        best_lines.extend(["", "## Diagnostic Horizons"])
        if diag_2p5:
            best_lines.append(f"- h=2.5 validation MAE: {diag_2p5.get('h2p5_val_mae')} ({diag_2p5.get('member_runs')})")
        if diag_1:
            best_lines.append(f"- h=1 diagnostic validation MAE: {diag_1.get('h1_diagnostic_val_mae')} ({diag_1.get('member_runs')})")
    final_rows = read_csv(out / "final_test_metrics.csv")
    if final_rows:
        best_lines.extend(["", "## Final Test"])
        for row in final_rows:
            best_lines.append(
                f"- {row.get('run_name')}: MAE={row.get('test_mae')}, RMSE={row.get('test_rmse')}, R2={row.get('test_r2')}"
            )
            metrics_path = Path(row.get("metrics_path", ""))
            if metrics_path.exists():
                metrics = read_json(metrics_path).get("metrics", {})
                by_h = metrics.get("by_horizon", {})
                for h in ("5", "10", "15"):
                    if isinstance(by_h.get(h), Mapping):
                        best_lines.append(f"  - h={h} test MAE: {by_h[h].get('mae')}")
        best_lines.append("- Final test was generated once after validation-based selection.")
    (out / "best_model_summary.md").write_text("\n".join(best_lines) + "\n", encoding="utf-8")
    manifest = read_csv(out / "resume_manifest.csv")
    state = [
        "# RUN_STATE",
        "",
        f"- Updated: {now_iso()}",
        f"- Runtime seconds in this invocation: {time.time() - start_time:.1f}",
        f"- Completed runs: {sum(1 for row in manifest if row.get('status') == 'completed')}",
        f"- Failed runs: {sum(1 for row in manifest if row.get('status') == 'failed')}",
        f"- Interrupted runs: {sum(1 for row in manifest if row.get('status') == 'interrupted')}",
        f"- Best validation family: {best.get('family') if best else 'pending'}",
        "",
        "## Next Candidate",
    ]
    pending = [s for s in specs if not completed(Path(args.output_dir) / str(s["run_name"]))]
    if pending:
        state.append(f"- {pending[0]['run_name']}: {pending[0]['hypothesis']}")
    else:
        state.append("- No pending planned validation runs.")
    state.extend(
        [
            "",
            "## Stopped Directions",
            "- Static follow-up ablations were stopped because `baseline_static` underperformed `baseline_nostatic` on the primary mean validation MAE.",
            "- No further adaptive tuning is planned after the final test result.",
        ]
    )
    state.extend(["", "## Completed Validation Runs", markdown_table(completed_rows, ["run_name", "family", "horizon_seconds", "val_mae", "best_epoch"])])
    (out / "RUN_STATE.md").write_text("\n".join(state) + "\n", encoding="utf-8")


def run_final_test(args: argparse.Namespace, specs: Sequence[Mapping[str, Any]]) -> None:
    board = leaderboard_rows(args, specs)
    best = next((row for row in board if row.get("rank") == 1), None)
    if not best:
        return
    members = [name for name in str(best.get("member_runs", "")).split(";") if name]
    spec_by_name = {str(spec["run_name"]): spec for spec in specs}
    rows = []
    for name in members:
        spec = spec_by_name.get(name)
        if not spec:
            continue
        logs = Path(args.output_dir) / "logs"
        code = run_subprocess(
            eval_command(args, spec, "test"),
            ROOT,
            logs / f"{name}.final_test.stdout.log",
            logs / f"{name}.final_test.stderr.log",
        )
        eval_metrics_path = Path(args.output_dir) / name / "eval_test" / "metrics.json"
        metrics = read_json(eval_metrics_path).get("metrics", {}) if eval_metrics_path.exists() else {}
        rows.append(
            {
                "family": best.get("family"),
                "run_name": name,
                "status": "completed" if code == 0 else f"failed_{code}",
                "horizon_seconds": spec.get("horizon_seconds"),
                "val_family_mean_mae": best.get("mean_val_mae_h5_h10_h15"),
                "test_mae": metrics.get("mae", ""),
                "test_rmse": metrics.get("rmse", ""),
                "test_r2": metrics.get("r2", ""),
                "metrics_path": rel(eval_metrics_path),
                "prediction_csv_path": rel(Path(args.output_dir) / name / "eval_test" / "test_predictions.csv"),
            }
        )
    fields = ["family", "run_name", "status", "horizon_seconds", "val_family_mean_mae", "test_mae", "test_rmse", "test_r2", "metrics_path", "prediction_csv_path"]
    write_csv(Path(args.output_dir) / "final_test_metrics.csv", rows, fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run goal-specific DenseFMS MAE search.")
    parser.add_argument("--data_dir", default="./DenseFMS/Dataset")
    parser.add_argument("--config", default="configs/lc_sa_tcnformer.yaml")
    parser.add_argument("--split_file", default="./artifacts/densefms_split_seed42.json")
    parser.add_argument("--policy_file", default="docs/codex/goal_mae_search_policy.md")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_runs", type=int, default=75)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--final_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = Path(args.output_dir)
    args.policy_file = Path(args.policy_file)
    specs = build_plan()
    init_docs(args, specs)
    start = time.time()
    if args.dry_run:
        rows = [{"run_name": spec["run_name"], "command": " ".join(train_command(args, spec)), "hypothesis": spec["hypothesis"]} for spec in specs[: args.max_runs]]
        write_csv(Path(args.output_dir) / "dry_run_commands.csv", rows, ["run_name", "command", "hypothesis"])
        write_summaries(args, specs, start)
        return
    if not args.summary_only:
        for spec in specs[: args.max_runs]:
            run_one(args, spec)
            write_summaries(args, specs, start)
    write_summaries(args, specs, start)
    if args.final_test:
        run_final_test(args, specs)
        write_summaries(args, specs, start)


if __name__ == "__main__":
    main()
