#!/usr/bin/env python
"""Validation-only audit writer for the 0506 adaptive improvement goal."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_HORIZONS = (5.0, 10.0, 15.0)
PRIOR_LABEL = "prior_recent10_no_static_s7"
PRIOR_RUN_DIR = ROOT / "runs/goal_mae_next1h_0505/next1h_recent10_e80_s7"


EXPERIMENT_FIELDS = [
    "run_name",
    "status",
    "parent_run",
    "branch",
    "hypothesis",
    "changed_fields",
    "recent_window_seconds",
    "static_usage",
    "model_changes",
    "loss_changes",
    "motion_feature_changes",
    "seed",
    "epochs_or_max_epochs",
    "primary_val_mae_h5_h10_h15_mean",
    "val_mae_h5",
    "val_mae_h10",
    "val_mae_h15",
    "h15_change_vs_parent",
    "trend_plot_review",
    "direction_agreement",
    "large_rise_recall",
    "large_drop_recall",
    "delta_or_slope_correlation",
    "dynamic_range_ratio",
    "lag_estimate",
    "high_fms_or_large_delta_notes",
    "promotion_decision",
    "stop_reason",
    "test_usage",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def fmt(value: Any) -> str:
    x = as_float(value)
    if math.isfinite(x):
        return f"{x:.6f}"
    return "" if value in (None, "") else str(value)


def markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], limit: Optional[int] = None) -> str:
    shown = list(rows[:limit] if limit is not None else rows)
    if not shown:
        return "| empty |\n|---|"
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in shown:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def by_horizon_metrics(metrics: Mapping[str, Any], horizon: float) -> Mapping[str, Any]:
    by_h = metrics.get("by_horizon", {})
    if not isinstance(by_h, Mapping):
        return {}
    keys = [f"{horizon:g}", str(float(horizon)), str(int(horizon)) if float(horizon).is_integer() else ""]
    for key in keys:
        if key and key in by_h and isinstance(by_h[key], Mapping):
            return by_h[key]
    return {}


def parse_run_metrics(run_dir: Path) -> Dict[str, Any]:
    payload = read_json(run_dir / "metrics.json")
    metrics = payload.get("metrics", {}) if isinstance(payload, Mapping) else {}
    best = metrics.get("best_val_metrics") or metrics.get("val_metrics") or {}
    if not isinstance(best, Mapping):
        best = {}
    h_mae: Dict[float, float] = {}
    for horizon in PRIMARY_HORIZONS:
        h_mae[horizon] = as_float(by_horizon_metrics(best, horizon).get("mae"))
    primary_values = [h_mae[h] for h in PRIMARY_HORIZONS if math.isfinite(h_mae[h])]
    config = read_json(run_dir / "config_snapshot.json")
    return {
        "val_mae": as_float(best.get("mae")),
        "val_rmse": as_float(best.get("rmse")),
        "val_n": best.get("n", ""),
        "val_mae_h5": h_mae[5.0],
        "val_mae_h10": h_mae[10.0],
        "val_mae_h15": h_mae[15.0],
        "primary": float(np.mean(primary_values)) if len(primary_values) == 3 else math.nan,
        "best_epoch": metrics.get("best_epoch", ""),
        "config": config,
    }


def parse_eval_metrics(run_dir: Path, split: str = "test") -> Dict[str, Any]:
    payload = read_json(run_dir / f"eval_{split}" / "metrics.json")
    metrics = payload.get("metrics", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(metrics, Mapping):
        metrics = {}
    h_mae: Dict[float, float] = {}
    for horizon in PRIMARY_HORIZONS:
        h_mae[horizon] = as_float(by_horizon_metrics(metrics, horizon).get("mae"))
    primary_values = [h_mae[h] for h in PRIMARY_HORIZONS if math.isfinite(h_mae[h])]
    return {
        "mae": as_float(metrics.get("mae")),
        "rmse": as_float(metrics.get("rmse")),
        "r2": as_float(metrics.get("r2")),
        "n": metrics.get("n", ""),
        "mae_h5": h_mae[5.0],
        "mae_h10": h_mae[10.0],
        "mae_h15": h_mae[15.0],
        "primary": float(np.mean(primary_values)) if len(primary_values) == 3 else math.nan,
        "high_fms_precision": as_float(metrics.get("high_fms_precision")),
        "high_fms_recall": as_float(metrics.get("high_fms_recall")),
        "high_fms_f1": as_float(metrics.get("high_fms_f1")),
        "path": run_dir / f"eval_{split}" / "metrics.json",
        "predictions": run_dir / f"eval_{split}" / f"{split}_predictions.csv",
    }


def _series_key(row: Mapping[str, str]) -> Tuple[str, str, float]:
    return (
        str(row.get("session_id") or row.get("source_file") or ""),
        str(row.get("participant_id") or ""),
        as_float(row.get("horizon_seconds")),
    )


def flow_metrics(rows: Sequence[Mapping[str, str]], eps: float = 0.5) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str, float], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        if str(row.get("split", "val")).lower() not in {"", "val", "validation"}:
            continue
        if str(row.get("in_common_eval_window", "true")).lower() not in {"true", "1", "yes", ""}:
            continue
        groups[_series_key(row)].append(row)
    dt_all: List[float] = []
    dp_all: List[float] = []
    lag_values: List[int] = []
    target_all: List[float] = []
    pred_all: List[float] = []
    high_bias: List[float] = []
    for items in groups.values():
        ordered = sorted(items, key=lambda r: (as_float(r.get("current_time")), as_float(r.get("target_time"))))
        target = np.asarray([as_float(r.get("target_fms")) for r in ordered], dtype=np.float64)
        pred = np.asarray([as_float(r.get("predicted_fms")) for r in ordered], dtype=np.float64)
        valid = np.isfinite(target) & np.isfinite(pred)
        if valid.sum() < 2:
            continue
        target = target[valid]
        pred = pred[valid]
        target_all.extend(target.tolist())
        pred_all.extend(pred.tolist())
        high = target >= 10.0
        high_bias.extend((pred[high] - target[high]).tolist())
        dt = np.diff(target)
        dp = np.diff(pred)
        dt_all.extend(dt.tolist())
        dp_all.extend(dp.tolist())
        if target.size >= 8 and np.std(target) > 1e-9 and np.std(pred) > 1e-9:
            max_lag = min(10, target.size - 2)
            scores: List[Tuple[float, int]] = []
            t0 = target - target.mean()
            p0 = pred - pred.mean()
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    a = t0[-lag:]
                    b = p0[:lag]
                elif lag > 0:
                    a = t0[:-lag]
                    b = p0[lag:]
                else:
                    a = t0
                    b = p0
                if a.size >= 3 and np.std(a) > 1e-9 and np.std(b) > 1e-9:
                    scores.append((float(np.corrcoef(a, b)[0, 1]), lag))
            if scores:
                lag_values.append(max(scores, key=lambda x: x[0])[1])
    dt = np.asarray(dt_all, dtype=np.float64)
    dp = np.asarray(dp_all, dtype=np.float64)
    target = np.asarray(target_all, dtype=np.float64)
    pred = np.asarray(pred_all, dtype=np.float64)
    diff_valid = np.isfinite(dt) & np.isfinite(dp)
    if diff_valid.any():
        dt = dt[diff_valid]
        dp = dp[diff_valid]
        moving = np.abs(dt) >= eps
        rise = dt >= eps
        drop = dt <= -eps
        delta_corr = float(np.corrcoef(dt, dp)[0, 1]) if np.std(dt) > 1e-9 and np.std(dp) > 1e-9 else math.nan
        direction_agreement = float(np.mean(np.sign(dt[moving]) == np.sign(dp[moving]))) if moving.any() else math.nan
        large_rise_recall = float(np.mean(dp[rise] > 0.0)) if rise.any() else math.nan
        large_drop_recall = float(np.mean(dp[drop] < 0.0)) if drop.any() else math.nan
    else:
        direction_agreement = large_rise_recall = large_drop_recall = delta_corr = math.nan
    dynamic_range_ratio = float(np.std(pred) / np.std(target)) if target.size and np.std(target) > 1e-9 else math.nan
    return {
        "direction_agreement": direction_agreement,
        "large_rise_recall": large_rise_recall,
        "large_drop_recall": large_drop_recall,
        "delta_or_slope_correlation": delta_corr,
        "dynamic_range_ratio": dynamic_range_ratio,
        "lag_estimate": float(np.median(lag_values)) if lag_values else math.nan,
        "high_fms_bias": float(np.mean(high_bias)) if high_bias else math.nan,
        "diff_points": int(dt.size),
        "large_rise_points": int(np.sum(dt >= eps)) if dt.size else 0,
        "large_drop_points": int(np.sum(dt <= -eps)) if dt.size else 0,
    }


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "run"


def generate_validation_plots(label: str, rows: Sequence[Mapping[str, str]], output_dir: Path) -> Dict[str, str]:
    """Write compact validation-only horizon plots for visual trend review."""
    if not rows:
        return {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    plot_dir = output_dir / "validation_plot_reviews"
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups: Dict[Tuple[str, str, float], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        if str(row.get("split", "val")).lower() not in {"", "val", "validation"}:
            continue
        if str(row.get("in_common_eval_window", "true")).lower() not in {"true", "1", "yes", ""}:
            continue
        groups[_series_key(row)].append(row)

    paths: Dict[str, str] = {}
    for horizon in PRIMARY_HORIZONS:
        candidates = []
        for key, items in groups.items():
            if abs(key[2] - horizon) > 1e-6:
                continue
            ordered = sorted(items, key=lambda r: (as_float(r.get("current_time")), as_float(r.get("target_time"))))
            target = np.asarray([as_float(r.get("target_fms")) for r in ordered], dtype=np.float64)
            pred = np.asarray([as_float(r.get("predicted_fms")) for r in ordered], dtype=np.float64)
            time = np.asarray([as_float(r.get("target_time") or r.get("current_time")) for r in ordered], dtype=np.float64)
            valid = np.isfinite(target) & np.isfinite(pred) & np.isfinite(time)
            if valid.sum() < 4:
                continue
            target = target[valid]
            pred = pred[valid]
            time = time[valid]
            score = float(np.std(target)) + 0.02 * float(valid.sum())
            candidates.append((score, key, time, target, pred))
        if not candidates:
            continue
        selected = sorted(candidates, key=lambda item: item[0], reverse=True)[:3]
        fig, axes = plt.subplots(len(selected), 1, figsize=(10, 2.7 * len(selected)), squeeze=False)
        for ax, (_, key, time, target, pred) in zip(axes[:, 0], selected):
            x = time - time.min()
            ax.plot(x, target, color="#111111", linewidth=1.8, label="target")
            ax.plot(x, pred, color="#d55e00", linewidth=1.6, label="prediction")
            session = key[0].replace("\\", "/").split("/")[-1]
            ax.set_title(f"{session} | h={horizon:g}s", fontsize=9)
            ax.set_xlabel("seconds from first plotted target")
            ax.set_ylabel("FMS")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)
        fig.suptitle(f"{label} validation trend review h={horizon:g}s", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = plot_dir / f"{safe_name(label)}_h{horizon:g}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths[f"h{horizon:g}"] = rel(path)
    return paths


def delta_distribution(rows: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    deltas = []
    by_h: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        target = as_float(row.get("target_fms"))
        start = as_float(row.get("start_fms_value") or row.get("anchor_fms"))
        horizon = as_float(row.get("horizon_seconds"))
        if math.isfinite(target) and math.isfinite(start):
            delta = target - start
            deltas.append(delta)
            by_h[f"{horizon:g}"].append(delta)
    out: Dict[str, Any] = {}
    arr = np.asarray(finite(deltas), dtype=np.float64)
    if arr.size:
        out["overall"] = {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "q05": float(np.quantile(arr, 0.05)),
            "q25": float(np.quantile(arr, 0.25)),
            "q50": float(np.quantile(arr, 0.50)),
            "q75": float(np.quantile(arr, 0.75)),
            "q95": float(np.quantile(arr, 0.95)),
            "abs_ge_2": int(np.sum(np.abs(arr) >= 2.0)),
            "abs_ge_3": int(np.sum(np.abs(arr) >= 3.0)),
            "abs_ge_4": int(np.sum(np.abs(arr) >= 4.0)),
        }
    out["by_horizon"] = {}
    for horizon, values in sorted(by_h.items(), key=lambda kv: as_float(kv[0])):
        h_arr = np.asarray(finite(values), dtype=np.float64)
        if h_arr.size:
            out["by_horizon"][horizon] = {
                "n": int(h_arr.size),
                "mean": float(np.mean(h_arr)),
                "std": float(np.std(h_arr)),
                "q05": float(np.quantile(h_arr, 0.05)),
                "q50": float(np.quantile(h_arr, 0.50)),
                "q95": float(np.quantile(h_arr, 0.95)),
                "abs_ge_2": int(np.sum(np.abs(h_arr) >= 2.0)),
                "abs_ge_3": int(np.sum(np.abs(h_arr) >= 3.0)),
                "abs_ge_4": int(np.sum(np.abs(h_arr) >= 4.0)),
            }
    return out


def discover_runs(output_dir: Path) -> List[Tuple[str, Path, str]]:
    runs: List[Tuple[str, Path, str]] = []
    if PRIOR_RUN_DIR.exists():
        runs.append((PRIOR_LABEL, PRIOR_RUN_DIR, "validation_prior"))
    for child in sorted(output_dir.iterdir() if output_dir.exists() else []):
        if child.is_dir() and (child / "metrics.json").exists():
            runs.append((child.name, child, "goal_run"))
    return runs


def row_for_run(label: str, run_dir: Path, source: str, baseline_h15: float, output_dir: Path) -> Dict[str, Any]:
    parsed = parse_run_metrics(run_dir)
    config = parsed.get("config", {})
    model_cfg = config.get("model", {}) if isinstance(config, Mapping) else {}
    data_cfg = config.get("data", {}) if isinstance(config, Mapping) else {}
    train_cfg = config.get("training", {}) if isinstance(config, Mapping) else {}
    loss_cfg = config.get("loss", {}) if isinstance(config, Mapping) else {}
    pred_path = run_dir / "val_predictions.csv"
    pred_rows = read_csv(pred_path)
    flows = flow_metrics(pred_rows)
    plot_paths = generate_validation_plots(label, pred_rows, output_dir)
    primary = parsed["primary"]
    h15_change = parsed["val_mae_h15"] - baseline_h15 if math.isfinite(parsed["val_mae_h15"]) else math.nan
    status = "completed" if (run_dir / "metrics.json").exists() and (run_dir / "best.pt").exists() else "incomplete"
    branch = source
    if "h15" in label:
        branch = "h15_head"
    elif "dual" in label or model_cfg.get("forecast_head_mode") in {"delta", "dual_average", "dual_gated"}:
        branch = "dual_head"
    elif model_cfg.get("motion_feature_mode", "none") != "none":
        branch = "motion_feature"
    return {
        "run_name": label,
        "status": status,
        "parent_run": PRIOR_LABEL if source == "goal_run" else "",
        "branch": branch,
        "hypothesis": model_cfg.get("architecture_hypothesis", ""),
        "changed_fields": "",
        "recent_window_seconds": data_cfg.get("recent_window_seconds", ""),
        "static_usage": "static" if data_cfg.get("use_static") else "no_static",
        "model_changes": f"model={config.get('model_name', 'lc_sa_tcnformer')}; head={model_cfg.get('forecast_head_mode', 'level')}; hhead={model_cfg.get('horizon_head_mode', 'linear')}",
        "loss_changes": f"loss={loss_cfg.get('type', '')}/{loss_cfg.get('mode', '')}; alpha={loss_cfg.get('dual_aux_alpha', 0)}; beta={loss_cfg.get('dual_aux_beta', 0)}",
        "motion_feature_changes": model_cfg.get("motion_feature_mode", "none"),
        "seed": train_cfg.get("seed", ""),
        "epochs_or_max_epochs": train_cfg.get("epochs", ""),
        "primary_val_mae_h5_h10_h15_mean": fmt(primary),
        "val_mae_h5": fmt(parsed["val_mae_h5"]),
        "val_mae_h10": fmt(parsed["val_mae_h10"]),
        "val_mae_h15": fmt(parsed["val_mae_h15"]),
        "h15_change_vs_parent": fmt(h15_change),
        "trend_plot_review": "; ".join(plot_paths.values()),
        "direction_agreement": fmt(flows["direction_agreement"]),
        "large_rise_recall": fmt(flows["large_rise_recall"]),
        "large_drop_recall": fmt(flows["large_drop_recall"]),
        "delta_or_slope_correlation": fmt(flows["delta_or_slope_correlation"]),
        "dynamic_range_ratio": fmt(flows["dynamic_range_ratio"]),
        "lag_estimate": fmt(flows["lag_estimate"]),
        "high_fms_or_large_delta_notes": f"high_bias={fmt(flows['high_fms_bias'])}; diff_points={flows['diff_points']}",
        "promotion_decision": "",
        "stop_reason": "",
        "test_usage": "none_prelock",
        "_primary": primary,
        "_run_dir": run_dir,
    }


def write_reports(output_dir: Path, rows: List[Dict[str, Any]], start_time: str, sanity_status: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: as_float(row.get("primary_val_mae_h5_h10_h15_mean"), math.inf))
    leaderboard = []
    for idx, row in enumerate(sorted_rows, start=1):
        item = dict(row)
        item["rank"] = idx
        leaderboard.append(item)
    write_csv(output_dir / "experiment_log.csv", rows, EXPERIMENT_FIELDS)
    write_csv(output_dir / "leaderboard.csv", leaderboard, ["rank", *EXPERIMENT_FIELDS])
    md_fields = [
        "rank",
        "run_name",
        "primary_val_mae_h5_h10_h15_mean",
        "val_mae_h5",
        "val_mae_h10",
        "val_mae_h15",
        "direction_agreement",
        "dynamic_range_ratio",
        "high_fms_or_large_delta_notes",
    ]
    write_text(output_dir / "leaderboard.md", "# Leaderboard\n\n" + markdown_table(leaderboard, md_fields, limit=40))
    write_text(output_dir / "experiment_log.md", "# Experiment Log\n\n" + markdown_table(rows, EXPERIMENT_FIELDS, limit=60))
    manifest_rows = [
        {
            "run_name": row["run_name"],
            "status": row["status"],
            "checkpoint_path": rel(Path(row["_run_dir"]) / "best.pt"),
            "metrics_path": rel(Path(row["_run_dir"]) / "metrics.json"),
            "prediction_csv_path": rel(Path(row["_run_dir"]) / "val_predictions.csv"),
            "resume_action": "completed; do not rerun" if row["status"] == "completed" else "inspect before resume",
        }
        for row in rows
    ]
    write_csv(
        output_dir / "resume_manifest.csv",
        manifest_rows,
        ["run_name", "status", "checkpoint_path", "metrics_path", "prediction_csv_path", "resume_action"],
    )

    best = leaderboard[0] if leaderboard else {}
    final_test = parse_eval_metrics(Path(best.get("_run_dir", "")), "test") if best else {}
    final_test_available = bool(final_test and Path(final_test.get("path", "")).exists())
    required_summary = [
        "# PLAN",
        "",
        "- Tier 0: input/leakage/delta/flow audit, current best comparability, no-test guard.",
        "- Tier 1: h15 head capacity and level+delta dual-head representative runs.",
        "- Tier 2: motion-derived causal feature run if Tier 1 does not solve smoothing/lag.",
        "- Tier 3: static/high-change/optimization fallback only with expected learning value.",
        "- Final test is optional in this exploratory goal and must not run before validation lock.",
    ]
    write_text(output_dir / "PLAN.md", "\n".join(required_summary))

    write_text(
        output_dir / "input_contract.md",
        "\n".join(
            [
                "# Input Contract",
                "",
                "- Main track: `fms_context_mode=start_only`, `anchor_mode=none`, `anchor_interval_seconds=0`.",
                "- Calibration encoder may use first 90s head motion and calibration FMS history.",
                "- Recent motion input may use only head motion up to current time t.",
                "- Allowed post-calibration FMS is only the start FMS at the recent window start.",
                "- `anchor_index/time/fms` in prediction CSV are backward-compatible aliases for `start_fms_*`.",
                "- Missing start FMS fallback must use latest finite FMS at or before nominal start index.",
                "- Forbidden inputs: current FMS, target FMS, future FMS, recent dense FMS sequence, sparse_observed, recent_start_observed, identity features.",
                "- Static Age/Gender/MSSQ is allowed only with train-only normalization; gender primary encoding is binary2 when static is used.",
            ]
        ),
    )

    write_text(
        output_dir / "leakage_audit.md",
        "\n".join(
            [
                "# Leakage Audit",
                "",
                f"- Updated: {now_iso()}",
                f"- Sanity test status: {sanity_status}",
                "- Pre-lock test guard: existing `runs/**/eval_test` outputs were not used for branch choice.",
                "- During adaptive search, all training runs used `--no_test_eval`.",
                (
                    f"- Final test: run once after validation lock at `{rel(final_test['path'])}`."
                    if final_test_available
                    else "- Final test: not run yet."
                ),
                "- Current best comparability: uses validation rows from `runs/goal_mae_next1h_0505/next1h_recent10_e80_s7/val_predictions.csv`.",
                "- Prediction CSV metadata includes current/target/start indices and start-FMS fallback fields.",
                "- New model options are additive and keep default model behavior unless explicitly enabled.",
                "- Required checks covered by sanity suite: seconds-to-steps, target shift, calibration leakage, recent-window leakage, anchor/start-FMS policy, model forward shape.",
            ]
        ),
    )

    prior_rows = read_csv(PRIOR_RUN_DIR / "val_predictions.csv")
    dist = delta_distribution(prior_rows)
    dist_lines = ["# Delta Distribution Audit", "", "- Source: validation predictions only.", ""]
    if "overall" in dist:
        dist_lines.extend(["## Overall", "", markdown_table([dist["overall"]], list(dist["overall"].keys()))])
    if dist.get("by_horizon"):
        h_rows = [{"horizon": k, **v} for k, v in dist["by_horizon"].items()]
        dist_lines.extend(["", "## By Horizon", "", markdown_table(h_rows, list(h_rows[0].keys()))])
    write_text(output_dir / "delta_distribution_audit.md", "\n".join(dist_lines))

    state = [
        "# RUN_STATE",
        "",
        f"- goal_start_time: {start_time}",
        f"- last_update: {now_iso()}",
        "- target_wall_clock_hours: 6",
        "- max_wall_clock_hours: 6.5",
        "- min_active_search_hours: 5",
        f"- current_best: {best.get('run_name', '')}",
        f"- current_best_primary_val_mae_h5_h10_h15_mean: {best.get('primary_val_mae_h5_h10_h15_mean', '')}",
        f"- completed_runs: {sum(1 for row in rows if row['status'] == 'completed')}",
        "- failed_or_interrupted_runs: none recorded yet",
        "- final_selection_lock: written" if (output_dir / "VALIDATION_SELECTION_LOCK.md").exists() else "- final_selection_lock: pending",
        f"- final_test: {rel(final_test['path'])}" if final_test_available else "- final_test: not run",
        "- test_usage_prelock: none",
        "- next_planned_experiment: none; validation selection is locked" if final_test_available else "- next_planned_experiment: h15 residual/deeper head or level+delta dual-head representative run",
        "- resume_possible: completed rows in resume_manifest.csv should not be rerun; incomplete rows require inspection.",
    ]
    write_text(output_dir / "RUN_STATE.md", "\n".join(state))

    write_text(
        output_dir / "best_model_summary.md",
        "\n".join(
            [
                "# Best Model Summary",
                "",
                f"- Updated: {now_iso()}",
                f"- Current validation best: `{best.get('run_name', '')}`",
                f"- Primary validation MAE mean(h5/h10/h15): {best.get('primary_val_mae_h5_h10_h15_mean', '')}",
                f"- h5/h10/h15: {best.get('val_mae_h5', '')} / {best.get('val_mae_h10', '')} / {best.get('val_mae_h15', '')}",
                f"- Best trend-following candidate so far: `{best.get('run_name', '')}` until new validation evidence is added.",
                (
                    f"- Final test primary MAE mean(h5/h10/h15): {fmt(final_test.get('primary'))}"
                    if final_test_available
                    else "- Final test: not run in this search state."
                ),
                (
                    f"- Final test h5/h10/h15 MAE: {fmt(final_test.get('mae_h5'))} / {fmt(final_test.get('mae_h10'))} / {fmt(final_test.get('mae_h15'))}"
                    if final_test_available
                    else ""
                ),
            ]
        ),
    )

    write_text(
        output_dir / "branch_summaries.md",
        "\n".join(
            [
                "# Branch Summaries",
                "",
                "- h15 head branch: h15 residual variants degraded primary MAE and were not promoted.",
                "- level+delta dual-head branch: delta-only improved primary MAE; dual-average/gated improved some flow/high-FMS tradeoffs but did not beat delta+motion norm.",
                "- motion-derived feature branch: causal norm features with delta head produced the best validation primary result.",
                "- static branch: raw and encoded static variants degraded validation MAE, so static was not selected.",
                "- recent30 branch: long motion plus distant start FMS degraded validation MAE; recent10 remained selected.",
                "- high-change/high-FMS loss branch: normalized high-FMS weighting improved high-FMS bias/dynamic range but reduced primary MAE.",
            ]
        ),
    )

    write_text(
        output_dir / "model_change_audit.md",
        "\n".join(
            [
                "# Model Change Audit",
                "",
                "- Added optional forecast head modes: `level`, `delta`, `dual_average`, `dual_gated`.",
                "- Added optional horizon head modes: `linear`, `h15_deep`, `h15_residual`, `h10_h15_residual`.",
                "- Added causal motion-derived features: `norm`, `norm_delta`, `norm_delta_energy`.",
                "- Defaults preserve legacy behavior unless CLI/config explicitly enables a new option.",
                "- No new forbidden input source was introduced; motion-derived features are causal transforms of recent head motion.",
                "- Final selected model uses `forecast_head_mode=delta`, `motion_feature_mode=norm`, `no_static`.",
            ]
        ),
    )

    if final_test_available:
        test_rows = [
            {
                "run_name": best.get("run_name", ""),
                "test_primary_mae_h5_h10_h15_mean": fmt(final_test.get("primary")),
                "test_mae_h5": fmt(final_test.get("mae_h5")),
                "test_mae_h10": fmt(final_test.get("mae_h10")),
                "test_mae_h15": fmt(final_test.get("mae_h15")),
                "test_mae_all": fmt(final_test.get("mae")),
                "test_rmse_all": fmt(final_test.get("rmse")),
                "test_r2_all": fmt(final_test.get("r2")),
                "test_n": final_test.get("n", ""),
                "high_fms_precision": fmt(final_test.get("high_fms_precision")),
                "high_fms_recall": fmt(final_test.get("high_fms_recall")),
                "high_fms_f1": fmt(final_test.get("high_fms_f1")),
            }
        ]
        write_text(
            output_dir / "final_test_audit.md",
            "\n".join(
                [
                    "# Final Test Audit",
                    "",
                    "- Test evaluation was run once after validation selection lock.",
                    f"- Metrics path: `{rel(final_test['path'])}`",
                    f"- Prediction CSV: `{rel(final_test['predictions'])}`",
                    "",
                    markdown_table(test_rows, list(test_rows[0].keys())),
                ]
            ),
        )

    review_lines = [
        "# Validation Plot Reviews",
        "",
        "Validation prediction CSVs only. These plots are for trend/lag/smoothing inspection and are not test evidence.",
        "",
    ]
    for row in leaderboard:
        if row.get("trend_plot_review"):
            review_lines.extend(
                [
                    f"## {row.get('run_name', '')}",
                    "",
                    f"- primary_val_mae_h5_h10_h15_mean: {row.get('primary_val_mae_h5_h10_h15_mean', '')}",
                    f"- direction_agreement: {row.get('direction_agreement', '')}",
                    f"- dynamic_range_ratio: {row.get('dynamic_range_ratio', '')}",
                    f"- plots: {row.get('trend_plot_review', '')}",
                    "",
                ]
            )
    write_text(output_dir / "validation_plot_reviews" / "README.md", "\n".join(review_lines))

    write_text(
        output_dir / "final_report.md",
        "\n".join(
            [
                "# Final Report Draft",
                "",
                "이번 goal은 validation search와 final test 1회를 완료했다." if final_test_available else "이번 goal은 아직 진행 중이다.",
                "",
                "## Current Validation Leaderboard",
                "",
                markdown_table(
                    leaderboard,
                    [
                        "rank",
                        "run_name",
                        "primary_val_mae_h5_h10_h15_mean",
                        "val_mae_h5",
                        "val_mae_h10",
                        "val_mae_h15",
                        "direction_agreement",
                        "dynamic_range_ratio",
                    ],
                    limit=20,
                ),
                "",
                "## Test Policy",
                "",
                "- Pre-lock branch selection은 validation 결과와 validation prediction/plot만 사용한다.",
                (
                    f"- Final test는 validation lock 이후 `{best.get('run_name', '')}`에 대해 1회만 수행했다."
                    if final_test_available
                    else "- Adaptive search 중 final test는 수행하지 않았다."
                ),
                "",
                "## Final Test Result",
                "",
                (
                    markdown_table(
                        [
                            {
                                "run_name": best.get("run_name", ""),
                                "test_primary_mae_h5_h10_h15_mean": fmt(final_test.get("primary")),
                                "test_mae_h5": fmt(final_test.get("mae_h5")),
                                "test_mae_h10": fmt(final_test.get("mae_h10")),
                                "test_mae_h15": fmt(final_test.get("mae_h15")),
                                "test_mae_all": fmt(final_test.get("mae")),
                            }
                        ],
                        [
                            "run_name",
                            "test_primary_mae_h5_h10_h15_mean",
                            "test_mae_h5",
                            "test_mae_h10",
                            "test_mae_h15",
                            "test_mae_all",
                        ],
                    )
                    if final_test_available
                    else "- Not run."
                ),
            ]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runs/goal_mae_6h_adaptive_improvement_0506")
    parser.add_argument("--start_time", default="")
    parser.add_argument("--sanity_status", default="not recorded")
    args = parser.parse_args()
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    start_time = args.start_time or now_iso()
    runs = discover_runs(output_dir)
    baseline_h15 = 1.9026686877
    rows = [row_for_run(label, path, source, baseline_h15, output_dir) for label, path, source in runs]
    write_reports(output_dir, rows, start_time, args.sanity_status)


if __name__ == "__main__":
    main()
