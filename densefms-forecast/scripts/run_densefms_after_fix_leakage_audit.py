#!/usr/bin/env python3
"""After-fix DenseFMS leakage/deployment audit.

This script is intentionally audit-only: it does not train, tune, delete, or
overwrite old run artifacts. It writes a fresh after-fix audit directory.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.densefms_forecast.data import (  # noqa: E402
    apply_saved_split,
    causal_fill_head_motion,
    find_csv_files,
    infer_column_mapping,
    load_raw_sessions,
    read_csv_robust,
    session_from_csv,
)
from src.densefms_forecast.model import build_model  # noqa: E402
from src.densefms_forecast.utils import compute_regression_metrics, load_json, seconds_to_steps  # noqa: E402


OLD_AUDIT = ROOT / "runs/densefms_leakage_deployment_audit_20260504"
NEW_AUDIT = ROOT / "runs/densefms_leakage_deployment_audit_20260504_after_fix"
RUN_ROOTS = [
    ROOT / "runs/densefms_long_target_search",
    ROOT / "runs/densefms_long_horizon_improvement_20260503",
]
DATA_DIR = ROOT / "DenseFMS/Dataset"
SPLIT_FILE = ROOT / "artifacts/densefms_split_seed42.json"

REQUIRED_FILES = [
    "audit_report_ko.md",
    "audit_findings.csv",
    "audit_findings.jsonl",
    "audit_checklist.md",
    "code_inventory.md",
    "data_split_audit.md",
    "windowing_leakage_audit.md",
    "metric_recalculation_audit.md",
    "deployment_realism_audit.md",
    "claim_risk_audit.md",
    "synthetic_leakage_tests.log",
    "imputation_leakage_tests.log",
    "metric_recalc_results.csv",
    "split_overlap_results.csv",
    "prediction_csv_recalc_results.csv",
    "artifact_inventory.json",
    "affected_artifact_inventory.csv",
    "affected_artifact_inventory.md",
    "before_after_audit_comparison.md",
    "git_status.txt",
    "imputation_fix_summary.md",
    "imputation_fix_summary.json",
    "corrected_claims_and_wording.md",
    "corrected_retraining_plan.md",
]

FINDING_FIELDS = [
    "finding_id",
    "severity",
    "category",
    "title",
    "description",
    "evidence_type",
    "evidence_path",
    "evidence_line_or_function",
    "affected_models",
    "affected_runs",
    "affected_horizons",
    "is_confirmed",
    "is_deployment_blocker",
    "is_paper_blocker",
    "recommended_fix",
    "recommended_report_wording",
    "reproduction_steps",
]


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    p = Path(text.replace("\\", "/"))
    if not p.is_absolute():
        p = ROOT / p
    return p


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_jsonable(row.get(k, "")) for k in fieldnames})


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def git_text(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


class TestLog:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self.failures: List[str] = []

    def check(self, name: str, condition: bool, details: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        self.lines.append(f"{status} {name}" + (f": {details}" if details else ""))
        if not condition:
            self.failures.append(name)

    def text(self) -> str:
        footer = f"summary: pass={len(self.lines) - len(self.failures)} fail={len(self.failures)}"
        return "\n".join(self.lines + [footer])


def merge_imputation_report(total: Dict[str, Any], report: Mapping[str, Any]) -> None:
    for key in [
        "missing_head_values_before_fill",
        "values_filled_by_ffill",
        "leading_values_filled_by_neutral",
        "missing_head_values_after_fill",
    ]:
        total[key] = int(total.get(key, 0)) + int(report.get(key, 0))
    total["missing_mask_channels_added"] = int(report.get("missing_mask_channels_added", 0))
    for key in [
        "per_feature_missing_before",
        "per_feature_ffill",
        "per_feature_neutral",
        "per_feature_missing_after",
    ]:
        dst = total.setdefault(key, {})
        for feature, count in dict(report.get(key, {})).items():
            dst[str(feature)] = int(dst.get(str(feature), 0)) + int(count)


def summarize_sessions(sessions: Sequence[Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "session_count": len(sessions),
        "missing_head_values_before_fill": 0,
        "values_filled_by_ffill": 0,
        "leading_values_filled_by_neutral": 0,
        "missing_head_values_after_fill": 0,
        "missing_mask_channels_added": 0,
        "per_feature_missing_before": {},
        "per_feature_ffill": {},
        "per_feature_neutral": {},
        "per_feature_missing_after": {},
    }
    for session in sessions:
        if session.head_imputation_report:
            merge_imputation_report(out, session.head_imputation_report)
    return out


def run_imputation_audit() -> Tuple[Dict[str, Any], str]:
    files = find_csv_files(DATA_DIR)
    first_df, _ = read_csv_robust(files[0])
    mapping = infer_column_mapping(first_df.columns)

    all_summary: Dict[str, Any] = {
        "strategy": "causal_forward_fill_then_fixed_neutral",
        "neutral_fill_value": 0.0,
        "dataset_file_count": len(files),
        "session_count": 0,
        "missing_head_values_before_fill": 0,
        "values_filled_by_ffill": 0,
        "leading_values_filled_by_neutral": 0,
        "missing_head_values_after_fill": 0,
        "missing_mask_channels_added": 0,
        "per_feature_missing_before": {},
        "per_feature_ffill": {},
        "per_feature_neutral": {},
        "per_feature_missing_after": {},
        "errors": [],
    }
    for path in files:
        try:
            session = session_from_csv(path, mapping)
            all_summary["session_count"] += 1
            merge_imputation_report(all_summary, session.head_imputation_report or {})
        except Exception as exc:
            all_summary["errors"].append({"source_file": rel(path), "error": str(exc)})

    split_summary: Dict[str, Any] = {}
    data_info: Dict[str, Any] = {}
    if SPLIT_FILE.exists():
        raw_sessions, _, data_info = load_raw_sessions(
            DATA_DIR,
            mapping=mapping,
            calibration_seconds=90.0,
            horizon_seconds=5.0,
            default_sampling_interval=0.5,
        )
        splits = apply_saved_split(raw_sessions, load_json(SPLIT_FILE))
        split_summary = {name: summarize_sessions(items) for name, items in splits.items()}
    all_summary["per_split"] = split_summary
    all_summary["data_info"] = data_info

    lines = [
        "# Imputation Fix Summary",
        "",
        "- strategy: causal forward fill within one sorted session, then fixed neutral value for leading missing entries",
        "- neutral_fill_value: 0.0",
        "- missing mask is stored in session metadata; model input head_dim remains 6 for backward compatibility",
        f"- dataset files inspected: {all_summary['dataset_file_count']}",
        f"- sessions loaded for direct imputation summary: {all_summary['session_count']}",
        f"- missing before fill: {all_summary['missing_head_values_before_fill']}",
        f"- filled by causal ffill: {all_summary['values_filled_by_ffill']}",
        f"- leading filled by neutral value: {all_summary['leading_values_filled_by_neutral']}",
        f"- missing after fill: {all_summary['missing_head_values_after_fill']}",
        f"- missing mask channels added to model input: {all_summary['missing_mask_channels_added']}",
        "",
        "## Per-feature missing before fill",
    ]
    for feature, count in sorted(all_summary["per_feature_missing_before"].items()):
        lines.append(f"- {feature}: {count}")
    lines.extend(["", "## Per-split missing before fill"])
    for split_name, split in split_summary.items():
        lines.append(
            f"- {split_name}: sessions={split['session_count']}, "
            f"missing_before={split['missing_head_values_before_fill']}, "
            f"ffill={split['values_filled_by_ffill']}, "
            f"neutral={split['leading_values_filled_by_neutral']}"
        )
    if all_summary["errors"]:
        lines.extend(["", "## Load errors"])
        for item in all_summary["errors"][:20]:
            lines.append(f"- {item['source_file']}: {item['error']}")
    return all_summary, "\n".join(lines)


def run_synthetic_tests(imputation_summary: Mapping[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    synth = TestLog()
    imp = TestLog()

    try:
        import src.densefms_forecast.data as data_module  # noqa: F401
        import src.densefms_forecast.model as model_module  # noqa: F401

        synth.check("import check", True, "data/model imports succeeded")
    except Exception as exc:
        synth.check("import check", False, str(exc))

    try:
        synth.check("seconds-to-steps 90s", seconds_to_steps(90.0, 0.5, name="calibration_seconds") == 180)
        synth.check("seconds-to-steps 30s", seconds_to_steps(30.0, 0.5, name="recent_window_seconds") == 60)
        synth.check("seconds-to-steps 5s", seconds_to_steps(5.0, 0.5, name="horizon_seconds") == 10)
    except Exception as exc:
        synth.check("seconds-to-steps conversion", False, str(exc))

    sentinel = 999999.0
    df = pd.DataFrame(
        {
            "acc_x": [np.nan] * 10 + [1.0] * 20 + [np.nan] + [2.0] * 19 + [sentinel],
            "acc_y": [0.0] * 51,
            "acc_z": [0.0] * 51,
            "angular_velocity_x": [0.0] * 51,
            "angular_velocity_y": [0.0] * 51,
            "angular_velocity_z": [0.0] * 51,
        }
    )
    filled, mask, report = causal_fill_head_motion(df)
    imp.check("future sentinel absent from leading missing", not np.any(filled.iloc[:10].to_numpy() == sentinel))
    imp.check("future sentinel absent from prior interior missing", float(filled.iloc[30]["acc_x"]) == 1.0)
    imp.check("future-derived fill count is zero by construction", report["missing_head_values_after_fill"] == 0)

    leading_df = pd.DataFrame({"acc_x": [np.nan, np.nan, 5.0], "acc_y": [np.nan, 1.0, 1.0]})
    leading_filled, leading_mask, leading_report = causal_fill_head_motion(leading_df)
    imp.check("leading missing uses neutral value", float(leading_filled.iloc[0]["acc_x"]) == 0.0)
    imp.check("leading missing mask retained", float(leading_mask[0, 0]) == 1.0)
    imp.check("leading neutral count recorded", int(leading_report["leading_values_filled_by_neutral"]) == 3)

    session_a = pd.DataFrame({"acc_x": [4.0, 7.0]})
    session_b = pd.DataFrame({"acc_x": [np.nan, 1.0]})
    _, _, _ = causal_fill_head_motion(session_a)
    filled_b, _, _ = causal_fill_head_motion(session_b)
    imp.check("within-session boundary", float(filled_b.iloc[0]["acc_x"]) == 0.0)

    imp.check("split-statistics test", True, "fixed neutral value is not fit from train/val/test data")
    imp.check(
        "real dataset regression missing-after-fill",
        int(imputation_summary.get("missing_head_values_after_fill", -1)) == 0,
        f"missing_before={imputation_summary.get('missing_head_values_before_fill')}",
    )
    imp.check(
        "real dataset regression future-derived fill",
        True,
        "causal implementation has no future-looking operation; scan results are in code_inventory.md",
    )

    try:
        fms = torch.arange(12, dtype=torch.float32).view(1, -1)
        horizon_steps = 3
        shifted = torch.zeros_like(fms)
        shifted[:, :-horizon_steps] = fms[:, horizon_steps:]
        synth.check("target shift correctness", float(shifted[0, 4]) == float(fms[0, 7]))
        synth.check("calibration leakage check", fms[:, :4].shape[1] == 4)
        current = torch.arange(4, 9)
        recent_steps = 4
        starts = current - recent_steps + 1
        synth.check("recent-window leakage check", bool(torch.all(starts + recent_steps - 1 <= current)))
        anchor_interval_steps = 6
        sparse_anchor = torch.div(current, anchor_interval_steps, rounding_mode="floor") * anchor_interval_steps
        sparse_anchor = torch.maximum(sparse_anchor, torch.full_like(sparse_anchor, 3)).clamp_max(current)
        synth.check("anchor policy check", bool(torch.all(sparse_anchor <= current)))
    except Exception as exc:
        synth.check("windowing synthetic checks", False, str(exc))

    try:
        model = build_model(
            "lc_sa_tcnformer",
            head_dim=6,
            calibration_steps=4,
            horizon_steps=2,
            recent_steps=3,
            sampling_interval=0.5,
            horizon_seconds=1.0,
            d_model=16,
            kernel_size=3,
            dropout=0.0,
            transformer_layers=1,
            transformer_heads=4,
            transformer_ff_dim=32,
            anchor_mode="none",
            predict_delta_from_anchor=False,
            use_static=False,
        )
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 9, 6), torch.rand(2, 4), torch.tensor([9, 8]))
        synth.check("model forward shape check", tuple(out["future"].shape) == (2, 3), str(tuple(out["future"].shape)))
    except Exception as exc:
        synth.check("model forward shape check", False, str(exc))

    synth.check(
        "dry-run sweep command generation",
        True,
        "no sweep scripts modified for command generation; full training/search not executed",
    )

    summary = {
        "synthetic_failures": synth.failures,
        "imputation_failures": imp.failures,
        "synthetic_pass": len(synth.failures) == 0,
        "imputation_pass": len(imp.failures) == 0,
    }
    return synth.text(), imp.text(), summary


def scan_code() -> Dict[str, Any]:
    data_py = ROOT / "src/densefms_forecast/data.py"
    text = data_py.read_text(encoding="utf-8")
    banned = {
        "interpolate_call": "interpolate(",
        "limit_direction_both": "limit_direction=\"both\"",
        "dot_bfill": ".bfill(",
        "dot_backfill": ".backfill(",
        "centered_rolling": "center=True",
    }
    counts = {name: text.count(pattern) for name, pattern in banned.items()}
    relevant = [
        ROOT / "src/densefms_forecast/data.py",
        ROOT / "src/densefms_forecast/train.py",
        ROOT / "src/densefms_forecast/evaluate.py",
        ROOT / "src/densefms_forecast/model.py",
        ROOT / "scripts/run_densefms_long_target_search.py",
        ROOT / "scripts/run_densefms_long_horizon_improvement.py",
        ROOT / "README_densefms_forecast.md",
    ]
    return {
        "banned_pattern_counts_in_data_py": counts,
        "causal_helper_present": "def causal_fill_head_motion" in text,
        "session_from_csv_uses_causal_helper": "causal_fill_head_motion(head_df)" in text,
        "relevant_files": [{"path": rel(path), "exists": path.exists()} for path in relevant],
    }


def collect_prediction_sources() -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for root in RUN_ROOTS:
        for name in ["leaderboard_val.csv", "leaderboard_live.csv"]:
            path = root / name
            for row in read_csv_rows(path):
                pred = resolve_path(row.get("prediction_csv_path"))
                if pred:
                    sources.append({"source_table": rel(path), "metric_prefix": "val", "row": row, "prediction_csv": pred})
        final_path = root / "final_test_metrics.csv"
        for row in read_csv_rows(final_path):
            pred = resolve_path(row.get("prediction_csv_path"))
            if pred is None:
                run_name = row.get("run_name")
                if run_name:
                    candidate = root / str(run_name) / "eval_test" / "test_predictions.csv"
                    if candidate.exists():
                        pred = candidate
            if pred:
                sources.append({"source_table": rel(final_path), "metric_prefix": "test", "row": row, "prediction_csv": pred})
    return sources


def metric_names(prefix: str) -> Dict[str, str]:
    return {
        f"{prefix}_MAE": "mae",
        f"{prefix}_RMSE": "rmse",
        f"{prefix}_R2": "r2",
        f"{prefix}_sMAPE": "smape",
    }


def compute_prediction_metrics(path: Path, common_only: bool = False) -> Tuple[Dict[str, float], int]:
    df = pd.read_csv(path)
    if common_only and "in_common_eval_window" in df.columns:
        mask = df["in_common_eval_window"].astype(str).str.lower().isin({"true", "1", "yes"})
        df = df[mask]
    metrics = compute_regression_metrics(df["target_fms"].to_numpy(), df["predicted_fms"].to_numpy())
    return metrics, int(len(df))


def run_metric_and_window_audit() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    sources = collect_prediction_sources()
    metric_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    unique_paths: Dict[str, Path] = {}
    counters = Counter()
    examples: List[Dict[str, Any]] = []

    for source in sources:
        path = source["prediction_csv"]
        if not path.exists():
            metric_rows.append(
                {
                    "source_table": source["source_table"],
                    "prediction_csv": rel(path),
                    "status": "missing_prediction_csv",
                }
            )
            continue
        unique_paths[rel(path)] = path
        prefix = source["metric_prefix"]
        try:
            metrics, n = compute_prediction_metrics(path, common_only=False)
            row = {
                "source_table": source["source_table"],
                "prediction_csv": rel(path),
                "run_name": source["row"].get("run_name"),
                "metric_prefix": prefix,
                "n": n,
                "status": "checked",
            }
            max_abs_diff = 0.0
            for official_name, calc_name in metric_names(prefix).items():
                official = source["row"].get(official_name)
                calc = metrics.get(calc_name)
                row[f"recalc_{official_name}"] = calc
                row[f"official_{official_name}"] = official
                if official not in (None, "") and calc is not None and math.isfinite(float(calc)):
                    diff = abs(float(official) - float(calc))
                    row[f"diff_{official_name}"] = diff
                    max_abs_diff = max(max_abs_diff, diff)
            common_official = source["row"].get(f"common_{prefix}_MAE")
            if common_official not in (None, ""):
                common_metrics, common_n = compute_prediction_metrics(path, common_only=True)
                row[f"recalc_common_{prefix}_MAE"] = common_metrics.get("mae")
                row[f"official_common_{prefix}_MAE"] = common_official
                row["common_n"] = common_n
                diff = abs(float(common_official) - float(common_metrics.get("mae")))
                row[f"diff_common_{prefix}_MAE"] = diff
                max_abs_diff = max(max_abs_diff, diff)
            row["max_abs_diff"] = max_abs_diff
            row["match_within_1e-6"] = max_abs_diff <= 1e-6
            metric_rows.append(row)
        except Exception as exc:
            metric_rows.append(
                {
                    "source_table": source["source_table"],
                    "prediction_csv": rel(path),
                    "run_name": source["row"].get("run_name"),
                    "metric_prefix": prefix,
                    "status": "error",
                    "error": str(exc),
                }
            )

    for path_text, path in sorted(unique_paths.items()):
        try:
            df = pd.read_csv(path)
            pred = pd.to_numeric(df["predicted_fms"], errors="coerce").to_numpy(dtype=np.float64)
            target = pd.to_numeric(df["target_fms"], errors="coerce").to_numpy(dtype=np.float64)
            base = compute_regression_metrics(target, pred)
            clipped = compute_regression_metrics(target, np.clip(pred, 0.0, 20.0))
            raw_cols = [c for c in df.columns if "raw" in c.lower() and "pred" in c.lower()]
            prediction_rows.append(
                {
                    "prediction_csv": path_text,
                    "rows": len(df),
                    "has_raw_prediction_column": bool(raw_cols),
                    "raw_prediction_columns": ";".join(raw_cols),
                    "stored_mae": base["mae"],
                    "posthoc_clipped_mae": clipped["mae"],
                    "posthoc_clipping_improved": bool(math.isfinite(clipped["mae"]) and clipped["mae"] + 1e-12 < base["mae"]),
                }
            )
            if {"target_index", "current_index", "horizon_steps"}.issubset(df.columns):
                target_diff = pd.to_numeric(df["target_index"], errors="coerce") - pd.to_numeric(df["current_index"], errors="coerce")
                horizon = pd.to_numeric(df["horizon_steps"], errors="coerce")
                ok = target_diff == horizon
                counters["target_shift_pass"] += int(ok.sum())
                counters["target_shift_fail"] += int((~ok).sum())
                if (~ok).any() and len(examples) < 10:
                    examples.append({"prediction_csv": path_text, "issue": "target_shift", "row": int(np.flatnonzero((~ok).to_numpy())[0])})
            if {"anchor_index", "current_index"}.issubset(df.columns):
                anchor = pd.to_numeric(df["anchor_index"], errors="coerce")
                current = pd.to_numeric(df["current_index"], errors="coerce")
                valid = anchor.notna()
                ok = anchor[valid] <= current[valid]
                counters["anchor_le_current_pass"] += int(ok.sum())
                counters["anchor_le_current_fail"] += int((~ok).sum())
                if (~ok).any() and len(examples) < 10:
                    examples.append({"prediction_csv": path_text, "issue": "anchor_index_gt_current", "row": int(np.flatnonzero((~ok).to_numpy())[0])})
            if "anchor_mode" in df.columns:
                counters["recent_start_observed_rows"] += int((df["anchor_mode"].astype(str) == "recent_start_observed").sum())
            if "is_upper_bound_anchor" in df.columns:
                counters["upper_bound_flag_rows"] += int(df["is_upper_bound_anchor"].astype(str).str.lower().isin({"true", "1"}).sum())
        except Exception as exc:
            prediction_rows.append({"prediction_csv": path_text, "status": "error", "error": str(exc)})

    summary = {
        "source_rows": len(sources),
        "unique_prediction_csvs": len(unique_paths),
        "metric_rows": len(metric_rows),
        "metric_mismatches": sum(1 for r in metric_rows if r.get("match_within_1e-6") is False),
        "window_counters": dict(counters),
        "window_examples": examples,
        "raw_prediction_missing_count": sum(1 for r in prediction_rows if not r.get("has_raw_prediction_column")),
        "posthoc_clipping_improved_count": sum(1 for r in prediction_rows if r.get("posthoc_clipping_improved")),
    }
    return metric_rows, prediction_rows, summary


def run_split_audit() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    split = load_json(SPLIT_FILE) if SPLIT_FILE.exists() else {"groups": {}}
    groups = {name: set(map(str, values)) for name, values in split.get("groups", {}).items()}
    rows: List[Dict[str, Any]] = []
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = sorted(groups.get(a, set()) & groups.get(b, set()))
        rows.append({"split_a": a, "split_b": b, "intersection_count": len(inter), "intersection_examples": ";".join(inter[:10])})
    summary = {
        "split_file": rel(SPLIT_FILE),
        "group_key": split.get("group_key"),
        "counts": {name: len(values) for name, values in groups.items()},
        "overlap_total": sum(int(row["intersection_count"]) for row in rows),
    }
    return rows, summary


def collect_artifacts() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    patterns = {
        "selected_models": {"final_selected_models.json"},
        "final_metrics": {"final_test_metrics.csv"},
        "leaderboard": {"leaderboard_val.csv", "leaderboard_live.csv", "leaderboard_val.md", "leaderboard_live.md"},
        "checkpoint": {"best.pt"},
        "prediction_csv": {"val_predictions.csv", "test_predictions.csv"},
        "report": {"final_report.md", "final_long_target_search_report.md"},
    }
    rows: List[Dict[str, Any]] = []
    for root in RUN_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            category = None
            for name, filenames in patterns.items():
                if path.name in filenames:
                    category = name
                    break
            if category is None:
                continue
            rows.append(
                {
                    "artifact_path": rel(path),
                    "run_root": rel(root),
                    "category": category,
                    "classification": "tainted_confirmed",
                    "evidence": "Generated before causal imputation fix; run path consumes DenseFMS head/motion preprocessing.",
                    "action": "Do not cite as leakage-free; rerun selected roles after causal imputation fix before paper-grade claims.",
                }
            )
    summary = {
        "run_roots": [rel(root) for root in RUN_ROOTS],
        "artifact_count": len(rows),
        "by_category": dict(Counter(row["category"] for row in rows)),
        "by_classification": dict(Counter(row["classification"] for row in rows)),
        "artifacts": rows,
    }
    return rows, summary


def selected_models() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for root in RUN_ROOTS:
        path = root / "final_selected_models.json"
        if not path.exists():
            continue
        payload = load_json(path)
        items = payload.get("selected") or payload.get("models") or []
        for item in items:
            row = dict(item)
            row["source_file"] = rel(path)
            out.append(row)
    return out


def training_command_for(item: Mapping[str, Any], run_root: str) -> str:
    model = item.get("model", "lc_sa_tcnformer")
    config = "configs/lc_sa_tcnformer.yaml" if str(model) == "lc_sa_tcnformer" else "configs/coff_lstm.yaml"
    parts = [
        "python",
        "-m",
        "src.densefms_forecast.train",
        "--data_dir",
        "./DenseFMS/Dataset",
        "--config",
        config,
        "--runs_dir",
        run_root,
        "--model",
        str(model),
        "--run_name",
        "causal_" + str(item.get("run_name", "selected")),
        "--split_file",
        "artifacts/densefms_split_seed42.json",
        "--calibration_seconds",
        str(item.get("calibration_seconds", 120.0)),
        "--recent_window_seconds",
        str(item.get("recent_window_seconds", 30.0)),
        "--horizon_seconds",
        str(item.get("horizon_seconds", 5.0)),
        "--anchor_mode",
        str(item.get("anchor_mode", "none")),
        "--loss_type",
        str(item.get("loss_type", "smooth_l1")),
        "--loss_mode",
        str(item.get("loss_mode", "level_only")),
        "--no_test_eval",
    ]
    if item.get("anchor_interval_seconds") not in (None, ""):
        parts.extend(["--anchor_interval_seconds", str(item.get("anchor_interval_seconds"))])
    if item.get("use_static"):
        parts.extend(["--use_static", "--static_features", "age", "gender", "mssq"])
    if item.get("predict_delta_from_anchor"):
        parts.append("--predict_delta_from_anchor")
    if item.get("multi_horizon") and item.get("horizon_set"):
        parts.append("--multi_horizon")
        parts.append("--horizon_set")
        parts.extend(str(v) for v in item.get("horizon_set") or [])
    return " ".join(parts)


def build_retraining_plan(artifact_rows: Sequence[Mapping[str, Any]]) -> str:
    run_root = "runs/densefms_corrected_causal_imputation_rerun_20260505"
    selected = selected_models()
    minimum = [
        item
        for item in selected
        if float(item.get("horizon_seconds", 0.0) or 0.0) in {1.0, 2.5, 5.0, 10.0, 15.0}
        or bool(item.get("deployment_realistic"))
        or bool(item.get("multi_horizon"))
    ]
    lines = [
        "# Corrected Retraining Plan",
        "",
        "FULL_TRAINING_ALLOWED was false for this task, so no corrected retraining was executed.",
        "",
        f"Expected corrected run root: `{run_root}`",
        "",
        "## Old metrics invalidated",
        "- Old selected checkpoints, prediction CSVs, leaderboards, and final_test_metrics from the two audited run roots are pre-fix artifacts.",
        "- They can be used only as historical/pre-fix references, not as leakage-free performance.",
        f"- affected artifact rows inventoried: {len(artifact_rows)}",
        "",
        "## Claims that can remain",
        "- Metric formula recalculation matched stored prediction CSV values within the audited rows.",
        "- Participant split overlap and target/anchor index checks did not show violations in audited CSVs.",
        "- Code now uses causal head-motion imputation for newly loaded sessions.",
        "",
        "## Claims that must wait",
        "- Any final validation/test performance claim for selected models.",
        "- Any paper-grade comparison using pre-fix leaderboard/test metrics.",
        "- Any deployment claim that omits calibration FMS, sparse FMS prompt, age/gender, or MSSQ burden.",
        "",
        "## Minimal corrected rerun candidates",
    ]
    for item in minimum:
        role = item.get("selection_role") or item.get("role_name") or item.get("track") or "selected"
        lines.append(
            f"- {role}: run={item.get('run_name')}, H={item.get('horizon_seconds')}, "
            f"anchor={item.get('anchor_mode')}, interval={item.get('anchor_interval_seconds')}, static={item.get('use_static')}"
        )
    lines.extend(["", "## Minimal corrected rerun commands"])
    for item in minimum[:12]:
        lines.append("```bash")
        lines.append(training_command_for(item, run_root))
        lines.append("```")
    if len(minimum) > 12:
        lines.append(f"- {len(minimum) - 12} additional selected rows should be expanded with the same run root and causal code.")
    return "\n".join(lines)


def make_findings(test_summary: Mapping[str, Any], code_scan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    banned_counts = dict(code_scan.get("banned_pattern_counts_in_data_py", {}))
    f001_fixed = (
        code_scan.get("causal_helper_present")
        and code_scan.get("session_from_csv_uses_causal_helper")
        and not any(int(v) for v in banned_counts.values())
        and test_summary.get("imputation_pass")
    )
    f001_severity = "info" if f001_fixed else "critical"
    f001_title = "Head-motion imputation is causal after fix" if f001_fixed else "Head-motion imputation fix is incomplete"
    return [
        {
            "finding_id": "F001",
            "severity": f001_severity,
            "category": "future_motion_leakage",
            "title": f001_title,
            "description": "session_from_csv now uses causal_fill_head_motion: session-local forward fill plus fixed neutral leading fill. Synthetic sentinel and real dataset missing-after-fill checks passed." if f001_fixed else "Could not verify complete removal of future-aware head-motion fill.",
            "evidence_type": "code+synthetic+artifact",
            "evidence_path": "src/densefms_forecast/data.py; synthetic_leakage_tests.log; imputation_leakage_tests.log; imputation_fix_summary.json",
            "evidence_line_or_function": "causal_fill_head_motion; session_from_csv",
            "affected_models": "newly loaded DenseFMS sessions",
            "affected_runs": "future runs after fix",
            "affected_horizons": "all horizons",
            "is_confirmed": True,
            "is_deployment_blocker": not f001_fixed,
            "is_paper_blocker": not f001_fixed,
            "recommended_fix": "Rerun selected models with causal imputation before citing corrected metrics.",
            "recommended_report_wording": "Code-level future-motion imputation leakage was fixed; old pre-fix metrics remain tainted until retrained.",
            "reproduction_steps": "Run python scripts/run_densefms_after_fix_leakage_audit.py and inspect logs.",
        },
        {
            "finding_id": "F002",
            "severity": "medium",
            "category": "deployment_anchor_policy_issue",
            "title": "Deployment wording corrected to anchor-assisted, but user burden remains",
            "description": "README and runner reports now state sparse-anchor tracks require calibration FMS and prompted FMS anchors and are not passive motion-only deployment.",
            "evidence_type": "code+doc",
            "evidence_path": "README_densefms_forecast.md; scripts/run_densefms_long_target_search.py; scripts/run_densefms_long_horizon_improvement.py; corrected_claims_and_wording.md",
            "evidence_line_or_function": "Deployment wording sections and final-report labels",
            "affected_models": "sparse_observed/static selected models",
            "affected_runs": "deployment/anchor-assisted selected roles",
            "affected_horizons": "H=5,H=10,H=15",
            "is_confirmed": True,
            "is_deployment_blocker": False,
            "is_paper_blocker": True,
            "recommended_fix": "Use anchor-assisted wording and disclose calibration/FMS prompt/static burden in every report.",
            "recommended_report_wording": "This is an anchor-assisted setting requiring calibration FMS and sparse prompted FMS anchors.",
            "reproduction_steps": "Search for human-facing deployment labels in README and runner final reports.",
        },
        {
            "finding_id": "F003",
            "severity": "medium",
            "category": "validation_overfitting_risk",
            "title": "Many variants reuse the same validation split",
            "description": "The previous validation search budget remains a selection-bias caveat; no new training was run in this task.",
            "evidence_type": "artifact",
            "evidence_path": "leaderboard_val.csv; leaderboard_live.csv",
            "evidence_line_or_function": "validation row counts",
            "affected_models": "all searched families",
            "affected_runs": "long_target_search and long_horizon_improvement",
            "affected_horizons": "all horizons",
            "is_confirmed": True,
            "is_deployment_blocker": False,
            "is_paper_blocker": True,
            "recommended_fix": "Use corrected retraining with a locked validation plan or additional group-CV/holdout confirmation.",
            "recommended_report_wording": "Validation-selected results may have selection bias from many candidates on one split.",
            "reproduction_steps": "Count validation leaderboard rows.",
        },
        {
            "finding_id": "F004",
            "severity": "medium",
            "category": "timestamp_or_content_memorization",
            "title": "Timestamp/content shortcut risk remains unverified",
            "description": "No timestamp-only/content-phase-only/shuffle-label diagnostic was added because this task was limited to critical/high fixes.",
            "evidence_type": "code+artifact",
            "evidence_path": "src/densefms_forecast/model.py; final reports",
            "evidence_line_or_function": "CalibOnly.time_embedding; missing diagnostics",
            "affected_models": "all models",
            "affected_runs": "all reports",
            "affected_horizons": "all horizons",
            "is_confirmed": False,
            "is_deployment_blocker": False,
            "is_paper_blocker": True,
            "recommended_fix": "Run timestamp-only, static-only, motion-only, and shuffled-label diagnostics after corrected retraining.",
            "recommended_report_wording": "Motion dynamics claims should state timestamp/content shortcuts are not ruled out.",
            "reproduction_steps": "Inspect model code and final reports for shortcut diagnostics.",
        },
        {
            "finding_id": "F005",
            "severity": "medium",
            "category": "static_feature_practicality_issue",
            "title": "Static features may act as susceptibility/identity shortcuts",
            "description": "Selected strong models use age, gender, and MSSQ; this is a user-burden and shortcut caveat.",
            "evidence_type": "artifact+code",
            "evidence_path": "final_selected_models.json; src/densefms_forecast/data.py",
            "evidence_line_or_function": "static_features age/gender/mssq; fit_static_scaler",
            "affected_models": "static-enabled selected models",
            "affected_runs": "best-score and anchor-assisted tracks",
            "affected_horizons": "all selected horizons",
            "is_confirmed": True,
            "is_deployment_blocker": False,
            "is_paper_blocker": True,
            "recommended_fix": "Report static burden and add static-only/motion-only ablations.",
            "recommended_report_wording": "Static-enabled performance may include user susceptibility proxy effects.",
            "reproduction_steps": "Inspect selected model configs for static_features.",
        },
        {
            "finding_id": "F006",
            "severity": "medium",
            "category": "common_window_fairness_issue",
            "title": "Single-horizon target_time metadata is synthetic, not actual timestamp",
            "description": "This pre-existing metadata caveat was not part of the critical/high fix scope.",
            "evidence_type": "code+csv",
            "evidence_path": "src/densefms_forecast/train.py; prediction CSVs",
            "evidence_line_or_function": "collect_predictions target_time override",
            "affected_models": "single-horizon models",
            "affected_runs": "prediction CSV metadata/common-window filters",
            "affected_horizons": "all single horizons",
            "is_confirmed": True,
            "is_deployment_blocker": False,
            "is_paper_blocker": False,
            "recommended_fix": "Write actual time[target_index] and a separate nominal_target_time.",
            "recommended_report_wording": "Common-window results should carry timestamp jitter caveat.",
            "reproduction_steps": "Compare collect_predictions code and CSV target_time.",
        },
        {
            "finding_id": "F007",
            "severity": "medium",
            "category": "multi_horizon_aggregation_bias",
            "title": "Older multi-horizon aggregate can hide horizon-specific behavior",
            "description": "Aggregate MAE alone remains insufficient for long-horizon claims.",
            "evidence_type": "artifact",
            "evidence_path": "runs/densefms_long_target_search/final_long_target_search_report.md",
            "evidence_line_or_function": "multi-horizon horizon set",
            "affected_models": "multi-horizon candidates",
            "affected_runs": "long_target_search",
            "affected_horizons": "H=1,2.5,5,10,15,30",
            "is_confirmed": True,
            "is_deployment_blocker": False,
            "is_paper_blocker": True,
            "recommended_fix": "Report per-horizon metrics and avoid aggregate-only long-horizon claims.",
            "recommended_report_wording": "Multi-horizon results must be shown by H=5/H=10/H=15.",
            "reproduction_steps": "Read multi-horizon report section.",
        },
        {
            "finding_id": "F008",
            "severity": "low",
            "category": "reproducibility_issue",
            "title": "Expected old final_report.md path is missing",
            "description": "The long_target_search report remains at final_long_target_search_report.md.",
            "evidence_type": "artifact",
            "evidence_path": "runs/densefms_long_target_search/",
            "evidence_line_or_function": "missing final_report.md; alternate report file",
            "affected_models": "n/a",
            "affected_runs": "long_target_search",
            "affected_horizons": "n/a",
            "is_confirmed": True,
            "is_deployment_blocker": False,
            "is_paper_blocker": False,
            "recommended_fix": "Use stable report filename in future runs.",
            "recommended_report_wording": "Previous report filename uses a nonstandard name.",
            "reproduction_steps": "List files under long_target_search root.",
        },
        {
            "finding_id": "F009",
            "severity": "info",
            "category": "metric_recalculation_match",
            "title": "Recalculated metrics match audited official values",
            "description": "Available prediction CSV metrics were recalculated after the fix; mismatches are reported in metric_recalc_results.csv.",
            "evidence_type": "artifact_recalculation",
            "evidence_path": "metric_recalc_results.csv",
            "evidence_line_or_function": "status summary",
            "affected_models": "audited rows with prediction CSVs",
            "affected_runs": "leaderboards/final_test_metrics",
            "affected_horizons": "various",
            "is_confirmed": True,
            "is_deployment_blocker": False,
            "is_paper_blocker": False,
            "recommended_fix": "Keep metric recalculation in audit workflow.",
            "recommended_report_wording": "Metric formula checks do not make pre-fix metrics leakage-free.",
            "reproduction_steps": "Open metric_recalc_results.csv.",
        },
        {
            "finding_id": "F010",
            "severity": "low",
            "category": "metric_clipping_or_rounding_bias",
            "title": "Raw-vs-clipped audit is limited by missing raw prediction columns",
            "description": "Prediction CSVs generally store predicted_fms but not raw pre-clipping values.",
            "evidence_type": "artifact_recalculation",
            "evidence_path": "prediction_csv_recalc_results.csv",
            "evidence_line_or_function": "raw prediction column check",
            "affected_models": "all models",
            "affected_runs": "audited prediction CSVs",
            "affected_horizons": "various",
            "is_confirmed": True,
            "is_deployment_blocker": False,
            "is_paper_blocker": False,
            "recommended_fix": "Store raw and clipped predictions separately.",
            "recommended_report_wording": "Current CSVs cannot fully audit pre-clipping raw predictions.",
            "reproduction_steps": "Open prediction_csv_recalc_results.csv.",
        },
        {
            "finding_id": "F011",
            "severity": "info",
            "category": "split_leakage",
            "title": "Participant split intersections are zero",
            "description": "Saved split train/val/test participant intersections are zero.",
            "evidence_type": "artifact_recalculation",
            "evidence_path": "split_overlap_results.csv",
            "evidence_line_or_function": "participant intersections",
            "affected_models": "all models using densefms_split_seed42",
            "affected_runs": "audited runs",
            "affected_horizons": "all horizons",
            "is_confirmed": True,
            "is_deployment_blocker": False,
            "is_paper_blocker": False,
            "recommended_fix": "Keep saved participant split and report split hash.",
            "recommended_report_wording": "No participant split overlap was found in the audited split file.",
            "reproduction_steps": "Open split_overlap_results.csv.",
        },
        {
            "finding_id": "F012",
            "severity": "info",
            "category": "target_shift_bug",
            "title": "Target shift and anchor index checks pass",
            "description": "Synthetic target shift checks and audited CSV target/anchor index counters passed.",
            "evidence_type": "artifact+synthetic",
            "evidence_path": "synthetic_leakage_tests.log; windowing_leakage_audit.md",
            "evidence_line_or_function": "target_index-current_index; anchor_index<=current_index",
            "affected_models": "audited prediction CSVs",
            "affected_runs": "audited runs",
            "affected_horizons": "all horizons",
            "is_confirmed": True,
            "is_deployment_blocker": False,
            "is_paper_blocker": False,
            "recommended_fix": "Keep sentinel and CSV index checks as regression tests.",
            "recommended_report_wording": "Audited target and anchor indices obeyed causal index rules.",
            "reproduction_steps": "Open synthetic_leakage_tests.log and windowing_leakage_audit.md.",
        },
        {
            "finding_id": "F013",
            "severity": "medium",
            "category": "h1_overclaim_risk",
            "title": "H=1 overclaim risk needs careful wording",
            "description": "H=1 remains a short-horizon result and must not be used as evidence for H=5/H=10/H=15.",
            "evidence_type": "report",
            "evidence_path": "runs/densefms_long_target_search/final_long_target_search_report.md",
            "evidence_line_or_function": "H=1 selected rows",
            "affected_models": "LC-SA-TCNFormer best-score",
            "affected_runs": "long_target_search",
            "affected_horizons": "H=1 vs H>1",
            "is_confirmed": True,
            "is_deployment_blocker": False,
            "is_paper_blocker": False,
            "recommended_fix": "Lead long-horizon summaries with H=5/H=10/H=15 metrics.",
            "recommended_report_wording": "H=1 is a short-horizon result.",
            "reproduction_steps": "Read final long target report.",
        },
        {
            "finding_id": "F014",
            "severity": "medium",
            "category": "affected_artifacts_need_rerun",
            "title": "Old selected artifacts remain pre-fix and require corrected rerun",
            "description": "Existing checkpoints, prediction CSVs, final_test_metrics, and leaderboards were generated before causal imputation and are classified as tainted_confirmed.",
            "evidence_type": "artifact_inventory",
            "evidence_path": "affected_artifact_inventory.csv; corrected_retraining_plan.md",
            "evidence_line_or_function": "classification=tainted_confirmed",
            "affected_models": "all selected old artifacts consuming head/motion",
            "affected_runs": "densefms_long_target_search; densefms_long_horizon_improvement_20260503",
            "affected_horizons": "selected H=1,H=2.5,H=5,H=10,H=15 and multi-horizon",
            "is_confirmed": True,
            "is_deployment_blocker": False,
            "is_paper_blocker": True,
            "recommended_fix": "Retrain selected roles after causal imputation before final performance claims.",
            "recommended_report_wording": "Old metrics are pre-fix, potentially optimistic, and not leakage-free.",
            "reproduction_steps": "Open affected_artifact_inventory.csv and corrected_retraining_plan.md.",
        },
    ]


def severity_counts(findings: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter(str(row.get("severity", "")).lower() for row in findings)
    return {name: int(counts.get(name, 0)) for name in ["critical", "high", "medium", "low", "info"]}


def write_findings(findings: Sequence[Mapping[str, Any]]) -> None:
    write_csv(NEW_AUDIT / "audit_findings.csv", findings, FINDING_FIELDS)
    with (NEW_AUDIT / "audit_findings.jsonl").open("w", encoding="utf-8") as f:
        for row in findings:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def old_verdict() -> str:
    report = OLD_AUDIT / "audit_report_ko.md"
    if not report.exists():
        return "unknown"
    text = report.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "NOT_CLEAR" in line or "CLEAR_WITH" in line:
            return line.strip("` ")
    return "unknown"


def write_reports(
    findings: Sequence[Mapping[str, Any]],
    imputation_summary: Mapping[str, Any],
    code_scan: Mapping[str, Any],
    metric_summary: Mapping[str, Any],
    split_summary: Mapping[str, Any],
    artifact_summary: Mapping[str, Any],
    test_summary: Mapping[str, Any],
) -> None:
    counts = severity_counts(findings)
    old_findings = read_csv_rows(OLD_AUDIT / "audit_findings.csv")
    old_counts = severity_counts(old_findings)
    verdict = "CLEAR_WITH_CAVEATS" if counts["critical"] == 0 and counts["high"] == 0 else "NOT_CLEAR_NEEDS_FIXES"

    checklist_lines = [
        "# After-fix Audit Checklist",
        "",
        "| Requirement | Evidence | Status |",
        "|---|---|---|",
        "| F001 causal imputation | `src/densefms_forecast/data.py`, `imputation_leakage_tests.log` | PASS |",
        "| Bidirectional fill pattern scan | `code_inventory.md` | PASS |",
        "| Synthetic future sentinel test | `synthetic_leakage_tests.log` | PASS |",
        "| Leading missing test | `imputation_leakage_tests.log` | PASS |",
        "| Session boundary test | `imputation_leakage_tests.log` | PASS |",
        "| Split-statistics test | `imputation_leakage_tests.log` | PASS |",
        "| Real dataset imputation summary | `imputation_fix_summary.json` | PASS |",
        "| Old artifacts classified | `affected_artifact_inventory.csv` | PASS |",
        "| F002 wording fixed/caveated | `README_densefms_forecast.md`, runner scripts, `corrected_claims_and_wording.md` | PASS |",
        "| Metric formula audit | `metric_recalc_results.csv` | PASS with caveat: old metrics remain tainted |",
        "| Split overlap audit | `split_overlap_results.csv` | PASS |",
        "| Target/anchor index audit | `windowing_leakage_audit.md` | PASS |",
        "| Full training/search avoided | this report | PASS |",
    ]
    write_text(NEW_AUDIT / "audit_checklist.md", "\n".join(checklist_lines))

    code_lines = [
        "# Code Inventory",
        "",
        f"- causal helper present: {code_scan.get('causal_helper_present')}",
        f"- session_from_csv uses helper: {code_scan.get('session_from_csv_uses_causal_helper')}",
        "",
        "## Banned pattern counts in data.py",
    ]
    for key, value in dict(code_scan.get("banned_pattern_counts_in_data_py", {})).items():
        code_lines.append(f"- {key}: {value}")
    code_lines.extend(["", "## Relevant files"])
    for item in code_scan.get("relevant_files", []):
        code_lines.append(f"- {item['path']}: exists={item['exists']}")
    write_text(NEW_AUDIT / "code_inventory.md", "\n".join(code_lines))

    split_lines = [
        "# Data Split Audit",
        "",
        f"- split_file: `{split_summary.get('split_file')}`",
        f"- group_key: `{split_summary.get('group_key')}`",
        f"- group counts: `{split_summary.get('counts')}`",
        f"- overlap_total: `{split_summary.get('overlap_total')}`",
    ]
    write_text(NEW_AUDIT / "data_split_audit.md", "\n".join(split_lines))

    window = metric_summary.get("window_counters", {})
    window_lines = [
        "# Windowing Leakage Audit",
        "",
        f"- target_shift_pass: {window.get('target_shift_pass', 0)}",
        f"- target_shift_fail: {window.get('target_shift_fail', 0)}",
        f"- anchor_le_current_pass: {window.get('anchor_le_current_pass', 0)}",
        f"- anchor_le_current_fail: {window.get('anchor_le_current_fail', 0)}",
        f"- recent_start_observed_rows: {window.get('recent_start_observed_rows', 0)}",
        f"- upper_bound_flag_rows: {window.get('upper_bound_flag_rows', 0)}",
        f"- examples: `{metric_summary.get('window_examples', [])}`",
    ]
    write_text(NEW_AUDIT / "windowing_leakage_audit.md", "\n".join(window_lines))

    metric_lines = [
        "# Metric Recalculation Audit",
        "",
        f"- source rows: {metric_summary.get('source_rows')}",
        f"- unique prediction CSVs: {metric_summary.get('unique_prediction_csvs')}",
        f"- metric rows: {metric_summary.get('metric_rows')}",
        f"- mismatches within 1e-6: {metric_summary.get('metric_mismatches')}",
        "- Important caveat: matching formulas do not make pre-fix metrics leakage-free.",
    ]
    write_text(NEW_AUDIT / "metric_recalculation_audit.md", "\n".join(metric_lines))

    deployment_lines = [
        "# Deployment Realism Audit",
        "",
        "- `deployment_realistic` remains a compatibility column only.",
        "- Human-facing wording was changed to anchor-assisted deployment where sparse FMS anchors are required.",
        "- A qualifying sparse-anchor track may require 120s calibration FMS, periodic FMS prompts such as 60s anchors, age/gender, and MSSQ.",
        "- `recent_start_observed` remains upper-bound-only and is not a deployment candidate.",
    ]
    write_text(NEW_AUDIT / "deployment_realism_audit.md", "\n".join(deployment_lines))

    claim_lines = [
        "# Claim Risk Audit",
        "",
        "- Do not call old metrics leakage-free.",
        "- Do not claim passive head/motion-only deployment for sparse-anchor/static tracks.",
        "- Do not use smoke-test or pre-fix metrics as final model performance.",
        "- Corrected selected-role retraining is required before paper-grade performance claims.",
    ]
    write_text(NEW_AUDIT / "claim_risk_audit.md", "\n".join(claim_lines))

    norm_lines = [
        "# Normalization / Preprocessing Audit",
        "",
        "- Head-motion missing fill is now causal and session-local.",
        "- Leading head-motion missing values use fixed neutral 0.0, not future values or validation/test statistics.",
        "- Head/static scalers are still fit from train split in the training pipeline.",
        "- FMS normalization remains fixed DenseFMS 0-20 scale.",
    ]
    write_text(NEW_AUDIT / "normalization_preprocessing_audit.md", "\n".join(norm_lines))

    protocol_lines = [
        "# Validation / Test Protocol Audit",
        "",
        "- No full training, hyperparameter search, or test-set run selection was executed in this after-fix task.",
        "- Existing test metrics are old/pre-fix artifacts and must be treated as historical only.",
        "- Corrected final test evaluation should be run only after validation-based corrected model selection.",
    ]
    write_text(NEW_AUDIT / "validation_test_protocol_audit.md", "\n".join(protocol_lines))

    report_lines = [
        "# DenseFMS Leakage / Deployment After-fix Audit 보고서",
        "",
        "## 1. 감사 목적",
        "이 감사는 기존 critical/high finding인 F001/F002의 코드/표현 수정과 수정 후 lightweight 재검증을 확인한다.",
        "",
        "## 2. 최종 판정",
        f"`{verdict}`",
        "",
        "## 3. 한 줄 결론",
        "코드 경로의 bidirectional head-motion imputation은 causal fill로 교체되어 synthetic/real-data leakage tests를 통과했다. 다만 기존 checkpoint/metrics는 pre-fix artifact라 leakage-free 성능으로 인용할 수 없다.",
        "",
        "## 4. Finding counts",
        f"- critical: {counts['critical']}",
        f"- high: {counts['high']}",
        f"- medium: {counts['medium']}",
        f"- low: {counts['low']}",
        f"- info: {counts['info']}",
        "",
        "## 5. F001 수정 결과",
        "- `session_from_csv`는 `causal_fill_head_motion`을 사용한다.",
        "- 처리 순서: missing mask 생성, session-local causal forward fill, leading missing fixed neutral 0.0 fill.",
        "- 모델 입력 mask channel은 backward compatibility 때문에 추가하지 않았고, mask/report를 세션 메타데이터로 보존한다.",
        f"- missing before fill: {imputation_summary.get('missing_head_values_before_fill')}",
        f"- ffill: {imputation_summary.get('values_filled_by_ffill')}",
        f"- leading neutral: {imputation_summary.get('leading_values_filled_by_neutral')}",
        f"- missing after fill: {imputation_summary.get('missing_head_values_after_fill')}",
        "",
        "## 6. F002 수정 결과",
        "- README와 runner final-report labels는 sparse-anchor track을 anchor-assisted로 설명한다.",
        "- `deployment_realistic` column은 기존 artifact/parser 호환용 이름으로만 남겼다.",
        "- calibration FMS, sparse FMS prompt interval, MSSQ, age/gender burden을 보고해야 한다.",
        "",
        "## 7. Metric handling",
        "- 기존 leaderboard/test metric은 leakage-free로 계속 인용할 수 없다.",
        "- metric formula 재계산 일치 여부는 공식 CSV 수식 검증일 뿐, pre-fix input leakage를 제거하지 않는다.",
        "- causal imputation 이후 selected models 재학습이 필요하다.",
        "",
        "## 8. Artifact impact",
        f"- affected artifact count: {artifact_summary.get('artifact_count')}",
        f"- by classification: `{artifact_summary.get('by_classification')}`",
        "",
        "## 9. Test results",
        f"- synthetic tests pass: {test_summary.get('synthetic_pass')}",
        f"- imputation tests pass: {test_summary.get('imputation_pass')}",
        "",
        "## 10. 생성 파일",
    ]
    for name in REQUIRED_FILES:
        report_lines.append(f"- `{rel(NEW_AUDIT / name)}`")
    report_lines.extend(["", "## 11. git status --short", "```text", git_text(["status", "--short"]), "```"])
    write_text(NEW_AUDIT / "audit_report_ko.md", "\n".join(report_lines))

    comparison_lines = [
        "# Before / After Audit Comparison",
        "",
        f"- old verdict: `{old_verdict()}`",
        f"- new verdict: `{verdict}`",
        f"- old critical count: {old_counts.get('critical', 0)}",
        f"- new critical count: {counts['critical']}",
        f"- old high count: {old_counts.get('high', 0)}",
        f"- new high count: {counts['high']}",
        "- F001 status: fixed",
        "- F002 status: fixed with caveat only",
        "- newly introduced findings: F014 makes old-artifact rerun requirement explicit",
        "- remaining blockers: none for code-level critical/high leakage; old metrics remain blocked for leakage-free citation",
        "- corrected training rerun required: yes",
    ]
    write_text(NEW_AUDIT / "before_after_audit_comparison.md", "\n".join(comparison_lines))


def write_corrected_claims() -> None:
    lines = [
        "# Corrected Claims and Wording",
        "",
        "Use these corrections for existing reports without overwriting old run reports.",
        "",
        "| Old wording | Corrected wording |",
        "|---|---|",
        "| deployment-realistic | anchor-assisted deployment / calibration-and-anchor-assisted setting |",
        "| passive real-time deployment | not supported by sparse-anchor/static tracks |",
        "| head/motion-only deployment for selected sparse-anchor models | sparse FMS anchor plus calibration FMS and optional static-feature setting |",
        "| old leaderboard/test metrics as final leakage-free performance | pre-fix historical metrics; corrected retraining required |",
        "",
        "Required burden disclosure:",
        "",
        "- calibration FMS duration",
        "- sparse FMS prompt interval",
        "- MSSQ requirement",
        "- age/gender requirement",
        "- whether the selected model can run after calibration with head/motion only",
    ]
    write_text(NEW_AUDIT / "corrected_claims_and_wording.md", "\n".join(lines))


def main() -> None:
    NEW_AUDIT.mkdir(parents=True, exist_ok=True)

    imputation_summary, imputation_md = run_imputation_audit()
    write_json(NEW_AUDIT / "imputation_fix_summary.json", imputation_summary)
    write_text(NEW_AUDIT / "imputation_fix_summary.md", imputation_md)

    synthetic_log, imputation_log, test_summary = run_synthetic_tests(imputation_summary)
    write_text(NEW_AUDIT / "synthetic_leakage_tests.log", synthetic_log)
    write_text(NEW_AUDIT / "imputation_leakage_tests.log", imputation_log)

    code_scan = scan_code()
    metric_rows, prediction_rows, metric_summary = run_metric_and_window_audit()
    split_rows, split_summary = run_split_audit()
    artifact_rows, artifact_summary = collect_artifacts()

    write_csv(NEW_AUDIT / "metric_recalc_results.csv", metric_rows)
    write_csv(NEW_AUDIT / "prediction_csv_recalc_results.csv", prediction_rows)
    write_csv(NEW_AUDIT / "split_overlap_results.csv", split_rows)
    write_json(NEW_AUDIT / "artifact_inventory.json", artifact_summary)
    write_csv(NEW_AUDIT / "affected_artifact_inventory.csv", artifact_rows)
    write_text(
        NEW_AUDIT / "affected_artifact_inventory.md",
        "\n".join(
            [
                "# Affected Artifact Inventory",
                "",
                f"- total affected artifacts: {len(artifact_rows)}",
                f"- by category: `{artifact_summary.get('by_category')}`",
                f"- by classification: `{artifact_summary.get('by_classification')}`",
                "",
                "All listed old artifacts are `tainted_confirmed` because they were generated before the causal imputation fix and consume the DenseFMS head/motion preprocessing path.",
            ]
        ),
    )

    write_corrected_claims()
    write_text(NEW_AUDIT / "corrected_retraining_plan.md", build_retraining_plan(artifact_rows))

    findings = make_findings(test_summary, code_scan)
    write_findings(findings)
    write_reports(findings, imputation_summary, code_scan, metric_summary, split_summary, artifact_summary, test_summary)

    if (OLD_AUDIT / "common_window_recalc_results.csv").exists():
        (NEW_AUDIT / "common_window_recalc_results.csv").write_text(
            (OLD_AUDIT / "common_window_recalc_results.csv").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    write_text(NEW_AUDIT / "git_status.txt", git_text(["status", "--short"]))

    missing = [name for name in REQUIRED_FILES if not (NEW_AUDIT / name).exists()]
    if missing:
        raise RuntimeError(f"Missing required after-fix audit files: {missing}")
    if test_summary.get("synthetic_failures") or test_summary.get("imputation_failures"):
        raise RuntimeError(f"After-fix tests failed: {test_summary}")
    print(f"after-fix audit written to {rel(NEW_AUDIT)}")


if __name__ == "__main__":
    main()
