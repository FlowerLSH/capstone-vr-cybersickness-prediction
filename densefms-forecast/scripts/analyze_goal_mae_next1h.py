"""Validation-only audit for the post-lock DenseFMS next-step search.

This script intentionally reads validation artifacts only. It does not read
test predictions or metrics, and it writes a compact audit/report under a new
output directory so the locked v2 result remains untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BEST_RUN = "v2_lcsa_per_horizon_heads_adaptive_seed7_ff192_recent20_mh_nostatic_lr0p0003_wd0p0001_drop0p05_e80_s7"
PRIMARY_HORIZONS = (5.0, 10.0, 15.0)


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_name(value: Any) -> str:
    return str(value).replace("\\", "/").split("/")[-1]


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
            writer.writerow({key: row.get(key, "") for key in fields})


def markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def summarize_numeric(values: Iterable[float]) -> Dict[str, float]:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return {"n": 0, "mae": math.nan, "rmse": math.nan, "bias": math.nan, "p90_abs": math.nan}
    return {"n": len(vals)}


def aggregate(rows: Sequence[Mapping[str, Any]], key_fn) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        if key is not None:
            buckets[str(key)].append(row)
    out: List[Dict[str, Any]] = []
    for key, items in buckets.items():
        errors = [as_float(item.get("absolute_error")) for item in items]
        signed = [as_float(item.get("predicted_fms")) - as_float(item.get("target_fms")) for item in items]
        squared = [as_float(item.get("squared_error")) for item in items]
        finite_errors = sorted(v for v in errors if math.isfinite(v))
        finite_signed = [v for v in signed if math.isfinite(v)]
        finite_squared = [v for v in squared if math.isfinite(v)]
        n = len(finite_errors)
        p90 = finite_errors[min(n - 1, int(math.ceil(n * 0.9)) - 1)] if n else math.nan
        out.append(
            {
                "bucket": key,
                "n": n,
                "mae": sum(finite_errors) / n if n else math.nan,
                "rmse": math.sqrt(sum(finite_squared) / len(finite_squared)) if finite_squared else math.nan,
                "bias_pred_minus_target": sum(finite_signed) / len(finite_signed) if finite_signed else math.nan,
                "p90_abs_error": p90,
            }
        )
    out.sort(key=lambda row: as_float(row["mae"], -math.inf), reverse=True)
    return out


def bin_numeric(value: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    if not math.isfinite(value):
        return "missing"
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def load_best_validation_rows(v2_dir: Path) -> List[Dict[str, Any]]:
    pred_path = v2_dir / BEST_RUN / "val_predictions.csv"
    rows = read_csv(pred_path)
    return [row for row in rows if str(row.get("in_common_eval_window", "")).lower() in {"true", "1", "yes"}]


def write_error_audit(v2_dir: Path, out: Path) -> None:
    rows = load_best_validation_rows(v2_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["bucket", "n", "mae", "rmse", "bias_pred_minus_target", "p90_abs_error"]

    by_horizon = aggregate(rows, lambda row: f"h={as_float(row.get('horizon_seconds')):g}")
    by_target = aggregate(
        rows,
        lambda row: bin_numeric(
            as_float(row.get("target_fms")),
            [2.0, 4.0, 6.0, 8.0],
            ["target<2", "2<=target<4", "4<=target<6", "6<=target<8", "target>=8"],
        ),
    )
    by_start = aggregate(
        rows,
        lambda row: bin_numeric(
            as_float(row.get("start_fms_value")),
            [2.0, 4.0, 6.0, 8.0],
            ["start<2", "2<=start<4", "4<=start<6", "6<=start<8", "start>=8"],
        ),
    )
    by_delta = aggregate(
        rows,
        lambda row: bin_numeric(
            as_float(row.get("target_fms")) - as_float(row.get("start_fms_value")),
            [-2.0, -0.5, 0.5, 2.0],
            ["target-start<-2", "-2<=delta<-0.5", "-0.5<=delta<0.5", "0.5<=delta<2", "delta>=2"],
        ),
    )
    by_time = aggregate(
        rows,
        lambda row: bin_numeric(
            as_float(row.get("current_time")),
            [120.0, 150.0, 180.0, 210.0],
            ["current<120s", "120<=current<150s", "150<=current<180s", "180<=current<210s", "current>=210s"],
        ),
    )

    write_csv(out / "error_by_horizon.csv", by_horizon, fields)
    write_csv(out / "error_by_target_fms.csv", by_target, fields)
    write_csv(out / "error_by_start_fms.csv", by_start, fields)
    write_csv(out / "error_by_target_minus_start.csv", by_delta, fields)
    write_csv(out / "error_by_current_time.csv", by_time, fields)

    worst_sessions = aggregate(rows, lambda row: safe_name(row.get("session_id") or row.get("source_file")))
    write_csv(out / "worst_val_sessions.csv", worst_sessions, fields)

    lines = [
        "# Next 1h Validation Error Audit",
        "",
        "- Source: validation predictions only.",
        f"- Best source run: `{BEST_RUN}`",
        "- Test predictions/metrics are intentionally not read by this audit.",
        f"- Common-window rows analyzed: {len(rows)}",
        "",
        "## Horizon",
        "",
        markdown_table(by_horizon, fields),
        "",
        "## Target FMS Buckets",
        "",
        markdown_table(by_target, fields),
        "",
        "## Start FMS Buckets",
        "",
        markdown_table(by_start, fields),
        "",
        "## Target Minus Start",
        "",
        markdown_table(by_delta, fields),
        "",
        "## Current-Time Buckets",
        "",
        markdown_table(by_time, fields),
        "",
        "## Worst Validation Sessions",
        "",
        markdown_table(worst_sessions, fields, limit=12),
        "",
        "## Interpretation",
        "",
        "- Positive bias means predictions are high; negative bias means underprediction.",
        "- The target-minus-start table is the main check for whether fast FMS increases dominate the remaining error.",
        "- This audit should drive validation-only candidate choice; it must not be mixed with final test feedback.",
    ]
    (out / "validation_error_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    val = metrics.get("best_val_metrics", {})
    by_h = val.get("by_horizon", {}) if isinstance(val, Mapping) else {}
    row = {
        "best_epoch": metrics.get("best_epoch", ""),
        "val_mae": val.get("mae", ""),
        "val_rmse": val.get("rmse", ""),
        "val_n": val.get("n", ""),
    }
    scores = []
    for h in PRIMARY_HORIZONS:
        item = by_h.get(str(h)) or by_h.get(f"{h:g}") if isinstance(by_h, Mapping) else None
        mae = item.get("mae", "") if isinstance(item, Mapping) else ""
        row[f"h{h:g}_val_mae"] = mae
        score = as_float(mae)
        if math.isfinite(score):
            scores.append(score)
    row["primary_mean_h5_h10_h15"] = sum(scores) / 3.0 if len(scores) == 3 else ""
    return row


def summarize_search(out: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(out.glob("next1h_*")):
        config_path = run_dir / "config_snapshot.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        parsed = parse_metrics(run_dir / "metrics.json")
        if not parsed:
            status = "pending" if not run_dir.exists() else "missing_metrics"
        else:
            status = "completed"
        row = {
            "run_name": run_dir.name,
            "status": status,
            "recent_window_seconds": config.get("data", {}).get("recent_window_seconds", ""),
            "use_static": config.get("data", {}).get("use_static", ""),
            "horizon_encoder_dim": config.get("model", {}).get("horizon_encoder_dim", "default"),
            "horizon_context_mode": config.get("model", {}).get("horizon_context_mode", "encoded"),
            "start_fms_context_mode": config.get("model", {}).get("start_fms_context_mode", "encoded"),
            "static_context_mode": config.get("model", {}).get("static_context_mode", "encoded"),
            "epochs": config.get("training", {}).get("epochs", ""),
            "patience": config.get("training", {}).get("patience", ""),
            "seed": config.get("training", {}).get("seed", ""),
            "model": config.get("model", {}).get("model_name", "lc_sa_tcnformer"),
        }
        row.update(parsed)
        rows.append(row)
    fields = [
        "run_name",
        "status",
        "recent_window_seconds",
        "use_static",
        "start_fms_context_mode",
        "horizon_context_mode",
        "horizon_encoder_dim",
        "static_context_mode",
        "primary_mean_h5_h10_h15",
        "h5_val_mae",
        "h10_val_mae",
        "h15_val_mae",
        "best_epoch",
        "epochs",
        "patience",
        "seed",
    ]
    rows.sort(key=lambda row: as_float(row.get("primary_mean_h5_h10_h15"), math.inf))
    write_csv(out / "recent_window_cheap_leaderboard.csv", rows, fields)
    lines = [
        "# Next 1h Recent-Window Cheap Search",
        "",
        "- Scope: validation-only, no test evaluation.",
        "- Fixed settings: `lcsa_per_horizon_heads`, `d_model=96`, `transformer_ff_dim=192`, `seed=7`, no static, start_only/no anchor.",
        "- Variable: `recent_window_seconds`.",
        "- Horizon branch ablation rows set `horizon_encoder_dim`; `default` means scalar horizon -> `d_model`.",
        "- Compact-context rows set `start_fms_context_mode`, `horizon_context_mode`, and `static_context_mode`.",
        "",
        markdown_table(rows, fields),
        "",
        "## Next Recommendation",
        "",
    ]
    completed = [row for row in rows if row.get("status") == "completed" and row.get("primary_mean_h5_h10_h15") not in ("", None)]
    if completed:
        best = min(completed, key=lambda row: as_float(row.get("primary_mean_h5_h10_h15"), math.inf))
        lines.append(
            f"- Best cheap candidate so far: `{best.get('run_name')}` with primary validation MAE `{best.get('primary_mean_h5_h10_h15')}`."
        )
        lines.append("- Promote this only if it clearly beats the locked v2 validation best under comparable training budget; otherwise keep v2 best as the main model.")
    else:
        lines.append("- No completed cheap candidates yet.")
    (out / "recent_window_cheap_search.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2_dir", default="runs/goal_mae_search_v2")
    parser.add_argument("--out_dir", default="runs/goal_mae_next1h_0505")
    args = parser.parse_args()

    v2_dir = (ROOT / args.v2_dir).resolve()
    out = (ROOT / args.out_dir).resolve()
    write_error_audit(v2_dir, out)
    summarize_search(out)


if __name__ == "__main__":
    main()
