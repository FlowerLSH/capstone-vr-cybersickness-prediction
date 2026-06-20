"""Summarize online-current warning-extension validation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


RECIPES = [
    "selected_risk035",
    "risk045_smooth005",
    "zero_anchor_highgate_delta2",
    "range_scaled_delta2",
]


def _get(mapping: Mapping[str, Any], dotted: str, default: float = float("nan")) -> float:
    value: Any = mapping
    for key in dotted.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _fmt(value: float) -> str:
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.4f}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _load_metrics(run_dir: Path) -> Dict[str, Any]:
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload["metrics"]
    return {
        "payload": payload,
        "metrics": metrics,
        "val": metrics.get("best_val_metrics", metrics.get("val_metrics", {})),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize warning-extension validation runs.")
    parser.add_argument("--runs_dir", default="runs/online_current_warning_extension_0514")
    parser.add_argument("--report_dir", default="reports/online_current_warning_extension_0514")
    parser.add_argument("--suffix", default="warnext_stage1_full_seed42")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for recipe in RECIPES:
        run_dir = Path(args.runs_dir) / f"{recipe}_{args.suffix}"
        if not (run_dir / "metrics.json").exists():
            continue
        loaded = _load_metrics(run_dir)
        metrics = loaded["metrics"]
        val = loaded["val"]
        rows.append(
            {
                "recipe": recipe,
                "run_dir": str(run_dir),
                "best_epoch": metrics.get("best_epoch"),
                "selection_value": metrics.get("best_selection_value"),
                "val_mae": _get(val, "mae"),
                "val_rmse": _get(val, "rmse"),
                "val_r2": _get(val, "current_fms_r2"),
                "sigma_mean": _get(val, "uncertainty_sigma_mean"),
                "sigma_median": _get(val, "uncertainty_sigma_median"),
                "future_mae_5s": _get(val, "future_aux.5s.mae"),
                "future_mae_10s": _get(val, "future_aux.10s.mae"),
                "future_mae_20s": _get(val, "future_aux.20s.mae"),
                "future_mae_30s": _get(val, "future_aux.30s.mae"),
                "high8_20s_auprc": _get(val, "high_risk.20s_thr8.auprc"),
                "high8_20s_auroc": _get(val, "high_risk.20s_thr8.auroc"),
                "high8_20s_f1": _get(val, "high_risk.20s_thr8.f1"),
                "high12_20s_auprc": _get(val, "high_risk.20s_thr12.auprc"),
                "high12_20s_auroc": _get(val, "high_risk.20s_thr12.auroc"),
                "high12_20s_f1": _get(val, "high_risk.20s_thr12.f1"),
                "rise10_auprc": _get(val, "rapid_rise.10s.auprc"),
                "rise20_auprc": _get(val, "rapid_rise.20s.auprc"),
                "drop10_auprc": _get(val, "rapid_drop.10s.auprc"),
                "drop20_auprc": _get(val, "rapid_drop.20s.auprc"),
                "final_warning_f1": _get(val, "final_warning.f1"),
            }
        )

    fieldnames = [
        "recipe",
        "best_epoch",
        "selection_value",
        "val_mae",
        "val_rmse",
        "val_r2",
        "sigma_mean",
        "sigma_median",
        "future_mae_5s",
        "future_mae_10s",
        "future_mae_20s",
        "future_mae_30s",
        "high8_20s_auprc",
        "high8_20s_auroc",
        "high8_20s_f1",
        "high12_20s_auprc",
        "high12_20s_auroc",
        "high12_20s_f1",
        "rise10_auprc",
        "rise20_auprc",
        "drop10_auprc",
        "drop20_auprc",
        "final_warning_f1",
        "run_dir",
    ]
    report_dir = Path(args.report_dir)
    _write_csv(report_dir / "stage1_validation_summary.csv", rows, fieldnames)

    lines = ["# Online Current Warning Extension Stage 1 Summary", ""]
    lines.append("Scope: backbone frozen, warning/future/uncertainty heads only, validation selection only. Test evaluation was not run.")
    lines.append("Direct `high_risk`, `rapid_rise`, `rapid_drop`, and `future_aux` metrics are the primary outputs here.")
    lines.append("")
    lines.append("## Validation Summary")
    lines.append("")
    lines.append(
        "| recipe | val MAE | future MAE 10s | future MAE 20s | high12 AUPRC | rise10 AUPRC | drop20 AUPRC | sigma mean |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {recipe} | {mae} | {f10} | {f20} | {h12} | {r10} | {d20} | {sigma} |".format(
                recipe=row["recipe"],
                mae=_fmt(row["val_mae"]),
                f10=_fmt(row["future_mae_10s"]),
                f20=_fmt(row["future_mae_20s"]),
                h12=_fmt(row["high12_20s_auprc"]),
                r10=_fmt(row["rise10_auprc"]),
                d20=_fmt(row["drop20_auprc"]),
                sigma=_fmt(row["sigma_mean"]),
            )
        )
    lines.append("")
    if rows:
        best_high12 = max(rows, key=lambda row: row["high12_20s_auprc"])
        best_future10 = min(rows, key=lambda row: row["future_mae_10s"])
        best_mae = min(rows, key=lambda row: row["val_mae"])
        lines.append("## Current Best Reads")
        lines.append("")
        lines.append(f"- Best validation current MAE: `{best_mae['recipe']}` ({_fmt(best_mae['val_mae'])}).")
        lines.append(
            f"- Best high12 20s AUPRC: `{best_high12['recipe']}` ({_fmt(best_high12['high12_20s_auprc'])})."
        )
        lines.append(
            f"- Best future 10s MAE: `{best_future10['recipe']}` ({_fmt(best_future10['future_mae_10s'])})."
        )
    lines.append("")
    lines.append("## Run Directories")
    lines.append("")
    for row in rows:
        lines.append(f"- `{row['recipe']}`: `{row['run_dir']}`")
    (report_dir / "stage1_validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "report_dir": str(report_dir)}, indent=2))


if __name__ == "__main__":
    main()
