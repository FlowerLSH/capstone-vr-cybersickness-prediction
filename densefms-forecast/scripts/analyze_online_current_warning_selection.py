"""Summarize warning-focused full runs and tune validation-only thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EVENTS: Sequence[Tuple[str, str, str]] = (
    ("future_high8_20s", "p_high_risk_20s_thr8", "high_risk_label_20s_thr8"),
    ("future_high12_20s", "p_high_risk_20s_thr12", "high_risk_label_20s_thr12"),
    ("rapid_rise_10s", "p_rapid_rise_10s", "rapid_rise_label_10s"),
    ("rapid_rise_20s", "p_rapid_rise_20s", "rapid_rise_label_20s"),
    ("rapid_drop_10s", "p_rapid_drop_10s", "rapid_drop_label_10s"),
    ("rapid_drop_20s", "p_rapid_drop_20s", "rapid_drop_label_20s"),
)


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


def _fmt(value: object) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if not math.isfinite(f) else f"{f:.4f}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).astype(float).to_numpy() > 0.5
    return values.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def _binary_prf(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = scores >= threshold
    tp = float(np.sum(pred & labels))
    fp = float(np.sum(pred & ~labels))
    fn = float(np.sum(~pred & labels))
    tn = float(np.sum(~pred & ~labels))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_rate": float(np.mean(pred)),
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _tune_threshold(labels: np.ndarray, scores: np.ndarray, mode: str) -> Dict[str, float]:
    mask = np.isfinite(scores)
    labels = labels[mask].astype(bool)
    scores = scores[mask].astype(float)
    if scores.size == 0 or labels.sum() == 0:
        return _binary_prf(labels, scores, 0.5)
    thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 99), np.quantile(scores, np.linspace(0, 1, 101))])
    best: Optional[Dict[str, float]] = None
    for threshold in thresholds:
        row = _binary_prf(labels, scores, float(threshold))
        if mode == "precision80":
            ok = row["precision"] >= 0.80
            score = (row["recall"] if ok else -1.0, row["f1"], -row["false_alarm_rate"])
        elif mode == "recall80":
            ok = row["recall"] >= 0.80
            score = (row["precision"] if ok else -1.0, row["f1"], -row["false_alarm_rate"])
        else:
            score = (row["f1"], row["precision"], row["recall"], -row["false_alarm_rate"])
        row["_score_tuple"] = score  # type: ignore[assignment]
        if best is None or score > best["_score_tuple"]:  # type: ignore[index]
            best = row
    assert best is not None
    best.pop("_score_tuple", None)
    return best


def _sigma_calibration(df: pd.DataFrame) -> Dict[str, float]:
    sigma = df["predicted_fms_sigma"].astype(float).to_numpy()
    err = df["fms_absolute_error"].astype(float).to_numpy()
    mask = np.isfinite(sigma) & np.isfinite(err) & (sigma > 1e-9)
    sigma = sigma[mask]
    err = err[mask]
    ratio = err / sigma
    alpha_68 = float(np.quantile(ratio, 0.683)) if ratio.size else float("nan")
    alpha_90 = float(np.quantile(ratio, 0.900)) if ratio.size else float("nan")
    alpha_95 = float(np.quantile(ratio, 0.954)) if ratio.size else float("nan")
    def cov(alpha: float, scale: float = 1.0) -> float:
        return float(np.mean(err <= alpha * scale * sigma)) if ratio.size and math.isfinite(alpha) else float("nan")
    return {
        "sigma_mean": float(np.mean(sigma)) if sigma.size else float("nan"),
        "sigma_median": float(np.median(sigma)) if sigma.size else float("nan"),
        "sigma_coverage_1x": cov(1.0),
        "sigma_coverage_2x": cov(2.0),
        "sigma_alpha_for_68": alpha_68,
        "sigma_alpha_for_90": alpha_90,
        "sigma_alpha_for_95": alpha_95,
        "sigma_cov_alpha68_1x": cov(alpha_68),
        "sigma_cov_alpha68_2x": cov(alpha_68, 2.0),
    }


def _summarize_run(run_dir: Path) -> Dict[str, object]:
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    val = metrics["best_val_metrics"]
    return {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "best_epoch": metrics.get("best_epoch"),
        "selection_metric": metrics.get("selection_metric"),
        "selection_value": metrics.get("best_selection_value"),
        "val_mae": _get(val, "mae"),
        "val_rmse": _get(val, "rmse"),
        "val_r2": _get(val, "current_fms_r2"),
        "future_mae_10s": _get(val, "future_aux.10s.mae"),
        "future_mae_20s": _get(val, "future_aux.20s.mae"),
        "high8_20s_auprc": _get(val, "high_risk.20s_thr8.auprc"),
        "high12_20s_auprc": _get(val, "high_risk.20s_thr12.auprc"),
        "rise10_auprc": _get(val, "rapid_rise.10s.auprc"),
        "rise20_auprc": _get(val, "rapid_rise.20s.auprc"),
        "drop10_auprc": _get(val, "rapid_drop.10s.auprc"),
        "drop20_auprc": _get(val, "rapid_drop.20s.auprc"),
        "sigma_mean": _get(val, "uncertainty_sigma_mean"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze warning-focused validation runs.")
    parser.add_argument("--runs_dir", default="runs/online_current_warning_selection_0514")
    parser.add_argument("--report_dir", default="reports/online_current_warning_selection_0514")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(path for path in runs_dir.glob("*_full_seed42") if (path / "metrics.json").exists())
    rows = [_summarize_run(path) for path in run_dirs]
    rows = sorted(rows, key=lambda row: (-float(row["high12_20s_auprc"]), float(row["val_mae"])))
    _write_csv(report_dir / "warning_selection_validation_leaderboard.csv", rows)

    threshold_rows: List[Dict[str, object]] = []
    sigma_rows: List[Dict[str, object]] = []
    for row in rows:
        run_dir = Path(str(row["run_dir"]))
        pred_path = run_dir / "val_predictions.csv"
        if not pred_path.exists():
            continue
        df = pd.read_csv(pred_path)
        sigma = _sigma_calibration(df)
        sigma_rows.append({"run": row["run"], **sigma})
        for event, score_col, label_col in EVENTS:
            if score_col not in df.columns or label_col not in df.columns:
                continue
            valid_col = label_col.replace("_label_", "_valid_")
            valid = _as_bool(df[valid_col]) if valid_col in df.columns else np.ones(len(df), dtype=bool)
            scores = df.loc[valid, score_col].astype(float).to_numpy()
            labels = _as_bool(df.loc[valid, label_col])
            modes = ["best_f1"]
            if "high8" in event:
                modes.append("recall80")
            if "high12" in event or "rapid" in event:
                modes.append("precision80")
            for mode in modes:
                tuned = _tune_threshold(labels, scores, mode)
                threshold_rows.append(
                    {
                        "run": row["run"],
                        "event": event,
                        "mode": mode,
                        "positive_rate": float(np.mean(labels)) if labels.size else float("nan"),
                        **tuned,
                    }
                )
    _write_csv(report_dir / "warning_threshold_tuning_val.csv", threshold_rows)
    _write_csv(report_dir / "sigma_calibration_val.csv", sigma_rows)

    lines = ["# Online Current Warning Selection Summary", ""]
    lines.append("Validation-only summary. Test set is not used for model or threshold selection.")
    lines.append("")
    lines.append("## Validation Leaderboard")
    lines.append("")
    lines.append(
        "| run | best epoch | val MAE | future 10s MAE | future 20s MAE | high8 AUPRC | high12 AUPRC | rise20 AUPRC | drop20 AUPRC | sigma mean |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {run} | {epoch} | {mae} | {f10} | {f20} | {h8} | {h12} | {r20} | {d20} | {sig} |".format(
                run=row["run"],
                epoch=row["best_epoch"],
                mae=_fmt(row["val_mae"]),
                f10=_fmt(row["future_mae_10s"]),
                f20=_fmt(row["future_mae_20s"]),
                h8=_fmt(row["high8_20s_auprc"]),
                h12=_fmt(row["high12_20s_auprc"]),
                r20=_fmt(row["rise20_auprc"]),
                d20=_fmt(row["drop20_auprc"]),
                sig=_fmt(row["sigma_mean"]),
            )
        )
    if rows:
        selected = min(rows, key=lambda row: (-(float(row["high12_20s_auprc"]) - 0.001 * float(row["val_mae"]))))
        lines.append("")
        lines.append("## Suggested Read")
        lines.append("")
        lines.append(
            "High12 AUPRC is nearly tied between the top runs, so current MAE/future MAE/drop warning should be considered before final selection."
        )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `warning_selection_validation_leaderboard.csv`")
    lines.append("- `warning_threshold_tuning_val.csv`")
    lines.append("- `sigma_calibration_val.csv`")
    (report_dir / "warning_selection_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote warning selection analysis to {report_dir}")


if __name__ == "__main__":
    main()
