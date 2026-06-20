"""Diagnose whether calibration-window FMS tendencies persist after calibration.

The goal is split diagnostics, not model evaluation.  It compares how strongly
calibration level/trend/range explain the later prediction window in each split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from densefms_forecast.data import apply_saved_split, load_raw_sessions
from densefms_forecast.utils import load_json, seconds_to_steps


def _parse_named_path(spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip() or Path(path).stem, Path(path)
    path = Path(spec)
    return path.stem, path


def _finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _mean(values: Sequence[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _std(values: Sequence[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(np.std(arr, ddof=0)) if arr.size else float("nan")


def _range(values: Sequence[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(np.max(arr) - np.min(arr)) if arr.size else float("nan")


def _last(values: Sequence[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(arr[-1]) if arr.size else float("nan")


def _slope_per_minute(values: Sequence[float] | np.ndarray, sampling_interval: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    if int(mask.sum()) < 2:
        return float("nan")
    y = arr[mask]
    x = np.arange(arr.shape[0], dtype=np.float64)[mask] * float(sampling_interval)
    x = x - float(np.mean(x))
    denom = float(np.sum(x * x))
    if denom <= 1e-12:
        return float("nan")
    slope_per_second = float(np.sum(x * (y - float(np.mean(y)))) / denom)
    return slope_per_second * 60.0


def _corr(x_values: Sequence[float], y_values: Sequence[float]) -> Tuple[float, int]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan"), int(mask.sum())
    x = x[mask]
    y = y[mask]
    if float(np.std(x, ddof=0)) <= 1e-12 or float(np.std(y, ddof=0)) <= 1e-12:
        return float("nan"), int(mask.sum())
    return float(np.corrcoef(x, y)[0, 1]), int(mask.sum())


def _rate(flags: Sequence[bool]) -> float:
    return float(sum(1 for flag in flags if flag) / len(flags)) if flags else float("nan")


def _format(value: float) -> str:
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.4f}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _session_features(
    split_name: str,
    split: str,
    session: Any,
    calibration_steps: int,
    recent_steps: int,
    segment_steps: int,
    sampling_interval: float,
    trend_threshold: float,
) -> Dict[str, Any]:
    fms = np.asarray(session.fms_raw if session.fms_raw is not None else session.fms, dtype=np.float64)
    prediction_start = max(int(calibration_steps), int(recent_steps) - 1)
    calib = fms[:calibration_steps] if calibration_steps > 0 else np.asarray([], dtype=np.float64)
    pred = fms[prediction_start : session.length]
    cal_first = calib[:segment_steps]
    cal_last = calib[-segment_steps:] if calib.size else calib
    pred_first = pred[:segment_steps]
    pred_last = pred[-segment_steps:] if pred.size else pred

    cal_first_mean = _mean(cal_first)
    cal_last_mean = _mean(cal_last)
    pred_first_mean = _mean(pred_first)
    pred_last_mean = _mean(pred_last)
    cal_net = cal_last_mean - cal_first_mean if math.isfinite(cal_last_mean) and math.isfinite(cal_first_mean) else float("nan")
    pred_net = pred_last_mean - pred_first_mean if math.isfinite(pred_last_mean) and math.isfinite(pred_first_mean) else float("nan")
    cal_end = _last(calib)
    pred_mean = _mean(pred)
    cal_slope = _slope_per_minute(calib, sampling_interval)
    pred_slope = _slope_per_minute(pred, sampling_interval)
    trend_eligible = math.isfinite(cal_net) and math.isfinite(pred_net) and abs(cal_net) >= trend_threshold and abs(pred_net) >= trend_threshold
    sign_match = bool(trend_eligible and (cal_net > 0) == (pred_net > 0))
    return {
        "split_name": split_name,
        "split": split,
        "participant_id": session.participant_id or "",
        "session_id": session.session_id,
        "source_file": session.source_file,
        "length": int(session.length),
        "prediction_start_index": int(prediction_start),
        "calibration_points": int(_finite(calib).size),
        "prediction_points": int(_finite(pred).size),
        "calibration_mean": _mean(calib),
        "calibration_end": cal_end,
        "calibration_first_segment_mean": cal_first_mean,
        "calibration_last_segment_mean": cal_last_mean,
        "calibration_net_delta": cal_net,
        "calibration_slope_per_min": cal_slope,
        "calibration_range": _range(calib),
        "prediction_mean": pred_mean,
        "prediction_first_segment_mean": pred_first_mean,
        "prediction_last_segment_mean": pred_last_mean,
        "prediction_net_delta": pred_net,
        "prediction_slope_per_min": pred_slope,
        "prediction_range": _range(pred),
        "prediction_mean_minus_calibration_end": pred_mean - cal_end if math.isfinite(pred_mean) and math.isfinite(cal_end) else float("nan"),
        "prediction_first_minus_calibration_end": pred_first_mean - cal_end if math.isfinite(pred_first_mean) and math.isfinite(cal_end) else float("nan"),
        "prediction_last_minus_calibration_last_segment": (
            pred_last_mean - cal_last_mean if math.isfinite(pred_last_mean) and math.isfinite(cal_last_mean) else float("nan")
        ),
        "abs_prediction_mean_minus_calibration_end": abs(pred_mean - cal_end) if math.isfinite(pred_mean) and math.isfinite(cal_end) else float("nan"),
        "abs_prediction_first_minus_calibration_end": (
            abs(pred_first_mean - cal_end) if math.isfinite(pred_first_mean) and math.isfinite(cal_end) else float("nan")
        ),
        "trend_eligible": bool(trend_eligible),
        "trend_sign_match": sign_match,
        "calibration_up": bool(math.isfinite(cal_net) and cal_net >= trend_threshold),
        "calibration_down": bool(math.isfinite(cal_net) and cal_net <= -trend_threshold),
        "prediction_up": bool(math.isfinite(pred_net) and pred_net >= trend_threshold),
        "prediction_down": bool(math.isfinite(pred_net) and pred_net <= -trend_threshold),
    }


def _summarize_rows(split_name: str, split: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "split_name": split_name,
        "split": split,
        "sessions": len(rows),
        "participants": len({str(row["participant_id"]) for row in rows if row.get("participant_id")}),
    }
    metrics = [
        "calibration_mean",
        "calibration_end",
        "calibration_net_delta",
        "calibration_slope_per_min",
        "calibration_range",
        "prediction_mean",
        "prediction_net_delta",
        "prediction_slope_per_min",
        "prediction_range",
        "prediction_mean_minus_calibration_end",
        "prediction_first_minus_calibration_end",
        "prediction_last_minus_calibration_last_segment",
        "abs_prediction_mean_minus_calibration_end",
        "abs_prediction_first_minus_calibration_end",
    ]
    for metric in metrics:
        values = [float(row.get(metric, float("nan"))) for row in rows]
        out[f"{metric}_mean"] = _mean(values)
        out[f"{metric}_std"] = _std(values)
    correlations = {
        "corr_calibration_end_prediction_mean": ("calibration_end", "prediction_mean"),
        "corr_calibration_mean_prediction_mean": ("calibration_mean", "prediction_mean"),
        "corr_calibration_net_prediction_net": ("calibration_net_delta", "prediction_net_delta"),
        "corr_calibration_slope_prediction_slope": ("calibration_slope_per_min", "prediction_slope_per_min"),
        "corr_calibration_range_prediction_range": ("calibration_range", "prediction_range"),
        "corr_calibration_end_prediction_first": ("calibration_end", "prediction_first_segment_mean"),
        "corr_calibration_last_prediction_last": ("calibration_last_segment_mean", "prediction_last_segment_mean"),
    }
    for name, (x_key, y_key) in correlations.items():
        corr, n = _corr([float(row.get(x_key, float("nan"))) for row in rows], [float(row.get(y_key, float("nan"))) for row in rows])
        out[name] = corr
        out[f"{name}_n"] = n
    eligible = [row for row in rows if bool(row.get("trend_eligible"))]
    out["trend_eligible_sessions"] = len(eligible)
    out["trend_sign_match_rate"] = _rate([bool(row.get("trend_sign_match")) for row in eligible])
    up = [row for row in rows if bool(row.get("calibration_up"))]
    down = [row for row in rows if bool(row.get("calibration_down"))]
    out["calibration_up_sessions"] = len(up)
    out["calibration_down_sessions"] = len(down)
    out["calibration_up_prediction_up_rate"] = _rate([bool(row.get("prediction_up")) for row in up])
    out["calibration_up_prediction_down_rate"] = _rate([bool(row.get("prediction_down")) for row in up])
    out["calibration_down_prediction_down_rate"] = _rate([bool(row.get("prediction_down")) for row in down])
    out["calibration_down_prediction_up_rate"] = _rate([bool(row.get("prediction_up")) for row in down])
    return out


def _fit_linear_transfer(train_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    feature_names = [
        "calibration_end",
        "calibration_mean",
        "calibration_net_delta",
        "calibration_slope_per_min",
        "calibration_range",
    ]
    x_raw: List[List[float]] = []
    y_raw: List[float] = []
    for row in train_rows:
        features = [float(row.get(name, float("nan"))) for name in feature_names]
        target = float(row.get("prediction_mean", float("nan")))
        if all(math.isfinite(value) for value in features) and math.isfinite(target):
            x_raw.append(features)
            y_raw.append(target)
    if len(x_raw) < len(feature_names) + 2:
        return {"feature_names": feature_names, "available": False}
    x = np.asarray(x_raw, dtype=np.float64)
    y = np.asarray(y_raw, dtype=np.float64)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x_std = (x - mean) / std
    design = np.concatenate([np.ones((x_std.shape[0], 1), dtype=np.float64), x_std], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {
        "feature_names": feature_names,
        "available": True,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "coef": coef.tolist(),
        "train_n": int(len(y_raw)),
    }


def _eval_linear_transfer(model: Mapping[str, Any], split_name: str, split: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not model.get("available"):
        return {"split_name": split_name, "split": split, "n": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    feature_names = list(model["feature_names"])
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    preds: List[float] = []
    targets: List[float] = []
    for row in rows:
        features = [float(row.get(name, float("nan"))) for name in feature_names]
        target = float(row.get("prediction_mean", float("nan")))
        if all(math.isfinite(value) for value in features) and math.isfinite(target):
            x = (np.asarray(features, dtype=np.float64) - mean) / std
            design = np.concatenate([np.ones(1, dtype=np.float64), x])
            preds.append(float(design @ coef))
            targets.append(target)
    if not targets:
        return {"split_name": split_name, "split": split, "n": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    pred = np.asarray(preds, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    err = pred - target
    return {
        "split_name": split_name,
        "split": split,
        "n": int(target.shape[0]),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "bias": float(np.mean(err)),
    }


def _markdown_report(
    out_dir: Path,
    args: argparse.Namespace,
    summary_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# DenseFMS Calibration Persistence Diagnostic",
        "",
        "## Configuration",
        "",
        f"- calibration seconds: `{float(args.calibration_seconds):g}`",
        f"- later/prediction window starts at `max(calibration_steps, recent_steps - 1)`",
        f"- segment seconds for start/end trend: `{float(args.segment_seconds):g}`",
        f"- trend threshold: `{float(args.trend_threshold):g}` FMS points",
        f"- max session points: `{args.max_session_points}`",
        "",
        "## Calibration-To-Later Summary",
        "",
        "| split file | split | sessions | cal_end->later_mean r | cal_trend->later_trend r | cal_range->later_range r | later_mean-cal_end | abs later_mean-cal_end | trend sign match | up->up | up->down |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['split_name']} | {row['split']} | {int(row['sessions'])} | "
            f"{_format(float(row['corr_calibration_end_prediction_mean']))} | "
            f"{_format(float(row['corr_calibration_net_prediction_net']))} | "
            f"{_format(float(row['corr_calibration_range_prediction_range']))} | "
            f"{_format(float(row['prediction_mean_minus_calibration_end_mean']))} | "
            f"{_format(float(row['abs_prediction_mean_minus_calibration_end_mean']))} | "
            f"{_format(float(row['trend_sign_match_rate']))} | "
            f"{_format(float(row['calibration_up_prediction_up_rate']))} | "
            f"{_format(float(row['calibration_up_prediction_down_rate']))} |"
        )
    lines.extend(
        [
            "",
            "## Train-Fit Calibration Transfer",
            "",
            "A simple linear model is fit only on each split file's train sessions using calibration end/mean/trend/range, then evaluated on train/val/test session-level later-window mean FMS. Similar val/test errors indicate that the calibration-to-later relation is not strongly shifted.",
            "",
            "| split file | eval split | n | MAE | RMSE | bias |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in transfer_rows:
        lines.append(
            "| "
            f"{row['split_name']} | {row['split']} | {int(row['n'])} | "
            f"{_format(float(row['mae']))} | {_format(float(row['rmse']))} | {_format(float(row['bias']))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- High `cal_end->later_mean r` means the calibration endpoint is a useful anchor for later FMS level.",
            "- High `cal_trend->later_trend r` or high `trend sign match` means the calibration trend tends to continue later.",
            "- If these values differ strongly between train/val/test, validation may not represent test for calibration-aware models.",
            "- If they are low everywhere, the problem is intrinsically hard for calibration-based prediction because calibration does not reliably determine later trajectory shape.",
            "",
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_persistence_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    raw_sessions, mapping, data_info = load_raw_sessions(
        args.data_dir,
        calibration_seconds=float(args.calibration_seconds),
        horizon_seconds=float(args.horizon_seconds),
        default_sampling_interval=float(args.sampling_interval),
        max_session_points=int(args.max_session_points) if args.max_session_points is not None else None,
    )
    sampling_interval = float(data_info["sampling_interval"])
    calibration_steps = int(data_info["calibration_steps"])
    recent_steps = seconds_to_steps(float(args.recent_window_seconds), sampling_interval, name="recent_window_seconds")
    segment_steps = seconds_to_steps(float(args.segment_seconds), sampling_interval, name="segment_seconds")
    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    transfer_rows: List[Dict[str, Any]] = []
    by_splitfile_split: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for split_name, split_path in [_parse_named_path(spec) for spec in args.splits]:
        split_info = load_json(split_path)
        split_sessions = apply_saved_split(raw_sessions, split_info)
        for split in ["train", "val", "test"]:
            for session in split_sessions.get(split, []):
                row = _session_features(
                    split_name,
                    split,
                    session,
                    calibration_steps,
                    recent_steps,
                    segment_steps,
                    sampling_interval,
                    float(args.trend_threshold),
                )
                all_rows.append(row)
                by_splitfile_split[(split_name, split)].append(row)
        for split in ["train", "val", "test"]:
            summary_rows.append(_summarize_rows(split_name, split, by_splitfile_split[(split_name, split)]))
        transfer_model = _fit_linear_transfer(by_splitfile_split[(split_name, "train")])
        for split in ["train", "val", "test"]:
            transfer_rows.append(_eval_linear_transfer(transfer_model, split_name, split, by_splitfile_split[(split_name, split)]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if all_rows:
        _write_csv(out_dir / "session_calibration_persistence.csv", all_rows, list(all_rows[0].keys()))
    if summary_rows:
        _write_csv(out_dir / "calibration_persistence_summary.csv", summary_rows, list(summary_rows[0].keys()))
    if transfer_rows:
        _write_csv(out_dir / "calibration_transfer_linear_metrics.csv", transfer_rows, list(transfer_rows[0].keys()))
    payload = {
        "args": vars(args),
        "data_info": data_info,
        "mapping": mapping,
        "summary": summary_rows,
        "linear_transfer": transfer_rows,
    }
    (out_dir / "calibration_persistence_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _markdown_report(out_dir, args, summary_rows, transfer_rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--splits", nargs="+", required=True, help="Named split files as name=path.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sampling_interval", type=float, default=0.5)
    parser.add_argument("--calibration_seconds", type=float, default=120.0)
    parser.add_argument("--recent_window_seconds", type=float, default=10.0)
    parser.add_argument("--horizon_seconds", type=float, default=10.0)
    parser.add_argument("--max_session_points", type=int, default=420)
    parser.add_argument("--segment_seconds", type=float, default=30.0)
    parser.add_argument("--trend_threshold", type=float, default=1.0)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"out_dir": args.out_dir, "summary": payload["summary"], "linear_transfer": payload["linear_transfer"]}, indent=2))


if __name__ == "__main__":
    main()
