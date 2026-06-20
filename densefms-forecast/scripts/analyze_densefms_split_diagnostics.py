"""Create split-distribution diagnostics for DenseFMS online-current experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from densefms_forecast.data import apply_saved_split, load_raw_sessions
from densefms_forecast.utils import load_json, seconds_to_steps


FMS_BINS = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0]


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _format(value: float, digits: int = 4) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.{digits}f}"


def _quantile(values: Sequence[float], q: float) -> float:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, q))


def _mean(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _std(values: Sequence[float]) -> float:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    return float(np.std(finite, ddof=0)) if finite.size else float("nan")


def _median(values: Sequence[float]) -> float:
    return _quantile(values, 0.5)


def _bin_label(left: float, right: float) -> str:
    return f"{left:g}_{right:g}"


def _bin_counts(values: np.ndarray, bins: Sequence[float]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    finite = values[np.isfinite(values)]
    for idx in range(len(bins) - 1):
        left = float(bins[idx])
        right = float(bins[idx + 1])
        if idx == len(bins) - 2:
            mask = (finite >= left) & (finite <= right)
        else:
            mask = (finite >= left) & (finite < right)
        out[_bin_label(left, right)] = int(mask.sum())
    return out


def _max_delta(values: np.ndarray, horizon_steps: int, direction: str) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size <= horizon_steps or horizon_steps <= 0:
        return 0.0
    delta = finite[horizon_steps:] - finite[:-horizon_steps]
    if direction == "drop":
        delta = -delta
    return float(np.max(delta)) if delta.size else 0.0


def _motion_stats(head: np.ndarray) -> Dict[str, float]:
    if head.size == 0:
        return {
            "motion_norm_mean": float("nan"),
            "motion_norm_std": float("nan"),
            "motion_delta_norm_mean": float("nan"),
            "motion_delta_norm_std": float("nan"),
        }
    arr = np.asarray(head, dtype=np.float64)
    finite_rows = np.isfinite(arr).all(axis=1)
    arr = arr[finite_rows]
    if arr.size == 0:
        return {
            "motion_norm_mean": float("nan"),
            "motion_norm_std": float("nan"),
            "motion_delta_norm_mean": float("nan"),
            "motion_delta_norm_std": float("nan"),
        }
    norm = np.linalg.norm(arr, axis=1)
    delta_norm = np.linalg.norm(np.diff(arr, axis=0), axis=1) if arr.shape[0] > 1 else np.asarray([], dtype=np.float64)
    return {
        "motion_norm_mean": float(np.mean(norm)),
        "motion_norm_std": float(np.std(norm, ddof=0)),
        "motion_delta_norm_mean": float(np.mean(delta_norm)) if delta_norm.size else 0.0,
        "motion_delta_norm_std": float(np.std(delta_norm, ddof=0)) if delta_norm.size else 0.0,
    }


def _session_row(
    split: str,
    session: Any,
    calibration_steps: int,
    recent_steps: int,
    sampling_interval: float,
    rise_drop_horizon_steps: int,
    rise_drop_threshold: float,
    flat_range_threshold: float,
) -> Dict[str, Any]:
    start = max(int(calibration_steps), int(recent_steps) - 1)
    fms = np.asarray(session.fms_raw if session.fms_raw is not None else session.fms, dtype=np.float64)
    head = np.asarray(session.head_raw if session.head_raw is not None else session.head, dtype=np.float64)
    pred = fms[start : session.length]
    calib = fms[:calibration_steps] if calibration_steps > 0 else np.asarray([], dtype=np.float64)
    pred_finite = pred[np.isfinite(pred)]
    calib_finite = calib[np.isfinite(calib)]
    whole_finite = fms[np.isfinite(fms)]
    pred_range = float(np.max(pred_finite) - np.min(pred_finite)) if pred_finite.size else float("nan")
    pred_mean = float(np.mean(pred_finite)) if pred_finite.size else float("nan")
    pred_std = float(np.std(pred_finite, ddof=0)) if pred_finite.size else float("nan")
    cal_end = float(calib_finite[-1]) if calib_finite.size else float("nan")
    cal_mean = float(np.mean(calib_finite)) if calib_finite.size else float("nan")
    cal_range = float(np.max(calib_finite) - np.min(calib_finite)) if calib_finite.size else float("nan")
    motion_pred = _motion_stats(head[start : session.length])
    motion_calib = _motion_stats(head[:calibration_steps])
    max_rise = _max_delta(pred_finite, rise_drop_horizon_steps, "rise")
    max_drop = _max_delta(pred_finite, rise_drop_horizon_steps, "drop")
    low_fraction = float(np.mean(pred_finite <= 2.0)) if pred_finite.size else float("nan")
    midlow_fraction = float(np.mean((pred_finite > 2.0) & (pred_finite < 5.0))) if pred_finite.size else float("nan")
    high_fraction = float(np.mean(pred_finite >= 15.0)) if pred_finite.size else float("nan")
    warning_fraction = float(np.mean(pred_finite >= 12.0)) if pred_finite.size else float("nan")
    gender = str(session.gender or "unknown")
    return {
        "split": split,
        "participant_id": session.participant_id or "",
        "session_id": session.session_id,
        "source_file": session.source_file,
        "length": int(session.length),
        "prediction_start_index": int(start),
        "prediction_points": int(pred_finite.size),
        "age": _safe_float(session.age),
        "mssq": _safe_float(session.mssq),
        "gender": gender,
        "calibration_end_fms": cal_end,
        "calibration_mean_fms": cal_mean,
        "calibration_range_fms": cal_range,
        "prediction_mean_fms": pred_mean,
        "prediction_std_fms": pred_std,
        "prediction_min_fms": float(np.min(pred_finite)) if pred_finite.size else float("nan"),
        "prediction_max_fms": float(np.max(pred_finite)) if pred_finite.size else float("nan"),
        "prediction_range_fms": pred_range,
        "whole_mean_fms": float(np.mean(whole_finite)) if whole_finite.size else float("nan"),
        "whole_range_fms": float(np.max(whole_finite) - np.min(whole_finite)) if whole_finite.size else float("nan"),
        "low_fraction": low_fraction,
        "midlow_fraction": midlow_fraction,
        "warning_fraction": warning_fraction,
        "high_fraction": high_fraction,
        "max_rise_10s": max_rise,
        "max_drop_10s": max_drop,
        "net_delta": float(pred_finite[-1] - pred_finite[0]) if pred_finite.size >= 2 else 0.0,
        "flat_session": bool(math.isfinite(pred_range) and pred_range < flat_range_threshold),
        "rise_session": bool(max_rise >= rise_drop_threshold),
        "drop_session": bool(max_drop >= rise_drop_threshold),
        "low_dominant_session": bool(math.isfinite(low_fraction) and low_fraction >= 0.5),
        "high_dominant_session": bool(math.isfinite(high_fraction) and high_fraction >= 0.5),
        "motion_norm_mean": motion_pred["motion_norm_mean"],
        "motion_norm_std": motion_pred["motion_norm_std"],
        "motion_delta_norm_mean": motion_pred["motion_delta_norm_mean"],
        "motion_delta_norm_std": motion_pred["motion_delta_norm_std"],
        "calibration_motion_norm_mean": motion_calib["motion_norm_mean"],
        "calibration_motion_delta_norm_mean": motion_calib["motion_delta_norm_mean"],
    }


def _aggregate_split(split: str, rows: Sequence[Mapping[str, Any]], point_bins: Mapping[str, int]) -> Dict[str, Any]:
    participants = {str(row["participant_id"]) for row in rows if row.get("participant_id")}
    pred_points = int(sum(int(row["prediction_points"]) for row in rows))
    genders = Counter(str(row.get("gender") or "unknown") for row in rows)
    out: Dict[str, Any] = {
        "split": split,
        "participants": len(participants),
        "sessions": len(rows),
        "prediction_points": pred_points,
        "gender_counts": dict(genders),
    }
    metrics = [
        "age",
        "mssq",
        "calibration_end_fms",
        "calibration_mean_fms",
        "calibration_range_fms",
        "prediction_mean_fms",
        "prediction_std_fms",
        "prediction_range_fms",
        "whole_mean_fms",
        "whole_range_fms",
        "low_fraction",
        "midlow_fraction",
        "warning_fraction",
        "high_fraction",
        "max_rise_10s",
        "max_drop_10s",
        "net_delta",
        "motion_norm_mean",
        "motion_delta_norm_mean",
        "calibration_motion_norm_mean",
        "calibration_motion_delta_norm_mean",
    ]
    for name in metrics:
        values = [_safe_float(row.get(name)) for row in rows]
        out[f"{name}_mean"] = _mean(values)
        out[f"{name}_median"] = _median(values)
        out[f"{name}_std"] = _std(values)
        out[f"{name}_q25"] = _quantile(values, 0.25)
        out[f"{name}_q75"] = _quantile(values, 0.75)
    for flag in ["flat_session", "rise_session", "drop_session", "low_dominant_session", "high_dominant_session"]:
        out[f"{flag}_rate"] = float(sum(1 for row in rows if bool(row.get(flag))) / len(rows)) if rows else float("nan")
    for label, count in point_bins.items():
        out[f"bin_{label}_count"] = int(count)
        out[f"bin_{label}_fraction"] = float(count / pred_points) if pred_points else float("nan")
    return out


def _participant_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row.get("participant_id") or ""))].append(row)
    out: List[Dict[str, Any]] = []
    for (split, participant_id), items in sorted(grouped.items()):
        if not participant_id:
            continue
        out.append(
            {
                "split": split,
                "participant_id": participant_id,
                "sessions": len(items),
                "prediction_mean_fms": _mean([_safe_float(row["prediction_mean_fms"]) for row in items]),
                "prediction_range_fms": _mean([_safe_float(row["prediction_range_fms"]) for row in items]),
                "calibration_end_fms": _mean([_safe_float(row["calibration_end_fms"]) for row in items]),
                "low_fraction": _mean([_safe_float(row["low_fraction"]) for row in items]),
                "high_fraction": _mean([_safe_float(row["high_fraction"]) for row in items]),
                "flat_session_rate": _mean([1.0 if row.get("flat_session") else 0.0 for row in items]),
                "rise_session_rate": _mean([1.0 if row.get("rise_session") else 0.0 for row in items]),
                "drop_session_rate": _mean([1.0 if row.get("drop_session") else 0.0 for row in items]),
            }
        )
    return out


def _smd(train: Mapping[str, Any], other: Mapping[str, Any], metric: str) -> float:
    train_mean = _safe_float(train.get(f"{metric}_mean"))
    other_mean = _safe_float(other.get(f"{metric}_mean"))
    train_std = _safe_float(train.get(f"{metric}_std"))
    other_std = _safe_float(other.get(f"{metric}_std"))
    pooled = math.sqrt(max(train_std * train_std + other_std * other_std, 0.0) / 2.0)
    if not (math.isfinite(train_mean) and math.isfinite(other_mean) and pooled > 1e-8):
        return float("nan")
    return (other_mean - train_mean) / pooled


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _markdown_report(
    out_dir: Path,
    data_info: Mapping[str, Any],
    split_summary: Sequence[Mapping[str, Any]],
    smd_rows: Sequence[Mapping[str, Any]],
    val_test_smd_rows: Sequence[Mapping[str, Any]],
    participant_summary: Sequence[Mapping[str, Any]],
    session_rows: Sequence[Mapping[str, Any]],
) -> None:
    summary_by_split = {row["split"]: row for row in split_summary}
    lines: List[str] = [
        "# DenseFMS Split Diagnostic Report",
        "",
        "## Configuration",
        "",
        f"- sampling interval: `{_format(float(data_info['sampling_interval']))}` seconds",
        f"- calibration steps: `{int(data_info['calibration_steps'])}`",
        f"- recent/prediction start uses current-index start from the current experiment setting",
        f"- max session points: `{data_info.get('max_session_points')}`",
        f"- loaded sessions: `{data_info.get('session_count')}`",
        f"- participants: `{data_info.get('participant_count')}`",
        "",
        "## Split Summary",
        "",
        "| split | participants | sessions | points | mean FMS | calib-end | range | low frac | high frac | flat rate | rise rate | drop rate | motion delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in split_summary:
        lines.append(
            "| "
            f"{row['split']} | {int(row['participants'])} | {int(row['sessions'])} | {int(row['prediction_points'])} | "
            f"{_format(float(row['prediction_mean_fms_mean']))} | {_format(float(row['calibration_end_fms_mean']))} | "
            f"{_format(float(row['prediction_range_fms_mean']))} | {_format(float(row['low_fraction_mean']))} | "
            f"{_format(float(row['high_fraction_mean']))} | {_format(float(row['flat_session_rate']))} | "
            f"{_format(float(row['rise_session_rate']))} | {_format(float(row['drop_session_rate']))} | "
            f"{_format(float(row['motion_delta_norm_mean_mean']))} |"
        )
    lines.extend(
        [
            "",
            "## Prediction-Window FMS Bin Fractions",
            "",
            "| split | 0-2 | 2-5 | 5-10 | 10-15 | 15-20 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in split_summary:
        values = [
            row.get("bin_0_2_fraction", float("nan")),
            row.get("bin_2_5_fraction", float("nan")),
            row.get("bin_5_10_fraction", float("nan")),
            row.get("bin_10_15_fraction", float("nan")),
            row.get("bin_15_20_fraction", float("nan")),
        ]
        lines.append("| " + row["split"] + " | " + " | ".join(_format(float(v)) for v in values) + " |")
    lines.extend(
        [
            "",
            "## Standardized Difference vs Train",
            "",
            "Absolute values above about 0.5 are worth inspecting; above 0.8 is a large split shift signal.",
            "",
            "| split | metric | SMD | train mean | split mean |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in smd_rows:
        if row["split"] == "train":
            continue
        lines.append(
            "| "
            f"{row['split']} | {row['metric']} | {_format(float(row['smd']))} | "
            f"{_format(float(row['train_mean']))} | {_format(float(row['split_mean']))} |"
        )
    lines.extend(
        [
            "",
            "## Standardized Difference: Test vs Validation",
            "",
            "This section is the direct check for whether validation represents the final test split.",
            "",
            "| metric | SMD | val mean | test mean |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in val_test_smd_rows:
        lines.append(
            "| "
            f"{row['metric']} | {_format(float(row['smd']))} | "
            f"{_format(float(row['reference_mean']))} | {_format(float(row['compared_mean']))} |"
        )
    lines.extend(["", "## Extreme Participants", ""])
    for split in ["train", "val", "test"]:
        items = [row for row in participant_summary if row["split"] == split]
        high = sorted(items, key=lambda row: _safe_float(row["prediction_mean_fms"]), reverse=True)[:5]
        low = sorted(items, key=lambda row: _safe_float(row["prediction_mean_fms"]))[:5]
        lines.extend(
            [
                f"### {split}",
                "",
                "| group | participant | sessions | mean FMS | range | low frac | high frac |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, group_items in [("highest_mean", high), ("lowest_mean", low)]:
            for row in group_items:
                lines.append(
                    "| "
                    f"{label} | {row['participant_id']} | {int(row['sessions'])} | "
                    f"{_format(float(row['prediction_mean_fms']))} | {_format(float(row['prediction_range_fms']))} | "
                    f"{_format(float(row['low_fraction']))} | {_format(float(row['high_fraction']))} |"
                )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Hints",
            "",
            "- If validation has more rise/range sessions than test, dynamic range heads can look good on validation but over-move on test.",
            "- If test has more low or flat sessions, low-FMS overprediction will dominate test MAE.",
            "- If participant-level means differ strongly, validation selection may not represent the held-out participants.",
            "",
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    raw_sessions, mapping, data_info = load_raw_sessions(
        args.data_dir,
        calibration_seconds=float(args.calibration_seconds),
        horizon_seconds=float(args.horizon_seconds),
        default_sampling_interval=float(args.sampling_interval),
        max_session_points=int(args.max_session_points) if args.max_session_points is not None else None,
    )
    split_info = load_json(args.split_file)
    split_sessions = apply_saved_split(raw_sessions, split_info)
    sampling_interval = float(data_info["sampling_interval"])
    calibration_steps = int(data_info["calibration_steps"])
    recent_steps = seconds_to_steps(
        float(args.recent_window_seconds),
        sampling_interval,
        name="recent_window_seconds",
    )
    rise_drop_horizon_steps = seconds_to_steps(float(args.rise_drop_horizon_seconds), sampling_interval, name="rise_drop_horizon_seconds")
    rows: List[Dict[str, Any]] = []
    bin_counts_by_split: Dict[str, Counter[str]] = {split: Counter() for split in ["train", "val", "test"]}
    for split, sessions in split_sessions.items():
        for session in sessions:
            row = _session_row(
                split,
                session,
                calibration_steps,
                recent_steps,
                sampling_interval,
                rise_drop_horizon_steps,
                float(args.rise_drop_threshold),
                float(args.flat_range_threshold),
            )
            rows.append(row)
            start = int(row["prediction_start_index"])
            fms = np.asarray(session.fms_raw if session.fms_raw is not None else session.fms, dtype=np.float64)
            bin_counts_by_split[split].update(_bin_counts(fms[start : session.length], FMS_BINS))

    split_summary = [
        _aggregate_split(split, [row for row in rows if row["split"] == split], dict(bin_counts_by_split[split]))
        for split in ["train", "val", "test"]
    ]
    summary_by_split = {row["split"]: row for row in split_summary}
    smd_metrics = [
        "age",
        "mssq",
        "calibration_end_fms",
        "calibration_mean_fms",
        "calibration_range_fms",
        "prediction_mean_fms",
        "prediction_range_fms",
        "low_fraction",
        "warning_fraction",
        "high_fraction",
        "max_rise_10s",
        "max_drop_10s",
        "motion_delta_norm_mean",
        "calibration_motion_delta_norm_mean",
    ]
    smd_rows: List[Dict[str, Any]] = []
    train_summary = summary_by_split["train"]
    for split, split_row in summary_by_split.items():
        for metric in smd_metrics:
            smd_rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "smd": 0.0 if split == "train" else _smd(train_summary, split_row, metric),
                    "train_mean": train_summary.get(f"{metric}_mean"),
                    "split_mean": split_row.get(f"{metric}_mean"),
                }
            )
    val_test_smd_rows: List[Dict[str, Any]] = []
    val_summary = summary_by_split["val"]
    test_summary = summary_by_split["test"]
    for metric in smd_metrics:
        val_test_smd_rows.append(
            {
                "reference_split": "val",
                "compared_split": "test",
                "metric": metric,
                "smd": _smd(val_summary, test_summary, metric),
                "reference_mean": val_summary.get(f"{metric}_mean"),
                "compared_mean": test_summary.get(f"{metric}_mean"),
            }
        )
    participant_summary = _participant_rows(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        _write_csv(out_dir / "session_split_diagnostics.csv", rows, list(rows[0].keys()))
    if split_summary:
        _write_csv(out_dir / "split_summary.csv", split_summary, list(split_summary[0].keys()))
    if smd_rows:
        _write_csv(out_dir / "split_standardized_differences.csv", smd_rows, list(smd_rows[0].keys()))
    if val_test_smd_rows:
        _write_csv(
            out_dir / "val_test_standardized_differences.csv",
            val_test_smd_rows,
            list(val_test_smd_rows[0].keys()),
        )
    if participant_summary:
        _write_csv(out_dir / "participant_split_summary.csv", participant_summary, list(participant_summary[0].keys()))
    payload = {
        "args": vars(args),
        "data_info": data_info,
        "mapping": mapping,
        "split_info": split_info,
        "split_summary": split_summary,
        "standardized_differences": smd_rows,
        "val_test_standardized_differences": val_test_smd_rows,
        "participant_summary": participant_summary,
    }
    (out_dir / "split_diagnostic_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _markdown_report(out_dir, data_info, split_summary, smd_rows, val_test_smd_rows, participant_summary, rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sampling_interval", type=float, default=0.5)
    parser.add_argument("--calibration_seconds", type=float, default=120.0)
    parser.add_argument("--recent_window_seconds", type=float, default=10.0)
    parser.add_argument("--horizon_seconds", type=float, default=10.0)
    parser.add_argument("--max_session_points", type=int, default=420)
    parser.add_argument("--rise_drop_horizon_seconds", type=float, default=10.0)
    parser.add_argument("--rise_drop_threshold", type=float, default=3.0)
    parser.add_argument("--flat_range_threshold", type=float, default=2.5)
    args = parser.parse_args()
    payload = analyze(args)
    print(json.dumps({"out_dir": args.out_dir, "splits": payload["split_summary"]}, indent=2))


if __name__ == "__main__":
    main()
