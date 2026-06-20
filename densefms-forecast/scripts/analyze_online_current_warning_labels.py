"""Analyze leakage-safe warning-label prevalence for online-current DenseFMS.

This script does not evaluate model predictions.  It only summarizes future
high-FMS and rise/drop labels from DenseFMS ground truth under a saved split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from densefms_forecast.data import apply_saved_split, load_raw_sessions
from densefms_forecast.utils import ensure_dir, load_json, save_json, seconds_to_steps


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def _summarize_fms(values: Sequence[float], thresholds: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    out: Dict[str, Any] = {
        "n": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr)) if arr.size else float("nan"),
        "min": float(np.min(arr)) if arr.size else float("nan"),
        "max": float(np.max(arr)) if arr.size else float("nan"),
        "p10": float(np.percentile(arr, 10)) if arr.size else float("nan"),
        "p25": float(np.percentile(arr, 25)) if arr.size else float("nan"),
        "p50": float(np.percentile(arr, 50)) if arr.size else float("nan"),
        "p75": float(np.percentile(arr, 75)) if arr.size else float("nan"),
        "p90": float(np.percentile(arr, 90)) if arr.size else float("nan"),
    }
    for threshold in thresholds:
        out[f"ge_{float(threshold):g}"] = float(np.mean(arr >= float(threshold))) if arr.size else float("nan")
    return out


def _iter_prediction_indices(length: int, calibration_steps: int, max_horizon_steps: int) -> Iterable[int]:
    start = int(calibration_steps)
    end = int(length) - int(max_horizon_steps)
    for idx in range(start, max(start, end)):
        yield idx


def _analyze_split(
    sessions: Sequence[Any],
    split_name: str,
    calibration_steps: int,
    high_horizon_steps: Sequence[int],
    high_thresholds: Sequence[float],
    rise_specs: Sequence[tuple[int, float]],
    drop_specs: Sequence[tuple[int, float]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    max_horizon = max([1, *high_horizon_steps, *[h for h, _ in rise_specs], *[h for h, _ in drop_specs]])
    fms_all: List[float] = []
    fms_post: List[float] = []
    counters: Dict[str, List[float]] = {}
    for horizon_steps in high_horizon_steps:
        for threshold in high_thresholds:
            counters[f"high_{horizon_steps}_thr{float(threshold):g}"] = []
    for horizon_steps, delta in rise_specs:
        counters[f"rise_{horizon_steps}_d{float(delta):g}"] = []
    for horizon_steps, delta in drop_specs:
        counters[f"drop_{horizon_steps}_d{float(delta):g}"] = []

    for session in sessions:
        fms = np.asarray(session.fms_raw if session.fms_raw is not None else session.fms, dtype=np.float64)
        fms_all.extend(fms.tolist())
        fms_post.extend(fms[int(calibration_steps):].tolist())
        for idx in _iter_prediction_indices(len(fms), calibration_steps, max_horizon):
            current = float(fms[idx])
            for horizon_steps in high_horizon_steps:
                future = fms[idx + 1 : idx + horizon_steps + 1]
                future_max = float(np.max(future)) if future.size else float("nan")
                for threshold in high_thresholds:
                    counters[f"high_{horizon_steps}_thr{float(threshold):g}"].append(float(future_max >= float(threshold)))
            for horizon_steps, delta in rise_specs:
                future = fms[idx + 1 : idx + horizon_steps + 1]
                future_max = float(np.max(future)) if future.size else float("nan")
                counters[f"rise_{horizon_steps}_d{float(delta):g}"].append(float(future_max - current >= float(delta)))
            for horizon_steps, delta in drop_specs:
                future = fms[idx + 1 : idx + horizon_steps + 1]
                future_min = float(np.min(future)) if future.size else float("nan")
                counters[f"drop_{horizon_steps}_d{float(delta):g}"].append(float(current - future_min >= float(delta)))

    rows: List[Dict[str, Any]] = []
    for name, values in counters.items():
        rows.append(
            {
                "split": split_name,
                "label": name,
                "n": len(values),
                "positive_rate": _mean(values),
                "positive_count": int(np.sum(np.asarray(values, dtype=np.float64))) if values else 0,
            }
        )
    summary = {
        "split": split_name,
        "session_count": len(sessions),
        "fms_all": _summarize_fms(fms_all, high_thresholds),
        "fms_post_calibration": _summarize_fms(fms_post, high_thresholds),
    }
    return rows, summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "label", "n", "positive_count", "positive_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_report(path: Path, summaries: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Online Current Warning Label Diagnostic", ""]
    lines.append("이 보고서는 모델 prediction을 보지 않고 FMS label만으로 warning label 빈도를 계산한다.")
    lines.append("")
    lines.append("## FMS Distribution")
    lines.append("")
    lines.append("| split | post n | post mean | post p50 | post p75 | post p90 | ge8 | ge12 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for summary in summaries:
        post = summary["fms_post_calibration"]
        lines.append(
            "| {split} | {n} | {mean:.3f} | {p50:.3f} | {p75:.3f} | {p90:.3f} | {ge8:.4f} | {ge12:.4f} |".format(
                split=summary["split"],
                n=post["n"],
                mean=post["mean"],
                p50=post["p50"],
                p75=post["p75"],
                p90=post["p90"],
                ge8=post.get("ge_8", float("nan")),
                ge12=post.get("ge_12", float("nan")),
            )
        )
    lines.append("")
    lines.append("## Warning Label Prevalence")
    lines.append("")
    lines.append("| split | label | n | positives | positive rate |")
    lines.append("|---|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {split} | `{label}` | {n} | {pos} | {rate:.4f} |".format(
                split=row["split"],
                label=row["label"],
                n=row["n"],
                pos=row["positive_count"],
                rate=row["positive_rate"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze online-current warning label prevalence.")
    parser.add_argument("--data_dir", default="DenseFMS/Dataset")
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--report_dir", default="reports/online_current_warning_extension_0514/label_diagnostic")
    parser.add_argument("--sampling_interval", type=float, default=0.5)
    parser.add_argument("--calibration_seconds", type=float, default=120.0)
    parser.add_argument("--horizon_seconds", type=float, default=10.0)
    parser.add_argument("--max_session_points", type=int, default=420)
    parser.add_argument("--high_risk_horizon_seconds", nargs="+", type=float, default=[20.0])
    parser.add_argument("--high_risk_thresholds", nargs="+", type=float, default=[8.0, 12.0])
    parser.add_argument("--rise_specs", nargs="+", default=["10:2", "20:3"])
    parser.add_argument("--drop_specs", nargs="+", default=["10:2", "20:3"])
    args = parser.parse_args()

    calibration_steps = seconds_to_steps(args.calibration_seconds, args.sampling_interval, name="calibration_seconds")
    high_horizon_steps = [
        seconds_to_steps(value, args.sampling_interval, name="high_risk_horizon_seconds")
        for value in args.high_risk_horizon_seconds
    ]

    def _parse_specs(values: Sequence[str], name: str) -> List[tuple[int, float]]:
        specs: List[tuple[int, float]] = []
        for value in values:
            seconds_text, delta_text = str(value).split(":", maxsplit=1)
            specs.append((seconds_to_steps(float(seconds_text), args.sampling_interval, name=name), float(delta_text)))
        return specs

    rise_specs = _parse_specs(args.rise_specs, "rise_specs")
    drop_specs = _parse_specs(args.drop_specs, "drop_specs")
    raw_sessions, _mapping, data_info = load_raw_sessions(
        args.data_dir,
        calibration_seconds=float(args.calibration_seconds),
        horizon_seconds=float(args.horizon_seconds),
        default_sampling_interval=float(args.sampling_interval),
        max_session_points=int(args.max_session_points) if args.max_session_points is not None else None,
    )
    split_info = load_json(args.split_file) if args.split_file else None
    if split_info is not None:
        split_sessions = apply_saved_split(raw_sessions, split_info)
    else:
        split_sessions = {"all": raw_sessions}

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for split_name, sessions in split_sessions.items():
        rows, summary = _analyze_split(
            sessions,
            split_name,
            calibration_steps,
            high_horizon_steps,
            [float(v) for v in args.high_risk_thresholds],
            rise_specs,
            drop_specs,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    report_dir = ensure_dir(args.report_dir)
    _write_csv(report_dir / "warning_label_prevalence.csv", all_rows)
    payload = {
        "config": vars(args),
        "data_info": data_info,
        "calibration_steps": calibration_steps,
        "high_risk_horizon_steps": high_horizon_steps,
        "rise_specs_steps": rise_specs,
        "drop_specs_steps": drop_specs,
        "split_summaries": summaries,
        "label_rows": all_rows,
    }
    save_json(report_dir / "warning_label_diagnostic.json", payload)
    _write_report(report_dir / "warning_label_diagnostic.md", summaries, all_rows)
    print(json.dumps({"report_dir": str(report_dir), "rows": len(all_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
