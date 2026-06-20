"""Offline causal motion-dynamics diagnostics for online current-FMS tracking."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.densefms_forecast.data import inspect_dataset, load_raw_sessions
from src.densefms_forecast.model import append_motion_features
from src.densefms_forecast.utils import ensure_dir, seconds_to_steps


CAUSAL_DYNAMICS_V1_FEATURES = [
    "accel_norm",
    "gyro_norm",
    "motion_norm",
    "accel_delta_norm",
    "gyro_delta_norm",
    "motion_delta_norm",
    "accel_jerk_norm",
    "gyro_jerk_norm",
    "accel_energy_short",
    "gyro_energy_short",
    "motion_energy_long",
    "motion_delta_energy_long",
    "short_long_energy_ratio",
    "motion_jerk_energy",
    "sign_change_rate",
    "spectral_proxy",
    "channel_energy_entropy",
    "channel_participation_ratio",
]


def _safe_corr(x: Sequence[float], y: Sequence[float], *, spearman: bool = False) -> float:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[valid]
    yy = yy[valid]
    if xx.size < 3 or float(np.std(xx)) <= 1e-12 or float(np.std(yy)) <= 1e-12:
        return float("nan")
    if spearman:
        xx = pd.Series(xx).rank(method="average").to_numpy(dtype=np.float64)
        yy = pd.Series(yy).rank(method="average").to_numpy(dtype=np.float64)
    return float(np.corrcoef(xx, yy)[0, 1])


def _collect_rows(args: argparse.Namespace) -> pd.DataFrame:
    report, mapping = inspect_dataset(args.data_dir, artifacts_dir=args.out_dir)
    sessions, _, _ = load_raw_sessions(
        args.data_dir,
        mapping=mapping,
        calibration_seconds=args.calibration_seconds,
        horizon_seconds=args.horizon_seconds,
        default_sampling_interval=args.sampling_interval,
        max_session_points=args.max_session_points,
    )
    del report
    horizon_steps = seconds_to_steps(args.horizon_seconds, args.sampling_interval, name="horizon_seconds")
    calibration_steps = seconds_to_steps(args.calibration_seconds, args.sampling_interval, name="calibration_seconds")
    rows: List[Dict[str, float | str | int]] = []
    for session in sessions:
        if session.length <= calibration_steps + horizon_steps:
            continue
        with torch.no_grad():
            features = append_motion_features(
                torch.as_tensor(session.head, dtype=torch.float32).unsqueeze(0),
                "causal_dynamics_v1",
            )[0, :, 6:].cpu().numpy()
        fms = np.asarray(session.fms_raw if session.fms_raw is not None else session.fms, dtype=np.float64)
        last_current = min(session.length - horizon_steps, args.max_session_points or session.length)
        for idx in range(calibration_steps, last_current):
            current = float(fms[idx])
            future = float(fms[idx + horizon_steps])
            delta = future - current
            row: Dict[str, float | str | int] = {
                "participant_id": session.participant_id or "",
                "session_id": session.session_id,
                "source_file": session.source_file,
                "current_index": int(idx),
                "current_time": float(session.time[idx]) if idx < len(session.time) else float(idx) * args.sampling_interval,
                "fms": current,
                f"delta_{args.horizon_seconds:g}s": delta,
                f"rapid_rise_{args.horizon_seconds:g}s": int(delta >= args.rapid_rise_threshold),
            }
            for name, value in zip(CAUSAL_DYNAMICS_V1_FEATURES, features[idx]):
                row[name] = float(value)
            rows.append(row)
    if not rows:
        raise RuntimeError("No diagnostic rows were available after calibration/horizon filtering.")
    return pd.DataFrame(rows)


def _summarize(frame: pd.DataFrame, args: argparse.Namespace) -> List[Dict[str, float | str]]:
    delta_col = f"delta_{args.horizon_seconds:g}s"
    rapid_col = f"rapid_rise_{args.horizon_seconds:g}s"
    low = frame["fms"] < args.low_fms_threshold
    high = frame["fms"] >= args.high_fms_threshold
    rows: List[Dict[str, float | str]] = []
    for feature in CAUSAL_DYNAMICS_V1_FEATURES:
        low_values = frame.loc[low, feature].to_numpy(dtype=np.float64)
        high_values = frame.loc[high, feature].to_numpy(dtype=np.float64)
        low_mean = float(np.nanmean(low_values)) if low_values.size else float("nan")
        high_mean = float(np.nanmean(high_values)) if high_values.size else float("nan")
        pooled = frame[feature].to_numpy(dtype=np.float64)
        pooled_std = float(np.nanstd(pooled))
        rows.append(
            {
                "feature": feature,
                "pearson_fms": _safe_corr(frame[feature], frame["fms"]),
                "spearman_fms": _safe_corr(frame[feature], frame["fms"], spearman=True),
                "pearson_future_delta": _safe_corr(frame[feature], frame[delta_col]),
                "spearman_future_delta": _safe_corr(frame[feature], frame[delta_col], spearman=True),
                "pearson_rapid_rise": _safe_corr(frame[feature], frame[rapid_col]),
                "low_fms_mean": low_mean,
                "high_fms_mean": high_mean,
                "high_minus_low_effect": (high_mean - low_mean) / max(pooled_std, 1e-8) if math.isfinite(low_mean + high_mean) else float("nan"),
                "n": int(frame[feature].notna().sum()),
            }
        )
    rows.sort(key=lambda row: abs(float(row["spearman_fms"])) if math.isfinite(float(row["spearman_fms"])) else -1.0, reverse=True)
    return rows


def _write_report(rows: Sequence[Dict[str, float | str]], frame: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> None:
    md = [
        "# Online Current Motion Dynamics Diagnostic",
        "",
        f"- data_dir: `{args.data_dir}`",
        f"- max_session_points: {args.max_session_points}",
        f"- calibration_seconds: {args.calibration_seconds}",
        f"- horizon_seconds: {args.horizon_seconds}",
        f"- rows: {len(frame)}",
        "- features are computed causally at current time t; future FMS is used only as a diagnostic label.",
        "",
        "| feature | spearman_fms | pearson_delta | pearson_rapid | high-low effect |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows[:18]:
        md.append(
            "| {feature} | {spearman_fms:.4f} | {pearson_future_delta:.4f} | {pearson_rapid_rise:.4f} | {high_minus_low_effect:.4f} |".format(
                **row
            )
        )
    (out_dir / "motion_dynamics_diagnostic.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    sample = frame[["fms", "channel_energy_entropy"]].dropna()
    if len(sample) > args.max_plot_points:
        sample = sample.sample(args.max_plot_points, random_state=42)
    plt.figure(figsize=(7.0, 4.5))
    plt.scatter(sample["channel_energy_entropy"], sample["fms"], s=4, alpha=0.25)
    plt.xlabel("causal channel energy entropy")
    plt.ylabel("FMS")
    plt.tight_layout()
    plt.savefig(out_dir / "complexity_vs_fms.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose causal motion dynamics against FMS labels.")
    parser.add_argument("--data_dir", default="DenseFMS/Dataset")
    parser.add_argument("--out_dir", default="runs/online_fms_current_tracking_0509_integrated/motion_dynamics_diagnostic")
    parser.add_argument("--sampling_interval", type=float, default=0.5)
    parser.add_argument("--max_session_points", type=int, default=420)
    parser.add_argument("--calibration_seconds", type=float, default=120.0)
    parser.add_argument("--horizon_seconds", type=float, default=5.0)
    parser.add_argument("--rapid_rise_threshold", type=float, default=1.0)
    parser.add_argument("--low_fms_threshold", type=float, default=4.0)
    parser.add_argument("--high_fms_threshold", type=float, default=10.0)
    parser.add_argument("--max_plot_points", type=int, default=20000)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    frame = _collect_rows(args)
    rows = _summarize(frame, args)
    frame.to_csv(out_dir / "motion_dynamics_samples.csv", index=False)
    with (out_dir / "motion_dynamics_correlations.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_report(rows, frame, args, out_dir)
    print(f"Saved motion dynamics diagnostic to {out_dir}")


if __name__ == "__main__":
    main()
