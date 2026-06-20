"""Analyze online current-FMS tracker validation predictions.

This script intentionally works from saved prediction CSVs so it can compare
validation-only candidates without re-running training or touching test data.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FMS_BINS = [
    ("low", -float("inf"), 4.0),
    ("mid", 4.0, 10.0),
    ("high", 10.0, float("inf")),
]
TIME_BUCKETS = [
    ("early", -float("inf"), 180.0),
    ("middle", 180.0, 300.0),
    ("late", 300.0, float("inf")),
]


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_")[:90] or "session"


def _safe_corr(a: Sequence[float], b: Sequence[float], method: str = "pearson") -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[valid]
    bb = bb[valid]
    if aa.size < 2 or float(np.std(aa)) <= 1e-12 or float(np.std(bb)) <= 1e-12:
        return float("nan")
    if method == "spearman":
        aa = pd.Series(aa).rank(method="average").to_numpy(dtype=np.float64)
        bb = pd.Series(bb).rank(method="average").to_numpy(dtype=np.float64)
    return float(np.corrcoef(aa, bb)[0, 1])


def _nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _nanmedian(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _binary_prf(true_positive: np.ndarray, pred_positive: np.ndarray) -> Dict[str, float]:
    tp = float(np.sum(true_positive & pred_positive))
    fp = float(np.sum(~true_positive & pred_positive))
    fn = float(np.sum(true_positive & ~pred_positive))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _load_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    path = run_dir / f"{split}_predictions.csv"
    if not path.exists():
        eval_path = run_dir / f"eval_{split}" / f"{split}_predictions.csv"
        if eval_path.exists():
            path = eval_path
    if not path.exists():
        raise FileNotFoundError(f"Missing {split} prediction CSV for {run_dir}")
    frame = pd.read_csv(path)
    required = {"session_id", "current_time", "target_fms_now", "predicted_fms_now", "fms_absolute_error"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def _load_metrics(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _best_epoch_and_curve_mae(run_dir: Path) -> Tuple[float, float]:
    curves_path = run_dir / "training_curves.csv"
    if not curves_path.exists():
        return float("nan"), float("nan")
    curves = pd.read_csv(curves_path)
    if "val_mae" not in curves or curves.empty:
        return float("nan"), float("nan")
    idx = int(curves["val_mae"].idxmin())
    return float(curves.loc[idx, "epoch"]), float(curves.loc[idx, "val_mae"])


def _affine_from_train(train: Optional[pd.DataFrame]) -> Tuple[float, float, float]:
    if train is None or train.empty:
        return float("nan"), float("nan"), float("nan")
    pred = train["predicted_fms_now"].to_numpy(dtype=np.float64)
    target = train["target_fms_now"].to_numpy(dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(target)
    pred = pred[valid]
    target = target[valid]
    if pred.size < 2 or float(np.var(pred)) <= 1e-12:
        return float("nan"), float("nan"), float("nan")
    a, b = np.polyfit(pred, target, 1)
    corrected = a * pred + b
    return float(a), float(b), float(np.mean(np.abs(corrected - target)))


def _level_metrics(frame: pd.DataFrame, train_frame: Optional[pd.DataFrame]) -> Dict[str, float]:
    true = frame["target_fms_now"].to_numpy(dtype=np.float64)
    pred = frame["predicted_fms_now"].to_numpy(dtype=np.float64)
    err = pred - true
    metrics: Dict[str, float] = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float("nan"),
        "prediction_std": float(np.std(pred)),
        "target_std": float(np.std(true)),
        "prediction_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(true)),
        "calibration_bias": float(np.mean(pred - true)),
        "n": float(len(frame)),
    }
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    if ss_tot > 1e-12:
        metrics["r2"] = float(1.0 - ss_res / ss_tot)
    for name, lo, hi in FMS_BINS:
        mask = (true >= lo) & (true < hi)
        metrics[f"mae_fms_{name}"] = float(np.mean(np.abs(err[mask]))) if mask.any() else float("nan")
    time = frame["current_time"].to_numpy(dtype=np.float64)
    for name, lo, hi in TIME_BUCKETS:
        mask = (time >= lo) & (time < hi)
        metrics[f"mae_time_{name}"] = float(np.mean(np.abs(err[mask]))) if mask.any() else float("nan")
    a, b, train_mae = _affine_from_train(train_frame)
    metrics["train_affine_a"] = a
    metrics["train_affine_b"] = b
    metrics["train_affine_train_mae"] = train_mae
    if math.isfinite(a) and math.isfinite(b):
        corrected = a * pred + b
        metrics["train_affine_val_mae"] = float(np.mean(np.abs(corrected - true)))
    else:
        metrics["train_affine_val_mae"] = float("nan")
    return metrics


def _session_shape_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    pearson: List[float] = []
    spearman: List[float] = []
    centered_mae: List[float] = []
    z_mae: List[float] = []
    pred_range_ratios: List[float] = []
    flat_failures = 0
    range_sessions = 0
    for _, session in frame.groupby("session_id", sort=False):
        session = session.sort_values("current_time")
        true = session["target_fms_now"].to_numpy(dtype=np.float64)
        pred = session["predicted_fms_now"].to_numpy(dtype=np.float64)
        if true.size < 2:
            continue
        p_corr = _safe_corr(true, pred, "pearson")
        s_corr = _safe_corr(true, pred, "spearman")
        if math.isfinite(p_corr):
            pearson.append(p_corr)
        if math.isfinite(s_corr):
            spearman.append(s_corr)
        centered_mae.append(float(np.mean(np.abs((pred - np.mean(pred)) - (true - np.mean(true))))))
        true_std = float(np.std(true))
        pred_std = float(np.std(pred))
        if true_std > 1e-12 and pred_std > 1e-12:
            z_true = (true - np.mean(true)) / true_std
            z_pred = (pred - np.mean(pred)) / pred_std
            z_mae.append(float(np.mean(np.abs(z_pred - z_true))))
        true_range = float(np.max(true) - np.min(true))
        pred_range = float(np.max(pred) - np.min(pred))
        if true_range > 1e-12:
            ratio = pred_range / true_range
            pred_range_ratios.append(ratio)
            range_sessions += 1
            if ratio < 0.25:
                flat_failures += 1
    return {
        "pearson_session_mean": _nanmean(pearson),
        "pearson_session_median": _nanmedian(pearson),
        "pearson_session_count": float(len(pearson)),
        "spearman_session_mean": _nanmean(spearman),
        "spearman_session_median": _nanmedian(spearman),
        "spearman_session_count": float(len(spearman)),
        "centered_mae_session_mean": _nanmean(centered_mae),
        "shape_z_mae_session_mean": _nanmean(z_mae),
        "pred_true_range_ratio_mean": _nanmean(pred_range_ratios),
        "flat_range_lt25pct_session_rate": float(flat_failures / range_sessions) if range_sessions else float("nan"),
        "range_session_count": float(range_sessions),
    }


def _delta_metrics(frame: pd.DataFrame, horizons_seconds: Sequence[float], flat_threshold: float = 0.3) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    by_kind: Dict[str, List[float]] = {"rise": [], "drop": [], "plateau": []}
    for horizon_seconds in horizons_seconds:
        true_deltas: List[float] = []
        pred_deltas: List[float] = []
        lag_values: List[float] = []
        for _, session in frame.groupby("session_id", sort=False):
            session = session.sort_values("current_time")
            if session.empty:
                continue
            interval = float(session["sampling_interval"].dropna().iloc[0]) if "sampling_interval" in session else 0.5
            step = max(1, int(round(float(horizon_seconds) / max(interval, 1e-8))))
            true = session["target_fms_now"].to_numpy(dtype=np.float64)
            pred = session["predicted_fms_now"].to_numpy(dtype=np.float64)
            if true.size <= step:
                continue
            dt = true[step:] - true[:-step]
            dp = pred[step:] - pred[:-step]
            true_deltas.extend(dt.tolist())
            pred_deltas.extend(dp.tolist())
            if abs(float(horizon_seconds) - 5.0) < 1e-9:
                for value in dt:
                    if value >= 1.0:
                        by_kind["rise"].append(abs(float(value)))
                    elif value <= -1.0:
                        by_kind["drop"].append(abs(float(value)))
                    elif abs(float(value)) < flat_threshold:
                        by_kind["plateau"].append(abs(float(value)))
                lag = _best_lag_seconds(true, pred, interval, max_lag_seconds=5.0)
                if math.isfinite(lag):
                    lag_values.append(lag)

        dt_arr = np.asarray(true_deltas, dtype=np.float64)
        dp_arr = np.asarray(pred_deltas, dtype=np.float64)
        key = f"{float(horizon_seconds):g}s"
        valid = np.isfinite(dt_arr) & np.isfinite(dp_arr)
        dt_arr = dt_arr[valid]
        dp_arr = dp_arr[valid]
        if dt_arr.size:
            metrics[f"delta_mae_{key}"] = float(np.mean(np.abs(dp_arr - dt_arr)))
            metrics[f"delta_corr_{key}"] = _safe_corr(dt_arr, dp_arr)
            move = np.abs(dt_arr) >= flat_threshold
            rise = dt_arr >= flat_threshold
            drop = dt_arr <= -flat_threshold
            pred_sign = np.sign(dp_arr)
            true_sign = np.sign(dt_arr)
            metrics[f"direction_acc_{key}"] = float(np.mean(pred_sign[move] == true_sign[move])) if move.any() else float("nan")
            metrics[f"direction_acc_rise_{key}"] = float(np.mean(pred_sign[rise] > 0)) if rise.any() else float("nan")
            metrics[f"direction_acc_drop_{key}"] = float(np.mean(pred_sign[drop] < 0)) if drop.any() else float("nan")
            true_rise = dt_arr >= 1.0
            true_drop = dt_arr <= -1.0
            pred_rise = dp_arr >= 1.0
            pred_drop = dp_arr <= -1.0
            metrics[f"rise_f1_{key}"] = _binary_prf(true_rise, pred_rise)["f1"]
            metrics[f"drop_f1_{key}"] = _binary_prf(true_drop, pred_drop)["f1"]
            metrics[f"movement_balanced_f1_{key}"] = float(np.mean([metrics[f"rise_f1_{key}"], metrics[f"drop_f1_{key}"]]))
        else:
            for name in (
                "delta_mae",
                "delta_corr",
                "direction_acc",
                "direction_acc_rise",
                "direction_acc_drop",
                "rise_f1",
                "drop_f1",
                "movement_balanced_f1",
            ):
                metrics[f"{name}_{key}"] = float("nan")
        if abs(float(horizon_seconds) - 5.0) < 1e-9:
            metrics["lag_best_seconds_median_abs_5s_window"] = _nanmedian([abs(v) for v in lag_values])
            metrics["lag_best_seconds_mean_5s_window"] = _nanmean(lag_values)
    for kind, values in by_kind.items():
        metrics[f"segment_{kind}_mean_abs_delta_5s"] = _nanmean(values)
    return metrics


def _best_lag_seconds(true: np.ndarray, pred: np.ndarray, interval: float, max_lag_seconds: float = 5.0) -> float:
    if true.size < 4:
        return float("nan")
    true_slope = np.diff(true)
    pred_slope = np.diff(pred)
    if float(np.std(true_slope)) <= 1e-12 or float(np.std(pred_slope)) <= 1e-12:
        return float("nan")
    max_lag = max(1, int(round(max_lag_seconds / max(interval, 1e-8))))
    best_corr = -float("inf")
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = true_slope[-lag:]
            b = pred_slope[: len(pred_slope) + lag]
        elif lag > 0:
            a = true_slope[: len(true_slope) - lag]
            b = pred_slope[lag:]
        else:
            a = true_slope
            b = pred_slope
        corr = _safe_corr(a, b)
        if math.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag
    return float(best_lag * interval) if math.isfinite(best_corr) else float("nan")


def analyze_run(run_dir: Path, label: str, split: str, train_frame: Optional[pd.DataFrame]) -> Dict[str, float | str]:
    frame = _load_predictions(run_dir, split)
    metrics_json = _load_metrics(run_dir)
    best_epoch, best_curve_mae = _best_epoch_and_curve_mae(run_dir)
    row: Dict[str, float | str] = {
        "label": label,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_curve_val_mae": best_curve_mae,
        "test_eval_skipped": str(metrics_json.get("task", {}).get("test_eval_skipped", "")),
    }
    row.update(_level_metrics(frame, train_frame))
    row.update(_session_shape_metrics(frame))
    row.update(_delta_metrics(frame, [5.0, 10.0, 20.0, 30.0]))
    return row


def _select_sessions(primary: pd.DataFrame, count: int, mode: str) -> List[str]:
    grouped = primary.groupby("session_id")
    if mode == "worst":
        scores = grouped["fms_absolute_error"].mean().sort_values(ascending=False)
    elif mode == "dynamic":
        scores = grouped["target_fms_now"].agg(lambda s: float(np.max(s) - np.min(s))).sort_values(ascending=False)
    else:
        scores = grouped["fms_absolute_error"].mean().sort_values()
    return [str(v) for v in scores.head(max(1, int(count))).index.tolist()]


def _plot_trajectories(
    runs: List[Tuple[str, Path, pd.DataFrame]],
    out_dir: Path,
    primary_label: str,
    count: int,
    mode: str,
) -> None:
    primary = next((frame for label, _, frame in runs if label == primary_label), runs[0][2])
    sessions = _select_sessions(primary, count, mode)
    for idx, session_id in enumerate(sessions, start=1):
        plt.figure(figsize=(11, 4.8))
        true_done = False
        for label, _, frame in runs:
            session = frame[frame["session_id"] == session_id].sort_values("current_time")
            if session.empty:
                continue
            if not true_done:
                plt.plot(
                    session["current_time"],
                    session["target_fms_now"],
                    color="black",
                    linewidth=2.4,
                    label="true current FMS",
                )
                true_done = True
            plt.plot(
                session["current_time"],
                session["predicted_fms_now"],
                linewidth=1.65,
                alpha=0.92,
                label=label,
            )
        plt.xlabel("Time (s)")
        plt.ylabel("FMS")
        plt.title(session_id)
        plt.grid(alpha=0.22)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"trajectory_{mode}_{idx:02d}_{_safe_name(session_id)}.png", dpi=160)
        plt.close()


def _direction_accuracy(true: np.ndarray, pred: np.ndarray, step: int = 10, eps: float = 0.3) -> float:
    if true.size <= step or pred.size <= step:
        return float("nan")
    dt = true[step:] - true[:-step]
    dp = pred[step:] - pred[:-step]
    valid = np.isfinite(dt) & np.isfinite(dp)
    dt = dt[valid]
    dp = dp[valid]
    if dt.size == 0:
        return float("nan")
    true_cls = np.zeros(dt.shape, dtype=np.int64)
    pred_cls = np.zeros(dp.shape, dtype=np.int64)
    true_cls[dt > eps] = 1
    true_cls[dt < -eps] = -1
    pred_cls[dp > eps] = 1
    pred_cls[dp < -eps] = -1
    return float(np.mean(true_cls == pred_cls))


def _judge_session_curve(frame: pd.DataFrame) -> Dict[str, float | str]:
    session = frame.sort_values("current_time")
    true = session["target_fms_now"].to_numpy(dtype=np.float64)
    pred = session["predicted_fms_now"].to_numpy(dtype=np.float64)
    valid = np.isfinite(true) & np.isfinite(pred)
    true = true[valid]
    pred = pred[valid]
    if true.size < 4:
        return {
            "plot_judgment": "bad",
            "plot_notes": "too few valid points",
            "plot_corr": float("nan"),
            "plot_centered_mae": float("nan"),
            "plot_range_ratio": float("nan"),
            "plot_regime_error": float("nan"),
            "plot_direction_acc_5s": float("nan"),
        }

    corr = _safe_corr(true, pred)
    centered_mae = float(np.mean(np.abs((pred - np.mean(pred)) - (true - np.mean(true)))))
    true_range = float(np.max(true) - np.min(true))
    pred_range = float(np.max(pred) - np.min(pred))
    range_ratio = pred_range / true_range if true_range > 1e-8 else float("nan")
    regime_error = float(abs(np.mean(pred) - np.mean(true)))
    direction_acc = _direction_accuracy(true, pred, step=10, eps=0.3)

    severe: List[str] = []
    medium: List[str] = []
    if true_range >= 3.0 and math.isfinite(corr) and corr < -0.05:
        severe.append("opposite-direction")
    if true_range >= 4.0 and math.isfinite(range_ratio) and range_ratio < 0.25:
        severe.append("major-transition-underfit")
    if true_range < 2.0 and pred_range > 4.0:
        severe.append("false-rise-drop-on-plateau")
    if regime_error > 4.0:
        severe.append("wrong-fms-regime")
    if centered_mae > 2.6:
        severe.append("large-centered-error")
    if true_range >= 3.0 and math.isfinite(direction_acc) and direction_acc < 0.45:
        severe.append("poor-direction-5s")

    if not severe:
        if true_range >= 3.0 and math.isfinite(range_ratio) and range_ratio < 0.4:
            medium.append("weak-transition-amplitude")
        if regime_error > 2.5:
            medium.append("noticeable-regime-bias")
        if centered_mae > 1.8:
            medium.append("noticeable-shape-error")
        if math.isfinite(corr) and corr < 0.25 and true_range >= 3.0:
            medium.append("weak-trend-correlation")

    if severe:
        judgment = "bad"
        notes = ",".join(severe)
    elif medium:
        judgment = "medium"
        notes = ",".join(medium)
    else:
        judgment = "good"
        notes = "tracks main trend/regime"

    return {
        "plot_judgment": judgment,
        "plot_notes": notes,
        "plot_corr": float(corr),
        "plot_centered_mae": centered_mae,
        "plot_range_ratio": float(range_ratio),
        "plot_regime_error": regime_error,
        "plot_direction_acc_5s": float(direction_acc),
    }


def _write_plot_judgments(
    runs: List[Tuple[str, Path, pd.DataFrame]],
    out_dir: Path,
    primary_label: str,
    count: int,
) -> None:
    primary = next((frame for label, _, frame in runs if label == primary_label), runs[0][2])
    plot_sessions: List[Tuple[str, str]] = []
    for mode in ("best", "worst", "dynamic"):
        for session_id in _select_sessions(primary, count, mode):
            plot_sessions.append((mode, session_id))
    rows: List[Dict[str, float | str]] = []
    for mode, session_id in plot_sessions:
        for label, run_dir, frame in runs:
            session = frame[frame["session_id"] == session_id]
            if session.empty:
                continue
            row: Dict[str, float | str] = {
                "mode": mode,
                "session_id": session_id,
                "label": label,
                "run_dir": str(run_dir),
            }
            row.update(_judge_session_curve(session))
            rows.append(row)
    judgments = pd.DataFrame(rows)
    judgments.to_csv(out_dir / "plot_judgment_sessions.csv", index=False)
    if judgments.empty:
        pd.DataFrame(columns=["label", "plot_good", "plot_medium", "plot_bad", "plot_total"]).to_csv(
            out_dir / "plot_judgment_summary.csv",
            index=False,
        )
        return
    summary_rows: List[Dict[str, float | str]] = []
    for label, group in judgments.groupby("label", sort=False):
        counts = group["plot_judgment"].value_counts()
        summary_rows.append(
            {
                "label": str(label),
                "plot_good": int(counts.get("good", 0)),
                "plot_medium": int(counts.get("medium", 0)),
                "plot_bad": int(counts.get("bad", 0)),
                "plot_total": int(len(group)),
                "plot_set_primary_label": primary_label,
                "plot_set_count_per_mode": int(count),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "plot_judgment_summary.csv", index=False)


def _plot_scatter(runs: List[Tuple[str, Path, pd.DataFrame]], out_dir: Path, max_points: int) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(4.5 * len(runs), 4.6), sharex=True, sharey=True)
    if len(runs) == 1:
        axes = [axes]
    rng = np.random.default_rng(42)
    for ax, (label, _, frame) in zip(axes, runs):
        shown = frame
        if len(frame) > max_points:
            shown = frame.iloc[rng.choice(len(frame), size=max_points, replace=False)]
        ax.scatter(shown["target_fms_now"], shown["predicted_fms_now"], s=7, alpha=0.22)
        ax.plot([0, 20], [0, 20], color="black", linewidth=1.1)
        ax.set_title(label)
        ax.set_xlabel("True current FMS")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted current FMS")
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_scatter_all.png", dpi=160)
    plt.close(fig)


def _plot_dynamic_range(leaderboard: pd.DataFrame, out_dir: Path) -> None:
    plot_cols = ["mae", "pearson_session_mean", "centered_mae_session_mean", "flat_range_lt25pct_session_rate"]
    available = [col for col in plot_cols if col in leaderboard.columns]
    if not available:
        return
    fig, axes = plt.subplots(len(available), 1, figsize=(10, 2.8 * len(available)))
    if len(available) == 1:
        axes = [axes]
    labels = leaderboard["label"].astype(str).tolist()
    for ax, col in zip(axes, available):
        ax.bar(labels, leaderboard[col].astype(float).to_numpy())
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "trend_metric_summary.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze online current-FMS tracker predictions.")
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--primary_label", default=None)
    parser.add_argument("--trajectory_count", type=int, default=6)
    parser.add_argument("--max_scatter_points", type=int, default=8000)
    args = parser.parse_args()

    if len(args.run_dirs) != len(args.labels):
        raise ValueError("--run_dirs and --labels must have the same length.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_paths = [Path(value) for value in args.run_dirs]
    labels = [str(value) for value in args.labels]

    rows: List[Dict[str, float | str]] = []
    run_frames: List[Tuple[str, Path, pd.DataFrame]] = []
    for label, run_dir in zip(labels, run_paths):
        train_frame: Optional[pd.DataFrame]
        try:
            train_frame = _load_predictions(run_dir, "train")
        except FileNotFoundError:
            train_frame = None
        rows.append(analyze_run(run_dir, label, args.split, train_frame))
        run_frames.append((label, run_dir, _load_predictions(run_dir, args.split)))

    leaderboard = pd.DataFrame(rows).sort_values(["mae", "centered_mae_session_mean"], ascending=[True, True])
    leaderboard.to_csv(out_dir / "online_current_validation_leaderboard.csv", index=False)
    with open(out_dir / "online_current_validation_leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(leaderboard.to_json(orient="records")), f, indent=2)

    primary = args.primary_label or str(leaderboard.iloc[0]["label"])
    _plot_scatter(run_frames, out_dir, args.max_scatter_points)
    _plot_dynamic_range(leaderboard, out_dir)
    _plot_trajectories(run_frames, out_dir, primary, args.trajectory_count, "best")
    _plot_trajectories(run_frames, out_dir, primary, args.trajectory_count, "worst")
    _plot_trajectories(run_frames, out_dir, primary, args.trajectory_count, "dynamic")
    _write_plot_judgments(run_frames, out_dir, primary, args.trajectory_count)
    print(f"Saved analysis to {out_dir}")
    shown = [
        "label",
        "mae",
        "rmse",
        "pearson_session_mean",
        "centered_mae_session_mean",
        "delta_corr_5s",
        "direction_acc_5s",
        "flat_range_lt25pct_session_rate",
        "train_affine_val_mae",
    ]
    print(leaderboard[[col for col in shown if col in leaderboard.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
