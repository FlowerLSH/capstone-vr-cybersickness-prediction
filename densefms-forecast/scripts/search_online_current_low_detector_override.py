"""Train-only low-FMS detector with validation-selected override policy.

This script fits a low-state detector on train predictions only. Validation
selects a threshold and deployment-style override policy. Test can be supplied
only after validation selection has fixed the policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from densefms_forecast.data import read_csv_robust


DEFAULT_MEMBERS = [
    "member_pred_selected_risk035",
    "member_pred_risk045",
    "member_pred_zero_anchor",
    "member_pred_range_scaled",
]

MOTION_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
]


def _gender_numeric(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.lower().str.strip()
    return values.map({"male": 0.0, "m": 0.0, "female": 1.0, "f": 1.0}).fillna(0.5)


def _resolve_source_path(source_file: str, source_root: Path) -> Optional[Path]:
    text = str(source_file or "").strip()
    if not text:
        return None
    normalized = Path(text.replace("\\", "/"))
    candidates = [normalized]
    if not normalized.is_absolute():
        candidates.append(source_root / normalized)
        candidates.append(source_root / normalized.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _slope(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if int(finite.sum()) < 2:
        return 0.0
    y = values[finite].astype(np.float64)
    x = np.arange(values.size, dtype=np.float64)[finite]
    x = x - float(x.mean())
    denom = float(np.sum(x * x))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x * (y - float(y.mean()))) / denom)


def _safe_stats(prefix: str, values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)].astype(np.float64)
    if finite.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_first": 0.0,
            f"{prefix}_last": 0.0,
            f"{prefix}_delta": 0.0,
            f"{prefix}_slope": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_first": float(finite[0]),
        f"{prefix}_last": float(finite[-1]),
        f"{prefix}_delta": float(finite[-1] - finite[0]),
        f"{prefix}_slope": _slope(values),
    }


def _calibration_summary_for_file(path: Path, calibration_steps: int) -> Dict[str, float]:
    raw, _meta = read_csv_robust(path)
    n = max(1, int(calibration_steps))
    cal = raw.iloc[:n].copy()
    fms = pd.to_numeric(cal.get("fms", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float64)
    row = _safe_stats("calib_fms", fms)
    if fms.size:
        row["calib_fms_frac_low2"] = float(np.mean(fms[np.isfinite(fms)] < 2.0)) if np.isfinite(fms).any() else 0.0
        row["calib_fms_frac_low5"] = float(np.mean(fms[np.isfinite(fms)] < 5.0)) if np.isfinite(fms).any() else 0.0
        row["calib_fms_frac_high8"] = float(np.mean(fms[np.isfinite(fms)] >= 8.0)) if np.isfinite(fms).any() else 0.0
        row["calib_fms_frac_high12"] = float(np.mean(fms[np.isfinite(fms)] >= 12.0)) if np.isfinite(fms).any() else 0.0
        for window in (20, 60):
            tail = fms[-min(window, fms.size) :]
            row[f"calib_fms_last{window}_mean"] = float(np.nanmean(tail)) if np.isfinite(tail).any() else 0.0
            row[f"calib_fms_last{window}_std"] = float(np.nanstd(tail)) if np.isfinite(tail).any() else 0.0
    motion_arrays: List[np.ndarray] = []
    for col in MOTION_COLUMNS:
        if col in cal.columns:
            values = pd.to_numeric(cal[col], errors="coerce").to_numpy(dtype=np.float64)
            row.update(_safe_stats(f"motion_calib_{col}", values))
            motion_arrays.append(values)
    if motion_arrays:
        motion = np.vstack(motion_arrays).T
        acc = motion[:, :3] if motion.shape[1] >= 3 else motion
        gyro = motion[:, 3:6] if motion.shape[1] >= 6 else motion[:, :0]
        acc_mag = np.linalg.norm(np.nan_to_num(acc, nan=0.0), axis=1)
        row.update(_safe_stats("motion_calib_acc_mag", acc_mag))
        if gyro.size:
            gyro_mag = np.linalg.norm(np.nan_to_num(gyro, nan=0.0), axis=1)
            row.update(_safe_stats("motion_calib_gyro_mag", gyro_mag))
        if motion.shape[0] > 1:
            diff_mag = np.linalg.norm(np.diff(np.nan_to_num(motion, nan=0.0), axis=0), axis=1)
            row.update(_safe_stats("motion_calib_diff_mag", diff_mag))
    return row


def _add_calibration_summary_features(frame: pd.DataFrame, source_root: Path) -> pd.DataFrame:
    if "source_file" not in frame.columns or "calibration_steps" not in frame.columns:
        return frame.copy()
    out = frame.copy()
    cache: Dict[Tuple[str, int], Dict[str, float]] = {}
    summary_rows: List[Dict[str, float]] = []
    for source_file, steps in zip(out["source_file"], out["calibration_steps"]):
        key = (str(source_file), int(float(steps)))
        if key not in cache:
            path = _resolve_source_path(key[0], source_root)
            cache[key] = _calibration_summary_for_file(path, key[1]) if path is not None else {}
        summary_rows.append(cache[key])
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return out
    summary = summary.fillna(0.0)
    return pd.concat([out.reset_index(drop=True), summary.reset_index(drop=True)], axis=1)


def _add_features(frame: pd.DataFrame, members: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    out = frame.copy().sort_values(["session_id", "current_index"]).reset_index(drop=True)
    member_values = out.loc[:, list(members)].astype(float).to_numpy(dtype=np.float64)
    out["member_mean"] = np.mean(member_values, axis=1)
    out["member_std"] = np.std(member_values, axis=1)
    out["member_min"] = np.min(member_values, axis=1)
    out["member_max"] = np.max(member_values, axis=1)
    out["member_range"] = out["member_max"] - out["member_min"]
    if "gender" in out.columns:
        out["gender_numeric"] = _gender_numeric(out["gender"])
    if "anchor_fms" in out.columns:
        anchor = pd.to_numeric(out["anchor_fms"], errors="coerce").astype(float)
        out["member_mean_minus_anchor"] = out["member_mean"] - anchor
        out["member_min_minus_anchor"] = out["member_min"] - anchor
        out["member_max_minus_anchor"] = out["member_max"] - anchor
        for col in members:
            out[f"{col}_minus_anchor"] = pd.to_numeric(out[col], errors="coerce").astype(float) - anchor
    if {"current_time", "calibration_seconds"}.issubset(out.columns):
        out["time_since_calibration"] = (
            pd.to_numeric(out["current_time"], errors="coerce").astype(float)
            - pd.to_numeric(out["calibration_seconds"], errors="coerce").astype(float)
        )
    # Causal prediction-history features are deployable because they use only
    # previous model outputs within the same session.
    grouped = out.groupby("session_id", sort=False)
    for source in ["member_mean", "member_min", "member_max"]:
        values = pd.to_numeric(out[source], errors="coerce").astype(float)
        by_session = values.groupby(out["session_id"], sort=False)
        count = by_session.cumcount() + 1
        out[f"{source}_causal_mean"] = by_session.expanding().mean().reset_index(level=0, drop=True)
        out[f"{source}_causal_std"] = by_session.expanding().std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
        out[f"{source}_causal_min"] = by_session.cummin()
        out[f"{source}_causal_max"] = by_session.cummax()
        first = by_session.transform("first")
        out[f"{source}_delta_from_first"] = values - first
        out[f"{source}_slope_from_first"] = out[f"{source}_delta_from_first"] / count.clip(lower=1).astype(float)
        for window in (20, 60):
            rolling = by_session.rolling(window=window, min_periods=1)
            out[f"{source}_roll{window}_mean"] = rolling.mean().reset_index(level=0, drop=True)
            out[f"{source}_roll{window}_std"] = rolling.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
            out[f"{source}_roll{window}_min"] = rolling.min().reset_index(level=0, drop=True)
            out[f"{source}_roll{window}_max"] = rolling.max().reset_index(level=0, drop=True)

    base = [
        "anchor_fms",
        "current_time",
        "current_index",
        "age",
        "mssq",
        "gender_numeric",
        "time_since_calibration",
    ]
    features = [
        col
        for col in out.columns
        if col in base
        or col in members
        or col.startswith("member_")
        or col.startswith("calib_")
        or col.startswith("motion_calib_")
        or col.endswith("_minus_anchor")
    ]
    return out, features


def _matrix(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median(numeric_only=True)).fillna(0.0)
    return values.to_numpy(dtype=np.float64)


def _models(seed: int) -> Dict[str, Any]:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "hgb": HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=64,
            random_state=int(seed),
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=48,
            max_features=0.8,
            random_state=int(seed),
            n_jobs=-1,
            class_weight="balanced",
        ),
    }


def _base_prediction(frame: pd.DataFrame, base_col: str) -> np.ndarray:
    if base_col == "equal4":
        members = sorted(col for col in frame.columns if col.startswith("member_pred_"))
        return frame.loc[:, members].astype(float).to_numpy(dtype=np.float64).mean(axis=1)
    if base_col in frame.columns:
        return pd.to_numeric(frame[base_col], errors="coerce").to_numpy(dtype=np.float64)
    raise ValueError(f"Missing base prediction column: {base_col}")


def _override_prediction(
    frame: pd.DataFrame,
    base: np.ndarray,
    active: np.ndarray,
    mode: str,
    value: float,
    strength: float,
) -> np.ndarray:
    if mode == "constant":
        replacement = np.full_like(base, float(value), dtype=np.float64)
    elif mode == "min_member":
        members = sorted(col for col in frame.columns if col.startswith("member_pred_"))
        replacement = frame.loc[:, members].astype(float).to_numpy(dtype=np.float64).min(axis=1)
    elif mode == "min_member_cap":
        members = sorted(col for col in frame.columns if col.startswith("member_pred_"))
        replacement = np.minimum(frame.loc[:, members].astype(float).to_numpy(dtype=np.float64).min(axis=1), float(value))
    elif mode == "anchor_cap":
        replacement = np.minimum(base, pd.to_numeric(frame["anchor_fms"], errors="coerce").to_numpy(dtype=np.float64) + float(value))
    else:
        raise ValueError(f"Unknown override mode: {mode}")
    mixed = base * (1.0 - float(strength)) + replacement * float(strength)
    return np.clip(np.where(active, mixed, base), 0.0, 20.0)


def _high_metrics(y: np.ndarray, pred: np.ndarray, threshold: float) -> Dict[str, float]:
    true = y >= float(threshold)
    hit = pred >= float(threshold)
    tp = float(np.sum(true & hit))
    fp = float(np.sum(~true & hit))
    fn = float(np.sum(true & ~hit))
    tn = float(np.sum(~true & ~hit))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        f"high{threshold:g}_precision": precision,
        f"high{threshold:g}_recall": recall,
        f"high{threshold:g}_f1": f1,
        f"high{threshold:g}_fpr": fp / (fp + tn) if fp + tn > 0 else 0.0,
        f"high{threshold:g}_fnr": fn / (tp + fn) if tp + fn > 0 else 0.0,
    }


def _metrics(frame: pd.DataFrame, pred: np.ndarray, low_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y = pd.to_numeric(frame["target_fms_now"], errors="coerce").to_numpy(dtype=np.float64)
    low = (y >= 0.0) & (y < 2.0)
    active = low_prob >= float(threshold)
    err = pred - y
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    tp = float(np.sum(active & low))
    fp = float(np.sum(active & ~low))
    fn = float(np.sum(~active & low))
    row = {
        "n": int(y.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "r2": 1.0 - float(np.sum(err * err)) / ss_tot if ss_tot > 1e-12 else float("nan"),
        "original_low_0_2_n": int(np.sum(low)),
        "original_low_0_2_bias": float(np.mean(err[low])) if np.any(low) else float("nan"),
        "original_low_0_2_mae": float(np.mean(np.abs(err[low]))) if np.any(low) else float("nan"),
        "low_detector_precision": tp / (tp + fp) if tp + fp > 0 else 0.0,
        "low_detector_recall": tp / (tp + fn) if tp + fn > 0 else 0.0,
        "low_detector_active_rate": float(np.mean(active)),
    }
    row.update(_high_metrics(y, pred, 8.0))
    row.update(_high_metrics(y, pred, 12.0))
    row["goal_composite_strict120"] = (
        float(row["mae"])
        + 0.25 * max(0.0, float(row["original_low_0_2_bias"]) - 2.5)
        + 2.0 * max(0.0, 0.70 - float(row["r2"]))
        + 0.5 * max(0.0, 0.76 - float(row["high12_f1"]))
    )
    row["goal_pass_count"] = int(row["mae"] <= 1.8) + int(row["r2"] >= 0.75) + int(row["original_low_0_2_bias"] <= 2.5)
    return row


def _score(row: Mapping[str, Any], mode: str) -> tuple[float, ...]:
    if mode == "score_only":
        return (
            float(row["goal_composite_strict120"]),
            float(row["mae"]),
            max(0.0, float(row["original_low_0_2_bias"]) - 2.5),
            -float(row["r2"]),
        )
    if mode == "low_then_score":
        return (
            max(0.0, float(row["original_low_0_2_bias"]) - 2.5),
            float(row["goal_composite_strict120"]),
            float(row["mae"]),
            -float(row["r2"]),
        )
    raise ValueError("selection_mode must be score_only or low_then_score.")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_predictions(path: Path, frame: pd.DataFrame, pred: np.ndarray, label: str, split: str) -> None:
    out = frame.copy()
    out["run_name"] = label
    out["model_name"] = "online_current_low_detector_override"
    out["split"] = split
    out["base_predicted_fms_now"] = out["predicted_fms_now"] if "predicted_fms_now" in out.columns else np.nan
    out["predicted_fms_now"] = pred
    out["fms_absolute_error"] = np.abs(pred - pd.to_numeric(out["target_fms_now"], errors="coerce").to_numpy(dtype=np.float64))
    if "alarm_caution" in out.columns:
        out["alarm_caution"] = out["predicted_fms_now"] >= 8.0
    if "alarm_warning_high_fms" in out.columns:
        out["alarm_warning_high_fms"] = out["predicted_fms_now"] >= 12.0
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--member_cols", nargs="*", default=DEFAULT_MEMBERS)
    parser.add_argument("--base_col", default="predicted_fms_now")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--override_modes", nargs="+", default=["constant", "min_member", "min_member_cap", "anchor_cap"])
    parser.add_argument("--override_values", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--selection_mode", choices=["score_only", "low_then_score"], default="score_only")
    parser.add_argument("--label", default="low_detector_override")
    parser.add_argument("--use_calibration_summary", action="store_true")
    parser.add_argument("--source_root", default=".")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_raw = pd.read_csv(args.train_csv)
    val_raw = pd.read_csv(args.val_csv)
    if args.use_calibration_summary:
        source_root = Path(args.source_root)
        train_raw = _add_calibration_summary_features(train_raw, source_root)
        val_raw = _add_calibration_summary_features(val_raw, source_root)
    train, features = _add_features(train_raw, args.member_cols)
    val, _ = _add_features(val_raw, args.member_cols)
    x_train = _matrix(train, features)
    x_val = _matrix(val, features)
    y_train = pd.to_numeric(train["target_fms_now"], errors="coerce").to_numpy(dtype=np.float64)
    low_train = ((y_train >= 0.0) & (y_train < 2.0)).astype(np.int64)
    base_val = _base_prediction(val, args.base_col)

    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_pred: Optional[np.ndarray] = None
    best_prob: Optional[np.ndarray] = None
    fitted_models: Dict[str, Any] = {}
    for model_name, model in _models(args.seed).items():
        model.fit(x_train, low_train)
        fitted_models[model_name] = model
        if not hasattr(model, "predict_proba"):
            continue
        low_prob = np.asarray(model.predict_proba(x_val)[:, 1], dtype=np.float64)
        for threshold in args.thresholds:
            active = low_prob >= float(threshold)
            for override_mode in args.override_modes:
                values = [0.0] if override_mode == "min_member" else args.override_values
                for override_value in values:
                    for strength in args.strengths:
                        pred = _override_prediction(val, base_val, active, override_mode, float(override_value), float(strength))
                        row: Dict[str, Any] = {
                            "model": model_name,
                            "threshold": float(threshold),
                            "override_mode": override_mode,
                            "override_value": float(override_value),
                            "strength": float(strength),
                        }
                        row.update(_metrics(val, pred, low_prob, threshold))
                        rows.append(row)
                        if best is None or _score(row, args.selection_mode) < _score(best, args.selection_mode):
                            best = row
                            best_pred = pred.copy()
                            best_prob = low_prob.copy()
    assert best is not None and best_pred is not None and best_prob is not None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "validation_low_detector_override_grid.csv", rows)
    _save_predictions(out_dir / "val_predictions.csv", val, best_pred, args.label, "val")
    payload: Dict[str, Any] = {
        "selection_mode": args.selection_mode,
        "label": args.label,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "features": features,
        "selected_validation": best,
        "validation_metrics": _metrics(val, best_pred, best_prob, float(best["threshold"])),
    }
    if args.test_csv:
        test_raw = pd.read_csv(args.test_csv)
        if args.use_calibration_summary:
            test_raw = _add_calibration_summary_features(test_raw, Path(args.source_root))
        test, _ = _add_features(test_raw, args.member_cols)
        x_test = _matrix(test, features)
        model = fitted_models[str(best["model"])]
        low_prob_test = np.asarray(model.predict_proba(x_test)[:, 1], dtype=np.float64)
        base_test = _base_prediction(test, args.base_col)
        pred_test = _override_prediction(
            test,
            base_test,
            low_prob_test >= float(best["threshold"]),
            str(best["override_mode"]),
            float(best["override_value"]),
            float(best["strength"]),
        )
        _save_predictions(out_dir / "test_predictions.csv", test, pred_test, args.label, "test")
        payload["test_metrics"] = _metrics(test, pred_test, low_prob_test, float(best["threshold"]))
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"selected_validation": best, "validation_metrics": payload["validation_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
