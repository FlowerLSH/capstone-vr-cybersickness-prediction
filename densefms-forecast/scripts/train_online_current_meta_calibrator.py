"""Train a train-split-only meta calibrator for online current-FMS predictions.

The script fits a lightweight post-hoc model on saved train predictions and
applies it to another split. It is intended for validation-only selection:
train targets fit the calibrator, validation targets choose the method, and
test targets should be used only after the validation choice is fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_FEATURE_COLUMNS = [
    "predicted_fms_now",
    "predicted_fms_regression",
    "predicted_fms_ordinal",
    "anchor_fms",
    "current_time",
    "current_index",
    "age",
    "mssq",
    "p_rapid_rise_5s",
    "p_rapid_rise_10s",
    "p_rapid_drop_5s",
    "p_rapid_drop_10s",
]
KEY_COLUMNS = ["session_id", "current_index", "current_time", "target_fms_now"]


def _gender_numeric(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.lower().str.strip()
    return values.map({"male": 0.0, "m": 0.0, "female": 1.0, "f": 1.0}).fillna(0.5)


def _available_features(frame: pd.DataFrame) -> List[str]:
    features = [column for column in BASE_FEATURE_COLUMNS if column in frame.columns]
    features.extend(sorted(column for column in frame.columns if column.startswith("member_pred_")))
    return features


def _add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "gender" in out.columns:
        out["gender_numeric"] = _gender_numeric(out["gender"])
    if {"predicted_fms_now", "anchor_fms"}.issubset(out.columns):
        out["pred_minus_anchor"] = out["predicted_fms_now"].astype(float) - out["anchor_fms"].astype(float)
        out["pred_anchor_abs_gap"] = np.abs(out["pred_minus_anchor"].to_numpy(dtype=np.float64))
    if {"current_time", "calibration_seconds"}.issubset(out.columns):
        out["time_since_calibration"] = out["current_time"].astype(float) - out["calibration_seconds"].astype(float)
    if {"p_rapid_rise_5s", "p_rapid_drop_5s"}.issubset(out.columns):
        out["risk_balance_5s"] = out["p_rapid_rise_5s"].astype(float) - out["p_rapid_drop_5s"].astype(float)
    if {"p_rapid_rise_10s", "p_rapid_drop_10s"}.issubset(out.columns):
        out["risk_balance_10s"] = out["p_rapid_rise_10s"].astype(float) - out["p_rapid_drop_10s"].astype(float)
    member_cols = sorted(column for column in out.columns if column.startswith("member_pred_"))
    if member_cols:
        member_values = out[member_cols].astype(float).to_numpy(dtype=np.float64)
        out["member_pred_mean"] = np.mean(member_values, axis=1)
        out["member_pred_std"] = np.std(member_values, axis=1)
        out["member_pred_min"] = np.min(member_values, axis=1)
        out["member_pred_max"] = np.max(member_values, axis=1)
        out["member_pred_range"] = out["member_pred_max"] - out["member_pred_min"]
    if {"session_id", "current_index", "predicted_fms_now"}.issubset(out.columns):
        out = out.sort_values(["session_id", "current_index"]).copy()
        pred = pd.to_numeric(out["predicted_fms_now"], errors="coerce").astype(float)
        grouped = pred.groupby(out["session_id"], sort=False)
        count = grouped.cumcount() + 1
        causal_mean = grouped.expanding().mean().reset_index(level=0, drop=True)
        causal_std = grouped.expanding().std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
        causal_min = grouped.cummin()
        causal_max = grouped.cummax()
        first_pred = grouped.transform("first")
        out["pred_causal_count"] = count.astype(float)
        out["pred_causal_mean"] = causal_mean
        out["pred_causal_std"] = causal_std
        out["pred_causal_min"] = causal_min
        out["pred_causal_max"] = causal_max
        out["pred_causal_range"] = causal_max - causal_min
        out["pred_causal_delta_from_first"] = pred - first_pred
        out["pred_causal_slope_from_first"] = out["pred_causal_delta_from_first"] / count.clip(lower=1).astype(float)
        for window in (20, 60):
            rolling = grouped.rolling(window=window, min_periods=1)
            roll_mean = rolling.mean().reset_index(level=0, drop=True)
            roll_std = rolling.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
            roll_min = rolling.min().reset_index(level=0, drop=True)
            roll_max = rolling.max().reset_index(level=0, drop=True)
            out[f"pred_roll{window}_mean"] = roll_mean
            out[f"pred_roll{window}_std"] = roll_std
            out[f"pred_roll{window}_min"] = roll_min
            out[f"pred_roll{window}_max"] = roll_max
            out[f"pred_roll{window}_range"] = roll_max - roll_min
        if "anchor_fms" in out.columns:
            anchor = pd.to_numeric(out["anchor_fms"], errors="coerce").astype(float)
            out["pred_causal_mean_minus_anchor"] = out["pred_causal_mean"] - anchor
            out["pred_causal_delta_first_minus_anchor"] = first_pred - anchor
    return out


def _feature_columns(train: pd.DataFrame, eval_frame: pd.DataFrame) -> List[str]:
    train_aug = _add_derived_features(train.head(1))
    eval_aug = _add_derived_features(eval_frame.head(1))
    candidates = _available_features(train_aug)
    candidates.extend(
        [
            "gender_numeric",
            "pred_minus_anchor",
            "pred_anchor_abs_gap",
            "time_since_calibration",
            "risk_balance_5s",
            "risk_balance_10s",
            "member_pred_mean",
            "member_pred_std",
            "member_pred_min",
            "member_pred_max",
            "member_pred_range",
            "pred_causal_count",
            "pred_causal_mean",
            "pred_causal_std",
            "pred_causal_min",
            "pred_causal_max",
            "pred_causal_range",
            "pred_causal_delta_from_first",
            "pred_causal_slope_from_first",
            "pred_roll20_mean",
            "pred_roll20_std",
            "pred_roll20_min",
            "pred_roll20_max",
            "pred_roll20_range",
            "pred_roll60_mean",
            "pred_roll60_std",
            "pred_roll60_min",
            "pred_roll60_max",
            "pred_roll60_range",
            "pred_causal_mean_minus_anchor",
            "pred_causal_delta_first_minus_anchor",
        ]
    )
    seen = set()
    features: List[str] = []
    for column in candidates:
        if column in seen:
            continue
        seen.add(column)
        if column in train_aug.columns and column in eval_aug.columns:
            features.append(column)
    return features


def _matrix(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, list(features)].copy()
    for column in values.columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    fill = values.median(numeric_only=True)
    values = values.fillna(fill).fillna(0.0)
    return values.to_numpy(dtype=np.float64)


def _build_model(method: str, seed: int, alpha: float, max_iter: int):
    if method == "isotonic":
        return IsotonicRegression(y_min=0.0, y_max=20.0, out_of_bounds="clip")
    if method == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(alpha), random_state=int(seed)))
    if method == "huber":
        return make_pipeline(StandardScaler(), HuberRegressor(alpha=float(alpha), max_iter=int(max_iter)))
    if method == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=int(max_iter),
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=float(alpha),
            min_samples_leaf=64,
            random_state=int(seed),
        )
    if method == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(max_iter),
            max_depth=8,
            min_samples_leaf=64,
            max_features=0.8,
            random_state=int(seed),
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported method: {method}")


def _metrics(frame: pd.DataFrame) -> Dict[str, float]:
    target = frame["target_fms_now"].to_numpy(dtype=np.float64)
    pred = frame["predicted_fms_now"].to_numpy(dtype=np.float64)
    err = pred - target
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "prediction_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(target)),
        "prediction_std": float(np.std(pred)),
        "target_std": float(np.std(target)),
        "n": float(len(frame)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a train-only post-hoc calibrator and apply it to a split.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--method", choices=["isotonic", "ridge", "huber", "hgb", "extra_trees"], default="ridge")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip_min", type=float, default=0.0)
    parser.add_argument("--clip_max", type=float, default=20.0)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv).sort_values(KEY_COLUMNS).reset_index(drop=True)
    eval_frame = pd.read_csv(args.eval_csv).sort_values(KEY_COLUMNS).reset_index(drop=True)
    train_aug = _add_derived_features(train)
    eval_aug = _add_derived_features(eval_frame)
    if args.method == "isotonic":
        features = ["predicted_fms_now"]
    else:
        features = _feature_columns(train, eval_frame)
    if "target_fms_now" not in train_aug.columns or "target_fms_now" not in eval_aug.columns:
        raise ValueError("Both CSVs must include target_fms_now.")
    if not features:
        raise ValueError("No usable feature columns found.")

    x_train = _matrix(train_aug, features)
    y_train = train_aug["target_fms_now"].to_numpy(dtype=np.float64)
    x_eval = _matrix(eval_aug, features)
    model = _build_model(args.method, args.seed, args.alpha, args.max_iter)
    if args.method == "isotonic":
        model.fit(x_train[:, 0], y_train)
        pred = np.asarray(model.predict(x_eval[:, 0]), dtype=np.float64)
    else:
        model.fit(x_train, y_train)
        pred = np.asarray(model.predict(x_eval), dtype=np.float64)
    pred = np.clip(pred, float(args.clip_min), float(args.clip_max))

    out = eval_frame.copy()
    out["base_predicted_fms_now"] = out["predicted_fms_now"].astype(float)
    out["run_name"] = args.label or f"meta_calibrator_{args.method}"
    out["model_name"] = f"online_current_meta_calibrator_{args.method}"
    out["split"] = args.split
    out["predicted_fms_now"] = pred
    out["fms_absolute_error"] = np.abs(pred - out["target_fms_now"].to_numpy(dtype=np.float64))
    if "ordinal_bin_pred" in out.columns:
        bins = np.asarray([0, 2, 4, 6, 8, 10, 12, 15, 20], dtype=np.float64)
        out["ordinal_bin_pred"] = np.digitize(pred, bins[1:-1], right=False).astype(int)
    if "alarm_caution" in out.columns:
        out["alarm_caution"] = out["predicted_fms_now"] >= 8.0
    if "alarm_warning_high_fms" in out.columns:
        out["alarm_warning_high_fms"] = out["predicted_fms_now"] >= 12.0
    if "final_warning" in out.columns and "alarm_warning_high_fms" in out.columns:
        rapid = out["alarm_warning_rapid_rise"] if "alarm_warning_rapid_rise" in out.columns else False
        out["final_warning"] = out["alarm_warning_high_fms"] | rapid

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"{args.split}_predictions.csv", index=False)
    payload = {
        "task": {
            "meta_calibrator": True,
            "fit_split": "train",
            "eval_split": args.split,
            "test_eval_skipped": args.split != "test",
        },
        "method": args.method,
        "alpha": float(args.alpha),
        "max_iter": int(args.max_iter),
        "features": features,
        "train_csv": args.train_csv,
        "eval_csv": args.eval_csv,
        "metrics": _metrics(out),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload["metrics"], indent=2))
    print(f"Saved calibrated predictions to {out_dir}")


if __name__ == "__main__":
    main()
