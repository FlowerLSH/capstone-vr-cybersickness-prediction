"""Shared utilities for DenseFMS forecasting experiments."""

from __future__ import annotations

import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_jsonable(obj: Any) -> Any:
    """Convert NumPy/PyTorch scalar containers into JSON-native values."""
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def timestamp_for_run() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def smooth_l1_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """SmoothL1 averaged only over valid elements."""
    safe_pred = torch.where(mask, pred, torch.zeros_like(pred))
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    loss = torch.nn.functional.smooth_l1_loss(safe_pred, safe_target, reduction="none")
    mask_f = mask.to(loss.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (loss * mask_f).sum() / denom


def denormalize_fms(values: np.ndarray | torch.Tensor, scaler: Mapping[str, float]) -> np.ndarray:
    arr = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    f_min = float(scaler["min"])
    f_max = float(scaler["max"])
    return arr * (f_max - f_min) + f_min


def normalize_fms(values: np.ndarray, scaler: Mapping[str, float]) -> np.ndarray:
    f_min = float(scaler["min"])
    f_max = float(scaler["max"])
    denom = max(f_max - f_min, 1e-8)
    return (values - f_min) / denom


def compute_regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if valid.sum() == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "smape": float("nan"),
            "acc_within_0.5": float("nan"),
            "acc_within_1.0": float("nan"),
            "acc_within_2.0": float("nan"),
            "n": 0,
        }
    yt = y_true_arr[valid]
    yp = y_pred_arr[valid]
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    denom = np.maximum((np.abs(yt) + np.abs(yp)) / 2.0, 1e-8)
    smape = float(np.mean(np.abs(err) / denom) * 100.0)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "smape": smape,
        "acc_within_0.5": float(np.mean(np.abs(err) <= 0.5)),
        "acc_within_1.0": float(np.mean(np.abs(err) <= 1.0)),
        "acc_within_2.0": float(np.mean(np.abs(err) <= 2.0)),
        "n": int(valid.sum()),
    }


def compute_high_fms_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    threshold: float = 7.0,
) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(yt) & np.isfinite(yp)
    if valid.sum() == 0:
        return {
            "high_fms_threshold": float(threshold),
            "high_fms_false_positive_rate": float("nan"),
            "high_fms_false_negative_rate": float("nan"),
            "high_fms_precision": float("nan"),
            "high_fms_recall": float("nan"),
            "high_fms_f1": float("nan"),
        }
    yt = yt[valid] >= threshold
    yp = yp[valid] >= threshold
    tp = float(np.sum(yp & yt))
    fp = float(np.sum(yp & ~yt))
    tn = float(np.sum(~yp & ~yt))
    fn = float(np.sum(~yp & yt))
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    f1 = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0 else float("nan")
    return {
        "high_fms_threshold": float(threshold),
        "high_fms_false_positive_rate": fp / (fp + tn) if fp + tn > 0 else float("nan"),
        "high_fms_false_negative_rate": fn / (fn + tp) if fn + tp > 0 else float("nan"),
        "high_fms_precision": precision,
        "high_fms_recall": recall,
        "high_fms_f1": f1,
    }


def classify_trend(diff: np.ndarray, eps: float) -> np.ndarray:
    """Classify FMS differences as down/stable/up using an FMS-point threshold."""
    cls = np.zeros_like(diff, dtype=np.int64)
    cls[diff > eps] = 1
    cls[diff < -eps] = -1
    return cls


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _binary_prf(true_positive: np.ndarray, pred_positive: np.ndarray) -> Dict[str, float]:
    tp = float(np.sum(true_positive & pred_positive))
    fp = float(np.sum(~true_positive & pred_positive))
    fn = float(np.sum(true_positive & ~pred_positive))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _class_f1(true_cls: np.ndarray, pred_cls: np.ndarray, cls_value: int) -> float:
    true_positive = true_cls == cls_value
    pred_positive = pred_cls == cls_value
    return _binary_prf(true_positive, pred_positive)["f1"]


def _trend_metrics_for_diffs(dt: np.ndarray, dp: np.ndarray, eps_fms: float, prefix: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    eps_key = f"eps{eps_fms:g}"
    if dt.size == 0:
        for name in (
            "trend_acc",
            "trend_macro_f1",
            "trend_up_f1",
            "trend_down_f1",
            "trend_stable_f1",
            "change_precision",
            "change_recall",
            "change_f1",
        ):
            metrics[f"{name}_{prefix}_{eps_key}"] = float("nan")
        return metrics

    true_cls = classify_trend(dt, eps_fms)
    pred_cls = classify_trend(dp, eps_fms)
    down_f1 = _class_f1(true_cls, pred_cls, -1)
    stable_f1 = _class_f1(true_cls, pred_cls, 0)
    up_f1 = _class_f1(true_cls, pred_cls, 1)
    change = _binary_prf(true_cls != 0, pred_cls != 0)
    metrics.update(
        {
            f"trend_acc_{prefix}_{eps_key}": float(np.mean(pred_cls == true_cls)),
            f"trend_macro_f1_{prefix}_{eps_key}": float(np.mean([down_f1, stable_f1, up_f1])),
            f"trend_up_f1_{prefix}_{eps_key}": up_f1,
            f"trend_down_f1_{prefix}_{eps_key}": down_f1,
            f"trend_stable_f1_{prefix}_{eps_key}": stable_f1,
            f"change_precision_{prefix}_{eps_key}": change["precision"],
            f"change_recall_{prefix}_{eps_key}": change["recall"],
            f"change_f1_{prefix}_{eps_key}": change["f1"],
        }
    )
    return metrics


def _series_from_arrays(pred: Any, target: Any, valid_mask: Any = None) -> List[Dict[str, Any]]:
    pred_arr = np.asarray(pred, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    if pred_arr.shape != target_arr.shape:
        raise ValueError(f"Prediction/target shapes differ: {pred_arr.shape} vs {target_arr.shape}")
    if pred_arr.ndim == 1:
        pred_arr = pred_arr[None, :]
        target_arr = target_arr[None, :]
    if pred_arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D prediction/target arrays, got {pred_arr.shape}")
    if valid_mask is None:
        mask_arr = np.ones_like(pred_arr, dtype=bool)
    else:
        mask_arr = np.asarray(valid_mask, dtype=bool)
        if mask_arr.ndim == 1:
            mask_arr = mask_arr[None, :]
        if mask_arr.shape != pred_arr.shape:
            raise ValueError(f"Mask shape {mask_arr.shape} does not match sequence shape {pred_arr.shape}")
    return [
        {
            "target_full": target_arr[i].tolist(),
            "prediction_full": pred_arr[i].tolist(),
            "mask": mask_arr[i].tolist(),
        }
        for i in range(pred_arr.shape[0])
    ]


def compute_sequence_analysis_metrics(
    series_or_pred: Sequence[Mapping[str, Any]] | Any,
    target: Any = None,
    valid_mask: Any = None,
    fms_scale: float = 20.0,
    values_are_normalized: bool = False,
    eps_fms: float = 0.5,
) -> Dict[str, Any]:
    """Compute sequence/trend metrics within each session.

    Metrics are defined on original FMS points. If normalized values are passed
    directly, set ``values_are_normalized=True`` to map them back with the fixed
    DenseFMS 0-20 scale before computing trend classes.
    """
    if target is None:
        series = list(series_or_pred)
    else:
        series = _series_from_arrays(series_or_pred, target, valid_mask)

    diffs_by_step: Dict[int, Dict[str, List[float]]] = {
        1: {"true": [], "pred": []},
        4: {"true": [], "pred": []},
        10: {"true": [], "pred": []},
    }
    sign_hits_raw_exact: List[bool] = []
    pearsons: List[float] = []

    for item in series:
        target = np.asarray(item["target_full"], dtype=np.float64)
        pred = np.asarray(item["prediction_full"], dtype=np.float64)
        mask = np.asarray(item["mask"], dtype=bool)
        if values_are_normalized:
            target = target * float(fms_scale)
            pred = pred * float(fms_scale)
        valid = mask & np.isfinite(target) & np.isfinite(pred)
        if valid.sum() >= 2:
            target_valid = target[valid]
            pred_valid = pred[valid]
            if np.std(target_valid) > 1e-12 and np.std(pred_valid) > 1e-12:
                pearsons.append(float(np.corrcoef(target_valid, pred_valid)[0, 1]))

        for step in diffs_by_step:
            if len(target) <= step:
                continue
            diff_mask = valid[step:] & valid[:-step]
            if not diff_mask.any():
                continue
            dt = (target[step:] - target[:-step])[diff_mask]
            dp = (pred[step:] - pred[:-step])[diff_mask]
            diffs_by_step[step]["true"].extend(dt.tolist())
            diffs_by_step[step]["pred"].extend(dp.tolist())
            if step == 1:
                sign_hits_raw_exact.extend((np.sign(dp) == np.sign(dt)).tolist())

    dt_1 = np.asarray(diffs_by_step[1]["true"], dtype=np.float64)
    dp_1 = np.asarray(diffs_by_step[1]["pred"], dtype=np.float64)
    if dt_1.size:
        deriv_err = np.abs(dp_1 - dt_1)
        true_cls_1 = classify_trend(dt_1, eps_fms)
        moving_mask = true_cls_1 != 0
        stationary_mask = true_cls_1 == 0
        derivative_mae_all = float(np.mean(deriv_err))
        derivative_mae_stationary = _safe_mean(deriv_err[stationary_mask])
        derivative_mae_moving = _safe_mean(deriv_err[moving_mask])
        raw_exact = float(np.mean(np.asarray(sign_hits_raw_exact, dtype=bool)))
        pred_cls_1 = classify_trend(dp_1, eps_fms)
        stationary_accuracy = _safe_mean((pred_cls_1[stationary_mask] == 0).astype(np.float64))
        moving_accuracy = _safe_mean((pred_cls_1[moving_mask] == true_cls_1[moving_mask]).astype(np.float64))
    else:
        derivative_mae_all = float("nan")
        derivative_mae_stationary = float("nan")
        derivative_mae_moving = float("nan")
        raw_exact = float("nan")
        stationary_accuracy = float("nan")
        moving_accuracy = float("nan")

    metrics: Dict[str, Any] = {
        "derivative_mae": derivative_mae_all,
        "derivative_mae_all": derivative_mae_all,
        f"derivative_mae_stationary_eps{eps_fms:g}": derivative_mae_stationary,
        f"derivative_mae_moving_eps{eps_fms:g}": derivative_mae_moving,
        "trend_sign_accuracy": raw_exact,
        "trend_sign_accuracy_raw_exact": raw_exact,
        f"trend_stationary_accuracy_1step_eps{eps_fms:g}": stationary_accuracy,
        f"trend_moving_accuracy_1step_eps{eps_fms:g}": moving_accuracy,
        "pearson_per_session_mean": float(np.mean(pearsons)) if pearsons else float("nan"),
        "pearson_per_session_std": float(np.std(pearsons)) if pearsons else float("nan"),
        "pearson_per_session_n": len(pearsons),
        "dtw_available": False,
        "dtw_mean": float("nan"),
        "dtw_std": float("nan"),
    }
    step_names = {1: "0p5s", 4: "2s", 10: "5s"}
    for step, label in step_names.items():
        dt = np.asarray(diffs_by_step[step]["true"], dtype=np.float64)
        dp = np.asarray(diffs_by_step[step]["pred"], dtype=np.float64)
        metrics.update(_trend_metrics_for_diffs(dt, dp, eps_fms, label))
        if step == 1:
            alias_metrics = _trend_metrics_for_diffs(dt, dp, eps_fms, "1step")
            metrics.update(alias_metrics)
    return metrics


def check_disjoint(name_a: str, groups_a: Iterable[str], name_b: str, groups_b: Iterable[str]) -> None:
    overlap = set(groups_a) & set(groups_b)
    if overlap:
        shown = sorted(overlap)[:10]
        raise AssertionError(f"{name_a}/{name_b} group overlap detected: {shown}")


def update_nested(config: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    node: MutableMapping[str, Any] = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, MutableMapping):
            raise ValueError(f"Cannot set {dotted_key}; {part} is not a mapping")
        node = child
    node[parts[-1]] = value


def normalize_time_config(config: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Populate canonical/legacy time config aliases in-place."""
    data = config.setdefault("data", {})
    sampling_interval = float(data.get("sampling_interval", data.get("default_sampling_interval", 0.5)))
    data["sampling_interval"] = sampling_interval
    data["default_sampling_interval"] = float(data.get("default_sampling_interval", sampling_interval))
    data["calibration_seconds"] = float(data.get("calibration_seconds", 30.0))
    data["horizon_seconds"] = float(data.get("horizon_seconds", 5.0))
    recent = float(data.get("recent_window_seconds", data.get("recent_seconds", 10.0)))
    data["recent_window_seconds"] = recent
    data["recent_seconds"] = float(data.get("recent_seconds", recent))
    max_points = data.get("max_session_points", 420)
    if max_points is None:
        max_points = 420
    data["max_session_points"] = min(int(max_points), 420)
    return config


def seconds_to_steps(
    seconds: float,
    sampling_interval: float,
    name: str = "seconds",
    warn: bool = True,
    allow_zero: bool = False,
) -> int:
    seconds_f = float(seconds)
    interval_f = float(sampling_interval)
    if interval_f <= 0:
        raise ValueError(f"sampling_interval must be positive, got {sampling_interval}")
    steps = int(round(seconds_f / interval_f))
    if steps == 0 and bool(allow_zero):
        return 0
    if steps <= 0:
        raise ValueError(f"{name}={seconds_f:g} and sampling_interval={interval_f:g} produce non-positive steps")
    actual = steps * interval_f
    tolerance = max(1e-6, interval_f * 0.01)
    if warn and abs(actual - seconds_f) > tolerance:
        warnings.warn(
            f"{name}={seconds_f:g}s is not close to a multiple of sampling_interval={interval_f:g}s; "
            f"using {steps} steps ({actual:g}s).",
            RuntimeWarning,
            stacklevel=2,
        )
    return steps


def int_steps(seconds: float, sampling_interval: float) -> int:
    return seconds_to_steps(seconds, sampling_interval, warn=False)


def human_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"
