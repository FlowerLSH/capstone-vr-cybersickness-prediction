"""Training entry point for DenseFMS online future forecasting."""

from __future__ import annotations

import argparse
import csv
import copy
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import (
    DenseFMSSession,
    DenseFMSSessionDataset,
    apply_saved_split,
    build_static_report,
    collate_sessions,
    current_sequence_times,
    fit_scalers,
    fit_static_scaler,
    future_sequence_targets,
    future_sequence_times,
    inspect_dataset,
    load_calibration_residual_features,
    load_raw_sessions,
    make_group_kfold_splits,
    normalize_head_channel_mode,
    normalize_gender_encoding,
    normalize_static_features,
    run_data_sanity_checks,
    split_sessions,
    static_feature_dim,
    transform_sessions,
    validate_static_availability,
)
from .losses import FutureSequenceLoss
from .model import build_model
from .utils import (
    compute_high_fms_metrics,
    compute_regression_metrics,
    compute_sequence_analysis_metrics,
    denormalize_fms,
    ensure_dir,
    human_float,
    int_steps,
    load_config,
    load_json,
    normalize_time_config,
    save_json,
    seconds_to_steps,
    set_seed,
    timestamp_for_run,
)


def prepare_data(
    data_dir: str | Path,
    config: Mapping[str, Any],
    limit_sessions: Optional[int] = None,
    saved_split: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    normalize_time_config(config)
    data_cfg = config["data"]
    use_static = bool(data_cfg.get("use_static", False))
    allow_missing_static = bool(data_cfg.get("allow_missing_static", False))
    static_features = normalize_static_features(data_cfg.get("static_features", ["age", "gender"]))
    gender_encoding = normalize_gender_encoding(data_cfg.get("gender_encoding", "category3"))
    head_channel_mode = normalize_head_channel_mode(data_cfg.get("head_channel_mode", "all"))
    report, mapping = inspect_dataset(data_dir, artifacts_dir=config.get("artifacts_dir", "artifacts"))
    raw_sessions, mapping, data_info = load_raw_sessions(
        data_dir,
        mapping=mapping,
        calibration_seconds=float(data_cfg["calibration_seconds"]),
        horizon_seconds=float(data_cfg["horizon_seconds"]),
        default_sampling_interval=float(data_cfg.get("sampling_interval", data_cfg.get("default_sampling_interval", 0.5))),
        max_session_points=data_cfg.get("max_session_points"),
    )
    if limit_sessions:
        raw_sessions = raw_sessions[: int(limit_sessions)]
        data_info["session_count"] = len(raw_sessions)
        data_info["participant_count"] = len({s.participant_id for s in raw_sessions if s.participant_id})
    data_info["head_channel_mode"] = head_channel_mode

    if saved_split is None:
        split_info = split_sessions(
            raw_sessions,
            seed=int(config["training"].get("seed", 42)),
            train_frac=float(data_cfg.get("train_frac", 0.70)),
            val_frac=float(data_cfg.get("val_frac", 0.15)),
            test_frac=float(data_cfg.get("test_frac", 0.15)),
        )
        split_raw = split_info["sessions"]
        split_info_public = {k: v for k, v in split_info.items() if k != "sessions"}
        if bool(data_cfg.get("group_kfold", False)):
            split_info_public["group_kfold"] = make_group_kfold_splits(
                raw_sessions,
                n_splits=int(data_cfg.get("group_kfold_splits", 5)),
                seed=int(config["training"].get("seed", 42)),
            )
    else:
        split_raw = apply_saved_split(raw_sessions, saved_split)
        split_info_public = dict(saved_split)

    train_sessions = split_raw["train"]
    if not train_sessions:
        raise RuntimeError("No training sessions available after split.")
    scalers = fit_scalers(train_sessions, data_info["calibration_steps"], data_info["horizon_steps"])
    static_report = None
    if use_static:
        all_split_sessions = split_raw["train"] + split_raw.get("val", []) + split_raw.get("test", [])
        validate_static_availability(
            all_split_sessions,
            static_features=static_features,
            allow_missing_static=allow_missing_static,
        )
        scalers["static"] = fit_static_scaler(
            train_sessions,
            static_features=static_features,
            allow_missing_static=allow_missing_static,
            gender_encoding=gender_encoding,
        )
        static_report = build_static_report(split_raw, scalers["static"], static_features=static_features)
        data_info["static_report"] = static_report
    residual_feature_map = None
    residual_feature_names = None
    residual_feature_info = None
    residual_paths = data_cfg.get("calibration_residual_features_path")
    residual_required = bool(data_cfg.get("require_calibration_residual_features", False))
    if residual_paths:
        residual_feature_map, residual_feature_names, residual_feature_info = load_calibration_residual_features(residual_paths)
        data_info["calibration_residual_feature_names"] = list(residual_feature_names)
        data_info["calibration_residual_feature_dim"] = int(len(residual_feature_names))
        data_info["calibration_residual_feature_artifacts"] = dict(residual_feature_info)
    elif residual_required:
        raise ValueError("data.require_calibration_residual_features=true requires data.calibration_residual_features_path.")
    no_test_eval = bool(config.get("evaluation", {}).get("no_test_eval", False))
    split_norm = {
        name: transform_sessions(
            sessions,
            scalers,
            use_static=use_static,
            static_features=static_features,
            allow_missing_static=allow_missing_static,
            head_channel_mode=head_channel_mode,
            calibration_residual_feature_map=residual_feature_map,
            calibration_residual_feature_names=residual_feature_names,
            require_calibration_residual_features=bool(residual_required and not (name == "test" and no_test_eval)),
        )
        for name, sessions in split_raw.items()
    }

    return {
        "report": report,
        "mapping": mapping,
        "data_info": data_info,
        "split_info": split_info_public,
        "splits": split_norm,
        "scalers": scalers,
        "static_report": static_report,
    }


def build_participant_balanced_session_weights(sessions: Sequence[DenseFMSSession]) -> torch.Tensor:
    """Return per-session weights whose participant-level totals are balanced."""
    keys: List[str] = []
    counts: Dict[str, int] = {}
    for idx, session in enumerate(sessions):
        key = str(session.participant_id or session.session_id or session.source_file or f"session_{idx}")
        keys.append(key)
        counts[key] = counts.get(key, 0) + 1
    if not keys:
        return torch.empty(0, dtype=torch.double)
    weights = np.asarray([1.0 / float(counts[key]) for key in keys], dtype=np.float64)
    weights = weights / max(float(weights.mean()), 1e-12)
    return torch.as_tensor(weights, dtype=torch.double)


def make_loaders(splits: Mapping[str, List[DenseFMSSession]], config: Mapping[str, Any]) -> Dict[str, DataLoader]:
    train_cfg = config["training"]
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 0))
    participant_balanced_sampling = bool(train_cfg.get("participant_balanced_sampling", False))
    loaders = {}
    for name, sessions in splits.items():
        if not sessions:
            continue
        sampler = None
        shuffle = name == "train"
        if name == "train" and participant_balanced_sampling:
            weights = build_participant_balanced_session_weights(sessions)
            sampler = WeightedRandomSampler(weights, num_samples=len(sessions), replacement=True)
            shuffle = False
        loaders[name] = DataLoader(
            DenseFMSSessionDataset(sessions),
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_sessions,
        )
    return loaders


def _lds_kernel_window(kernel: str = "gaussian", kernel_size: int = 5, sigma: float = 2.0) -> np.ndarray:
    kernel_name = str(kernel or "gaussian").lower()
    size = int(kernel_size)
    if size <= 0 or size % 2 == 0:
        raise ValueError("loss.lds_kernel_size must be a positive odd integer.")
    half = size // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    if kernel_name == "gaussian":
        scale = max(float(sigma), 1e-8)
        weights = np.exp(-(offsets**2) / (2.0 * scale * scale))
    elif kernel_name == "triangular":
        weights = (half + 1.0) - np.abs(offsets)
    elif kernel_name in {"laplace", "laplacian"}:
        scale = max(float(sigma), 1e-8)
        weights = np.exp(-np.abs(offsets) / scale)
    else:
        raise ValueError("loss.lds_kernel must be one of: gaussian, triangular, laplace.")
    weights = weights / max(float(weights.sum()), 1e-12)
    return weights.astype(np.float64)


def _lds_bin_indices(values: np.ndarray, min_value: float, max_value: float, bin_size: float, num_bins: int) -> np.ndarray:
    safe = np.asarray(values, dtype=np.float64)
    clipped = np.clip(safe, float(min_value), float(max_value))
    idx = np.floor((clipped - float(min_value)) / max(float(bin_size), 1e-8)).astype(np.int64)
    return np.clip(idx, 0, int(num_bins) - 1)


def build_lds_weight_info(
    train_sessions: Sequence[DenseFMSSession],
    prediction_start: int,
    fms_scaler: Mapping[str, float],
    *,
    min_value: float = 0.0,
    max_value: float = 20.0,
    bin_size: float = 1.0,
    kernel: str = "gaussian",
    kernel_size: int = 5,
    sigma: float = 2.0,
    gamma: float = 0.5,
    weight_min: float = 0.5,
    weight_max: float = 3.0,
) -> Dict[str, Any]:
    """Build Yang et al. LDS loss weights from train-split current-FMS targets only."""

    bin_width = float(bin_size)
    if bin_width <= 0:
        raise ValueError("loss.lds_bin_size must be positive.")
    lo = float(min_value)
    hi = float(max_value)
    if hi <= lo:
        raise ValueError("loss.lds_max must be greater than loss.lds_min.")
    weight_floor = float(weight_min)
    weight_ceil = float(weight_max)
    if weight_floor <= 0 or weight_ceil < weight_floor:
        raise ValueError("loss.lds_weight_min/max must satisfy 0 < min <= max.")
    num_bins = int(np.floor((hi - lo) / bin_width)) + 1
    raw_targets: List[np.ndarray] = []
    f_min = float(fms_scaler["min"])
    f_max = float(fms_scaler["max"])
    f_range = max(f_max - f_min, 1e-8)
    start = int(prediction_start)
    for sess in train_sessions:
        if sess.length <= start:
            continue
        if sess.fms_raw is not None:
            values = np.asarray(sess.fms_raw[start : sess.length], dtype=np.float64)
        else:
            values = np.asarray(sess.fms[start : sess.length], dtype=np.float64) * f_range + f_min
        values = values[np.isfinite(values)]
        if values.size:
            raw_targets.append(values)
    if not raw_targets:
        raise ValueError("LDS weighting requested, but no train current-FMS targets were available.")
    targets = np.concatenate(raw_targets, axis=0)
    target_bins = _lds_bin_indices(targets, lo, hi, bin_width, num_bins)
    empirical = np.bincount(target_bins, minlength=num_bins).astype(np.float64)
    kernel_window = _lds_kernel_window(kernel=kernel, kernel_size=kernel_size, sigma=sigma)
    effective = np.convolve(empirical, kernel_window, mode="same")
    effective = np.maximum(effective, 1e-8)
    reference_density = float(np.mean(effective[target_bins]))
    weights = np.power(reference_density / effective, float(gamma))
    weights = np.clip(weights, weight_floor, weight_ceil)
    sample_mean = float(np.mean(weights[target_bins]))
    if sample_mean > 1e-8:
        weights = weights / sample_mean
    weights = np.clip(weights, weight_floor, weight_ceil)
    sample_weights = weights[target_bins]
    return {
        "enabled": True,
        "prediction_start": int(start),
        "min": lo,
        "max": hi,
        "bin_size": bin_width,
        "num_bins": int(num_bins),
        "kernel": str(kernel),
        "kernel_size": int(kernel_size),
        "sigma": float(sigma),
        "gamma": float(gamma),
        "weight_min": weight_floor,
        "weight_max": weight_ceil,
        "kernel_window": kernel_window.astype(float).tolist(),
        "empirical_density": empirical.astype(float).tolist(),
        "effective_density": effective.astype(float).tolist(),
        "weights": weights.astype(float).tolist(),
        "train_target_count": int(targets.size),
        "train_nonempty_bins": int(np.count_nonzero(empirical)),
        "sample_weight_mean": float(np.mean(sample_weights)),
        "sample_weight_std": float(np.std(sample_weights)),
        "sample_weight_min": float(np.min(sample_weights)),
        "sample_weight_max": float(np.max(sample_weights)),
    }


def build_ordinal_class_count_info(
    train_sessions: Sequence[DenseFMSSession],
    prediction_start: int,
    fms_scaler: Mapping[str, float],
    ordinal_bins: Sequence[float],
) -> Dict[str, Any]:
    """Count train-split ordinal current-FMS labels for SLACE proximity."""

    bins = np.asarray([float(v) for v in ordinal_bins], dtype=np.float64)
    if bins.ndim != 1 or bins.size == 0:
        raise ValueError("ordinal_bins must contain at least one bin for SLACE.")
    if np.any(np.diff(bins) <= 0):
        raise ValueError("ordinal_bins must be strictly increasing for SLACE.")
    raw_targets: List[np.ndarray] = []
    f_min = float(fms_scaler["min"])
    f_max = float(fms_scaler["max"])
    f_range = max(f_max - f_min, 1e-8)
    start = int(prediction_start)
    for sess in train_sessions:
        if sess.length <= start:
            continue
        if sess.fms_raw is not None:
            values = np.asarray(sess.fms_raw[start : sess.length], dtype=np.float64)
        else:
            values = np.asarray(sess.fms[start : sess.length], dtype=np.float64) * f_range + f_min
        values = values[np.isfinite(values)]
        if values.size:
            raw_targets.append(values)
    if not raw_targets:
        raise ValueError("SLACE requested, but no train current-FMS targets were available.")
    targets = np.concatenate(raw_targets, axis=0)
    labels = np.argmin(np.abs(targets.reshape(-1, 1) - bins.reshape(1, -1)), axis=1)
    counts = np.bincount(labels, minlength=int(bins.size)).astype(np.float64)
    return {
        "enabled": True,
        "prediction_start": int(start),
        "ordinal_bins": bins.astype(float).tolist(),
        "class_counts": counts.astype(float).tolist(),
        "train_target_count": int(targets.size),
        "train_nonempty_classes": int(np.count_nonzero(counts)),
        "min_count": float(np.min(counts)) if counts.size else 0.0,
        "max_count": float(np.max(counts)) if counts.size else 0.0,
    }


def _tensor_scalar_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return int(default)
        return int(value.detach().cpu().flatten()[0].item())
    return int(value)


def _future_targets_for_horizons(
    fms: torch.Tensor,
    lengths: torch.Tensor,
    prediction_start: int,
    horizon_steps_list: Sequence[int],
    max_pred_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if lengths.device != fms.device:
        lengths = lengths.to(fms.device)
    pred_steps = int(max_pred_steps)
    horizons = torch.tensor([int(v) for v in horizon_steps_list], dtype=torch.long, device=fms.device)
    if pred_steps == 0 or horizons.numel() == 0:
        return (
            fms.new_zeros((fms.shape[0], 0, int(horizons.numel()))),
            torch.zeros((fms.shape[0], 0, int(horizons.numel())), dtype=torch.bool, device=fms.device),
        )
    positions = int(prediction_start) + torch.arange(pred_steps, dtype=torch.long, device=fms.device)
    target_positions = positions.view(1, -1, 1) + horizons.view(1, 1, -1)
    safe_idx = target_positions.clamp_max(fms.shape[1] - 1).expand(fms.shape[0], -1, -1)
    source = fms.unsqueeze(-1).expand(-1, -1, int(horizons.numel()))
    target = torch.gather(source, 1, safe_idx)
    mask = target_positions < lengths.view(-1, 1, 1)
    mask = mask & torch.isfinite(target)
    return target, mask


def compute_session_summary_targets(
    fms: torch.Tensor,
    lengths: torch.Tensor,
    calibration_steps: int,
    high_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return post-calibration session summaries used only as auxiliary targets."""
    if lengths.device != fms.device:
        lengths = lengths.to(fms.device)
    total_steps = int(fms.shape[1])
    steps = torch.arange(total_steps, dtype=torch.long, device=fms.device).view(1, -1)
    valid = (steps >= int(calibration_steps)) & (steps < lengths.view(-1, 1)) & torch.isfinite(fms)
    valid_f = valid.to(fms.dtype)
    denom = valid_f.sum(dim=1).clamp_min(1.0)
    safe = torch.where(valid, fms, torch.zeros_like(fms))
    post_mean = safe.sum(dim=1) / denom
    post_max = torch.where(valid, fms, torch.full_like(fms, -float("inf"))).amax(dim=1)
    end_idx = (lengths - 1).clamp_min(0).clamp_max(total_steps - 1)
    post_end = fms.gather(1, end_idx.view(-1, 1)).squeeze(1)
    post_high_frac = ((safe >= float(high_threshold)) & valid).to(fms.dtype).sum(dim=1) / denom
    summary = torch.stack([post_mean, post_max, post_end, post_high_frac], dim=-1)
    session_valid = valid.any(dim=1) & torch.isfinite(summary).all(dim=-1)
    summary = torch.where(torch.isfinite(summary), summary, torch.zeros_like(summary)).clamp(0.0, 1.0)
    return summary, session_valid.unsqueeze(-1).expand_as(summary)


def compute_loss(
    outputs: Mapping[str, torch.Tensor],
    fms: torch.Tensor,
    lengths: torch.Tensor,
    calibration_steps: int,
    horizon_steps: int,
    loss_fn: FutureSequenceLoss,
    dual_aux_alpha: float = 0.0,
    dual_aux_beta: float = 0.0,
    change_aux_weight: float = 0.0,
    change_aux_threshold: float = 0.1,
    current_aux_weight: float = 0.0,
    current_delta_aux_weight: float = 0.0,
    session_aux_weight: float = 0.0,
    session_aux_loss_type: str = "smooth_l1",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_future = outputs["future"]
    prediction_start = _tensor_scalar_int(outputs.get("prediction_start"), calibration_steps)
    if pred_future.ndim == 3:
        horizon_values = outputs.get("horizon_steps_list")
        horizon_steps_list = (
            [int(v) for v in horizon_values.detach().cpu().tolist()]
            if isinstance(horizon_values, torch.Tensor)
            else [int(horizon_steps)]
        )
        target_future, target_mask = _future_targets_for_horizons(
            fms,
            lengths,
            prediction_start,
            horizon_steps_list,
            max_pred_steps=pred_future.shape[1],
        )
    else:
        target_future, target_mask = future_sequence_targets(
            fms,
            lengths,
            calibration_steps,
            horizon_steps,
            max_pred_steps=pred_future.shape[1],
            prediction_start_steps=prediction_start,
        )
    valid_mask = outputs["mask"].to(target_mask.device) & target_mask
    loss, parts = loss_fn(pred_future, target_future, valid_mask)
    aux_level = pred_future.new_tensor(0.0)
    aux_delta = pred_future.new_tensor(0.0)
    valid = valid_mask.bool() & torch.isfinite(target_future)

    def _aux_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        finite = mask.bool() & torch.isfinite(pred) & torch.isfinite(target)
        safe_pred = torch.where(finite, pred, torch.zeros_like(pred))
        safe_target = torch.where(finite, target, torch.zeros_like(target))
        raw = torch.nn.functional.smooth_l1_loss(safe_pred, safe_target, reduction="none")
        return (raw * finite.to(raw.dtype)).sum() / finite.to(raw.dtype).sum().clamp_min(1.0)

    if float(dual_aux_alpha) > 0 and "future_level" in outputs:
        aux_level = _aux_loss(outputs["future_level"].to(target_future.device), target_future, valid)
        loss = loss + float(dual_aux_alpha) * aux_level
    if float(dual_aux_beta) > 0 and "future_delta_pred" in outputs and "future_delta_base" in outputs:
        base = outputs["future_delta_base"].to(target_future.device)
        if base.ndim == 2 and target_future.ndim == 3:
            base = base.unsqueeze(-1).expand_as(target_future)
        target_delta = target_future - base
        aux_delta = _aux_loss(outputs["future_delta_pred"].to(target_future.device), target_delta, valid)
        loss = loss + float(dual_aux_beta) * aux_delta
    aux_change = pred_future.new_tensor(0.0)
    change_points = 0
    if float(change_aux_weight) > 0 and "future_change_logits" in outputs and "anchor_fms" in outputs:
        logits = outputs["future_change_logits"].to(target_future.device)
        base = outputs["anchor_fms"].to(target_future.device)
        if base.ndim == 2 and target_future.ndim == 3:
            base = base.unsqueeze(-1).expand_as(target_future)
        target_delta = target_future - base
        labels = torch.ones_like(target_delta, dtype=torch.long)
        threshold = float(change_aux_threshold)
        labels = torch.where(target_delta <= -threshold, torch.zeros_like(labels), labels)
        labels = torch.where(target_delta >= threshold, torch.full_like(labels, 2), labels)
        finite = valid.bool() & torch.isfinite(target_delta)
        if logits.ndim == 3:
            logits_flat = logits.reshape(-1, logits.shape[-1])
        elif logits.ndim == 4:
            logits_flat = logits.reshape(-1, logits.shape[-1])
        else:
            raise ValueError(f"future_change_logits must be [B,P,3] or [B,P,H,3], got {logits.shape}")
        labels_flat = labels.reshape(-1)
        finite_flat = finite.reshape(-1)
        raw_change = torch.nn.functional.cross_entropy(logits_flat, labels_flat, reduction="none")
        aux_change = (raw_change * finite_flat.to(raw_change.dtype)).sum() / finite_flat.to(raw_change.dtype).sum().clamp_min(1.0)
        change_points = int(finite_flat.sum().detach().cpu())
        loss = loss + float(change_aux_weight) * aux_change
    aux_current = pred_future.new_tensor(0.0)
    aux_current_delta = pred_future.new_tensor(0.0)
    aux_session = pred_future.new_tensor(0.0)
    current_points = 0
    current_delta_points = 0
    if float(current_aux_weight) > 0 and "current" in outputs:
        current_pred = outputs["current"].to(target_future.device)
        if current_pred.ndim == 3 and current_pred.shape[-1] == 1:
            current_pred = current_pred.squeeze(-1)
        if current_pred.ndim != 2:
            raise ValueError(f"current auxiliary prediction must be [B,P], got {current_pred.shape}")
        pred_steps = min(int(current_pred.shape[1]), int(pred_future.shape[1]))
        positions = int(prediction_start) + torch.arange(pred_steps, device=fms.device, dtype=torch.long)
        safe_idx = positions.clamp_max(fms.shape[1] - 1)
        target_current = fms.index_select(1, safe_idx).to(target_future.device)
        current_mask = positions.view(1, -1) < lengths.to(fms.device).view(-1, 1)
        current_mask = current_mask.to(target_future.device) & torch.isfinite(target_current)
        aux_current = _aux_loss(current_pred[:, :pred_steps], target_current, current_mask)
        current_points = int(current_mask.sum().detach().cpu())
        loss = loss + float(current_aux_weight) * aux_current
        if float(current_delta_aux_weight) > 0 and pred_steps > 1:
            pred_delta = current_pred[:, 1:pred_steps] - current_pred[:, : pred_steps - 1]
            target_delta = target_current[:, 1:pred_steps] - target_current[:, : pred_steps - 1]
            delta_mask = current_mask[:, 1:pred_steps] & current_mask[:, : pred_steps - 1]
            aux_current_delta = _aux_loss(pred_delta, target_delta, delta_mask)
            current_delta_points = int(delta_mask.sum().detach().cpu())
            loss = loss + float(current_delta_aux_weight) * aux_current_delta
    session_points = 0
    if float(session_aux_weight) > 0:
        if "session_summary" not in outputs:
            raise ValueError("session_aux_weight > 0 requires model outputs['session_summary']; use --session_context_mode summary.")
        if session_aux_loss_type not in {"smooth_l1", "mse", "l1", "mae"}:
            raise ValueError("session_aux_loss_type must be one of: smooth_l1, mse, l1, mae.")
        session_pred = outputs["session_summary"].to(target_future.device)
        if session_pred.ndim != 2:
            raise ValueError(f"session_summary prediction must be [B,D], got {session_pred.shape}")
        session_target, session_mask = compute_session_summary_targets(
            fms.to(target_future.device),
            lengths.to(target_future.device),
            calibration_steps,
        )
        if session_target.shape != session_pred.shape:
            raise ValueError(f"session_summary target shape {session_target.shape} does not match prediction {session_pred.shape}.")
        session_mask = session_mask & torch.isfinite(session_pred) & torch.isfinite(session_target)
        safe_pred = torch.where(session_mask, session_pred, torch.zeros_like(session_pred))
        safe_target = torch.where(session_mask, session_target, torch.zeros_like(session_target))
        if session_aux_loss_type == "smooth_l1":
            raw_session = torch.nn.functional.smooth_l1_loss(safe_pred, safe_target, reduction="none")
        else:
            session_err = safe_pred - safe_target
            raw_session = session_err.abs() if session_aux_loss_type in {"l1", "mae"} else session_err.square()
        session_mask_f = session_mask.to(raw_session.dtype)
        aux_session = (raw_session * session_mask_f).sum() / session_mask_f.sum().clamp_min(1.0)
        session_points = int(session_mask.sum().detach().cpu())
        loss = loss + float(session_aux_weight) * aux_session
    parts["loss_total"] = float(loss.detach().cpu())
    parts["loss_dual_level"] = float(aux_level.detach().cpu())
    parts["loss_dual_delta"] = float(aux_delta.detach().cpu())
    parts["loss_change_aux"] = float(aux_change.detach().cpu())
    parts["loss_current_aux"] = float(aux_current.detach().cpu())
    parts["loss_current_delta_aux"] = float(aux_current_delta.detach().cpu())
    parts["loss_session_aux"] = float(aux_session.detach().cpu())
    parts["dual_aux_alpha"] = float(dual_aux_alpha)
    parts["dual_aux_beta"] = float(dual_aux_beta)
    parts["change_aux_weight"] = float(change_aux_weight)
    parts["change_aux_threshold"] = float(change_aux_threshold)
    parts["change_aux_points"] = int(change_points)
    parts["current_aux_weight"] = float(current_aux_weight)
    parts["current_aux_points"] = int(current_points)
    parts["current_delta_aux_weight"] = float(current_delta_aux_weight)
    parts["current_delta_aux_points"] = int(current_delta_points)
    parts["session_aux_weight"] = float(session_aux_weight)
    parts["session_aux_points"] = int(session_points)
    return loss, parts


def compute_online_current_risk_targets(
    fms: torch.Tensor,
    lengths: torch.Tensor,
    prediction_start: int,
    pred_steps: int,
    rise_horizon_steps: Sequence[int],
    rise_thresholds_normalized: Sequence[float],
    ordinal_bins_normalized: Sequence[float] | torch.Tensor,
    fall_horizon_steps: Optional[Sequence[int]] = None,
    fall_thresholds_normalized: Optional[Sequence[float]] = None,
    high_risk_horizon_steps: Optional[Sequence[int]] = None,
    high_risk_thresholds_normalized: Optional[Sequence[float]] = None,
    high_risk_label_mode: str = "future_any",
    high_risk_onset_past_steps: int = 0,
    future_horizon_steps: Optional[Sequence[int]] = None,
    event_delta_threshold_normalized: float = 1.0 / 20.0,
) -> Dict[str, torch.Tensor]:
    """Build leakage-safe current-FMS and future rapid-rise/drop targets."""
    device = fms.device
    if lengths.device != device:
        lengths = lengths.to(device)
    positions = int(prediction_start) + torch.arange(int(pred_steps), dtype=torch.long, device=device)
    fall_horizon_steps = [int(v) for v in (fall_horizon_steps or rise_horizon_steps)]
    fall_thresholds_normalized = [
        float(v)
        for v in (
            fall_thresholds_normalized
            if fall_thresholds_normalized is not None
            else rise_thresholds_normalized
        )
    ]
    if len(fall_thresholds_normalized) != len(fall_horizon_steps):
        raise ValueError("fall_thresholds_normalized must have the same length as fall_horizon_steps.")
    high_risk_horizon_steps = [int(v) for v in (high_risk_horizon_steps or [])]
    high_risk_thresholds_normalized = [float(v) for v in (high_risk_thresholds_normalized or [])]
    high_risk_label_mode = str(high_risk_label_mode or "future_any").lower()
    if high_risk_label_mode not in {"future_any", "current_below", "onset", "current_or_future"}:
        raise ValueError(
            "high_risk_label_mode must be one of: future_any, current_below, onset, current_or_future."
        )
    high_risk_onset_past_steps = int(max(high_risk_onset_past_steps, 0))
    future_horizon_steps = [int(v) for v in (future_horizon_steps or [])]
    if int(pred_steps) == 0:
        h_count = len(rise_horizon_steps)
        fall_h_count = len(fall_horizon_steps)
        high_h_count = len(high_risk_horizon_steps)
        high_t_count = len(high_risk_thresholds_normalized)
        f_count = len(future_horizon_steps)
        return {
            "current": fms.new_zeros((fms.shape[0], 0)),
            "current_mask": torch.zeros((fms.shape[0], 0), dtype=torch.bool, device=device),
            "rise_labels": fms.new_zeros((fms.shape[0], 0, h_count)),
            "rise_mask": torch.zeros((fms.shape[0], 0, h_count), dtype=torch.bool, device=device),
            "fall_labels": fms.new_zeros((fms.shape[0], 0, fall_h_count)),
            "fall_mask": torch.zeros((fms.shape[0], 0, fall_h_count), dtype=torch.bool, device=device),
            "high_risk_labels": fms.new_zeros((fms.shape[0], 0, high_h_count, high_t_count)),
            "high_risk_mask": torch.zeros((fms.shape[0], 0, high_h_count, high_t_count), dtype=torch.bool, device=device),
            "future": fms.new_zeros((fms.shape[0], 0, f_count)),
            "future_mask": torch.zeros((fms.shape[0], 0, f_count), dtype=torch.bool, device=device),
            "future_delta": fms.new_zeros((fms.shape[0], 0, f_count)),
            "event_labels": torch.zeros((fms.shape[0], 0, f_count), dtype=torch.long, device=device),
            "ordinal_labels": torch.zeros((fms.shape[0], 0), dtype=torch.long, device=device),
            "positions": positions,
        }
    safe_idx = positions.clamp_max(fms.shape[1] - 1)
    current = fms.index_select(1, safe_idx)
    current_mask = (positions.view(1, -1) < lengths.view(-1, 1)) & torch.isfinite(current)
    rise_labels: List[torch.Tensor] = []
    rise_masks: List[torch.Tensor] = []
    for horizon_steps, threshold in zip(rise_horizon_steps, rise_thresholds_normalized):
        horizon_steps = int(horizon_steps)
        future_offsets = torch.arange(1, horizon_steps + 1, dtype=torch.long, device=device)
        future_idx = positions.view(1, -1, 1) + future_offsets.view(1, 1, -1)
        safe_future_idx = future_idx.clamp_max(fms.shape[1] - 1).expand(fms.shape[0], -1, -1)
        future_values = torch.gather(fms.unsqueeze(-1).expand(-1, -1, horizon_steps), 1, safe_future_idx)
        full_future_mask = (positions.view(1, -1) + horizon_steps) < lengths.view(-1, 1)
        valid_future = future_idx < lengths.view(-1, 1, 1)
        future_values = torch.where(valid_future & torch.isfinite(future_values), future_values, fms.new_full(future_values.shape, -float("inf")))
        future_max = future_values.amax(dim=-1)
        delta = future_max - current
        labels = (delta >= float(threshold)).to(fms.dtype)
        mask = current_mask & full_future_mask & torch.isfinite(delta)
        rise_labels.append(torch.where(mask, labels, torch.zeros_like(labels)))
        rise_masks.append(mask)
    rise_label_tensor = torch.stack(rise_labels, dim=-1) if rise_labels else fms.new_zeros((fms.shape[0], int(pred_steps), 0))
    rise_mask_tensor = (
        torch.stack(rise_masks, dim=-1)
        if rise_masks
        else torch.zeros((fms.shape[0], int(pred_steps), 0), dtype=torch.bool, device=device)
    )
    fall_labels: List[torch.Tensor] = []
    fall_masks: List[torch.Tensor] = []
    for horizon_steps, threshold in zip(fall_horizon_steps, fall_thresholds_normalized):
        horizon_steps = int(horizon_steps)
        future_offsets = torch.arange(1, horizon_steps + 1, dtype=torch.long, device=device)
        future_idx = positions.view(1, -1, 1) + future_offsets.view(1, 1, -1)
        safe_future_idx = future_idx.clamp_max(fms.shape[1] - 1).expand(fms.shape[0], -1, -1)
        future_values = torch.gather(fms.unsqueeze(-1).expand(-1, -1, horizon_steps), 1, safe_future_idx)
        full_future_mask = (positions.view(1, -1) + horizon_steps) < lengths.view(-1, 1)
        valid_future = future_idx < lengths.view(-1, 1, 1)
        future_values = torch.where(valid_future & torch.isfinite(future_values), future_values, fms.new_full(future_values.shape, float("inf")))
        future_min = future_values.amin(dim=-1)
        drop = current - future_min
        labels = (drop >= float(threshold)).to(fms.dtype)
        mask = current_mask & full_future_mask & torch.isfinite(drop)
        fall_labels.append(torch.where(mask, labels, torch.zeros_like(labels)))
        fall_masks.append(mask)
    fall_label_tensor = (
        torch.stack(fall_labels, dim=-1)
        if fall_labels
        else fms.new_zeros((fms.shape[0], int(pred_steps), 0))
    )
    fall_mask_tensor = (
        torch.stack(fall_masks, dim=-1)
        if fall_masks
        else torch.zeros((fms.shape[0], int(pred_steps), 0), dtype=torch.bool, device=device)
    )
    high_risk_labels: List[torch.Tensor] = []
    high_risk_masks: List[torch.Tensor] = []
    for horizon_steps in high_risk_horizon_steps:
        horizon_steps = int(horizon_steps)
        future_offsets = torch.arange(1, horizon_steps + 1, dtype=torch.long, device=device)
        future_idx = positions.view(1, -1, 1) + future_offsets.view(1, 1, -1)
        safe_future_idx = future_idx.clamp_max(fms.shape[1] - 1).expand(fms.shape[0], -1, -1)
        future_values = torch.gather(fms.unsqueeze(-1).expand(-1, -1, horizon_steps), 1, safe_future_idx)
        full_future_mask = (positions.view(1, -1) + horizon_steps) < lengths.view(-1, 1)
        valid_future = future_idx < lengths.view(-1, 1, 1)
        future_values = torch.where(valid_future & torch.isfinite(future_values), future_values, fms.new_full(future_values.shape, -float("inf")))
        future_max = future_values.amax(dim=-1)
        threshold_labels: List[torch.Tensor] = []
        threshold_masks: List[torch.Tensor] = []
        for threshold in high_risk_thresholds_normalized:
            future_high = future_max >= float(threshold)
            if high_risk_label_mode == "current_or_future":
                current_high = torch.isfinite(current) & (current >= float(threshold))
                labels = (future_high | current_high).to(fms.dtype)
            else:
                labels = future_high.to(fms.dtype)
            mask = current_mask & full_future_mask & torch.isfinite(future_max)
            if high_risk_label_mode == "current_below":
                current_below = torch.isfinite(current) & (current < float(threshold))
                mask = mask & current_below
            elif high_risk_label_mode == "onset":
                past_offsets = torch.arange(
                    -high_risk_onset_past_steps,
                    1,
                    dtype=torch.long,
                    device=device,
                )
                past_idx = positions.view(1, -1, 1) + past_offsets.view(1, 1, -1)
                safe_past_idx = past_idx.clamp(0, fms.shape[1] - 1).expand(fms.shape[0], -1, -1)
                past_values = torch.gather(
                    fms.unsqueeze(-1).expand(-1, -1, past_offsets.numel()),
                    1,
                    safe_past_idx,
                )
                valid_past = (past_idx >= 0) & (past_idx < lengths.view(-1, 1, 1))
                finite_past = valid_past & torch.isfinite(past_values)
                past_values = torch.where(finite_past, past_values, fms.new_full(past_values.shape, float("inf")))
                full_past_mask = finite_past.all(dim=-1)
                past_max = past_values.amax(dim=-1)
                onset_candidate = full_past_mask & torch.isfinite(past_max) & (past_max < float(threshold))
                mask = mask & onset_candidate
            threshold_labels.append(torch.where(mask, labels, torch.zeros_like(labels)))
            threshold_masks.append(mask)
        if threshold_labels:
            high_risk_labels.append(torch.stack(threshold_labels, dim=-1))
            high_risk_masks.append(torch.stack(threshold_masks, dim=-1))
    high_risk_label_tensor = (
        torch.stack(high_risk_labels, dim=2)
        if high_risk_labels
        else fms.new_zeros((fms.shape[0], int(pred_steps), 0, len(high_risk_thresholds_normalized)))
    )
    high_risk_mask_tensor = (
        torch.stack(high_risk_masks, dim=2)
        if high_risk_masks
        else torch.zeros(
            (fms.shape[0], int(pred_steps), 0, len(high_risk_thresholds_normalized)),
            dtype=torch.bool,
            device=device,
        )
    )
    future_values: List[torch.Tensor] = []
    future_masks: List[torch.Tensor] = []
    future_deltas: List[torch.Tensor] = []
    event_labels: List[torch.Tensor] = []
    event_threshold = float(event_delta_threshold_normalized)
    for horizon_steps in future_horizon_steps:
        horizon_steps = int(horizon_steps)
        future_idx = positions + horizon_steps
        safe_future_idx = future_idx.clamp_max(fms.shape[1] - 1)
        future = fms.index_select(1, safe_future_idx)
        mask = (future_idx.view(1, -1) < lengths.view(-1, 1)) & current_mask & torch.isfinite(future)
        delta = future - current
        labels = torch.ones(delta.shape, dtype=torch.long, device=device)
        labels = torch.where(delta <= -event_threshold, torch.zeros_like(labels), labels)
        labels = torch.where(delta >= event_threshold, torch.full_like(labels, 2), labels)
        future_values.append(torch.where(mask, future, torch.zeros_like(future)))
        future_masks.append(mask)
        future_deltas.append(torch.where(mask, delta, torch.zeros_like(delta)))
        event_labels.append(torch.where(mask, labels, torch.zeros_like(labels)))
    future_tensor = (
        torch.stack(future_values, dim=-1)
        if future_values
        else fms.new_zeros((fms.shape[0], int(pred_steps), 0))
    )
    future_mask_tensor = (
        torch.stack(future_masks, dim=-1)
        if future_masks
        else torch.zeros((fms.shape[0], int(pred_steps), 0), dtype=torch.bool, device=device)
    )
    future_delta_tensor = (
        torch.stack(future_deltas, dim=-1)
        if future_deltas
        else fms.new_zeros((fms.shape[0], int(pred_steps), 0))
    )
    event_label_tensor = (
        torch.stack(event_labels, dim=-1)
        if event_labels
        else torch.zeros((fms.shape[0], int(pred_steps), 0), dtype=torch.long, device=device)
    )
    if isinstance(ordinal_bins_normalized, torch.Tensor):
        bins = ordinal_bins_normalized.to(device=device, dtype=fms.dtype)
    else:
        bins = torch.tensor([float(v) for v in ordinal_bins_normalized], dtype=fms.dtype, device=device)
    ordinal_labels = torch.argmin((current.unsqueeze(-1) - bins.view(1, 1, -1)).abs(), dim=-1)
    ordinal_labels = torch.where(current_mask, ordinal_labels, torch.zeros_like(ordinal_labels))
    return {
        "current": current,
        "current_mask": current_mask,
        "rise_labels": rise_label_tensor,
        "rise_mask": rise_mask_tensor,
        "fall_labels": fall_label_tensor,
        "fall_mask": fall_mask_tensor,
        "high_risk_labels": high_risk_label_tensor,
        "high_risk_mask": high_risk_mask_tensor,
        "future": future_tensor,
        "future_mask": future_mask_tensor,
        "future_delta": future_delta_tensor,
        "event_labels": event_label_tensor,
        "ordinal_labels": ordinal_labels,
        "positions": positions,
    }


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_regression_loss(pred, target, mask, loss_type="smooth_l1")


def _masked_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "smooth_l1",
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss_name = str(loss_type).lower()
    if loss_name not in {"smooth_l1", "mse", "l1", "mae"}:
        raise ValueError("loss_type must be one of: smooth_l1, mse, l1, mae.")
    finite = mask.bool() & torch.isfinite(pred) & torch.isfinite(target)
    safe_pred = torch.where(finite, pred, torch.zeros_like(pred))
    safe_target = torch.where(finite, target, torch.zeros_like(target))
    if loss_name == "smooth_l1":
        raw = torch.nn.functional.smooth_l1_loss(safe_pred, safe_target, reduction="none")
    else:
        err = safe_pred - safe_target
        raw = err.abs() if loss_name in {"l1", "mae"} else err.square()
    valid_f = finite.to(raw.dtype)
    if sample_weight is not None:
        weights = torch.where(finite, sample_weight.to(raw.device, dtype=raw.dtype), torch.zeros_like(raw))
        weighted_valid = valid_f * weights
        return (raw * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)
    return (raw * valid_f).sum() / valid_f.sum().clamp_min(1.0)


def _masked_trajectory_shape_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta_steps: Optional[Sequence[int]] = None,
    loss_type: str = "mae",
    delta_weight: float = 1.0,
    centered_weight: float = 0.5,
    range_weight: float = 0.2,
    min_points: int = 4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """DILATE-lite trajectory loss over causal current-FMS output sequences.

    This is not full soft-DTW DILATE. It keeps the level MAE objective separate
    and adds lightweight shape/timing surrogates that are stable for long
    per-session current-FMS trajectories.
    """
    if pred.ndim != 2 or target.ndim != 2 or mask.ndim != 2:
        raise ValueError("trajectory shape loss expects pred, target, and mask with shape [B,P].")
    finite = mask.bool() & torch.isfinite(pred) & torch.isfinite(target)
    counts = finite.sum(dim=1)
    row_mask = counts >= int(max(min_points, 1))
    zero = pred.new_tensor(0.0)
    delta_steps = [int(v) for v in (delta_steps or []) if int(v) > 0 and int(v) < int(pred.shape[1])]

    loss_centered = zero
    centered_points = 0
    if float(centered_weight) > 0 and row_mask.any():
        finite_f = finite.to(pred.dtype)
        denom = finite_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        pred_mean = (torch.where(finite, pred, torch.zeros_like(pred)) * finite_f).sum(dim=1, keepdim=True) / denom
        target_mean = (torch.where(finite, target, torch.zeros_like(target)) * finite_f).sum(dim=1, keepdim=True) / denom
        centered_pred = pred - pred_mean
        centered_target = target - target_mean
        centered_mask = finite & row_mask.view(-1, 1)
        loss_centered = _masked_regression_loss(centered_pred, centered_target, centered_mask, loss_type=loss_type)
        centered_points = int(centered_mask.sum().detach().cpu())

    loss_range = zero
    range_points = 0
    if float(range_weight) > 0 and row_mask.any():
        pos_inf = torch.full_like(pred, float("inf"))
        neg_inf = torch.full_like(pred, -float("inf"))
        pred_min = torch.where(finite, pred, pos_inf).amin(dim=1)
        pred_max = torch.where(finite, pred, neg_inf).amax(dim=1)
        target_min = torch.where(finite, target, pos_inf).amin(dim=1)
        target_max = torch.where(finite, target, neg_inf).amax(dim=1)
        pred_range = pred_max - pred_min
        target_range = target_max - target_min
        range_mask = row_mask & torch.isfinite(pred_range) & torch.isfinite(target_range)
        loss_range = _masked_regression_loss(pred_range, target_range, range_mask, loss_type=loss_type)
        range_points = int(range_mask.sum().detach().cpu())

    loss_delta = zero
    delta_points = 0
    if float(delta_weight) > 0 and delta_steps:
        weighted_delta = zero
        weighted_points = 0.0
        for step in delta_steps:
            delta_mask = finite[:, step:] & finite[:, :-step] & row_mask.view(-1, 1)
            points = int(delta_mask.sum().detach().cpu())
            if points <= 0:
                continue
            pred_delta = pred[:, step:] - pred[:, :-step]
            target_delta = target[:, step:] - target[:, :-step]
            step_loss = _masked_regression_loss(pred_delta, target_delta, delta_mask, loss_type=loss_type)
            weighted_delta = weighted_delta + step_loss * float(points)
            weighted_points += float(points)
            delta_points += points
        if weighted_points > 0:
            loss_delta = weighted_delta / float(weighted_points)

    loss = (
        float(delta_weight) * loss_delta
        + float(centered_weight) * loss_centered
        + float(range_weight) * loss_range
    )
    parts = {
        "loss_trajectory": float(loss.detach().cpu()),
        "loss_trajectory_delta": float(loss_delta.detach().cpu()),
        "loss_trajectory_centered": float(loss_centered.detach().cpu()),
        "loss_trajectory_range": float(loss_range.detach().cpu()),
        "trajectory_delta_points": int(delta_points),
        "trajectory_centered_points": int(centered_points),
        "trajectory_range_points": int(range_points),
        "trajectory_delta_step_count": int(len(delta_steps)),
        "trajectory_delta_weight": float(delta_weight),
        "trajectory_centered_weight": float(centered_weight),
        "trajectory_range_weight": float(range_weight),
        "trajectory_min_points": int(min_points),
    }
    return loss, parts


def _ordinal_soft_targets(
    labels: torch.Tensor,
    num_classes: int,
    *,
    sigma: float = 1.0,
    kernel: str = "gaussian",
) -> torch.Tensor:
    classes = torch.arange(num_classes, device=labels.device, dtype=torch.float32).view(1, 1, -1)
    centers = labels.to(torch.float32).unsqueeze(-1)
    distance = torch.abs(classes - centers)
    sigma_f = max(float(sigma), 1e-6)
    kernel_name = str(kernel or "gaussian").lower()
    if kernel_name in {"laplace", "laplacian", "exponential"}:
        weights = torch.exp(-distance / sigma_f)
    elif kernel_name in {"triangular", "linear"}:
        weights = torch.clamp(1.0 - distance / sigma_f, min=0.0)
    else:
        weights = torch.exp(-0.5 * torch.square(distance / sigma_f))
    hard = torch.nn.functional.one_hot(labels.clamp(min=0, max=num_classes - 1), num_classes=num_classes).to(weights.dtype)
    weights = torch.where(weights.sum(dim=-1, keepdim=True) > 0, weights, hard)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _apply_point_weight(raw: torch.Tensor, finite: torch.Tensor, point_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    finite_f = finite.to(raw.dtype)
    while finite_f.ndim < raw.ndim:
        finite_f = finite_f.unsqueeze(-1)
    weight = finite_f
    if point_weight is not None:
        point_weight_f = point_weight.to(device=raw.device, dtype=raw.dtype)
        while point_weight_f.ndim < raw.ndim:
            point_weight_f = point_weight_f.unsqueeze(-1)
        weight = weight * point_weight_f
    return (raw * weight).sum() / weight.sum().clamp_min(1.0)


def _ordinal_cdf_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    finite: torch.Tensor,
    point_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    cdf_pred = torch.cumsum(probs, dim=-1).clamp(1e-6, 1.0 - 1e-6)
    class_ids = torch.arange(logits.shape[-1], device=logits.device).view(1, 1, -1)
    cdf_target = (class_ids >= labels.unsqueeze(-1)).to(cdf_pred.dtype)
    raw = torch.nn.functional.binary_cross_entropy(cdf_pred, cdf_target, reduction="none")
    return _apply_point_weight(raw, finite, point_weight=point_weight)


def _ordinal_emd_loss(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    finite: torch.Tensor,
    point_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_cdf = torch.cumsum(probs, dim=-1)
    target_cdf = torch.cumsum(target_probs.to(dtype=probs.dtype), dim=-1)
    raw = torch.square(pred_cdf - target_cdf).mean(dim=-1)
    return _apply_point_weight(raw, finite, point_weight=point_weight)


def _ordinal_bin_edges_from_centers(
    centers: torch.Tensor,
    *,
    left: float,
    right: float,
) -> torch.Tensor:
    centers = centers.flatten()
    if centers.numel() < 1:
        raise ValueError("ordinal bin centers must contain at least one value.")
    if centers.numel() == 1:
        return torch.tensor([float(left), float(right)], device=centers.device, dtype=centers.dtype)
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[:1] - (mids[:1] - centers[:1])
    last = centers[-1:] + (centers[-1:] - mids[-1:])
    edges = torch.cat([first, mids, last], dim=0)
    edges = edges.clone()
    edges[0] = torch.maximum(edges[0], torch.as_tensor(float(left), device=edges.device, dtype=edges.dtype))
    edges[-1] = torch.minimum(edges[-1], torch.as_tensor(float(right), device=edges.device, dtype=edges.dtype))
    return edges


def _ordinal_tpt_gaussian_targets(
    target_raw: torch.Tensor,
    bin_centers_raw: torch.Tensor,
    *,
    sigma: float,
    left: float,
    right: float,
) -> torch.Tensor:
    """OCE-TS target-to-probability transform using a truncated Gaussian."""

    sigma_f = max(float(sigma), 1e-6)
    centers = bin_centers_raw.to(device=target_raw.device, dtype=target_raw.dtype).flatten()
    edges = _ordinal_bin_edges_from_centers(centers, left=float(left), right=float(right))
    sqrt_two = math.sqrt(2.0)
    target = target_raw.unsqueeze(-1)
    lower = edges[:-1].view(*([1] * target_raw.ndim), -1)
    upper = edges[1:].view(*([1] * target_raw.ndim), -1)
    left_t = torch.as_tensor(float(left), device=target_raw.device, dtype=target_raw.dtype)
    right_t = torch.as_tensor(float(right), device=target_raw.device, dtype=target_raw.dtype)
    denom = 0.5 * (
        torch.erf((right_t - target_raw) / (sigma_f * sqrt_two))
        - torch.erf((left_t - target_raw) / (sigma_f * sqrt_two))
    )
    probs = 0.5 * (
        torch.erf((upper - target) / (sigma_f * sqrt_two))
        - torch.erf((lower - target) / (sigma_f * sqrt_two))
    )
    probs = probs / denom.unsqueeze(-1).clamp_min(1e-8)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    hard_idx = torch.argmin(torch.abs(target - centers.view(*([1] * target_raw.ndim), -1)), dim=-1)
    hard = torch.nn.functional.one_hot(hard_idx.clamp(min=0, max=int(centers.numel()) - 1), num_classes=int(centers.numel())).to(
        probs.dtype
    )
    probs = torch.where(probs.sum(dim=-1, keepdim=True) > 0, probs, hard)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _ordinal_oce_loss(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    finite: torch.Tensor,
    point_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_cdf = torch.cumsum(probs, dim=-1)[..., :-1].clamp(1e-6, 1.0 - 1e-6)
    target_cdf = torch.cumsum(target_probs.to(dtype=probs.dtype), dim=-1)[..., :-1].clamp(0.0, 1.0)
    raw = -(target_cdf * torch.log(pred_cdf) + (1.0 - target_cdf) * torch.log(1.0 - pred_cdf))
    return _apply_point_weight(raw, finite, point_weight=point_weight)


def _ordinal_slace_closeness_matrix(
    class_counts: torch.Tensor,
    *,
    proximity: bool,
    normalize_proximity: bool,
    count_smoothing: float,
) -> torch.Tensor:
    min_count = max(float(count_smoothing), 1e-12)
    counts = class_counts.flatten().to(dtype=torch.float64).clamp_min(min_count)
    num_classes = int(counts.numel())
    if num_classes < 1:
        raise ValueError("SLACE class_counts must contain at least one class.")
    if not bool(proximity):
        ids = torch.arange(num_classes, device=counts.device, dtype=torch.float64)
        return -torch.abs(ids.view(1, -1) - ids.view(-1, 1))
    total = counts.sum().clamp_min(1e-12)
    prefix = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)], dim=0)
    closeness = counts.new_empty((num_classes, num_classes))
    for true_idx in range(num_classes):
        half_true = 0.5 * counts[true_idx]
        for cls_idx in range(num_classes):
            if cls_idx > true_idx:
                mass = half_true + (prefix[cls_idx + 1] - prefix[true_idx + 1])
            elif cls_idx == true_idx:
                mass = half_true
            else:
                mass = half_true + (prefix[true_idx] - prefix[cls_idx])
            closeness[true_idx, cls_idx] = -torch.log((mass / total).clamp_min(1e-12))
    if normalize_proximity:
        closeness = closeness / closeness.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return closeness


def _ordinal_sord_targets_from_closeness(
    labels: torch.Tensor,
    closeness: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    rows = closeness.to(device=labels.device).index_select(0, labels.reshape(-1)).to(torch.float32)
    distance = rows.max(dim=-1, keepdim=True).values - rows
    target_probs = torch.nn.functional.softmax(-float(alpha) * distance, dim=-1)
    return target_probs.view(*labels.shape, int(closeness.shape[-1]))


def _ordinal_slace_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    finite: torch.Tensor,
    *,
    class_counts: Optional[torch.Tensor],
    alpha: float,
    proximity: bool = True,
    normalize_proximity: bool = False,
    count_smoothing: float = 1e-6,
    point_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_classes = int(logits.shape[-1])
    if class_counts is None:
        class_counts_t = torch.ones(num_classes, device=logits.device, dtype=logits.dtype)
    else:
        class_counts_t = class_counts.to(device=logits.device, dtype=logits.dtype).flatten()
        if int(class_counts_t.numel()) != num_classes:
            raise ValueError("SLACE class_counts must match the ordinal head class count.")
    probs = torch.nn.functional.softmax(logits, dim=-1).clamp_min(1e-8)
    closeness = _ordinal_slace_closeness_matrix(
        class_counts_t,
        proximity=bool(proximity),
        normalize_proximity=bool(normalize_proximity),
        count_smoothing=float(count_smoothing),
    ).to(device=logits.device, dtype=probs.dtype)
    target_probs = _ordinal_sord_targets_from_closeness(labels, closeness, alpha=float(alpha)).to(dtype=probs.dtype)
    row_closeness = closeness.index_select(0, labels.reshape(-1)).view(*labels.shape, num_classes)
    dominance = row_closeness.unsqueeze(-1) <= row_closeness.unsqueeze(-2)
    accumulated = (dominance.to(probs.dtype) * probs.unsqueeze(-2)).sum(dim=-1).clamp_min(1e-8)
    raw = -(target_probs * torch.log(accumulated)).sum(dim=-1)
    return _apply_point_weight(raw, finite, point_weight=point_weight)


def _corn_ordinal_bce_loss(
    binary_logits: torch.Tensor,
    labels: torch.Tensor,
    finite: torch.Tensor,
    point_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    threshold_ids = torch.arange(binary_logits.shape[-1], device=binary_logits.device).view(1, 1, -1)
    labels_expanded = labels.unsqueeze(-1)
    binary_targets = (labels_expanded > threshold_ids).to(binary_logits.dtype)
    conditional_mask = labels_expanded >= threshold_ids
    finite_conditional = finite.unsqueeze(-1) & conditional_mask
    raw = torch.nn.functional.binary_cross_entropy_with_logits(binary_logits, binary_targets, reduction="none")
    return _apply_point_weight(raw, finite_conditional, point_weight=point_weight)


def compute_online_current_risk_loss(
    outputs: Mapping[str, torch.Tensor],
    fms: torch.Tensor,
    lengths: torch.Tensor,
    fms_scaler: Mapping[str, float],
    rise_horizon_steps: Sequence[int],
    rise_thresholds: Sequence[float],
    ordinal_bins: Sequence[float],
    current_reg_aux_weight: float = 0.4,
    ordinal_loss_weight: float = 0.6,
    ordinal_loss_mode: str = "ce",
    ordinal_soft_label_sigma: float = 1.0,
    ordinal_soft_label_kernel: str = "gaussian",
    ordinal_ev_loss_weight: float = 0.0,
    ordinal_low_weight: float = 1.0,
    ordinal_low_threshold: float = 2.0,
    ordinal_slace_alpha: float = 1.0,
    ordinal_slace_proximity: bool = True,
    ordinal_slace_normalize_proximity: bool = False,
    ordinal_slace_count_smoothing: float = 1e-6,
    ordinal_class_counts: Optional[torch.Tensor] = None,
    risk_loss_weight: float = 1.0,
    fall_horizon_steps: Optional[Sequence[int]] = None,
    fall_thresholds: Optional[Sequence[float]] = None,
    fall_loss_weight: float = 0.0,
    high_risk_horizon_steps: Optional[Sequence[int]] = None,
    high_risk_thresholds: Optional[Sequence[float]] = None,
    high_risk_loss_weight: float = 0.0,
    high_risk_label_mode: str = "future_any",
    high_risk_onset_past_steps: int = 0,
    smoothness_weight: float = 0.02,
    risk_pos_weight: str | float | Sequence[float] = "auto",
    fall_risk_pos_weight: str | float | Sequence[float] = "auto",
    high_risk_pos_weight: str | float | Sequence[float] = "auto",
    loss_type: str = "smooth_l1",
    future_aux_horizon_steps: Optional[Sequence[int]] = None,
    future_aux_loss_weight: float = 0.0,
    delta_aux_loss_weight: float = 0.0,
    event_aux_loss_weight: float = 0.0,
    event_delta_threshold: float = 1.0,
    anchor_break_weight: float = 0.0,
    anchor_break_threshold: float = 4.0,
    anchor_break_max_weight: float = 3.0,
    lds_weight_table: Optional[torch.Tensor] = None,
    lds_min: float = 0.0,
    lds_bin_size: float = 1.0,
    transition_weighting: bool = False,
    transition_horizon_steps: Optional[Sequence[int]] = None,
    transition_drop_threshold: float = 2.0,
    transition_recovery_threshold: float = 3.0,
    transition_high_threshold: float = 8.0,
    transition_low_threshold: float = 5.0,
    transition_rise_threshold: float = 3.0,
    transition_drop_weight: float = 2.0,
    transition_recovery_weight: float = 3.0,
    transition_rise_weight: float = 1.5,
    transition_max_weight: float = 4.0,
    trajectory_loss_weight: float = 0.0,
    trajectory_decoder_loss_weight: float = 0.0,
    trajectory_delta_steps: Optional[Sequence[int]] = None,
    trajectory_delta_weight: float = 1.0,
    trajectory_centered_weight: float = 0.5,
    trajectory_range_weight: float = 0.2,
    trajectory_loss_type: str = "mae",
    trajectory_min_points: int = 4,
    coarse_band_loss_weight: float = 0.0,
    coarse_residual_loss_weight: float = 0.0,
    regime_loss_weight: float = 0.0,
    regime_delta_slow_threshold: float = 0.5,
    regime_delta_rapid_threshold: float = 2.0,
    regime_high_threshold: float = 12.0,
    uncertainty_loss_weight: float = 0.0,
    session_affine_scale_regularization_weight: float = 0.0,
    session_affine_bias_regularization_weight: float = 0.0,
    calibration_residual_regularization_weight: float = 0.0,
    low_overprediction_weight: float = 0.0,
    high_underprediction_weight: float = 0.0,
    low_overprediction_threshold: float = 2.0,
    high_underprediction_threshold: float = 15.0,
    low_suppressor_gate_loss_weight: float = 0.0,
    low_suppressor_threshold: float = 2.0,
    low_suppressor_gate_pos_weight: float = 1.0,
    low_suppressor_gate_target_mode: str = "low",
    low_suppressor_anchor_threshold: float = 5.0,
    low_suppressor_recovery_delta: float = 4.0,
    low_suppressor_correction_regularization_weight: float = 0.0,
    anchor_gate_loss_weight: float = 0.0,
    anchor_gate_threshold: float = 10.0,
    anchor_gate_pos_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    current_pred = outputs["current"]
    if current_pred.ndim != 2:
        raise ValueError(f"online current prediction must be [B,P], got {current_pred.shape}")
    loss_name = str(loss_type).lower()
    if loss_name not in {"smooth_l1", "mse", "l1", "mae"}:
        raise ValueError("loss_type must be one of: smooth_l1, mse, l1, mae.")
    prediction_start = _tensor_scalar_int(outputs.get("prediction_start"), 0)
    f_min = float(fms_scaler["min"])
    f_max = float(fms_scaler["max"])
    f_range = max(f_max - f_min, 1e-8)
    thresholds_norm = [float(v) / f_range for v in rise_thresholds]
    fall_horizon_steps = [int(v) for v in (fall_horizon_steps or rise_horizon_steps)]
    fall_thresholds = [float(v) for v in (fall_thresholds or rise_thresholds)]
    if len(fall_thresholds) != len(fall_horizon_steps):
        raise ValueError("fall_thresholds must have the same length as fall_horizon_steps.")
    fall_thresholds_norm = [float(v) / f_range for v in fall_thresholds]
    bins_norm = [float(v - f_min) / f_range for v in ordinal_bins]
    high_risk_steps = [int(v) for v in (high_risk_horizon_steps or [])]
    high_risk_thresholds = [float(v) for v in (high_risk_thresholds or [])]
    high_risk_thresholds_norm = [float(v - f_min) / f_range for v in high_risk_thresholds]
    future_steps = [int(v) for v in (future_aux_horizon_steps or [])]
    targets = compute_online_current_risk_targets(
        fms,
        lengths,
        prediction_start,
        int(current_pred.shape[1]),
        rise_horizon_steps,
        thresholds_norm,
        bins_norm,
        fall_horizon_steps=fall_horizon_steps,
        fall_thresholds_normalized=fall_thresholds_norm,
        high_risk_horizon_steps=high_risk_steps,
        high_risk_thresholds_normalized=high_risk_thresholds_norm,
        high_risk_label_mode=high_risk_label_mode,
        high_risk_onset_past_steps=high_risk_onset_past_steps,
        future_horizon_steps=future_steps,
        event_delta_threshold_normalized=float(event_delta_threshold) / f_range,
    )
    mask = outputs["mask"].to(current_pred.device).bool() & targets["current_mask"].to(current_pred.device)
    target_current = targets["current"].to(current_pred.device)
    sample_weight: Optional[torch.Tensor] = None
    lds_mean_weight = 1.0
    lds_min_weight = 1.0
    lds_max_weight = 1.0
    lds_points = 0
    if lds_weight_table is not None:
        table = lds_weight_table.to(current_pred.device, dtype=current_pred.dtype).flatten()
        if table.numel() == 0:
            raise ValueError("lds_weight_table must contain at least one bin weight.")
        raw_target = target_current * f_range + f_min
        bin_size = max(float(lds_bin_size), 1e-8)
        bin_idx = torch.floor((raw_target - float(lds_min)) / bin_size).long().clamp(0, int(table.numel()) - 1)
        lds_weight = table[bin_idx]
        lds_weight = torch.where(torch.isfinite(raw_target), lds_weight, torch.ones_like(lds_weight))
        sample_weight = lds_weight
        finite_lds = mask & torch.isfinite(lds_weight)
        if finite_lds.any():
            lds_points = int(finite_lds.sum().detach().cpu())
            lds_values = lds_weight[finite_lds]
            lds_mean_weight = float(lds_values.mean().detach().cpu())
            lds_min_weight = float(lds_values.min().detach().cpu())
            lds_max_weight = float(lds_values.max().detach().cpu())
    anchor_break_points = 0
    anchor_break_mean_weight = 1.0
    if float(anchor_break_weight) > 0 and "calibration_end_fms" in outputs:
        base_fms = outputs["calibration_end_fms"].to(current_pred.device)
        if base_fms.ndim == 1:
            base_fms = base_fms.unsqueeze(1)
        anchor_delta_raw = (target_current - base_fms).abs() * f_range
        threshold = max(float(anchor_break_threshold), 1e-8)
        extra = ((anchor_delta_raw - threshold) / threshold).clamp_min(0.0) * float(anchor_break_weight)
        anchor_weight = (1.0 + extra).clamp(max=max(float(anchor_break_max_weight), 1.0))
        sample_weight = anchor_weight if sample_weight is None else sample_weight * anchor_weight
        finite_weight = mask & torch.isfinite(sample_weight)
        if finite_weight.any():
            anchor_break_points = int(((anchor_delta_raw > threshold) & finite_weight).sum().detach().cpu())
            anchor_break_mean_weight = float(sample_weight[finite_weight].mean().detach().cpu())
    transition_horizon_steps = [int(v) for v in (transition_horizon_steps or [])]
    transition_points = 0
    transition_drop_points = 0
    transition_recovery_points = 0
    transition_rise_points = 0
    transition_mean_weight = 1.0
    transition_min_weight = 1.0
    transition_max_observed_weight = 1.0
    if bool(transition_weighting) and transition_horizon_steps:
        transition_targets = compute_online_current_risk_targets(
            fms,
            lengths,
            prediction_start,
            int(current_pred.shape[1]),
            [],
            [],
            bins_norm,
            future_horizon_steps=transition_horizon_steps,
            event_delta_threshold_normalized=max(float(transition_drop_threshold), 1e-8) / f_range,
        )
        transition_mask = transition_targets["future_mask"].to(current_pred.device) & outputs["mask"].to(current_pred.device).bool().unsqueeze(-1)
        current_raw = target_current * f_range + f_min
        future_raw = transition_targets["future"].to(current_pred.device) * f_range + f_min
        delta_raw = transition_targets["future_delta"].to(current_pred.device) * f_range
        finite_transition = transition_mask & torch.isfinite(delta_raw) & torch.isfinite(future_raw) & torch.isfinite(current_raw).unsqueeze(-1)
        drop_event = finite_transition & (delta_raw <= -max(float(transition_drop_threshold), 1e-8))
        recovery_event = finite_transition & (
            ((current_raw.unsqueeze(-1) >= float(transition_high_threshold)) & (future_raw <= float(transition_low_threshold)))
            | (delta_raw <= -max(float(transition_recovery_threshold), 1e-8))
        )
        rise_event = finite_transition & (delta_raw >= max(float(transition_rise_threshold), 1e-8))
        drop_any = drop_event.any(dim=-1)
        recovery_any = recovery_event.any(dim=-1)
        rise_any = rise_event.any(dim=-1)
        transition_weight = torch.ones_like(current_pred)
        max_weight = max(float(transition_max_weight), 1.0)
        if float(transition_drop_weight) > 1.0:
            transition_weight = torch.where(
                drop_any,
                torch.maximum(transition_weight, torch.full_like(transition_weight, float(transition_drop_weight))),
                transition_weight,
            )
        if float(transition_recovery_weight) > 1.0:
            transition_weight = torch.where(
                recovery_any,
                torch.maximum(transition_weight, torch.full_like(transition_weight, float(transition_recovery_weight))),
                transition_weight,
            )
        if float(transition_rise_weight) > 1.0:
            transition_weight = torch.where(
                rise_any,
                torch.maximum(transition_weight, torch.full_like(transition_weight, float(transition_rise_weight))),
                transition_weight,
            )
        transition_weight = transition_weight.clamp(min=1.0, max=max_weight)
        transition_weight = torch.where(mask, transition_weight, torch.ones_like(transition_weight))
        sample_weight = transition_weight if sample_weight is None else sample_weight * transition_weight
        finite_weight = mask & finite_transition.any(dim=-1) & torch.isfinite(transition_weight)
        if finite_weight.any():
            transition_points = int(finite_weight.sum().detach().cpu())
            transition_drop_points = int((drop_any & mask).sum().detach().cpu())
            transition_recovery_points = int((recovery_any & mask).sum().detach().cpu())
            transition_rise_points = int((rise_any & mask).sum().detach().cpu())
            transition_values = transition_weight[mask & torch.isfinite(transition_weight)]
            transition_mean_weight = float(transition_values.mean().detach().cpu())
            transition_min_weight = float(transition_values.min().detach().cpu())
            transition_max_observed_weight = float(transition_values.max().detach().cpu())
    loss_current = _masked_regression_loss(current_pred, target_current, mask, loss_type=loss_name, sample_weight=sample_weight)
    loss_reg = current_pred.new_tensor(0.0)
    if float(current_reg_aux_weight) > 0 and "current_reg" in outputs:
        loss_reg = _masked_regression_loss(
            outputs["current_reg"].to(current_pred.device),
            target_current,
            mask,
            loss_type=loss_name,
            sample_weight=sample_weight,
        )
    loss_ordinal = current_pred.new_tensor(0.0)
    loss_ordinal_ev = current_pred.new_tensor(0.0)
    ordinal_points = 0
    ordinal_ev_points = 0
    if float(ordinal_loss_weight) > 0 and "ordinal_logits" in outputs:
        ordinal_mode = str(ordinal_loss_mode or "ce").lower()
        labels = targets["ordinal_labels"].to(current_pred.device)
        if ordinal_mode == "cross_entropy":
            ordinal_mode = "ce"
        target_raw_for_ordinal = target_current * f_range + f_min
        ordinal_point_weight: Optional[torch.Tensor] = None
        if abs(float(ordinal_low_weight) - 1.0) > 1e-8:
            low_weight = max(float(ordinal_low_weight), 0.0)
            ordinal_point_weight = torch.where(
                target_raw_for_ordinal <= float(ordinal_low_threshold),
                torch.full_like(target_current, low_weight),
                torch.ones_like(target_current),
            )
        if ordinal_mode in {"corn", "corn_bce"} and "ordinal_binary_logits" in outputs:
            binary_logits = outputs["ordinal_binary_logits"].to(current_pred.device)
            finite = mask & torch.isfinite(binary_logits).all(dim=-1)
            if finite.any():
                loss_ordinal = _corn_ordinal_bce_loss(
                    binary_logits,
                    labels,
                    finite,
                    point_weight=ordinal_point_weight,
                )
                ordinal_points = int(finite.sum().detach().cpu())
        elif ordinal_mode in {"cumulative", "cumulative_bce", "coral_bce"} and "ordinal_binary_logits" in outputs:
            binary_logits = outputs["ordinal_binary_logits"].to(current_pred.device)
            finite = mask & torch.isfinite(binary_logits).all(dim=-1)
            if finite.any():
                threshold_ids = torch.arange(binary_logits.shape[-1], device=current_pred.device).view(1, 1, -1)
                binary_targets = (labels.unsqueeze(-1) > threshold_ids).to(binary_logits.dtype)
                raw = torch.nn.functional.binary_cross_entropy_with_logits(binary_logits, binary_targets, reduction="none")
                loss_ordinal = _apply_point_weight(raw, finite, point_weight=ordinal_point_weight)
                ordinal_points = int(finite.sum().detach().cpu())
        else:
            logits = outputs["ordinal_logits"].to(current_pred.device)
            finite = mask & torch.isfinite(logits).all(dim=-1)
            if finite.any():
                if ordinal_mode in {"soft_ce", "soft_label_ce", "unimodal_soft_ce"}:
                    target_probs = _ordinal_soft_targets(
                        labels,
                        logits.shape[-1],
                        sigma=ordinal_soft_label_sigma,
                        kernel=ordinal_soft_label_kernel,
                    )
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    raw = -(target_probs * log_probs).sum(dim=-1)
                    loss_ordinal = _apply_point_weight(raw, finite, point_weight=ordinal_point_weight)
                elif ordinal_mode in {"slace", "slace_prox", "slace_paper"}:
                    loss_ordinal = _ordinal_slace_loss(
                        logits,
                        labels,
                        finite,
                        class_counts=ordinal_class_counts,
                        alpha=ordinal_slace_alpha,
                        proximity=ordinal_slace_proximity,
                        normalize_proximity=ordinal_slace_normalize_proximity,
                        count_smoothing=ordinal_slace_count_smoothing,
                        point_weight=ordinal_point_weight,
                    )
                elif ordinal_mode in {"slace_index", "slace_no_prox"}:
                    loss_ordinal = _ordinal_slace_loss(
                        logits,
                        labels,
                        finite,
                        class_counts=ordinal_class_counts,
                        alpha=ordinal_slace_alpha,
                        proximity=False,
                        normalize_proximity=False,
                        count_smoothing=ordinal_slace_count_smoothing,
                        point_weight=ordinal_point_weight,
                    )
                elif ordinal_mode in {"slace_norm_prox", "slace_prox_norm"}:
                    loss_ordinal = _ordinal_slace_loss(
                        logits,
                        labels,
                        finite,
                        class_counts=ordinal_class_counts,
                        alpha=ordinal_slace_alpha,
                        proximity=True,
                        normalize_proximity=True,
                        count_smoothing=ordinal_slace_count_smoothing,
                        point_weight=ordinal_point_weight,
                    )
                elif ordinal_mode in {"emd", "emd2", "soft_emd", "dldl_emd", "dldl", "wasserstein"}:
                    target_probs = _ordinal_soft_targets(
                        labels,
                        logits.shape[-1],
                        sigma=ordinal_soft_label_sigma,
                        kernel=ordinal_soft_label_kernel,
                    )
                    loss_ordinal = _ordinal_emd_loss(logits, target_probs, finite, point_weight=ordinal_point_weight)
                elif ordinal_mode in {"cdf_bce", "ordinal_cdf_bce", "oce"}:
                    loss_ordinal = _ordinal_cdf_bce_loss(logits, labels, finite, point_weight=ordinal_point_weight)
                elif ordinal_mode in {"oce_ts", "tpt_oce", "soft_oce_ts"}:
                    bin_centers_raw = torch.tensor(
                        [float(v) for v in ordinal_bins],
                        device=current_pred.device,
                        dtype=target_raw_for_ordinal.dtype,
                    )
                    target_probs = _ordinal_tpt_gaussian_targets(
                        target_raw_for_ordinal,
                        bin_centers_raw,
                        sigma=ordinal_soft_label_sigma,
                        left=f_min,
                        right=f_max,
                    )
                    loss_ordinal = _ordinal_oce_loss(logits, target_probs, finite, point_weight=ordinal_point_weight)
                else:
                    raw = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
                    loss_ordinal = _apply_point_weight(raw.view_as(finite), finite, point_weight=ordinal_point_weight)
                ordinal_points = int(finite.sum().detach().cpu())
    if float(ordinal_ev_loss_weight) > 0 and isinstance(outputs.get("ordinal_probs"), torch.Tensor):
        ordinal_probs = outputs["ordinal_probs"].to(current_pred.device)
        ordinal_bins_tensor = outputs.get("ordinal_bins")
        if isinstance(ordinal_bins_tensor, torch.Tensor) and ordinal_bins_tensor.numel() == ordinal_probs.shape[-1]:
            ordinal_bins_tensor = ordinal_bins_tensor.to(current_pred.device, dtype=ordinal_probs.dtype).view(1, 1, -1)
            pred_ev = (ordinal_probs * ordinal_bins_tensor).sum(dim=-1)
            finite_ev = mask & torch.isfinite(pred_ev) & torch.isfinite(target_current)
            if finite_ev.any():
                loss_ordinal_ev = _masked_regression_loss(
                    pred_ev,
                    target_current,
                    finite_ev,
                    loss_type=loss_name,
                    sample_weight=None,
                )
                ordinal_ev_points = int(finite_ev.sum().detach().cpu())
    loss_coarse_band = current_pred.new_tensor(0.0)
    coarse_band_points = 0
    if float(coarse_band_loss_weight) > 0 and isinstance(outputs.get("coarse_band_logits"), torch.Tensor):
        coarse_logits = outputs["coarse_band_logits"].to(current_pred.device)
        coarse_bins = outputs.get("coarse_band_bins")
        if isinstance(coarse_bins, torch.Tensor) and coarse_bins.numel() > 0 and coarse_logits.shape[-1] == int(coarse_bins.numel()) + 1:
            coarse_bins = coarse_bins.to(current_pred.device, dtype=target_current.dtype).flatten()
            labels = torch.bucketize(target_current.contiguous(), coarse_bins.contiguous()).long()
            finite = mask & torch.isfinite(coarse_logits).all(dim=-1)
            if finite.any():
                raw = torch.nn.functional.cross_entropy(
                    coarse_logits.reshape(-1, coarse_logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="none",
                )
                finite_flat = finite.reshape(-1).to(raw.dtype)
                loss_coarse_band = (raw * finite_flat).sum() / finite_flat.sum().clamp_min(1.0)
                coarse_band_points = int(finite.sum().detach().cpu())
    loss_coarse_residual = current_pred.new_tensor(0.0)
    coarse_residual_points = 0
    if float(coarse_residual_loss_weight) > 0 and isinstance(outputs.get("current_coarse_residual"), torch.Tensor):
        coarse_residual_pred = outputs["current_coarse_residual"].to(current_pred.device, dtype=current_pred.dtype)
        if coarse_residual_pred.shape == target_current.shape:
            finite = mask & torch.isfinite(coarse_residual_pred) & torch.isfinite(target_current)
            if finite.any():
                loss_coarse_residual = _masked_regression_loss(
                    coarse_residual_pred,
                    target_current,
                    finite,
                    loss_type=loss_name,
                )
                coarse_residual_points = int(finite.sum().detach().cpu())
    loss_regime = current_pred.new_tensor(0.0)
    regime_points = 0
    regime_logits_source = outputs.get("current_regime_gate_logits", outputs.get("regime_logits"))
    if float(regime_loss_weight) > 0 and isinstance(regime_logits_source, torch.Tensor):
        regime_logits = regime_logits_source.to(current_pred.device)
        if regime_logits.shape[-1] >= 5 and int(current_pred.shape[1]) > 0:
            positions = prediction_start + torch.arange(current_pred.shape[1], device=current_pred.device)
            regime_horizon = int(rise_horizon_steps[0]) if rise_horizon_steps else int(max(1, current_pred.shape[1]))
            future_idx = positions + regime_horizon
            valid_future = future_idx.view(1, -1) < lengths.to(current_pred.device).view(-1, 1)
            gather_idx = future_idx.clamp(max=max(int(fms.shape[1]) - 1, 0)).view(1, -1).expand(fms.shape[0], -1)
            fms_device = fms.to(current_pred.device)
            future = fms_device.gather(1, gather_idx)
            delta_raw = (future - target_current) * f_range
            current_raw = target_current * f_range + f_min
            slow_thr = max(float(regime_delta_slow_threshold), 0.0)
            rapid_thr = max(float(regime_delta_rapid_threshold), slow_thr + 1e-6)
            high_thr = float(regime_high_threshold)
            labels = torch.zeros_like(target_current, dtype=torch.long)
            labels = torch.where(delta_raw <= -slow_thr, torch.full_like(labels, 4), labels)
            labels = torch.where((delta_raw > -slow_thr) & (delta_raw >= rapid_thr), torch.full_like(labels, 2), labels)
            labels = torch.where(
                (delta_raw > -slow_thr) & (delta_raw >= slow_thr) & (delta_raw < rapid_thr),
                torch.full_like(labels, 1),
                labels,
            )
            labels = torch.where(
                (delta_raw.abs() < slow_thr) & (current_raw >= high_thr),
                torch.full_like(labels, 3),
                labels,
            )
            finite = mask & valid_future & torch.isfinite(regime_logits).all(dim=-1) & torch.isfinite(delta_raw)
            if finite.any():
                raw = torch.nn.functional.cross_entropy(
                    regime_logits.reshape(-1, regime_logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="none",
                )
                finite_flat = finite.reshape(-1).to(raw.dtype)
                loss_regime = (raw * finite_flat).sum() / finite_flat.sum().clamp_min(1.0)
                regime_points = int(finite.sum().detach().cpu())
    loss_uncertainty = current_pred.new_tensor(0.0)
    uncertainty_points = 0
    if float(uncertainty_loss_weight) > 0 and isinstance(outputs.get("current_log_sigma"), torch.Tensor):
        log_sigma = outputs["current_log_sigma"].to(current_pred.device, dtype=current_pred.dtype)
        if log_sigma.shape == current_pred.shape:
            finite = mask & torch.isfinite(current_pred) & torch.isfinite(target_current) & torch.isfinite(log_sigma)
            if finite.any():
                log_sigma_valid = log_sigma[finite].clamp(-2.5, 1.0)
                sigma = torch.exp(log_sigma_valid).clamp_min(0.05)
                err = (current_pred.detach()[finite] - target_current[finite]).clamp(-1.0, 1.0)
                nll = 0.5 * (err / sigma).square() + log_sigma_valid
                loss_uncertainty = torch.nan_to_num(nll, nan=50.0, posinf=50.0, neginf=-5.0).clamp(-5.0, 50.0).mean()
                uncertainty_points = int(finite.sum().detach().cpu())
        if not torch.isfinite(loss_uncertainty):
            loss_uncertainty = current_pred.new_tensor(0.0)
            uncertainty_points = 0
    loss_risk = current_pred.new_tensor(0.0)
    risk_points = 0
    risk_pos_counts: List[float] = []
    risk_total_counts: List[float] = []
    if float(risk_loss_weight) > 0 and "risk_logits" in outputs:
        logits = outputs["risk_logits"].to(current_pred.device)
        labels = targets["rise_labels"].to(current_pred.device)
        risk_mask = targets["rise_mask"].to(current_pred.device)
        risk_mask = risk_mask & outputs["mask"].to(current_pred.device).bool().unsqueeze(-1)
        finite = risk_mask & torch.isfinite(logits) & torch.isfinite(labels)
        if isinstance(risk_pos_weight, str) and risk_pos_weight.lower() == "auto":
            pos = (labels * finite.to(labels.dtype)).sum(dim=(0, 1))
            total = finite.to(labels.dtype).sum(dim=(0, 1))
            neg = (total - pos).clamp_min(0.0)
            pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, 20.0)
        elif isinstance(risk_pos_weight, (list, tuple, np.ndarray)):
            pos_weight = torch.tensor([float(v) for v in risk_pos_weight], dtype=logits.dtype, device=logits.device)
        else:
            pos_weight = torch.full((logits.shape[-1],), float(risk_pos_weight), dtype=logits.dtype, device=logits.device)
        raw_risk = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
            pos_weight=pos_weight.view(1, 1, -1),
        )
        finite_f = finite.to(raw_risk.dtype)
        loss_risk = (raw_risk * finite_f).sum() / finite_f.sum().clamp_min(1.0)
        risk_points = int(finite.sum().detach().cpu())
        risk_pos_counts = (labels * finite.to(labels.dtype)).sum(dim=(0, 1)).detach().cpu().tolist()
        risk_total_counts = finite.to(labels.dtype).sum(dim=(0, 1)).detach().cpu().tolist()
    loss_fall_risk = current_pred.new_tensor(0.0)
    fall_risk_points = 0
    fall_risk_pos_counts: List[float] = []
    fall_risk_total_counts: List[float] = []
    if float(fall_loss_weight) > 0 and "fall_risk_logits" in outputs:
        logits = outputs["fall_risk_logits"].to(current_pred.device)
        labels = targets["fall_labels"].to(current_pred.device)
        fall_mask = targets["fall_mask"].to(current_pred.device)
        fall_mask = fall_mask & outputs["mask"].to(current_pred.device).bool().unsqueeze(-1)
        if logits.shape != labels.shape:
            raise ValueError(f"fall_risk_logits must match fall labels {labels.shape}, got {logits.shape}")
        finite = fall_mask & torch.isfinite(logits) & torch.isfinite(labels)
        if isinstance(fall_risk_pos_weight, str) and fall_risk_pos_weight.lower() == "auto":
            pos = (labels * finite.to(labels.dtype)).sum(dim=(0, 1))
            total = finite.to(labels.dtype).sum(dim=(0, 1))
            neg = (total - pos).clamp_min(0.0)
            pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, 20.0)
        elif isinstance(fall_risk_pos_weight, (list, tuple, np.ndarray)):
            pos_weight = torch.tensor([float(v) for v in fall_risk_pos_weight], dtype=logits.dtype, device=logits.device)
        else:
            pos_weight = torch.full((logits.shape[-1],), float(fall_risk_pos_weight), dtype=logits.dtype, device=logits.device)
        raw_fall_risk = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
            pos_weight=pos_weight.view(1, 1, -1),
        )
        finite_f = finite.to(raw_fall_risk.dtype)
        loss_fall_risk = (raw_fall_risk * finite_f).sum() / finite_f.sum().clamp_min(1.0)
        fall_risk_points = int(finite.sum().detach().cpu())
        fall_risk_pos_counts = (labels * finite.to(labels.dtype)).sum(dim=(0, 1)).detach().cpu().tolist()
        fall_risk_total_counts = finite.to(labels.dtype).sum(dim=(0, 1)).detach().cpu().tolist()
    loss_high_risk = current_pred.new_tensor(0.0)
    high_risk_points = 0
    high_risk_pos_counts: List[float] = []
    high_risk_total_counts: List[float] = []
    if float(high_risk_loss_weight) > 0 and "high_risk_logits" in outputs:
        logits = outputs["high_risk_logits"].to(current_pred.device)
        labels = targets["high_risk_labels"].to(current_pred.device)
        high_mask = targets["high_risk_mask"].to(current_pred.device)
        high_mask = high_mask & outputs["mask"].to(current_pred.device).bool().unsqueeze(-1).unsqueeze(-1)
        if logits.shape != labels.shape:
            raise ValueError(f"high_risk_logits must match high-risk labels {labels.shape}, got {logits.shape}")
        finite = high_mask & torch.isfinite(logits) & torch.isfinite(labels)
        if isinstance(high_risk_pos_weight, str) and high_risk_pos_weight.lower() == "auto":
            pos = (labels * finite.to(labels.dtype)).sum(dim=(0, 1))
            total = finite.to(labels.dtype).sum(dim=(0, 1))
            neg = (total - pos).clamp_min(0.0)
            pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, 20.0)
        elif isinstance(high_risk_pos_weight, (list, tuple, np.ndarray)):
            raw_weights = torch.tensor([float(v) for v in high_risk_pos_weight], dtype=logits.dtype, device=logits.device)
            if raw_weights.numel() == logits.shape[-1]:
                pos_weight = raw_weights.view(1, -1).expand(logits.shape[-2], -1)
            elif raw_weights.numel() == logits.shape[-2] * logits.shape[-1]:
                pos_weight = raw_weights.view(logits.shape[-2], logits.shape[-1])
            else:
                raise ValueError(
                    "high_risk_pos_weight list must have one value per threshold or one value per horizon-threshold pair."
                )
        else:
            pos_weight = torch.full(
                (logits.shape[-2], logits.shape[-1]),
                float(high_risk_pos_weight),
                dtype=logits.dtype,
                device=logits.device,
            )
        raw_high = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
            pos_weight=pos_weight.view(1, 1, logits.shape[-2], logits.shape[-1]),
        )
        finite_f = finite.to(raw_high.dtype)
        loss_high_risk = (raw_high * finite_f).sum() / finite_f.sum().clamp_min(1.0)
        high_risk_points = int(finite.sum().detach().cpu())
        high_risk_pos_counts = (labels * finite.to(labels.dtype)).sum(dim=(0, 1)).detach().cpu().reshape(-1).tolist()
        high_risk_total_counts = finite.to(labels.dtype).sum(dim=(0, 1)).detach().cpu().reshape(-1).tolist()
    loss_smooth = current_pred.new_tensor(0.0)
    smooth_points = 0
    if float(smoothness_weight) > 0 and int(current_pred.shape[1]) > 1:
        pred_delta = current_pred[:, 1:] - current_pred[:, :-1]
        target_delta = target_current[:, 1:] - target_current[:, :-1]
        delta_mask = mask[:, 1:] & mask[:, :-1]
        loss_smooth = _masked_regression_loss(pred_delta, target_delta, delta_mask, loss_type=loss_name)
        smooth_points = int(delta_mask.sum().detach().cpu())
    loss_future_aux = current_pred.new_tensor(0.0)
    future_aux_points = 0
    future_pred: Optional[torch.Tensor] = None
    future_mask: Optional[torch.Tensor] = None
    if (
        float(future_aux_loss_weight) > 0
        and future_steps
        and isinstance(outputs.get("future_aux"), torch.Tensor)
    ):
        future_pred = outputs["future_aux"].to(current_pred.device)
        target_future = targets["future"].to(current_pred.device)
        future_mask = targets["future_mask"].to(current_pred.device) & outputs["mask"].to(current_pred.device).bool().unsqueeze(-1)
        if future_pred.shape != target_future.shape:
            raise ValueError(f"future_aux must be {target_future.shape}, got {future_pred.shape}")
        loss_future_aux = _masked_regression_loss(future_pred, target_future, future_mask, loss_type=loss_name)
        future_aux_points = int(future_mask.sum().detach().cpu())
    loss_delta_aux = current_pred.new_tensor(0.0)
    delta_aux_points = 0
    if (
        float(delta_aux_loss_weight) > 0
        and future_steps
        and isinstance(outputs.get("future_aux"), torch.Tensor)
    ):
        if future_pred is None or future_mask is None:
            future_pred = outputs["future_aux"].to(current_pred.device)
            target_future = targets["future"].to(current_pred.device)
            future_mask = targets["future_mask"].to(current_pred.device) & outputs["mask"].to(current_pred.device).bool().unsqueeze(-1)
            if future_pred.shape != target_future.shape:
                raise ValueError(f"future_aux must be {target_future.shape}, got {future_pred.shape}")
        pred_future_delta = future_pred - current_pred.unsqueeze(-1)
        target_future_delta = targets["future_delta"].to(current_pred.device)
        loss_delta_aux = _masked_regression_loss(pred_future_delta, target_future_delta, future_mask, loss_type=loss_name)
        delta_aux_points = int(future_mask.sum().detach().cpu())
    loss_event_aux = current_pred.new_tensor(0.0)
    event_aux_points = 0
    if (
        float(event_aux_loss_weight) > 0
        and future_steps
        and isinstance(outputs.get("event_logits"), torch.Tensor)
    ):
        event_logits = outputs["event_logits"].to(current_pred.device)
        event_labels = targets["event_labels"].to(current_pred.device)
        event_mask = targets["future_mask"].to(current_pred.device) & outputs["mask"].to(current_pred.device).bool().unsqueeze(-1)
        if event_logits.shape[:3] != event_labels.shape or event_logits.shape[-1] != 3:
            raise ValueError(f"event_logits must be [B,P,H,3] for labels {event_labels.shape}, got {event_logits.shape}")
        finite = event_mask & torch.isfinite(event_logits).all(dim=-1)
        if finite.any():
            raw = torch.nn.functional.cross_entropy(
                event_logits.reshape(-1, 3),
                event_labels.reshape(-1),
                reduction="none",
            )
            finite_flat = finite.reshape(-1).to(raw.dtype)
            loss_event_aux = (raw * finite_flat).sum() / finite_flat.sum().clamp_min(1.0)
            event_aux_points = int(finite.sum().detach().cpu())
    loss_trajectory_decoder = current_pred.new_tensor(0.0)
    trajectory_decoder_points = 0
    trajectory_decoder_offset_count = 0
    if float(trajectory_decoder_loss_weight) > 0:
        trajectory_pred = outputs.get("current_trajectory")
        if not isinstance(trajectory_pred, torch.Tensor):
            raise ValueError("trajectory_decoder_loss_weight requires model output 'current_trajectory'.")
        trajectory_pred = trajectory_pred.to(current_pred.device, dtype=current_pred.dtype)
        if trajectory_pred.ndim != 3 or trajectory_pred.shape[:2] != current_pred.shape:
            raise ValueError(f"current_trajectory must be [B,P,K] matching current, got {trajectory_pred.shape}.")
        offsets_tensor = outputs.get("current_trajectory_offsets")
        if isinstance(offsets_tensor, torch.Tensor):
            offsets = offsets_tensor.to(current_pred.device, dtype=torch.long).flatten()
        else:
            offsets = torch.arange(trajectory_pred.shape[-1], dtype=torch.long, device=current_pred.device)
        if int(offsets.numel()) != int(trajectory_pred.shape[-1]):
            raise ValueError("current_trajectory_offsets length must match current_trajectory last dimension.")
        trajectory_decoder_offset_count = int(offsets.numel())
        positions = prediction_start + torch.arange(current_pred.shape[1], device=current_pred.device)
        fms_device = fms.to(current_pred.device, dtype=current_pred.dtype)
        lengths_device = lengths.to(current_pred.device)
        max_index = max(int(fms_device.shape[1]) - 1, 0)
        target_slices: List[torch.Tensor] = []
        mask_slices: List[torch.Tensor] = []
        for offset_value in offsets.detach().cpu().tolist():
            index = positions + int(offset_value)
            valid = (index.view(1, -1) >= 0) & (index.view(1, -1) < lengths_device.view(-1, 1))
            gather_index = index.clamp(0, max_index).view(1, -1).expand(fms_device.shape[0], -1)
            target_slices.append(fms_device.gather(1, gather_index))
            mask_slices.append(valid)
        if target_slices:
            target_trajectory = torch.stack(target_slices, dim=-1)
            trajectory_mask = torch.stack(mask_slices, dim=-1) & outputs["mask"].to(current_pred.device).bool().unsqueeze(-1)
            finite = trajectory_mask & torch.isfinite(trajectory_pred) & torch.isfinite(target_trajectory)
            decoder_weight = sample_weight.unsqueeze(-1).expand_as(trajectory_pred) if sample_weight is not None else None
            loss_trajectory_decoder = _masked_regression_loss(
                trajectory_pred,
                target_trajectory,
                finite,
                loss_type=loss_name,
                sample_weight=decoder_weight,
            )
            trajectory_decoder_points = int(finite.sum().detach().cpu())
    loss_trajectory = current_pred.new_tensor(0.0)
    trajectory_parts = {
        "loss_trajectory": 0.0,
        "loss_trajectory_delta": 0.0,
        "loss_trajectory_centered": 0.0,
        "loss_trajectory_range": 0.0,
        "trajectory_delta_points": 0,
        "trajectory_centered_points": 0,
        "trajectory_range_points": 0,
        "trajectory_delta_step_count": 0,
        "trajectory_delta_weight": float(trajectory_delta_weight),
        "trajectory_centered_weight": float(trajectory_centered_weight),
        "trajectory_range_weight": float(trajectory_range_weight),
        "trajectory_min_points": int(trajectory_min_points),
    }
    if float(trajectory_loss_weight) > 0:
        loss_trajectory, trajectory_parts = _masked_trajectory_shape_loss(
            current_pred,
            target_current,
            mask,
            delta_steps=trajectory_delta_steps,
            loss_type=trajectory_loss_type,
            delta_weight=trajectory_delta_weight,
            centered_weight=trajectory_centered_weight,
            range_weight=trajectory_range_weight,
            min_points=trajectory_min_points,
        )
    loss_session_affine_scale_reg = current_pred.new_tensor(0.0)
    loss_session_affine_bias_reg = current_pred.new_tensor(0.0)
    session_affine_scale_points = 0
    session_affine_bias_points = 0
    valid_session = mask.any(dim=1, keepdim=True)
    scale = outputs.get("current_session_affine_scale")
    if float(session_affine_scale_regularization_weight) > 0 and isinstance(scale, torch.Tensor):
        scale = scale.to(current_pred.device, dtype=current_pred.dtype)
        if scale.ndim == 1:
            scale = scale.view(-1, 1)
        if scale.shape[0] != current_pred.shape[0] or scale.ndim != 2:
            raise ValueError(f"current_session_affine_scale must be [B] or [B,K], got {scale.shape}.")
        if scale.shape[1] not in {1, current_pred.shape[1]}:
            raise ValueError(f"current_session_affine_scale must have one value per session or prediction step, got {scale.shape}.")
        finite_scale = valid_session.expand_as(scale) & torch.isfinite(scale)
        if finite_scale.any():
            loss_session_affine_scale_reg = (scale[finite_scale] - 1.0).square().mean()
            session_affine_scale_points = int(finite_scale.sum().detach().cpu())
    bias = outputs.get("current_session_affine_bias")
    if float(session_affine_bias_regularization_weight) > 0 and isinstance(bias, torch.Tensor):
        bias = bias.to(current_pred.device, dtype=current_pred.dtype)
        if bias.ndim == 1:
            bias = bias.view(-1, 1)
        if bias.shape[0] != current_pred.shape[0] or bias.ndim != 2:
            raise ValueError(f"current_session_affine_bias must be [B] or [B,K], got {bias.shape}.")
        if bias.shape[1] not in {1, current_pred.shape[1]}:
            raise ValueError(f"current_session_affine_bias must have one value per session or prediction step, got {bias.shape}.")
        finite_bias = valid_session.expand_as(bias) & torch.isfinite(bias)
        if finite_bias.any():
            loss_session_affine_bias_reg = bias[finite_bias].square().mean()
            session_affine_bias_points = int(finite_bias.sum().detach().cpu())
    loss_calibration_residual_reg = current_pred.new_tensor(0.0)
    calibration_residual_reg_points = 0
    correction = outputs.get("current_residual_adapter_correction")
    if float(calibration_residual_regularization_weight) > 0 and isinstance(correction, torch.Tensor):
        correction = correction.to(current_pred.device, dtype=current_pred.dtype)
        if correction.shape != current_pred.shape:
            raise ValueError(
                f"current_residual_adapter_correction must match current prediction {current_pred.shape}, got {correction.shape}."
            )
        finite_correction = mask & torch.isfinite(correction)
        if finite_correction.any():
            loss_calibration_residual_reg = correction[finite_correction].square().mean()
            calibration_residual_reg_points = int(finite_correction.sum().detach().cpu())
    loss_low_overprediction = current_pred.new_tensor(0.0)
    loss_high_underprediction = current_pred.new_tensor(0.0)
    low_overprediction_points = 0
    high_underprediction_points = 0
    if float(low_overprediction_weight) > 0 or float(high_underprediction_weight) > 0:
        target_raw = target_current * f_range + f_min
        pred_raw = current_pred * f_range + f_min
        finite_bias = mask & torch.isfinite(target_raw) & torch.isfinite(pred_raw)
        if float(low_overprediction_weight) > 0:
            low_mask = finite_bias & (target_raw <= float(low_overprediction_threshold)) & (pred_raw > target_raw)
            if low_mask.any():
                loss_low_overprediction = ((pred_raw[low_mask] - target_raw[low_mask]) / f_range).mean()
                low_overprediction_points = int(low_mask.sum().detach().cpu())
        if float(high_underprediction_weight) > 0:
            high_mask = finite_bias & (target_raw >= float(high_underprediction_threshold)) & (pred_raw < target_raw)
            if high_mask.any():
                loss_high_underprediction = ((target_raw[high_mask] - pred_raw[high_mask]) / f_range).mean()
                high_underprediction_points = int(high_mask.sum().detach().cpu())
    loss_low_suppressor_gate = current_pred.new_tensor(0.0)
    loss_low_suppressor_correction_reg = current_pred.new_tensor(0.0)
    loss_anchor_gate = current_pred.new_tensor(0.0)
    low_suppressor_gate_points = 0
    low_suppressor_correction_points = 0
    anchor_gate_points = 0
    gate_logits = outputs.get("current_low_suppressor_gate_logits")
    if not isinstance(gate_logits, torch.Tensor):
        gate_logits = outputs.get("current_calib_prior_gate_logits")
    if not isinstance(gate_logits, torch.Tensor):
        gate_logits = outputs.get("current_anchor_gate_logits")
    if float(low_suppressor_gate_loss_weight) > 0 and isinstance(gate_logits, torch.Tensor):
        gate_target_mode = str(low_suppressor_gate_target_mode or "low").strip().lower()
        if gate_target_mode not in {"low", "recovery_low", "anchor_drop_low"}:
            raise ValueError("low_suppressor_gate_target_mode must be one of: low, recovery_low, anchor_drop_low.")
        gate_logits = gate_logits.to(current_pred.device, dtype=current_pred.dtype)
        if gate_logits.shape != current_pred.shape:
            raise ValueError(f"current_low_suppressor_gate_logits must match {current_pred.shape}, got {gate_logits.shape}.")
        target_raw = target_current * f_range + f_min
        finite_gate = mask & torch.isfinite(gate_logits) & torch.isfinite(target_raw)
        if finite_gate.any():
            low_target = target_raw <= float(low_suppressor_threshold)
            if gate_target_mode == "low":
                gate_target_bool = low_target
            else:
                anchor = outputs.get("model_anchor_fms", outputs.get("calibration_end_fms"))
                if not isinstance(anchor, torch.Tensor):
                    raise ValueError(f"{gate_target_mode} gate target requires model_anchor_fms or calibration_end_fms.")
                anchor = anchor.to(current_pred.device, dtype=current_pred.dtype)
                if anchor.ndim == 1:
                    anchor = anchor.view(-1, 1)
                if anchor.shape[0] != current_pred.shape[0]:
                    raise ValueError(f"anchor tensor batch size must match current prediction, got {anchor.shape}.")
                if anchor.shape[1] == 1:
                    anchor = anchor.expand(-1, current_pred.shape[1])
                elif anchor.shape[1] != current_pred.shape[1]:
                    raise ValueError(f"anchor tensor must be [B] or [B,T], got {anchor.shape}.")
                anchor_raw = anchor * f_range + f_min
                anchor_high = anchor_raw >= float(low_suppressor_anchor_threshold)
                if gate_target_mode == "recovery_low":
                    gate_target_bool = low_target & anchor_high
                else:
                    gate_target_bool = low_target & ((anchor_raw - target_raw) >= float(low_suppressor_recovery_delta))
                finite_gate = finite_gate & torch.isfinite(anchor_raw)
            gate_target = gate_target_bool.to(dtype=gate_logits.dtype)
            pos_weight = torch.full(
                (),
                max(float(low_suppressor_gate_pos_weight), 1e-8),
                dtype=gate_logits.dtype,
                device=gate_logits.device,
            )
            raw_gate_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                gate_logits,
                gate_target,
                pos_weight=pos_weight,
                reduction="none",
            )
            loss_low_suppressor_gate = raw_gate_loss[finite_gate].mean()
            low_suppressor_gate_points = int(finite_gate.sum().detach().cpu())
    anchor_gate_logits = outputs.get("current_anchor_gate_logits")
    if float(anchor_gate_loss_weight) > 0 and isinstance(anchor_gate_logits, torch.Tensor):
        anchor_gate_logits = anchor_gate_logits.to(current_pred.device, dtype=current_pred.dtype)
        if anchor_gate_logits.shape != current_pred.shape:
            raise ValueError(f"current_anchor_gate_logits must match {current_pred.shape}, got {anchor_gate_logits.shape}.")
        target_raw = target_current * f_range + f_min
        finite_anchor_gate = mask & torch.isfinite(anchor_gate_logits) & torch.isfinite(target_raw)
        if finite_anchor_gate.any():
            anchor_gate_target = (target_raw >= float(anchor_gate_threshold)).to(dtype=anchor_gate_logits.dtype)
            anchor_pos_weight = torch.full(
                (),
                max(float(anchor_gate_pos_weight), 1e-8),
                dtype=anchor_gate_logits.dtype,
                device=anchor_gate_logits.device,
            )
            raw_anchor_gate_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                anchor_gate_logits,
                anchor_gate_target,
                pos_weight=anchor_pos_weight,
                reduction="none",
            )
            loss_anchor_gate = raw_anchor_gate_loss[finite_anchor_gate].mean()
            anchor_gate_points = int(finite_anchor_gate.sum().detach().cpu())
    suppressor_correction = outputs.get("current_low_suppressor_correction")
    if float(low_suppressor_correction_regularization_weight) > 0 and isinstance(suppressor_correction, torch.Tensor):
        suppressor_correction = suppressor_correction.to(current_pred.device, dtype=current_pred.dtype)
        if suppressor_correction.shape != current_pred.shape:
            raise ValueError(
                f"current_low_suppressor_correction must match {current_pred.shape}, got {suppressor_correction.shape}."
            )
        finite_correction = mask & torch.isfinite(suppressor_correction)
        if finite_correction.any():
            loss_low_suppressor_correction_reg = suppressor_correction[finite_correction].square().mean()
            low_suppressor_correction_points = int(finite_correction.sum().detach().cpu())
    loss = (
        loss_current
        + float(current_reg_aux_weight) * loss_reg
        + float(ordinal_loss_weight) * loss_ordinal
        + float(ordinal_ev_loss_weight) * loss_ordinal_ev
        + float(coarse_band_loss_weight) * loss_coarse_band
        + float(coarse_residual_loss_weight) * loss_coarse_residual
        + float(regime_loss_weight) * loss_regime
        + float(uncertainty_loss_weight) * loss_uncertainty
        + float(risk_loss_weight) * loss_risk
        + float(fall_loss_weight) * loss_fall_risk
        + float(high_risk_loss_weight) * loss_high_risk
        + float(smoothness_weight) * loss_smooth
        + float(future_aux_loss_weight) * loss_future_aux
        + float(delta_aux_loss_weight) * loss_delta_aux
        + float(event_aux_loss_weight) * loss_event_aux
        + float(trajectory_decoder_loss_weight) * loss_trajectory_decoder
        + float(trajectory_loss_weight) * loss_trajectory
        + float(session_affine_scale_regularization_weight) * loss_session_affine_scale_reg
        + float(session_affine_bias_regularization_weight) * loss_session_affine_bias_reg
        + float(calibration_residual_regularization_weight) * loss_calibration_residual_reg
        + float(low_overprediction_weight) * loss_low_overprediction
        + float(high_underprediction_weight) * loss_high_underprediction
        + float(low_suppressor_gate_loss_weight) * loss_low_suppressor_gate
        + float(low_suppressor_correction_regularization_weight) * loss_low_suppressor_correction_reg
        + float(anchor_gate_loss_weight) * loss_anchor_gate
    )
    valid_points = int(mask.sum().detach().cpu())
    parts = {
        "loss_total": float(loss.detach().cpu()),
        "loss_level": float(loss_current.detach().cpu()),
        "loss_trend": float(loss_smooth.detach().cpu()),
        "loss_current": float(loss_current.detach().cpu()),
        "loss_current_reg_aux": float(loss_reg.detach().cpu()),
        "loss_ordinal": float(loss_ordinal.detach().cpu()),
        "loss_ordinal_ev": float(loss_ordinal_ev.detach().cpu()),
        "loss_coarse_band": float(loss_coarse_band.detach().cpu()),
        "loss_coarse_residual": float(loss_coarse_residual.detach().cpu()),
        "loss_regime": float(loss_regime.detach().cpu()),
        "loss_uncertainty": float(loss_uncertainty.detach().cpu()),
        "loss_risk": float(loss_risk.detach().cpu()),
        "loss_fall_risk": float(loss_fall_risk.detach().cpu()),
        "loss_high_risk": float(loss_high_risk.detach().cpu()),
        "loss_smoothness": float(loss_smooth.detach().cpu()),
        "loss_future_aux": float(loss_future_aux.detach().cpu()),
        "loss_delta_aux": float(loss_delta_aux.detach().cpu()),
        "loss_event_aux": float(loss_event_aux.detach().cpu()),
        "loss_trajectory_decoder": float(loss_trajectory_decoder.detach().cpu()),
        "loss_trajectory": float(loss_trajectory.detach().cpu()),
        "loss_session_affine_scale_reg": float(loss_session_affine_scale_reg.detach().cpu()),
        "loss_session_affine_bias_reg": float(loss_session_affine_bias_reg.detach().cpu()),
        "loss_calibration_residual_reg": float(loss_calibration_residual_reg.detach().cpu()),
        "loss_low_overprediction": float(loss_low_overprediction.detach().cpu()),
        "loss_high_underprediction": float(loss_high_underprediction.detach().cpu()),
        "loss_low_suppressor_gate": float(loss_low_suppressor_gate.detach().cpu()),
        "loss_low_suppressor_correction_reg": float(loss_low_suppressor_correction_reg.detach().cpu()),
        "loss_anchor_gate": float(loss_anchor_gate.detach().cpu()),
        "loss_trajectory_delta": float(trajectory_parts["loss_trajectory_delta"]),
        "loss_trajectory_centered": float(trajectory_parts["loss_trajectory_centered"]),
        "loss_trajectory_range": float(trajectory_parts["loss_trajectory_range"]),
        "valid_points": int(valid_points),
        "ordinal_points": int(ordinal_points),
        "ordinal_ev_points": int(ordinal_ev_points),
        "coarse_band_points": int(coarse_band_points),
        "coarse_residual_points": int(coarse_residual_points),
        "regime_points": int(regime_points),
        "uncertainty_points": int(uncertainty_points),
        "risk_points": int(risk_points),
        "fall_risk_points": int(fall_risk_points),
        "high_risk_points": int(high_risk_points),
        "smoothness_points": int(smooth_points),
        "future_aux_points": int(future_aux_points),
        "delta_aux_points": int(delta_aux_points),
        "event_aux_points": int(event_aux_points),
        "trajectory_decoder_points": int(trajectory_decoder_points),
        "session_affine_scale_points": int(session_affine_scale_points),
        "session_affine_bias_points": int(session_affine_bias_points),
        "calibration_residual_reg_points": int(calibration_residual_reg_points),
        "low_overprediction_points": int(low_overprediction_points),
        "high_underprediction_points": int(high_underprediction_points),
        "low_suppressor_gate_points": int(low_suppressor_gate_points),
        "low_suppressor_correction_points": int(low_suppressor_correction_points),
        "anchor_gate_points": int(anchor_gate_points),
        "trajectory_decoder_offset_count": int(trajectory_decoder_offset_count),
        "trajectory_delta_points": int(trajectory_parts["trajectory_delta_points"]),
        "trajectory_centered_points": int(trajectory_parts["trajectory_centered_points"]),
        "trajectory_range_points": int(trajectory_parts["trajectory_range_points"]),
        "trajectory_delta_step_count": int(trajectory_parts["trajectory_delta_step_count"]),
        "current_reg_aux_weight": float(current_reg_aux_weight),
        "ordinal_loss_weight": float(ordinal_loss_weight),
        "ordinal_loss_mode": str(ordinal_loss_mode or "ce").lower(),
        "ordinal_soft_label_sigma": float(ordinal_soft_label_sigma),
        "ordinal_soft_label_kernel": str(ordinal_soft_label_kernel or "gaussian").lower(),
        "ordinal_ev_loss_weight": float(ordinal_ev_loss_weight),
        "ordinal_low_weight": float(ordinal_low_weight),
        "ordinal_low_threshold": float(ordinal_low_threshold),
        "ordinal_slace_alpha": float(ordinal_slace_alpha),
        "ordinal_slace_proximity": bool(ordinal_slace_proximity),
        "ordinal_slace_normalize_proximity": bool(ordinal_slace_normalize_proximity),
        "ordinal_slace_count_smoothing": float(ordinal_slace_count_smoothing),
        "ordinal_slace_class_count_available": bool(ordinal_class_counts is not None),
        "coarse_band_loss_weight": float(coarse_band_loss_weight),
        "coarse_residual_loss_weight": float(coarse_residual_loss_weight),
        "regime_loss_weight": float(regime_loss_weight),
        "regime_delta_slow_threshold": float(regime_delta_slow_threshold),
        "regime_delta_rapid_threshold": float(regime_delta_rapid_threshold),
        "regime_high_threshold": float(regime_high_threshold),
        "uncertainty_loss_weight": float(uncertainty_loss_weight),
        "risk_loss_weight": float(risk_loss_weight),
        "fall_loss_weight": float(fall_loss_weight),
        "high_risk_loss_weight": float(high_risk_loss_weight),
        "smoothness_weight": float(smoothness_weight),
        "future_aux_loss_weight": float(future_aux_loss_weight),
        "delta_aux_loss_weight": float(delta_aux_loss_weight),
        "event_aux_loss_weight": float(event_aux_loss_weight),
        "event_delta_threshold": float(event_delta_threshold),
        "trajectory_decoder_loss_weight": float(trajectory_decoder_loss_weight),
        "trajectory_loss_weight": float(trajectory_loss_weight),
        "trajectory_delta_weight": float(trajectory_delta_weight),
        "trajectory_centered_weight": float(trajectory_centered_weight),
        "trajectory_range_weight": float(trajectory_range_weight),
        "trajectory_loss_type": str(trajectory_loss_type),
        "trajectory_min_points": int(trajectory_min_points),
        "session_affine_scale_regularization_weight": float(session_affine_scale_regularization_weight),
        "session_affine_bias_regularization_weight": float(session_affine_bias_regularization_weight),
        "calibration_residual_regularization_weight": float(calibration_residual_regularization_weight),
        "low_overprediction_weight": float(low_overprediction_weight),
        "high_underprediction_weight": float(high_underprediction_weight),
        "low_overprediction_threshold": float(low_overprediction_threshold),
        "high_underprediction_threshold": float(high_underprediction_threshold),
        "low_suppressor_gate_loss_weight": float(low_suppressor_gate_loss_weight),
        "low_suppressor_threshold": float(low_suppressor_threshold),
        "low_suppressor_gate_pos_weight": float(low_suppressor_gate_pos_weight),
        "low_suppressor_gate_target_mode": str(low_suppressor_gate_target_mode or "low"),
        "low_suppressor_anchor_threshold": float(low_suppressor_anchor_threshold),
        "low_suppressor_recovery_delta": float(low_suppressor_recovery_delta),
        "low_suppressor_correction_regularization_weight": float(low_suppressor_correction_regularization_weight),
        "anchor_gate_loss_weight": float(anchor_gate_loss_weight),
        "anchor_gate_threshold": float(anchor_gate_threshold),
        "anchor_gate_pos_weight": float(anchor_gate_pos_weight),
        "anchor_break_weight": float(anchor_break_weight),
        "anchor_break_threshold": float(anchor_break_threshold),
        "anchor_break_max_weight": float(anchor_break_max_weight),
        "anchor_break_points": int(anchor_break_points),
        "anchor_break_mean_weight": float(anchor_break_mean_weight),
        "lds_weighting": bool(lds_weight_table is not None),
        "lds_points": int(lds_points),
        "lds_mean_weight": float(lds_mean_weight),
        "lds_min_weight": float(lds_min_weight),
        "lds_max_weight": float(lds_max_weight),
        "transition_weighting": bool(transition_weighting and transition_horizon_steps),
        "transition_horizon_count": int(len(transition_horizon_steps)),
        "transition_horizon_max_steps": int(max(transition_horizon_steps) if transition_horizon_steps else 0),
        "transition_drop_threshold": float(transition_drop_threshold),
        "transition_recovery_threshold": float(transition_recovery_threshold),
        "transition_high_threshold": float(transition_high_threshold),
        "transition_low_threshold": float(transition_low_threshold),
        "transition_rise_threshold": float(transition_rise_threshold),
        "transition_drop_weight": float(transition_drop_weight),
        "transition_recovery_weight": float(transition_recovery_weight),
        "transition_rise_weight": float(transition_rise_weight),
        "transition_max_weight": float(transition_max_weight),
        "transition_points": int(transition_points),
        "transition_drop_points": int(transition_drop_points),
        "transition_recovery_points": int(transition_recovery_points),
        "transition_rise_points": int(transition_rise_points),
        "transition_mean_weight": float(transition_mean_weight),
        "transition_min_weight": float(transition_min_weight),
        "transition_max_observed_weight": float(transition_max_observed_weight),
    }
    for idx, horizon_steps in enumerate(rise_horizon_steps):
        total = float(risk_total_counts[idx]) if idx < len(risk_total_counts) else 0.0
        pos = float(risk_pos_counts[idx]) if idx < len(risk_pos_counts) else 0.0
        parts[f"risk_h{int(horizon_steps)}_prevalence"] = pos / total if total > 0 else 0.0
    for idx, horizon_steps in enumerate(fall_horizon_steps):
        total = float(fall_risk_total_counts[idx]) if idx < len(fall_risk_total_counts) else 0.0
        pos = float(fall_risk_pos_counts[idx]) if idx < len(fall_risk_pos_counts) else 0.0
        parts[f"fall_risk_h{int(horizon_steps)}_prevalence"] = pos / total if total > 0 else 0.0
    for h_idx, horizon_steps in enumerate(high_risk_steps):
        for t_idx, threshold in enumerate(high_risk_thresholds):
            flat_idx = h_idx * len(high_risk_thresholds) + t_idx
            total = float(high_risk_total_counts[flat_idx]) if flat_idx < len(high_risk_total_counts) else 0.0
            pos = float(high_risk_pos_counts[flat_idx]) if flat_idx < len(high_risk_pos_counts) else 0.0
            parts[f"high_risk_h{int(horizon_steps)}_thr{float(threshold):g}_prevalence"] = (
                pos / total if total > 0 else 0.0
            )
    return loss, parts


def _binary_auc(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(s)
    y = y[valid] > 0.5
    s = s[valid]
    pos = int(y.sum())
    neg = int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        rank_sums = np.bincount(inverse, weights=ranks)
        ranks = rank_sums[inverse] / counts[inverse]
    rank_sum_pos = float(ranks[y].sum())
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg)


def _binary_auprc(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(s)
    y = y[valid] > 0.5
    s = s[valid]
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted, dtype=np.float64)
    fp = np.cumsum(~y_sorted, dtype=np.float64)
    recall = tp / float(pos)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapezoid(precision, recall))


def _binary_metrics(y_true: Sequence[float], y_score: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(s)
    if valid.sum() == 0:
        return {
            "n": 0,
            "prevalence": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
        }
    yt = y[valid] > 0.5
    yp = s[valid] >= float(threshold)
    tp = float(np.sum(yp & yt))
    fp = float(np.sum(yp & ~yt))
    fn = float(np.sum(~yp & yt))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "n": int(valid.sum()),
        "prevalence": float(np.mean(yt)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": _binary_auc(yt.astype(float), s[valid]),
        "auprc": _binary_auprc(yt.astype(float), s[valid]),
    }


def _finite_corr(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if a.size < 2 or float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _metric_by_path(metrics: Mapping[str, Any], path: str, default: float = float("nan")) -> float:
    current: Any = metrics
    for token in str(path).split("."):
        if isinstance(current, Mapping) and token in current:
            current = current[token]
        else:
            return float(default)
    try:
        return float(current)
    except (TypeError, ValueError):
        return float(default)


def _slice_signed_bias_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    lower: float,
    upper: float,
    include_upper: bool = False,
) -> Dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    finite = np.isfinite(true) & np.isfinite(pred)
    if include_upper:
        mask = finite & (true >= float(lower)) & (true <= float(upper))
    else:
        mask = finite & (true >= float(lower)) & (true < float(upper))
    if not mask.any():
        return {
            "n": 0,
            "target_mean": float("nan"),
            "pred_mean": float("nan"),
            "signed_bias": float("nan"),
            "mae": float("nan"),
        }
    diff = pred[mask] - true[mask]
    return {
        "n": int(mask.sum()),
        "target_mean": float(np.mean(true[mask])),
        "pred_mean": float(np.mean(pred[mask])),
        "signed_bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
    }


def compute_online_current_goal_selection_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    mae: float,
    r2: float,
    high8_f1: float,
    high12_f1: float,
    low_bias_target: float = 2.5,
    r2_floor: float = 0.70,
    high12_f1_floor: float = 0.76,
) -> Dict[str, Any]:
    """Metrics used to select goal-oriented online-current checkpoints.

    The strict low-FMS target in the goal uses the original [0, 2) bin, not
    the inclusive <=2 helper bin used by older summaries.
    """
    low_0_2 = _slice_signed_bias_metrics(y_true, y_pred, lower=0.0, upper=2.0, include_upper=False)
    low_0_2_inclusive = _slice_signed_bias_metrics(y_true, y_pred, lower=0.0, upper=2.0, include_upper=True)
    low_bias = float(low_0_2["signed_bias"])
    low_penalty = max(0.0, low_bias - float(low_bias_target)) if np.isfinite(low_bias) else 10.0
    r2_penalty = max(0.0, float(r2_floor) - float(r2)) if np.isfinite(float(r2)) else 1.0
    high12_penalty = max(0.0, float(high12_f1_floor) - float(high12_f1)) if np.isfinite(float(high12_f1)) else 1.0
    strict120 = float(mae) + 0.25 * low_penalty + 2.0 * r2_penalty + 0.5 * high12_penalty
    return {
        "low_fms": {
            "0_2": low_0_2,
            "0_2_inclusive": low_0_2_inclusive,
        },
        "goal_composite": {
            "strict120": float(strict120),
            "low_bias_target": float(low_bias_target),
            "r2_floor": float(r2_floor),
            "high12_f1_floor": float(high12_f1_floor),
            "low_bias_penalty": float(low_penalty),
            "r2_penalty": float(r2_penalty),
            "high12_f1_penalty": float(high12_penalty),
            "high8_f1": float(high8_f1),
            "high12_f1": float(high12_f1),
        },
    }


def _selection_mode_for_metric(metric_path: str, explicit_mode: Optional[str] = None) -> str:
    if explicit_mode:
        mode = str(explicit_mode).lower()
        if mode not in {"min", "max"}:
            raise ValueError("training.selection_mode must be 'min' or 'max'.")
        return mode
    metric = str(metric_path).lower()
    return "min" if any(token in metric for token in ("mae", "rmse", "loss", "error", "smape")) else "max"


def _online_current_trajectory_metrics(
    records: Sequence[Mapping[str, Any]],
    horizons_seconds: Sequence[float] = (5.0, 10.0),
    flat_threshold: float = 0.3,
) -> Dict[str, float]:
    by_session: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        session_id = str(record.get("session_id") or record.get("source_file") or "unknown")
        by_session.setdefault(session_id, []).append(record)

    pearson_values: List[float] = []
    centered_mae_values: List[float] = []
    z_mae_values: List[float] = []
    range_ratios: List[float] = []
    flat_failures = 0
    range_sessions = 0
    delta_true_by_horizon: Dict[str, List[float]] = {f"{float(v):g}s": [] for v in horizons_seconds}
    delta_pred_by_horizon: Dict[str, List[float]] = {f"{float(v):g}s": [] for v in horizons_seconds}

    for session_records in by_session.values():
        ordered = sorted(session_records, key=lambda item: float(item.get("current_time", item.get("current_index", 0.0)) or 0.0))
        target = np.asarray([float(item.get("target_fms_now", float("nan"))) for item in ordered], dtype=np.float64)
        pred = np.asarray([float(item.get("predicted_fms_now", float("nan"))) for item in ordered], dtype=np.float64)
        valid = np.isfinite(target) & np.isfinite(pred)
        target = target[valid]
        pred = pred[valid]
        if target.size < 2:
            continue
        corr = _finite_corr(target, pred)
        if np.isfinite(corr):
            pearson_values.append(float(corr))
        centered_mae_values.append(float(np.mean(np.abs((pred - np.mean(pred)) - (target - np.mean(target))))))
        target_std = float(np.std(target))
        pred_std = float(np.std(pred))
        if target_std > 1e-12 and pred_std > 1e-12:
            z_true = (target - np.mean(target)) / target_std
            z_pred = (pred - np.mean(pred)) / pred_std
            z_mae_values.append(float(np.mean(np.abs(z_pred - z_true))))
        target_range = float(np.max(target) - np.min(target))
        pred_range = float(np.max(pred) - np.min(pred))
        if target_range > 1e-12:
            ratio = pred_range / target_range
            range_ratios.append(float(ratio))
            range_sessions += 1
            if ratio < 0.25:
                flat_failures += 1

        interval_values = [
            float(item.get("sampling_interval", float("nan")))
            for item in ordered
            if np.isfinite(float(item.get("sampling_interval", float("nan"))))
        ]
        interval = interval_values[0] if interval_values else 0.5
        for horizon in horizons_seconds:
            step = max(1, int(round(float(horizon) / max(float(interval), 1e-8))))
            if target.size <= step:
                continue
            key = f"{float(horizon):g}s"
            delta_true_by_horizon[key].extend((target[step:] - target[:-step]).tolist())
            delta_pred_by_horizon[key].extend((pred[step:] - pred[:-step]).tolist())

    def _nanmean(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else float("nan")

    metrics: Dict[str, float] = {
        "pearson_session_mean": _nanmean(pearson_values),
        "centered_mae_session_mean": _nanmean(centered_mae_values),
        "shape_z_mae_session_mean": _nanmean(z_mae_values),
        "pred_true_range_ratio_mean": _nanmean(range_ratios),
        "flat_range_lt25pct_session_rate": float(flat_failures / range_sessions) if range_sessions else float("nan"),
        "session_count": float(len(by_session)),
    }
    for horizon in horizons_seconds:
        key = f"{float(horizon):g}s"
        dt = np.asarray(delta_true_by_horizon[key], dtype=np.float64)
        dp = np.asarray(delta_pred_by_horizon[key], dtype=np.float64)
        valid = np.isfinite(dt) & np.isfinite(dp)
        dt = dt[valid]
        dp = dp[valid]
        if dt.size:
            move = np.abs(dt) >= float(flat_threshold)
            rise = dt >= float(flat_threshold)
            drop = dt <= -float(flat_threshold)
            pred_sign = np.sign(dp)
            true_sign = np.sign(dt)
            metrics[f"delta_mae_{key}"] = float(np.mean(np.abs(dp - dt)))
            metrics[f"delta_corr_{key}"] = _finite_corr(dt, dp)
            metrics[f"direction_acc_{key}"] = float(np.mean(pred_sign[move] == true_sign[move])) if move.any() else float("nan")
            metrics[f"direction_acc_rise_{key}"] = float(np.mean(pred_sign[rise] > 0)) if rise.any() else float("nan")
            metrics[f"direction_acc_drop_{key}"] = float(np.mean(pred_sign[drop] < 0)) if drop.any() else float("nan")
        else:
            metrics[f"delta_mae_{key}"] = float("nan")
            metrics[f"delta_corr_{key}"] = float("nan")
            metrics[f"direction_acc_{key}"] = float("nan")
            metrics[f"direction_acc_rise_{key}"] = float("nan")
            metrics[f"direction_acc_drop_{key}"] = float("nan")
    return metrics


@torch.no_grad()
def collect_online_current_risk_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    calibration_steps: int,
    fms_scaler: Mapping[str, float],
    rise_horizon_steps: Sequence[int],
    rise_thresholds: Sequence[float],
    ordinal_bins: Sequence[float],
    fall_horizon_steps: Optional[Sequence[int]] = None,
    fall_thresholds: Optional[Sequence[float]] = None,
    high_risk_horizon_steps: Optional[Sequence[int]] = None,
    high_risk_thresholds: Optional[Sequence[float]] = None,
    high_risk_label_mode: str = "future_any",
    high_risk_onset_past_steps: int = 0,
    high_fms_caution_threshold: float = 8.0,
    high_fms_warning_threshold: float = 12.0,
    rapid_rise_probability_threshold: float = 0.5,
    rapid_drop_probability_threshold: Optional[float] = None,
    final_warning_mode: str = "high_or_rapid",
    use_static: bool = False,
    calibration_seconds: Optional[float] = None,
    recent_window_seconds: Optional[float] = None,
    sampling_interval: float = 0.5,
    recent_window_steps: Optional[int] = None,
    run_name: Optional[str] = None,
    model_name: Optional[str] = None,
    split_name: Optional[str] = None,
    max_eval_batches: Optional[int] = None,
    future_aux_horizon_steps: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    model.eval()
    f_min = float(fms_scaler["min"])
    f_max = float(fms_scaler["max"])
    f_range = max(f_max - f_min, 1e-8)
    thresholds_norm = [float(v) / f_range for v in rise_thresholds]
    fall_horizon_steps = [int(v) for v in (fall_horizon_steps or rise_horizon_steps)]
    fall_thresholds = [float(v) for v in (fall_thresholds or rise_thresholds)]
    if len(fall_thresholds) != len(fall_horizon_steps):
        raise ValueError("fall_thresholds must have the same length as fall_horizon_steps.")
    fall_thresholds_norm = [float(v) / f_range for v in fall_thresholds]
    high_risk_horizon_steps = [int(v) for v in (high_risk_horizon_steps or [])]
    high_risk_thresholds = [float(v) for v in (high_risk_thresholds or [])]
    high_risk_thresholds_norm = [float(v - f_min) / f_range for v in high_risk_thresholds]
    bins_norm = [float(v - f_min) / f_range for v in ordinal_bins]
    ordinal_bins_raw = np.asarray([float(v) for v in ordinal_bins], dtype=np.float64)
    rapid_drop_probability_threshold = (
        float(rapid_drop_probability_threshold)
        if rapid_drop_probability_threshold is not None
        else float(rapid_rise_probability_threshold)
    )
    future_aux_horizon_steps = [int(v) for v in (future_aux_horizon_steps or [])]
    y_true: List[float] = []
    y_pred: List[float] = []
    y_pred_regression: List[float] = []
    y_pred_ordinal_value: List[float] = []
    y_pred_ordinal_hard_value: List[float] = []
    y_pred_sigma: List[float] = []
    ordinal_true: List[int] = []
    ordinal_pred: List[int] = []
    risk_true: Dict[str, List[float]] = {f"{int(h)}": [] for h in rise_horizon_steps}
    risk_score: Dict[str, List[float]] = {f"{int(h)}": [] for h in rise_horizon_steps}
    fall_true: Dict[str, List[float]] = {f"{int(h)}": [] for h in fall_horizon_steps}
    fall_score: Dict[str, List[float]] = {f"{int(h)}": [] for h in fall_horizon_steps}
    high_risk_true: Dict[str, List[float]] = {
        f"{int(h)}_{float(threshold):g}": [] for h in high_risk_horizon_steps for threshold in high_risk_thresholds
    }
    high_risk_score: Dict[str, List[float]] = {
        f"{int(h)}_{float(threshold):g}": [] for h in high_risk_horizon_steps for threshold in high_risk_thresholds
    }
    future_true: Dict[str, List[float]] = {f"{int(h)}": [] for h in future_aux_horizon_steps}
    future_pred: Dict[str, List[float]] = {f"{int(h)}": [] for h in future_aux_horizon_steps}
    delta_true: Dict[str, List[float]] = {f"{int(h)}": [] for h in future_aux_horizon_steps}
    delta_pred: Dict[str, List[float]] = {f"{int(h)}": [] for h in future_aux_horizon_steps}
    event_true: Dict[str, List[int]] = {f"{int(h)}": [] for h in future_aux_horizon_steps}
    event_pred: Dict[str, List[int]] = {f"{int(h)}": [] for h in future_aux_horizon_steps}
    rapid_any_true: List[float] = []
    rapid_any_score: List[float] = []
    rapid_any_alarm: List[bool] = []
    drop_any_true: List[float] = []
    drop_any_score: List[float] = []
    drop_any_alarm: List[bool] = []
    final_true: List[bool] = []
    final_alarm: List[bool] = []
    plot_series: List[Dict[str, Any]] = []
    prediction_records: List[Dict[str, Any]] = []
    final_warning_mode = str(final_warning_mode or "high_or_rapid").lower()
    if final_warning_mode not in {"high_or_rapid", "rapid_rise_only"}:
        raise ValueError("final_warning_mode must be 'high_or_rapid' or 'rapid_rise_only'.")

    for batch_idx, batch in enumerate(loader):
        if max_eval_batches is not None and batch_idx >= int(max_eval_batches):
            break
        head = batch["head"].to(device)
        fms = batch["fms"].to(device)
        lengths = batch["lengths"].to(device)
        static = batch.get("static")
        if use_static:
            if static is None:
                raise ValueError("Model was configured with use_static=True, but batch['static'] is missing.")
            static = static.to(device)
        residual_features = batch.get("calibration_residual_features")
        residual_feature_mask = batch.get("calibration_residual_feature_mask")
        residual_adapter_enabled = bool(getattr(model, "calibration_residual_adapter_enabled", False))
        summary_fusion_enabled = bool(getattr(model, "calibration_summary_fusion_enabled", False))
        residual_feature_dependent = residual_adapter_enabled or summary_fusion_enabled
        if residual_feature_dependent:
            if residual_features is None:
                raise ValueError("Calibration residual-feature evaluation requested but batch['calibration_residual_features'] is missing.")
            residual_features = residual_features.to(device)
            residual_feature_mask = residual_feature_mask.to(device) if residual_feature_mask is not None else None
        model_kwargs: Dict[str, Any] = {"static": static}
        if residual_feature_dependent:
            model_kwargs["calibration_residual_features"] = residual_features
            model_kwargs["calibration_residual_feature_mask"] = residual_feature_mask
        outputs = model(head, fms[:, :calibration_steps], lengths, **model_kwargs)
        current_pred = outputs["current"].to(device)
        prediction_start = _tensor_scalar_int(outputs.get("prediction_start"), calibration_steps)
        targets = compute_online_current_risk_targets(
            fms,
            lengths,
            prediction_start,
            int(current_pred.shape[1]),
            rise_horizon_steps,
            thresholds_norm,
            bins_norm,
            fall_horizon_steps=fall_horizon_steps,
            fall_thresholds_normalized=fall_thresholds_norm,
            high_risk_horizon_steps=high_risk_horizon_steps,
            high_risk_thresholds_normalized=high_risk_thresholds_norm,
            high_risk_label_mode=high_risk_label_mode,
            high_risk_onset_past_steps=high_risk_onset_past_steps,
            future_horizon_steps=future_aux_horizon_steps,
        )
        mask = outputs["mask"].to(device).bool() & targets["current_mask"].to(device)
        current_target = targets["current"].to(device)
        current_pred_raw = denormalize_fms(current_pred, fms_scaler)
        current_target_raw = denormalize_fms(current_target, fms_scaler)
        current_reg = outputs.get("current_reg")
        current_ordinal = outputs.get("current_ordinal")
        current_sigma = outputs.get("current_sigma")
        current_pre_session_affine = outputs.get("current_pre_session_affine")
        current_session_affine_scale = outputs.get("current_session_affine_scale")
        current_session_affine_bias = outputs.get("current_session_affine_bias")
        current_pre_affine = outputs.get("current_pre_affine")
        current_affine_scale = outputs.get("current_affine_scale")
        current_affine_bias = outputs.get("current_affine_bias")
        current_pre_binned_affine = outputs.get("current_pre_binned_affine")
        current_binned_affine_scale = outputs.get("current_binned_affine_scale")
        current_binned_affine_bias = outputs.get("current_binned_affine_bias")
        current_binned_affine_bin = outputs.get("current_binned_affine_bin")
        current_pre_residual_adapter = outputs.get("current_pre_residual_adapter")
        current_residual_adapter_correction = outputs.get("current_residual_adapter_correction")
        current_residual_adapter_gate = outputs.get("current_residual_adapter_gate")
        current_pre_low_suppressor = outputs.get("current_pre_low_suppressor")
        current_low_suppressor_correction = outputs.get("current_low_suppressor_correction")
        current_low_suppressor_gate = outputs.get("current_low_suppressor_gate")
        current_calib_prior_gate = outputs.get("current_calib_prior_gate")
        current_calib_prior_cap = outputs.get("current_calib_prior_cap")
        current_calib_prior_capped_value = outputs.get("current_calib_prior_capped_value")
        current_reg_raw = (
            denormalize_fms(current_reg.to(device), fms_scaler)
            if isinstance(current_reg, torch.Tensor)
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_ordinal_raw = (
            denormalize_fms(current_ordinal.to(device), fms_scaler)
            if isinstance(current_ordinal, torch.Tensor)
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_sigma_raw = (
            current_sigma.to(device).detach().cpu().numpy() * f_range
            if isinstance(current_sigma, torch.Tensor) and current_sigma.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_pre_session_affine_raw = (
            denormalize_fms(current_pre_session_affine.to(device), fms_scaler)
            if isinstance(current_pre_session_affine, torch.Tensor) and current_pre_session_affine.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )

        def _session_affine_array(value: Any, multiplier: float = 1.0) -> np.ndarray:
            if not isinstance(value, torch.Tensor):
                return np.full_like(current_pred_raw, float("nan"))
            arr = value.to(device).detach().cpu().numpy() * float(multiplier)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.shape == current_pred_raw.shape:
                return arr
            if arr.shape == (current_pred_raw.shape[0], 1):
                return np.broadcast_to(arr, current_pred_raw.shape)
            return np.full_like(current_pred_raw, float("nan"))

        current_session_affine_scale_np = _session_affine_array(current_session_affine_scale)
        current_session_affine_bias_raw = _session_affine_array(current_session_affine_bias, multiplier=f_range)
        current_pre_affine_raw = (
            denormalize_fms(current_pre_affine.to(device), fms_scaler)
            if isinstance(current_pre_affine, torch.Tensor) and current_pre_affine.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_affine_scale_np = (
            current_affine_scale.to(device).detach().cpu().numpy()
            if isinstance(current_affine_scale, torch.Tensor) and current_affine_scale.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_affine_bias_raw = (
            current_affine_bias.to(device).detach().cpu().numpy() * f_range
            if isinstance(current_affine_bias, torch.Tensor) and current_affine_bias.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_pre_binned_affine_raw = (
            denormalize_fms(current_pre_binned_affine.to(device), fms_scaler)
            if isinstance(current_pre_binned_affine, torch.Tensor) and current_pre_binned_affine.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_binned_affine_scale_np = (
            current_binned_affine_scale.to(device).detach().cpu().numpy()
            if isinstance(current_binned_affine_scale, torch.Tensor) and current_binned_affine_scale.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_binned_affine_bias_raw = (
            current_binned_affine_bias.to(device).detach().cpu().numpy() * f_range
            if isinstance(current_binned_affine_bias, torch.Tensor) and current_binned_affine_bias.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_binned_affine_bin_np = (
            current_binned_affine_bin.to(device).detach().cpu().numpy()
            if isinstance(current_binned_affine_bin, torch.Tensor) and current_binned_affine_bin.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_pre_residual_adapter_raw = (
            denormalize_fms(current_pre_residual_adapter.to(device), fms_scaler)
            if isinstance(current_pre_residual_adapter, torch.Tensor) and current_pre_residual_adapter.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_residual_adapter_correction_raw = (
            current_residual_adapter_correction.to(device).detach().cpu().numpy() * f_range
            if isinstance(current_residual_adapter_correction, torch.Tensor)
            and current_residual_adapter_correction.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_residual_adapter_gate_np = (
            current_residual_adapter_gate.to(device).detach().cpu().numpy()
            if isinstance(current_residual_adapter_gate, torch.Tensor)
            and current_residual_adapter_gate.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_pre_low_suppressor_raw = (
            denormalize_fms(current_pre_low_suppressor.to(device), fms_scaler)
            if isinstance(current_pre_low_suppressor, torch.Tensor) and current_pre_low_suppressor.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_low_suppressor_correction_raw = (
            current_low_suppressor_correction.to(device).detach().cpu().numpy() * f_range
            if isinstance(current_low_suppressor_correction, torch.Tensor)
            and current_low_suppressor_correction.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_low_suppressor_gate_np = (
            current_low_suppressor_gate.to(device).detach().cpu().numpy()
            if isinstance(current_low_suppressor_gate, torch.Tensor)
            and current_low_suppressor_gate.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_calib_prior_gate_np = (
            current_calib_prior_gate.to(device).detach().cpu().numpy()
            if isinstance(current_calib_prior_gate, torch.Tensor)
            and current_calib_prior_gate.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_calib_prior_cap_raw = (
            denormalize_fms(_session_affine_array(current_calib_prior_cap), fms_scaler)
            if isinstance(current_calib_prior_cap, torch.Tensor)
            else np.full_like(current_pred_raw, float("nan"))
        )
        current_calib_prior_capped_value_raw = (
            denormalize_fms(current_calib_prior_capped_value.to(device), fms_scaler)
            if isinstance(current_calib_prior_capped_value, torch.Tensor)
            and current_calib_prior_capped_value.shape == current_pred.shape
            else np.full_like(current_pred_raw, float("nan"))
        )
        mask_np = mask.detach().cpu().numpy()
        pred_np = current_pred_raw
        target_np = current_target_raw
        pred_reg_np = current_reg_raw
        pred_ordinal_np = current_ordinal_raw
        positions_np = targets["positions"].detach().cpu().numpy()
        time_np = batch["time"].detach().cpu().numpy()
        lengths_np = lengths.detach().cpu().numpy()
        risk_probs_np = outputs["risk_probs"].detach().cpu().numpy()
        risk_labels_np = targets["rise_labels"].detach().cpu().numpy()
        risk_mask_np = targets["rise_mask"].detach().cpu().numpy()
        fall_probs = outputs.get("fall_risk_probs")
        fall_probs_np = (
            fall_probs.detach().cpu().numpy()
            if isinstance(fall_probs, torch.Tensor)
            else np.zeros((current_pred.shape[0], current_pred.shape[1], len(fall_horizon_steps)), dtype=np.float64)
        )
        fall_labels_np = targets["fall_labels"].detach().cpu().numpy()
        fall_mask_np = targets["fall_mask"].detach().cpu().numpy()
        high_risk_probs = outputs.get("high_risk_probs")
        high_risk_probs_np = (
            high_risk_probs.detach().cpu().numpy()
            if isinstance(high_risk_probs, torch.Tensor)
            else np.zeros(
                (current_pred.shape[0], current_pred.shape[1], len(high_risk_horizon_steps), len(high_risk_thresholds)),
                dtype=np.float64,
            )
        )
        high_risk_labels_np = targets["high_risk_labels"].detach().cpu().numpy()
        high_risk_mask_np = targets["high_risk_mask"].detach().cpu().numpy()
        ordinal_labels_np = targets["ordinal_labels"].detach().cpu().numpy()
        ordinal_probs_np = outputs["ordinal_probs"].detach().cpu().numpy()
        ordinal_pred_np = np.argmax(ordinal_probs_np, axis=-1)
        ordinal_pred_clipped_np = np.clip(ordinal_pred_np, 0, max(len(ordinal_bins_raw) - 1, 0))
        pred_ordinal_hard_np = (
            ordinal_bins_raw[ordinal_pred_clipped_np]
            if len(ordinal_bins_raw) > 0
            else np.full_like(pred_ordinal_np, float("nan"), dtype=np.float64)
        )
        future_aux_np = outputs.get("future_aux")
        future_aux_np = (
            denormalize_fms(future_aux_np.to(device), fms_scaler)
            if isinstance(future_aux_np, torch.Tensor)
            else np.zeros((current_pred.shape[0], current_pred.shape[1], 0), dtype=np.float64)
        )
        future_target_np = denormalize_fms(targets["future"].to(device), fms_scaler)
        future_delta_target_np = targets["future_delta"].detach().cpu().numpy() * f_range
        future_delta_pred_np = future_aux_np - pred_np[..., None]
        future_mask_np = targets["future_mask"].detach().cpu().numpy()
        event_labels_np = targets["event_labels"].detach().cpu().numpy()
        event_probs = outputs.get("event_probs")
        event_probs_np = (
            event_probs.detach().cpu().numpy()
            if isinstance(event_probs, torch.Tensor)
            else np.zeros((current_pred.shape[0], current_pred.shape[1], 0, 3), dtype=np.float64)
        )
        event_pred_np = np.argmax(event_probs_np, axis=-1) if event_probs_np.shape[-1] else np.zeros_like(event_labels_np)
        fms_raw_np = batch.get("fms_raw", fms).detach().cpu().numpy()

        y_true.extend(target_np[mask_np].tolist())
        y_pred.extend(pred_np[mask_np].tolist())
        y_pred_regression.extend(pred_reg_np[mask_np].tolist())
        y_pred_ordinal_value.extend(pred_ordinal_np[mask_np].tolist())
        y_pred_ordinal_hard_value.extend(pred_ordinal_hard_np[mask_np].tolist())
        y_pred_sigma.extend(current_sigma_raw[mask_np].tolist())
        ordinal_true.extend(ordinal_labels_np[mask_np].astype(int).tolist())
        ordinal_pred.extend(ordinal_pred_np[mask_np].astype(int).tolist())
        for h_idx, horizon_steps in enumerate(rise_horizon_steps):
            key = f"{int(horizon_steps)}"
            h_mask = risk_mask_np[..., h_idx] & mask_np
            risk_true[key].extend(risk_labels_np[..., h_idx][h_mask].tolist())
            risk_score[key].extend(risk_probs_np[..., h_idx][h_mask].tolist())
        for h_idx, horizon_steps in enumerate(fall_horizon_steps):
            key = f"{int(horizon_steps)}"
            h_mask = fall_mask_np[..., h_idx] & mask_np
            fall_true[key].extend(fall_labels_np[..., h_idx][h_mask].tolist())
            fall_score[key].extend(fall_probs_np[..., h_idx][h_mask].tolist())
        for h_idx, horizon_steps in enumerate(high_risk_horizon_steps):
            for t_idx, threshold in enumerate(high_risk_thresholds):
                key = f"{int(horizon_steps)}_{float(threshold):g}"
                h_mask = high_risk_mask_np[..., h_idx, t_idx] & mask_np
                high_risk_true[key].extend(high_risk_labels_np[..., h_idx, t_idx][h_mask].tolist())
                high_risk_score[key].extend(high_risk_probs_np[..., h_idx, t_idx][h_mask].tolist())
        for h_idx, horizon_steps in enumerate(future_aux_horizon_steps):
            key = f"{int(horizon_steps)}"
            h_mask = future_mask_np[..., h_idx] & mask_np
            future_true[key].extend(future_target_np[..., h_idx][h_mask].tolist())
            future_pred[key].extend(future_aux_np[..., h_idx][h_mask].tolist())
            delta_true[key].extend(future_delta_target_np[..., h_idx][h_mask].tolist())
            delta_pred[key].extend(future_delta_pred_np[..., h_idx][h_mask].tolist())
            event_true[key].extend(event_labels_np[..., h_idx][h_mask].astype(int).tolist())
            event_pred[key].extend(event_pred_np[..., h_idx][h_mask].astype(int).tolist())

        for b in range(mask_np.shape[0]):
            valid_j = np.where(mask_np[b])[0]
            if len(valid_j) and len(plot_series) < 12:
                plot_series.append(
                    {
                        "metadata": batch["metadata"][b],
                        "target_time": [float(time_np[b, int(positions_np[j])]) for j in valid_j],
                        "target": target_np[b, valid_j].tolist(),
                        "prediction": pred_np[b, valid_j].tolist(),
                        "prediction_regression": pred_reg_np[b, valid_j].tolist(),
                        "prediction_ordinal": pred_ordinal_np[b, valid_j].tolist(),
                        "prediction_ordinal_hard": pred_ordinal_hard_np[b, valid_j].tolist(),
                    }
                )
            meta = batch["metadata"][b]
            session_length = int(lengths_np[b])
            for j in valid_j:
                current_index = int(positions_np[j])
                current_time = float(time_np[b, current_index]) if np.isfinite(time_np[b, current_index]) else float("nan")
                true_value = float(target_np[b, j])
                pred_value = float(pred_np[b, j])
                pred_reg_value = float(pred_reg_np[b, j])
                pred_ordinal_value = float(pred_ordinal_np[b, j])
                pred_ordinal_hard_value = float(pred_ordinal_hard_np[b, j])
                pred_sigma_value = float(current_sigma_raw[b, j])
                pre_session_affine_value = float(current_pre_session_affine_raw[b, j])
                session_affine_scale_value = float(current_session_affine_scale_np[b, j])
                session_affine_bias_value = float(current_session_affine_bias_raw[b, j])
                pre_affine_value = float(current_pre_affine_raw[b, j])
                affine_scale_value = float(current_affine_scale_np[b, j])
                affine_bias_value = float(current_affine_bias_raw[b, j])
                pre_binned_affine_value = float(current_pre_binned_affine_raw[b, j])
                binned_affine_scale_value = float(current_binned_affine_scale_np[b, j])
                binned_affine_bias_value = float(current_binned_affine_bias_raw[b, j])
                binned_affine_bin_value = float(current_binned_affine_bin_np[b, j])
                pre_residual_adapter_value = float(current_pre_residual_adapter_raw[b, j])
                residual_adapter_correction_value = float(current_residual_adapter_correction_raw[b, j])
                residual_adapter_gate_value = float(current_residual_adapter_gate_np[b, j])
                pre_low_suppressor_value = float(current_pre_low_suppressor_raw[b, j])
                low_suppressor_correction_value = float(current_low_suppressor_correction_raw[b, j])
                low_suppressor_gate_value = float(current_low_suppressor_gate_np[b, j])
                calib_prior_gate_value = float(current_calib_prior_gate_np[b, j])
                calib_prior_cap_value = float(current_calib_prior_cap_raw[b, j])
                calib_prior_capped_value = float(current_calib_prior_capped_value_raw[b, j])
                rapid_flags = [
                    bool(
                        h_idx < risk_probs_np.shape[-1]
                        and h_idx < risk_mask_np.shape[-1]
                        and risk_mask_np[b, j, h_idx]
                        and risk_probs_np[b, j, h_idx] >= float(rapid_rise_probability_threshold)
                    )
                    for h_idx in range(len(rise_horizon_steps))
                ]
                rapid_valid_labels = [
                    bool(h_idx < risk_mask_np.shape[-1] and risk_mask_np[b, j, h_idx])
                    for h_idx in range(len(rise_horizon_steps))
                ]
                rapid_label_values = [
                    bool(risk_labels_np[b, j, h_idx] > 0.5) if rapid_valid_labels[h_idx] else False
                    for h_idx in range(len(rise_horizon_steps))
                ]
                rapid_score_values = [
                    float(risk_probs_np[b, j, h_idx]) if h_idx < risk_probs_np.shape[-1] and rapid_valid_labels[h_idx] else float("nan")
                    for h_idx in range(len(rise_horizon_steps))
                ]
                drop_flags = [
                    bool(
                        h_idx < fall_probs_np.shape[-1]
                        and h_idx < fall_mask_np.shape[-1]
                        and fall_mask_np[b, j, h_idx]
                        and fall_probs_np[b, j, h_idx] >= float(rapid_drop_probability_threshold)
                    )
                    for h_idx in range(len(fall_horizon_steps))
                ]
                drop_valid_labels = [
                    bool(h_idx < fall_mask_np.shape[-1] and fall_mask_np[b, j, h_idx])
                    for h_idx in range(len(fall_horizon_steps))
                ]
                drop_label_values = [
                    bool(fall_labels_np[b, j, h_idx] > 0.5) if drop_valid_labels[h_idx] else False
                    for h_idx in range(len(fall_horizon_steps))
                ]
                drop_score_values = [
                    float(fall_probs_np[b, j, h_idx]) if h_idx < fall_probs_np.shape[-1] and drop_valid_labels[h_idx] else float("nan")
                    for h_idx in range(len(fall_horizon_steps))
                ]
                high_risk_entries: List[Tuple[float, float, bool, Optional[int], float]] = []
                for h_idx, horizon_steps in enumerate(high_risk_horizon_steps):
                    horizon_seconds = float(horizon_steps) * float(sampling_interval)
                    for t_idx, threshold in enumerate(high_risk_thresholds):
                        label_valid = bool(
                            h_idx < high_risk_mask_np.shape[-2]
                            and t_idx < high_risk_mask_np.shape[-1]
                            and high_risk_mask_np[b, j, h_idx, t_idx]
                        )
                        label_value = (
                            int(high_risk_labels_np[b, j, h_idx, t_idx])
                            if label_valid
                            else None
                        )
                        score_value = (
                            float(high_risk_probs_np[b, j, h_idx, t_idx])
                            if h_idx < high_risk_probs_np.shape[-2] and t_idx < high_risk_probs_np.shape[-1]
                            else float("nan")
                        )
                        high_risk_entries.append(
                            (horizon_seconds, float(threshold), label_valid, label_value, score_value)
                        )
                warning_high = pred_value >= float(high_fms_warning_threshold)
                warning_rapid = any(rapid_flags)
                true_high = true_value >= float(high_fms_warning_threshold)
                true_rapid = any(rapid_label_values)
                warning_drop = any(drop_flags)
                true_drop = any(drop_label_values)
                high_risk_warning_scores = [
                    score
                    for _horizon_seconds, threshold, label_valid, _label_value, score in high_risk_entries
                    if label_valid and threshold >= float(high_fms_warning_threshold) and np.isfinite(score)
                ]
                high_risk_warning_labels = [
                    bool(label_value)
                    for _horizon_seconds, threshold, label_valid, label_value, _score in high_risk_entries
                    if label_valid and threshold >= float(high_fms_warning_threshold) and label_value is not None
                ]
                warning_high_future = any(
                    score >= float(rapid_rise_probability_threshold) for score in high_risk_warning_scores
                )
                true_high_future = any(high_risk_warning_labels)
                if final_warning_mode == "rapid_rise_only":
                    final_label = true_rapid
                    final_pred = warning_rapid
                else:
                    final_label = true_high or true_high_future or true_rapid
                    final_pred = warning_high or warning_high_future or warning_rapid
                rapid_any_true.append(float(true_rapid))
                rapid_any_score.append(float(np.nanmax(rapid_score_values)) if any(np.isfinite(rapid_score_values)) else float("nan"))
                rapid_any_alarm.append(bool(warning_rapid))
                drop_any_true.append(float(true_drop))
                drop_any_score.append(float(np.nanmax(drop_score_values)) if any(np.isfinite(drop_score_values)) else float("nan"))
                drop_any_alarm.append(bool(warning_drop))
                final_true.append(bool(final_label))
                final_alarm.append(bool(final_pred))
                record: Dict[str, Any] = {
                    "run_name": run_name,
                    "model_name": model_name,
                    "split": split_name,
                    "participant_id": meta.get("participant_id"),
                    "session_id": meta.get("session_id"),
                    "source_file": meta.get("source_file"),
                    "age": meta.get("age"),
                    "gender": meta.get("gender"),
                    "mssq": meta.get("mssq"),
                    "current_index": current_index,
                    "current_time": current_time,
                    "session_length_steps": session_length,
                    "calibration_seconds": calibration_seconds,
                    "recent_window_seconds": recent_window_seconds,
                    "sampling_interval": float(sampling_interval),
                    "calibration_steps": int(calibration_steps),
                    "recent_window_steps": int(recent_window_steps if recent_window_steps is not None else 0),
                    "anchor_mode": "calibration_only",
                    "fms_context_mode": "calibration_history",
                    "anchor_index": int(calibration_steps) - 1,
                    "anchor_time": (
                        float(time_np[b, int(calibration_steps) - 1])
                        if int(calibration_steps) - 1 < time_np.shape[1] and np.isfinite(time_np[b, int(calibration_steps) - 1])
                        else float("nan")
                    ),
                    "anchor_fms": (
                        float(fms_raw_np[b, int(calibration_steps) - 1])
                        if int(calibration_steps) - 1 < fms_raw_np.shape[1] and np.isfinite(fms_raw_np[b, int(calibration_steps) - 1])
                        else float("nan")
                    ),
                    "use_static": bool(use_static),
                    "target_fms_now": true_value,
                    "predicted_fms_now": pred_value,
                    "predicted_fms_regression": pred_reg_value,
                    "predicted_fms_ordinal": pred_ordinal_value,
                    "predicted_fms_ordinal_hard": pred_ordinal_hard_value,
                    "predicted_fms_sigma": pred_sigma_value,
                    "predicted_fms_pre_session_affine": pre_session_affine_value,
                    "current_session_affine_scale": session_affine_scale_value,
                    "current_session_affine_bias": session_affine_bias_value,
                    "predicted_fms_pre_affine": pre_affine_value,
                    "current_affine_scale": affine_scale_value,
                    "current_affine_bias": affine_bias_value,
                    "predicted_fms_pre_binned_affine": pre_binned_affine_value,
                    "current_binned_affine_scale": binned_affine_scale_value,
                    "current_binned_affine_bias": binned_affine_bias_value,
                    "current_binned_affine_bin": binned_affine_bin_value,
                    "predicted_fms_pre_residual_adapter": pre_residual_adapter_value,
                    "current_residual_adapter_correction": residual_adapter_correction_value,
                    "current_residual_adapter_gate": residual_adapter_gate_value,
                    "predicted_fms_pre_low_suppressor": pre_low_suppressor_value,
                    "current_low_suppressor_correction": low_suppressor_correction_value,
                    "current_low_suppressor_gate": low_suppressor_gate_value,
                    "current_calib_prior_gate": calib_prior_gate_value,
                    "current_calib_prior_cap": calib_prior_cap_value,
                    "current_calib_prior_capped_value": calib_prior_capped_value,
                    "fms_absolute_error": abs(pred_value - true_value),
                    "fms_regression_absolute_error": abs(pred_reg_value - true_value),
                    "fms_ordinal_absolute_error": abs(pred_ordinal_value - true_value),
                    "fms_ordinal_hard_absolute_error": abs(pred_ordinal_hard_value - true_value),
                    "ordinal_bin_true": int(ordinal_labels_np[b, j]),
                    "ordinal_bin_pred": int(ordinal_pred_np[b, j]),
                    "alarm_caution": bool(pred_value >= float(high_fms_caution_threshold)),
                    "alarm_warning_high_fms": bool(warning_high),
                    "alarm_warning_high_risk": bool(warning_high_future),
                    "alarm_warning_rapid_rise": bool(warning_rapid),
                    "alarm_warning_rapid_drop": bool(warning_drop),
                    "final_warning_mode": final_warning_mode,
                    "final_warning": bool(final_pred),
                    "final_warning_label": bool(final_label),
                    "high_risk_label_mode": str(high_risk_label_mode or "future_any"),
                    "high_risk_onset_past_steps": int(max(high_risk_onset_past_steps, 0)),
                }
                for h_idx, horizon_steps in enumerate(rise_horizon_steps):
                    horizon_seconds = float(horizon_steps) * float(sampling_interval)
                    label_valid = bool(h_idx < risk_mask_np.shape[-1] and risk_mask_np[b, j, h_idx])
                    record[f"p_rapid_rise_{horizon_seconds:g}s"] = (
                        float(risk_probs_np[b, j, h_idx]) if h_idx < risk_probs_np.shape[-1] else None
                    )
                    record[f"rapid_rise_label_{horizon_seconds:g}s"] = (
                        int(risk_labels_np[b, j, h_idx]) if label_valid else None
                    )
                    record[f"rapid_rise_valid_{horizon_seconds:g}s"] = label_valid
                for h_idx, horizon_steps in enumerate(fall_horizon_steps):
                    horizon_seconds = float(horizon_steps) * float(sampling_interval)
                    label_valid = bool(h_idx < fall_mask_np.shape[-1] and fall_mask_np[b, j, h_idx])
                    record[f"p_rapid_drop_{horizon_seconds:g}s"] = (
                        float(fall_probs_np[b, j, h_idx]) if h_idx < fall_probs_np.shape[-1] else None
                    )
                    record[f"rapid_drop_label_{horizon_seconds:g}s"] = (
                        int(fall_labels_np[b, j, h_idx]) if label_valid else None
                    )
                    record[f"rapid_drop_valid_{horizon_seconds:g}s"] = label_valid
                for horizon_seconds, threshold, label_valid, label_value, score_value in high_risk_entries:
                    suffix = f"{horizon_seconds:g}s_thr{threshold:g}"
                    record[f"p_high_risk_{suffix}"] = score_value
                    record[f"high_risk_label_{suffix}"] = label_value if label_valid else None
                    record[f"high_risk_valid_{suffix}"] = label_valid
                for h_idx, horizon_steps in enumerate(future_aux_horizon_steps):
                    horizon_seconds = float(horizon_steps) * float(sampling_interval)
                    aux_valid = bool(h_idx < future_mask_np.shape[-1] and future_mask_np[b, j, h_idx] and mask_np[b, j])
                    record[f"future_aux_target_{horizon_seconds:g}s"] = (
                        float(future_target_np[b, j, h_idx]) if aux_valid else None
                    )
                    record[f"future_aux_pred_{horizon_seconds:g}s"] = (
                        float(future_aux_np[b, j, h_idx]) if h_idx < future_aux_np.shape[-1] and aux_valid else None
                    )
                    record[f"delta_aux_target_{horizon_seconds:g}s"] = (
                        float(future_delta_target_np[b, j, h_idx]) if aux_valid else None
                    )
                    record[f"delta_aux_pred_{horizon_seconds:g}s"] = (
                        float(future_delta_pred_np[b, j, h_idx]) if h_idx < future_delta_pred_np.shape[-1] and aux_valid else None
                    )
                    record[f"future_aux_valid_{horizon_seconds:g}s"] = aux_valid
                    record[f"delta_aux_valid_{horizon_seconds:g}s"] = aux_valid
                    record[f"event_aux_label_{horizon_seconds:g}s"] = (
                        int(event_labels_np[b, j, h_idx]) if aux_valid else None
                    )
                    record[f"event_aux_pred_{horizon_seconds:g}s"] = (
                        int(event_pred_np[b, j, h_idx]) if h_idx < event_pred_np.shape[-1] and aux_valid else None
                    )
                    record[f"event_aux_valid_{horizon_seconds:g}s"] = aux_valid
                prediction_records.append(record)

    metrics = compute_regression_metrics(y_true, y_pred)
    metrics = {f"current_fms_{key}": value for key, value in metrics.items()}
    metrics["mae"] = metrics.get("current_fms_mae", float("nan"))
    metrics["rmse"] = metrics.get("current_fms_rmse", float("nan"))
    finite_sigma = np.asarray([value for value in y_pred_sigma if np.isfinite(value)], dtype=np.float64)
    metrics["uncertainty_sigma_mean"] = float(np.mean(finite_sigma)) if finite_sigma.size else float("nan")
    metrics["uncertainty_sigma_median"] = float(np.median(finite_sigma)) if finite_sigma.size else float("nan")
    metrics["n"] = metrics.get("current_fms_n", 0)
    metrics["regression_head"] = compute_regression_metrics(y_true, y_pred_regression)
    metrics["ordinal_head_value"] = compute_regression_metrics(y_true, y_pred_ordinal_value)
    metrics["ordinal_head_hard_value"] = compute_regression_metrics(y_true, y_pred_ordinal_hard_value)
    if y_pred_ordinal_hard_value:
        yt_arr = np.asarray(y_true, dtype=np.float64)
        yp_arr = np.asarray(y_pred_ordinal_hard_value, dtype=np.float64)
        finite_int = np.isfinite(yt_arr) & np.isfinite(yp_arr)
        if finite_int.any():
            yt_int = np.rint(yt_arr[finite_int]).astype(np.int64)
            yp_int = np.rint(yp_arr[finite_int]).astype(np.int64)
            diff_int = np.abs(yt_int - yp_int)
            metrics["integer_exact_accuracy"] = float(np.mean(diff_int == 0))
            metrics["integer_off_by_one_accuracy"] = float(np.mean(diff_int <= 1))
            metrics["integer_hard_mae"] = float(np.mean(diff_int))
        else:
            metrics["integer_exact_accuracy"] = float("nan")
            metrics["integer_off_by_one_accuracy"] = float("nan")
            metrics["integer_hard_mae"] = float("nan")
    else:
        metrics["integer_exact_accuracy"] = float("nan")
        metrics["integer_off_by_one_accuracy"] = float("nan")
        metrics["integer_hard_mae"] = float("nan")
    metrics.update({f"caution_{key}": value for key, value in compute_high_fms_metrics(y_true, y_pred, threshold=high_fms_caution_threshold).items()})
    metrics.update({f"warning_{key}": value for key, value in compute_high_fms_metrics(y_true, y_pred, threshold=high_fms_warning_threshold).items()})
    metrics.update(
        compute_online_current_goal_selection_metrics(
            y_true,
            y_pred,
            mae=float(metrics.get("current_fms_mae", float("nan"))),
            r2=float(metrics.get("current_fms_r2", float("nan"))),
            high8_f1=float(metrics.get("caution_high_fms_f1", float("nan"))),
            high12_f1=float(metrics.get("warning_high_fms_f1", float("nan"))),
        )
    )
    if ordinal_true:
        ot = np.asarray(ordinal_true, dtype=np.int64)
        op = np.asarray(ordinal_pred, dtype=np.int64)
        metrics["ordinal_accuracy"] = float(np.mean(ot == op))
        metrics["ordinal_off_by_one_accuracy"] = float(np.mean(np.abs(ot - op) <= 1))
    else:
        metrics["ordinal_accuracy"] = float("nan")
        metrics["ordinal_off_by_one_accuracy"] = float("nan")
    metrics["rapid_rise"] = {}
    for horizon_steps in rise_horizon_steps:
        key = f"{int(horizon_steps)}"
        horizon_seconds = float(horizon_steps) * float(sampling_interval)
        metrics["rapid_rise"][f"{horizon_seconds:g}s"] = _binary_metrics(
            risk_true[key],
            risk_score[key],
            threshold=rapid_rise_probability_threshold,
        )
    metrics["rapid_drop"] = {}
    for horizon_steps in fall_horizon_steps:
        key = f"{int(horizon_steps)}"
        horizon_seconds = float(horizon_steps) * float(sampling_interval)
        metrics["rapid_drop"][f"{horizon_seconds:g}s"] = _binary_metrics(
            fall_true[key],
            fall_score[key],
            threshold=rapid_drop_probability_threshold,
        )
    metrics["high_risk"] = {}
    metrics["high_risk_label_mode"] = str(high_risk_label_mode or "future_any")
    metrics["high_risk_onset_past_steps"] = int(max(high_risk_onset_past_steps, 0))
    for horizon_steps in high_risk_horizon_steps:
        horizon_seconds = float(horizon_steps) * float(sampling_interval)
        for threshold in high_risk_thresholds:
            key = f"{int(horizon_steps)}_{float(threshold):g}"
            metrics["high_risk"][f"{horizon_seconds:g}s_thr{float(threshold):g}"] = _binary_metrics(
                high_risk_true[key],
                high_risk_score[key],
                threshold=rapid_rise_probability_threshold,
            )
    metrics["future_aux"] = {}
    metrics["delta_aux"] = {}
    metrics["event_aux"] = {}
    for horizon_steps in future_aux_horizon_steps:
        key = f"{int(horizon_steps)}"
        horizon_seconds = float(horizon_steps) * float(sampling_interval)
        label = f"{horizon_seconds:g}s"
        metrics["future_aux"][label] = compute_regression_metrics(future_true[key], future_pred[key])
        delta_metrics = compute_regression_metrics(delta_true[key], delta_pred[key])
        delta_metrics["corr"] = _finite_corr(delta_true[key], delta_pred[key])
        metrics["delta_aux"][label] = delta_metrics
        metrics["event_aux"][label] = {
            "n": int(len(event_true[key])),
            "accuracy": float(np.mean(np.asarray(event_true[key], dtype=np.int64) == np.asarray(event_pred[key], dtype=np.int64)))
            if event_true[key]
            else float("nan"),
        }
    metrics["rapid_rise_any"] = _binary_metrics(rapid_any_true, rapid_any_score, threshold=rapid_rise_probability_threshold)
    metrics["rapid_drop_any"] = _binary_metrics(drop_any_true, drop_any_score, threshold=rapid_drop_probability_threshold)
    final_prf = _binary_metrics(final_true, [1.0 if value else 0.0 for value in final_alarm], threshold=0.5)
    metrics["final_warning_mode"] = final_warning_mode
    metrics["final_warning"] = final_prf
    metrics["trajectory"] = _online_current_trajectory_metrics(
        prediction_records,
        horizons_seconds=(5.0, 10.0),
    )
    return {
        "metrics": metrics,
        "series": plot_series,
        "y_true": y_true,
        "y_pred": y_pred,
        "prediction_records": prediction_records,
    }


def _future_horizon_steps_from_outputs(outputs: Mapping[str, torch.Tensor], fallback_horizon_steps: int) -> List[int]:
    values = outputs.get("horizon_steps_list")
    if isinstance(values, torch.Tensor):
        return [int(v) for v in values.detach().cpu().flatten().tolist()]
    if values is not None:
        return [int(v) for v in values]  # type: ignore[arg-type]
    future = outputs["future"]
    if future.ndim == 3 and future.shape[-1] != 1:
        raise ValueError("Multi-horizon outputs must include horizon_steps_list for teacher distillation.")
    return [int(fallback_horizon_steps)]


def _future_and_mask_as_3d(outputs: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    future = outputs["future"]
    mask = outputs["mask"].to(future.device).bool()
    if future.ndim == 2:
        future = future.unsqueeze(-1)
        mask = mask.unsqueeze(-1) if mask.ndim == 2 else mask
    elif future.ndim == 3:
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1).expand_as(future)
        elif mask.shape != future.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match future shape {future.shape}.")
    else:
        raise ValueError(f"Expected future output to be [B,P] or [B,P,H], got {future.shape}.")
    return future, mask


def _tensor_and_mask_as_3d(outputs: Mapping[str, torch.Tensor], key: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if key not in outputs:
        raise ValueError(f"Expected model outputs to include {key!r}.")
    value = outputs[key]
    mask = outputs["mask"].to(value.device).bool()
    if value.ndim == 2:
        value = value.unsqueeze(-1)
        mask = mask.unsqueeze(-1) if mask.ndim == 2 else mask
    elif value.ndim == 3:
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1).expand_as(value)
        elif mask.shape != value.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match {key} shape {value.shape}.")
    else:
        raise ValueError(f"Expected {key} to be [B,P] or [B,P,H], got {value.shape}.")
    return value, mask


def compute_teacher_future_distillation_loss(
    student_outputs: Mapping[str, torch.Tensor],
    teacher_outputs: Mapping[str, torch.Tensor],
    horizon_steps: int,
    loss_type: str = "smooth_l1",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Match student future predictions to a privileged teacher on shared times/horizons."""
    if loss_type not in {"smooth_l1", "mse", "l1", "mae"}:
        raise ValueError("teacher_distill_loss_type must be one of: smooth_l1, mse, l1, mae.")
    student_future, student_mask = _future_and_mask_as_3d(student_outputs)
    teacher_future, teacher_mask = _future_and_mask_as_3d(teacher_outputs)
    device = student_future.device
    teacher_future = teacher_future.to(device).detach()
    teacher_mask = teacher_mask.to(device)

    student_start = _tensor_scalar_int(student_outputs.get("prediction_start"), 0)
    teacher_start = _tensor_scalar_int(teacher_outputs.get("prediction_start"), 0)
    student_steps = int(student_future.shape[1])
    teacher_steps = int(teacher_future.shape[1])
    common_start = max(student_start, teacher_start)
    common_end = min(student_start + student_steps, teacher_start + teacher_steps)
    if common_end <= common_start:
        zero = student_future.new_tensor(0.0)
        return zero, {
            "loss_teacher_distill": 0.0,
            "teacher_distill_points": 0,
            "teacher_distill_weight": 0.0,
        }

    student_horizons = _future_horizon_steps_from_outputs(student_outputs, horizon_steps)
    teacher_horizons = _future_horizon_steps_from_outputs(teacher_outputs, horizon_steps)
    teacher_horizon_to_idx = {int(value): idx for idx, value in enumerate(teacher_horizons)}
    shared_horizons = [value for value in student_horizons if int(value) in teacher_horizon_to_idx]
    if not shared_horizons:
        raise ValueError(
            f"Teacher/student horizon sets do not overlap: student={student_horizons}, teacher={teacher_horizons}."
        )
    student_horizon_idx = [student_horizons.index(value) for value in shared_horizons]
    teacher_horizon_idx = [teacher_horizon_to_idx[int(value)] for value in shared_horizons]

    positions = torch.arange(common_start, common_end, device=device, dtype=torch.long)
    student_pos_idx = positions - int(student_start)
    teacher_pos_idx = positions - int(teacher_start)
    student_sel = student_future.index_select(1, student_pos_idx).index_select(2, torch.tensor(student_horizon_idx, device=device))
    teacher_sel = teacher_future.index_select(1, teacher_pos_idx).index_select(2, torch.tensor(teacher_horizon_idx, device=device))
    student_mask_sel = student_mask.index_select(1, student_pos_idx).index_select(2, torch.tensor(student_horizon_idx, device=device))
    teacher_mask_sel = teacher_mask.index_select(1, teacher_pos_idx).index_select(2, torch.tensor(teacher_horizon_idx, device=device))
    valid = student_mask_sel & teacher_mask_sel & torch.isfinite(student_sel) & torch.isfinite(teacher_sel)
    safe_student = torch.where(valid, student_sel, torch.zeros_like(student_sel))
    safe_teacher = torch.where(valid, teacher_sel, torch.zeros_like(teacher_sel))
    if loss_type == "smooth_l1":
        raw = torch.nn.functional.smooth_l1_loss(safe_student, safe_teacher, reduction="none")
    else:
        err = safe_student - safe_teacher
        raw = err.abs() if loss_type in {"l1", "mae"} else err.square()
    valid_f = valid.to(raw.dtype)
    loss = (raw * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    return loss, {
        "loss_teacher_distill": float(loss.detach().cpu()),
        "teacher_distill_points": int(valid.sum().detach().cpu()),
        "teacher_distill_weight": 0.0,
    }


def compute_teacher_current_distillation_loss(
    student_outputs: Mapping[str, torch.Tensor],
    teacher_outputs: Mapping[str, torch.Tensor],
    loss_type: str = "smooth_l1",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Match online-current student predictions to a fixed teacher on shared current times."""
    if loss_type not in {"smooth_l1", "mse", "l1", "mae"}:
        raise ValueError("teacher_distill_loss_type must be one of: smooth_l1, mse, l1, mae.")
    if "current" not in student_outputs or "current" not in teacher_outputs:
        raise ValueError("Current-FMS teacher distillation requires both outputs to include 'current'.")
    student_current = student_outputs["current"]
    teacher_current = teacher_outputs["current"].to(student_current.device).detach()
    student_mask = student_outputs["mask"].to(student_current.device).bool()
    teacher_mask = teacher_outputs["mask"].to(student_current.device).bool()
    if student_current.ndim != 2 or teacher_current.ndim != 2:
        raise ValueError(
            f"Current-FMS teacher distillation expects [B,P] tensors, got "
            f"student={student_current.shape}, teacher={teacher_current.shape}."
        )
    if student_mask.shape != student_current.shape:
        raise ValueError(f"Student mask shape {student_mask.shape} does not match current shape {student_current.shape}.")
    if teacher_mask.shape != teacher_current.shape:
        raise ValueError(f"Teacher mask shape {teacher_mask.shape} does not match current shape {teacher_current.shape}.")
    device = student_current.device
    student_start = _tensor_scalar_int(student_outputs.get("prediction_start"), 0)
    teacher_start = _tensor_scalar_int(teacher_outputs.get("prediction_start"), 0)
    common_start = max(student_start, teacher_start)
    common_end = min(student_start + int(student_current.shape[1]), teacher_start + int(teacher_current.shape[1]))
    if common_end <= common_start:
        zero = student_current.new_tensor(0.0)
        return zero, {
            "loss_teacher_distill": 0.0,
            "teacher_distill_points": 0,
            "teacher_distill_weight": 0.0,
            "teacher_distill_target": "current",
        }
    positions = torch.arange(common_start, common_end, dtype=torch.long, device=device)
    student_idx = positions - int(student_start)
    teacher_idx = positions - int(teacher_start)
    student_sel = student_current.index_select(1, student_idx)
    teacher_sel = teacher_current.index_select(1, teacher_idx)
    student_mask_sel = student_mask.index_select(1, student_idx)
    teacher_mask_sel = teacher_mask.index_select(1, teacher_idx)
    valid = student_mask_sel & teacher_mask_sel & torch.isfinite(student_sel) & torch.isfinite(teacher_sel)
    loss = _masked_regression_loss(student_sel, teacher_sel, valid, loss_type=loss_type)
    return loss, {
        "loss_teacher_distill": float(loss.detach().cpu()),
        "teacher_distill_points": int(valid.sum().detach().cpu()),
        "teacher_distill_weight": 0.0,
        "teacher_distill_target": "current",
    }


def compute_teacher_delta_distillation_loss(
    student_outputs: Mapping[str, torch.Tensor],
    teacher_outputs: Mapping[str, torch.Tensor],
    horizon_steps: int,
    loss_type: str = "smooth_l1",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Match student future-delta head to teacher future targets relative to student current state."""
    if loss_type not in {"smooth_l1", "mse", "l1", "mae"}:
        raise ValueError("teacher_delta_distill_loss_type must be one of: smooth_l1, mse, l1, mae.")
    student_delta, student_delta_mask = _tensor_and_mask_as_3d(student_outputs, "future_delta_pred")
    student_base, student_base_mask = _tensor_and_mask_as_3d(student_outputs, "future_delta_base")
    teacher_future, teacher_mask = _future_and_mask_as_3d(teacher_outputs)
    device = student_delta.device
    student_base = student_base.to(device).detach()
    teacher_future = teacher_future.to(device).detach()
    teacher_mask = teacher_mask.to(device)

    student_start = _tensor_scalar_int(student_outputs.get("prediction_start"), 0)
    teacher_start = _tensor_scalar_int(teacher_outputs.get("prediction_start"), 0)
    common_start = max(student_start, teacher_start)
    common_end = min(student_start + int(student_delta.shape[1]), teacher_start + int(teacher_future.shape[1]))
    if common_end <= common_start:
        zero = student_delta.new_tensor(0.0)
        return zero, {
            "loss_teacher_delta_distill": 0.0,
            "teacher_delta_distill_points": 0,
            "teacher_delta_distill_weight": 0.0,
            "teacher_delta_distill_effective_weight": 0.0,
        }

    student_horizons = _future_horizon_steps_from_outputs(student_outputs, horizon_steps)
    teacher_horizons = _future_horizon_steps_from_outputs(teacher_outputs, horizon_steps)
    teacher_horizon_to_idx = {int(value): idx for idx, value in enumerate(teacher_horizons)}
    shared_horizons = [value for value in student_horizons if int(value) in teacher_horizon_to_idx]
    if not shared_horizons:
        raise ValueError(
            f"Teacher/student horizon sets do not overlap for delta distillation: "
            f"student={student_horizons}, teacher={teacher_horizons}."
        )
    student_horizon_idx = [student_horizons.index(value) for value in shared_horizons]
    teacher_horizon_idx = [teacher_horizon_to_idx[int(value)] for value in shared_horizons]

    positions = torch.arange(common_start, common_end, device=device, dtype=torch.long)
    student_pos_idx = positions - int(student_start)
    teacher_pos_idx = positions - int(teacher_start)
    student_h_idx = torch.tensor(student_horizon_idx, device=device, dtype=torch.long)
    teacher_h_idx = torch.tensor(teacher_horizon_idx, device=device, dtype=torch.long)
    student_delta_sel = student_delta.index_select(1, student_pos_idx).index_select(2, student_h_idx)
    student_base_sel = student_base.index_select(1, student_pos_idx).index_select(2, student_h_idx)
    teacher_future_sel = teacher_future.index_select(1, teacher_pos_idx).index_select(2, teacher_h_idx)
    student_delta_mask_sel = student_delta_mask.index_select(1, student_pos_idx).index_select(2, student_h_idx)
    student_base_mask_sel = student_base_mask.index_select(1, student_pos_idx).index_select(2, student_h_idx)
    teacher_mask_sel = teacher_mask.index_select(1, teacher_pos_idx).index_select(2, teacher_h_idx)
    target_delta = teacher_future_sel - student_base_sel
    valid = (
        student_delta_mask_sel
        & student_base_mask_sel
        & teacher_mask_sel
        & torch.isfinite(student_delta_sel)
        & torch.isfinite(target_delta)
    )
    safe_student = torch.where(valid, student_delta_sel, torch.zeros_like(student_delta_sel))
    safe_target = torch.where(valid, target_delta, torch.zeros_like(target_delta))
    if loss_type == "smooth_l1":
        raw = torch.nn.functional.smooth_l1_loss(safe_student, safe_target, reduction="none")
    else:
        err = safe_student - safe_target
        raw = err.abs() if loss_type in {"l1", "mae"} else err.square()
    valid_f = valid.to(raw.dtype)
    loss = (raw * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    return loss, {
        "loss_teacher_delta_distill": float(loss.detach().cpu()),
        "teacher_delta_distill_points": int(valid.sum().detach().cpu()),
        "teacher_delta_distill_weight": 0.0,
        "teacher_delta_distill_effective_weight": 0.0,
    }


def _repr_and_mask_as_4d(outputs: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if "distill_repr" not in outputs:
        raise ValueError("Teacher representation distillation requires model outputs['distill_repr'].")
    repr_tensor = outputs["distill_repr"]
    mask = outputs["mask"].to(repr_tensor.device).bool()
    if repr_tensor.ndim == 3:
        repr_tensor = repr_tensor.unsqueeze(2)
        mask = mask.unsqueeze(-1) if mask.ndim == 2 else mask
    elif repr_tensor.ndim == 4:
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1).expand(repr_tensor.shape[:3])
        elif mask.shape != repr_tensor.shape[:3]:
            raise ValueError(f"Mask shape {mask.shape} does not match distill_repr shape {repr_tensor.shape}.")
    else:
        raise ValueError(f"Expected distill_repr to be [B,P,D] or [B,P,H,D], got {repr_tensor.shape}.")
    return repr_tensor, mask


def compute_teacher_repr_distillation_loss(
    student_outputs: Mapping[str, torch.Tensor],
    teacher_outputs: Mapping[str, torch.Tensor],
    horizon_steps: int,
    student_projector: Optional[torch.nn.Module] = None,
    loss_type: str = "smooth_l1",
    normalize: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Align student latent states with a privileged teacher representation."""
    if loss_type not in {"smooth_l1", "mse", "l1", "mae"}:
        raise ValueError("teacher_repr_distill_loss_type must be one of: smooth_l1, mse, l1, mae.")
    student_repr, student_mask = _repr_and_mask_as_4d(student_outputs)
    teacher_repr, teacher_mask = _repr_and_mask_as_4d(teacher_outputs)
    device = student_repr.device
    teacher_repr = teacher_repr.to(device).detach()
    teacher_mask = teacher_mask.to(device)

    student_start = _tensor_scalar_int(student_outputs.get("prediction_start"), 0)
    teacher_start = _tensor_scalar_int(teacher_outputs.get("prediction_start"), 0)
    common_start = max(student_start, teacher_start)
    common_end = min(student_start + int(student_repr.shape[1]), teacher_start + int(teacher_repr.shape[1]))
    if common_end <= common_start:
        zero = student_repr.new_tensor(0.0)
        return zero, {
            "loss_teacher_repr_distill": 0.0,
            "teacher_repr_distill_points": 0,
            "teacher_repr_distill_weight": 0.0,
            "teacher_repr_distill_effective_weight": 0.0,
        }

    student_horizons = _future_horizon_steps_from_outputs(student_outputs, horizon_steps)
    teacher_horizons = _future_horizon_steps_from_outputs(teacher_outputs, horizon_steps)
    teacher_horizon_to_idx = {int(value): idx for idx, value in enumerate(teacher_horizons)}
    shared_horizons = [value for value in student_horizons if int(value) in teacher_horizon_to_idx]
    if not shared_horizons:
        raise ValueError(
            f"Teacher/student horizon sets do not overlap for repr distillation: "
            f"student={student_horizons}, teacher={teacher_horizons}."
        )
    student_horizon_idx = [student_horizons.index(value) for value in shared_horizons]
    teacher_horizon_idx = [teacher_horizon_to_idx[int(value)] for value in shared_horizons]

    positions = torch.arange(common_start, common_end, device=device, dtype=torch.long)
    student_pos_idx = positions - int(student_start)
    teacher_pos_idx = positions - int(teacher_start)
    student_h_idx = torch.tensor(student_horizon_idx, device=device, dtype=torch.long)
    teacher_h_idx = torch.tensor(teacher_horizon_idx, device=device, dtype=torch.long)
    student_sel = student_repr.index_select(1, student_pos_idx).index_select(2, student_h_idx)
    teacher_sel = teacher_repr.index_select(1, teacher_pos_idx).index_select(2, teacher_h_idx)
    if student_projector is not None:
        student_sel = student_projector(student_sel)
    if student_sel.shape[-1] != teacher_sel.shape[-1]:
        raise ValueError(
            f"Representation dimensions differ after projection: student={student_sel.shape[-1]}, "
            f"teacher={teacher_sel.shape[-1]}."
        )
    if normalize:
        student_sel = torch.nn.functional.layer_norm(student_sel, (student_sel.shape[-1],))
        teacher_sel = torch.nn.functional.layer_norm(teacher_sel, (teacher_sel.shape[-1],))
    student_mask_sel = student_mask.index_select(1, student_pos_idx).index_select(2, student_h_idx)
    teacher_mask_sel = teacher_mask.index_select(1, teacher_pos_idx).index_select(2, teacher_h_idx)
    valid = student_mask_sel & teacher_mask_sel & torch.isfinite(student_sel).all(dim=-1) & torch.isfinite(teacher_sel).all(dim=-1)
    valid_4d = valid.unsqueeze(-1).expand_as(student_sel)
    safe_student = torch.where(valid_4d, student_sel, torch.zeros_like(student_sel))
    safe_teacher = torch.where(valid_4d, teacher_sel, torch.zeros_like(teacher_sel))
    if loss_type == "smooth_l1":
        raw = torch.nn.functional.smooth_l1_loss(safe_student, safe_teacher, reduction="none")
    else:
        err = safe_student - safe_teacher
        raw = err.abs() if loss_type in {"l1", "mae"} else err.square()
    valid_f = valid_4d.to(raw.dtype)
    loss = (raw * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    return loss, {
        "loss_teacher_repr_distill": float(loss.detach().cpu()),
        "teacher_repr_distill_points": int(valid.sum().detach().cpu()),
        "teacher_repr_distill_weight": 0.0,
        "teacher_repr_distill_effective_weight": 0.0,
    }


def distill_weight_for_epoch(weight: float, epoch: int, start_epoch: int = 1, warmup_epochs: int = 0) -> float:
    if float(weight) <= 0:
        return 0.0
    if int(epoch) < int(start_epoch):
        return 0.0
    if int(warmup_epochs) <= 0:
        return float(weight)
    progress = (int(epoch) - int(start_epoch) + 1) / float(warmup_epochs)
    return float(weight) * max(0.0, min(1.0, progress))


def load_teacher_model(checkpoint_path: str | Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    path = Path(checkpoint_path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name")
    model_kwargs = ckpt.get("model_kwargs")
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if not model_name or not isinstance(model_kwargs, Mapping) or state_dict is None:
        raise ValueError(f"Teacher checkpoint {path} does not contain model_name, model_kwargs, and model_state_dict.")
    teacher = build_model(str(model_name), **dict(model_kwargs)).to(device)
    teacher.load_state_dict(state_dict)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher, {
        "checkpoint": str(path),
        "model_name": str(model_name),
        "model_kwargs": dict(model_kwargs),
        "requires_full_fms": bool(getattr(teacher, "requires_full_fms", False)),
        "use_static": bool(getattr(teacher, "use_static", False)),
    }


def validate_teacher_compatibility(teacher_info: Mapping[str, Any], student_kwargs: Mapping[str, Any]) -> None:
    teacher_kwargs = teacher_info.get("model_kwargs", {})
    if not isinstance(teacher_kwargs, Mapping):
        return
    for key in ("head_dim", "calibration_steps"):
        if key in teacher_kwargs and key in student_kwargs and int(teacher_kwargs[key]) != int(student_kwargs[key]):
            raise ValueError(
                f"Teacher checkpoint {teacher_info.get('checkpoint')} has {key}={teacher_kwargs[key]}, "
                f"but this run uses {key}={student_kwargs[key]}."
            )
    if "sampling_interval" in teacher_kwargs and "sampling_interval" in student_kwargs:
        if abs(float(teacher_kwargs["sampling_interval"]) - float(student_kwargs["sampling_interval"])) > 1e-8:
            raise ValueError(
                f"Teacher checkpoint {teacher_info.get('checkpoint')} has sampling_interval={teacher_kwargs['sampling_interval']}, "
                f"but this run uses sampling_interval={student_kwargs['sampling_interval']}."
            )


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    calibration_steps: int,
    horizon_steps: int,
    fms_scaler: Mapping[str, float],
    high_fms_threshold: float = 7.0,
    use_static: bool = False,
    calibration_seconds: Optional[float] = None,
    horizon_seconds: Optional[float] = None,
    recent_window_seconds: Optional[float] = None,
    common_eval_current_start: Optional[float] = None,
    common_eval_current_end: Optional[float] = None,
    common_eval_target_start: Optional[float] = None,
    common_eval_target_end: Optional[float] = None,
    common_eval_max_horizon_seconds: Optional[float] = None,
    sampling_interval: float = 0.5,
    recent_window_steps: Optional[int] = None,
    run_name: Optional[str] = None,
    model_name: Optional[str] = None,
    split_name: Optional[str] = None,
    anchor_mode: str = "none",
    anchor_interval_seconds: float = 60.0,
    fms_context_mode: str = "calibration_history",
    is_upper_bound_anchor: bool = False,
    max_eval_batches: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    y_true: List[float] = []
    y_pred: List[float] = []
    half_true = {"first": [], "second": []}
    half_pred = {"first": [], "second": []}
    plot_series: List[Dict[str, Any]] = []
    sequence_series: List[Dict[str, Any]] = []
    common_sequence_series: List[Dict[str, Any]] = []
    prediction_records: List[Dict[str, Any]] = []
    common_y_true: List[float] = []
    common_y_pred: List[float] = []
    subgroup_true: Dict[str, List[float]] = {}
    subgroup_pred: Dict[str, List[float]] = {}
    age_records: List[Tuple[float, List[float], List[float]]] = []
    mssq_records: List[Tuple[float, List[float], List[float]]] = []
    horizon_true: Dict[str, List[float]] = {}
    horizon_pred: Dict[str, List[float]] = {}
    current_true: List[float] = []
    current_pred: List[float] = []

    for batch_idx, batch in enumerate(loader):
        if max_eval_batches is not None and batch_idx >= int(max_eval_batches):
            break
        head = batch["head"].to(device)
        fms = batch["fms"].to(device)
        lengths = batch["lengths"].to(device)
        static = batch.get("static")
        if use_static:
            if static is None:
                raise ValueError("Model was configured with use_static=True, but batch['static'] is missing.")
            static = static.to(device)
        fms_input = fms if getattr(model, "requires_full_fms", False) else fms[:, :calibration_steps]
        outputs = model(head, fms_input, lengths, static=static)
        pred_future = outputs["future"]
        prediction_start = _tensor_scalar_int(outputs.get("prediction_start"), calibration_steps)
        multi_horizon = pred_future.ndim == 3
        if multi_horizon:
            horizon_values = outputs.get("horizon_steps_list")
            horizon_steps_list = (
                [int(v) for v in horizon_values.detach().cpu().tolist()]
                if isinstance(horizon_values, torch.Tensor)
                else [int(horizon_steps)]
            )
            target_future, target_mask = _future_targets_for_horizons(
                fms,
                lengths,
                prediction_start,
                horizon_steps_list,
                max_pred_steps=pred_future.shape[1],
            )
            pred_steps = int(pred_future.shape[1])
            h_count = len(horizon_steps_list)
            current_positions = prediction_start + torch.arange(pred_steps, dtype=torch.long, device=device)
            safe_current_idx = current_positions.clamp_max(batch["time"].shape[1] - 1)
            current_base = batch["time"].to(device).index_select(1, safe_current_idx)
            current_time = current_base.unsqueeze(-1).expand(-1, -1, h_count)
            horizon_steps_tensor = torch.tensor(horizon_steps_list, dtype=torch.long, device=device)
            target_positions = current_positions.view(1, -1, 1) + horizon_steps_tensor.view(1, 1, -1)
            safe_target_idx = target_positions.clamp_max(batch["time"].shape[1] - 1).expand(batch["time"].shape[0], -1, -1)
            target_time = torch.gather(batch["time"].to(device).unsqueeze(-1).expand(-1, -1, h_count), 1, safe_target_idx)
            time_mask = target_positions < lengths.view(-1, 1, 1)
            current_time_mask = current_positions.view(1, -1, 1) < lengths.view(-1, 1, 1)
        else:
            if pred_future.ndim != 2:
                raise ValueError(f"collect_predictions expects [B,T] or [B,T,H], got {pred_future.shape}.")
            horizon_steps_list = [int(horizon_steps)]
            target_future, target_mask = future_sequence_targets(
                fms,
                lengths,
                calibration_steps,
                horizon_steps,
                max_pred_steps=pred_future.shape[1],
                prediction_start_steps=prediction_start,
            )
            target_time, time_mask = future_sequence_times(
                batch["time"].to(device),
                lengths,
                calibration_steps,
                horizon_steps,
                max_pred_steps=pred_future.shape[1],
                prediction_start_steps=prediction_start,
            )
            current_time, current_time_mask = current_sequence_times(
                batch["time"].to(device),
                lengths,
                calibration_steps,
                horizon_steps,
                max_pred_steps=pred_future.shape[1],
                prediction_start_steps=prediction_start,
            )
        mask = outputs["mask"].to(device) & target_mask & time_mask
        mask = mask & current_time_mask
        pred_raw_all = denormalize_fms(pred_future, fms_scaler)
        target_raw_all = denormalize_fms(target_future, fms_scaler)
        current_pred_raw: Optional[np.ndarray] = None
        current_target_raw: Optional[np.ndarray] = None
        current_aux_mask_np: Optional[np.ndarray] = None
        if "current" in outputs:
            current_aux = outputs["current"].to(device)
            if current_aux.ndim == 3 and current_aux.shape[-1] == 1:
                current_aux = current_aux.squeeze(-1)
            if current_aux.ndim != 2:
                raise ValueError(f"current auxiliary prediction must be [B,P], got {current_aux.shape}")
            current_steps = int(current_aux.shape[1])
            current_positions_aux = prediction_start + torch.arange(current_steps, dtype=torch.long, device=device)
            safe_current_idx_aux = current_positions_aux.clamp_max(fms.shape[1] - 1)
            current_target_aux = fms.index_select(1, safe_current_idx_aux)
            current_aux_mask = current_positions_aux.view(1, -1) < lengths.view(-1, 1)
            if multi_horizon:
                current_aux_mask = current_aux_mask & outputs["mask"].to(device).bool().any(dim=-1)[:, :current_steps]
            else:
                current_aux_mask = current_aux_mask & outputs["mask"].to(device).bool()[:, :current_steps]
            current_aux_mask = current_aux_mask & torch.isfinite(current_target_aux)
            current_pred_raw = denormalize_fms(current_aux, fms_scaler)
            current_target_raw = denormalize_fms(current_target_aux, fms_scaler)
            current_aux_mask_np = current_aux_mask.detach().cpu().numpy()
            current_true.extend(current_target_raw[current_aux_mask_np].tolist())
            current_pred.extend(current_pred_raw[current_aux_mask_np].tolist())
        if multi_horizon:
            h_count = len(horizon_steps_list)
            pred_raw_all = pred_raw_all.reshape(pred_raw_all.shape[0], -1)
            target_raw_all = target_raw_all.reshape(target_raw_all.shape[0], -1)
            mask = mask.reshape(mask.shape[0], -1)
            target_time = target_time.reshape(target_time.shape[0], -1)
            current_time = current_time.reshape(current_time.shape[0], -1)
            time_mask = time_mask.reshape(time_mask.shape[0], -1)
            current_time_mask = current_time_mask.reshape(current_time_mask.shape[0], -1)
            position_offsets = np.repeat(np.arange(pred_future.shape[1], dtype=np.int64), h_count)
            horizon_steps_flat = np.tile(np.asarray(horizon_steps_list, dtype=np.int64), pred_future.shape[1])
            horizon_seconds_flat = horizon_steps_flat.astype(np.float64) * float(sampling_interval)
        else:
            position_offsets = np.arange(pred_future.shape[1], dtype=np.int64)
            horizon_steps_flat = np.full(pred_future.shape[1], int(horizon_steps), dtype=np.int64)
            horizon_seconds_flat = np.full(pred_future.shape[1], float(horizon_seconds if horizon_seconds is not None else horizon_steps * sampling_interval), dtype=np.float64)

        mask_np = mask.cpu().numpy()
        pred_np = pred_raw_all
        target_np = target_raw_all
        lengths_np = lengths.cpu().numpy()
        target_time_np = target_time.detach().cpu().numpy()
        current_time_np = current_time.detach().cpu().numpy()
        if horizon_seconds is not None and not multi_horizon:
            target_time_np = current_time_np + float(horizon_seconds)
        time_np = batch["time"].detach().cpu().numpy()
        fms_raw_np = batch.get("fms_raw", fms).detach().cpu().numpy()
        common_mask_np = mask_np.copy()
        if common_eval_current_start is not None:
            common_mask_np &= current_time_np >= float(common_eval_current_start)
        if common_eval_current_end is not None:
            common_mask_np &= current_time_np <= float(common_eval_current_end)
        if common_eval_target_start is not None:
            common_mask_np &= target_time_np >= float(common_eval_target_start)
        if common_eval_target_end is not None:
            common_mask_np &= target_time_np <= float(common_eval_target_end)
        if common_eval_max_horizon_seconds is not None:
            session_end_time = np.asarray(
                [
                    time_np[b, int(lengths_np[b]) - 1] if lengths_np[b] > 0 else np.nan
                    for b in range(time_np.shape[0])
                ],
                dtype=np.float64,
            )
            common_current_end_by_session = session_end_time[:, None] - float(common_eval_max_horizon_seconds)
            common_mask_np &= current_time_np <= common_current_end_by_session

        y_true.extend(target_np[mask_np].tolist())
        y_pred.extend(pred_np[mask_np].tolist())
        common_y_true.extend(target_np[common_mask_np].tolist())
        common_y_pred.extend(pred_np[common_mask_np].tolist())
        for h_sec in np.unique(horizon_seconds_flat):
            h_key = f"{float(h_sec):g}"
            h_mask = mask_np & (horizon_seconds_flat[None, :] == h_sec)
            horizon_true.setdefault(h_key, []).extend(target_np[h_mask].tolist())
            horizon_pred.setdefault(h_key, []).extend(pred_np[h_mask].tolist())

        for b in range(mask_np.shape[0]):
            valid_j = np.where(mask_np[b])[0]
            if not multi_horizon:
                sequence_series.append(
                    {
                        "metadata": batch["metadata"][b],
                        "target_full": target_np[b].tolist(),
                        "prediction_full": pred_np[b].tolist(),
                        "mask": mask_np[b].tolist(),
                    }
                )
                common_sequence_series.append(
                    {
                        "metadata": batch["metadata"][b],
                        "target_full": target_np[b].tolist(),
                        "prediction_full": pred_np[b].tolist(),
                        "mask": common_mask_np[b].tolist(),
                    }
                )
            if len(valid_j) == 0:
                continue
            meta = batch["metadata"][b]
            for j in valid_j:
                true_value = float(target_np[b, j])
                pred_value = float(pred_np[b, j])
                current_index = int(prediction_start + int(position_offsets[j]))
                current_value = None
                current_prediction = None
                current_abs_error = None
                current_position_offset = int(position_offsets[j])
                if (
                    current_pred_raw is not None
                    and current_target_raw is not None
                    and current_aux_mask_np is not None
                    and current_position_offset < current_pred_raw.shape[1]
                    and current_aux_mask_np[b, current_position_offset]
                ):
                    current_value = float(current_target_raw[b, current_position_offset])
                    current_prediction = float(current_pred_raw[b, current_position_offset])
                    current_abs_error = abs(current_prediction - current_value)
                current_horizon_steps = int(horizon_steps_flat[j])
                current_horizon_seconds = float(horizon_seconds_flat[j])
                target_index = int(current_index + current_horizon_steps)
                session_length = int(lengths_np[b])
                session_duration = (
                    float(time_np[b, session_length - 1] - time_np[b, 0])
                    if session_length > 1 and np.isfinite(time_np[b, session_length - 1]) and np.isfinite(time_np[b, 0])
                    else float("nan")
                )
                recent_steps = int(recent_window_steps if recent_window_steps is not None else 0)
                anchor_index: Optional[int] = None
                nominal_start_index: Optional[int] = None
                nominal_start_time: Optional[float] = None
                anchor_is_fallback: Optional[bool] = None
                mode = str(anchor_mode or "none").lower()
                context_mode = str(fms_context_mode or "none").lower()
                if context_mode == "start_only":
                    nominal_start_index = current_index - recent_steps + 1 if recent_steps > 0 else current_index
                    nominal_start_index = int(max(0, min(nominal_start_index, current_index, session_length - 1)))
                    nominal_start_time = (
                        float(time_np[b, nominal_start_index])
                        if np.isfinite(time_np[b, nominal_start_index])
                        else float("nan")
                    )
                    anchor_index = nominal_start_index
                elif mode == "calibration_end":
                    anchor_index = calibration_steps - 1
                elif mode == "sparse_observed":
                    interval_steps = max(1, seconds_to_steps(float(anchor_interval_seconds), float(sampling_interval), name="anchor_interval_seconds", warn=False))
                    anchor_index = (current_index // interval_steps) * interval_steps
                    if anchor_index < calibration_steps:
                        anchor_index = calibration_steps - 1
                elif mode == "recent_start_observed":
                    anchor_index = current_index - recent_steps + 1 if recent_steps > 0 else current_index
                if anchor_index is not None:
                    anchor_index = int(max(0, min(anchor_index, current_index, session_length - 1)))
                    if not np.isfinite(fms_raw_np[b, anchor_index]):
                        finite_candidates = np.where(np.isfinite(fms_raw_np[b, : anchor_index + 1]))[0]
                        if len(finite_candidates):
                            anchor_index = int(finite_candidates[-1])
                    if context_mode == "start_only" and nominal_start_index is not None:
                        anchor_is_fallback = bool(anchor_index != nominal_start_index)
                    anchor_time = float(time_np[b, anchor_index]) if np.isfinite(time_np[b, anchor_index]) else float("nan")
                    anchor_fms = float(fms_raw_np[b, anchor_index]) if np.isfinite(fms_raw_np[b, anchor_index]) else float("nan")
                    time_since_anchor = float(current_time_np[b, j] - anchor_time) if np.isfinite(anchor_time) else float("nan")
                else:
                    anchor_time = None
                    anchor_fms = None
                    time_since_anchor = None
                prediction_records.append(
                    {
                        "run_name": run_name,
                        "model_name": model_name,
                        "split": split_name,
                        "participant_id": meta.get("participant_id"),
                        "session_id": meta.get("session_id"),
                        "source_file": meta.get("source_file"),
                        "age": meta.get("age"),
                        "gender": meta.get("gender"),
                        "mssq": meta.get("mssq"),
                        "static_feature_names": meta.get("static_feature_names"),
                        "current_index": current_index,
                        "target_index": target_index,
                        "current_time": float(current_time_np[b, j]),
                        "target_time": float(target_time_np[b, j]),
                        "session_length_steps": session_length,
                        "session_duration": session_duration,
                        "calibration_seconds": calibration_seconds,
                        "recent_window_seconds": recent_window_seconds,
                        "horizon_seconds": current_horizon_seconds,
                        "calibration_steps": int(calibration_steps),
                        "recent_window_steps": recent_steps,
                        "horizon_steps": current_horizon_steps,
                        "anchor_mode": mode,
                        "fms_context_mode": fms_context_mode,
                        "anchor_index": anchor_index,
                        "anchor_time": anchor_time,
                        "anchor_fms": anchor_fms,
                        "anchor_is_fallback": anchor_is_fallback,
                        "nominal_start_index": nominal_start_index,
                        "nominal_start_time": nominal_start_time,
                        "start_fms_index": anchor_index if context_mode == "start_only" else None,
                        "start_fms_time": anchor_time if context_mode == "start_only" else None,
                        "start_fms_value": anchor_fms if context_mode == "start_only" else None,
                        "time_since_anchor": time_since_anchor,
                        "is_upper_bound_anchor": bool(is_upper_bound_anchor),
                        "use_static": bool(use_static),
                        "current_fms": current_value,
                        "predicted_current_fms": current_prediction,
                        "current_absolute_error": current_abs_error,
                        "target_fms": true_value,
                        "predicted_fms": pred_value,
                        "absolute_error": abs(pred_value - true_value),
                        "squared_error": float((pred_value - true_value) ** 2),
                        "in_common_eval_window": bool(common_mask_np[b, j]),
                    }
                )
            if use_static:
                gender = str(meta.get("gender") or "unknown")
                key = gender if gender in {"male", "female", "unknown"} else "unknown"
                subgroup_true.setdefault(key, []).extend(target_np[b, valid_j].tolist())
                subgroup_pred.setdefault(key, []).extend(pred_np[b, valid_j].tolist())
                age = meta.get("age")
                if age is not None and np.isfinite(age):
                    age_records.append((float(age), target_np[b, valid_j].tolist(), pred_np[b, valid_j].tolist()))
                mssq = meta.get("mssq")
                if mssq is not None and np.isfinite(mssq):
                    mssq_records.append((float(mssq), target_np[b, valid_j].tolist(), pred_np[b, valid_j].tolist()))
            split_j = max(1, (int(lengths_np[b]) - horizon_steps - calibration_steps) // 2)
            first_j = valid_j[position_offsets[valid_j] < split_j]
            second_j = valid_j[position_offsets[valid_j] >= split_j]
            half_true["first"].extend(target_np[b, first_j].tolist())
            half_pred["first"].extend(pred_np[b, first_j].tolist())
            half_true["second"].extend(target_np[b, second_j].tolist())
            half_pred["second"].extend(pred_np[b, second_j].tolist())
            if len(plot_series) < 12:
                plot_series.append(
                    {
                        "metadata": batch["metadata"][b],
                        "target_time": target_time_np[b, valid_j].tolist(),
                        "target": target_np[b, valid_j].tolist(),
                        "prediction": pred_np[b, valid_j].tolist(),
                    }
                )

    metrics = compute_regression_metrics(y_true, y_pred)
    metrics["by_horizon"] = {h: compute_regression_metrics(horizon_true[h], horizon_pred[h]) for h in sorted(horizon_true, key=float)}
    metrics["current_aux"] = compute_regression_metrics(current_true, current_pred)
    metrics["first_half"] = compute_regression_metrics(half_true["first"], half_pred["first"])
    metrics["second_half"] = compute_regression_metrics(half_true["second"], half_pred["second"])
    metrics.update(compute_sequence_analysis_metrics(sequence_series))
    metrics.update(compute_high_fms_metrics(y_true, y_pred, threshold=high_fms_threshold))
    common_metrics = compute_regression_metrics(common_y_true, common_y_pred)
    common_metrics.update(compute_sequence_analysis_metrics(common_sequence_series))
    common_metrics.update(compute_high_fms_metrics(common_y_true, common_y_pred, threshold=high_fms_threshold))
    metrics["common_window"] = {
        "current_start": common_eval_current_start,
        "current_end": common_eval_current_end,
        "target_start": common_eval_target_start,
        "target_end": common_eval_target_end,
        "max_horizon_seconds": common_eval_max_horizon_seconds,
        "n": common_metrics.get("n", 0),
    }
    for key, value in common_metrics.items():
        metrics[f"common_{key}"] = value
    if use_static:
        metrics["by_gender"] = {
            gender: compute_regression_metrics(subgroup_true.get(gender, []), subgroup_pred.get(gender, []))
            for gender in ("male", "female", "unknown")
            if len(subgroup_true.get(gender, [])) >= 10
        }
        metrics["by_age_bin"] = {}
        ages = np.asarray([record[0] for record in age_records], dtype=np.float64)
        if len(ages) >= 3 and np.unique(ages).size >= 3:
            q1, q2 = np.quantile(ages, [1.0 / 3.0, 2.0 / 3.0])
            bins = {
                f"age_low_<=_{q1:.2f}": (lambda a, q1=q1, q2=q2: a <= q1),
                f"age_mid_{q1:.2f}_{q2:.2f}": (lambda a, q1=q1, q2=q2: q1 < a <= q2),
                f"age_high_>_{q2:.2f}": (lambda a, q1=q1, q2=q2: a > q2),
            }
            for name, pred_fn in bins.items():
                bt: List[float] = []
                bp: List[float] = []
                for age, true_values, pred_values in age_records:
                    if pred_fn(age):
                        bt.extend(true_values)
                        bp.extend(pred_values)
                if len(bt) >= 10:
                    metrics["by_age_bin"][name] = compute_regression_metrics(bt, bp)
        else:
            metrics["age_bin_warning"] = "Not enough distinct ages to compute age-binned metrics."
        metrics["by_mssq_bin"] = {}
        mssqs = np.asarray([record[0] for record in mssq_records], dtype=np.float64)
        if len(mssqs) >= 3 and np.unique(mssqs).size >= 3:
            q1, q2 = np.quantile(mssqs, [1.0 / 3.0, 2.0 / 3.0])
            bins = {
                f"mssq_low_<=_{q1:.2f}": (lambda m, q1=q1, q2=q2: m <= q1),
                f"mssq_mid_{q1:.2f}_{q2:.2f}": (lambda m, q1=q1, q2=q2: q1 < m <= q2),
                f"mssq_high_>_{q2:.2f}": (lambda m, q1=q1, q2=q2: m > q2),
            }
            for name, selector in bins.items():
                bt: List[float] = []
                bp: List[float] = []
                for mssq, target_values, pred_values in mssq_records:
                    if selector(mssq):
                        bt.extend(target_values)
                        bp.extend(pred_values)
                if len(bt) >= 10:
                    metrics["by_mssq_bin"][name] = compute_regression_metrics(bt, bp)
        else:
            metrics["mssq_bin_warning"] = "Not enough distinct MSSQ values to compute MSSQ-binned metrics."
    return {
        "metrics": metrics,
        "series": plot_series,
        "y_true": y_true,
        "y_pred": y_pred,
        "prediction_records": prediction_records,
    }


def save_prediction_plots(series: List[Mapping[str, Any]], plots_dir: str | Path, split_name: str) -> None:
    plots_dir = ensure_dir(plots_dir)
    for idx, item in enumerate(series):
        meta = item["metadata"]
        plt.figure(figsize=(10, 4))
        plt.plot(item["target_time"], item["target"], label="target FMS", linewidth=2)
        plt.plot(item["target_time"], item["prediction"], label="predicted FMS", linewidth=2)
        plt.xlabel("target timestamp (s)")
        plt.ylabel("FMS")
        plt.title(str(meta.get("session_id", f"session_{idx}"))[:80])
        plt.legend()
        plt.tight_layout()
        out = plots_dir / f"{split_name}_{idx:02d}.png"
        plt.savefig(out, dpi=140)
        plt.close()


def save_prediction_csv(records: List[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_training_curves(history: List[Mapping[str, Any]], run_dir: str | Path, plot: bool = True) -> None:
    run_dir = Path(run_dir)
    if not history:
        return
    csv_path = run_dir / "training_curves.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "epoch",
            "loss_total",
            "loss_level",
            "loss_trend",
            "val_mae",
            "val_rmse",
            "seconds",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            train_loss = row.get("train_loss", {})
            val_metrics = row.get("val_metrics", {})
            writer.writerow(
                {
                    "epoch": row.get("epoch"),
                    "loss_total": train_loss.get("loss_total"),
                    "loss_level": train_loss.get("loss_level"),
                    "loss_trend": train_loss.get("loss_trend"),
                    "val_mae": val_metrics.get("mae"),
                    "val_rmse": val_metrics.get("rmse"),
                    "seconds": row.get("seconds"),
                }
            )

    if not plot:
        return
    epochs = [int(row["epoch"]) for row in history]
    loss_total = [float(row["train_loss"]["loss_total"]) for row in history]
    val_mae = [float(row["val_metrics"]["mae"]) for row in history]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, loss_total, label="train loss_total", color="tab:blue")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss_total", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_mae, label="val MAE", color="tab:orange")
    ax2.set_ylabel("val MAE", color="tab:orange")
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=140)
    plt.close(fig)


def train_one_run(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)
    normalize_time_config(config)
    config.setdefault("loss", {})
    config.setdefault("model", {})
    config.setdefault("training", {})
    config.setdefault("task", {})
    config.setdefault("evaluation", {})
    if args.task_mode is not None:
        config["task"]["mode"] = args.task_mode
    if args.rise_horizon_seconds is not None:
        config["task"]["rise_horizon_seconds"] = [float(v) for v in args.rise_horizon_seconds]
    if args.rise_thresholds is not None:
        config["task"]["rise_thresholds"] = [float(v) for v in args.rise_thresholds]
    if args.fall_horizon_seconds is not None:
        config["task"]["fall_horizon_seconds"] = [float(v) for v in args.fall_horizon_seconds]
    if args.fall_thresholds is not None:
        config["task"]["fall_thresholds"] = [float(v) for v in args.fall_thresholds]
    if args.high_risk_horizon_seconds is not None:
        config["task"]["high_risk_horizon_seconds"] = [float(v) for v in args.high_risk_horizon_seconds]
    if args.high_risk_thresholds is not None:
        config["task"]["high_risk_thresholds"] = [float(v) for v in args.high_risk_thresholds]
    if args.high_risk_label_mode is not None:
        config["task"]["high_risk_label_mode"] = str(args.high_risk_label_mode)
    if args.high_risk_onset_past_seconds is not None:
        config["task"]["high_risk_onset_past_seconds"] = float(args.high_risk_onset_past_seconds)
    if args.future_aux_horizon_seconds is not None:
        config["task"]["future_aux_horizon_seconds"] = [float(v) for v in args.future_aux_horizon_seconds]
    if args.high_fms_caution_threshold is not None:
        config["evaluation"]["high_fms_caution_threshold"] = float(args.high_fms_caution_threshold)
    if args.high_fms_warning_threshold is not None:
        config["evaluation"]["high_fms_warning_threshold"] = float(args.high_fms_warning_threshold)
        config["evaluation"]["high_fms_threshold"] = float(args.high_fms_warning_threshold)
    if args.rapid_rise_probability_threshold is not None:
        config["evaluation"]["rapid_rise_probability_threshold"] = float(args.rapid_rise_probability_threshold)
    if args.rapid_drop_probability_threshold is not None:
        config["evaluation"]["rapid_drop_probability_threshold"] = float(args.rapid_drop_probability_threshold)
    if args.final_warning_mode is not None:
        config["evaluation"]["final_warning_mode"] = args.final_warning_mode
    if args.ordinal_bins is not None:
        config["model"]["ordinal_bins"] = [float(v) for v in args.ordinal_bins]
    if args.fms_combine_weight_ordinal is not None:
        config["model"]["fms_combine_weight_ordinal"] = float(args.fms_combine_weight_ordinal)
    if args.current_head_mode is not None:
        config["model"]["current_head_mode"] = args.current_head_mode
    if args.ordinal_head_mode is not None:
        config["model"]["ordinal_head_mode"] = args.ordinal_head_mode
    if args.current_delta_scale is not None:
        config["model"]["current_delta_scale"] = float(args.current_delta_scale)
    if args.current_anchor_delta_growth_scale is not None:
        config["model"]["current_anchor_delta_growth_scale"] = float(args.current_anchor_delta_growth_scale)
    if args.current_anchor_delta_growth_horizon_seconds is not None:
        config["model"]["current_anchor_delta_growth_horizon_seconds"] = float(
            args.current_anchor_delta_growth_horizon_seconds
        )
    if args.current_anchor_delta_growth_power is not None:
        config["model"]["current_anchor_delta_growth_power"] = float(args.current_anchor_delta_growth_power)
    if args.current_trajectory_offsets is not None:
        config["model"]["current_trajectory_offsets"] = [int(v) for v in args.current_trajectory_offsets]
    if args.current_range_guard_low_threshold is not None:
        config["model"]["current_range_guard_low_threshold"] = float(args.current_range_guard_low_threshold)
    if args.current_range_guard_temperature is not None:
        config["model"]["current_range_guard_temperature"] = float(args.current_range_guard_temperature)
    if args.current_range_guard_floor is not None:
        config["model"]["current_range_guard_floor"] = float(args.current_range_guard_floor)
    if args.current_range_guard_cap is not None:
        config["model"]["current_range_guard_cap"] = float(args.current_range_guard_cap)
    if args.current_range_guard_cap_strength is not None:
        config["model"]["current_range_guard_cap_strength"] = float(args.current_range_guard_cap_strength)
    if args.motion_encoder_context is not None:
        config["model"]["motion_encoder_context"] = args.motion_encoder_context
    if args.motion_encoder_layers is not None:
        config["model"]["motion_encoder_layers"] = int(args.motion_encoder_layers)
    if args.risk_temporal_context is not None:
        config["model"]["risk_temporal_context"] = args.risk_temporal_context
    if args.risk_temporal_layers is not None:
        config["model"]["risk_temporal_layers"] = int(args.risk_temporal_layers)
    if args.risk_head_enabled is not None:
        config["model"]["risk_head_enabled"] = bool(args.risk_head_enabled)
    if args.fall_risk_head_enabled is not None:
        config["model"]["fall_risk_head_enabled"] = bool(args.fall_risk_head_enabled)
    if args.high_risk_head_enabled is not None:
        config["model"]["high_risk_head_enabled"] = bool(args.high_risk_head_enabled)
    if args.current_low_suppressor_enabled is not None:
        config["model"]["current_low_suppressor_enabled"] = bool(args.current_low_suppressor_enabled)
    if args.current_low_suppressor_hidden_dim is not None:
        config["model"]["current_low_suppressor_hidden_dim"] = int(args.current_low_suppressor_hidden_dim)
    if args.current_low_suppressor_delta_range is not None:
        config["model"]["current_low_suppressor_delta_range"] = float(args.current_low_suppressor_delta_range)
    if args.current_low_suppressor_gate_init_bias is not None:
        config["model"]["current_low_suppressor_gate_init_bias"] = float(args.current_low_suppressor_gate_init_bias)
    if args.deep_tcn_dilations is not None:
        config["model"]["deep_tcn_dilations"] = [int(v) for v in args.deep_tcn_dilations]
    if args.selection_metric is not None:
        config["training"]["selection_metric"] = args.selection_metric
    if args.selection_mode is not None:
        config["training"]["selection_mode"] = args.selection_mode
    if args.risk_loss_weight is not None:
        config["loss"]["risk_loss_weight"] = float(args.risk_loss_weight)
    if args.fall_loss_weight is not None:
        config["loss"]["fall_loss_weight"] = float(args.fall_loss_weight)
    if args.high_risk_loss_weight is not None:
        config["loss"]["high_risk_loss_weight"] = float(args.high_risk_loss_weight)
    if args.smoothness_weight is not None:
        config["loss"]["smoothness_weight"] = float(args.smoothness_weight)
    if args.anchor_break_weight is not None:
        config["loss"]["anchor_break_weight"] = float(args.anchor_break_weight)
    if args.anchor_break_threshold is not None:
        config["loss"]["anchor_break_threshold"] = float(args.anchor_break_threshold)
    if args.anchor_break_max_weight is not None:
        config["loss"]["anchor_break_max_weight"] = float(args.anchor_break_max_weight)
    if args.transition_weighting:
        config["loss"]["transition_weighting"] = True
    if args.no_transition_weighting:
        config["loss"]["transition_weighting"] = False
    if args.transition_horizon_seconds is not None:
        config["loss"]["transition_horizon_seconds"] = [float(v) for v in args.transition_horizon_seconds]
    if args.transition_drop_threshold is not None:
        config["loss"]["transition_drop_threshold"] = float(args.transition_drop_threshold)
    if args.transition_recovery_threshold is not None:
        config["loss"]["transition_recovery_threshold"] = float(args.transition_recovery_threshold)
    if args.transition_high_threshold is not None:
        config["loss"]["transition_high_threshold"] = float(args.transition_high_threshold)
    if args.transition_low_threshold is not None:
        config["loss"]["transition_low_threshold"] = float(args.transition_low_threshold)
    if args.transition_rise_threshold is not None:
        config["loss"]["transition_rise_threshold"] = float(args.transition_rise_threshold)
    if args.transition_drop_weight is not None:
        config["loss"]["transition_drop_weight"] = float(args.transition_drop_weight)
    if args.transition_recovery_weight is not None:
        config["loss"]["transition_recovery_weight"] = float(args.transition_recovery_weight)
    if args.transition_rise_weight is not None:
        config["loss"]["transition_rise_weight"] = float(args.transition_rise_weight)
    if args.transition_max_weight is not None:
        config["loss"]["transition_max_weight"] = float(args.transition_max_weight)
    if args.lds_weighting:
        config["loss"]["lds_weighting"] = True
    if args.no_lds_weighting:
        config["loss"]["lds_weighting"] = False
    if args.lds_min is not None:
        config["loss"]["lds_min"] = float(args.lds_min)
    if args.lds_max is not None:
        config["loss"]["lds_max"] = float(args.lds_max)
    if args.lds_gamma is not None:
        config["loss"]["lds_gamma"] = float(args.lds_gamma)
    if args.lds_kernel_size is not None:
        config["loss"]["lds_kernel_size"] = int(args.lds_kernel_size)
    if args.lds_sigma is not None:
        config["loss"]["lds_sigma"] = float(args.lds_sigma)
    if args.lds_bin_size is not None:
        config["loss"]["lds_bin_size"] = float(args.lds_bin_size)
    if args.lds_weight_min is not None:
        config["loss"]["lds_weight_min"] = float(args.lds_weight_min)
    if args.lds_weight_max is not None:
        config["loss"]["lds_weight_max"] = float(args.lds_weight_max)
    if args.lds_kernel is not None:
        config["loss"]["lds_kernel"] = args.lds_kernel
    if args.fds_enabled:
        config["model"]["fds_enabled"] = True
    if args.no_fds_enabled:
        config["model"]["fds_enabled"] = False
    if args.fds_kernel is not None:
        config["model"]["fds_kernel"] = args.fds_kernel
    if args.fds_min is not None:
        config["model"]["fds_min"] = float(args.fds_min)
    if args.fds_max is not None:
        config["model"]["fds_max"] = float(args.fds_max)
    if args.fds_num_bins is not None:
        config["model"]["fds_num_bins"] = int(args.fds_num_bins)
    if args.fds_kernel_size is not None:
        config["model"]["fds_kernel_size"] = int(args.fds_kernel_size)
    if args.fds_sigma is not None:
        config["model"]["fds_sigma"] = float(args.fds_sigma)
    if args.fds_momentum is not None:
        config["model"]["fds_momentum"] = float(args.fds_momentum)
    if args.fds_blend is not None:
        config["model"]["fds_blend"] = float(args.fds_blend)
    if args.fds_bin_size is not None:
        config["model"]["fds_bin_size"] = float(args.fds_bin_size)
    if args.fds_start_smooth is not None:
        config["model"]["fds_start_smooth"] = int(args.fds_start_smooth)
    if args.fds_start_update is not None:
        config["model"]["fds_start_update"] = int(args.fds_start_update)
    if args.current_reg_aux_weight is not None:
        config["loss"]["current_reg_aux_weight"] = float(args.current_reg_aux_weight)
    if args.ordinal_loss_weight is not None:
        config["loss"]["ordinal_loss_weight"] = float(args.ordinal_loss_weight)
    if args.ordinal_loss_mode is not None:
        config["loss"]["ordinal_loss_mode"] = args.ordinal_loss_mode
    if args.ordinal_soft_label_sigma is not None:
        config["loss"]["ordinal_soft_label_sigma"] = float(args.ordinal_soft_label_sigma)
    if args.ordinal_soft_label_kernel is not None:
        config["loss"]["ordinal_soft_label_kernel"] = args.ordinal_soft_label_kernel
    if args.ordinal_ev_loss_weight is not None:
        config["loss"]["ordinal_ev_loss_weight"] = float(args.ordinal_ev_loss_weight)
    if args.ordinal_low_weight is not None:
        config["loss"]["ordinal_low_weight"] = float(args.ordinal_low_weight)
    if args.ordinal_low_threshold is not None:
        config["loss"]["ordinal_low_threshold"] = float(args.ordinal_low_threshold)
    if args.ordinal_slace_alpha is not None:
        config["loss"]["ordinal_slace_alpha"] = float(args.ordinal_slace_alpha)
    if args.ordinal_slace_proximity is not None:
        config["loss"]["ordinal_slace_proximity"] = bool(args.ordinal_slace_proximity)
    if args.ordinal_slace_normalize_proximity is not None:
        config["loss"]["ordinal_slace_normalize_proximity"] = bool(args.ordinal_slace_normalize_proximity)
    if args.ordinal_slace_count_smoothing is not None:
        config["loss"]["ordinal_slace_count_smoothing"] = float(args.ordinal_slace_count_smoothing)
    if args.coarse_band_bins is not None:
        config["model"]["coarse_band_bins"] = [float(v) for v in args.coarse_band_bins]
    if args.coarse_band_loss_weight is not None:
        config["loss"]["coarse_band_loss_weight"] = float(args.coarse_band_loss_weight)
    if args.coarse_residual_loss_weight is not None:
        config["loss"]["coarse_residual_loss_weight"] = float(args.coarse_residual_loss_weight)
    if args.regime_head_enabled is not None:
        config["model"]["regime_head_enabled"] = bool(args.regime_head_enabled)
    if args.regime_loss_weight is not None:
        config["loss"]["regime_loss_weight"] = float(args.regime_loss_weight)
    if args.regime_delta_slow_threshold is not None:
        config["loss"]["regime_delta_slow_threshold"] = float(args.regime_delta_slow_threshold)
    if args.regime_delta_rapid_threshold is not None:
        config["loss"]["regime_delta_rapid_threshold"] = float(args.regime_delta_rapid_threshold)
    if args.regime_high_threshold is not None:
        config["loss"]["regime_high_threshold"] = float(args.regime_high_threshold)
    if args.uncertainty_head_enabled is not None:
        config["model"]["uncertainty_head_enabled"] = bool(args.uncertainty_head_enabled)
    if args.uncertainty_loss_weight is not None:
        config["loss"]["uncertainty_loss_weight"] = float(args.uncertainty_loss_weight)
    if args.motion_pretrain_checkpoint is not None:
        config["model"]["motion_pretrain_checkpoint"] = args.motion_pretrain_checkpoint
    if args.risk_pos_weight is not None:
        config["loss"]["risk_pos_weight"] = args.risk_pos_weight
    if args.fall_risk_pos_weight is not None:
        config["loss"]["fall_risk_pos_weight"] = args.fall_risk_pos_weight
    if args.high_risk_pos_weight is not None:
        config["loss"]["high_risk_pos_weight"] = args.high_risk_pos_weight
    if args.rollout_mode is not None:
        config["model"]["state_feedback_mode"] = "predicted_current" if args.rollout_mode == "predicted" else "none"
    if args.motion_stats_branch is not None:
        config["model"]["motion_stats_branch"] = bool(args.motion_stats_branch)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["training"]["batch_size"] = int(args.batch_size)
    if args.participant_balanced_sampling is not None:
        config["training"]["participant_balanced_sampling"] = bool(args.participant_balanced_sampling)
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = float(args.learning_rate)
    if args.weight_decay is not None:
        config["training"]["weight_decay"] = float(args.weight_decay)
    if args.patience is not None:
        config["training"]["patience"] = int(args.patience)
    if args.seed is not None:
        config["training"]["seed"] = int(args.seed)
    if args.num_workers is not None:
        config["training"]["num_workers"] = int(args.num_workers)
    if args.init_checkpoint is not None:
        config["training"]["init_checkpoint"] = args.init_checkpoint
    if args.freeze_loaded_parameters is not None:
        config["training"]["freeze_loaded_parameters"] = bool(args.freeze_loaded_parameters)
    if args.trainable_parameter_patterns is not None:
        config["training"]["trainable_parameter_patterns"] = list(args.trainable_parameter_patterns)
    if args.runs_dir is not None:
        config["runs_dir"] = args.runs_dir
    if args.max_train_batches is not None:
        config["training"]["max_train_batches"] = int(args.max_train_batches)
    if args.max_eval_batches is not None:
        config["training"]["max_eval_batches"] = int(args.max_eval_batches)
    if args.calibration_seconds is not None:
        config["data"]["calibration_seconds"] = float(args.calibration_seconds)
    if args.horizon_seconds is not None:
        config["data"]["horizon_seconds"] = float(args.horizon_seconds)
    if args.recent_window_seconds is not None:
        config["data"]["recent_window_seconds"] = float(args.recent_window_seconds)
        config["data"]["recent_seconds"] = float(args.recent_window_seconds)
    if args.max_session_points is not None:
        config["data"]["max_session_points"] = int(args.max_session_points)
    if args.head_channel_mode is not None:
        config["data"]["head_channel_mode"] = normalize_head_channel_mode(args.head_channel_mode)
    else:
        config["data"]["head_channel_mode"] = normalize_head_channel_mode(config["data"].get("head_channel_mode", "all"))
    if args.calibration_residual_features_path is not None:
        config["data"]["calibration_residual_features_path"] = (
            list(args.calibration_residual_features_path)
            if len(args.calibration_residual_features_path) > 1
            else str(args.calibration_residual_features_path[0])
        )
    if args.require_calibration_residual_features is not None:
        config["data"]["require_calibration_residual_features"] = bool(args.require_calibration_residual_features)
    normalize_time_config(config)
    if args.no_film:
        config["model"]["no_film"] = True
    if args.no_recent_encoder:
        config["model"]["no_recent_encoder"] = True
    if args.recent_encoder is not None:
        config["model"]["recent_encoder"] = args.recent_encoder
    if args.recent_attn_heads is not None:
        config["model"]["recent_attn_heads"] = int(args.recent_attn_heads)
    if args.recent_attn_layers is not None:
        config["model"]["recent_attn_layers"] = int(args.recent_attn_layers)
    if args.recent_attn_dropout is not None:
        config["model"]["recent_attn_dropout"] = float(args.recent_attn_dropout)
    if args.no_aux_now:
        config["model"]["no_aux_now"] = True
    if args.loss_mode is not None:
        config["loss"]["mode"] = args.loss_mode
    if args.loss_type is not None:
        config["loss"]["type"] = args.loss_type
    if args.trend_weight is not None:
        config["loss"]["trend_weight"] = float(args.trend_weight)
    if args.horizon_loss_weights is not None:
        config["loss"]["horizon_weights"] = [float(v) for v in args.horizon_loss_weights]
    if args.change_weight is not None:
        config["loss"]["change_weight"] = float(args.change_weight)
    if args.high_target_weight is not None:
        config["loss"]["high_target_weight"] = float(args.high_target_weight)
    if args.high_target_threshold is not None:
        config["loss"]["high_target_threshold"] = float(args.high_target_threshold)
    if args.low_target_weight is not None:
        config["loss"]["low_target_weight"] = float(args.low_target_weight)
    if args.low_target_threshold is not None:
        config["loss"]["low_target_threshold"] = float(args.low_target_threshold)
    if args.dual_aux_alpha is not None:
        config["loss"]["dual_aux_alpha"] = float(args.dual_aux_alpha)
    if args.dual_aux_beta is not None:
        config["loss"]["dual_aux_beta"] = float(args.dual_aux_beta)
    if args.change_aux_weight is not None:
        config["loss"]["change_aux_weight"] = float(args.change_aux_weight)
    if args.change_aux_threshold is not None:
        config["loss"]["change_aux_threshold"] = float(args.change_aux_threshold)
    if args.current_aux_weight is not None:
        config["loss"]["current_aux_weight"] = float(args.current_aux_weight)
    if args.current_delta_aux_weight is not None:
        config["loss"]["current_delta_aux_weight"] = float(args.current_delta_aux_weight)
    if args.future_aux_loss_weight is not None:
        config["loss"]["future_aux_loss_weight"] = float(args.future_aux_loss_weight)
    if args.delta_aux_loss_weight is not None:
        config["loss"]["delta_aux_loss_weight"] = float(args.delta_aux_loss_weight)
    if args.event_aux_loss_weight is not None:
        config["loss"]["event_aux_loss_weight"] = float(args.event_aux_loss_weight)
    if args.event_delta_threshold is not None:
        config["loss"]["event_delta_threshold"] = float(args.event_delta_threshold)
    if args.trajectory_loss_weight is not None:
        config["loss"]["trajectory_loss_weight"] = float(args.trajectory_loss_weight)
    if args.trajectory_decoder_loss_weight is not None:
        config["loss"]["trajectory_decoder_loss_weight"] = float(args.trajectory_decoder_loss_weight)
    if args.trajectory_delta_seconds is not None:
        config["loss"]["trajectory_delta_seconds"] = [float(v) for v in args.trajectory_delta_seconds]
    if args.trajectory_delta_weight is not None:
        config["loss"]["trajectory_delta_weight"] = float(args.trajectory_delta_weight)
    if args.trajectory_centered_weight is not None:
        config["loss"]["trajectory_centered_weight"] = float(args.trajectory_centered_weight)
    if args.trajectory_range_weight is not None:
        config["loss"]["trajectory_range_weight"] = float(args.trajectory_range_weight)
    if args.trajectory_loss_type is not None:
        config["loss"]["trajectory_loss_type"] = args.trajectory_loss_type
    if args.trajectory_min_points is not None:
        config["loss"]["trajectory_min_points"] = int(args.trajectory_min_points)
    if args.session_affine_scale_regularization_weight is not None:
        config["loss"]["session_affine_scale_regularization_weight"] = float(args.session_affine_scale_regularization_weight)
    if args.session_affine_bias_regularization_weight is not None:
        config["loss"]["session_affine_bias_regularization_weight"] = float(args.session_affine_bias_regularization_weight)
    if args.calibration_residual_regularization_weight is not None:
        config["loss"]["calibration_residual_regularization_weight"] = float(args.calibration_residual_regularization_weight)
    if args.low_overprediction_weight is not None:
        config["loss"]["low_overprediction_weight"] = float(args.low_overprediction_weight)
    if args.high_underprediction_weight is not None:
        config["loss"]["high_underprediction_weight"] = float(args.high_underprediction_weight)
    if args.low_overprediction_threshold is not None:
        config["loss"]["low_overprediction_threshold"] = float(args.low_overprediction_threshold)
    if args.high_underprediction_threshold is not None:
        config["loss"]["high_underprediction_threshold"] = float(args.high_underprediction_threshold)
    if args.low_suppressor_gate_loss_weight is not None:
        config["loss"]["low_suppressor_gate_loss_weight"] = float(args.low_suppressor_gate_loss_weight)
    if args.low_suppressor_threshold is not None:
        config["loss"]["low_suppressor_threshold"] = float(args.low_suppressor_threshold)
    if args.low_suppressor_gate_pos_weight is not None:
        config["loss"]["low_suppressor_gate_pos_weight"] = float(args.low_suppressor_gate_pos_weight)
    if args.low_suppressor_gate_target_mode is not None:
        config["loss"]["low_suppressor_gate_target_mode"] = str(args.low_suppressor_gate_target_mode)
    if args.low_suppressor_anchor_threshold is not None:
        config["loss"]["low_suppressor_anchor_threshold"] = float(args.low_suppressor_anchor_threshold)
    if args.low_suppressor_recovery_delta is not None:
        config["loss"]["low_suppressor_recovery_delta"] = float(args.low_suppressor_recovery_delta)
    if args.low_suppressor_correction_regularization_weight is not None:
        config["loss"]["low_suppressor_correction_regularization_weight"] = float(
            args.low_suppressor_correction_regularization_weight
        )
    if args.anchor_gate_loss_weight is not None:
        config["loss"]["anchor_gate_loss_weight"] = float(args.anchor_gate_loss_weight)
    if args.anchor_gate_threshold is not None:
        config["loss"]["anchor_gate_threshold"] = float(args.anchor_gate_threshold)
    if args.anchor_gate_pos_weight is not None:
        config["loss"]["anchor_gate_pos_weight"] = float(args.anchor_gate_pos_weight)
    if args.session_aux_weight is not None:
        config["loss"]["session_aux_weight"] = float(args.session_aux_weight)
    if args.session_aux_loss_type is not None:
        config["loss"]["session_aux_loss_type"] = args.session_aux_loss_type
    if args.teacher_checkpoint is not None:
        config["loss"]["teacher_checkpoint"] = args.teacher_checkpoint
    if args.teacher_distill_weight is not None:
        config["loss"]["teacher_distill_weight"] = float(args.teacher_distill_weight)
    if args.teacher_distill_loss_type is not None:
        config["loss"]["teacher_distill_loss_type"] = args.teacher_distill_loss_type
    if args.teacher_delta_distill_weight is not None:
        config["loss"]["teacher_delta_distill_weight"] = float(args.teacher_delta_distill_weight)
    if args.teacher_delta_distill_loss_type is not None:
        config["loss"]["teacher_delta_distill_loss_type"] = args.teacher_delta_distill_loss_type
    if args.teacher_delta_distill_start_epoch is not None:
        config["loss"]["teacher_delta_distill_start_epoch"] = int(args.teacher_delta_distill_start_epoch)
    if args.teacher_delta_distill_warmup_epochs is not None:
        config["loss"]["teacher_delta_distill_warmup_epochs"] = int(args.teacher_delta_distill_warmup_epochs)
    if args.teacher_repr_distill_weight is not None:
        config["loss"]["teacher_repr_distill_weight"] = float(args.teacher_repr_distill_weight)
    if args.teacher_repr_distill_loss_type is not None:
        config["loss"]["teacher_repr_distill_loss_type"] = args.teacher_repr_distill_loss_type
    if args.teacher_repr_distill_start_epoch is not None:
        config["loss"]["teacher_repr_distill_start_epoch"] = int(args.teacher_repr_distill_start_epoch)
    if args.teacher_repr_distill_warmup_epochs is not None:
        config["loss"]["teacher_repr_distill_warmup_epochs"] = int(args.teacher_repr_distill_warmup_epochs)
    if args.high_fms_threshold is not None:
        config.setdefault("evaluation", {})["high_fms_threshold"] = float(args.high_fms_threshold)
    if args.use_static:
        config["model"]["use_static"] = True
        config["data"]["use_static"] = True
    if args.no_static:
        config["model"]["use_static"] = False
        config["data"]["use_static"] = False
    for key in (
        "d_model",
        "hidden_dim",
        "mlp_layers",
        "gru_layers",
        "branch_dropout",
        "anchor_dropout",
        "delta_scale",
        "kernel_size",
        "dropout",
        "transformer_layers",
        "transformer_heads",
        "transformer_ff_dim",
        "pooling",
        "anchor_mode",
        "anchor_interval_seconds",
        "fms_context_mode",
        "predict_delta_from_anchor",
        "multi_horizon",
        "horizon_set",
        "per_horizon_heads",
        "horizon_encoder_dim",
        "horizon_context_mode",
        "start_fms_context_mode",
        "static_context_mode",
        "static_hidden_dim",
        "static_dropout",
        "forecast_head_mode",
        "horizon_head_mode",
        "horizon_head_hidden_dim",
        "motion_feature_mode",
        "current_trajectory_offsets",
        "coarse_band_bins",
        "coarse_residual_head_enabled",
        "coarse_residual_range",
        "coarse_residual_combine_weight",
        "regime_head_enabled",
        "regime_class_count",
        "uncertainty_head_enabled",
        "motion_pretrain_checkpoint",
        "stream_time_features",
        "stream_context_mode",
        "stream_prepend_calibration",
        "stream_calib_condition_mode",
        "stream_calib_condition_strength",
        "calib_summary_features",
        "calibration_fusion_mode",
        "calibration_fusion_hidden_dim",
        "calibration_fusion_output_dim",
        "calibration_encoder_mode",
        "state_feedback_mode",
        "deep_tcn_dilations",
        "calibration_tcn_adaptive_dilations",
        "calibration_tcn_max_padding_steps",
        "calibration_tcn_max_padding_fraction",
        "decoder_hidden_dim",
        "decoder_context_mode",
        "decoder_temporal_context",
        "decoder_temporal_layers",
        "calib_fms_dropout",
        "calibration_end_fms_dropout",
        "current_session_affine_head_enabled",
        "current_session_affine_hidden_dim",
        "current_session_affine_scale_range",
        "current_session_affine_bias_range",
        "current_affine_head_enabled",
        "current_affine_hidden_dim",
        "current_affine_scale_range",
        "current_affine_bias_range",
        "current_binned_affine_head_enabled",
        "current_binned_affine_anchor_bins",
        "current_binned_affine_pred_bins",
        "current_binned_affine_time_bins",
        "current_binned_affine_scale_range",
        "current_binned_affine_bias_range",
        "calibration_residual_adapter_enabled",
        "calibration_residual_feature_dim",
        "calibration_residual_adapter_hidden_dim",
        "calibration_residual_adapter_mode",
        "calibration_residual_delta_range",
        "calibration_residual_decay_seconds",
        "calibration_residual_gate_low_threshold",
        "calibration_residual_gate_high_threshold",
        "calibration_residual_gate_anchor_threshold",
        "calibration_residual_gate_temperature",
        "calibration_summary_fusion_enabled",
        "calibration_summary_fusion_feature_dim",
        "calibration_summary_fusion_hidden_dim",
        "calibration_summary_fusion_mode",
        "calibration_summary_fusion_strength",
        "session_context_mode",
        "change_aux_head",
        "calib_dilations",
        "recent_dilations",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config["model"][key] = value
    if args.static_features:
        config["data"]["static_features"] = args.static_features
    config["data"]["static_features"] = normalize_static_features(config["data"].get("static_features", ["age", "gender"]))
    if args.gender_encoding is not None:
        config["data"]["gender_encoding"] = normalize_gender_encoding(args.gender_encoding)
    else:
        config["data"]["gender_encoding"] = normalize_gender_encoding(config["data"].get("gender_encoding", "category3"))
    config.setdefault("model", {})
    config["model"]["static_features"] = list(config["data"]["static_features"])
    config["model"]["gender_encoding"] = config["data"]["gender_encoding"]
    config["model"]["static_dim"] = static_feature_dim(
        config["data"]["static_features"],
        gender_encoding=config["data"]["gender_encoding"],
    )
    if args.allow_missing_static:
        config["data"]["allow_missing_static"] = True

    loss_mode = str(config["loss"].get("mode", "level_only"))
    if loss_mode == "level_plus_trend":
        loss_mode = "level_trend_raw"
        config["loss"]["mode"] = loss_mode
    loss_type = str(config["loss"].get("type", "smooth_l1"))
    trend_weight = float(config["loss"].get("trend_weight", 0.0 if loss_mode == "level_only" else 0.1))
    horizon_weights = config["loss"].get("horizon_weights")
    change_weight = float(config["loss"].get("change_weight", 0.0))
    high_target_weight = float(config["loss"].get("high_target_weight", 0.0))
    high_target_threshold = float(config["loss"].get("high_target_threshold", 0.5))
    low_target_weight = float(config["loss"].get("low_target_weight", 0.0))
    low_target_threshold = float(config["loss"].get("low_target_threshold", 0.15))
    dual_aux_alpha = float(config["loss"].get("dual_aux_alpha", 0.0))
    dual_aux_beta = float(config["loss"].get("dual_aux_beta", 0.0))
    change_aux_weight = float(config["loss"].get("change_aux_weight", 0.0))
    change_aux_threshold = float(config["loss"].get("change_aux_threshold", 0.1))
    current_aux_weight = float(config["loss"].get("current_aux_weight", 0.0))
    current_delta_aux_weight = float(config["loss"].get("current_delta_aux_weight", 0.0))
    session_aux_weight = float(config["loss"].get("session_aux_weight", 0.0))
    session_aux_loss_type = str(config["loss"].get("session_aux_loss_type", "smooth_l1"))
    teacher_checkpoint = config["loss"].get("teacher_checkpoint")
    teacher_distill_weight = float(config["loss"].get("teacher_distill_weight", 0.0))
    teacher_distill_loss_type = str(config["loss"].get("teacher_distill_loss_type", "smooth_l1"))
    teacher_delta_distill_weight = float(config["loss"].get("teacher_delta_distill_weight", 0.0))
    teacher_delta_distill_loss_type = str(config["loss"].get("teacher_delta_distill_loss_type", "smooth_l1"))
    teacher_delta_distill_start_epoch = int(config["loss"].get("teacher_delta_distill_start_epoch", 1))
    teacher_delta_distill_warmup_epochs = int(config["loss"].get("teacher_delta_distill_warmup_epochs", 0))
    teacher_repr_distill_weight = float(config["loss"].get("teacher_repr_distill_weight", 0.0))
    teacher_repr_distill_loss_type = str(config["loss"].get("teacher_repr_distill_loss_type", "smooth_l1"))
    teacher_repr_distill_start_epoch = int(config["loss"].get("teacher_repr_distill_start_epoch", 1))
    teacher_repr_distill_warmup_epochs = int(config["loss"].get("teacher_repr_distill_warmup_epochs", 0))
    task_mode = str(config.get("task", {}).get("mode", config.get("task_mode", "future_forecast"))).lower()
    if task_mode not in {"future_forecast", "online_current_risk"}:
        raise ValueError("task mode must be either future_forecast or online_current_risk.")
    rise_horizon_seconds = [float(v) for v in config.get("task", {}).get("rise_horizon_seconds", [5.0, 10.0])]
    rise_thresholds = [float(v) for v in config.get("task", {}).get("rise_thresholds", [2.0, 3.0])]
    if len(rise_horizon_seconds) != len(rise_thresholds):
        raise ValueError("task.rise_horizon_seconds and task.rise_thresholds must have the same length.")
    fall_horizon_seconds = [float(v) for v in config.get("task", {}).get("fall_horizon_seconds", rise_horizon_seconds)]
    fall_thresholds = [float(v) for v in config.get("task", {}).get("fall_thresholds", rise_thresholds)]
    if len(fall_horizon_seconds) != len(fall_thresholds):
        raise ValueError("task.fall_horizon_seconds and task.fall_thresholds must have the same length.")
    if any(value <= 0 for value in fall_horizon_seconds):
        raise ValueError("task.fall_horizon_seconds must contain only positive seconds.")
    high_risk_horizon_seconds = [float(v) for v in config.get("task", {}).get("high_risk_horizon_seconds", [])]
    high_risk_thresholds = [float(v) for v in config.get("task", {}).get("high_risk_thresholds", [])]
    high_risk_label_mode = str(config.get("task", {}).get("high_risk_label_mode", "future_any")).lower()
    if high_risk_label_mode not in {"future_any", "current_below", "onset", "current_or_future"}:
        raise ValueError(
            "task.high_risk_label_mode must be one of: future_any, current_below, onset, current_or_future."
        )
    high_risk_onset_past_seconds = float(config.get("task", {}).get("high_risk_onset_past_seconds", 0.0))
    if high_risk_onset_past_seconds < 0:
        raise ValueError("task.high_risk_onset_past_seconds must be nonnegative.")
    if any(value <= 0 for value in high_risk_horizon_seconds):
        raise ValueError("task.high_risk_horizon_seconds must contain only positive seconds.")
    if any(value < 0.0 or value > 20.0 for value in high_risk_thresholds):
        raise ValueError("task.high_risk_thresholds must be on the raw DenseFMS 0-20 scale.")
    future_aux_horizon_seconds = [float(v) for v in config.get("task", {}).get("future_aux_horizon_seconds", [])]
    if any(value <= 0 for value in future_aux_horizon_seconds):
        raise ValueError("task.future_aux_horizon_seconds must contain only positive seconds.")
    ordinal_bins = [float(v) for v in config["model"].get("ordinal_bins", [0, 2, 4, 6, 8, 10, 12, 15, 20])]
    coarse_band_bins = [float(v) for v in config["model"].get("coarse_band_bins", [])]
    current_reg_aux_weight = float(config["loss"].get("current_reg_aux_weight", 0.4))
    ordinal_loss_weight = float(config["loss"].get("ordinal_loss_weight", 0.6))
    ordinal_head_mode = str(config["model"].get("ordinal_head_mode") or "").lower()
    if ordinal_head_mode == "corn":
        default_ordinal_loss_mode = "corn_bce"
    elif ordinal_head_mode in {"cumulative", "clm", "coral"}:
        default_ordinal_loss_mode = "cumulative_bce"
    else:
        default_ordinal_loss_mode = "ce"
    ordinal_loss_mode = str(config["loss"].get("ordinal_loss_mode", default_ordinal_loss_mode)).lower()
    valid_ordinal_loss_modes = {
        "ce",
        "cross_entropy",
        "cumulative",
        "cumulative_bce",
        "coral_bce",
        "clm_nll",
        "cumulative_nll",
        "corn",
        "corn_bce",
        "soft_ce",
        "soft_label_ce",
        "unimodal_soft_ce",
        "slace",
        "slace_prox",
        "slace_paper",
        "slace_index",
        "slace_no_prox",
        "slace_norm_prox",
        "slace_prox_norm",
        "cdf_bce",
        "ordinal_cdf_bce",
        "oce",
        "oce_ts",
        "tpt_oce",
        "soft_oce_ts",
        "emd",
        "emd2",
        "soft_emd",
        "dldl_emd",
        "dldl",
        "wasserstein",
    }
    if ordinal_loss_mode not in valid_ordinal_loss_modes:
        raise ValueError(
            "loss.ordinal_loss_mode must be one of: ce, cross_entropy, cumulative, cumulative_bce, "
            "coral_bce, clm_nll, corn_bce, soft_ce, unimodal_soft_ce, slace, slace_index, "
            "slace_norm_prox, cdf_bce, oce, "
            "oce_ts, tpt_oce, soft_oce_ts, emd, dldl_emd."
        )
    if ordinal_loss_mode == "cross_entropy":
        ordinal_loss_mode = "ce"
    if ordinal_head_mode in {"cumulative", "clm", "coral"} and ordinal_loss_mode not in {
        "ce",
        "cumulative",
        "cumulative_bce",
        "coral_bce",
        "clm_nll",
        "cumulative_nll",
    }:
        raise ValueError(
            "model.ordinal_head_mode='cumulative'/'clm'/'coral' requires a cumulative/CLM-compatible ordinal loss."
        )
    if ordinal_head_mode == "corn" and ordinal_loss_mode not in {"corn", "corn_bce"}:
        raise ValueError("model.ordinal_head_mode='corn' requires loss.ordinal_loss_mode='corn_bce'.")
    if ordinal_head_mode in {"", "softmax"} and ordinal_loss_mode in {
        "cumulative",
        "cumulative_bce",
        "coral_bce",
        "clm_nll",
        "cumulative_nll",
        "corn",
        "corn_bce",
    }:
        raise ValueError("This ordinal loss mode requires a cumulative/CLM/CORAL/CORN ordinal head.")
    ordinal_soft_label_sigma = float(config["loss"].get("ordinal_soft_label_sigma", 1.0))
    if ordinal_soft_label_sigma <= 0:
        raise ValueError("loss.ordinal_soft_label_sigma must be positive.")
    ordinal_soft_label_kernel = str(config["loss"].get("ordinal_soft_label_kernel", "gaussian")).lower()
    if ordinal_soft_label_kernel not in {"gaussian", "laplace", "laplacian", "exponential", "triangular", "linear"}:
        raise ValueError("loss.ordinal_soft_label_kernel must be one of: gaussian, laplace, triangular.")
    ordinal_ev_loss_weight = float(config["loss"].get("ordinal_ev_loss_weight", 0.0))
    ordinal_low_weight = float(config["loss"].get("ordinal_low_weight", 1.0))
    ordinal_low_threshold = float(config["loss"].get("ordinal_low_threshold", 2.0))
    if ordinal_low_weight < 0:
        raise ValueError("loss.ordinal_low_weight must be nonnegative.")
    ordinal_slace_alpha = float(config["loss"].get("ordinal_slace_alpha", ordinal_soft_label_sigma))
    if ordinal_slace_alpha <= 0:
        raise ValueError("loss.ordinal_slace_alpha must be positive.")
    ordinal_slace_proximity = bool(config["loss"].get("ordinal_slace_proximity", True))
    ordinal_slace_normalize_proximity = bool(config["loss"].get("ordinal_slace_normalize_proximity", False))
    ordinal_slace_count_smoothing = float(config["loss"].get("ordinal_slace_count_smoothing", 1e-6))
    if ordinal_slace_count_smoothing < 0:
        raise ValueError("loss.ordinal_slace_count_smoothing must be nonnegative.")
    coarse_band_loss_weight = float(config["loss"].get("coarse_band_loss_weight", 0.0))
    if coarse_band_loss_weight > 0 and not coarse_band_bins:
        raise ValueError("loss.coarse_band_loss_weight > 0 requires model.coarse_band_bins.")
    coarse_residual_loss_weight = float(config["loss"].get("coarse_residual_loss_weight", 0.0))
    if coarse_residual_loss_weight > 0 and not bool(config["model"].get("coarse_residual_head_enabled", False)):
        raise ValueError("loss.coarse_residual_loss_weight > 0 requires model.coarse_residual_head_enabled=true.")
    regime_loss_weight = float(config["loss"].get("regime_loss_weight", 0.0))
    regime_delta_slow_threshold = float(config["loss"].get("regime_delta_slow_threshold", 0.5))
    regime_delta_rapid_threshold = float(config["loss"].get("regime_delta_rapid_threshold", 2.0))
    regime_high_threshold = float(config["loss"].get("regime_high_threshold", 12.0))
    current_head_mode_for_checks = str(config["model"].get("current_head_mode", "basic")).lower()
    if (
        regime_loss_weight > 0
        and not bool(config["model"].get("regime_head_enabled", False))
        and current_head_mode_for_checks != "regime_gated"
    ):
        raise ValueError("loss.regime_loss_weight > 0 requires model.regime_head_enabled=true or current_head_mode='regime_gated'.")
    uncertainty_loss_weight = float(config["loss"].get("uncertainty_loss_weight", 0.0))
    if uncertainty_loss_weight > 0 and not bool(config["model"].get("uncertainty_head_enabled", False)):
        raise ValueError("loss.uncertainty_loss_weight > 0 requires model.uncertainty_head_enabled=true.")
    risk_loss_weight = float(config["loss"].get("risk_loss_weight", 1.0))
    fall_loss_weight = float(config["loss"].get("fall_loss_weight", 0.0))
    high_risk_loss_weight = float(config["loss"].get("high_risk_loss_weight", 0.0))
    smoothness_weight = float(config["loss"].get("smoothness_weight", 0.02))
    future_aux_loss_weight = float(config["loss"].get("future_aux_loss_weight", 0.0))
    delta_aux_loss_weight = float(config["loss"].get("delta_aux_loss_weight", 0.0))
    event_aux_loss_weight = float(config["loss"].get("event_aux_loss_weight", 0.0))
    event_delta_threshold = float(config["loss"].get("event_delta_threshold", 1.0))
    trajectory_loss_weight = float(config["loss"].get("trajectory_loss_weight", 0.0))
    trajectory_decoder_loss_weight = float(config["loss"].get("trajectory_decoder_loss_weight", 0.0))
    trajectory_delta_seconds_raw = config["loss"].get("trajectory_delta_seconds")
    trajectory_delta_seconds = (
        [float(v) for v in trajectory_delta_seconds_raw]
        if trajectory_delta_seconds_raw is not None
        else list(rise_horizon_seconds)
    )
    trajectory_delta_weight = float(config["loss"].get("trajectory_delta_weight", 1.0))
    trajectory_centered_weight = float(config["loss"].get("trajectory_centered_weight", 0.5))
    trajectory_range_weight = float(config["loss"].get("trajectory_range_weight", 0.2))
    trajectory_loss_type = str(config["loss"].get("trajectory_loss_type", loss_type)).lower()
    trajectory_min_points = int(config["loss"].get("trajectory_min_points", 4))
    session_affine_scale_regularization_weight = float(
        config["loss"].get("session_affine_scale_regularization_weight", 0.0)
    )
    session_affine_bias_regularization_weight = float(
        config["loss"].get("session_affine_bias_regularization_weight", 0.0)
    )
    calibration_residual_regularization_weight = float(
        config["loss"].get("calibration_residual_regularization_weight", 0.0)
    )
    low_overprediction_weight = float(config["loss"].get("low_overprediction_weight", 0.0))
    high_underprediction_weight = float(config["loss"].get("high_underprediction_weight", 0.0))
    low_overprediction_threshold = float(config["loss"].get("low_overprediction_threshold", 2.0))
    high_underprediction_threshold = float(config["loss"].get("high_underprediction_threshold", 15.0))
    low_suppressor_gate_loss_weight = float(config["loss"].get("low_suppressor_gate_loss_weight", 0.0))
    low_suppressor_threshold = float(config["loss"].get("low_suppressor_threshold", 2.0))
    low_suppressor_gate_pos_weight = float(config["loss"].get("low_suppressor_gate_pos_weight", 1.0))
    low_suppressor_gate_target_mode = str(config["loss"].get("low_suppressor_gate_target_mode", "low")).strip().lower()
    if low_suppressor_gate_target_mode not in {"low", "recovery_low", "anchor_drop_low"}:
        raise ValueError("loss.low_suppressor_gate_target_mode must be one of: low, recovery_low, anchor_drop_low.")
    low_suppressor_anchor_threshold = float(config["loss"].get("low_suppressor_anchor_threshold", 5.0))
    low_suppressor_recovery_delta = float(config["loss"].get("low_suppressor_recovery_delta", 4.0))
    low_suppressor_correction_regularization_weight = float(
        config["loss"].get("low_suppressor_correction_regularization_weight", 0.0)
    )
    anchor_gate_loss_weight = float(config["loss"].get("anchor_gate_loss_weight", 0.0))
    anchor_gate_threshold = float(config["loss"].get("anchor_gate_threshold", 10.0))
    anchor_gate_pos_weight = float(config["loss"].get("anchor_gate_pos_weight", 1.0))
    anchor_break_weight = float(config["loss"].get("anchor_break_weight", 0.0))
    anchor_break_threshold = float(config["loss"].get("anchor_break_threshold", 4.0))
    anchor_break_max_weight = float(config["loss"].get("anchor_break_max_weight", 3.0))
    transition_weighting = bool(config["loss"].get("transition_weighting", False))
    transition_horizon_seconds_raw = config["loss"].get("transition_horizon_seconds")
    transition_horizon_seconds = (
        [float(v) for v in transition_horizon_seconds_raw]
        if transition_horizon_seconds_raw is not None
        else list(rise_horizon_seconds)
    )
    transition_drop_threshold = float(config["loss"].get("transition_drop_threshold", 2.0))
    transition_recovery_threshold = float(config["loss"].get("transition_recovery_threshold", 3.0))
    transition_high_threshold = float(config["loss"].get("transition_high_threshold", 8.0))
    transition_low_threshold = float(config["loss"].get("transition_low_threshold", 5.0))
    transition_rise_threshold = float(config["loss"].get("transition_rise_threshold", 3.0))
    transition_drop_weight = float(config["loss"].get("transition_drop_weight", 2.0))
    transition_recovery_weight = float(config["loss"].get("transition_recovery_weight", 3.0))
    transition_rise_weight = float(config["loss"].get("transition_rise_weight", 1.5))
    transition_max_weight = float(config["loss"].get("transition_max_weight", 4.0))
    lds_weighting = bool(config["loss"].get("lds_weighting", False))
    lds_min = float(config["loss"].get("lds_min", 0.0))
    lds_max = float(config["loss"].get("lds_max", 20.0))
    lds_bin_size = float(config["loss"].get("lds_bin_size", 1.0))
    lds_kernel = str(config["loss"].get("lds_kernel", "gaussian"))
    lds_kernel_size = int(config["loss"].get("lds_kernel_size", 5))
    lds_sigma = float(config["loss"].get("lds_sigma", 2.0))
    lds_gamma = float(config["loss"].get("lds_gamma", 0.5))
    lds_weight_min = float(config["loss"].get("lds_weight_min", 0.5))
    lds_weight_max = float(config["loss"].get("lds_weight_max", 3.0))
    fds_enabled = bool(config["model"].get("fds_enabled", False))
    fds_min = float(config["model"].get("fds_min", 0.0))
    fds_max = float(config["model"].get("fds_max", 20.0))
    fds_bin_size = float(config["model"].get("fds_bin_size", 1.0))
    fds_num_bins = int(config["model"].get("fds_num_bins", int(round((fds_max - fds_min) / max(fds_bin_size, 1e-8))) + 1))
    fds_kernel = str(config["model"].get("fds_kernel", "gaussian"))
    fds_kernel_size = int(config["model"].get("fds_kernel_size", 5))
    fds_sigma = float(config["model"].get("fds_sigma", 2.0))
    fds_momentum = float(config["model"].get("fds_momentum", 0.9))
    fds_blend = float(config["model"].get("fds_blend", 1.0))
    fds_start_update = int(config["model"].get("fds_start_update", 1))
    fds_start_smooth = int(config["model"].get("fds_start_smooth", 2))
    risk_pos_weight = config["loss"].get("risk_pos_weight", "auto")
    fall_risk_pos_weight = config["loss"].get("fall_risk_pos_weight", "auto")
    high_risk_pos_weight = config["loss"].get("high_risk_pos_weight", "auto")
    high_fms_caution_threshold = float(config.get("evaluation", {}).get("high_fms_caution_threshold", 8.0))
    high_fms_warning_threshold = float(config.get("evaluation", {}).get("high_fms_warning_threshold", 12.0))
    rapid_rise_probability_threshold = float(config.get("evaluation", {}).get("rapid_rise_probability_threshold", 0.5))
    rapid_drop_probability_threshold = float(
        config.get("evaluation", {}).get("rapid_drop_probability_threshold", rapid_rise_probability_threshold)
    )
    final_warning_mode = str(config.get("evaluation", {}).get("final_warning_mode", "high_or_rapid"))
    selection_metric = str(config["training"].get("selection_metric", "mae"))
    selection_mode = _selection_mode_for_metric(selection_metric, config["training"].get("selection_mode"))
    if task_mode == "online_current_risk" and (teacher_delta_distill_weight > 0 or teacher_repr_distill_weight > 0):
        raise ValueError(
            "Teacher delta/repr distillation is not supported for task.mode=online_current_risk; "
            "use teacher_distill_weight for current-FMS distillation."
        )
    if (teacher_distill_weight > 0 or teacher_delta_distill_weight > 0 or teacher_repr_distill_weight > 0) and not teacher_checkpoint:
        raise ValueError("Teacher distillation requires --teacher_checkpoint.")
    if loss_type not in {"smooth_l1", "mse", "l1", "mae"}:
        raise ValueError("--loss_type must be one of: smooth_l1, mse, l1, mae.")
    if trajectory_loss_type not in {"smooth_l1", "mse", "l1", "mae"}:
        raise ValueError("loss.trajectory_loss_type must be one of: smooth_l1, mse, l1, mae.")
    if teacher_distill_loss_type not in {"smooth_l1", "mse", "l1", "mae"}:
        raise ValueError("--teacher_distill_loss_type must be one of: smooth_l1, mse, l1, mae.")
    if teacher_delta_distill_loss_type not in {"smooth_l1", "mse", "l1", "mae"}:
        raise ValueError("--teacher_delta_distill_loss_type must be one of: smooth_l1, mse, l1, mae.")
    if teacher_repr_distill_loss_type not in {"smooth_l1", "mse", "l1", "mae"}:
        raise ValueError("--teacher_repr_distill_loss_type must be one of: smooth_l1, mse, l1, mae.")
    if not bool(config["model"].get("risk_head_enabled", True)) and risk_loss_weight > 0:
        raise ValueError("model.risk_head_enabled=false requires loss.risk_loss_weight=0.0.")
    if not bool(config["model"].get("fall_risk_head_enabled", False)) and fall_loss_weight > 0:
        raise ValueError("model.fall_risk_head_enabled=false requires loss.fall_loss_weight=0.0.")
    if not bool(config["model"].get("high_risk_head_enabled", False)) and high_risk_loss_weight > 0:
        raise ValueError("model.high_risk_head_enabled=false requires loss.high_risk_loss_weight=0.0.")
    if bool(config["model"].get("high_risk_head_enabled", False)) and (
        not high_risk_horizon_seconds or not high_risk_thresholds
    ):
        raise ValueError("model.high_risk_head_enabled=true requires task.high_risk_horizon_seconds and task.high_risk_thresholds.")
    low_gate_head_available = bool(config["model"].get("current_low_suppressor_enabled", False)) or (
        current_head_mode_for_checks in {
            "zero_anchor_mixture",
            "calib_prior_range_scaled_delta",
            "calib_lowcap_range_scaled_delta",
        }
    )
    if not low_gate_head_available and (
        low_suppressor_gate_loss_weight > 0 or low_suppressor_correction_regularization_weight > 0
    ):
        raise ValueError(
            "low suppressor/gate losses require model.current_low_suppressor_enabled=true "
            "or a current head mode with an auxiliary gate."
        )
    if current_head_mode_for_checks in {
        "zero_anchor_mixture",
        "calib_prior_range_scaled_delta",
        "calib_lowcap_range_scaled_delta",
    } and low_suppressor_correction_regularization_weight > 0:
        raise ValueError(f"{current_head_mode_for_checks} supports gate loss but not suppressor correction regularization.")
    if anchor_gate_loss_weight > 0 and current_head_mode_for_checks != "zero_anchor_mixture":
        raise ValueError("loss.anchor_gate_loss_weight > 0 requires model.current_head_mode='zero_anchor_mixture'.")
    if session_aux_loss_type not in {"smooth_l1", "mse", "l1", "mae"}:
        raise ValueError("--session_aux_loss_type must be one of: smooth_l1, mse, l1, mae.")
    if lds_weighting:
        _lds_kernel_window(kernel=lds_kernel, kernel_size=lds_kernel_size, sigma=lds_sigma)
        if lds_bin_size <= 0 or lds_max <= lds_min:
            raise ValueError("LDS weighting requires lds_bin_size > 0 and lds_max > lds_min.")
        if lds_weight_min <= 0 or lds_weight_max < lds_weight_min:
            raise ValueError("LDS weighting requires 0 < lds_weight_min <= lds_weight_max.")
    if transition_weighting:
        if task_mode != "online_current_risk":
            raise ValueError("transition weighting is supported only for task.mode=online_current_risk.")
        if not transition_horizon_seconds or any(value <= 0 for value in transition_horizon_seconds):
            raise ValueError("loss.transition_horizon_seconds must contain positive seconds when transition_weighting is enabled.")
        if min(transition_drop_threshold, transition_recovery_threshold, transition_rise_threshold) <= 0:
            raise ValueError("transition thresholds must be positive FMS-point values.")
        if transition_high_threshold < transition_low_threshold:
            raise ValueError("transition_high_threshold must be >= transition_low_threshold.")
        if min(transition_drop_weight, transition_recovery_weight, transition_rise_weight, transition_max_weight) < 1.0:
            raise ValueError("transition weights must be >= 1.0.")
        if transition_max_weight < max(transition_drop_weight, transition_recovery_weight, transition_rise_weight):
            raise ValueError("transition_max_weight must be >= the individual transition weights.")
    if trajectory_loss_weight > 0:
        if task_mode != "online_current_risk":
            raise ValueError("trajectory loss is supported only for task.mode=online_current_risk.")
        if not trajectory_delta_seconds or any(value <= 0 for value in trajectory_delta_seconds):
            raise ValueError("loss.trajectory_delta_seconds must contain positive seconds when trajectory_loss_weight > 0.")
        if min(trajectory_delta_weight, trajectory_centered_weight, trajectory_range_weight) < 0:
            raise ValueError("trajectory component weights must be nonnegative.")
        if trajectory_delta_weight + trajectory_centered_weight + trajectory_range_weight <= 0:
            raise ValueError("At least one trajectory component weight must be positive.")
        if trajectory_min_points < 2:
            raise ValueError("trajectory_min_points must be >= 2.")
    if trajectory_decoder_loss_weight > 0:
        if task_mode != "online_current_risk":
            raise ValueError("trajectory decoder loss is supported only for task.mode=online_current_risk.")
        if current_head_mode_for_checks != "trajectory_decoder":
            raise ValueError("trajectory_decoder_loss_weight requires model.current_head_mode='trajectory_decoder'.")
    if min(session_affine_scale_regularization_weight, session_affine_bias_regularization_weight) < 0:
        raise ValueError("session affine regularization weights must be nonnegative.")
    if min(
        calibration_residual_regularization_weight,
        low_overprediction_weight,
        high_underprediction_weight,
    ) < 0:
        raise ValueError("calibration residual/asymmetric bias loss weights must be nonnegative.")
    if low_overprediction_threshold < 0.0 or high_underprediction_threshold > 20.0:
        raise ValueError("low/high asymmetric thresholds must be expressed on the raw DenseFMS 0-20 scale.")
    if low_overprediction_threshold >= high_underprediction_threshold:
        raise ValueError("low_overprediction_threshold must be smaller than high_underprediction_threshold.")
    if (
        session_affine_scale_regularization_weight > 0 or session_affine_bias_regularization_weight > 0
    ) and not bool(config["model"].get("current_session_affine_head_enabled", False)):
        raise ValueError(
            "session affine regularization requires model.current_session_affine_head_enabled=true."
        )
    if calibration_residual_regularization_weight > 0 and not bool(
        config["model"].get("calibration_residual_adapter_enabled", False)
    ):
        raise ValueError(
            "calibration residual regularization requires model.calibration_residual_adapter_enabled=true."
        )
    if fds_enabled:
        if task_mode != "online_current_risk":
            raise ValueError("FDS is currently supported only for task.mode=online_current_risk.")
        if args.model not in {"online_fms_risk_tracker", "online_risk_tracker", "online_current_risk"}:
            raise ValueError("FDS is currently supported only with --model online_fms_risk_tracker.")
        _lds_kernel_window(kernel=fds_kernel, kernel_size=fds_kernel_size, sigma=fds_sigma)
        if fds_bin_size <= 0 or fds_max <= fds_min:
            raise ValueError("FDS requires fds_bin_size > 0 and fds_max > fds_min.")
        if not (0.0 <= fds_blend <= 1.0):
            raise ValueError("FDS requires 0.0 <= fds_blend <= 1.0.")
        if not (0.0 <= fds_momentum < 1.0):
            raise ValueError("FDS requires 0.0 <= fds_momentum < 1.0.")
    if change_aux_weight > 0:
        config["model"]["change_aux_head"] = True
    high_fms_threshold = float(config.get("evaluation", {}).get("high_fms_threshold", high_fms_warning_threshold))
    use_static = bool(config["model"].get("use_static", False))
    allow_missing_static = bool(config["data"].get("allow_missing_static", False))
    calibration_seconds = float(config["data"]["calibration_seconds"])
    horizon_seconds = float(config["data"]["horizon_seconds"])
    recent_window_seconds = float(config["data"]["recent_window_seconds"])
    residual_adapter_enabled = bool(config["model"].get("calibration_residual_adapter_enabled", False))
    summary_fusion_enabled = bool(config["model"].get("calibration_summary_fusion_enabled", False))
    residual_feature_dependent = residual_adapter_enabled or summary_fusion_enabled
    if residual_feature_dependent:
        config["data"]["require_calibration_residual_features"] = True
        if not config["data"].get("calibration_residual_features_path"):
            raise ValueError(
                "calibration residual-feature dependent model options require "
                "data.calibration_residual_features_path."
            )

    seed = int(config["training"].get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    saved_split = load_json(args.split_file) if args.split_file and Path(args.split_file).exists() else None
    prepared = prepare_data(args.data_dir, config, limit_sessions=args.limit_sessions, saved_split=saved_split)
    if residual_feature_dependent:
        residual_dim = int(prepared["data_info"].get("calibration_residual_feature_dim", 0))
        if residual_dim <= 0:
            raise ValueError("Calibration residual features were requested, but no residual feature dimension was loaded.")
        config["model"]["calibration_residual_feature_dim"] = residual_dim
        if summary_fusion_enabled:
            config["model"]["calibration_summary_fusion_feature_dim"] = residual_dim
        print(
            "Calibration residual features: "
            f"dim={residual_dim} sources={prepared['data_info'].get('calibration_residual_feature_artifacts', {}).get('sources', [])}"
        )
    if use_static and prepared.get("static_report"):
        report = prepared["static_report"]
        print("Static feature report:")
        print(f"  features: {report.get('static_features')}")
        print(f"  gender_encoding: {report.get('gender_encoding')}")
        print(f"  static_dim: {report['static_scaler'].get('static_dim')}")
        if "age" in report.get("static_features", []):
            print(f"  train age mean/std: {report['static_scaler']['age_mean']:.4f} / {report['static_scaler']['age_std']:.4f}")
        if "mssq" in report.get("static_features", []):
            print(f"  train MSSQ mean/std: {report['static_scaler']['mssq_mean']:.4f} / {report['static_scaler']['mssq_std']:.4f}")
        for split_name, split_report in report["splits"].items():
            print(
                f"  {split_name}: age_available={split_report['age_available']}/{split_report['session_count']} "
                f"gender_available={split_report['gender_available']}/{split_report['session_count']} "
                f"mssq_available={split_report['mssq_available']}/{split_report['session_count']} "
                f"gender_counts={split_report['gender_counts']}"
            )
    if args.overfit_sessions:
        n = int(args.overfit_sessions)
        sessions = prepared["splits"]["train"][:n]
        if len(sessions) < n:
            all_sessions = prepared["splits"]["train"] + prepared["splits"].get("val", []) + prepared["splits"].get("test", [])
            sessions = all_sessions[:n]
        prepared["splits"] = {"train": sessions, "val": sessions, "test": sessions}
        prepared["split_info"]["overfit_sessions"] = n
        print(f"Overfit mode: using {len(sessions)} sessions for train/val/test.")

    loaders = make_loaders(prepared["splits"], config)
    calibration_steps = int(prepared["data_info"]["calibration_steps"])
    horizon_steps = int(prepared["data_info"]["horizon_steps"])
    recent_steps = seconds_to_steps(
        float(config["data"]["recent_window_seconds"]),
        float(prepared["data_info"]["sampling_interval"]),
        name="recent_window_seconds",
    )
    rise_horizon_steps = [
        seconds_to_steps(
            float(value),
            float(prepared["data_info"]["sampling_interval"]),
            name="rise_horizon_seconds",
        )
        for value in rise_horizon_seconds
    ]
    fall_horizon_steps = [
        seconds_to_steps(
            float(value),
            float(prepared["data_info"]["sampling_interval"]),
            name="fall_horizon_seconds",
        )
        for value in fall_horizon_seconds
    ]
    high_risk_horizon_steps = [
        seconds_to_steps(
            float(value),
            float(prepared["data_info"]["sampling_interval"]),
            name="high_risk_horizon_seconds",
        )
        for value in high_risk_horizon_seconds
    ]
    high_risk_onset_past_steps = seconds_to_steps(
        high_risk_onset_past_seconds,
        float(prepared["data_info"]["sampling_interval"]),
        name="high_risk_onset_past_seconds",
        allow_zero=True,
    )
    future_aux_horizon_steps = [
        seconds_to_steps(
            float(value),
            float(prepared["data_info"]["sampling_interval"]),
            name="future_aux_horizon_seconds",
        )
        for value in future_aux_horizon_seconds
    ]
    transition_horizon_steps = [
        seconds_to_steps(
            float(value),
            float(prepared["data_info"]["sampling_interval"]),
            name="transition_horizon_seconds",
        )
        for value in transition_horizon_seconds
    ]
    trajectory_delta_steps = [
        seconds_to_steps(
            float(value),
            float(prepared["data_info"]["sampling_interval"]),
            name="trajectory_delta_seconds",
        )
        for value in trajectory_delta_seconds
    ]
    fms_min = float(prepared["scalers"]["fms"]["min"])
    fms_max = float(prepared["scalers"]["fms"]["max"])
    fms_range = max(fms_max - fms_min, 1e-8)
    rise_thresholds_norm = [float(v) / fms_range for v in rise_thresholds]
    ordinal_bins_norm = [float(v - fms_min) / fms_range for v in ordinal_bins]
    lds_info: Optional[Dict[str, Any]] = None
    lds_weight_table: Optional[torch.Tensor] = None
    if task_mode == "online_current_risk" and lds_weighting:
        lds_prediction_start = max(int(calibration_steps), int(recent_steps) - 1)
        lds_info = build_lds_weight_info(
            prepared["splits"]["train"],
            lds_prediction_start,
            prepared["scalers"]["fms"],
            min_value=lds_min,
            max_value=lds_max,
            bin_size=lds_bin_size,
            kernel=lds_kernel,
            kernel_size=lds_kernel_size,
            sigma=lds_sigma,
            gamma=lds_gamma,
            weight_min=lds_weight_min,
            weight_max=lds_weight_max,
        )
        lds_weight_table = torch.tensor(lds_info["weights"], dtype=torch.float32)
        print(
            "LDS weighting: "
            f"targets={lds_info['train_target_count']} bins={lds_info['num_bins']} "
            f"nonempty={lds_info['train_nonempty_bins']} kernel={lds_kernel} "
            f"ks={lds_kernel_size} sigma={lds_sigma:g} gamma={lds_gamma:g} "
            f"sample_weight_mean={lds_info['sample_weight_mean']:.4f} "
            f"range=[{lds_info['sample_weight_min']:.4f},{lds_info['sample_weight_max']:.4f}]"
        )
    ordinal_class_count_info: Optional[Dict[str, Any]] = None
    ordinal_class_counts: Optional[torch.Tensor] = None
    if task_mode == "online_current_risk" and ordinal_loss_mode in {
        "slace",
        "slace_prox",
        "slace_paper",
        "slace_index",
        "slace_no_prox",
        "slace_norm_prox",
        "slace_prox_norm",
    }:
        slace_prediction_start = max(int(calibration_steps), int(recent_steps) - 1)
        ordinal_class_count_info = build_ordinal_class_count_info(
            prepared["splits"]["train"],
            slace_prediction_start,
            prepared["scalers"]["fms"],
            ordinal_bins,
        )
        ordinal_class_counts = torch.tensor(ordinal_class_count_info["class_counts"], dtype=torch.float32)
        print(
            "SLACE ordinal counts: "
            f"targets={ordinal_class_count_info['train_target_count']} "
            f"classes={len(ordinal_class_count_info['class_counts'])} "
            f"nonempty={ordinal_class_count_info['train_nonempty_classes']} "
            f"range=[{ordinal_class_count_info['min_count']:.0f},{ordinal_class_count_info['max_count']:.0f}] "
            f"alpha={ordinal_slace_alpha:g} prox={ordinal_slace_proximity} "
            f"norm_prox={ordinal_slace_normalize_proximity}"
        )
    if not args.overfit_sessions:
        first_batch = next(iter(loaders["train"]))
        run_data_sanity_checks(prepared["split_info"], first_batch, calibration_steps, horizon_steps)

    model_kwargs = {
        "head_dim": int(config["model"].get("head_dim", 6)),
        "calibration_steps": calibration_steps,
        "horizon_steps": horizon_steps,
        "recent_steps": recent_steps,
        "sampling_interval": float(prepared["data_info"]["sampling_interval"]),
        "horizon_seconds": horizon_seconds,
        "rise_horizon_steps": rise_horizon_steps,
        "rise_thresholds": rise_thresholds,
        "fall_horizon_steps": fall_horizon_steps,
        "fall_thresholds": fall_thresholds,
        "high_risk_horizon_steps": high_risk_horizon_steps,
        "high_risk_thresholds": high_risk_thresholds,
        "future_aux_horizon_steps": future_aux_horizon_steps,
        "delta_max": float(prepared["scalers"]["delta_max"]),
        "no_film": bool(config["model"].get("no_film", False)),
        "no_recent_encoder": bool(config["model"].get("no_recent_encoder", False)),
        "use_legacy_multihead": bool(config["model"].get("use_legacy_multihead", False)),
        "use_static": use_static,
        "static_dim": int(
            config["model"].get(
                "static_dim",
                static_feature_dim(
                    config["data"].get("static_features", ["age", "gender"]),
                    gender_encoding=config["data"].get("gender_encoding", "category3"),
                ),
            )
        ),
        "static_hidden_dim": int(config["model"].get("static_hidden_dim", 64)),
        "static_dropout": float(config["model"].get("static_dropout", 0.1)),
        "recent_encoder": str(config["model"].get("recent_encoder", "tcn")),
        "recent_attn_heads": int(config["model"].get("recent_attn_heads", 4)),
        "recent_attn_layers": int(config["model"].get("recent_attn_layers", 1)),
        "recent_attn_dropout": float(config["model"].get("recent_attn_dropout", 0.1)),
        "max_time_steps": int(config["model"].get("max_time_steps", 2048)),
        "d_model": int(config["model"].get("d_model", 64)),
        "kernel_size": int(config["model"].get("kernel_size", 3)),
        "dropout": float(config["model"].get("dropout", 0.1)),
        "calib_dilations": config["model"].get("calib_dilations", [1, 2, 4, 8, 16]),
        "recent_dilations": config["model"].get("recent_dilations", "auto"),
        "transformer_layers": int(config["model"].get("transformer_layers", 1)),
        "transformer_heads": int(config["model"].get("transformer_heads", 4)),
        "transformer_ff_dim": int(config["model"].get("transformer_ff_dim", 128)),
        "pooling": str(config["model"].get("pooling", "mean")),
        "anchor_mode": str(config["model"].get("anchor_mode", "calibration_end")),
        "anchor_interval_seconds": float(config["model"].get("anchor_interval_seconds", 60.0)),
        "fms_context_mode": str(config["model"].get("fms_context_mode", "calibration_history")),
        "predict_delta_from_anchor": bool(config["model"].get("predict_delta_from_anchor", False)),
        "multi_horizon": bool(config["model"].get("multi_horizon", False)),
        "horizon_set": config["model"].get("horizon_set"),
        "per_horizon_heads": bool(config["model"].get("per_horizon_heads", False)),
        "horizon_encoder_dim": (
            None
            if config["model"].get("horizon_encoder_dim") is None
            else int(config["model"].get("horizon_encoder_dim"))
        ),
        "horizon_context_mode": str(config["model"].get("horizon_context_mode", "encoded")),
        "start_fms_context_mode": str(config["model"].get("start_fms_context_mode", "encoded")),
        "static_context_mode": str(config["model"].get("static_context_mode", "encoded")),
        "forecast_head_mode": str(config["model"].get("forecast_head_mode", "level")),
        "horizon_head_mode": str(config["model"].get("horizon_head_mode", "linear")),
        "horizon_head_hidden_dim": (
            None
            if config["model"].get("horizon_head_hidden_dim") is None
            else int(config["model"].get("horizon_head_hidden_dim"))
        ),
        "motion_feature_mode": str(config["model"].get("motion_feature_mode", "none")),
        "motion_stats_branch": bool(config["model"].get("motion_stats_branch", False)),
        "stream_time_features": bool(config["model"].get("stream_time_features", False)),
        "stream_context_mode": str(config["model"].get("stream_context_mode", "gru")),
        "stream_prepend_calibration": bool(config["model"].get("stream_prepend_calibration", False)),
        "stream_calib_condition_mode": str(config["model"].get("stream_calib_condition_mode", "none")),
        "stream_calib_condition_strength": float(config["model"].get("stream_calib_condition_strength", 0.1)),
        "calib_summary_features": bool(config["model"].get("calib_summary_features", False)),
        "calibration_fusion_mode": str(config["model"].get("calibration_fusion_mode", "add")),
        "calibration_fusion_hidden_dim": config["model"].get("calibration_fusion_hidden_dim"),
        "calibration_fusion_output_dim": config["model"].get("calibration_fusion_output_dim"),
        "calibration_encoder_mode": str(config["model"].get("calibration_encoder_mode", "tcn_transformer")),
        "state_feedback_mode": str(config["model"].get("state_feedback_mode", "none")),
        "session_context_mode": str(config["model"].get("session_context_mode", "none")),
        "change_aux_head": bool(config["model"].get("change_aux_head", False)),
        "ordinal_bins": ordinal_bins,
        "coarse_band_bins": coarse_band_bins,
        "coarse_residual_head_enabled": bool(config["model"].get("coarse_residual_head_enabled", False)),
        "coarse_residual_range": float(config["model"].get("coarse_residual_range", 3.0)),
        "coarse_residual_combine_weight": float(config["model"].get("coarse_residual_combine_weight", 0.0)),
        "fms_combine_weight_ordinal": float(config["model"].get("fms_combine_weight_ordinal", 0.6)),
        "current_head_mode": str(config["model"].get("current_head_mode", "basic")),
        "ordinal_head_mode": config["model"].get("ordinal_head_mode"),
        "current_delta_scale": float(config["model"].get("current_delta_scale", 0.75)),
        "current_anchor_delta_growth_scale": float(config["model"].get("current_anchor_delta_growth_scale", 0.0)),
        "current_anchor_delta_growth_horizon_seconds": float(
            config["model"].get("current_anchor_delta_growth_horizon_seconds", 90.0)
        ),
        "current_anchor_delta_growth_power": float(config["model"].get("current_anchor_delta_growth_power", 1.0)),
        "current_trajectory_offsets": config["model"].get("current_trajectory_offsets"),
        "current_range_guard_low_threshold": float(config["model"].get("current_range_guard_low_threshold", 5.0)),
        "current_range_guard_temperature": float(config["model"].get("current_range_guard_temperature", 1.0)),
        "current_range_guard_floor": float(config["model"].get("current_range_guard_floor", 0.10)),
        "current_range_guard_cap": float(config["model"].get("current_range_guard_cap", 2.0)),
        "current_range_guard_cap_strength": float(config["model"].get("current_range_guard_cap_strength", 1.0)),
        "motion_encoder_context": str(config["model"].get("motion_encoder_context", "linear")),
        "motion_encoder_layers": int(config["model"].get("motion_encoder_layers", 0)),
        "risk_head_enabled": bool(config["model"].get("risk_head_enabled", True)),
        "fall_risk_head_enabled": bool(config["model"].get("fall_risk_head_enabled", False)),
        "high_risk_head_enabled": bool(config["model"].get("high_risk_head_enabled", False)),
        "risk_temporal_context": str(config["model"].get("risk_temporal_context", "none")),
        "risk_temporal_layers": int(config["model"].get("risk_temporal_layers", 0)),
        "regime_head_enabled": bool(config["model"].get("regime_head_enabled", False)),
        "regime_class_count": int(config["model"].get("regime_class_count", 5)),
        "uncertainty_head_enabled": bool(config["model"].get("uncertainty_head_enabled", False)),
        "deep_tcn_dilations": config["model"].get("deep_tcn_dilations", [1, 2, 4, 8, 16, 32]),
        "calibration_tcn_adaptive_dilations": bool(
            config["model"].get("calibration_tcn_adaptive_dilations", False)
        ),
        "calibration_tcn_max_padding_steps": int(config["model"].get("calibration_tcn_max_padding_steps", 8)),
        "calibration_tcn_max_padding_fraction": float(
            config["model"].get("calibration_tcn_max_padding_fraction", 0.1)
        ),
        "decoder_hidden_dim": config["model"].get("decoder_hidden_dim"),
        "decoder_context_mode": str(config["model"].get("decoder_context_mode", "fused")),
        "decoder_temporal_context": str(config["model"].get("decoder_temporal_context", "none")),
        "decoder_temporal_layers": int(config["model"].get("decoder_temporal_layers", 0)),
        "fds_enabled": fds_enabled,
        "fds_min": fds_min,
        "fds_max": fds_max,
        "fds_bin_size": fds_bin_size,
        "fds_num_bins": fds_num_bins,
        "fds_kernel": fds_kernel,
        "fds_kernel_size": fds_kernel_size,
        "fds_sigma": fds_sigma,
        "fds_momentum": fds_momentum,
        "fds_blend": fds_blend,
        "calib_fms_dropout": float(config["model"].get("calib_fms_dropout", 0.0)),
        "calibration_end_fms_dropout": float(config["model"].get("calibration_end_fms_dropout", 0.0)),
        "current_session_affine_head_enabled": bool(config["model"].get("current_session_affine_head_enabled", False)),
        "current_session_affine_hidden_dim": config["model"].get("current_session_affine_hidden_dim"),
        "current_session_affine_scale_range": float(config["model"].get("current_session_affine_scale_range", 0.25)),
        "current_session_affine_bias_range": float(config["model"].get("current_session_affine_bias_range", 0.15)),
        "current_affine_head_enabled": bool(config["model"].get("current_affine_head_enabled", False)),
        "current_affine_hidden_dim": config["model"].get("current_affine_hidden_dim"),
        "current_affine_scale_range": float(config["model"].get("current_affine_scale_range", 0.5)),
        "current_affine_bias_range": float(config["model"].get("current_affine_bias_range", 0.25)),
        "current_binned_affine_head_enabled": bool(config["model"].get("current_binned_affine_head_enabled", False)),
        "current_binned_affine_anchor_bins": config["model"].get("current_binned_affine_anchor_bins"),
        "current_binned_affine_pred_bins": config["model"].get("current_binned_affine_pred_bins"),
        "current_binned_affine_time_bins": config["model"].get("current_binned_affine_time_bins"),
        "current_binned_affine_scale_range": float(config["model"].get("current_binned_affine_scale_range", 1.5)),
        "current_binned_affine_bias_range": float(config["model"].get("current_binned_affine_bias_range", 0.5)),
        "calibration_residual_adapter_enabled": bool(config["model"].get("calibration_residual_adapter_enabled", False)),
        "calibration_residual_feature_dim": int(config["model"].get("calibration_residual_feature_dim", 0)),
        "calibration_residual_adapter_hidden_dim": config["model"].get("calibration_residual_adapter_hidden_dim"),
        "calibration_residual_adapter_mode": str(config["model"].get("calibration_residual_adapter_mode", "mlp")),
        "calibration_residual_delta_range": float(config["model"].get("calibration_residual_delta_range", 0.15)),
        "calibration_residual_decay_seconds": float(config["model"].get("calibration_residual_decay_seconds", 120.0)),
        "calibration_residual_gate_low_threshold": float(
            config["model"].get("calibration_residual_gate_low_threshold", 8.0)
        ),
        "calibration_residual_gate_high_threshold": float(
            config["model"].get("calibration_residual_gate_high_threshold", 10.0)
        ),
        "calibration_residual_gate_anchor_threshold": float(
            config["model"].get("calibration_residual_gate_anchor_threshold", 10.0)
        ),
        "calibration_residual_gate_temperature": float(
            config["model"].get("calibration_residual_gate_temperature", 1.0)
        ),
        "calibration_summary_fusion_enabled": bool(config["model"].get("calibration_summary_fusion_enabled", False)),
        "calibration_summary_fusion_feature_dim": int(
            config["model"].get("calibration_summary_fusion_feature_dim", 0)
        ),
        "calibration_summary_fusion_hidden_dim": config["model"].get("calibration_summary_fusion_hidden_dim"),
        "calibration_summary_fusion_mode": str(
            config["model"].get("calibration_summary_fusion_mode", "additive_gated")
        ),
        "calibration_summary_fusion_strength": float(
            config["model"].get("calibration_summary_fusion_strength", 1.0)
        ),
        "current_low_suppressor_enabled": bool(config["model"].get("current_low_suppressor_enabled", False)),
        "current_low_suppressor_hidden_dim": config["model"].get("current_low_suppressor_hidden_dim"),
        "current_low_suppressor_delta_range": float(config["model"].get("current_low_suppressor_delta_range", 0.25)),
        "current_low_suppressor_gate_init_bias": float(
            config["model"].get("current_low_suppressor_gate_init_bias", -6.0)
        ),
        "hidden_dim": int(config["model"].get("hidden_dim", config["model"].get("d_model", 128))),
        "mlp_layers": config["model"].get("mlp_layers"),
        "gru_layers": int(config["model"].get("gru_layers", 1)),
        "branch_dropout": float(config["model"].get("branch_dropout", 0.0)),
        "anchor_dropout": float(config["model"].get("anchor_dropout", 0.0)),
        "delta_scale": float(config["model"].get("delta_scale", 0.5)),
    }
    model = build_model(args.model, **model_kwargs).to(device)
    motion_pretrain_info: Optional[Dict[str, Any]] = None
    motion_pretrain_checkpoint = config["model"].get("motion_pretrain_checkpoint")
    if motion_pretrain_checkpoint:
        if not hasattr(model, "deep_tcn_stream") or getattr(model, "deep_tcn_stream") is None:
            raise ValueError("--motion_pretrain_checkpoint requires a model with deep_tcn_stream.")
        pretrain_payload = torch.load(str(motion_pretrain_checkpoint), map_location=device, weights_only=False)
        encoder_state = pretrain_payload.get("encoder_state_dict")
        if encoder_state is None:
            raise ValueError(f"Motion pretrain checkpoint {motion_pretrain_checkpoint} lacks encoder_state_dict.")
        getattr(model, "deep_tcn_stream").load_state_dict(encoder_state, strict=True)
        motion_pretrain_info = {
            "checkpoint": str(motion_pretrain_checkpoint),
            "pretrain_config": pretrain_payload.get("config", {}),
            "metrics": pretrain_payload.get("metrics", {}),
        }
        print(f"Loaded motion pretraining checkpoint: {motion_pretrain_checkpoint}")
    init_checkpoint_info: Optional[Dict[str, Any]] = None
    init_checkpoint = config["training"].get("init_checkpoint")
    if init_checkpoint:
        payload = torch.load(str(init_checkpoint), map_location=device, weights_only=False)
        state_dict = payload.get("model_state_dict")
        if state_dict is None:
            raise ValueError(f"Initialization checkpoint {init_checkpoint} lacks model_state_dict.")
        target_state = model.state_dict()
        skipped_shape_mismatch: List[Dict[str, Any]] = []
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key in target_state and hasattr(value, "shape") and tuple(value.shape) != tuple(target_state[key].shape):
                skipped_shape_mismatch.append(
                    {
                        "key": key,
                        "checkpoint_shape": list(value.shape),
                        "model_shape": list(target_state[key].shape),
                    }
                )
                continue
            filtered_state_dict[key] = value
        state_dict = filtered_state_dict
        incompatible = model.load_state_dict(state_dict, strict=False)
        freeze_loaded = bool(config["training"].get("freeze_loaded_parameters", False))
        trainable_patterns = [str(value) for value in config["training"].get("trainable_parameter_patterns", [])]
        if freeze_loaded:
            if not trainable_patterns:
                raise ValueError("training.freeze_loaded_parameters=true requires training.trainable_parameter_patterns.")
            for name, parameter in model.named_parameters():
                parameter.requires_grad = any(pattern in name for pattern in trainable_patterns)
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        init_checkpoint_info = {
            "checkpoint": str(init_checkpoint),
            "freeze_loaded_parameters": freeze_loaded,
            "trainable_parameter_patterns": trainable_patterns,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "skipped_shape_mismatch": skipped_shape_mismatch,
            "trainable_parameter_names": trainable_names,
        }
        print(
            "Initialized model from checkpoint: "
            f"{init_checkpoint} missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)} "
            f"shape_skipped={len(skipped_shape_mismatch)} "
            f"freeze_loaded={freeze_loaded} trainable={len(trainable_names)} tensors"
        )
    teacher_model: Optional[torch.nn.Module] = None
    teacher_info: Optional[Dict[str, Any]] = None
    teacher_repr_projector: Optional[torch.nn.Module] = None
    teacher_repr_projection_info: Optional[Dict[str, Any]] = None
    if teacher_distill_weight > 0 or teacher_delta_distill_weight > 0 or teacher_repr_distill_weight > 0:
        teacher_model, teacher_info = load_teacher_model(str(teacher_checkpoint), device)
        validate_teacher_compatibility(teacher_info, model_kwargs)
        if teacher_repr_distill_weight > 0:
            student_repr_dim = int(getattr(model, "d_model"))
            teacher_repr_dim = int(getattr(teacher_model, "d_model"))
            if student_repr_dim == teacher_repr_dim:
                teacher_repr_projector = torch.nn.Identity().to(device)
            else:
                teacher_repr_projector = torch.nn.Linear(student_repr_dim, teacher_repr_dim).to(device)
            teacher_repr_projection_info = {
                "student_dim": student_repr_dim,
                "teacher_dim": teacher_repr_dim,
                "trainable": student_repr_dim != teacher_repr_dim,
                "normalize": True,
            }
        print(
            "Teacher distillation: "
            f"checkpoint={teacher_info['checkpoint']} model={teacher_info['model_name']} "
            f"requires_full_fms={teacher_info['requires_full_fms']} "
            f"{'current' if task_mode == 'online_current_risk' else 'future'}_weight={teacher_distill_weight:g} "
            f"delta_weight={teacher_delta_distill_weight:g} repr_weight={teacher_repr_distill_weight:g}"
        )
        if teacher_repr_projection_info is not None:
            print(
                "Teacher representation projection: "
                f"student_dim={teacher_repr_projection_info['student_dim']} "
                f"teacher_dim={teacher_repr_projection_info['teacher_dim']} "
                f"trainable={teacher_repr_projection_info['trainable']}"
            )
    if hasattr(model, "recent_rf_steps"):
        print(
            f"Recent TCN receptive field: {getattr(model, 'recent_rf_steps')} steps "
            f"({getattr(model, 'recent_rf_seconds'):.2f}s)"
        )
    if hasattr(model, "calibration_tcn_rf_steps") and getattr(model, "calibration_tcn_rf_steps"):
        print(
            "Calibration TCN receptive field: "
            f"{getattr(model, 'calibration_tcn_rf_steps')} steps, "
            f"dilations={getattr(model, 'calibration_deep_tcn_dilations', [])}, "
            f"causal_pad={getattr(model, 'calibration_tcn_pad_steps', 0)} steps"
        )
    if fds_enabled:
        print(
            "FDS feature smoothing: "
            f"bins={fds_num_bins} range=[{fds_min:g},{fds_max:g}] bin_size={fds_bin_size:g} "
            f"kernel={fds_kernel} ks={fds_kernel_size} sigma={fds_sigma:g} "
            f"momentum={fds_momentum:g} blend={fds_blend:g} "
            f"start_update={fds_start_update} start_smooth={fds_start_smooth} "
            "apply=train_only"
        )
    optimizer_params = list(model.parameters())
    if teacher_repr_projector is not None:
        optimizer_params.extend(p for p in teacher_repr_projector.parameters() if p.requires_grad)
    parameter_count = int(sum(p.numel() for p in optimizer_params if p.requires_grad))
    print(f"Trainable parameters: {parameter_count}")
    fds_info: Optional[Dict[str, Any]] = None
    if fds_enabled:
        fds_info = {
            "enabled": True,
            "min": fds_min,
            "max": fds_max,
            "bin_size": fds_bin_size,
            "num_bins": fds_num_bins,
            "kernel": fds_kernel,
            "kernel_size": fds_kernel_size,
            "sigma": fds_sigma,
            "momentum": fds_momentum,
            "blend": fds_blend,
            "start_update": fds_start_update,
            "start_smooth": fds_start_smooth,
            "apply": "train_only",
        }
    loss_fn = FutureSequenceLoss(
        mode=loss_mode,
        trend_weight=trend_weight,
        loss_type=loss_type,
        horizon_weights=horizon_weights,
        change_weight=change_weight,
        high_target_weight=high_target_weight,
        high_target_threshold=high_target_threshold,
        low_target_weight=low_target_weight,
        low_target_threshold=low_target_threshold,
    )
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    grad_clip = float(config["training"].get("gradient_clip", 1.0))
    epochs = int(config["training"].get("epochs", 80))
    patience = int(config["training"].get("patience", 12))
    max_train_batches = config["training"].get("max_train_batches")
    max_eval_batches = config["training"].get("max_eval_batches")

    static_tag = "static" if use_static else "no_static"
    loss_tag = "trend" if loss_mode == "level_trend_raw" else "level"
    if args.run_name:
        run_name = args.run_name
    elif loss_mode == "level_trend_raw":
        run_name = f"{timestamp_for_run()}_{args.model}_{static_tag}_{loss_tag}_w{trend_weight:g}"
    else:
        run_name = f"{timestamp_for_run()}_{args.model}_{static_tag}_{loss_tag}"
    run_dir = ensure_dir(Path(config.get("runs_dir", "runs")) / run_name)
    if args.skip_existing and (run_dir / "metrics.json").exists() and (run_dir / "best.pt").exists():
        print(f"Skipping existing completed run: {run_dir}")
        return load_json(run_dir / "metrics.json")
    if args.resume is not None:
        print("WARNING: --resume is accepted for CLI compatibility, but this trainer starts a fresh run unless skip_existing is used.")
    plots_dir = ensure_dir(run_dir / "plots")
    save_json(run_dir / "split.json", prepared["split_info"])
    save_json(run_dir / "config_snapshot.json", config)
    if args.split_file and saved_split is None:
        save_json(args.split_file, prepared["split_info"])

    best_selection_value = float("inf") if selection_mode == "min" else -float("inf")
    best_epoch = -1
    best_metrics: Dict[str, Any] = {}
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    fds_history: List[Dict[str, Any]] = []

    print(f"Training {args.model} task={task_mode} with {loss_mode} on {device} for up to {epochs} epochs. Run: {run_dir}")
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        if teacher_repr_projector is not None:
            teacher_repr_projector.train()
        if fds_enabled and hasattr(model, "reset_fds_epoch_stats"):
            model.reset_fds_epoch_stats()
        fds_update_now = bool(fds_enabled and epoch >= fds_start_update)
        fds_apply_now = bool(fds_enabled and epoch >= fds_start_smooth)
        fds_update_points = 0
        fds_apply_points = 0
        train_sums = {"loss_total": 0.0, "loss_level": 0.0, "loss_trend": 0.0}
        train_points = 0
        for batch_idx, batch in enumerate(loaders["train"]):
            if max_train_batches is not None and batch_idx >= int(max_train_batches):
                break
            head = batch["head"].to(device)
            fms = batch["fms"].to(device)
            lengths = batch["lengths"].to(device)
            static = batch.get("static")
            if use_static:
                if static is None:
                    raise ValueError("Static training requested but batch['static'] is missing.")
                static = static.to(device)
            residual_features = batch.get("calibration_residual_features")
            residual_feature_mask = batch.get("calibration_residual_feature_mask")
            if residual_feature_dependent:
                if residual_features is None:
                    raise ValueError("Calibration residual-feature training requested but batch['calibration_residual_features'] is missing.")
                residual_features = residual_features.to(device)
                residual_feature_mask = residual_feature_mask.to(device) if residual_feature_mask is not None else None
            optimizer.zero_grad(set_to_none=True)
            fms_input = fms if getattr(model, "requires_full_fms", False) else fms[:, :calibration_steps]
            fds_labels_raw = None
            fds_label_mask = None
            if fds_enabled and task_mode == "online_current_risk":
                pred_start, _pred_positions, pred_mask = model._prediction_positions(lengths, device)
                pred_steps = int(_pred_positions.numel())
                if pred_steps > 0:
                    fds_targets = compute_online_current_risk_targets(
                        fms,
                        lengths,
                        pred_start,
                        pred_steps,
                        rise_horizon_steps,
                        rise_thresholds_norm,
                        ordinal_bins_norm,
                    )
                    fds_labels_raw = fds_targets["current"].to(device) * fms_range + fms_min
                    fds_label_mask = pred_mask.to(device).bool() & fds_targets["current_mask"].to(device)
                else:
                    fds_labels_raw = fms.new_zeros((fms.shape[0], 0))
                    fds_label_mask = torch.zeros((fms.shape[0], 0), dtype=torch.bool, device=device)
            outputs = model(
                head,
                fms_input,
                lengths,
                static=static,
                fds_labels_raw=fds_labels_raw,
                fds_mask=fds_label_mask,
                fds_update=fds_update_now,
                fds_apply=fds_apply_now,
                calibration_residual_features=residual_features,
                calibration_residual_feature_mask=residual_feature_mask,
            )
            if fds_enabled:
                fds_update_points += _tensor_scalar_int(outputs.get("fds_updated_points"), 0)
                fds_apply_points += _tensor_scalar_int(outputs.get("fds_applied_points"), 0)
            if task_mode == "online_current_risk":
                loss, parts = compute_online_current_risk_loss(
                    outputs,
                    fms,
                    lengths,
                    prepared["scalers"]["fms"],
                    rise_horizon_steps=rise_horizon_steps,
                    rise_thresholds=rise_thresholds,
                    ordinal_bins=ordinal_bins,
                    current_reg_aux_weight=current_reg_aux_weight,
                    ordinal_loss_weight=ordinal_loss_weight,
                    ordinal_loss_mode=ordinal_loss_mode,
                    ordinal_soft_label_sigma=ordinal_soft_label_sigma,
                    ordinal_soft_label_kernel=ordinal_soft_label_kernel,
                    ordinal_ev_loss_weight=ordinal_ev_loss_weight,
                    ordinal_low_weight=ordinal_low_weight,
                    ordinal_low_threshold=ordinal_low_threshold,
                    ordinal_slace_alpha=ordinal_slace_alpha,
                    ordinal_slace_proximity=ordinal_slace_proximity,
                    ordinal_slace_normalize_proximity=ordinal_slace_normalize_proximity,
                    ordinal_slace_count_smoothing=ordinal_slace_count_smoothing,
                    ordinal_class_counts=ordinal_class_counts,
                    coarse_band_loss_weight=coarse_band_loss_weight,
                    coarse_residual_loss_weight=coarse_residual_loss_weight,
                    regime_loss_weight=regime_loss_weight,
                    regime_delta_slow_threshold=regime_delta_slow_threshold,
                    regime_delta_rapid_threshold=regime_delta_rapid_threshold,
                    regime_high_threshold=regime_high_threshold,
                    uncertainty_loss_weight=uncertainty_loss_weight,
                    risk_loss_weight=risk_loss_weight,
                    fall_horizon_steps=fall_horizon_steps,
                    fall_thresholds=fall_thresholds,
                    fall_loss_weight=fall_loss_weight,
                    high_risk_horizon_steps=high_risk_horizon_steps,
                    high_risk_thresholds=high_risk_thresholds,
                    high_risk_loss_weight=high_risk_loss_weight,
                    high_risk_label_mode=high_risk_label_mode,
                    high_risk_onset_past_steps=high_risk_onset_past_steps,
                    smoothness_weight=smoothness_weight,
                    risk_pos_weight=risk_pos_weight,
                    fall_risk_pos_weight=fall_risk_pos_weight,
                    high_risk_pos_weight=high_risk_pos_weight,
                    loss_type=loss_type,
                    future_aux_horizon_steps=future_aux_horizon_steps,
                    future_aux_loss_weight=future_aux_loss_weight,
                    delta_aux_loss_weight=delta_aux_loss_weight,
                    event_aux_loss_weight=event_aux_loss_weight,
                    event_delta_threshold=event_delta_threshold,
                    anchor_break_weight=anchor_break_weight,
                    anchor_break_threshold=anchor_break_threshold,
                    anchor_break_max_weight=anchor_break_max_weight,
                    lds_weight_table=lds_weight_table,
                    lds_min=lds_min,
                    lds_bin_size=lds_bin_size,
                    transition_weighting=transition_weighting,
                    transition_horizon_steps=transition_horizon_steps,
                    transition_drop_threshold=transition_drop_threshold,
                    transition_recovery_threshold=transition_recovery_threshold,
                    transition_high_threshold=transition_high_threshold,
                    transition_low_threshold=transition_low_threshold,
                    transition_rise_threshold=transition_rise_threshold,
                    transition_drop_weight=transition_drop_weight,
                    transition_recovery_weight=transition_recovery_weight,
                    transition_rise_weight=transition_rise_weight,
                    transition_max_weight=transition_max_weight,
                    trajectory_loss_weight=trajectory_loss_weight,
                    trajectory_decoder_loss_weight=trajectory_decoder_loss_weight,
                    trajectory_delta_steps=trajectory_delta_steps,
                    trajectory_delta_weight=trajectory_delta_weight,
                    trajectory_centered_weight=trajectory_centered_weight,
                    trajectory_range_weight=trajectory_range_weight,
                    trajectory_loss_type=trajectory_loss_type,
                    trajectory_min_points=trajectory_min_points,
                    session_affine_scale_regularization_weight=session_affine_scale_regularization_weight,
                    session_affine_bias_regularization_weight=session_affine_bias_regularization_weight,
                    calibration_residual_regularization_weight=calibration_residual_regularization_weight,
                    low_overprediction_weight=low_overprediction_weight,
                    high_underprediction_weight=high_underprediction_weight,
                    low_overprediction_threshold=low_overprediction_threshold,
                    high_underprediction_threshold=high_underprediction_threshold,
                    low_suppressor_gate_loss_weight=low_suppressor_gate_loss_weight,
                    low_suppressor_threshold=low_suppressor_threshold,
                    low_suppressor_gate_pos_weight=low_suppressor_gate_pos_weight,
                    low_suppressor_gate_target_mode=low_suppressor_gate_target_mode,
                    low_suppressor_anchor_threshold=low_suppressor_anchor_threshold,
                    low_suppressor_recovery_delta=low_suppressor_recovery_delta,
                    low_suppressor_correction_regularization_weight=low_suppressor_correction_regularization_weight,
                    anchor_gate_loss_weight=anchor_gate_loss_weight,
                    anchor_gate_threshold=anchor_gate_threshold,
                    anchor_gate_pos_weight=anchor_gate_pos_weight,
                )
            else:
                loss, parts = compute_loss(
                    outputs,
                    fms,
                    lengths,
                    calibration_steps,
                    horizon_steps,
                    loss_fn,
                    dual_aux_alpha=dual_aux_alpha,
                    dual_aux_beta=dual_aux_beta,
                    change_aux_weight=change_aux_weight,
                    change_aux_threshold=change_aux_threshold,
                    current_aux_weight=current_aux_weight,
                    current_delta_aux_weight=current_delta_aux_weight,
                    session_aux_weight=session_aux_weight,
                    session_aux_loss_type=session_aux_loss_type,
                )
            if teacher_model is not None:
                assert teacher_info is not None
                teacher_static = static if bool(teacher_info.get("use_static", False)) else None
                if bool(teacher_info.get("use_static", False)) and teacher_static is None:
                    raise ValueError("Teacher model requires static features, but batch['static'] is missing.")
                teacher_calibration_steps = int(
                    getattr(
                        teacher_model,
                        "calibration_steps",
                        teacher_info.get("model_kwargs", {}).get("calibration_steps", calibration_steps),
                    )
                )
                teacher_fms_input = (
                    fms
                    if bool(getattr(teacher_model, "requires_full_fms", False))
                    else fms[:, :teacher_calibration_steps]
                )
                with torch.no_grad():
                    teacher_outputs = teacher_model(head, teacher_fms_input, lengths, static=teacher_static)
                if teacher_distill_weight > 0:
                    if task_mode == "online_current_risk":
                        distill_loss, distill_parts = compute_teacher_current_distillation_loss(
                            outputs,
                            teacher_outputs,
                            loss_type=teacher_distill_loss_type,
                        )
                    else:
                        distill_loss, distill_parts = compute_teacher_future_distillation_loss(
                            outputs,
                            teacher_outputs,
                            horizon_steps=horizon_steps,
                            loss_type=teacher_distill_loss_type,
                        )
                    loss = loss + teacher_distill_weight * distill_loss
                    parts.update(distill_parts)
                    parts["teacher_distill_weight"] = float(teacher_distill_weight)
                    parts["loss_total"] = float(loss.detach().cpu())
                delta_effective_weight = distill_weight_for_epoch(
                    teacher_delta_distill_weight,
                    epoch,
                    start_epoch=teacher_delta_distill_start_epoch,
                    warmup_epochs=teacher_delta_distill_warmup_epochs,
                )
                if delta_effective_weight > 0:
                    delta_loss, delta_parts = compute_teacher_delta_distillation_loss(
                        outputs,
                        teacher_outputs,
                        horizon_steps=horizon_steps,
                        loss_type=teacher_delta_distill_loss_type,
                    )
                    loss = loss + delta_effective_weight * delta_loss
                    parts.update(delta_parts)
                    parts["teacher_delta_distill_weight"] = float(teacher_delta_distill_weight)
                    parts["teacher_delta_distill_effective_weight"] = float(delta_effective_weight)
                    parts["loss_total"] = float(loss.detach().cpu())
                repr_effective_weight = distill_weight_for_epoch(
                    teacher_repr_distill_weight,
                    epoch,
                    start_epoch=teacher_repr_distill_start_epoch,
                    warmup_epochs=teacher_repr_distill_warmup_epochs,
                )
                if repr_effective_weight > 0:
                    repr_loss, repr_parts = compute_teacher_repr_distillation_loss(
                        outputs,
                        teacher_outputs,
                        horizon_steps=horizon_steps,
                        student_projector=teacher_repr_projector,
                        loss_type=teacher_repr_distill_loss_type,
                    )
                    loss = loss + repr_effective_weight * repr_loss
                    parts.update(repr_parts)
                    parts["teacher_repr_distill_weight"] = float(teacher_repr_distill_weight)
                    parts["teacher_repr_distill_effective_weight"] = float(repr_effective_weight)
                    parts["loss_total"] = float(loss.detach().cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer_params, grad_clip)
            optimizer.step()
            weight = max(parts["valid_points"], 1)
            for key, value in parts.items():
                if key.startswith("loss_"):
                    train_sums.setdefault(key, 0.0)
                    train_sums[key] += float(value) * weight
            train_points += max(parts["valid_points"], 1)

        train_losses = {key: value / max(train_points, 1) for key, value in train_sums.items()}
        fds_epoch_summary: Dict[str, Any] = {}
        if fds_enabled and hasattr(model, "commit_fds_epoch_stats"):
            fds_epoch_summary = {
                "fds_update_points": int(fds_update_points),
                "fds_apply_points": int(fds_apply_points),
                "fds_update_active": bool(fds_update_now),
                "fds_apply_active": bool(fds_apply_now),
                **model.commit_fds_epoch_stats(),
            }
            fds_history.append({"epoch": epoch, **fds_epoch_summary})
        val_loader = loaders.get("val") or loaders["train"]
        if task_mode == "online_current_risk":
            val_result = collect_online_current_risk_predictions(
                model,
                val_loader,
                device,
                calibration_steps,
                prepared["scalers"]["fms"],
                rise_horizon_steps=rise_horizon_steps,
                rise_thresholds=rise_thresholds,
                ordinal_bins=ordinal_bins,
                fall_horizon_steps=fall_horizon_steps,
                fall_thresholds=fall_thresholds,
                high_risk_horizon_steps=high_risk_horizon_steps,
                high_risk_thresholds=high_risk_thresholds,
                high_risk_label_mode=high_risk_label_mode,
                high_risk_onset_past_steps=high_risk_onset_past_steps,
                high_fms_caution_threshold=high_fms_caution_threshold,
                high_fms_warning_threshold=high_fms_warning_threshold,
                rapid_rise_probability_threshold=rapid_rise_probability_threshold,
                rapid_drop_probability_threshold=rapid_drop_probability_threshold,
                final_warning_mode=final_warning_mode,
                use_static=use_static,
                calibration_seconds=calibration_seconds,
                recent_window_seconds=recent_window_seconds,
                sampling_interval=float(prepared["data_info"]["sampling_interval"]),
                recent_window_steps=recent_steps,
                run_name=run_name,
                model_name=args.model,
                split_name="val",
                max_eval_batches=max_eval_batches,
                future_aux_horizon_steps=future_aux_horizon_steps,
            )
        else:
            val_result = collect_predictions(
                model,
                val_loader,
                device,
                calibration_steps,
                horizon_steps,
                prepared["scalers"]["fms"],
                high_fms_threshold=high_fms_threshold,
                use_static=use_static,
                calibration_seconds=calibration_seconds,
                horizon_seconds=horizon_seconds,
                recent_window_seconds=recent_window_seconds,
                common_eval_current_start=args.common_eval_current_start,
                common_eval_current_end=args.common_eval_current_end,
                common_eval_target_start=args.common_eval_target_start,
                common_eval_target_end=args.common_eval_target_end,
                common_eval_max_horizon_seconds=args.common_eval_max_horizon_seconds,
                sampling_interval=float(prepared["data_info"]["sampling_interval"]),
                recent_window_steps=recent_steps,
                run_name=run_name,
                model_name=args.model,
                split_name="val",
                anchor_mode=str(config["model"].get("anchor_mode", "none" if args.model != "lc_sa_tcnformer" else "calibration_end")),
                anchor_interval_seconds=float(config["model"].get("anchor_interval_seconds", 60.0)),
                fms_context_mode=str(config["model"].get("fms_context_mode", "calibration_history")),
                is_upper_bound_anchor=str(config["model"].get("anchor_mode", "")) == "recent_start_observed",
                max_eval_batches=max_eval_batches,
            )
        val_mae = float(val_result["metrics"]["mae"])
        val_selection_value = _metric_by_path(val_result["metrics"], selection_metric)
        row = {
            "epoch": epoch,
            "train_loss": train_losses,
            "val_metrics": val_result["metrics"],
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "selection_value": val_selection_value,
            "seconds": time.time() - start,
        }
        if fds_epoch_summary:
            row["fds"] = fds_epoch_summary
        history.append(row)
        save_training_curves(history, run_dir, plot=False)
        fds_text = ""
        if fds_enabled:
            fds_text = (
                f" fds_update={fds_epoch_summary.get('fds_update_points', 0)}"
                f" fds_apply={fds_epoch_summary.get('fds_apply_points', 0)}"
                f" fds_bins={fds_epoch_summary.get('fds_running_bins', 0)}"
            )
        print(
            f"epoch {epoch:03d} loss_total={train_losses['loss_total']:.5f} "
            f"loss_level={train_losses['loss_level']:.5f} loss_trend={train_losses['loss_trend']:.5f} "
            f"val_mae={human_float(val_mae)} val_rmse={human_float(val_result['metrics']['rmse'])} "
            f"val_shape={human_float(_metric_by_path(val_result['metrics'], 'trajectory.centered_mae_session_mean'))} "
            f"val_dir5={human_float(_metric_by_path(val_result['metrics'], 'trajectory.direction_acc_5s'))} "
            f"val_select[{selection_metric}]={human_float(val_selection_value)} "
            f"time={row['seconds']:.1f}s{fds_text}"
        )

        is_better = (
            np.isfinite(val_selection_value)
            and (
                val_selection_value < best_selection_value
                if selection_mode == "min"
                else val_selection_value > best_selection_value
            )
        )
        if is_better:
            best_selection_value = val_selection_value
            best_epoch = epoch
            bad_epochs = 0
            best_metrics = copy.deepcopy(val_result["metrics"])
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "model_kwargs": model_kwargs,
                "config": config,
                "column_mapping": prepared["mapping"],
                "scalers": prepared["scalers"],
                "data_info": prepared["data_info"],
                "split_info": prepared["split_info"],
                "loss": {
                    "mode": loss_mode,
                    "trend_weight": trend_weight,
                    "type": loss_type,
                    "horizon_weights": horizon_weights,
                    "change_weight": change_weight,
                    "high_target_weight": high_target_weight,
                    "high_target_threshold": high_target_threshold,
                    "low_target_weight": low_target_weight,
                    "low_target_threshold": low_target_threshold,
                    "dual_aux_alpha": dual_aux_alpha,
                    "dual_aux_beta": dual_aux_beta,
                    "change_aux_weight": change_aux_weight,
                    "change_aux_threshold": change_aux_threshold,
                    "current_aux_weight": current_aux_weight,
                    "current_delta_aux_weight": current_delta_aux_weight,
                    "session_aux_weight": session_aux_weight,
                    "session_aux_loss_type": session_aux_loss_type,
                    "current_reg_aux_weight": current_reg_aux_weight,
                    "ordinal_loss_weight": ordinal_loss_weight,
                    "ordinal_loss_mode": ordinal_loss_mode,
                    "ordinal_soft_label_sigma": ordinal_soft_label_sigma,
                    "ordinal_soft_label_kernel": ordinal_soft_label_kernel,
                    "ordinal_ev_loss_weight": ordinal_ev_loss_weight,
                    "ordinal_low_weight": ordinal_low_weight,
                    "ordinal_low_threshold": ordinal_low_threshold,
                    "ordinal_slace_alpha": ordinal_slace_alpha,
                    "ordinal_slace_proximity": ordinal_slace_proximity,
                    "ordinal_slace_normalize_proximity": ordinal_slace_normalize_proximity,
                    "ordinal_slace_count_smoothing": ordinal_slace_count_smoothing,
                    "risk_loss_weight": risk_loss_weight,
                    "fall_loss_weight": fall_loss_weight,
                    "high_risk_loss_weight": high_risk_loss_weight,
                    "smoothness_weight": smoothness_weight,
                    "future_aux_loss_weight": future_aux_loss_weight,
                    "delta_aux_loss_weight": delta_aux_loss_weight,
                    "event_aux_loss_weight": event_aux_loss_weight,
                    "event_delta_threshold": event_delta_threshold,
                    "trajectory_decoder_loss_weight": trajectory_decoder_loss_weight,
                    "trajectory_loss_weight": trajectory_loss_weight,
                    "trajectory_delta_seconds": trajectory_delta_seconds,
                    "trajectory_delta_weight": trajectory_delta_weight,
                    "trajectory_centered_weight": trajectory_centered_weight,
                    "trajectory_range_weight": trajectory_range_weight,
                    "trajectory_loss_type": trajectory_loss_type,
                    "trajectory_min_points": trajectory_min_points,
                    "session_affine_scale_regularization_weight": session_affine_scale_regularization_weight,
                    "session_affine_bias_regularization_weight": session_affine_bias_regularization_weight,
                    "calibration_residual_regularization_weight": calibration_residual_regularization_weight,
                    "low_overprediction_weight": low_overprediction_weight,
                    "high_underprediction_weight": high_underprediction_weight,
                    "low_overprediction_threshold": low_overprediction_threshold,
                    "high_underprediction_threshold": high_underprediction_threshold,
                    "low_suppressor_gate_loss_weight": low_suppressor_gate_loss_weight,
                    "low_suppressor_threshold": low_suppressor_threshold,
                    "low_suppressor_gate_pos_weight": low_suppressor_gate_pos_weight,
                    "low_suppressor_gate_target_mode": low_suppressor_gate_target_mode,
                    "low_suppressor_anchor_threshold": low_suppressor_anchor_threshold,
                    "low_suppressor_recovery_delta": low_suppressor_recovery_delta,
                    "low_suppressor_correction_regularization_weight": low_suppressor_correction_regularization_weight,
                    "anchor_gate_loss_weight": anchor_gate_loss_weight,
                    "anchor_gate_threshold": anchor_gate_threshold,
                    "anchor_gate_pos_weight": anchor_gate_pos_weight,
                    "anchor_break_weight": anchor_break_weight,
                    "anchor_break_threshold": anchor_break_threshold,
                    "anchor_break_max_weight": anchor_break_max_weight,
                    "transition_weighting": transition_weighting,
                    "transition_horizon_seconds": transition_horizon_seconds,
                    "transition_drop_threshold": transition_drop_threshold,
                    "transition_recovery_threshold": transition_recovery_threshold,
                    "transition_high_threshold": transition_high_threshold,
                    "transition_low_threshold": transition_low_threshold,
                    "transition_rise_threshold": transition_rise_threshold,
                    "transition_drop_weight": transition_drop_weight,
                    "transition_recovery_weight": transition_recovery_weight,
                    "transition_rise_weight": transition_rise_weight,
                    "transition_max_weight": transition_max_weight,
                    "lds_weighting": lds_weighting,
                    "lds_min": lds_min,
                    "lds_max": lds_max,
                    "lds_bin_size": lds_bin_size,
                    "lds_kernel": lds_kernel,
                    "lds_kernel_size": lds_kernel_size,
                    "lds_sigma": lds_sigma,
                    "lds_gamma": lds_gamma,
                    "lds_weight_min": lds_weight_min,
                    "lds_weight_max": lds_weight_max,
                    "risk_pos_weight": risk_pos_weight,
                    "fall_risk_pos_weight": fall_risk_pos_weight,
                    "high_risk_pos_weight": high_risk_pos_weight,
                    "teacher_checkpoint": teacher_checkpoint,
                    "teacher_distill_weight": teacher_distill_weight,
                    "teacher_distill_loss_type": teacher_distill_loss_type,
                    "teacher_delta_distill_weight": teacher_delta_distill_weight,
                    "teacher_delta_distill_loss_type": teacher_delta_distill_loss_type,
                    "teacher_delta_distill_start_epoch": teacher_delta_distill_start_epoch,
                    "teacher_delta_distill_warmup_epochs": teacher_delta_distill_warmup_epochs,
                    "teacher_repr_distill_weight": teacher_repr_distill_weight,
                    "teacher_repr_distill_loss_type": teacher_repr_distill_loss_type,
                    "teacher_repr_distill_start_epoch": teacher_repr_distill_start_epoch,
                    "teacher_repr_distill_warmup_epochs": teacher_repr_distill_warmup_epochs,
                },
                "teacher_distillation": teacher_info,
                "teacher_repr_projection": teacher_repr_projection_info,
                "lds_info": lds_info,
                "ordinal_class_count_info": ordinal_class_count_info,
                "fds_info": fds_info,
                "fds_history": fds_history,
                "teacher_repr_projector_state_dict": (
                    teacher_repr_projector.state_dict() if teacher_repr_projector is not None else None
                ),
                "static_report": prepared.get("static_report"),
                "best_epoch": best_epoch,
                "best_val_metrics": best_metrics,
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
                "best_selection_value": best_selection_value,
                "parameter_count": parameter_count,
            }
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    if best_epoch < 0 or not (run_dir / "best.pt").exists():
        raise RuntimeError("Training produced no finite validation MAE and no best checkpoint.")
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": args.model,
            "model_kwargs": model_kwargs,
            "config": config,
            "column_mapping": prepared["mapping"],
            "scalers": prepared["scalers"],
            "data_info": prepared["data_info"],
            "split_info": prepared["split_info"],
            "loss": {
                "mode": loss_mode,
                "trend_weight": trend_weight,
                "type": loss_type,
                "horizon_weights": horizon_weights,
                "change_weight": change_weight,
                "high_target_weight": high_target_weight,
                "high_target_threshold": high_target_threshold,
                "low_target_weight": low_target_weight,
                "low_target_threshold": low_target_threshold,
                "dual_aux_alpha": dual_aux_alpha,
                "dual_aux_beta": dual_aux_beta,
                "change_aux_weight": change_aux_weight,
                "change_aux_threshold": change_aux_threshold,
                "current_aux_weight": current_aux_weight,
                "current_delta_aux_weight": current_delta_aux_weight,
                "session_aux_weight": session_aux_weight,
                "session_aux_loss_type": session_aux_loss_type,
                "current_reg_aux_weight": current_reg_aux_weight,
                "ordinal_loss_weight": ordinal_loss_weight,
                "ordinal_loss_mode": ordinal_loss_mode,
                "ordinal_soft_label_sigma": ordinal_soft_label_sigma,
                "ordinal_soft_label_kernel": ordinal_soft_label_kernel,
                "ordinal_ev_loss_weight": ordinal_ev_loss_weight,
                "ordinal_low_weight": ordinal_low_weight,
                "ordinal_low_threshold": ordinal_low_threshold,
                "ordinal_slace_alpha": ordinal_slace_alpha,
                "ordinal_slace_proximity": ordinal_slace_proximity,
                "ordinal_slace_normalize_proximity": ordinal_slace_normalize_proximity,
                "ordinal_slace_count_smoothing": ordinal_slace_count_smoothing,
                "risk_loss_weight": risk_loss_weight,
                "fall_loss_weight": fall_loss_weight,
                "smoothness_weight": smoothness_weight,
                "future_aux_loss_weight": future_aux_loss_weight,
                "delta_aux_loss_weight": delta_aux_loss_weight,
                "event_aux_loss_weight": event_aux_loss_weight,
                "event_delta_threshold": event_delta_threshold,
                "trajectory_decoder_loss_weight": trajectory_decoder_loss_weight,
                "trajectory_loss_weight": trajectory_loss_weight,
                "trajectory_delta_seconds": trajectory_delta_seconds,
                "trajectory_delta_weight": trajectory_delta_weight,
                "trajectory_centered_weight": trajectory_centered_weight,
                "trajectory_range_weight": trajectory_range_weight,
                "trajectory_loss_type": trajectory_loss_type,
                "trajectory_min_points": trajectory_min_points,
                "session_affine_scale_regularization_weight": session_affine_scale_regularization_weight,
                "session_affine_bias_regularization_weight": session_affine_bias_regularization_weight,
                "calibration_residual_regularization_weight": calibration_residual_regularization_weight,
                "low_overprediction_weight": low_overprediction_weight,
                "high_underprediction_weight": high_underprediction_weight,
                "low_overprediction_threshold": low_overprediction_threshold,
                "high_underprediction_threshold": high_underprediction_threshold,
                "anchor_break_weight": anchor_break_weight,
                "anchor_break_threshold": anchor_break_threshold,
                "anchor_break_max_weight": anchor_break_max_weight,
                "transition_weighting": transition_weighting,
                "transition_horizon_seconds": transition_horizon_seconds,
                "transition_drop_threshold": transition_drop_threshold,
                "transition_recovery_threshold": transition_recovery_threshold,
                "transition_high_threshold": transition_high_threshold,
                "transition_low_threshold": transition_low_threshold,
                "transition_rise_threshold": transition_rise_threshold,
                "transition_drop_weight": transition_drop_weight,
                "transition_recovery_weight": transition_recovery_weight,
                "transition_rise_weight": transition_rise_weight,
                "transition_max_weight": transition_max_weight,
                "lds_weighting": lds_weighting,
                "lds_min": lds_min,
                "lds_max": lds_max,
                "lds_bin_size": lds_bin_size,
                "lds_kernel": lds_kernel,
                "lds_kernel_size": lds_kernel_size,
                "lds_sigma": lds_sigma,
                "lds_gamma": lds_gamma,
                "lds_weight_min": lds_weight_min,
                "lds_weight_max": lds_weight_max,
                "risk_pos_weight": risk_pos_weight,
                "fall_risk_pos_weight": fall_risk_pos_weight,
                "teacher_checkpoint": teacher_checkpoint,
                "teacher_distill_weight": teacher_distill_weight,
                "teacher_distill_loss_type": teacher_distill_loss_type,
                "teacher_delta_distill_weight": teacher_delta_distill_weight,
                "teacher_delta_distill_loss_type": teacher_delta_distill_loss_type,
                "teacher_delta_distill_start_epoch": teacher_delta_distill_start_epoch,
                "teacher_delta_distill_warmup_epochs": teacher_delta_distill_warmup_epochs,
                "teacher_repr_distill_weight": teacher_repr_distill_weight,
                "teacher_repr_distill_loss_type": teacher_repr_distill_loss_type,
                "teacher_repr_distill_start_epoch": teacher_repr_distill_start_epoch,
                "teacher_repr_distill_warmup_epochs": teacher_repr_distill_warmup_epochs,
            },
            "teacher_distillation": teacher_info,
            "teacher_repr_projection": teacher_repr_projection_info,
            "lds_info": lds_info,
            "ordinal_class_count_info": ordinal_class_count_info,
            "fds_info": fds_info,
            "fds_history": fds_history,
            "teacher_repr_projector_state_dict": (
                teacher_repr_projector.state_dict() if teacher_repr_projector is not None else None
            ),
            "best_epoch": best_epoch,
            "best_val_metrics": best_metrics,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "best_selection_value": best_selection_value,
            "parameter_count": parameter_count,
        },
        run_dir / "final.pt",
    )
    save_training_curves(history, run_dir)
    all_metrics: Dict[str, Any] = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "best_selection_value": best_selection_value,
        "parameter_count": parameter_count,
        "recent_rf_steps": getattr(model, "recent_rf_steps", None),
        "recent_rf_seconds": getattr(model, "recent_rf_seconds", None),
        "recent_dilations": getattr(model, "recent_dilations", None),
        "lds_info": lds_info,
        "ordinal_class_count_info": ordinal_class_count_info,
        "fds_info": fds_info,
        "fds_history": fds_history,
    }
    eval_splits = ["val"]
    skip_test_eval = bool(args.no_test_eval or config.get("evaluation", {}).get("no_test_eval", False))
    if not skip_test_eval:
        eval_splits.append("test")
    for split_name in eval_splits:
        if split_name in loaders:
            if task_mode == "online_current_risk":
                result = collect_online_current_risk_predictions(
                    model,
                    loaders[split_name],
                    device,
                    calibration_steps,
                    prepared["scalers"]["fms"],
                    rise_horizon_steps=rise_horizon_steps,
                    rise_thresholds=rise_thresholds,
                    ordinal_bins=ordinal_bins,
                    fall_horizon_steps=fall_horizon_steps,
                    fall_thresholds=fall_thresholds,
                    high_risk_horizon_steps=high_risk_horizon_steps,
                    high_risk_thresholds=high_risk_thresholds,
                    high_risk_label_mode=high_risk_label_mode,
                    high_risk_onset_past_steps=high_risk_onset_past_steps,
                    high_fms_caution_threshold=high_fms_caution_threshold,
                    high_fms_warning_threshold=high_fms_warning_threshold,
                    rapid_rise_probability_threshold=rapid_rise_probability_threshold,
                    rapid_drop_probability_threshold=rapid_drop_probability_threshold,
                    final_warning_mode=final_warning_mode,
                    use_static=use_static,
                    calibration_seconds=calibration_seconds,
                    recent_window_seconds=recent_window_seconds,
                    sampling_interval=float(prepared["data_info"]["sampling_interval"]),
                    recent_window_steps=recent_steps,
                    run_name=run_name,
                    model_name=args.model,
                    split_name=split_name,
                    max_eval_batches=max_eval_batches,
                    future_aux_horizon_steps=future_aux_horizon_steps,
                )
            else:
                result = collect_predictions(
                    model,
                    loaders[split_name],
                    device,
                    calibration_steps,
                    horizon_steps,
                    prepared["scalers"]["fms"],
                    high_fms_threshold=high_fms_threshold,
                    use_static=use_static,
                    calibration_seconds=calibration_seconds,
                    horizon_seconds=horizon_seconds,
                    recent_window_seconds=recent_window_seconds,
                    common_eval_current_start=args.common_eval_current_start,
                    common_eval_current_end=args.common_eval_current_end,
                    common_eval_target_start=args.common_eval_target_start,
                    common_eval_target_end=args.common_eval_target_end,
                    common_eval_max_horizon_seconds=args.common_eval_max_horizon_seconds,
                    sampling_interval=float(prepared["data_info"]["sampling_interval"]),
                    recent_window_steps=recent_steps,
                    run_name=run_name,
                    model_name=args.model,
                    split_name=split_name,
                    anchor_mode=str(config["model"].get("anchor_mode", "none" if args.model != "lc_sa_tcnformer" else "calibration_end")),
                    anchor_interval_seconds=float(config["model"].get("anchor_interval_seconds", 60.0)),
                    fms_context_mode=str(config["model"].get("fms_context_mode", "calibration_history")),
                    is_upper_bound_anchor=str(config["model"].get("anchor_mode", "")) == "recent_start_observed",
                    max_eval_batches=max_eval_batches,
                )
            all_metrics[f"{split_name}_metrics"] = result["metrics"]
            if args.save_plots:
                save_prediction_plots(result["series"], plots_dir, f"{split_name}_{loss_mode}")
            if args.save_predictions:
                save_prediction_csv(result["prediction_records"], run_dir / f"{split_name}_predictions.csv")

    summary = {
        "run_dir": str(run_dir),
        "model": args.model,
        "task": {
            "mode": task_mode,
            "rise_horizon_seconds": rise_horizon_seconds,
            "rise_horizon_steps": rise_horizon_steps,
            "fall_horizon_seconds": fall_horizon_seconds,
            "fall_horizon_steps": fall_horizon_steps,
            "high_risk_horizon_seconds": high_risk_horizon_seconds,
            "high_risk_horizon_steps": high_risk_horizon_steps,
            "high_risk_label_mode": high_risk_label_mode,
            "high_risk_onset_past_seconds": high_risk_onset_past_seconds,
            "high_risk_onset_past_steps": high_risk_onset_past_steps,
            "future_aux_horizon_seconds": future_aux_horizon_seconds,
            "future_aux_horizon_steps": future_aux_horizon_steps,
            "rise_thresholds": rise_thresholds,
            "fall_thresholds": fall_thresholds,
            "high_risk_thresholds": high_risk_thresholds,
            "ordinal_bins": ordinal_bins,
            "high_fms_caution_threshold": high_fms_caution_threshold,
            "high_fms_warning_threshold": high_fms_warning_threshold,
            "rapid_rise_probability_threshold": rapid_rise_probability_threshold,
            "rapid_drop_probability_threshold": rapid_drop_probability_threshold,
            "final_warning_mode": final_warning_mode,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "test_eval_skipped": skip_test_eval,
        },
        "inferred_columns": prepared["mapping"],
        "data_info": prepared["data_info"],
        "split_info": prepared["split_info"],
        "scalers": prepared["scalers"],
        "loss": {
            "mode": loss_mode,
            "trend_weight": trend_weight,
            "type": loss_type,
            "horizon_weights": horizon_weights,
            "change_weight": change_weight,
            "high_target_weight": high_target_weight,
            "high_target_threshold": high_target_threshold,
            "low_target_weight": low_target_weight,
            "low_target_threshold": low_target_threshold,
            "dual_aux_alpha": dual_aux_alpha,
            "dual_aux_beta": dual_aux_beta,
            "change_aux_weight": change_aux_weight,
            "change_aux_threshold": change_aux_threshold,
            "current_aux_weight": current_aux_weight,
            "current_delta_aux_weight": current_delta_aux_weight,
            "session_aux_weight": session_aux_weight,
            "session_aux_loss_type": session_aux_loss_type,
            "current_reg_aux_weight": current_reg_aux_weight,
            "ordinal_loss_weight": ordinal_loss_weight,
            "ordinal_loss_mode": ordinal_loss_mode,
            "ordinal_soft_label_sigma": ordinal_soft_label_sigma,
            "ordinal_soft_label_kernel": ordinal_soft_label_kernel,
            "ordinal_ev_loss_weight": ordinal_ev_loss_weight,
            "ordinal_low_weight": ordinal_low_weight,
            "ordinal_low_threshold": ordinal_low_threshold,
            "ordinal_slace_alpha": ordinal_slace_alpha,
            "ordinal_slace_proximity": ordinal_slace_proximity,
            "ordinal_slace_normalize_proximity": ordinal_slace_normalize_proximity,
            "ordinal_slace_count_smoothing": ordinal_slace_count_smoothing,
            "coarse_band_loss_weight": coarse_band_loss_weight,
            "coarse_residual_loss_weight": coarse_residual_loss_weight,
            "regime_loss_weight": regime_loss_weight,
            "regime_delta_slow_threshold": regime_delta_slow_threshold,
            "regime_delta_rapid_threshold": regime_delta_rapid_threshold,
            "regime_high_threshold": regime_high_threshold,
            "uncertainty_loss_weight": uncertainty_loss_weight,
            "risk_loss_weight": risk_loss_weight,
            "fall_loss_weight": fall_loss_weight,
            "high_risk_loss_weight": high_risk_loss_weight,
            "smoothness_weight": smoothness_weight,
            "future_aux_loss_weight": future_aux_loss_weight,
            "delta_aux_loss_weight": delta_aux_loss_weight,
            "event_aux_loss_weight": event_aux_loss_weight,
            "event_delta_threshold": event_delta_threshold,
            "trajectory_decoder_loss_weight": trajectory_decoder_loss_weight,
            "trajectory_loss_weight": trajectory_loss_weight,
            "trajectory_delta_seconds": trajectory_delta_seconds,
            "trajectory_delta_weight": trajectory_delta_weight,
            "trajectory_centered_weight": trajectory_centered_weight,
            "trajectory_range_weight": trajectory_range_weight,
            "trajectory_loss_type": trajectory_loss_type,
            "trajectory_min_points": trajectory_min_points,
            "session_affine_scale_regularization_weight": session_affine_scale_regularization_weight,
            "session_affine_bias_regularization_weight": session_affine_bias_regularization_weight,
            "calibration_residual_regularization_weight": calibration_residual_regularization_weight,
            "low_overprediction_weight": low_overprediction_weight,
            "high_underprediction_weight": high_underprediction_weight,
            "low_overprediction_threshold": low_overprediction_threshold,
            "high_underprediction_threshold": high_underprediction_threshold,
            "low_suppressor_gate_loss_weight": low_suppressor_gate_loss_weight,
            "low_suppressor_threshold": low_suppressor_threshold,
            "low_suppressor_gate_pos_weight": low_suppressor_gate_pos_weight,
            "low_suppressor_gate_target_mode": low_suppressor_gate_target_mode,
            "low_suppressor_anchor_threshold": low_suppressor_anchor_threshold,
            "low_suppressor_recovery_delta": low_suppressor_recovery_delta,
            "low_suppressor_correction_regularization_weight": low_suppressor_correction_regularization_weight,
            "anchor_gate_loss_weight": anchor_gate_loss_weight,
            "anchor_gate_threshold": anchor_gate_threshold,
            "anchor_gate_pos_weight": anchor_gate_pos_weight,
            "anchor_break_weight": anchor_break_weight,
            "anchor_break_threshold": anchor_break_threshold,
            "anchor_break_max_weight": anchor_break_max_weight,
            "transition_weighting": transition_weighting,
            "transition_horizon_seconds": transition_horizon_seconds,
            "transition_drop_threshold": transition_drop_threshold,
            "transition_recovery_threshold": transition_recovery_threshold,
            "transition_high_threshold": transition_high_threshold,
            "transition_low_threshold": transition_low_threshold,
            "transition_rise_threshold": transition_rise_threshold,
            "transition_drop_weight": transition_drop_weight,
            "transition_recovery_weight": transition_recovery_weight,
            "transition_rise_weight": transition_rise_weight,
            "transition_max_weight": transition_max_weight,
            "lds_weighting": lds_weighting,
            "lds_min": lds_min,
            "lds_max": lds_max,
            "lds_bin_size": lds_bin_size,
            "lds_kernel": lds_kernel,
            "lds_kernel_size": lds_kernel_size,
            "lds_sigma": lds_sigma,
            "lds_gamma": lds_gamma,
            "lds_weight_min": lds_weight_min,
            "lds_weight_max": lds_weight_max,
            "risk_pos_weight": risk_pos_weight,
            "fall_risk_pos_weight": fall_risk_pos_weight,
            "high_risk_pos_weight": high_risk_pos_weight,
            "teacher_checkpoint": teacher_checkpoint,
            "teacher_distill_weight": teacher_distill_weight,
            "teacher_distill_loss_type": teacher_distill_loss_type,
            "teacher_delta_distill_weight": teacher_delta_distill_weight,
            "teacher_delta_distill_loss_type": teacher_delta_distill_loss_type,
            "teacher_delta_distill_start_epoch": teacher_delta_distill_start_epoch,
            "teacher_delta_distill_warmup_epochs": teacher_delta_distill_warmup_epochs,
            "teacher_repr_distill_weight": teacher_repr_distill_weight,
            "teacher_repr_distill_loss_type": teacher_repr_distill_loss_type,
            "teacher_repr_distill_start_epoch": teacher_repr_distill_start_epoch,
            "teacher_repr_distill_warmup_epochs": teacher_repr_distill_warmup_epochs,
        },
        "teacher_distillation": teacher_info,
        "teacher_repr_projection": teacher_repr_projection_info,
        "init_checkpoint": init_checkpoint_info,
        "motion_pretraining": motion_pretrain_info,
        "lds_info": lds_info,
        "ordinal_class_count_info": ordinal_class_count_info,
        "fds_info": fds_info,
        "fds_history": fds_history,
        "static_report": prepared.get("static_report"),
        "metrics": all_metrics,
    }
    save_json(run_dir / "metrics.json", summary)
    print(f"Saved best checkpoint to {run_dir / 'best.pt'}")
    print(f"Saved metrics to {run_dir / 'metrics.json'}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DenseFMS online future FMS forecasters.")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument(
        "--model",
        default="coff_lstm",
        choices=[
            "coff_lstm",
            "recent10_tcn",
            "calib_only",
            "lc_sa_tcnformer",
            "calib_init_state_forecaster",
            "online_fms_risk_tracker",
            "lcsa_cross_attn",
            "gru_state_mixer",
            "motion_conv_mixer",
            "anchor_delta_mlp",
            "anchor_delta_gru",
            "recent_tcn_summary_calib",
            "gated_fusion",
        ],
    )
    p.add_argument("--device", default=None)
    p.add_argument("--runs_dir", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--participant_balanced_sampling", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--init_checkpoint", default=None)
    p.add_argument("--freeze_loaded_parameters", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--trainable_parameter_patterns", nargs="+", default=None)
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--max_eval_batches", type=int, default=None)
    p.add_argument("--limit_sessions", type=int, default=None)
    p.add_argument("--overfit_sessions", type=int, default=None)
    p.add_argument("--task_mode", choices=["future_forecast", "online_current_risk"], default=None)
    p.add_argument("--rise_horizon_seconds", nargs="+", type=float, default=None)
    p.add_argument("--rise_thresholds", nargs="+", type=float, default=None)
    p.add_argument("--fall_horizon_seconds", nargs="+", type=float, default=None)
    p.add_argument("--fall_thresholds", nargs="+", type=float, default=None)
    p.add_argument("--high_risk_horizon_seconds", nargs="+", type=float, default=None)
    p.add_argument("--high_risk_thresholds", nargs="+", type=float, default=None)
    p.add_argument(
        "--high_risk_label_mode",
        choices=["future_any", "current_below", "onset", "current_or_future"],
        default=None,
    )
    p.add_argument("--high_risk_onset_past_seconds", type=float, default=None)
    p.add_argument("--future_aux_horizon_seconds", nargs="+", type=float, default=None)
    p.add_argument("--high_fms_caution_threshold", type=float, default=None)
    p.add_argument("--high_fms_warning_threshold", type=float, default=None)
    p.add_argument("--rapid_rise_probability_threshold", type=float, default=None)
    p.add_argument("--rapid_drop_probability_threshold", type=float, default=None)
    p.add_argument("--final_warning_mode", choices=["high_or_rapid", "rapid_rise_only"], default=None)
    p.add_argument("--ordinal_bins", nargs="+", type=float, default=None)
    p.add_argument("--fms_combine_weight_ordinal", type=float, default=None)
    p.add_argument(
        "--current_head_mode",
        choices=[
            "basic",
            "dual_delta_gate",
            "paper_ordreg",
            "residual_update",
            "person_prior",
            "trajectory_decoder",
            "regime_gated",
            "anchor_regime_gated",
            "state_space_delta",
            "range_scaled_delta",
            "guarded_range_scaled_delta",
            "calib_prior_range_scaled_delta",
            "calib_lowcap_range_scaled_delta",
            "zero_anchor_mixture",
        ],
        default=None,
    )
    p.add_argument("--ordinal_head_mode", choices=["softmax", "cumulative", "clm", "coral", "corn"], default=None)
    p.add_argument("--current_delta_scale", type=float, default=None)
    p.add_argument("--current_anchor_delta_growth_scale", type=float, default=None)
    p.add_argument("--current_anchor_delta_growth_horizon_seconds", type=float, default=None)
    p.add_argument("--current_anchor_delta_growth_power", type=float, default=None)
    p.add_argument("--current_trajectory_offsets", nargs="+", type=int, default=None)
    p.add_argument("--current_range_guard_low_threshold", type=float, default=None)
    p.add_argument("--current_range_guard_temperature", type=float, default=None)
    p.add_argument("--current_range_guard_floor", type=float, default=None)
    p.add_argument("--current_range_guard_cap", type=float, default=None)
    p.add_argument("--current_range_guard_cap_strength", type=float, default=None)
    p.add_argument("--motion_encoder_context", choices=["linear", "tcn"], default=None)
    p.add_argument("--motion_encoder_layers", type=int, default=None)
    p.add_argument("--risk_head_enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--fall_risk_head_enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--high_risk_head_enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--current_low_suppressor_enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--current_low_suppressor_hidden_dim", type=int, default=None)
    p.add_argument("--current_low_suppressor_delta_range", type=float, default=None)
    p.add_argument("--current_low_suppressor_gate_init_bias", type=float, default=None)
    p.add_argument("--risk_temporal_context", choices=["none", "tcn"], default=None)
    p.add_argument("--risk_temporal_layers", type=int, default=None)
    p.add_argument("--selection_metric", default=None)
    p.add_argument("--selection_mode", choices=["min", "max"], default=None)
    p.add_argument("--current_reg_aux_weight", type=float, default=None)
    p.add_argument("--ordinal_loss_weight", type=float, default=None)
    p.add_argument(
        "--ordinal_loss_mode",
        choices=[
            "ce",
            "cross_entropy",
            "cumulative",
            "cumulative_bce",
            "coral_bce",
            "clm_nll",
            "cumulative_nll",
            "corn",
            "corn_bce",
            "soft_ce",
            "soft_label_ce",
            "unimodal_soft_ce",
            "slace",
            "slace_prox",
            "slace_paper",
            "slace_index",
            "slace_no_prox",
            "slace_norm_prox",
            "slace_prox_norm",
            "cdf_bce",
            "ordinal_cdf_bce",
            "oce",
            "oce_ts",
            "tpt_oce",
            "soft_oce_ts",
            "emd",
            "emd2",
            "soft_emd",
            "dldl_emd",
            "dldl",
            "wasserstein",
        ],
        default=None,
    )
    p.add_argument("--ordinal_soft_label_sigma", type=float, default=None)
    p.add_argument(
        "--ordinal_soft_label_kernel",
        choices=["gaussian", "laplace", "laplacian", "exponential", "triangular", "linear"],
        default=None,
    )
    p.add_argument("--ordinal_ev_loss_weight", type=float, default=None)
    p.add_argument("--ordinal_low_weight", type=float, default=None)
    p.add_argument("--ordinal_low_threshold", type=float, default=None)
    p.add_argument("--ordinal_slace_alpha", type=float, default=None)
    p.add_argument("--ordinal_slace_proximity", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--ordinal_slace_normalize_proximity", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--ordinal_slace_count_smoothing", type=float, default=None)
    p.add_argument("--coarse_band_bins", nargs="+", type=float, default=None)
    p.add_argument("--coarse_band_loss_weight", type=float, default=None)
    p.add_argument("--coarse_residual_head_enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--coarse_residual_range", type=float, default=None)
    p.add_argument("--coarse_residual_combine_weight", type=float, default=None)
    p.add_argument("--coarse_residual_loss_weight", type=float, default=None)
    p.add_argument("--regime_head_enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--regime_class_count", type=int, default=None)
    p.add_argument("--regime_loss_weight", type=float, default=None)
    p.add_argument("--regime_delta_slow_threshold", type=float, default=None)
    p.add_argument("--regime_delta_rapid_threshold", type=float, default=None)
    p.add_argument("--regime_high_threshold", type=float, default=None)
    p.add_argument("--uncertainty_head_enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--uncertainty_loss_weight", type=float, default=None)
    p.add_argument("--risk_loss_weight", type=float, default=None)
    p.add_argument("--fall_loss_weight", type=float, default=None)
    p.add_argument("--high_risk_loss_weight", type=float, default=None)
    p.add_argument("--smoothness_weight", type=float, default=None)
    p.add_argument("--anchor_break_weight", type=float, default=None)
    p.add_argument("--anchor_break_threshold", type=float, default=None)
    p.add_argument("--anchor_break_max_weight", type=float, default=None)
    p.add_argument("--transition_weighting", action="store_true")
    p.add_argument("--no_transition_weighting", action="store_true")
    p.add_argument("--transition_horizon_seconds", nargs="+", type=float, default=None)
    p.add_argument("--transition_drop_threshold", type=float, default=None)
    p.add_argument("--transition_recovery_threshold", type=float, default=None)
    p.add_argument("--transition_high_threshold", type=float, default=None)
    p.add_argument("--transition_low_threshold", type=float, default=None)
    p.add_argument("--transition_rise_threshold", type=float, default=None)
    p.add_argument("--transition_drop_weight", type=float, default=None)
    p.add_argument("--transition_recovery_weight", type=float, default=None)
    p.add_argument("--transition_rise_weight", type=float, default=None)
    p.add_argument("--transition_max_weight", type=float, default=None)
    p.add_argument("--lds_weighting", action="store_true")
    p.add_argument("--no_lds_weighting", action="store_true")
    p.add_argument("--lds_min", type=float, default=None)
    p.add_argument("--lds_max", type=float, default=None)
    p.add_argument("--lds_gamma", type=float, default=None)
    p.add_argument("--lds_kernel", choices=["gaussian", "triangular", "laplace"], default=None)
    p.add_argument("--lds_kernel_size", type=int, default=None)
    p.add_argument("--lds_sigma", type=float, default=None)
    p.add_argument("--lds_bin_size", type=float, default=None)
    p.add_argument("--lds_weight_min", type=float, default=None)
    p.add_argument("--lds_weight_max", type=float, default=None)
    p.add_argument("--fds_enabled", action="store_true")
    p.add_argument("--no_fds_enabled", action="store_true")
    p.add_argument("--fds_min", type=float, default=None)
    p.add_argument("--fds_max", type=float, default=None)
    p.add_argument("--fds_num_bins", type=int, default=None)
    p.add_argument("--fds_kernel", choices=["gaussian", "triangular", "laplace"], default=None)
    p.add_argument("--fds_kernel_size", type=int, default=None)
    p.add_argument("--fds_sigma", type=float, default=None)
    p.add_argument("--fds_momentum", type=float, default=None)
    p.add_argument("--fds_blend", type=float, default=None)
    p.add_argument("--fds_bin_size", type=float, default=None)
    p.add_argument("--fds_start_update", type=int, default=None)
    p.add_argument("--fds_start_smooth", type=int, default=None)
    p.add_argument("--risk_pos_weight", default=None)
    p.add_argument("--fall_risk_pos_weight", default=None)
    p.add_argument("--high_risk_pos_weight", default=None)
    p.add_argument("--loss_type", choices=["mse", "smooth_l1", "l1", "mae"], default=None)
    p.add_argument("--loss_mode", choices=["level_only", "level_trend_raw", "level_plus_trend"], default=None)
    p.add_argument("--trend_weight", type=float, default=None)
    p.add_argument("--horizon_loss_weights", nargs="+", type=float, default=None)
    p.add_argument("--change_weight", type=float, default=None)
    p.add_argument("--high_target_weight", type=float, default=None)
    p.add_argument("--high_target_threshold", type=float, default=None)
    p.add_argument("--low_target_weight", type=float, default=None)
    p.add_argument("--low_target_threshold", type=float, default=None)
    p.add_argument("--dual_aux_alpha", type=float, default=None)
    p.add_argument("--dual_aux_beta", type=float, default=None)
    p.add_argument("--change_aux_weight", type=float, default=None)
    p.add_argument("--change_aux_threshold", type=float, default=None)
    p.add_argument("--current_aux_weight", type=float, default=None)
    p.add_argument("--current_delta_aux_weight", type=float, default=None)
    p.add_argument("--future_aux_loss_weight", type=float, default=None)
    p.add_argument("--delta_aux_loss_weight", type=float, default=None)
    p.add_argument("--event_aux_loss_weight", type=float, default=None)
    p.add_argument("--event_delta_threshold", type=float, default=None)
    p.add_argument("--trajectory_loss_weight", type=float, default=None)
    p.add_argument("--trajectory_decoder_loss_weight", type=float, default=None)
    p.add_argument("--trajectory_delta_seconds", nargs="+", type=float, default=None)
    p.add_argument("--trajectory_delta_weight", type=float, default=None)
    p.add_argument("--trajectory_centered_weight", type=float, default=None)
    p.add_argument("--trajectory_range_weight", type=float, default=None)
    p.add_argument("--trajectory_loss_type", choices=["smooth_l1", "mse", "l1", "mae"], default=None)
    p.add_argument("--trajectory_min_points", type=int, default=None)
    p.add_argument("--session_affine_scale_regularization_weight", type=float, default=None)
    p.add_argument("--session_affine_bias_regularization_weight", type=float, default=None)
    p.add_argument("--calibration_residual_regularization_weight", type=float, default=None)
    p.add_argument("--low_overprediction_weight", type=float, default=None)
    p.add_argument("--high_underprediction_weight", type=float, default=None)
    p.add_argument("--low_overprediction_threshold", type=float, default=None)
    p.add_argument("--high_underprediction_threshold", type=float, default=None)
    p.add_argument("--low_suppressor_gate_loss_weight", type=float, default=None)
    p.add_argument("--low_suppressor_threshold", type=float, default=None)
    p.add_argument("--low_suppressor_gate_pos_weight", type=float, default=None)
    p.add_argument("--low_suppressor_gate_target_mode", choices=["low", "recovery_low", "anchor_drop_low"], default=None)
    p.add_argument("--low_suppressor_anchor_threshold", type=float, default=None)
    p.add_argument("--low_suppressor_recovery_delta", type=float, default=None)
    p.add_argument("--low_suppressor_correction_regularization_weight", type=float, default=None)
    p.add_argument("--anchor_gate_loss_weight", type=float, default=None)
    p.add_argument("--anchor_gate_threshold", type=float, default=None)
    p.add_argument("--anchor_gate_pos_weight", type=float, default=None)
    p.add_argument("--session_aux_weight", type=float, default=None)
    p.add_argument("--session_aux_loss_type", choices=["smooth_l1", "mse", "l1", "mae"], default=None)
    p.add_argument("--teacher_checkpoint", default=None)
    p.add_argument("--teacher_distill_weight", type=float, default=None)
    p.add_argument("--teacher_distill_loss_type", choices=["smooth_l1", "mse", "l1", "mae"], default=None)
    p.add_argument("--teacher_delta_distill_weight", type=float, default=None)
    p.add_argument("--teacher_delta_distill_loss_type", choices=["smooth_l1", "mse", "l1", "mae"], default=None)
    p.add_argument("--teacher_delta_distill_start_epoch", type=int, default=None)
    p.add_argument("--teacher_delta_distill_warmup_epochs", type=int, default=None)
    p.add_argument("--teacher_repr_distill_weight", type=float, default=None)
    p.add_argument("--teacher_repr_distill_loss_type", choices=["smooth_l1", "mse", "l1", "mae"], default=None)
    p.add_argument("--teacher_repr_distill_start_epoch", type=int, default=None)
    p.add_argument("--teacher_repr_distill_warmup_epochs", type=int, default=None)
    p.add_argument("--motion_pretrain_checkpoint", default=None)
    p.add_argument("--high_fms_threshold", type=float, default=None)
    p.add_argument("--split_file", default=None)
    p.add_argument("--use_static", action="store_true")
    p.add_argument("--no_static", action="store_true")
    p.add_argument("--static_features", nargs="+", default=None)
    p.add_argument("--gender_encoding", choices=["category3", "binary2"], default=None)
    p.add_argument("--allow_missing_static", action="store_true")
    p.add_argument("--no_film", action="store_true")
    p.add_argument("--no_recent_encoder", action="store_true")
    p.add_argument("--recent_encoder", choices=["tcn", "transformer"], default=None)
    p.add_argument("--recent_attn_heads", type=int, default=None)
    p.add_argument("--recent_attn_layers", type=int, default=None)
    p.add_argument("--recent_attn_dropout", type=float, default=None)
    p.add_argument("--no_aux_now", action="store_true")
    p.add_argument("--calibration_seconds", type=float, default=None)
    p.add_argument("--horizon_seconds", type=float, default=None)
    p.add_argument("--recent_window_seconds", type=float, default=None)
    p.add_argument("--max_session_points", type=int, default=None)
    p.add_argument("--head_channel_mode", choices=["all", "linear_only", "angular_only"], default=None)
    p.add_argument("--calibration_residual_features_path", nargs="+", default=None)
    p.add_argument("--require_calibration_residual_features", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--common_eval_current_start", type=float, default=None)
    p.add_argument("--common_eval_current_end", type=float, default=None)
    p.add_argument("--common_eval_target_start", type=float, default=None)
    p.add_argument("--common_eval_target_end", type=float, default=None)
    p.add_argument("--common_eval_max_horizon_seconds", type=float, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--mlp_layers", nargs="+", type=int, default=None)
    p.add_argument("--gru_layers", type=int, default=None)
    p.add_argument("--branch_dropout", type=float, default=None)
    p.add_argument("--anchor_dropout", type=float, default=None)
    p.add_argument("--delta_scale", type=float, default=None)
    p.add_argument("--kernel_size", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--calib_dilations", nargs="+", type=int, default=None)
    p.add_argument("--recent_dilations", default=None)
    p.add_argument("--transformer_layers", type=int, default=None)
    p.add_argument("--transformer_heads", type=int, default=None)
    p.add_argument("--transformer_ff_dim", type=int, default=None)
    p.add_argument("--pooling", choices=["mean", "last", "attention"], default=None)
    p.add_argument("--anchor_mode", choices=["none", "calibration_end", "recent_start_observed", "sparse_observed"], default=None)
    p.add_argument("--anchor_interval_seconds", type=float, default=None)
    p.add_argument("--fms_context_mode", choices=["none", "start_only", "calibration_history", "sparse_anchor"], default=None)
    p.add_argument("--predict_delta_from_anchor", action="store_true", default=None)
    p.add_argument("--multi_horizon", action="store_true", default=None)
    p.add_argument("--horizon_set", nargs="+", type=float, default=None)
    p.add_argument("--per_horizon_heads", action="store_true", default=None)
    p.add_argument("--horizon_encoder_dim", type=int, default=None)
    p.add_argument("--horizon_context_mode", choices=["encoded", "scalar", "none"], default=None)
    p.add_argument("--start_fms_context_mode", choices=["encoded", "scalar", "scalar_time"], default=None)
    p.add_argument("--static_context_mode", choices=["encoded", "raw"], default=None)
    p.add_argument(
        "--forecast_head_mode",
        choices=["level", "delta", "dual_average", "dual_gated", "self_delta", "recent_start_delta", "rollin_start_delta"],
        default=None,
    )
    p.add_argument("--horizon_head_mode", choices=["linear", "h15_deep", "h15_residual", "h10_h15_residual"], default=None)
    p.add_argument("--horizon_head_hidden_dim", type=int, default=None)
    p.add_argument(
        "--motion_feature_mode",
        choices=["none", "norm", "norm_delta", "norm_delta_energy", "causal_dynamics_v1", "multi_timescale_v1"],
        default=None,
    )
    p.add_argument("--motion_stats_branch", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--stream_time_features", action="store_true", default=None)
    p.add_argument(
        "--stream_context_mode",
        choices=[
            "gru",
            "gru_multiscale",
            "gru_tcn",
            "gru_tcn_multiscale",
            "deep_tcn",
            "deep_tcn_latent_gru",
            "transformer",
            "transformer_latent_gru",
        ],
        default=None,
    )
    p.add_argument("--stream_prepend_calibration", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--stream_calib_condition_mode", choices=["none", "film"], default=None)
    p.add_argument("--stream_calib_condition_strength", type=float, default=None)
    p.add_argument("--calib_summary_features", action="store_true", default=None)
    p.add_argument(
        "--calibration_fusion_mode",
        choices=[
            "add",
            "mean_last_summary_concat",
            "mean_last_gated_summary",
            "mean_last_attention_summary",
            "mean_last_event_attention_summary",
        ],
        default=None,
    )
    p.add_argument("--calibration_fusion_hidden_dim", type=int, default=None)
    p.add_argument("--calibration_fusion_output_dim", type=int, default=None)
    p.add_argument(
        "--calibration_encoder_mode",
        choices=["tcn_transformer", "transformer", "transformer_cls", "deep_tcn", "deep_tcn_transformer"],
        default=None,
    )
    p.add_argument("--state_feedback_mode", choices=["none", "predicted_current"], default=None)
    p.add_argument("--deep_tcn_dilations", nargs="+", type=int, default=None)
    p.add_argument("--calibration_tcn_adaptive_dilations", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--calibration_tcn_max_padding_steps", type=int, default=None)
    p.add_argument("--calibration_tcn_max_padding_fraction", type=float, default=None)
    p.add_argument("--decoder_hidden_dim", type=int, default=None)
    p.add_argument("--decoder_context_mode", choices=["fused", "state"], default=None)
    p.add_argument("--decoder_temporal_context", choices=["none", "tcn"], default=None)
    p.add_argument("--decoder_temporal_layers", type=int, default=None)
    p.add_argument("--calib_fms_dropout", type=float, default=None)
    p.add_argument("--calibration_end_fms_dropout", type=float, default=None)
    p.add_argument("--current_session_affine_head_enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--current_session_affine_hidden_dim", type=int, default=None)
    p.add_argument("--current_session_affine_scale_range", type=float, default=None)
    p.add_argument("--current_session_affine_bias_range", type=float, default=None)
    p.add_argument("--current_affine_head_enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--current_affine_hidden_dim", type=int, default=None)
    p.add_argument("--current_affine_scale_range", type=float, default=None)
    p.add_argument("--current_affine_bias_range", type=float, default=None)
    p.add_argument("--current_binned_affine_head_enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--current_binned_affine_anchor_bins", nargs="*", type=float, default=None)
    p.add_argument("--current_binned_affine_pred_bins", nargs="*", type=float, default=None)
    p.add_argument("--current_binned_affine_time_bins", nargs="*", type=float, default=None)
    p.add_argument("--current_binned_affine_scale_range", type=float, default=None)
    p.add_argument("--current_binned_affine_bias_range", type=float, default=None)
    p.add_argument("--calibration_residual_adapter_enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--calibration_residual_feature_dim", type=int, default=None)
    p.add_argument("--calibration_residual_adapter_hidden_dim", type=int, default=None)
    p.add_argument(
        "--calibration_residual_adapter_mode",
        choices=["mean_decay", "mlp", "mlp_decay", "mlp_high_gate", "mlp_decay_high_gate"],
        default=None,
    )
    p.add_argument("--calibration_residual_delta_range", type=float, default=None)
    p.add_argument("--calibration_residual_decay_seconds", type=float, default=None)
    p.add_argument("--calibration_residual_gate_low_threshold", type=float, default=None)
    p.add_argument("--calibration_residual_gate_high_threshold", type=float, default=None)
    p.add_argument("--calibration_residual_gate_anchor_threshold", type=float, default=None)
    p.add_argument("--calibration_residual_gate_temperature", type=float, default=None)
    p.add_argument("--calibration_summary_fusion_enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--calibration_summary_fusion_feature_dim", type=int, default=None)
    p.add_argument("--calibration_summary_fusion_hidden_dim", type=int, default=None)
    p.add_argument("--calibration_summary_fusion_mode", choices=["additive_gated", "film"], default=None)
    p.add_argument("--calibration_summary_fusion_strength", type=float, default=None)
    p.add_argument("--rollout_mode", choices=["none", "predicted"], default=None)
    p.add_argument("--session_context_mode", choices=["none", "summary"], default=None)
    p.add_argument("--change_aux_head", action="store_true", default=None)
    p.add_argument("--static_hidden_dim", type=int, default=None)
    p.add_argument("--static_dropout", type=float, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_plots", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no_test_eval", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    train_one_run(args)


if __name__ == "__main__":
    main()
