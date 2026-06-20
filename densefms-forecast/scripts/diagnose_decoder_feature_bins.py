#!/usr/bin/env python3
"""Diagnose decoder feature organization by FMS target bins.

The script is intentionally read-only: it loads a validation-selected checkpoint,
captures the tensor fed into ``current_reg_head`` with a forward pre-hook, and
summarizes whether decoder features are organized by current FMS bins.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.densefms_forecast.data import (
    DenseFMSSessionDataset,
    apply_saved_split,
    collate_sessions,
    load_raw_sessions,
    transform_sessions,
)
from src.densefms_forecast.model import build_model
from src.densefms_forecast.train import compute_online_current_risk_targets
from src.densefms_forecast.utils import ensure_dir, load_json, normalize_time_config, save_json, seconds_to_steps, set_seed


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - target)))


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(pred - target))))


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return float("nan")
    denom = float(np.sum(np.square(target - np.mean(target))))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum(np.square(pred - target)) / denom)


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return float("nan")
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _load_checkpoint_and_data(
    checkpoint: Path,
    data_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, DataLoader, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    set_seed(int(ckpt.get("config", {}).get("training", {}).get("seed", 42)))
    config = ckpt["config"]
    normalize_time_config(config)
    data_cfg = config["data"]
    raw_sessions, _, data_info = load_raw_sessions(
        data_dir,
        mapping=ckpt.get("column_mapping"),
        calibration_seconds=float(data_cfg["calibration_seconds"]),
        horizon_seconds=float(data_cfg["horizon_seconds"]),
        default_sampling_interval=float(data_cfg.get("sampling_interval", data_cfg.get("default_sampling_interval", 0.5))),
        max_session_points=data_cfg.get("max_session_points"),
    )
    split_raw = apply_saved_split(raw_sessions, ckpt["split_info"])
    selected_raw = raw_sessions if split == "all" else split_raw.get(split, [])
    if not selected_raw:
        raise RuntimeError(f"No sessions available for split '{split}'.")
    use_static = bool(ckpt.get("model_kwargs", {}).get("use_static", False))
    sessions = transform_sessions(
        selected_raw,
        ckpt["scalers"],
        use_static=use_static,
        static_features=config.get("data", {}).get("static_features"),
        allow_missing_static=bool(config.get("data", {}).get("allow_missing_static", False)),
    )
    loader = DataLoader(
        DenseFMSSessionDataset(sessions),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sessions,
    )
    model_kwargs = dict(ckpt["model_kwargs"])
    model_kwargs["calibration_steps"] = int(data_info["calibration_steps"])
    model_kwargs["horizon_steps"] = int(data_info["horizon_steps"])
    model_kwargs["sampling_interval"] = float(data_info["sampling_interval"])
    model_kwargs["horizon_seconds"] = float(data_cfg["horizon_seconds"])
    model_kwargs["recent_steps"] = seconds_to_steps(
        float(data_cfg["recent_window_seconds"]),
        float(data_info["sampling_interval"]),
        name="recent_window_seconds",
    )
    model = build_model(ckpt["model_name"], **model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, loader, ckpt, config, data_info


def _collect_features(
    model: torch.nn.Module,
    loader: DataLoader,
    ckpt: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    fms_scaler = ckpt["scalers"]["fms"]
    f_min = float(fms_scaler["min"])
    f_max = float(fms_scaler["max"])
    f_range = max(f_max - f_min, 1e-8)
    task_cfg = config.get("task", {})
    model_kwargs = ckpt["model_kwargs"]
    rise_horizon_steps = [int(v) for v in model_kwargs.get("rise_horizon_steps", [int(model_kwargs.get("horizon_steps", 10))])]
    rise_thresholds = [float(v) for v in task_cfg.get("rise_thresholds", model_kwargs.get("rise_thresholds", [2.0]))]
    thresholds_norm = [value / f_range for value in rise_thresholds]
    ordinal_bins_raw = [float(v) for v in model_kwargs.get("ordinal_bins", list(range(21)))]
    ordinal_bins_norm = [(value - f_min) / f_range for value in ordinal_bins_raw]

    feature_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []
    pred_chunks: List[torch.Tensor] = []
    reg_chunks: List[torch.Tensor] = []
    ord_chunks: List[torch.Tensor] = []
    bin_chunks: List[torch.Tensor] = []
    session_ids: List[str] = []
    current_indices: List[int] = []

    captured: List[torch.Tensor] = []

    def hook(_module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach())

    handle = model.current_reg_head.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for batch in loader:
                head = batch["head"].to(device)
                fms = batch["fms"].to(device)
                lengths = batch["lengths"].to(device)
                static = batch.get("static")
                static = static.to(device) if static is not None else None
                captured.clear()
                outputs = model(head, fms[:, : int(ckpt["model_kwargs"]["calibration_steps"])], lengths, static=static)
                if not captured:
                    raise RuntimeError("current_reg_head hook did not capture decoder features.")
                features = captured[-1]
                targets = compute_online_current_risk_targets(
                    fms,
                    lengths,
                    int(outputs["prediction_start"].detach().cpu().item()),
                    int(outputs["current"].shape[1]),
                    rise_horizon_steps,
                    thresholds_norm,
                    ordinal_bins_norm,
                )
                mask = outputs["mask"].bool() & targets["current_mask"].to(device).bool()
                if int(mask.sum().item()) == 0:
                    continue
                target_raw = targets["current"].to(device) * f_range + f_min
                pred_raw = outputs["current"] * f_range + f_min
                reg_raw = outputs["current_reg"] * f_range + f_min
                ord_raw = outputs["current_ordinal"] * f_range + f_min
                bins = torch.round(target_raw).clamp(0, 20).long()
                feature_chunks.append(features[mask].detach().cpu())
                target_chunks.append(target_raw[mask].detach().cpu())
                pred_chunks.append(pred_raw[mask].detach().cpu())
                reg_chunks.append(reg_raw[mask].detach().cpu())
                ord_chunks.append(ord_raw[mask].detach().cpu())
                bin_chunks.append(bins[mask].detach().cpu())

                positions = targets["positions"].detach().cpu().numpy()
                mask_np = mask.detach().cpu().numpy()
                for batch_idx, metadata in enumerate(batch["metadata"]):
                    valid = np.where(mask_np[batch_idx])[0]
                    session_id = str(metadata.get("session_id", ""))
                    for j in valid:
                        session_ids.append(session_id)
                        current_indices.append(int(positions[int(j)]))
    finally:
        handle.remove()

    if not feature_chunks:
        raise RuntimeError("No valid decoder features were collected.")
    return {
        "features": torch.cat(feature_chunks, dim=0).numpy().astype(np.float64),
        "target": torch.cat(target_chunks, dim=0).numpy().astype(np.float64),
        "prediction": torch.cat(pred_chunks, dim=0).numpy().astype(np.float64),
        "regression": torch.cat(reg_chunks, dim=0).numpy().astype(np.float64),
        "ordinal": torch.cat(ord_chunks, dim=0).numpy().astype(np.float64),
        "bin": torch.cat(bin_chunks, dim=0).numpy().astype(np.int64),
        "session_id": np.asarray(session_ids, dtype=object),
        "current_index": np.asarray(current_indices, dtype=np.int64),
    }


def _bin_summary(data: Mapping[str, np.ndarray]) -> Tuple[List[Dict[str, Any]], Dict[int, np.ndarray], Dict[int, float]]:
    x = data["features"]
    y = data["target"]
    pred = data["prediction"]
    reg = data["regression"]
    ordv = data["ordinal"]
    bins = data["bin"]
    rows: List[Dict[str, Any]] = []
    means: Dict[int, np.ndarray] = {}
    within_rms: Dict[int, float] = {}
    for bin_idx in range(21):
        mask = bins == bin_idx
        count = int(mask.sum())
        row: Dict[str, Any] = {"bin": bin_idx, "count": count}
        if count:
            xb = x[mask]
            mean = xb.mean(axis=0)
            means[bin_idx] = mean
            centered = xb - mean.reshape(1, -1)
            sq_dist = np.sum(centered * centered, axis=1)
            within = float(np.sqrt(np.mean(sq_dist)))
            within_rms[bin_idx] = within
            row.update(
                {
                    "target_mean": float(np.mean(y[mask])),
                    "pred_mean": float(np.mean(pred[mask])),
                    "mae": _mae(pred[mask], y[mask]),
                    "reg_mae": _mae(reg[mask], y[mask]),
                    "ordinal_mae": _mae(ordv[mask], y[mask]),
                    "bias": float(np.mean(pred[mask] - y[mask])),
                    "feature_norm_mean": float(np.mean(np.linalg.norm(xb, axis=1))),
                    "feature_within_rms": within,
                    "feature_trace_var": float(np.mean(np.var(xb, axis=0))),
                }
            )
        rows.append(row)
    for row in rows:
        bin_idx = int(row["bin"])
        if bin_idx in means:
            for other_key, suffix in ((bin_idx - 1, "prev"), (bin_idx + 1, "next")):
                if other_key in means:
                    dist = float(np.linalg.norm(means[bin_idx] - means[other_key]))
                    denom = 0.5 * (within_rms[bin_idx] + within_rms[other_key])
                    row[f"centroid_dist_{suffix}"] = dist
                    row[f"centroid_dist_{suffix}_over_within"] = dist / denom if denom > 1e-12 else float("nan")
    return rows, means, within_rms


def _centroid_distance_summary(means: Mapping[int, np.ndarray], within_rms: Mapping[int, float]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    label_diffs: List[float] = []
    distances: List[float] = []
    overlap_ratios: List[float] = []
    keys = sorted(means.keys())
    for i, bin_i in enumerate(keys):
        for bin_j in keys[i + 1 :]:
            dist = float(np.linalg.norm(means[bin_i] - means[bin_j]))
            label_diff = float(abs(bin_j - bin_i))
            denom = 0.5 * (within_rms[bin_i] + within_rms[bin_j])
            ratio = dist / denom if denom > 1e-12 else float("nan")
            rows.append(
                {
                    "bin_i": bin_i,
                    "bin_j": bin_j,
                    "label_diff": label_diff,
                    "centroid_distance": dist,
                    "centroid_distance_over_within": ratio,
                }
            )
            label_diffs.append(label_diff)
            distances.append(dist)
            overlap_ratios.append(ratio)
    adjacent = [row for row in rows if int(row["label_diff"]) == 1]
    summary = {
        "centroid_distance_labeldiff_pearson": _pearson(label_diffs, distances),
        "centroid_distance_labeldiff_spearman": _spearman(label_diffs, distances),
        "adjacent_centroid_distance_mean": float(np.mean([row["centroid_distance"] for row in adjacent])) if adjacent else float("nan"),
        "adjacent_centroid_over_within_mean": float(np.nanmean([row["centroid_distance_over_within"] for row in adjacent])) if adjacent else float("nan"),
        "all_centroid_over_within_mean": float(np.nanmean(overlap_ratios)) if overlap_ratios else float("nan"),
        "valid_centroid_bins": int(len(keys)),
    }
    return rows, summary


def _fit_ridge_probe(
    train: Mapping[str, np.ndarray],
    eval_sets: Mapping[str, Mapping[str, np.ndarray]],
    alpha: float,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    x_train = train["features"]
    y_train = train["target"]
    feat_mean = x_train.mean(axis=0, keepdims=True)
    feat_std = x_train.std(axis=0, keepdims=True)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    x_std = (x_train - feat_mean) / feat_std
    x_aug = np.concatenate([x_std, np.ones((x_std.shape[0], 1), dtype=np.float64)], axis=1)
    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_train)
    metrics: Dict[str, Any] = {"ridge_alpha": float(alpha), "feature_dim": int(x_train.shape[1])}
    predictions: Dict[str, np.ndarray] = {}
    for split, data in eval_sets.items():
        xs = (data["features"] - feat_mean) / feat_std
        xs = np.concatenate([xs, np.ones((xs.shape[0], 1), dtype=np.float64)], axis=1)
        pred = xs @ weights
        pred = np.clip(pred, 0.0, 20.0)
        predictions[split] = pred
        target = data["target"]
        metrics[f"{split}_mae"] = _mae(pred, target)
        metrics[f"{split}_rmse"] = _rmse(pred, target)
        metrics[f"{split}_r2"] = _r2(pred, target)
        metrics[f"{split}_corr"] = _pearson(pred, target)
    return metrics, predictions


def _many_medium_few_rows(
    split: str,
    data: Mapping[str, np.ndarray],
    train_counts: Mapping[int, int],
    thresholds: Tuple[int, int],
    probe_pred: np.ndarray | None = None,
) -> List[Dict[str, Any]]:
    many_min, medium_min = thresholds
    groups = {"many": [], "medium": [], "few": []}
    for bin_idx in range(21):
        count = int(train_counts.get(bin_idx, 0))
        if count >= many_min:
            groups["many"].append(bin_idx)
        elif count >= medium_min:
            groups["medium"].append(bin_idx)
        else:
            groups["few"].append(bin_idx)
    rows: List[Dict[str, Any]] = []
    bins = data["bin"]
    for name, bin_list in groups.items():
        mask = np.isin(bins, np.asarray(bin_list, dtype=np.int64))
        row: Dict[str, Any] = {
            "split": split,
            "group": name,
            "bins": " ".join(str(v) for v in bin_list),
            "n": int(mask.sum()),
        }
        if int(mask.sum()) > 0:
            row.update(
                {
                    "mae": _mae(data["prediction"][mask], data["target"][mask]),
                    "reg_mae": _mae(data["regression"][mask], data["target"][mask]),
                    "ordinal_mae": _mae(data["ordinal"][mask], data["target"][mask]),
                }
            )
            if probe_pred is not None:
                row["linear_probe_mae"] = _mae(probe_pred[mask], data["target"][mask])
        rows.append(row)
    return rows


def _run_for_split(args: argparse.Namespace, split: str, device: torch.device) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
    model, loader, ckpt, config, _data_info = _load_checkpoint_and_data(
        Path(args.checkpoint),
        Path(args.data_dir),
        split,
        int(args.batch_size),
        device,
    )
    data = _collect_features(model, loader, ckpt, config, device)
    metadata = {
        "split": split,
        "n": int(data["features"].shape[0]),
        "feature_dim": int(data["features"].shape[1]),
        "decoder_context_mode": ckpt.get("model_kwargs", {}).get("decoder_context_mode", "fused"),
        "use_static": bool(ckpt.get("model_kwargs", {}).get("use_static", False)),
        "static_dim": int(ckpt.get("model_kwargs", {}).get("static_dim", 0)),
    }
    return data, metadata, dict(ckpt.get("model_kwargs", {}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose decoder feature bins for DIR/FDS feasibility.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test", "all"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--many_threshold", type=int, default=4000)
    parser.add_argument("--medium_threshold", type=int, default=1000)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_data: Dict[str, Dict[str, np.ndarray]] = {}
    split_metadata: Dict[str, Dict[str, Any]] = {}
    model_kwargs: Dict[str, Any] = {}
    for split in args.splits:
        data, metadata, model_kwargs = _run_for_split(args, split, device)
        all_data[split] = data
        split_metadata[split] = metadata
        rows, means, within = _bin_summary(data)
        distance_rows, distance_summary = _centroid_distance_summary(means, within)
        _write_csv(
            output_dir / f"{split}_feature_bin_summary.csv",
            rows,
            [
                "bin",
                "count",
                "target_mean",
                "pred_mean",
                "mae",
                "reg_mae",
                "ordinal_mae",
                "bias",
                "feature_norm_mean",
                "feature_within_rms",
                "feature_trace_var",
                "centroid_dist_prev",
                "centroid_dist_prev_over_within",
                "centroid_dist_next",
                "centroid_dist_next_over_within",
            ],
        )
        _write_csv(
            output_dir / f"{split}_feature_centroid_distances.csv",
            distance_rows,
            ["bin_i", "bin_j", "label_diff", "centroid_distance", "centroid_distance_over_within"],
        )
        split_metadata[split].update(distance_summary)

    probe_metrics: Dict[str, Any] = {}
    probe_predictions: Dict[str, np.ndarray] = {}
    if "train" in all_data:
        probe_metrics, probe_predictions = _fit_ridge_probe(all_data["train"], all_data, alpha=float(args.ridge_alpha))
    train_counts = {
        int(row["bin"]): int(row["count"])
        for row in csv.DictReader((output_dir / "train_feature_bin_summary.csv").open(newline="", encoding="utf-8"))
    } if "train" in all_data else {}
    mmf_rows: List[Dict[str, Any]] = []
    for split, data in all_data.items():
        mmf_rows.extend(
            _many_medium_few_rows(
                split,
                data,
                train_counts,
                (int(args.many_threshold), int(args.medium_threshold)),
                probe_predictions.get(split),
            )
        )
    _write_csv(
        output_dir / "many_medium_few_summary.csv",
        mmf_rows,
        ["split", "group", "bins", "n", "mae", "reg_mae", "ordinal_mae", "linear_probe_mae"],
    )
    summary = {
        "checkpoint": str(Path(args.checkpoint)),
        "splits": split_metadata,
        "model_kwargs": {
            "decoder_context_mode": model_kwargs.get("decoder_context_mode", "fused"),
            "use_static": bool(model_kwargs.get("use_static", False)),
            "static_dim": int(model_kwargs.get("static_dim", 0)),
            "current_head_mode": model_kwargs.get("current_head_mode"),
            "ordinal_head_mode": model_kwargs.get("ordinal_head_mode"),
            "hidden_dim": model_kwargs.get("hidden_dim"),
            "d_model": model_kwargs.get("d_model"),
        },
        "linear_probe": probe_metrics,
        "many_threshold": int(args.many_threshold),
        "medium_threshold": int(args.medium_threshold),
        "outputs": {
            "bin_summary": [f"{split}_feature_bin_summary.csv" for split in args.splits],
            "centroid_distances": [f"{split}_feature_centroid_distances.csv" for split in args.splits],
            "many_medium_few": "many_medium_few_summary.csv",
        },
    }
    save_json(output_dir / "feature_bin_diagnostic_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
