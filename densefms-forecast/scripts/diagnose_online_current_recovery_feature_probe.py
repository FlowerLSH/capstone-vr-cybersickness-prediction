#!/usr/bin/env python3
"""Probe whether online-current decoder features separate recovery-low states.

This is a diagnostic script, not a model-selection or training script.  It loads
an existing checkpoint, captures the final decoder fused feature with a forward
hook, trains a fixed linear probe on the train split, selects only a probability
threshold on validation, and reports how the probe transfers to validation/test.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.densefms_forecast.data import (
    DenseFMSSessionDataset,
    apply_saved_split,
    collate_sessions,
    load_calibration_residual_features,
    load_raw_sessions,
    transform_sessions,
)
from src.densefms_forecast.model import build_model
from src.densefms_forecast.train import compute_online_current_risk_targets
from src.densefms_forecast.utils import ensure_dir, load_json, normalize_time_config, save_json, seconds_to_steps, set_seed


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        starts = np.cumsum(np.r_[0, counts[:-1]])
        ranks = (starts + (counts - 1) / 2.0)[inverse]
    return ranks + 1.0


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    labels = labels[mask].astype(bool)
    scores = scores[mask].astype(float)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    pos_rank_sum = float(ranks[labels].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    labels = labels[mask].astype(bool)
    scores = scores[mask].astype(float)
    if labels.size == 0 or int(labels.sum()) == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order].astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(float(labels.sum()), 1.0)
    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _prf(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    mask = np.isfinite(scores)
    labels = labels[mask].astype(bool)
    scores = scores[mask].astype(float)
    pred = scores >= float(threshold)
    tp = float(np.sum(pred & labels))
    fp = float(np.sum(pred & ~labels))
    fn = float(np.sum(~pred & labels))
    tn = float(np.sum(~pred & ~labels))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_rate": float(np.mean(pred)) if pred.size else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _select_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 99), scores[np.isfinite(scores)]])
    best_threshold = 0.5
    best = _prf(labels, scores, best_threshold)
    for threshold in thresholds:
        row = _prf(labels, scores, float(threshold))
        key = (row["f1"], row["precision"], row["recall"], -abs(float(threshold) - 0.5))
        best_key = (best["f1"], best["precision"], best["recall"], -abs(best_threshold - 0.5))
        if key > best_key:
            best_threshold = float(threshold)
            best = row
    return best_threshold, best


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_checkpoint_and_data(
    checkpoint: Path,
    data_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
    split_file: Optional[Path] = None,
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
    split_info = load_json(split_file) if split_file else ckpt["split_info"]
    split_raw = apply_saved_split(raw_sessions, split_info)
    selected_raw = raw_sessions if split == "all" else split_raw.get(split, [])
    if not selected_raw:
        raise RuntimeError(f"No sessions available for split {split!r}.")
    use_static = bool(ckpt.get("model_kwargs", {}).get("use_static", False))
    residual_path = data_cfg.get("calibration_residual_features_path")
    residual_feature_map = None
    residual_feature_names = None
    if residual_path:
        residual_feature_map, residual_feature_names, _ = load_calibration_residual_features(residual_path)
    residual_required = bool(ckpt.get("model_kwargs", {}).get("calibration_residual_adapter_enabled", False))
    sessions = transform_sessions(
        selected_raw,
        ckpt["scalers"],
        use_static=use_static,
        static_features=data_cfg.get("static_features"),
        allow_missing_static=bool(data_cfg.get("allow_missing_static", False)),
        calibration_residual_feature_map=residual_feature_map,
        calibration_residual_feature_names=residual_feature_names,
        require_calibration_residual_features=residual_required,
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


def _feature_hook_module(model: torch.nn.Module) -> torch.nn.Module:
    decoder_blocks = getattr(model, "decoder_temporal_blocks", None)
    if decoder_blocks is not None:
        return decoder_blocks
    fusion = getattr(model, "fusion", None)
    if fusion is None:
        raise AttributeError("Model has neither decoder_temporal_blocks nor fusion module to hook.")
    return fusion


def _collect_split_features(
    model: torch.nn.Module,
    loader: DataLoader,
    ckpt: Mapping[str, Any],
    config: Mapping[str, Any],
    data_info: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    fms_scaler = ckpt["scalers"]["fms"]
    f_min = float(fms_scaler["min"])
    f_max = float(fms_scaler["max"])
    f_range = max(f_max - f_min, 1e-8)
    task_cfg = config.get("task", {})
    model_kwargs = ckpt["model_kwargs"]
    calibration_steps = int(data_info["calibration_steps"])
    sampling_interval = float(data_info["sampling_interval"])
    rise_horizon_steps = [int(v) for v in model_kwargs.get("rise_horizon_steps", [int(data_info["horizon_steps"])])]
    rise_thresholds = [float(v) for v in task_cfg.get("rise_thresholds", model_kwargs.get("rise_thresholds", [2.0]))]
    thresholds_norm = [float(value - f_min) / f_range for value in rise_thresholds]
    ordinal_bins = [float(v) for v in model_kwargs.get("ordinal_bins", [0, 2, 4, 6, 8, 10, 12, 15, 20])]
    ordinal_bins_norm = [float(value - f_min) / f_range for value in ordinal_bins]

    captured: List[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured.append(output.detach())

    handle = _feature_hook_module(model).register_forward_hook(hook)
    feature_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []
    pred_chunks: List[torch.Tensor] = []
    reg_chunks: List[torch.Tensor] = []
    ordinal_chunks: List[torch.Tensor] = []
    anchor_chunks: List[torch.Tensor] = []
    index_chunks: List[torch.Tensor] = []
    session_ids: List[str] = []
    participant_ids: List[str] = []
    try:
        with torch.no_grad():
            for batch in loader:
                head = batch["head"].to(device)
                fms = batch["fms"].to(device)
                lengths = batch["lengths"].to(device)
                static = batch.get("static")
                if static is not None:
                    static = static.to(device)
                residual_features = batch.get("calibration_residual_features")
                residual_feature_mask = batch.get("calibration_residual_feature_mask")
                residual_adapter_enabled = bool(getattr(model, "calibration_residual_adapter_enabled", False))
                model_kwargs_forward: Dict[str, Any] = {"static": static}
                if residual_adapter_enabled:
                    if residual_features is None:
                        raise RuntimeError("Residual adapter checkpoint requires calibration residual features.")
                    model_kwargs_forward["calibration_residual_features"] = residual_features.to(device)
                    model_kwargs_forward["calibration_residual_feature_mask"] = (
                        residual_feature_mask.to(device) if residual_feature_mask is not None else None
                    )
                captured.clear()
                outputs = model(head, fms[:, :calibration_steps], lengths, **model_kwargs_forward)
                if not captured:
                    raise RuntimeError("Decoder feature hook did not capture any tensor.")
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
                mask = outputs["mask"].to(device).bool() & targets["current_mask"].to(device).bool()
                if int(mask.sum().item()) == 0:
                    continue
                target_raw = targets["current"].to(device) * f_range + f_min
                pred_raw = outputs["current"].to(device) * f_range + f_min
                reg_raw = outputs.get("current_reg", outputs["current"]).to(device) * f_range + f_min
                ordinal_raw = outputs.get("current_ordinal", outputs["current"]).to(device) * f_range + f_min
                fms_raw = batch.get("fms_raw")
                if isinstance(fms_raw, torch.Tensor):
                    fms_raw = fms_raw.to(device)
                else:
                    fms_raw = fms * f_range + f_min
                anchor = fms_raw[:, calibration_steps - 1].view(-1, 1).expand(-1, int(outputs["current"].shape[1]))
                positions = targets["positions"].to(device).view(1, -1).expand_as(outputs["current"])

                feature_chunks.append(features[mask].detach().cpu())
                target_chunks.append(target_raw[mask].detach().cpu())
                pred_chunks.append(pred_raw[mask].detach().cpu())
                reg_chunks.append(reg_raw[mask].detach().cpu())
                ordinal_chunks.append(ordinal_raw[mask].detach().cpu())
                anchor_chunks.append(anchor[mask].detach().cpu())
                index_chunks.append(positions[mask].detach().cpu())

                mask_np = mask.detach().cpu().numpy()
                for batch_idx, metadata in enumerate(batch["metadata"]):
                    valid_count = int(mask_np[batch_idx].sum())
                    session_ids.extend([str(metadata.get("session_id", ""))] * valid_count)
                    participant_ids.extend([str(metadata.get("participant_id", ""))] * valid_count)
    finally:
        handle.remove()
    if not feature_chunks:
        raise RuntimeError("No valid features collected.")
    current_index = torch.cat(index_chunks, dim=0).numpy().astype(np.float64)
    target = torch.cat(target_chunks, dim=0).numpy().astype(np.float64)
    prediction = torch.cat(pred_chunks, dim=0).numpy().astype(np.float64)
    regression = torch.cat(reg_chunks, dim=0).numpy().astype(np.float64)
    ordinal = torch.cat(ordinal_chunks, dim=0).numpy().astype(np.float64)
    anchor = torch.cat(anchor_chunks, dim=0).numpy().astype(np.float64)
    context = np.stack(
        [
            anchor,
            prediction,
            regression,
            ordinal,
            prediction - anchor,
            regression - anchor,
            current_index * sampling_interval,
            np.maximum(current_index - calibration_steps, 0.0) * sampling_interval,
        ],
        axis=1,
    )
    return {
        "features": torch.cat(feature_chunks, dim=0).numpy().astype(np.float64),
        "context": context,
        "target": target,
        "prediction": prediction,
        "regression": regression,
        "ordinal": ordinal,
        "anchor": anchor,
        "current_index": current_index,
        "session_id": np.asarray(session_ids, dtype=object),
        "participant_id": np.asarray(participant_ids, dtype=object),
    }


def _labels(data: Mapping[str, np.ndarray], low_threshold: float, anchor_threshold: float, recovery_delta: float) -> Dict[str, np.ndarray]:
    target = data["target"]
    anchor = data["anchor"]
    finite = np.isfinite(target) & np.isfinite(anchor)
    low = finite & (target < float(low_threshold))
    return {
        "low": low,
        "recovery_low": low & (anchor >= float(anchor_threshold)),
        "anchor_drop_low": low & ((anchor - target) >= float(recovery_delta)),
    }


def _feature_sets(data: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "context_only": data["context"],
        "decoder_only": data["features"],
        "decoder_plus_context": np.concatenate([data["features"], data["context"]], axis=1),
    }


def _fit_probe(x: np.ndarray, y: np.ndarray, seed: int) -> Any:
    if int(y.sum()) == 0 or int((~y.astype(bool)).sum()) == 0:
        raise ValueError("Cannot fit probe with one class only.")
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=int(seed),
            solver="lbfgs",
        ),
    ).fit(x, y.astype(int))


def _evaluate_scores(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    row = {
        "n": int(labels.size),
        "positive_count": int(labels.astype(bool).sum()),
        "positive_rate": float(labels.astype(bool).mean()) if labels.size else float("nan"),
        "auroc": _auroc(labels, scores),
        "auprc": _auprc(labels, scores),
    }
    row.update(_prf(labels, scores, threshold))
    return row


def _bin_bias_rows(split: str, data: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
    bins = [(0, 2, "0_2"), (2, 5, "2_5"), (5, 10, "5_10"), (10, 15, "10_15"), (15, 20.000001, "15_20")]
    rows: List[Dict[str, Any]] = []
    target = data["target"]
    pred = data["prediction"]
    anchor = data["anchor"]
    for lo, hi, label in bins:
        mask = np.isfinite(target) & (target >= lo) & (target < hi)
        rows.append(
            {
                "split": split,
                "bin": label,
                "n": int(mask.sum()),
                "target_mean": float(np.mean(target[mask])) if mask.any() else float("nan"),
                "pred_mean": float(np.mean(pred[mask])) if mask.any() else float("nan"),
                "anchor_mean": float(np.mean(anchor[mask])) if mask.any() else float("nan"),
                "bias": float(np.mean(pred[mask] - target[mask])) if mask.any() else float("nan"),
            }
        )
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "nan"
    return f"{f:.{digits}f}"


def _write_report(
    path: Path,
    checkpoint: Path,
    model_kwargs: Mapping[str, Any],
    threshold_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    bin_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Recovery-Low Feature Probe Report",
        "",
        f"Checkpoint: `{checkpoint}`",
        "",
        "이 보고서는 checkpoint 내부 decoder feature가 low/recovery-low 상태를 선형적으로 구분할 수 있는지 보는 진단이다.",
        "Probe는 train split feature로만 학습했고, threshold는 validation F1 기준으로만 선택했다. Test 결과는 선택 이후 사후 확인이다.",
        "",
        "## Model",
        "",
        f"- current_head_mode: `{model_kwargs.get('current_head_mode')}`",
        f"- decoder_context_mode: `{model_kwargs.get('decoder_context_mode')}`",
        f"- calibration_steps: `{model_kwargs.get('calibration_steps')}`",
        f"- recent_steps: `{model_kwargs.get('recent_steps')}`",
        "",
        "## FMS Bin Bias",
        "",
        "| split | bin | n | target mean | pred mean | anchor mean | bias |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in bin_rows:
        lines.append(
            "| {split} | {bin} | {n} | {target} | {pred} | {anchor} | {bias} |".format(
                split=row["split"],
                bin=row["bin"],
                n=row["n"],
                target=_fmt(row["target_mean"]),
                pred=_fmt(row["pred_mean"]),
                anchor=_fmt(row["anchor_mean"]),
                bias=_fmt(row["bias"]),
            )
        )
    lines.extend(["", "## Validation-Selected Thresholds", "", "| label | feature set | threshold | val F1 | val P | val R |", "|---|---|---:|---:|---:|---:|"])
    for row in threshold_rows:
        lines.append(
            "| {label} | {feature_set} | {thr} | {f1} | {p} | {r} |".format(
                label=row["label"],
                feature_set=row["feature_set"],
                thr=_fmt(row["threshold"]),
                f1=_fmt(row["val_f1"]),
                p=_fmt(row["val_precision"]),
                r=_fmt(row["val_recall"]),
            )
        )
    lines.extend(["", "## Probe Metrics", "", "| label | feature set | split | pos rate | AUPRC | AUROC | P | R | F1 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in metric_rows:
        lines.append(
            "| {label} | {feature_set} | {split} | {pos} | {auprc} | {auroc} | {p} | {r} | {f1} |".format(
                label=row["label"],
                feature_set=row["feature_set"],
                split=row["split"],
                pos=_fmt(row["positive_rate"]),
                auprc=_fmt(row["auprc"]),
                auroc=_fmt(row["auroc"]),
                p=_fmt(row["precision"]),
                r=_fmt(row["recall"]),
                f1=_fmt(row["f1"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose recovery-low separability from online-current decoder features.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test", "all"])
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--low_threshold", type=float, default=2.0)
    parser.add_argument("--anchor_threshold", type=float, default=5.0)
    parser.add_argument("--recovery_delta", type=float, default=4.0)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = Path(args.checkpoint)
    split_file = Path(args.split_file) if args.split_file else None
    all_data: Dict[str, Dict[str, np.ndarray]] = {}
    model_kwargs: Dict[str, Any] = {}
    ckpt_seed = 42
    split_shapes: Dict[str, Dict[str, int]] = {}
    for split in args.splits:
        model, loader, ckpt, config, data_info = _load_checkpoint_and_data(
            checkpoint,
            Path(args.data_dir),
            split,
            int(args.batch_size),
            device,
            split_file=split_file,
        )
        ckpt_seed = int(ckpt.get("config", {}).get("training", {}).get("seed", 42))
        model_kwargs = dict(ckpt.get("model_kwargs", {}))
        data = _collect_split_features(model, loader, ckpt, config, data_info, device)
        all_data[split] = data
        split_shapes[split] = {"n": int(data["features"].shape[0]), "feature_dim": int(data["features"].shape[1])}

    if "train" not in all_data or "val" not in all_data:
        raise RuntimeError("This diagnostic requires at least train and val splits.")

    labels_by_split = {
        split: _labels(data, args.low_threshold, args.anchor_threshold, args.recovery_delta)
        for split, data in all_data.items()
    }
    metric_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    bin_rows: List[Dict[str, Any]] = []
    for split, data in all_data.items():
        bin_rows.extend(_bin_bias_rows(split, data))

    for label_name in ("low", "recovery_low", "anchor_drop_low"):
        y_train = labels_by_split["train"][label_name]
        y_val = labels_by_split["val"][label_name]
        for feature_set_name, x_train in _feature_sets(all_data["train"]).items():
            try:
                probe = _fit_probe(x_train, y_train, seed=ckpt_seed)
            except ValueError as exc:
                threshold_rows.append(
                    {
                        "label": label_name,
                        "feature_set": feature_set_name,
                        "threshold": float("nan"),
                        "val_f1": float("nan"),
                        "val_precision": float("nan"),
                        "val_recall": float("nan"),
                        "error": str(exc),
                    }
                )
                continue
            val_scores = probe.predict_proba(_feature_sets(all_data["val"])[feature_set_name])[:, 1]
            threshold, val_prf = _select_threshold(y_val, val_scores)
            threshold_rows.append(
                {
                    "label": label_name,
                    "feature_set": feature_set_name,
                    "threshold": threshold,
                    "val_f1": val_prf["f1"],
                    "val_precision": val_prf["precision"],
                    "val_recall": val_prf["recall"],
                }
            )
            for split, data in all_data.items():
                scores = probe.predict_proba(_feature_sets(data)[feature_set_name])[:, 1]
                row = _evaluate_scores(labels_by_split[split][label_name], scores, threshold)
                row.update({"label": label_name, "feature_set": feature_set_name, "split": split})
                metric_rows.append(row)

    _write_csv(output_dir / "feature_probe_metrics.csv", metric_rows)
    _write_csv(output_dir / "val_selected_thresholds.csv", threshold_rows)
    _write_csv(output_dir / "fms_bin_bias.csv", bin_rows)
    summary = {
        "checkpoint": str(checkpoint),
        "splits": split_shapes,
        "label_definition": {
            "low": f"target < {float(args.low_threshold):g}",
            "recovery_low": f"target < {float(args.low_threshold):g} and anchor >= {float(args.anchor_threshold):g}",
            "anchor_drop_low": f"target < {float(args.low_threshold):g} and anchor-target >= {float(args.recovery_delta):g}",
        },
        "model_kwargs": {
            "current_head_mode": model_kwargs.get("current_head_mode"),
            "decoder_context_mode": model_kwargs.get("decoder_context_mode"),
            "calibration_steps": model_kwargs.get("calibration_steps"),
            "recent_steps": model_kwargs.get("recent_steps"),
            "use_static": model_kwargs.get("use_static"),
        },
        "outputs": {
            "metrics": "feature_probe_metrics.csv",
            "thresholds": "val_selected_thresholds.csv",
            "bin_bias": "fms_bin_bias.csv",
            "report": "feature_probe_report.md",
        },
    }
    save_json(output_dir / "feature_probe_summary.json", summary)
    _write_report(output_dir / "feature_probe_report.md", checkpoint, model_kwargs, threshold_rows, metric_rows, bin_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
