"""Generate leakage-safe calibration residual features for online-current FMS runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from densefms_forecast.data import DenseFMSSessionDataset, apply_saved_split, collate_sessions, load_raw_sessions, transform_sessions
from densefms_forecast.model import build_model
from densefms_forecast.utils import denormalize_fms, ensure_dir, load_json, normalize_time_config, seconds_to_steps, set_seed


FEATURE_NAMES = [
    "residual_mean",
    "residual_std",
    "residual_last",
    "residual_slope_per_min",
    "residual_low_mean",
    "residual_mid_mean",
    "residual_high_mean",
    "has_low_residual",
    "has_high_residual",
    "probe_mae",
    "calibration_fms_mean",
    "calibration_fms_slope_per_min",
    "calibration_fms_range",
]


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else 0.0


def _safe_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.std()) if values.size else 0.0


def _safe_last(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values[-1]) if values.size else 0.0


def _slope_per_minute(times_seconds: np.ndarray, values: np.ndarray) -> float:
    times = np.asarray(times_seconds, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(times) & np.isfinite(vals)
    if int(finite.sum()) < 2:
        return 0.0
    x = times[finite] / 60.0
    x = x - float(x.mean())
    denom = float(np.sum(x * x))
    if denom <= 1e-12:
        return 0.0
    y = vals[finite] - float(vals[finite].mean())
    return float(np.sum(x * y) / denom)


def _session_feature_row(
    *,
    meta: Mapping[str, Any],
    split_name: str,
    residual_norm: np.ndarray,
    residual_raw: np.ndarray,
    target_raw: np.ndarray,
    pred_raw: np.ndarray,
    times_seconds: np.ndarray,
    calib_fms_norm: np.ndarray,
    calib_times_seconds: np.ndarray,
) -> Dict[str, Any]:
    low_mask = np.isfinite(target_raw) & (target_raw <= 2.0)
    high_mask = np.isfinite(target_raw) & (target_raw >= 15.0)
    mid_mask = np.isfinite(target_raw) & (target_raw > 2.0) & (target_raw < 15.0)
    features = [
        _safe_mean(residual_norm),
        _safe_std(residual_norm),
        _safe_last(residual_norm),
        _slope_per_minute(times_seconds, residual_norm),
        _safe_mean(residual_norm[low_mask]),
        _safe_mean(residual_norm[mid_mask]),
        _safe_mean(residual_norm[high_mask]),
        float(bool(np.any(low_mask))),
        float(bool(np.any(high_mask))),
        _safe_mean(np.abs(residual_norm)),
        _safe_mean(calib_fms_norm),
        _slope_per_minute(calib_times_seconds, calib_fms_norm),
        float(np.nanmax(calib_fms_norm) - np.nanmin(calib_fms_norm)) if np.isfinite(calib_fms_norm).any() else 0.0,
    ]
    source_file = str(meta.get("source_file") or "")
    source_path = Path(source_file)
    clean_features = [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in features]
    return {
        "split": split_name,
        "session_key": source_file or str(meta.get("session_id") or ""),
        "participant_id": meta.get("participant_id"),
        "session_id": meta.get("session_id"),
        "source_file": source_file,
        "source_file_name": source_path.name,
        "source_file_stem": source_path.stem,
        "feature_names": list(FEATURE_NAMES),
        "features": clean_features,
        "probe_points": int(np.isfinite(residual_norm).sum()),
        "probe_residual_mean_raw": _safe_mean(residual_raw),
        "probe_residual_std_raw": _safe_std(residual_raw),
        "probe_mae_raw": _safe_mean(np.abs(np.asarray(pred_raw, dtype=np.float64) - np.asarray(target_raw, dtype=np.float64))),
    }


@torch.no_grad()
def generate_features(
    *,
    checkpoint: str | Path,
    data_dir: str | Path,
    split_file: str | Path | None,
    splits: Sequence[str],
    output: str | Path,
    probe_prefix_seconds: float,
    probe_end_seconds: float,
    batch_size: int,
    device_name: str | None,
    max_session_points: int | None,
) -> Dict[str, Any]:
    if probe_prefix_seconds <= 0 or probe_end_seconds <= probe_prefix_seconds:
        raise ValueError("Require 0 < probe_prefix_seconds < probe_end_seconds.")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    set_seed(int(ckpt.get("config", {}).get("training", {}).get("seed", 42)))
    config = ckpt["config"]
    normalize_time_config(config)
    data_cfg = dict(config["data"])
    data_cfg["calibration_seconds"] = float(probe_prefix_seconds)
    if max_session_points is not None:
        data_cfg["max_session_points"] = int(max_session_points)
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
    sampling_interval = float(data_info["sampling_interval"])
    prefix_steps = int(data_info["calibration_steps"])
    probe_end_steps = seconds_to_steps(float(probe_end_seconds), sampling_interval, name="probe_end_seconds")
    if probe_end_steps <= prefix_steps:
        raise ValueError("probe_end_seconds must convert to a step after probe_prefix_seconds.")

    model_kwargs = dict(ckpt["model_kwargs"])
    model_kwargs["calibration_steps"] = prefix_steps
    model_kwargs["horizon_steps"] = int(data_info["horizon_steps"])
    model_kwargs["sampling_interval"] = sampling_interval
    model_kwargs["horizon_seconds"] = float(data_cfg["horizon_seconds"])
    model_kwargs["recent_steps"] = seconds_to_steps(
        float(data_cfg["recent_window_seconds"]),
        sampling_interval,
        name="recent_window_seconds",
    )
    model_kwargs["calibration_residual_adapter_enabled"] = False
    model = build_model(ckpt["model_name"], **model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    rows: List[Dict[str, Any]] = []
    fms_scaler = ckpt["scalers"]["fms"]
    use_static = bool(model_kwargs.get("use_static", False))
    allow_missing_static = bool(config.get("data", {}).get("allow_missing_static", False))
    for split_name in splits:
        selected_raw = raw_sessions if split_name == "all" else split_raw.get(split_name, [])
        if not selected_raw:
            continue
        sessions = transform_sessions(
            selected_raw,
            ckpt["scalers"],
            use_static=use_static,
            static_features=config.get("data", {}).get("static_features"),
            allow_missing_static=allow_missing_static,
        )
        loader = DataLoader(
            DenseFMSSessionDataset(sessions),
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_sessions,
        )
        for batch in loader:
            head = batch["head"].to(device)
            fms = batch["fms"].to(device)
            lengths = batch["lengths"].to(device)
            static = batch.get("static")
            if use_static:
                if static is None:
                    raise ValueError("Checkpoint uses static features, but batch['static'] is missing.")
                static = static.to(device)
            outputs = model(head, fms[:, :prefix_steps], lengths, static=static)
            current = outputs["current"].to(device)
            if int(current.shape[1]) == 0:
                continue
            prediction_start = int(outputs["prediction_start"].detach().cpu().item())
            positions = prediction_start + torch.arange(current.shape[1], device=device)
            target_idx = positions.clamp(max=max(int(fms.shape[1]) - 1, 0)).view(1, -1).expand(fms.shape[0], -1)
            target_norm = fms.gather(1, target_idx)
            valid = (
                outputs["mask"].to(device).bool()
                & (positions.view(1, -1) >= int(prefix_steps))
                & (positions.view(1, -1) < int(probe_end_steps))
                & (positions.view(1, -1) < lengths.view(-1, 1))
            )
            pred_raw = denormalize_fms(current, fms_scaler)
            target_raw = denormalize_fms(target_norm, fms_scaler)
            fms_raw_np = batch.get("fms_raw", batch["fms"]).detach().cpu().numpy()
            time_np = batch["time"].detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()
            residual_norm_np = (target_norm - current).detach().cpu().numpy()
            residual_raw_np = target_raw - pred_raw
            positions_np = positions.detach().cpu().numpy()
            for b, meta in enumerate(batch["metadata"]):
                valid_j = np.where(valid_np[b])[0]
                if valid_j.size == 0:
                    continue
                pos = positions_np[valid_j].astype(np.int64)
                calib_end = min(prefix_steps, int(lengths[b].detach().cpu().item()))
                calib_idx = np.arange(calib_end, dtype=np.int64)
                rows.append(
                    _session_feature_row(
                        meta=meta,
                        split_name=split_name,
                        residual_norm=residual_norm_np[b, valid_j],
                        residual_raw=residual_raw_np[b, valid_j],
                        target_raw=target_raw[b, valid_j],
                        pred_raw=pred_raw[b, valid_j],
                        times_seconds=time_np[b, pos],
                        calib_fms_norm=fms[b, :calib_end].detach().cpu().numpy(),
                        calib_times_seconds=time_np[b, calib_idx],
                    )
                )

    out_path = Path(output)
    ensure_dir(out_path.parent)
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "features": rows,
        "metadata": {
            "checkpoint": str(checkpoint),
            "data_dir": str(data_dir),
            "split_file": str(split_file) if split_file else None,
            "splits": list(splits),
            "probe_prefix_seconds": float(probe_prefix_seconds),
            "probe_end_seconds": float(probe_end_seconds),
            "probe_prefix_steps": int(prefix_steps),
            "probe_end_steps": int(probe_end_steps),
            "sampling_interval": float(sampling_interval),
            "leakage_policy": "FMS input limited to probe_prefix; residual labels limited to calibration probe window.",
        },
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    csv_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k not in {"feature_names", "features"}}
        flat.update({f"feature__{name}": value for name, value in zip(FEATURE_NAMES, row["features"])})
        csv_rows.append(flat)
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"Saved {len(rows)} session residual feature rows to {out_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate calibration residual features for online-current FMS runs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test", "all"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe_prefix_seconds", type=float, default=60.0)
    parser.add_argument("--probe_end_seconds", type=float, default=120.0)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_session_points", type=int, default=None)
    args = parser.parse_args()
    generate_features(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        split_file=args.split_file,
        splits=args.splits,
        output=args.output,
        probe_prefix_seconds=args.probe_prefix_seconds,
        probe_end_seconds=args.probe_end_seconds,
        batch_size=args.batch_size,
        device_name=args.device,
        max_session_points=args.max_session_points,
    )


if __name__ == "__main__":
    main()
