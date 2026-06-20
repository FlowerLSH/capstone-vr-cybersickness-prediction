"""Overlay level-only and level+trend predictions for matched DenseFMS samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .data import DenseFMSSessionDataset, apply_saved_split, collate_sessions, load_raw_sessions, transform_sessions
from .model import build_model
from .train import collect_predictions
from .utils import ensure_dir, normalize_time_config, set_seed


def _collect(checkpoint: str, data_dir: str, split: str, batch_size: int, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    normalize_time_config(config)
    raw_sessions, _, data_info = load_raw_sessions(
        data_dir,
        mapping=ckpt.get("column_mapping"),
        calibration_seconds=float(config["data"]["calibration_seconds"]),
        horizon_seconds=float(config["data"]["horizon_seconds"]),
        default_sampling_interval=float(config["data"].get("sampling_interval", config["data"].get("default_sampling_interval", 0.5))),
    )
    split_raw = apply_saved_split(raw_sessions, ckpt["split_info"])
    selected = split_raw[split]
    use_static = bool(ckpt.get("model_kwargs", {}).get("use_static", False))
    sessions = transform_sessions(
        selected,
        ckpt["scalers"],
        use_static=use_static,
        static_features=config.get("data", {}).get("static_features"),
        allow_missing_static=bool(config.get("data", {}).get("allow_missing_static", False)),
    )
    loader = DataLoader(
        DenseFMSSessionDataset(sessions),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sessions,
    )
    model = build_model(ckpt["model_name"], **ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    result = collect_predictions(
        model,
        loader,
        device,
        int(data_info["calibration_steps"]),
        int(data_info["horizon_steps"]),
        ckpt["scalers"]["fms"],
        high_fms_threshold=float(config.get("evaluation", {}).get("high_fms_threshold", 7.0)),
        use_static=use_static,
    )
    return result["series"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare level-only and level+raw-trend DenseFMS runs.")
    parser.add_argument("--level_only_checkpoint", required=True)
    parser.add_argument("--level_trend_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(42)
    level_series = _collect(args.level_only_checkpoint, args.data_dir, args.split, args.batch_size, device)
    trend_series = _collect(args.level_trend_checkpoint, args.data_dir, args.split, args.batch_size, device)
    out_dir = ensure_dir(args.output_dir or Path(args.level_trend_checkpoint).resolve().parent / "comparison_plots")

    for idx, (level, trend) in enumerate(zip(level_series, trend_series)):
        plt.figure(figsize=(10, 4))
        plt.plot(level["target_time"], level["target"], label="true FMS", linewidth=2)
        plt.plot(level["target_time"], level["prediction"], label="level_only prediction", linewidth=2)
        plt.plot(trend["target_time"], trend["prediction"], label="level_trend_raw prediction", linewidth=2)
        plt.xlabel("target timestamp (s)")
        plt.ylabel("FMS")
        plt.title(str(level["metadata"].get("session_id", f"session_{idx}"))[:80])
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"{args.split}_compare_{idx:02d}.png", dpi=140)
        plt.close()
    print(f"Saved comparison plots to {out_dir}")


if __name__ == "__main__":
    main()
