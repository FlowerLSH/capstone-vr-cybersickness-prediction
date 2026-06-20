"""Self-supervised motion encoder pretraining for online-current DenseFMS runs.

The objective is intentionally label-free: predict near-future motion-energy
summaries from causal motion history. The saved encoder can initialize
OnlineFMSRiskTracker.deep_tcn_stream when the input feature configuration
matches.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.densefms_forecast.data import DenseFMSSessionDataset, collate_sessions, apply_saved_split
from src.densefms_forecast.model import DeepTCNEncoder, append_motion_features
from src.densefms_forecast.train import prepare_data
from src.densefms_forecast.utils import ensure_dir, load_config, load_json, normalize_time_config, save_json, seconds_to_steps, set_seed


class MotionEnergyPretrainer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizons: Sequence[int],
        dilations: Sequence[int],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = DeepTCNEncoder(input_dim, hidden_dim, dilations=dilations, kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Linear(hidden_dim, len(list(horizons)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def _append_time_features(motion: torch.Tensor, calibration_steps: int, sampling_interval: float) -> torch.Tensor:
    steps = motion.shape[1]
    positions = torch.arange(steps, dtype=motion.dtype, device=motion.device)
    absolute_minutes = positions * float(sampling_interval) / 60.0
    since_calib_minutes = (positions - float(calibration_steps)) * float(sampling_interval) / 60.0
    time_features = torch.stack([absolute_minutes, since_calib_minutes], dim=-1).unsqueeze(0)
    return torch.cat([motion, time_features.expand(motion.shape[0], -1, -1)], dim=-1)


def _future_motion_energy_targets(head: torch.Tensor, lengths: torch.Tensor, horizons: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    motion_norm_sq = torch.linalg.vector_norm(head, dim=-1).square()
    cumsum = torch.cat([motion_norm_sq.new_zeros((head.shape[0], 1)), motion_norm_sq.cumsum(dim=1)], dim=1)
    positions = torch.arange(head.shape[1], device=head.device)
    targets = []
    masks = []
    for horizon in horizons:
        h = int(horizon)
        end = (positions + h + 1).clamp(max=head.shape[1])
        start = positions + 1
        denom = (end - start).to(head.dtype).clamp_min(1.0)
        energy = ((cumsum.index_select(1, end) - cumsum.index_select(1, start)) / denom.view(1, -1)).sqrt()
        valid = (positions.view(1, -1) + h) < lengths.to(head.device).view(-1, 1)
        targets.append(energy)
        masks.append(valid)
    return torch.stack(targets, dim=-1), torch.stack(masks, dim=-1)


def _run_epoch(
    model: MotionEnergyPretrainer,
    loader: DataLoader,
    device: torch.device,
    horizons: Sequence[int],
    motion_feature_mode: str,
    calibration_steps: int,
    sampling_interval: float,
    optimizer: torch.optim.Optimizer | None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_points = 0
    for batch in loader:
        head = batch["head"].to(device)
        lengths = batch["lengths"].to(device)
        features = append_motion_features(head, motion_feature_mode)
        features = _append_time_features(features, calibration_steps, sampling_interval)
        targets, mask = _future_motion_energy_targets(head, lengths, horizons)
        pred = model(features)
        finite = mask.to(device) & torch.isfinite(targets) & torch.isfinite(pred)
        if not finite.any():
            continue
        loss = ((pred - targets.to(device)).square() * finite.to(pred.dtype)).sum() / finite.to(pred.dtype).sum().clamp_min(1.0)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        points = int(finite.sum().detach().cpu())
        total_loss += float(loss.detach().cpu()) * points
        total_points += points
    return {"loss": total_loss / max(total_points, 1), "points": total_points}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain online-current motion encoder with motion-only future energy targets.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--out_dir", default="runs/online_fms_current_tracking_0509_remaining/motion_pretrain")
    parser.add_argument("--run_name", default="motion_energy_pretrain")
    parser.add_argument("--motion_feature_mode", default="causal_dynamics_v1")
    parser.add_argument("--future_energy_seconds", nargs="+", type=float, default=[5.0, 15.0])
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--kernel_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--deep_tcn_dilations", nargs="+", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    normalize_time_config(config)
    config.setdefault("model", {})
    config.setdefault("training", {})
    if args.motion_feature_mode is not None:
        config["model"]["motion_feature_mode"] = args.motion_feature_mode
    saved_split = load_json(args.split_file) if args.split_file else None
    prepared = prepare_data(args.data_dir, config, saved_split=saved_split)
    train_sessions = prepared["splits"]["train"]
    val_sessions = prepared["splits"].get("val", [])
    if not val_sessions:
        raise RuntimeError("Motion pretraining requires a validation split.")

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sampling_interval = float(prepared["data_info"]["sampling_interval"])
    calibration_steps = int(prepared["data_info"]["calibration_steps"])
    horizons = [seconds_to_steps(v, sampling_interval, name="future_energy_seconds") for v in args.future_energy_seconds]
    batch_size = int(args.batch_size or config["training"].get("batch_size", 48))
    loaders = {
        "train": DataLoader(DenseFMSSessionDataset(train_sessions), batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_sessions),
        "val": DataLoader(DenseFMSSessionDataset(val_sessions), batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_sessions),
    }

    hidden_dim = int(args.hidden_dim or config["model"].get("hidden_dim", config["model"].get("d_model", 192)))
    kernel_size = int(args.kernel_size or config["model"].get("kernel_size", 3))
    dropout = float(args.dropout if args.dropout is not None else config["model"].get("dropout", 0.1))
    dilations = [int(v) for v in (args.deep_tcn_dilations or config["model"].get("deep_tcn_dilations", [1, 2, 4, 8, 16]))]
    probe = append_motion_features(torch.zeros(1, 4, int(config["model"].get("head_dim", 6))), args.motion_feature_mode)
    input_dim = int(probe.shape[-1]) + 2
    model = MotionEnergyPretrainer(input_dim, hidden_dim, horizons, dilations, kernel_size, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)

    out_dir = ensure_dir(Path(args.out_dir) / args.run_name)
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    start_time = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_epoch(model, loaders["train"], device, horizons, args.motion_feature_mode, calibration_steps, sampling_interval, optimizer)
        with torch.no_grad():
            val_metrics = _run_epoch(model, loaders["val"], device, horizons, args.motion_feature_mode, calibration_steps, sampling_interval, None)
        row = {"epoch": epoch, "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"], "train_points": train_metrics["points"], "val_points": val_metrics["points"]}
        history.append(row)
        print(f"epoch {epoch:03d} train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f}")
        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "encoder_state_dict": model.encoder.state_dict(),
                    "config": {
                        "input_dim": input_dim,
                        "hidden_dim": hidden_dim,
                        "deep_tcn_dilations": dilations,
                        "kernel_size": kernel_size,
                        "dropout": dropout,
                        "motion_feature_mode": args.motion_feature_mode,
                        "stream_time_features": True,
                        "future_energy_seconds": [float(v) for v in args.future_energy_seconds],
                        "future_energy_steps": horizons,
                    },
                    "metrics": {"best_val_loss": best_val, "best_epoch": best_epoch},
                },
                out_dir / "best_motion_encoder.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                break
    save_json(
        out_dir / "metrics.json",
        {
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "elapsed_seconds": time.time() - start_time,
            "history": history,
            "checkpoint": str(out_dir / "best_motion_encoder.pt"),
        },
    )
    print(f"Saved motion pretraining checkpoint to {out_dir / 'best_motion_encoder.pt'}")


if __name__ == "__main__":
    main()
