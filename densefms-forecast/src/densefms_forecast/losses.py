"""Losses for future FMS sequence prediction."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from .utils import smooth_l1_masked


class FutureSequenceLoss(nn.Module):
    """SmoothL1 level loss with an optional raw adjacent-trend loss.

    The trend term compares first differences of the raw ordered prediction and
    target sequences. It only uses adjacent positions where both sequence
    positions are valid, so it never crosses session or padding boundaries.
    """

    VALID_MODES = {"level_only", "level_trend_raw", "level_plus_trend"}
    VALID_LOSSES = {"smooth_l1", "mse", "l1", "mae"}

    def __init__(
        self,
        mode: str = "level_only",
        trend_weight: float = 0.1,
        reduction: str = "mean",
        loss_type: str = "smooth_l1",
        horizon_weights: Optional[Sequence[float]] = None,
        change_weight: float = 0.0,
        high_target_weight: float = 0.0,
        high_target_threshold: float = 0.5,
        low_target_weight: float = 0.0,
        low_target_threshold: float = 0.15,
    ):
        super().__init__()
        if mode == "level_plus_trend":
            mode = "level_trend_raw"
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown loss mode '{mode}'. Expected one of {sorted(self.VALID_MODES)}.")
        if loss_type not in self.VALID_LOSSES:
            raise ValueError(f"Unknown loss_type '{loss_type}'. Expected one of {sorted(self.VALID_LOSSES)}.")
        if reduction != "mean":
            raise ValueError("FutureSequenceLoss currently supports reduction='mean' only.")
        self.mode = mode
        self.trend_weight = float(trend_weight)
        self.reduction = reduction
        self.loss_type = loss_type
        self.horizon_weights = [float(v) for v in horizon_weights] if horizon_weights else None
        self.change_weight = float(change_weight)
        self.high_target_weight = float(high_target_weight)
        self.high_target_threshold = float(high_target_threshold)
        self.low_target_weight = float(low_target_weight)
        self.low_target_threshold = float(low_target_threshold)

    def _element_weights(self, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target_safe = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        weights = torch.ones_like(target)
        if self.horizon_weights is not None and target.ndim == 3:
            if len(self.horizon_weights) != target.shape[-1]:
                raise ValueError(
                    f"horizon_weights length {len(self.horizon_weights)} does not match output horizons {target.shape[-1]}"
                )
            h = target.new_tensor(self.horizon_weights).view(1, 1, -1)
            weights = weights * h
        if self.change_weight > 0 and target.shape[1] >= 2:
            change = torch.zeros_like(target)
            change[:, 1:] = torch.abs(target_safe[:, 1:] - target_safe[:, :-1])
            weights = weights * (1.0 + self.change_weight * change.clamp_min(0.0))
        if self.high_target_weight > 0:
            high = (target_safe >= self.high_target_threshold).to(target.dtype)
            weights = weights * (1.0 + self.high_target_weight * high)
        if self.low_target_weight > 0:
            low = (target_safe <= self.low_target_threshold).to(target.dtype)
            weights = weights * (1.0 + self.low_target_weight * low)
        return weights * mask.to(target.dtype)

    def _masked_level_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "smooth_l1":
            valid = mask.bool() & torch.isfinite(pred) & torch.isfinite(target)
            safe_pred = torch.where(valid, pred, torch.zeros_like(pred))
            safe_target = torch.where(valid, target, torch.zeros_like(target))
            loss = torch.nn.functional.smooth_l1_loss(safe_pred, safe_target, reduction="none")
            weights = self._element_weights(target, valid)
            denom = weights.sum().clamp_min(1.0)
            return (loss * weights).sum() / denom
        valid = mask.bool() & torch.isfinite(pred) & torch.isfinite(target)
        err = torch.where(valid, pred - target, torch.zeros_like(pred))
        weights = self._element_weights(target, valid)
        denom = weights.sum().clamp_min(1.0)
        if self.loss_type in {"l1", "mae"}:
            return (err.abs() * weights).sum() / denom
        return (err.square() * weights).sum() / denom

    def forward(
        self,
        pred_future_seq: torch.Tensor,
        true_future_seq: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if pred_future_seq.ndim == 3 and pred_future_seq.shape[-1] == 1:
            pred_future_seq = pred_future_seq.squeeze(-1)
        if true_future_seq.ndim == 3 and true_future_seq.shape[-1] == 1:
            true_future_seq = true_future_seq.squeeze(-1)
        if pred_future_seq.shape != true_future_seq.shape:
            raise ValueError(f"Prediction/target shapes differ: {pred_future_seq.shape} vs {true_future_seq.shape}")
        if valid_mask.shape != pred_future_seq.shape:
            raise ValueError(f"Mask shape {valid_mask.shape} does not match sequence shape {pred_future_seq.shape}")

        valid_mask = valid_mask.bool() & torch.isfinite(pred_future_seq) & torch.isfinite(true_future_seq)
        loss_level = self._masked_level_loss(pred_future_seq, true_future_seq, valid_mask)
        zero = pred_future_seq.new_tensor(0.0)
        loss_trend = zero
        trend_points = 0

        if self.mode == "level_trend_raw" and pred_future_seq.shape[1] >= 2:
            diff_pred = pred_future_seq[:, 1:] - pred_future_seq[:, :-1]
            diff_true = true_future_seq[:, 1:] - true_future_seq[:, :-1]
            diff_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
            trend_points = int(diff_mask.sum().detach().cpu())
            loss_trend = self._masked_level_loss(diff_pred, diff_true, diff_mask)

        loss_total = loss_level + self.trend_weight * loss_trend
        parts = {
            "loss_total": float(loss_total.detach().cpu()),
            "loss_level": float(loss_level.detach().cpu()),
            "loss_trend": float(loss_trend.detach().cpu()),
            "trend_weight": self.trend_weight,
            "change_weight": self.change_weight,
            "high_target_weight": self.high_target_weight,
            "low_target_weight": self.low_target_weight,
            "valid_points": int(valid_mask.sum().detach().cpu()),
            "trend_points": trend_points,
        }
        return loss_total, parts
