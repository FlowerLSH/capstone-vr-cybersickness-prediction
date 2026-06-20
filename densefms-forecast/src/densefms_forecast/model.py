"""PyTorch models for calibration-conditioned online future FMS forecasting."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .online_current.heads import (
    _make_mlp_head,
    compute_current_head_outputs,
    current_head_risk_input_dim,
    make_calibration_prior_heads,
    make_current_regression_head,
    make_dual_delta_heads,
    make_ordinal_head,
    make_person_prior_heads,
    make_range_scaled_delta_heads,
    make_regime_gated_heads,
    make_residual_update_heads,
    make_state_space_delta_heads,
    make_trajectory_decoder_heads,
    make_zero_anchor_mixture_heads,
    normalize_current_head_mode,
    normalize_ordinal_head_mode,
)


FMS_CONTEXT_MODES = {"none", "start_only", "calibration_history", "sparse_anchor"}


def normalize_fms_context_mode(mode: str | None) -> str:
    mode = str(mode or "calibration_history").strip().lower()
    if mode not in FMS_CONTEXT_MODES:
        raise ValueError(f"fms_context_mode must be one of {sorted(FMS_CONTEXT_MODES)}, got {mode!r}")
    return mode


def calibration_context_fms(fms: torch.Tensor, calibration_steps: int, fms_context_mode: str) -> torch.Tensor:
    """Return the FMS sequence visible to the calibration encoder.

    ``start_only`` means the per-prediction recent-window start FMS anchor. The
    calibration encoder still receives the calibration FMS history, because that
    is part of the calibration phase rather than the rolling state anchor.
    """
    mode = normalize_fms_context_mode(fms_context_mode)
    if fms.ndim != 2:
        raise ValueError(f"FMS input must be [B,T] or [B,C], got {fms.shape}")
    if mode == "none":
        return fms.new_zeros((fms.shape[0], int(calibration_steps)))
    if fms.shape[1] < 1:
        raise ValueError("FMS input must contain at least one value.")
    if fms.shape[1] < int(calibration_steps):
        raise ValueError(f"{mode} needs {calibration_steps} calibration FMS steps, got {fms.shape[1]}.")
    return fms[:, : int(calibration_steps)]


def _uses_window_start_fms(fms_context_mode: str) -> bool:
    return normalize_fms_context_mode(fms_context_mode) == "start_only"


class CurrentAffineCalibrationHead(nn.Module):
    """Identity-initialized trainable scale/bias corrector for current-FMS outputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int],
        dropout: float,
        scale_range: float = 0.5,
        bias_range: float = 0.25,
    ):
        super().__init__()
        hidden = int(hidden_dim if hidden_dim is not None and int(hidden_dim) > 0 else input_dim)
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 2),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.scale_range = max(0.0, float(scale_range))
        self.bias_range = max(0.0, float(bias_range))

    def forward(self, current: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(features)
        scale = 1.0 + self.scale_range * torch.tanh(raw[..., 0])
        bias = self.bias_range * torch.tanh(raw[..., 1])
        corrected = torch.clamp(scale * current + bias, 0.0, 1.0)
        return corrected, scale, bias


class CurrentSessionAffineCalibrationHead(nn.Module):
    """Calibration-only session-level scale/bias corrector for current-FMS outputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int],
        dropout: float,
        scale_range: float = 0.25,
        bias_range: float = 0.15,
    ):
        super().__init__()
        hidden = int(hidden_dim if hidden_dim is not None and int(hidden_dim) > 0 else input_dim)
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 2),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.scale_range = max(0.0, float(scale_range))
        self.bias_range = max(0.0, float(bias_range))

    def forward(
        self,
        current: torch.Tensor,
        session_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if current.ndim != 2:
            raise ValueError(f"current must be [B,T], got {current.shape}.")
        if session_features.ndim != 2 or session_features.shape[0] != current.shape[0]:
            raise ValueError(f"session_features must be [B,D], got {session_features.shape}.")
        raw = self.net(session_features)
        scale = (1.0 + self.scale_range * torch.tanh(raw[:, 0])).view(-1, 1)
        bias = (self.bias_range * torch.tanh(raw[:, 1])).view(-1, 1)
        corrected = torch.clamp(scale * current + bias, 0.0, 1.0)
        return corrected, scale, bias


class CurrentLowFMSSuppressorHead(nn.Module):
    """Identity-initialized negative-only correction for low-FMS false positives."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int],
        dropout: float,
        delta_range: float = 0.25,
        gate_init_bias: float = -6.0,
    ):
        super().__init__()
        hidden = int(hidden_dim if hidden_dim is not None and int(hidden_dim) > 0 else input_dim)
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 2),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            final.bias.data[0] = float(gate_init_bias)
        self.delta_range = max(0.0, float(delta_range))

    def forward(
        self,
        current: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if current.ndim != 2:
            raise ValueError(f"current must be [B,T], got {current.shape}.")
        if features.ndim != 3 or features.shape[:2] != current.shape:
            raise ValueError(f"features must be [B,T,D] matching current, got {features.shape}.")
        raw = self.net(features)
        gate_logits = raw[..., 0]
        gate = torch.sigmoid(gate_logits)
        delta = self.delta_range * torch.sigmoid(raw[..., 1])
        correction = gate * delta
        corrected = torch.clamp(current - correction, 0.0, 1.0)
        return corrected, correction, gate, gate_logits


class CalibrationResidualAdapter(nn.Module):
    """Identity-initialized correction head driven by calibration probe residuals."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: Optional[int],
        dropout: float,
        mode: str = "mlp",
        delta_range: float = 0.15,
        decay_seconds: float = 120.0,
        gate_low_threshold: float = 8.0,
        gate_high_threshold: float = 10.0,
        gate_anchor_threshold: float = 10.0,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        feature_dim = int(feature_dim)
        if feature_dim <= 0:
            raise ValueError("calibration_residual_feature_dim must be positive when residual adapter is enabled.")
        mode = str(mode or "mlp").strip().lower()
        if mode not in {"mean_decay", "mlp", "mlp_decay", "mlp_high_gate", "mlp_decay_high_gate"}:
            raise ValueError(
                "calibration_residual_adapter_mode must be one of: "
                "mean_decay, mlp, mlp_decay, mlp_high_gate, mlp_decay_high_gate."
            )
        self.feature_dim = feature_dim
        self.mode = mode
        self.delta_range = max(0.0, float(delta_range))
        self.decay_seconds = max(float(decay_seconds), 1e-6)
        self.gate_low_threshold = max(0.0, min(20.0, float(gate_low_threshold))) / 20.0
        self.gate_high_threshold = max(0.0, min(20.0, float(gate_high_threshold))) / 20.0
        self.gate_anchor_threshold = max(0.0, min(20.0, float(gate_anchor_threshold))) / 20.0
        self.gate_temperature = max(float(gate_temperature), 1e-6) / 20.0
        self.mean_decay_gain = nn.Parameter(torch.ones(1)) if self.mode == "mean_decay" else None
        input_dim = feature_dim + 4
        hidden = int(hidden_dim if hidden_dim is not None and int(hidden_dim) > 0 else max(16, input_dim * 2))
        self.net = (
            nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, 1),
            )
            if self.mode in {"mlp", "mlp_decay", "mlp_high_gate", "mlp_decay_high_gate"}
            else None
        )
        if self.net is not None:
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def _high_regime_gate(self, current: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        pred_gate = torch.sigmoid((current - self.gate_high_threshold) / self.gate_temperature)
        anchor_gate = torch.sigmoid((anchor.expand_as(current) - self.gate_anchor_threshold) / self.gate_temperature)
        gate = torch.maximum(pred_gate, anchor_gate)
        low_mask = current >= self.gate_low_threshold
        return torch.where(low_mask, gate, torch.zeros_like(gate)).clamp(0.0, 1.0)

    def forward(
        self,
        current: torch.Tensor,
        residual_features: torch.Tensor,
        positions: torch.Tensor,
        calibration_steps: int,
        sampling_interval: float,
        anchor_fms: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if current.ndim != 2:
            raise ValueError(f"current must be [B,T], got {current.shape}.")
        if residual_features.ndim != 2 or residual_features.shape[0] != current.shape[0]:
            raise ValueError(f"residual_features must be [B,D], got {residual_features.shape}.")
        if residual_features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} residual features, got {residual_features.shape[1]}.")
        bsz, pred_steps = current.shape
        if pred_steps == 0:
            return current, current.new_zeros(current.shape), current.new_zeros(current.shape)
        if positions.ndim != 1 or int(positions.numel()) != pred_steps:
            raise ValueError(f"positions must be [T] matching current, got {positions.shape}.")
        dtype = current.dtype
        device = current.device
        residual_features = residual_features.to(device=device, dtype=dtype)
        anchor = anchor_fms.to(device=device, dtype=dtype)
        if anchor.ndim == 1:
            anchor = anchor.view(bsz, 1)
        if anchor.shape != (bsz, 1):
            raise ValueError(f"anchor_fms must be [B] or [B,1], got {anchor_fms.shape}.")
        positions_f = positions.to(device=device, dtype=dtype)
        absolute_time = positions_f.view(1, -1).expand(bsz, -1) * float(sampling_interval)
        since_calib = (positions_f - float(calibration_steps)).clamp_min(0.0).view(1, -1).expand(bsz, -1) * float(sampling_interval)
        decay = torch.exp(-since_calib / self.decay_seconds)
        if self.mode == "mean_decay":
            residual_mean = residual_features[:, 0].view(bsz, 1)
            assert self.mean_decay_gain is not None
            raw_delta = self.mean_decay_gain.to(dtype=dtype, device=device) * residual_mean * decay
            correction = raw_delta.clamp(min=-self.delta_range, max=self.delta_range)
        else:
            expanded = residual_features.unsqueeze(1).expand(-1, pred_steps, -1)
            scalar_context = torch.stack(
                [
                    current,
                    anchor.expand(-1, pred_steps),
                    absolute_time / 210.0,
                    since_calib / 210.0,
                ],
                dim=-1,
            )
            assert self.net is not None
            raw_delta = self.net(torch.cat([expanded, scalar_context], dim=-1)).squeeze(-1)
            correction = self.delta_range * torch.tanh(raw_delta)
            if self.mode in {"mlp_decay", "mlp_decay_high_gate"}:
                correction = correction * decay
        if self.mode in {"mlp_high_gate", "mlp_decay_high_gate"}:
            gate = self._high_regime_gate(current, anchor)
            correction = correction * gate
        else:
            gate = torch.ones_like(current)
        corrected = torch.clamp(current + correction, 0.0, 1.0)
        return corrected, correction, gate


class CalibrationSummaryFeatureFusion(nn.Module):
    """Identity-initialized early fusion driven by calibration-only summary features."""

    def __init__(
        self,
        feature_dim: int,
        fused_dim: int,
        hidden_dim: Optional[int],
        dropout: float,
        mode: str = "additive_gated",
        strength: float = 1.0,
    ):
        super().__init__()
        feature_dim = int(feature_dim)
        fused_dim = int(fused_dim)
        if feature_dim <= 0:
            raise ValueError("calibration_summary_fusion_feature_dim must be positive when summary fusion is enabled.")
        if fused_dim <= 0:
            raise ValueError("fused_dim must be positive for calibration summary fusion.")
        mode = str(mode or "additive_gated").strip().lower()
        if mode not in {"additive_gated", "film"}:
            raise ValueError("calibration_summary_fusion_mode must be one of: additive_gated, film.")
        self.feature_dim = feature_dim
        self.fused_dim = fused_dim
        self.mode = mode
        self.strength = max(0.0, float(strength))
        hidden = int(hidden_dim if hidden_dim is not None and int(hidden_dim) > 0 else max(fused_dim, feature_dim))
        self.summary_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(float(dropout)),
        )
        if self.mode == "additive_gated":
            self.delta = nn.Linear(hidden, fused_dim)
            self.gate = nn.Sequential(
                nn.Linear(hidden + fused_dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)
            final = self.gate[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
            self.film = None
        else:
            self.delta = None
            self.gate = None
            self.film = nn.Linear(hidden, fused_dim * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, fused: torch.Tensor, summary_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if fused.ndim != 3:
            raise ValueError(f"fused must be [B,T,D], got {fused.shape}.")
        if summary_features.ndim != 2 or summary_features.shape[0] != fused.shape[0]:
            raise ValueError(f"summary_features must be [B,F], got {summary_features.shape}.")
        if summary_features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} summary features, got {summary_features.shape[1]}.")
        bsz, pred_steps, _ = fused.shape
        if pred_steps == 0:
            return fused, fused.new_zeros(fused.shape), fused.new_zeros((bsz, pred_steps))
        encoded = self.summary_encoder(summary_features.to(device=fused.device, dtype=fused.dtype))
        expanded = encoded.unsqueeze(1).expand(-1, pred_steps, -1)
        if self.mode == "additive_gated":
            assert self.delta is not None and self.gate is not None
            delta = torch.tanh(self.delta(expanded))
            gate = torch.sigmoid(self.gate(torch.cat([fused, expanded], dim=-1))).squeeze(-1)
            fused_out = fused + self.strength * gate.unsqueeze(-1) * delta
            return fused_out, delta, gate
        assert self.film is not None
        gamma, beta = self.film(encoded).chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(1)
        beta = torch.tanh(beta).unsqueeze(1)
        delta = fused * (self.strength * gamma) + self.strength * beta
        fused_out = fused + delta
        gate = torch.ones((bsz, pred_steps), device=fused.device, dtype=fused.dtype)
        return fused_out, delta, gate


class CurrentBinnedAffineCalibrationHead(nn.Module):
    """Identity-initialized condition-binned affine corrector for current-FMS outputs."""

    def __init__(
        self,
        anchor_bins: Sequence[float] = (5.0, 10.0),
        pred_bins: Sequence[float] = (5.0, 10.0),
        time_bins: Sequence[float] = (160.0,),
        scale_range: float = 1.5,
        bias_range: float = 0.5,
    ):
        super().__init__()
        anchor = torch.tensor([float(v) / 20.0 for v in anchor_bins], dtype=torch.float32)
        pred = torch.tensor([float(v) / 20.0 for v in pred_bins], dtype=torch.float32)
        time = torch.tensor([float(v) for v in time_bins], dtype=torch.float32)
        if anchor.numel() > 1 and bool(torch.any(anchor[:-1] >= anchor[1:])):
            raise ValueError("current_binned_affine_anchor_bins must be strictly increasing.")
        if pred.numel() > 1 and bool(torch.any(pred[:-1] >= pred[1:])):
            raise ValueError("current_binned_affine_pred_bins must be strictly increasing.")
        if time.numel() > 1 and bool(torch.any(time[:-1] >= time[1:])):
            raise ValueError("current_binned_affine_time_bins must be strictly increasing.")
        self.register_buffer("anchor_bins_norm", anchor)
        self.register_buffer("pred_bins_norm", pred)
        self.register_buffer("time_bins_seconds", time)
        self.anchor_bin_count = int(anchor.numel()) + 1
        self.pred_bin_count = int(pred.numel()) + 1
        self.time_bin_count = int(time.numel()) + 1
        total_bins = self.anchor_bin_count * self.pred_bin_count * self.time_bin_count
        self.scale_raw = nn.Parameter(torch.zeros(total_bins))
        self.bias_raw = nn.Parameter(torch.zeros(total_bins))
        self.scale_range = max(0.0, float(scale_range))
        self.bias_range = max(0.0, float(bias_range))

    def forward(
        self,
        current: torch.Tensor,
        anchor_fms: torch.Tensor,
        current_time_seconds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if current.ndim != 2:
            raise ValueError(f"current must be [B,T], got {current.shape}.")
        if anchor_fms.ndim != 1 or anchor_fms.shape[0] != current.shape[0]:
            raise ValueError(f"anchor_fms must be [B], got {anchor_fms.shape}.")
        time = current_time_seconds.to(device=current.device, dtype=current.dtype)
        if time.ndim == 1:
            time = time.view(1, -1).expand_as(current)
        elif time.shape != current.shape:
            raise ValueError(f"current_time_seconds must be [T] or [B,T], got {current_time_seconds.shape}.")

        anchor_bins = self.anchor_bins_norm.to(device=current.device, dtype=current.dtype)
        pred_bins = self.pred_bins_norm.to(device=current.device, dtype=current.dtype)
        time_bins = self.time_bins_seconds.to(device=current.device, dtype=current.dtype)
        anchor_ids = torch.bucketize(anchor_fms.to(dtype=current.dtype).contiguous(), anchor_bins)
        pred_ids = torch.bucketize(current.detach().contiguous(), pred_bins)
        time_ids = torch.bucketize(time.contiguous(), time_bins)
        bin_ids = (anchor_ids.view(-1, 1) * self.pred_bin_count + pred_ids) * self.time_bin_count + time_ids

        scale = 1.0 + self.scale_range * torch.tanh(self.scale_raw[bin_ids])
        bias = self.bias_range * torch.tanh(self.bias_raw[bin_ids])
        corrected = torch.clamp(scale * current + bias, 0.0, 1.0)
        return corrected, scale, bias, bin_ids


class FeatureDistributionSmoothing(nn.Module):
    """Label-bin feature calibration for imbalanced regression training."""

    def __init__(
        self,
        feature_dim: int,
        num_bins: int = 21,
        min_value: float = 0.0,
        max_value: float = 20.0,
        bin_size: float = 1.0,
        kernel: str = "gaussian",
        kernel_size: int = 5,
        sigma: float = 2.0,
        momentum: float = 0.9,
        blend: float = 1.0,
    ):
        super().__init__()
        feature_dim = int(feature_dim)
        if feature_dim <= 0:
            raise ValueError("FDS feature_dim must be positive.")
        bin_size = float(bin_size)
        min_value = float(min_value)
        max_value = float(max_value)
        if bin_size <= 0 or max_value <= min_value:
            raise ValueError("FDS requires bin_size > 0 and max_value > min_value.")
        inferred_bins = int(round((max_value - min_value) / bin_size)) + 1
        num_bins = int(num_bins or inferred_bins)
        if num_bins <= 1:
            raise ValueError("FDS num_bins must be greater than 1.")
        if num_bins != inferred_bins:
            num_bins = inferred_bins
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("FDS kernel_size must be a positive odd integer.")
        kernel_name = str(kernel or "gaussian").lower()
        if kernel_name not in {"gaussian", "triangular", "laplace", "laplacian"}:
            raise ValueError("FDS kernel must be one of: gaussian, triangular, laplace.")

        self.feature_dim = feature_dim
        self.num_bins = num_bins
        self.min_value = min_value
        self.max_value = max_value
        self.bin_size = bin_size
        self.kernel_name = "laplace" if kernel_name == "laplacian" else kernel_name
        self.kernel_size = kernel_size
        self.sigma = float(sigma)
        self.momentum = max(0.0, min(0.999, float(momentum)))
        self.blend = max(0.0, min(1.0, float(blend)))

        self.register_buffer("running_mean", torch.zeros(num_bins, feature_dim))
        self.register_buffer("running_var", torch.ones(num_bins, feature_dim))
        self.register_buffer("running_count", torch.zeros(num_bins))
        self.register_buffer("smoothed_mean", torch.zeros(num_bins, feature_dim))
        self.register_buffer("smoothed_var", torch.ones(num_bins, feature_dim))
        self.register_buffer("smoothed_count", torch.zeros(num_bins))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("_epoch_sum", torch.zeros(num_bins, feature_dim), persistent=False)
        self.register_buffer("_epoch_sum_sq", torch.zeros(num_bins, feature_dim), persistent=False)
        self.register_buffer("_epoch_count", torch.zeros(num_bins), persistent=False)
        self.register_buffer("_kernel_window", self._make_kernel_window(), persistent=False)

    def _make_kernel_window(self) -> torch.Tensor:
        half = int(self.kernel_size) // 2
        offsets = torch.arange(-half, half + 1, dtype=torch.float32)
        if self.kernel_name == "gaussian":
            scale = max(float(self.sigma), 1e-8)
            weights = torch.exp(-(offsets.square()) / (2.0 * scale * scale))
        elif self.kernel_name == "triangular":
            weights = (half + 1.0) - offsets.abs()
        else:
            scale = max(float(self.sigma), 1e-8)
            weights = torch.exp(-offsets.abs() / scale)
        return weights / weights.sum().clamp_min(1e-12)

    def reset_epoch_stats(self) -> None:
        self._epoch_sum.zero_()
        self._epoch_sum_sq.zero_()
        self._epoch_count.zero_()

    def label_to_bin(self, labels_raw: torch.Tensor) -> torch.Tensor:
        idx = torch.floor((labels_raw - self.min_value) / self.bin_size + 1e-6).long()
        return idx.clamp(0, self.num_bins - 1)

    @torch.no_grad()
    def update_epoch_stats(self, features: torch.Tensor, labels_raw: torch.Tensor, mask: torch.Tensor) -> int:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"FDS features must be [B,P,{self.feature_dim}], got {features.shape}.")
        if labels_raw.shape != features.shape[:2] or mask.shape != features.shape[:2]:
            raise ValueError("FDS labels and mask must match feature [B,P] dimensions.")
        flat_features = features.detach().reshape(-1, self.feature_dim)
        flat_labels = labels_raw.detach().reshape(-1)
        flat_mask = mask.detach().reshape(-1).bool()
        finite = flat_mask & torch.isfinite(flat_labels) & torch.isfinite(flat_features).all(dim=-1)
        if not bool(finite.any()):
            return 0
        valid_features = flat_features[finite].to(dtype=self._epoch_sum.dtype)
        valid_bins = self.label_to_bin(flat_labels[finite]).to(self._epoch_count.device)
        ones = torch.ones(valid_bins.shape[0], dtype=self._epoch_count.dtype, device=self._epoch_count.device)
        self._epoch_count.index_add_(0, valid_bins, ones)
        self._epoch_sum.index_add_(0, valid_bins, valid_features.to(self._epoch_sum.device))
        self._epoch_sum_sq.index_add_(0, valid_bins, valid_features.square().to(self._epoch_sum_sq.device))
        return int(valid_bins.numel())

    @torch.no_grad()
    def commit_epoch_stats(self) -> Dict[str, float]:
        observed = self._epoch_count > 0
        points = int(self._epoch_count.sum().detach().cpu())
        bins_observed = int(observed.sum().detach().cpu())
        if bins_observed == 0:
            return {
                "fds_epoch_points": 0,
                "fds_bins_observed": 0,
                "fds_running_bins": int((self.running_count > 0).sum().detach().cpu()),
                "fds_initialized": bool(self.initialized.detach().cpu().item()),
            }

        count = self._epoch_count.clamp_min(1.0).unsqueeze(-1)
        epoch_mean = self._epoch_sum / count
        epoch_var = (self._epoch_sum_sq / count - epoch_mean.square()).clamp_min(1e-6)
        if not bool(self.initialized.detach().cpu().item()):
            self.running_mean[observed] = epoch_mean[observed]
            self.running_var[observed] = epoch_var[observed]
            self.running_count[observed] = self._epoch_count[observed]
            self.initialized.fill_(True)
        else:
            m = float(self.momentum)
            self.running_mean[observed] = m * self.running_mean[observed] + (1.0 - m) * epoch_mean[observed]
            self.running_var[observed] = m * self.running_var[observed] + (1.0 - m) * epoch_var[observed]
            self.running_count[observed] = m * self.running_count[observed] + (1.0 - m) * self._epoch_count[observed]
        self._smooth_running_stats()
        return {
            "fds_epoch_points": points,
            "fds_bins_observed": bins_observed,
            "fds_running_bins": int((self.running_count > 0).sum().detach().cpu()),
            "fds_initialized": bool(self.initialized.detach().cpu().item()),
        }

    @torch.no_grad()
    def _smooth_running_stats(self) -> None:
        valid = self.running_count > 0
        if not bool(valid.any()):
            return
        half = int(self.kernel_size) // 2
        kernel = self._kernel_window.to(device=self.running_mean.device, dtype=self.running_mean.dtype)
        smooth_mean = self.running_mean.clone()
        smooth_var = self.running_var.clone()
        smooth_count = torch.zeros_like(self.running_count)
        for bin_idx in range(self.num_bins):
            left = max(0, bin_idx - half)
            right = min(self.num_bins, bin_idx + half + 1)
            k_left = half - (bin_idx - left)
            k_right = k_left + (right - left)
            weights = kernel[k_left:k_right]
            neighbor_valid = valid[left:right]
            if not bool(neighbor_valid.any()):
                continue
            weights = weights * neighbor_valid.to(weights.dtype)
            weights = weights / weights.sum().clamp_min(1e-12)
            smooth_mean[bin_idx] = (self.running_mean[left:right] * weights.unsqueeze(-1)).sum(dim=0)
            smooth_var[bin_idx] = (self.running_var[left:right] * weights.unsqueeze(-1)).sum(dim=0).clamp_min(1e-6)
            smooth_count[bin_idx] = (self.running_count[left:right] * weights).sum()
        self.smoothed_mean.copy_(smooth_mean)
        self.smoothed_var.copy_(smooth_var)
        self.smoothed_count.copy_(smooth_count)

    def apply(self, features: torch.Tensor, labels_raw: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if self.blend <= 0.0 or not bool(self.initialized.detach().cpu().item()):
            return features, 0
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"FDS features must be [B,P,{self.feature_dim}], got {features.shape}.")
        if labels_raw.shape != features.shape[:2] or mask.shape != features.shape[:2]:
            raise ValueError("FDS labels and mask must match feature [B,P] dimensions.")
        flat_features = features.reshape(-1, self.feature_dim)
        flat_labels = labels_raw.reshape(-1)
        flat_mask = mask.reshape(-1).bool()
        bins = self.label_to_bin(flat_labels)
        stats_ready = (self.running_count[bins] > 0) & (self.smoothed_count[bins] > 0)
        finite = flat_mask & torch.isfinite(flat_labels) & torch.isfinite(flat_features).all(dim=-1) & stats_ready
        if not bool(finite.any()):
            return features, 0
        mean = self.running_mean[bins].to(device=features.device, dtype=features.dtype)
        var = self.running_var[bins].to(device=features.device, dtype=features.dtype).clamp_min(1e-6)
        smooth_mean = self.smoothed_mean[bins].to(device=features.device, dtype=features.dtype)
        smooth_var = self.smoothed_var[bins].to(device=features.device, dtype=features.dtype).clamp_min(1e-6)
        calibrated = (flat_features - mean) / var.sqrt() * smooth_var.sqrt() + smooth_mean
        if self.blend < 1.0:
            calibrated = flat_features + float(self.blend) * (calibrated - flat_features)
        out = flat_features.clone()
        out[finite] = calibrated[finite]
        return out.view_as(features), int(finite.sum().detach().cpu())


class CalibrationEncoder(nn.Module):
    """Encode first C steps of head motion, FMS, and delta-FMS."""

    def __init__(self, head_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(head_dim + 2, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )

    def forward(self, head_calib: torch.Tensor, y_calib: torch.Tensor) -> torch.Tensor:
        assert head_calib.ndim == 3, f"head_calib must be [B,C,H], got {head_calib.shape}"
        if y_calib.ndim == 2:
            y_calib = y_calib.unsqueeze(-1)
        assert y_calib.shape[:2] == head_calib.shape[:2], "calibration FMS must align with calibration head"
        delta = torch.zeros_like(y_calib)
        delta[:, 1:] = y_calib[:, 1:] - y_calib[:, :-1]
        x = torch.cat([head_calib, y_calib, delta], dim=-1)
        x = self.input(x)
        out, (h_n, _) = self.lstm(x)
        scores = self.attn(out).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        pooled = torch.sum(out * alpha.unsqueeze(-1), dim=1)
        final = h_n[-1]
        return self.out(torch.cat([final, pooled], dim=-1))


class StateInitializer(nn.Module):
    def __init__(self, calib_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.h = nn.Linear(calib_dim, hidden_dim)
        self.c = nn.Linear(calib_dim, hidden_dim)

    def forward(self, z_calib: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tanh(self.h(z_calib)), torch.tanh(self.c(z_calib))


class StaticEncoder(nn.Module):
    """Encode session-level static covariates [age_z, gender one-hot]."""

    def __init__(self, static_dim: int = 4, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Dropout(dropout),
            nn.Linear(32, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        if static.ndim != 2:
            raise ValueError(f"static must be [B, static_dim], got {static.shape}")
        return self.net(static)


class ContextFusion(nn.Module):
    """Fuse calibration context with optional static user context."""

    def __init__(self, calib_dim: int = 128, static_hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(calib_dim + static_hidden_dim, calib_dim),
            nn.GELU(),
            nn.LayerNorm(calib_dim),
        )

    def forward(self, z_calib: torch.Tensor, z_static: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_calib, z_static], dim=-1))


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels: int = 64, dilation: int = 1, dropout: float = 0.1, kernel_size: int = 3):
        super().__init__()
        self.conv1 = CausalConv1d(channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = self.dropout(F.gelu(self.conv1(y)))
        y = self.dropout(F.gelu(self.conv2(y)))
        y = y.transpose(1, 2)
        return self.norm(residual + y)


class DeepTCNEncoder(nn.Module):
    """Causal dilated TCN encoder for online motion streams."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dilations = [int(d) for d in dilations]
        self.input = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
        )
        self.blocks = nn.Sequential(
            *[
                TCNBlock(
                    int(hidden_dim),
                    dilation=int(d),
                    dropout=float(dropout),
                    kernel_size=int(kernel_size),
                )
                for d in self.dilations
            ]
        )
        self.out = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(self.input(x)))


class RecentWindowEncoder(nn.Module):
    """Causal TCN over the recent 10-second head-motion window."""

    def __init__(self, head_dim: int = 6, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(head_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.blocks = nn.Sequential(
            TCNBlock(64, dilation=1, dropout=dropout),
            TCNBlock(64, dilation=2, dropout=dropout),
            TCNBlock(64, dilation=4, dropout=dropout),
        )
        self.out = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )

    def forward(self, recent_head: torch.Tensor) -> torch.Tensor:
        assert recent_head.ndim == 3, f"recent_head must be [B,W,H], got {recent_head.shape}"
        x = self.input(recent_head)
        x = self.blocks(x)
        return self.out(x[:, -1])


class RecentTransformerEncoder(nn.Module):
    """Causal self-attention encoder over a recent head-motion window."""

    def __init__(
        self,
        head_dim: int = 6,
        recent_steps: int = 20,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_dim: int = 128,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.recent_steps = int(recent_steps)
        self.input = nn.Sequential(
            nn.Linear(head_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.positional = nn.Parameter(torch.zeros(1, self.recent_steps, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.out = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )

    def forward(self, recent_head: torch.Tensor) -> torch.Tensor:
        assert recent_head.ndim == 3, f"recent_head must be [B,W,H], got {recent_head.shape}"
        steps = recent_head.shape[1]
        if steps > self.recent_steps:
            raise ValueError(f"recent window length {steps} exceeds configured recent_steps={self.recent_steps}")
        x = self.input(recent_head) + self.positional[:, :steps, :]
        # The window itself is causal and ends at current time t. This mask
        # additionally prevents earlier tokens from attending to later tokens,
        # which keeps the block safe if more attention layers are added later.
        causal_mask = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=recent_head.device), diagonal=1)
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return self.out(x[:, -1])


class ForecastHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 320,
        delta_max: float = 0.1,
        dropout: float = 0.2,
        use_legacy_multihead: bool = False,
    ):
        super().__init__()
        self.delta_max = float(delta_max)
        self.use_legacy_multihead = use_legacy_multihead
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.now = nn.Linear(64, 1)
        self.future_level = nn.Linear(64, 1)
        self.future_delta = nn.Linear(64, 1)
        self.gate = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.shared(x)
        y_level = torch.sigmoid(self.future_level(z))
        y_future = y_level
        out = {
            "future": y_future.squeeze(-1),
            "y_future_hat": y_future.squeeze(-1),
            "y_future_level": y_level.squeeze(-1),
        }
        if self.use_legacy_multihead:
            y_now = torch.sigmoid(self.now(z))
            delta = self.delta_max * torch.tanh(self.future_delta(z))
            y_delta = torch.clamp(y_now + delta, 0.0, 1.0)
            gate = torch.sigmoid(self.gate(z))
            y_future = gate * y_level + (1.0 - gate) * y_delta
            out.update(
                {
                    "future": y_future.squeeze(-1),
                    "y_future_hat": y_future.squeeze(-1),
                    "y_now_hat": y_now.squeeze(-1),
                    "y_future_delta": y_delta.squeeze(-1),
                    "gate": gate.squeeze(-1),
                }
            )
        return out


class COFFLSTM(nn.Module):
    """Calibration-conditioned Online Future FMS LSTM.

    During forward, post-calibration FMS is not accepted as an argument. The
    recurrent state is updated causally from head[t] only, and recent windows
    are built from head[t-window+1:t+1].
    """

    def __init__(
        self,
        head_dim: int = 6,
        calibration_steps: int = 60,
        horizon_steps: int = 10,
        recent_steps: int = 20,
        sampling_interval: float = 0.5,
        delta_max: float = 0.1,
        no_film: bool = False,
        no_recent_encoder: bool = False,
        use_legacy_multihead: bool = False,
        use_static: bool = False,
        static_dim: int = 4,
        static_hidden_dim: int = 64,
        static_dropout: float = 0.1,
        recent_encoder: str = "tcn",
        recent_attn_heads: int = 4,
        recent_attn_layers: int = 1,
        recent_attn_dropout: float = 0.1,
        fms_context_mode: str = "calibration_history",
    ):
        super().__init__()
        if recent_attn_layers != 1:
            raise ValueError("recent_attn_layers currently supports only 1 layer")
        recent_encoder = recent_encoder.lower()
        if recent_encoder not in {"tcn", "transformer"}:
            raise ValueError(f"recent_encoder must be 'tcn' or 'transformer', got {recent_encoder!r}")
        self.head_dim = head_dim
        self.calibration_steps = calibration_steps
        self.horizon_steps = horizon_steps
        self.recent_steps = recent_steps
        self.sampling_interval = float(sampling_interval)
        self.recent_encoder_type = recent_encoder
        self.no_film = no_film
        self.no_recent_encoder = no_recent_encoder
        self.use_legacy_multihead = use_legacy_multihead
        self.use_static = use_static
        self.static_dim = static_dim
        self.fms_context_mode = normalize_fms_context_mode(fms_context_mode)
        self.uses_window_start_fms = _uses_window_start_fms(self.fms_context_mode)
        self.requires_full_fms = self.uses_window_start_fms

        self.calibration_encoder = CalibrationEncoder(head_dim=head_dim)
        self.static_encoder = StaticEncoder(static_dim, static_hidden_dim, static_dropout) if use_static else None
        self.context_fusion = ContextFusion(128, static_hidden_dim) if use_static else None
        self.state_initializer = StateInitializer()
        self.head_projection = nn.Sequential(
            nn.Linear(head_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.film = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
        self.cond_norm = nn.LayerNorm(64)
        self.cell = nn.LSTMCell(input_size=64, hidden_size=128)
        if recent_encoder == "tcn":
            self.recent_encoder = RecentWindowEncoder(head_dim=head_dim)
        else:
            self.recent_encoder = RecentTransformerEncoder(
                head_dim=head_dim,
                recent_steps=recent_steps,
                num_heads=recent_attn_heads,
                dropout=recent_attn_dropout,
            )
        recent_dim = 0 if no_recent_encoder else 64
        window_start_dim = 64 if self.uses_window_start_fms else 0
        self.window_start_encoder = (
            nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.LayerNorm(64))
            if self.uses_window_start_fms
            else None
        )
        self.forecast_head = ForecastHead(
            128 + 128 + recent_dim + window_start_dim,
            delta_max=delta_max,
            use_legacy_multihead=use_legacy_multihead,
        )

    def encode_calibration(self, head: torch.Tensor, y_calib: torch.Tensor) -> torch.Tensor:
        assert head.shape[1] >= self.calibration_steps, "sequence shorter than calibration_steps"
        context_fms = calibration_context_fms(y_calib, self.calibration_steps, self.fms_context_mode)
        return self.calibration_encoder(head[:, : self.calibration_steps], context_fms)

    def make_context(self, z_calib: torch.Tensor, static: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.use_static:
            return z_calib, None
        if static is None:
            raise ValueError("COFFLSTM was created with use_static=True, but static tensor was not provided.")
        if static.ndim != 2 or static.shape[0] != z_calib.shape[0] or static.shape[1] != self.static_dim:
            raise ValueError(f"static must be [B,{self.static_dim}], got {static.shape}")
        assert self.static_encoder is not None and self.context_fusion is not None
        z_static = self.static_encoder(static)
        return self.context_fusion(z_calib, z_static), z_static

    def initial_state(self, z_calib: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.state_initializer(z_calib)

    def condition_head(self, head_t: torch.Tensor, z_calib: torch.Tensor) -> torch.Tensor:
        e = self.head_projection(head_t)
        if self.no_film:
            return e
        gamma_beta = self.film(z_calib)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return self.cond_norm((1.0 + 0.5 * torch.tanh(gamma)) * e + beta)

    def step(
        self,
        head_t: torch.Tensor,
        recent_head: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        z_calib: torch.Tensor,
        window_start_context: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        assert head_t.ndim == 2 and head_t.shape[-1] == self.head_dim, "head_t must be [B,6]"
        assert recent_head.shape[1:] == (self.recent_steps, self.head_dim), "recent window has wrong shape"
        e = self.condition_head(head_t, z_calib)
        h, c = self.cell(e, state)
        parts = [z_calib, h]
        if not self.no_recent_encoder:
            parts.append(self.recent_encoder(recent_head))
        if self.window_start_encoder is not None:
            if window_start_context is None:
                window_start_context = z_calib.new_zeros((z_calib.shape[0], 2))
            parts.append(self.window_start_encoder(window_start_context))
        out = self.forecast_head(torch.cat(parts, dim=-1))
        out["use_static"] = self.use_static
        return out, (h, c)

    def _recent_representations(self, head: torch.Tensor) -> torch.Tensor:
        bsz, steps, head_dim = head.shape
        # Causal windows for every current time t. Left padding handles short
        # histories such as calibration_steps < recent_steps without negative
        # indexing, and every window ends exactly at t.
        pad = head.new_zeros((bsz, self.recent_steps - 1, head_dim))
        padded = torch.cat([pad, head], dim=1)
        windows = padded.unfold(dimension=1, size=self.recent_steps, step=1)
        windows = windows.permute(0, 1, 3, 2).contiguous()
        flat = windows.view(-1, self.recent_steps, head_dim)
        recent = self.recent_encoder(flat)
        return recent.view(bsz, steps, -1)

    def forward(
        self,
        head: torch.Tensor,
        y_calib: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        static: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        assert head.ndim == 3, f"head must be [B,T,6], got {head.shape}"
        assert head.shape[-1] == self.head_dim, f"expected head_dim={self.head_dim}, got {head.shape[-1]}"
        bsz, steps, _ = head.shape
        device = head.device
        if lengths is None:
            lengths = torch.full((bsz,), steps, dtype=torch.long, device=device)
        else:
            lengths = lengths.to(device)
        assert y_calib.shape[0] == bsz and y_calib.shape[1] >= self.calibration_steps
        fms = y_calib.to(device)
        if self.requires_full_fms and fms.shape[1] < steps:
            raise ValueError(
                f"fms_context_mode={self.fms_context_mode!r} requires full FMS input through current time."
            )

        z_calib = self.encode_calibration(head, fms)
        z_context, z_static = self.make_context(z_calib, static)
        h, c = self.initial_state(z_context)
        recent_all = None if self.no_recent_encoder else self._recent_representations(head)

        max_forecast_t = int(torch.clamp(lengths.max() - self.horizon_steps, min=self.calibration_steps).item())
        pred_steps = max(0, max_forecast_t - self.calibration_steps)
        compact = {
            key: torch.zeros((bsz, pred_steps), dtype=head.dtype, device=device)
            for key in ("future", "y_future_hat", "y_future_level")
        }
        outputs = {
            key: torch.zeros((bsz, steps), dtype=head.dtype, device=device)
            for key in ("future", "y_future_hat", "y_future_level")
        }
        if self.use_legacy_multihead:
            for key in ("y_now_hat", "y_future_delta", "gate"):
                compact[key] = torch.zeros((bsz, pred_steps), dtype=head.dtype, device=device)
                outputs[key] = torch.zeros((bsz, steps), dtype=head.dtype, device=device)
        pred_positions = self.calibration_steps + torch.arange(pred_steps, device=device)
        mask = (pred_positions.unsqueeze(0) + self.horizon_steps) < lengths.unsqueeze(1)
        anchor_indices = torch.zeros((bsz, pred_steps), dtype=torch.long, device=device) if self.uses_window_start_fms else None
        anchor_fms_seq = torch.zeros((bsz, pred_steps), dtype=head.dtype, device=device) if self.uses_window_start_fms else None
        for t in range(self.calibration_steps, max_forecast_t):
            active = (lengths > t).float().unsqueeze(-1)
            e = self.condition_head(head[:, t], z_context)
            h_new, c_new = self.cell(e, (h, c))
            h = active * h_new + (1.0 - active) * h
            c = active * c_new + (1.0 - active) * c
            parts = [z_context, h]
            if not self.no_recent_encoder:
                parts.append(recent_all[:, t])
            if self.window_start_encoder is not None:
                window_start_idx = torch.full((1,), max(t - self.recent_steps + 1, 0), dtype=torch.long, device=device)
                anchor_fms, actual_anchor_idx = _gather_anchor_fms_safe(fms, window_start_idx, "window_start_fms")
                time_since = (torch.full_like(actual_anchor_idx, t) - actual_anchor_idx).to(head.dtype) * self.sampling_interval
                window_start_context = torch.stack([anchor_fms.squeeze(1), (time_since / 120.0).squeeze(1)], dim=-1)
                parts.append(self.window_start_encoder(window_start_context))
            pred = self.forecast_head(torch.cat(parts, dim=-1))
            seq_idx = t - self.calibration_steps
            if anchor_indices is not None and anchor_fms_seq is not None:
                anchor_indices[:, seq_idx] = actual_anchor_idx.squeeze(1)
                anchor_fms_seq[:, seq_idx] = anchor_fms.squeeze(1)
            for key, value in pred.items():
                outputs[key][:, t] = value
                compact[key][:, seq_idx] = value
        outputs["future"] = compact["future"]
        outputs["mask"] = mask
        outputs["prediction_start"] = torch.tensor(self.calibration_steps, device=device)
        outputs["use_static"] = self.use_static
        outputs["z_calib_norm"] = z_calib.norm(dim=-1)
        outputs["z_context_norm"] = z_context.norm(dim=-1)
        if anchor_indices is not None and anchor_fms_seq is not None:
            outputs["anchor_index"] = anchor_indices
            outputs["anchor_fms"] = anchor_fms_seq
        if z_static is not None:
            outputs["z_static_norm"] = z_static.norm(dim=-1)
        for key, value in compact.items():
            if key != "future":
                outputs[f"{key}_seq"] = value
        return outputs


class Recent10TCN(nn.Module):
    """Recent 10-second head-window baseline with no calibration or online state."""

    def __init__(self, head_dim: int = 6, calibration_steps: int = 60, horizon_steps: int = 10, recent_steps: int = 20):
        super().__init__()
        self.head_dim = head_dim
        self.calibration_steps = calibration_steps
        self.horizon_steps = horizon_steps
        self.recent_steps = recent_steps
        self.recent_encoder = RecentWindowEncoder(head_dim=head_dim)
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.now = nn.Linear(64, 1)
        self.future = nn.Linear(64, 1)

    def forward(
        self,
        head: torch.Tensor,
        y_calib: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        static: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del y_calib
        del static
        assert head.ndim == 3 and head.shape[-1] == self.head_dim
        bsz, steps, head_dim = head.shape
        device = head.device
        if lengths is None:
            lengths = torch.full((bsz,), steps, dtype=torch.long, device=device)
        pad = head.new_zeros((bsz, self.recent_steps - 1, head_dim))
        padded = torch.cat([pad, head], dim=1)
        windows = padded.unfold(dimension=1, size=self.recent_steps, step=1)
        windows = windows.permute(0, 1, 3, 2).contiguous().view(-1, self.recent_steps, head_dim)
        z = self.head(self.recent_encoder(windows)).view(bsz, steps, 64)
        now = torch.sigmoid(self.now(z)).squeeze(-1)
        fut = torch.sigmoid(self.future(z)).squeeze(-1)
        out: Dict[str, torch.Tensor] = {}
        out["y_now_hat"] = now
        out["y_future_hat"] = fut
        out["y_future_level"] = fut
        out["y_future_delta"] = fut
        out["gate"] = torch.ones_like(fut)
        max_forecast_t = int(torch.clamp(lengths.max() - self.horizon_steps, min=self.calibration_steps).item())
        pred_steps = max(0, max_forecast_t - self.calibration_steps)
        positions = self.calibration_steps + torch.arange(pred_steps, device=device)
        out["future"] = out["y_future_hat"].index_select(1, positions) if pred_steps else out["y_future_hat"][:, :0]
        out["mask"] = (positions.unsqueeze(0) + self.horizon_steps) < lengths.unsqueeze(1)
        out["prediction_start"] = torch.tensor(self.calibration_steps, device=device)
        out["use_static"] = False
        return out


class CalibOnly(nn.Module):
    """Calibration-only baseline with a learned time-index embedding."""

    def __init__(
        self,
        head_dim: int = 6,
        calibration_steps: int = 60,
        horizon_steps: int = 10,
        max_time_steps: int = 2048,
        fms_context_mode: str = "calibration_history",
    ):
        super().__init__()
        self.head_dim = head_dim
        self.calibration_steps = calibration_steps
        self.horizon_steps = horizon_steps
        self.max_time_steps = max_time_steps
        self.fms_context_mode = normalize_fms_context_mode(fms_context_mode)
        self.calibration_encoder = CalibrationEncoder(head_dim=head_dim)
        self.time_embedding = nn.Embedding(max_time_steps, 64)
        self.head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.now = nn.Linear(64, 1)
        self.future = nn.Linear(64, 1)

    def forward(
        self,
        head: torch.Tensor,
        y_calib: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        static: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del static
        assert head.ndim == 3 and head.shape[-1] == self.head_dim
        bsz, steps, _ = head.shape
        device = head.device
        context_fms = calibration_context_fms(y_calib, self.calibration_steps, self.fms_context_mode)
        z_calib = self.calibration_encoder(head[:, : self.calibration_steps], context_fms)
        idx = torch.arange(steps, device=device).clamp_max(self.max_time_steps - 1)
        t_emb = self.time_embedding(idx).unsqueeze(0).expand(bsz, -1, -1)
        z = z_calib.unsqueeze(1).expand(-1, steps, -1)
        hidden = self.head(torch.cat([z, t_emb], dim=-1))
        now = torch.sigmoid(self.now(hidden)).squeeze(-1)
        fut = torch.sigmoid(self.future(hidden)).squeeze(-1)
        out = {
            "y_now_hat": now,
            "y_future_hat": fut,
            "y_future_level": fut,
            "y_future_delta": fut,
            "gate": torch.ones_like(fut),
        }
        if lengths is None:
            lengths = torch.full((bsz,), steps, dtype=torch.long, device=device)
        else:
            lengths = lengths.to(device)
        max_forecast_t = int(torch.clamp(lengths.max() - self.horizon_steps, min=self.calibration_steps).item())
        pred_steps = max(0, max_forecast_t - self.calibration_steps)
        positions = self.calibration_steps + torch.arange(pred_steps, device=device)
        out["future"] = fut.index_select(1, positions) if pred_steps else fut[:, :0]
        out["mask"] = (positions.unsqueeze(0) + self.horizon_steps) < lengths.unsqueeze(1)
        out["prediction_start"] = torch.tensor(self.calibration_steps, device=device)
        out["use_static"] = False
        return out


def recent_dilations_for_window(recent_window_seconds: float) -> List[int]:
    window = float(recent_window_seconds)
    if window <= 20.0:
        return [1, 2, 4]
    if window <= 45.0:
        return [1, 2, 4, 8]
    return [1, 2, 4, 8, 16]


def resolve_dilations(value: Any, recent_window_seconds: Optional[float] = None) -> List[int]:
    if value is None or str(value).lower() == "auto":
        if recent_window_seconds is None:
            raise ValueError("recent_window_seconds is required when recent_dilations='auto'.")
        return recent_dilations_for_window(float(recent_window_seconds))
    if isinstance(value, str):
        tokens = [token.strip() for token in value.replace(";", ",").split(",") if token.strip()]
        return [int(token) for token in tokens]
    return [int(item) for item in value]


def tcn_receptive_field_steps(dilations: Sequence[int], kernel_size: int = 3, convs_per_block: int = 2) -> int:
    return 1 + int(convs_per_block) * (int(kernel_size) - 1) * int(sum(int(v) for v in dilations))


def adaptive_tcn_dilations_for_sequence(
    dilations: Sequence[int],
    sequence_steps: int,
    kernel_size: int = 3,
    max_padding_steps: int = 8,
    max_padding_fraction: float = 0.1,
) -> Tuple[List[int], int, int]:
    """Reduce the largest dilations until RF shortage can be covered by causal padding."""
    effective = [int(v) for v in dilations]
    if not effective:
        raise ValueError("TCN dilations must contain at least one value.")
    steps = max(0, int(sequence_steps))
    padding_limit = max(int(max_padding_steps), int(math.ceil(max(steps, 1) * max(0.0, float(max_padding_fraction)))))
    while len(effective) > 1:
        rf_steps = tcn_receptive_field_steps(effective, kernel_size=kernel_size)
        if rf_steps - steps <= padding_limit:
            break
        effective = effective[:-1]
    rf_steps = tcn_receptive_field_steps(effective, kernel_size=kernel_size)
    pad_steps = max(0, rf_steps - steps)
    return effective, rf_steps, pad_steps


class SequencePooling(nn.Module):
    def __init__(self, dim: int, mode: str = "mean"):
        super().__init__()
        mode = mode.lower()
        if mode not in {"mean", "last", "attention"}:
            raise ValueError(f"pooling must be one of mean, last, attention; got {mode!r}")
        self.mode = mode
        self.attn = nn.Linear(dim, 1) if mode == "attention" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "last":
            return x[:, -1]
        if self.mode == "attention":
            assert self.attn is not None
            weights = torch.softmax(self.attn(x).squeeze(-1), dim=1)
            return torch.sum(x * weights.unsqueeze(-1), dim=1)
        return x.mean(dim=1)


MOTION_FEATURE_MODES = {
    "none",
    "norm",
    "norm_delta",
    "norm_delta_energy",
    "causal_dynamics_v1",
    "multi_timescale_v1",
}
FORECAST_HEAD_MODES = {"level", "delta", "dual_average", "dual_gated"}
HORIZON_HEAD_MODES = {"linear", "h15_deep", "h15_residual", "h10_h15_residual"}


class DeepRegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResidualRegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.out = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x + self.block(x)).squeeze(-1)


def _motion_feature_dim(mode: str) -> int:
    mode = str(mode or "none").lower()
    if mode == "none":
        return 0
    if mode == "norm":
        return 2
    if mode == "norm_delta":
        return 4
    if mode == "norm_delta_energy":
        return 6
    if mode == "causal_dynamics_v1":
        return 18
    if mode == "multi_timescale_v1":
        return 16
    raise ValueError(f"motion_feature_mode must be one of {sorted(MOTION_FEATURE_MODES)}, got {mode!r}")


def _causal_rolling_mean(x: torch.Tensor, window_steps: int) -> torch.Tensor:
    if window_steps <= 1:
        return x
    cumsum = torch.cat([x.new_zeros((x.shape[0], 1, x.shape[2])), x.cumsum(dim=1)], dim=1)
    steps = torch.arange(x.shape[1], device=x.device)
    start = (steps - int(window_steps) + 1).clamp_min(0)
    end = steps + 1
    denom = (end - start).to(x.dtype).view(1, -1, 1).clamp_min(1.0)
    return (cumsum.index_select(1, end) - cumsum.index_select(1, start)) / denom


def _causal_dynamics_features(head: torch.Tensor, rolling_steps: int = 6) -> torch.Tensor:
    """Causal derivative, energy, and complexity proxies for head-motion streams."""
    accel = head[..., :3]
    gyro = head[..., 3:6] if head.shape[-1] >= 6 else head[..., :3]
    motion_norm = torch.linalg.vector_norm(head, dim=-1, keepdim=True)
    accel_norm = torch.linalg.vector_norm(accel, dim=-1, keepdim=True)
    gyro_norm = torch.linalg.vector_norm(gyro, dim=-1, keepdim=True)

    delta = torch.zeros_like(head)
    delta[:, 1:] = head[:, 1:] - head[:, :-1]
    delta_accel = delta[..., :3]
    delta_gyro = delta[..., 3:6] if delta.shape[-1] >= 6 else delta[..., :3]
    motion_delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    accel_delta_norm = torch.linalg.vector_norm(delta_accel, dim=-1, keepdim=True)
    gyro_delta_norm = torch.linalg.vector_norm(delta_gyro, dim=-1, keepdim=True)

    jerk = torch.zeros_like(delta)
    jerk[:, 1:] = delta[:, 1:] - delta[:, :-1]
    jerk_accel = jerk[..., :3]
    jerk_gyro = jerk[..., 3:6] if jerk.shape[-1] >= 6 else jerk[..., :3]
    accel_jerk_norm = torch.linalg.vector_norm(jerk_accel, dim=-1, keepdim=True)
    gyro_jerk_norm = torch.linalg.vector_norm(jerk_gyro, dim=-1, keepdim=True)
    motion_jerk_norm = torch.linalg.vector_norm(jerk, dim=-1, keepdim=True)

    short_window = max(2, int(rolling_steps))
    long_window = max(short_window + 1, int(rolling_steps) * 5)
    accel_energy_short = _causal_rolling_mean(accel_norm.square(), short_window).sqrt()
    gyro_energy_short = _causal_rolling_mean(gyro_norm.square(), short_window).sqrt()
    motion_energy_long = _causal_rolling_mean(motion_norm.square(), long_window).sqrt()
    motion_delta_energy_long = _causal_rolling_mean(motion_delta_norm.square(), long_window).sqrt()
    motion_jerk_energy = _causal_rolling_mean(motion_jerk_norm.square(), short_window).sqrt()
    short_motion_energy = _causal_rolling_mean(motion_norm.square(), short_window).sqrt()
    short_long_energy_ratio = short_motion_energy / motion_energy_long.clamp_min(1e-6)
    spectral_proxy = motion_delta_energy_long / motion_energy_long.clamp_min(1e-6)

    signs = torch.sign(delta)
    sign_flip = head.new_zeros(head.shape[:2] + (1,))
    if head.shape[1] > 1:
        sign_flip[:, 1:] = ((signs[:, 1:] * signs[:, :-1]) < 0).to(head.dtype).mean(dim=-1, keepdim=True)
    sign_change_rate = _causal_rolling_mean(sign_flip, short_window)

    channel_energy = _causal_rolling_mean(head.square(), long_window).clamp_min(0.0)
    energy_sum = channel_energy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    probs = channel_energy / energy_sum
    entropy_denom = max(math.log(float(head.shape[-1])), 1e-8)
    channel_energy_entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True) / entropy_denom
    participation_ratio = energy_sum.square() / channel_energy.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    channel_participation_ratio = participation_ratio / max(float(head.shape[-1]), 1.0)

    return torch.cat(
        [
            accel_norm,
            gyro_norm,
            motion_norm,
            accel_delta_norm,
            gyro_delta_norm,
            motion_delta_norm,
            accel_jerk_norm,
            gyro_jerk_norm,
            accel_energy_short,
            gyro_energy_short,
            motion_energy_long,
            motion_delta_energy_long,
            short_long_energy_ratio,
            motion_jerk_energy,
            sign_change_rate,
            spectral_proxy,
            channel_energy_entropy,
            channel_participation_ratio,
        ],
        dim=-1,
    )


def _multi_timescale_motion_features(head: torch.Tensor) -> torch.Tensor:
    """Causal 5/15/30/60 second motion-response summaries for 0.5s DenseFMS streams."""
    motion_norm = torch.linalg.vector_norm(head, dim=-1, keepdim=True)
    delta = torch.zeros_like(head)
    delta[:, 1:] = head[:, 1:] - head[:, :-1]
    motion_delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    jerk = torch.zeros_like(delta)
    jerk[:, 1:] = delta[:, 1:] - delta[:, :-1]
    jerk_norm = torch.linalg.vector_norm(jerk, dim=-1, keepdim=True)

    windows = [10, 30, 60, 120]
    motion_energy = [_causal_rolling_mean(motion_norm.square(), w).sqrt() for w in windows]
    jerk_energy = [_causal_rolling_mean(jerk_norm.square(), w).sqrt() for w in windows]
    delta_energy_15 = _causal_rolling_mean(motion_delta_norm.square(), 30).sqrt()
    ratio_5_30 = motion_energy[0] / motion_energy[2].clamp_min(1e-6)
    ratio_15_60 = motion_energy[1] / motion_energy[3].clamp_min(1e-6)
    jerk_burst_5_30 = jerk_energy[0] / jerk_energy[2].clamp_min(1e-6)

    channel_energy_15 = _causal_rolling_mean(head.square(), 30).clamp_min(0.0)
    channel_energy_60 = _causal_rolling_mean(head.square(), 120).clamp_min(0.0)
    entropy_denom = max(math.log(float(head.shape[-1])), 1e-8)

    def entropy_and_participation(channel_energy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        energy_sum = channel_energy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        probs = channel_energy / energy_sum
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True) / entropy_denom
        participation = energy_sum.square() / channel_energy.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        participation = participation / max(float(head.shape[-1]), 1.0)
        return entropy, participation

    entropy_15, _ = entropy_and_participation(channel_energy_15)
    entropy_60, participation_60 = entropy_and_participation(channel_energy_60)
    complexity_drop = entropy_15 - entropy_60

    return torch.cat(
        [
            *motion_energy,
            *jerk_energy,
            ratio_5_30,
            ratio_15_60,
            jerk_burst_5_30,
            delta_energy_15,
            entropy_15,
            entropy_60,
            complexity_drop,
            participation_60,
        ],
        dim=-1,
    )


def append_motion_features(head: torch.Tensor, mode: str, rolling_steps: int = 6) -> torch.Tensor:
    mode = str(mode or "none").lower()
    if mode == "none":
        return head
    if mode not in MOTION_FEATURE_MODES:
        raise ValueError(f"motion_feature_mode must be one of {sorted(MOTION_FEATURE_MODES)}, got {mode!r}")
    if mode == "causal_dynamics_v1":
        return torch.cat([head, _causal_dynamics_features(head, rolling_steps=rolling_steps)], dim=-1)
    if mode == "multi_timescale_v1":
        return torch.cat([head, _multi_timescale_motion_features(head)], dim=-1)
    accel = head[..., :3]
    gyro = head[..., 3:6] if head.shape[-1] >= 6 else head[..., :3]
    accel_norm = torch.linalg.vector_norm(accel, dim=-1, keepdim=True)
    gyro_norm = torch.linalg.vector_norm(gyro, dim=-1, keepdim=True)
    parts = [head, accel_norm, gyro_norm]
    if mode in {"norm_delta", "norm_delta_energy"}:
        diff_accel = torch.zeros_like(accel)
        diff_gyro = torch.zeros_like(gyro)
        diff_accel[:, 1:] = accel[:, 1:] - accel[:, :-1]
        diff_gyro[:, 1:] = gyro[:, 1:] - gyro[:, :-1]
        parts.extend(
            [
                torch.linalg.vector_norm(diff_accel, dim=-1, keepdim=True),
                torch.linalg.vector_norm(diff_gyro, dim=-1, keepdim=True),
            ]
        )
    if mode == "norm_delta_energy":
        parts.extend(
            [
                _causal_rolling_mean(accel_norm.square(), rolling_steps).sqrt(),
                _causal_rolling_mean(gyro_norm.square(), rolling_steps).sqrt(),
            ]
        )
    return torch.cat(parts, dim=-1)


def _prediction_positions_for(
    lengths: torch.Tensor,
    calibration_steps: int,
    recent_steps: int,
    horizon_steps_list: Sequence[int],
    device: torch.device,
    multi_horizon: bool = False,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    start = max(int(calibration_steps), int(recent_steps) - 1)
    max_horizon_steps = max(int(v) for v in horizon_steps_list)
    max_forecast_t = int(torch.clamp(lengths.max() - max_horizon_steps, min=start).item())
    pred_steps = max(0, max_forecast_t - start)
    positions = start + torch.arange(pred_steps, device=device, dtype=torch.long)
    if multi_horizon:
        horizons = torch.tensor([int(v) for v in horizon_steps_list], device=device, dtype=torch.long)
        mask = (positions.view(1, -1, 1) + horizons.view(1, 1, -1)) < lengths.view(-1, 1, 1)
    else:
        mask = (positions.view(1, -1) + int(horizon_steps_list[0])) < lengths.view(-1, 1)
    return start, positions, mask


def _anchor_indices_for(
    positions: torch.Tensor,
    anchor_mode: str,
    calibration_steps: int,
    recent_steps: int,
    anchor_interval_steps: int,
) -> Optional[torch.Tensor]:
    if anchor_mode == "none":
        return None
    if anchor_mode == "calibration_end":
        return torch.full_like(positions, int(calibration_steps) - 1)
    if anchor_mode == "recent_start_observed":
        return positions - int(recent_steps) + 1
    anchor = torch.div(positions, int(anchor_interval_steps), rounding_mode="floor") * int(anchor_interval_steps)
    minimum = torch.full_like(anchor, int(calibration_steps) - 1)
    return torch.maximum(anchor, minimum).clamp_max(positions)


def _gather_anchor_fms_safe(fms: torch.Tensor, anchor_idx: torch.Tensor, anchor_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(anchor_idx.max().detach().cpu()) >= fms.shape[1]:
        raise ValueError(
            f"anchor_mode={anchor_mode!r} needs FMS through index {int(anchor_idx.max())}, "
            f"but only {fms.shape[1]} steps were provided."
        )
    idx = anchor_idx.unsqueeze(0).expand(fms.shape[0], -1)
    gathered = fms.gather(1, idx)
    if torch.isfinite(gathered).all():
        return gathered, idx
    steps = torch.arange(fms.shape[1], dtype=torch.long, device=fms.device).unsqueeze(0)
    valid_idx = torch.where(torch.isfinite(fms), steps.expand_as(fms), torch.zeros_like(steps).expand_as(fms))
    latest_valid_idx = torch.cummax(valid_idx, dim=1).values
    actual_idx = latest_valid_idx.gather(1, idx)
    return fms.gather(1, actual_idx), actual_idx


def _calibration_summary_features(head: torch.Tensor, fms: torch.Tensor, calibration_steps: int) -> torch.Tensor:
    calib_fms = fms[:, : int(calibration_steps)]
    calib_head = head[:, : int(calibration_steps)]
    safe_fms = torch.nan_to_num(calib_fms, nan=0.0)
    fms_summary = [
        safe_fms[:, :1],
        safe_fms[:, -1:],
        safe_fms.mean(dim=1, keepdim=True),
        safe_fms.std(dim=1, unbiased=False, keepdim=True),
        safe_fms.max(dim=1, keepdim=True).values,
        safe_fms.min(dim=1, keepdim=True).values,
        (safe_fms[:, -1:] - safe_fms[:, :1]),
        (safe_fms[:, -1:] - safe_fms[:, :1]) / max(int(calibration_steps) - 1, 1),
    ]
    head_summary = [
        calib_head.mean(dim=1),
        calib_head.std(dim=1, unbiased=False),
    ]
    return torch.cat(fms_summary + head_summary, dim=-1)


def _window_motion_stats(head: torch.Tensor, positions: torch.Tensor, recent_steps: int) -> torch.Tensor:
    if positions.numel() == 0:
        return head.new_zeros((head.shape[0], 0, head.shape[-1] * 4 + 9))
    windows = head.unfold(dimension=1, size=int(recent_steps), step=1).permute(0, 1, 3, 2).contiguous()
    window_idx = positions - int(recent_steps) + 1
    selected = windows.index_select(1, window_idx)
    mean = selected.mean(dim=2)
    std = selected.std(dim=2, unbiased=False)
    min_v = selected.min(dim=2).values
    max_v = selected.max(dim=2).values
    first = selected[..., :3]
    second = selected[..., 3:6] if selected.shape[-1] >= 6 else selected[..., :3]
    first_mag = torch.linalg.vector_norm(first, dim=-1)
    second_mag = torch.linalg.vector_norm(second, dim=-1)
    diff = selected[:, :, 1:, :3] - selected[:, :, :-1, :3]
    jerk_mag = torch.linalg.vector_norm(diff, dim=-1) if diff.numel() else first_mag[:, :, :0]
    def mag_stats(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 0:
            return x.new_zeros((*x.shape[:2], 3))
        return torch.stack([x.mean(dim=-1), x.std(dim=-1, unbiased=False), x.max(dim=-1).values], dim=-1)

    return torch.cat([mean, std, min_v, max_v, mag_stats(first_mag), mag_stats(second_mag), mag_stats(jerk_mag)], dim=-1)


class FeedForwardMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: Sequence[int], output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        prev = int(input_dim)
        for hidden in hidden_layers:
            layers.extend([nn.Linear(prev, int(hidden)), nn.GELU(), nn.LayerNorm(int(hidden)), nn.Dropout(float(dropout))])
            prev = int(hidden)
        layers.append(nn.Linear(prev, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureAnchorForecaster(nn.Module):
    ANCHOR_MODES = {"none", "calibration_end", "recent_start_observed", "sparse_observed"}

    def __init__(
        self,
        head_dim: int = 6,
        calibration_steps: int = 180,
        horizon_steps: int = 10,
        recent_steps: int = 60,
        sampling_interval: float = 0.5,
        horizon_seconds: Optional[float] = None,
        anchor_mode: str = "sparse_observed",
        anchor_interval_seconds: float = 60.0,
        predict_delta_from_anchor: bool = True,
        use_static: bool = True,
        static_dim: int = 5,
        dropout: float = 0.1,
        hidden_dim: int = 128,
        mlp_layers: Optional[Sequence[int]] = None,
        multi_horizon: bool = False,
        horizon_set: Optional[Sequence[float]] = None,
        delta_scale: float = 0.5,
        fms_context_mode: str = "calibration_history",
    ):
        super().__init__()
        anchor_mode = str(anchor_mode).lower()
        if anchor_mode not in self.ANCHOR_MODES:
            raise ValueError(f"anchor_mode must be one of {sorted(self.ANCHOR_MODES)}, got {anchor_mode!r}")
        fms_context_mode = normalize_fms_context_mode(fms_context_mode)
        uses_window_start_fms = _uses_window_start_fms(fms_context_mode)
        if predict_delta_from_anchor and anchor_mode == "none" and not uses_window_start_fms:
            raise ValueError("predict_delta_from_anchor requires an anchor mode.")
        self.requires_full_fms = anchor_mode in {"sparse_observed", "recent_start_observed"} or uses_window_start_fms
        self.head_dim = int(head_dim)
        self.calibration_steps = int(calibration_steps)
        self.horizon_steps = int(horizon_steps)
        self.recent_steps = int(recent_steps)
        self.sampling_interval = float(sampling_interval)
        self.horizon_seconds = float(horizon_seconds if horizon_seconds is not None else self.horizon_steps * self.sampling_interval)
        self.anchor_mode = anchor_mode
        self.fms_context_mode = fms_context_mode
        self.uses_window_start_fms = uses_window_start_fms
        self.anchor_interval_seconds = float(anchor_interval_seconds)
        self.anchor_interval_steps = max(1, int(round(self.anchor_interval_seconds / self.sampling_interval)))
        self.predict_delta_from_anchor = bool(predict_delta_from_anchor)
        self.use_static = bool(use_static)
        self.static_dim = int(static_dim)
        self.multi_horizon = bool(multi_horizon)
        self.horizon_set = [float(v) for v in horizon_set] if horizon_set else None
        if self.multi_horizon and not self.horizon_set:
            raise ValueError("multi_horizon=True requires horizon_set.")
        self.horizon_seconds_list = self.horizon_set if self.multi_horizon else [self.horizon_seconds]
        self.horizon_steps_list = [max(1, int(round(v / self.sampling_interval))) for v in self.horizon_seconds_list]
        self.output_dim = len(self.horizon_steps_list)
        self.delta_scale = float(delta_scale)
        self.summary_dim = 8 + self.head_dim * 2
        self.recent_stats_dim = self.head_dim * 4 + 9
        self.anchor_dim = 2 if self.anchor_mode != "none" or self.uses_window_start_fms else 0
        self.feature_dim = self.summary_dim + self.recent_stats_dim + self.anchor_dim + (self.static_dim if self.use_static else 0) + 1
        layers = [int(v) for v in (mlp_layers if mlp_layers else [hidden_dim, hidden_dim, max(32, hidden_dim // 2)])]
        self.mlp = FeedForwardMLP(self.feature_dim, layers, output_dim=1, dropout=dropout)

    def _base_features(
        self,
        head: torch.Tensor,
        fms: torch.Tensor,
        lengths: torch.Tensor,
        static: Optional[torch.Tensor],
    ) -> Tuple[int, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        device = head.device
        start, positions, mask = _prediction_positions_for(
            lengths, self.calibration_steps, self.recent_steps, self.horizon_steps_list, device, self.multi_horizon
        )
        pred_steps = int(positions.numel())
        if pred_steps == 0:
            return start, positions, mask, None, None, head.new_zeros((head.shape[0], 0, self.feature_dim - 1))
        context_fms = calibration_context_fms(fms, self.calibration_steps, self.fms_context_mode)
        z_calib = _calibration_summary_features(head, context_fms, self.calibration_steps).unsqueeze(1).expand(-1, pred_steps, -1)
        z_recent = _window_motion_stats(head, positions, self.recent_steps)
        anchor_idx = _anchor_indices_for(
            positions, self.anchor_mode, self.calibration_steps, self.recent_steps, self.anchor_interval_steps
        )
        if anchor_idx is None and self.uses_window_start_fms:
            anchor_idx = (positions - self.recent_steps + 1).clamp_min(0)
        anchor_fms = None
        actual_anchor_idx = None
        parts = [z_calib, z_recent]
        if anchor_idx is not None:
            anchor_label = self.anchor_mode if self.anchor_mode != "none" else "window_start_fms"
            anchor_fms, actual_anchor_idx = _gather_anchor_fms_safe(fms, anchor_idx, anchor_label)
            time_since = (positions.unsqueeze(0) - actual_anchor_idx).to(head.dtype) * self.sampling_interval
            parts.append(torch.stack([anchor_fms, time_since / 120.0], dim=-1))
        if self.use_static:
            if static is None:
                raise ValueError(f"{self.__class__.__name__} was created with use_static=True, but static tensor was not provided.")
            parts.append(static.unsqueeze(1).expand(-1, pred_steps, -1))
        return start, positions, mask, anchor_fms, actual_anchor_idx, torch.cat(parts, dim=-1)

    def _predict_from_features(self, base_features: torch.Tensor) -> torch.Tensor:
        bsz, pred_steps, _ = base_features.shape
        if self.multi_horizon:
            outputs = []
            for horizon in self.horizon_seconds_list:
                h = base_features.new_full((bsz, pred_steps, 1), float(horizon) / 60.0)
                raw = self.mlp(torch.cat([base_features, h], dim=-1)).squeeze(-1)
                outputs.append(raw)
            return torch.stack(outputs, dim=-1)
        h = base_features.new_full((bsz, pred_steps, 1), self.horizon_seconds / 60.0)
        return self.mlp(torch.cat([base_features, h], dim=-1)).squeeze(-1)

    def forward(
        self,
        head: torch.Tensor,
        y_calib: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        static: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, steps, _ = head.shape
        device = head.device
        lengths = lengths.to(device) if lengths is not None else torch.full((bsz,), steps, dtype=torch.long, device=device)
        fms = y_calib.to(device)
        if self.requires_full_fms and fms.shape[1] < steps:
            raise ValueError(
                f"fms_context_mode={self.fms_context_mode!r}, anchor_mode={self.anchor_mode!r} "
                "requires full FMS input through current time."
            )
        start, positions, mask, anchor_fms, actual_anchor_idx, base_features = self._base_features(head, fms, lengths, static)
        if positions.numel() == 0:
            empty = head.new_zeros((bsz, 0, self.output_dim)) if self.multi_horizon else head.new_zeros((bsz, 0))
            return {
                "future": empty,
                "mask": mask,
                "prediction_start": torch.tensor(start, device=device),
                "horizon_steps_list": torch.tensor(self.horizon_steps_list, dtype=torch.long, device=device),
                "use_static": torch.tensor(self.use_static, device=device),
            }
        raw_pred = self._predict_from_features(base_features)
        if self.predict_delta_from_anchor:
            assert anchor_fms is not None
            base = anchor_fms.unsqueeze(-1) if self.multi_horizon else anchor_fms
            pred = torch.clamp(base + self.delta_scale * torch.tanh(raw_pred), 0.0, 1.0)
        else:
            pred = torch.sigmoid(raw_pred)
        out: Dict[str, torch.Tensor] = {
            "future": pred,
            "mask": mask,
            "prediction_start": torch.tensor(start, device=device),
            "horizon_steps_list": torch.tensor(self.horizon_steps_list, dtype=torch.long, device=device),
            "use_static": torch.tensor(self.use_static, device=device),
        }
        if actual_anchor_idx is not None and anchor_fms is not None:
            out["anchor_index"] = actual_anchor_idx
            out["anchor_fms"] = anchor_fms
        return out


class AnchorDeltaMLP(FeatureAnchorForecaster):
    pass


class AnchorDeltaGRU(FeatureAnchorForecaster):
    def __init__(
        self,
        head_dim: int = 6,
        hidden_dim: int = 64,
        gru_layers: int = 1,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(head_dim=head_dim, hidden_dim=hidden_dim, dropout=dropout, **kwargs)
        self.gru_hidden_dim = int(hidden_dim)
        self.gru_layers = int(gru_layers)
        self.gru = nn.GRU(
            input_size=int(head_dim),
            hidden_size=self.gru_hidden_dim,
            num_layers=self.gru_layers,
            batch_first=True,
            dropout=float(dropout) if self.gru_layers > 1 else 0.0,
        )
        old_dim = self.feature_dim
        self.feature_dim = old_dim + self.gru_hidden_dim
        self.mlp = FeedForwardMLP(
            self.feature_dim,
            [self.gru_hidden_dim * 2, self.gru_hidden_dim * 2, self.gru_hidden_dim],
            output_dim=1,
            dropout=dropout,
        )

    def _base_features(
        self,
        head: torch.Tensor,
        fms: torch.Tensor,
        lengths: torch.Tensor,
        static: Optional[torch.Tensor],
    ) -> Tuple[int, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        start, positions, mask, anchor_fms, actual_anchor_idx, base = super()._base_features(head, fms, lengths, static)
        if positions.numel() == 0:
            return start, positions, mask, anchor_fms, actual_anchor_idx, base
        encoded, _ = self.gru(head)
        z_recent_seq = encoded.index_select(1, positions)
        return start, positions, mask, anchor_fms, actual_anchor_idx, torch.cat([base, z_recent_seq], dim=-1)


class RecentTCNSummaryCalib(FeatureAnchorForecaster):
    def __init__(
        self,
        head_dim: int = 6,
        hidden_dim: int = 64,
        d_model: int = 64,
        kernel_size: int = 3,
        recent_dilations: Sequence[int] | str = "auto",
        pooling: str = "mean",
        dropout: float = 0.1,
        sampling_interval: float = 0.5,
        recent_steps: int = 60,
        **kwargs: Any,
    ):
        super().__init__(
            head_dim=head_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            sampling_interval=sampling_interval,
            recent_steps=recent_steps,
            **kwargs,
        )
        self.d_model = int(d_model)
        self.recent_dilations = resolve_dilations(recent_dilations, int(recent_steps) * float(sampling_interval))
        self.recent_encoder = LCBranchTCN(head_dim, self.d_model, self.recent_dilations, int(kernel_size), float(dropout))
        self.recent_pool = SequencePooling(self.d_model, pooling)
        self.recent_rf_steps = tcn_receptive_field_steps(self.recent_dilations, int(kernel_size))
        self.recent_rf_seconds = self.recent_rf_steps * float(sampling_interval)
        old_dim = self.feature_dim
        self.feature_dim = old_dim + self.d_model
        self.mlp = FeedForwardMLP(self.feature_dim, [self.d_model * 2, self.d_model * 2, self.d_model], output_dim=1, dropout=dropout)

    def _recent_tcn_at(self, head: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        encoded = self.recent_encoder(head)
        if self.recent_pool.mode == "last":
            return encoded.index_select(1, positions)
        window_idx = positions - self.recent_steps + 1
        if self.recent_pool.mode == "mean":
            cumsum = torch.cat([encoded.new_zeros((encoded.shape[0], 1, encoded.shape[2])), encoded.cumsum(dim=1)], dim=1)
            end = positions + 1
            sums = cumsum.index_select(1, end) - cumsum.index_select(1, window_idx)
            return sums / float(self.recent_steps)
        windows = encoded.unfold(dimension=1, size=self.recent_steps, step=1).permute(0, 1, 3, 2).contiguous()
        selected = windows.index_select(1, window_idx)
        bsz, pred_steps, recent_steps, dim = selected.shape
        pooled = self.recent_pool(selected.view(-1, recent_steps, dim))
        return pooled.view(bsz, pred_steps, dim)

    def _base_features(
        self,
        head: torch.Tensor,
        fms: torch.Tensor,
        lengths: torch.Tensor,
        static: Optional[torch.Tensor],
    ) -> Tuple[int, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        start, positions, mask, anchor_fms, actual_anchor_idx, base = super()._base_features(head, fms, lengths, static)
        if positions.numel() == 0:
            return start, positions, mask, anchor_fms, actual_anchor_idx, base
        return start, positions, mask, anchor_fms, actual_anchor_idx, torch.cat([base, self._recent_tcn_at(head, positions)], dim=-1)


class GatedFusionForecaster(RecentTCNSummaryCalib):
    def __init__(
        self,
        branch_dropout: float = 0.0,
        anchor_dropout: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.branch_dropout = float(branch_dropout)
        self.anchor_dropout = float(anchor_dropout)
        base_dim = self.feature_dim - 1
        self.gate = nn.Sequential(nn.Linear(base_dim, base_dim), nn.Sigmoid())

    def _predict_from_features(self, base_features: torch.Tensor) -> torch.Tensor:
        if self.training and self.branch_dropout > 0:
            keep = torch.rand_like(base_features) > self.branch_dropout
            base_features = base_features * keep.to(base_features.dtype)
        gated = base_features * self.gate(base_features)
        return super()._predict_from_features(gated)


class LCBranchTCN(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dilations: Sequence[int], kernel_size: int, dropout: float):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.blocks = nn.Sequential(
            *[TCNBlock(d_model, dilation=int(d), dropout=dropout, kernel_size=kernel_size) for d in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input(x))


class LCSATCNFormer(nn.Module):
    """Long-Calibrated State-Anchored TCN-Transformer forecaster.

    The model accepts full normalized FMS only to compute the configured anchor
    value. Calibration uses only the first C steps; recent motion windows end at
    current time t; target positions are never indexed inside the forward pass.
    """

    ANCHOR_MODES = {"none", "calibration_end", "recent_start_observed", "sparse_observed"}

    def __init__(
        self,
        head_dim: int = 6,
        calibration_steps: int = 180,
        horizon_steps: int = 10,
        recent_steps: int = 60,
        sampling_interval: float = 0.5,
        horizon_seconds: Optional[float] = None,
        d_model: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
        calib_dilations: Sequence[int] = (1, 2, 4, 8, 16),
        recent_dilations: Sequence[int] | str = "auto",
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 128,
        pooling: str = "mean",
        anchor_mode: str = "calibration_end",
        anchor_interval_seconds: float = 60.0,
        predict_delta_from_anchor: bool = False,
        use_static: bool = False,
        static_dim: int = 4,
        static_hidden_dim: Optional[int] = None,
        static_dropout: float = 0.1,
        multi_horizon: bool = False,
        horizon_set: Optional[Sequence[float]] = None,
        per_horizon_heads: bool = False,
        horizon_encoder_dim: Optional[int] = None,
        horizon_context_mode: str = "encoded",
        start_fms_context_mode: str = "encoded",
        static_context_mode: str = "encoded",
        forecast_head_mode: str = "level",
        horizon_head_mode: str = "linear",
        horizon_head_hidden_dim: Optional[int] = None,
        motion_feature_mode: str = "none",
        change_aux_head: bool = False,
        fms_context_mode: str = "calibration_history",
    ):
        super().__init__()
        if d_model % transformer_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by transformer_heads={transformer_heads}")
        anchor_mode = anchor_mode.lower()
        if anchor_mode not in self.ANCHOR_MODES:
            raise ValueError(f"anchor_mode must be one of {sorted(self.ANCHOR_MODES)}, got {anchor_mode!r}")
        fms_context_mode = normalize_fms_context_mode(fms_context_mode)
        uses_window_start_fms = _uses_window_start_fms(fms_context_mode)
        if predict_delta_from_anchor and anchor_mode == "none" and not uses_window_start_fms:
            raise ValueError("predict_delta_from_anchor requires an anchor_mode other than 'none'.")
        self.requires_full_fms = anchor_mode in {"sparse_observed", "recent_start_observed"} or uses_window_start_fms
        self.head_dim = int(head_dim)
        self.calibration_steps = int(calibration_steps)
        self.horizon_steps = int(horizon_steps)
        self.recent_steps = int(recent_steps)
        self.sampling_interval = float(sampling_interval)
        self.horizon_seconds = float(horizon_seconds if horizon_seconds is not None else self.horizon_steps * self.sampling_interval)
        self.d_model = int(d_model)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.calib_dilations = [int(v) for v in calib_dilations]
        self.recent_dilations = resolve_dilations(recent_dilations, self.recent_steps * self.sampling_interval)
        self.transformer_layers = int(transformer_layers)
        self.transformer_heads = int(transformer_heads)
        self.transformer_ff_dim = int(transformer_ff_dim)
        self.pooling = pooling
        self.anchor_mode = anchor_mode
        self.fms_context_mode = fms_context_mode
        self.uses_window_start_fms = uses_window_start_fms
        self.anchor_interval_seconds = float(anchor_interval_seconds)
        self.anchor_interval_steps = max(1, int(round(self.anchor_interval_seconds / self.sampling_interval)))
        self.predict_delta_from_anchor = bool(predict_delta_from_anchor)
        self.use_static = bool(use_static)
        self.static_dim = int(static_dim)
        self.multi_horizon = bool(multi_horizon)
        self.per_horizon_heads = bool(per_horizon_heads)
        has_anchor_context = self.anchor_mode != "none" or self.uses_window_start_fms
        self.horizon_context_mode = str(horizon_context_mode).lower()
        if self.horizon_context_mode not in {"encoded", "scalar", "none"}:
            raise ValueError("horizon_context_mode must be one of: encoded, scalar, none")
        self.start_fms_context_mode = str(start_fms_context_mode).lower()
        if self.start_fms_context_mode not in {"encoded", "scalar", "scalar_time"}:
            raise ValueError("start_fms_context_mode must be one of: encoded, scalar, scalar_time")
        self.static_context_mode = str(static_context_mode).lower()
        if self.static_context_mode not in {"encoded", "raw"}:
            raise ValueError("static_context_mode must be one of: encoded, raw")
        self.forecast_head_mode = str(forecast_head_mode or "level").lower()
        if self.predict_delta_from_anchor and self.forecast_head_mode == "level":
            self.forecast_head_mode = "delta"
        if self.forecast_head_mode not in FORECAST_HEAD_MODES:
            raise ValueError(f"forecast_head_mode must be one of: {sorted(FORECAST_HEAD_MODES)}")
        if self.forecast_head_mode != "level" and not has_anchor_context:
            raise ValueError("delta/dual forecast head modes require start_only or another anchor context.")
        self.horizon_head_mode = str(horizon_head_mode or "linear").lower()
        if self.horizon_head_mode not in HORIZON_HEAD_MODES:
            raise ValueError(f"horizon_head_mode must be one of: {sorted(HORIZON_HEAD_MODES)}")
        self.horizon_head_hidden_dim = int(horizon_head_hidden_dim if horizon_head_hidden_dim is not None else self.d_model)
        self.motion_feature_mode = str(motion_feature_mode or "none").lower()
        self.motion_feature_dim = _motion_feature_dim(self.motion_feature_mode)
        self.change_aux_head_enabled = bool(change_aux_head)
        self.horizon_encoder_dim = self.d_model if horizon_encoder_dim is None else int(horizon_encoder_dim)
        if self.horizon_encoder_dim < 0:
            raise ValueError("horizon_encoder_dim must be >= 0")
        if self.horizon_context_mode == "encoded" and self.horizon_encoder_dim == 0:
            self.horizon_context_mode = "none"
        self.horizon_set = [float(v) for v in horizon_set] if horizon_set else None
        if self.multi_horizon and not self.horizon_set:
            raise ValueError("multi_horizon=True requires horizon_set.")
        self.horizon_seconds_list = self.horizon_set if self.multi_horizon else [self.horizon_seconds]
        self.horizon_steps_list = [max(1, int(round(v / self.sampling_interval))) for v in self.horizon_seconds_list]
        self.output_dim = len(self.horizon_steps_list)

        self.calibration_encoder = LCBranchTCN(
            input_dim=self.head_dim + 1,
            d_model=self.d_model,
            dilations=self.calib_dilations,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.transformer_heads,
            dim_feedforward=self.transformer_ff_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.calibration_transformer = nn.TransformerEncoder(transformer_layer, num_layers=self.transformer_layers)
        self.calibration_pool = SequencePooling(self.d_model, pooling)
        self.recent_encoder = LCBranchTCN(
            input_dim=self.head_dim + self.motion_feature_dim,
            d_model=self.d_model,
            dilations=self.recent_dilations,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )
        self.recent_pool = SequencePooling(self.d_model, pooling)
        self.anchor_encoder = (
            nn.Sequential(
                nn.Linear(2, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
            )
            if has_anchor_context and self.start_fms_context_mode == "encoded"
            else None
        )
        hidden_static = int(static_hidden_dim if static_hidden_dim is not None else self.d_model)
        self.static_encoder = (
            StaticEncoder(self.static_dim, hidden_static, static_dropout)
            if self.use_static and self.static_context_mode == "encoded"
            else None
        )
        self.static_projection = (
            nn.Sequential(nn.Linear(hidden_static, self.d_model), nn.GELU(), nn.LayerNorm(self.d_model))
            if self.use_static and self.static_context_mode == "encoded" and hidden_static != self.d_model
            else None
        )
        self.horizon_encoder = (
            nn.Sequential(
                nn.Linear(1, self.horizon_encoder_dim),
                nn.GELU(),
                nn.LayerNorm(self.horizon_encoder_dim),
            )
            if self.horizon_context_mode == "encoded" and self.horizon_encoder_dim > 0
            else None
        )
        anchor_context_dim = 0
        if has_anchor_context:
            anchor_context_dim = self.d_model if self.start_fms_context_mode == "encoded" else (1 if self.start_fms_context_mode == "scalar" else 2)
        static_context_dim = 0
        if self.use_static:
            static_context_dim = self.d_model if self.static_context_mode == "encoded" else self.static_dim
        horizon_context_dim = 0
        if self.horizon_context_mode == "encoded":
            horizon_context_dim = self.horizon_encoder_dim
        elif self.horizon_context_mode == "scalar":
            horizon_context_dim = 1
        fusion_dim = (self.d_model * 2) + anchor_context_dim + static_context_dim + horizon_context_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.head = self._make_output_head(None)
        self.horizon_heads = (
            nn.ModuleList([self._make_output_head(idx) for idx, _ in enumerate(self.horizon_steps_list)])
            if self.multi_horizon and self.per_horizon_heads
            else None
        )
        if self.forecast_head_mode != "level":
            self.delta_head = self._make_output_head(None)
            self.delta_horizon_heads = (
                nn.ModuleList([self._make_output_head(idx) for idx, _ in enumerate(self.horizon_steps_list)])
                if self.multi_horizon and self.per_horizon_heads
                else None
            )
        else:
            self.delta_head = None
            self.delta_horizon_heads = None
        if self.forecast_head_mode == "dual_gated":
            self.gate_head = self._make_output_head(None)
            self.gate_horizon_heads = (
                nn.ModuleList([self._make_output_head(idx) for idx, _ in enumerate(self.horizon_steps_list)])
                if self.multi_horizon and self.per_horizon_heads
                else None
            )
        else:
            self.gate_head = None
            self.gate_horizon_heads = None
        self.change_aux_head = nn.Linear(self.d_model, 3) if self.change_aux_head_enabled else None
        self.change_aux_horizon_heads = (
            nn.ModuleList([nn.Linear(self.d_model, 3) for _ in self.horizon_steps_list])
            if self.change_aux_head_enabled and self.multi_horizon and self.per_horizon_heads
            else None
        )
        self.recent_rf_steps = tcn_receptive_field_steps(self.recent_dilations, self.kernel_size)
        self.recent_rf_seconds = self.recent_rf_steps * self.sampling_interval

    def _horizon_seconds_for_idx(self, horizon_idx: Optional[int]) -> Optional[float]:
        if horizon_idx is None or horizon_idx >= len(self.horizon_seconds_list):
            return None
        return float(self.horizon_seconds_list[int(horizon_idx)])

    def _head_needs_capacity(self, horizon_idx: Optional[int]) -> bool:
        horizon = self._horizon_seconds_for_idx(horizon_idx)
        if horizon is None:
            return False
        if self.horizon_head_mode == "h15_deep":
            return abs(horizon - 15.0) < 1e-6
        if self.horizon_head_mode == "h15_residual":
            return abs(horizon - 15.0) < 1e-6
        if self.horizon_head_mode == "h10_h15_residual":
            return any(abs(horizon - value) < 1e-6 for value in (10.0, 15.0))
        return False

    def _make_output_head(self, horizon_idx: Optional[int]) -> nn.Module:
        if not self._head_needs_capacity(horizon_idx):
            return nn.Linear(self.d_model, 1)
        if self.horizon_head_mode == "h15_deep":
            return DeepRegressionHead(self.d_model, self.horizon_head_hidden_dim, self.dropout)
        return ResidualRegressionHead(self.d_model, self.horizon_head_hidden_dim, self.dropout)

    def _apply_output_head(self, head: nn.Module, fused: torch.Tensor) -> torch.Tensor:
        raw = head(fused)
        return raw.squeeze(-1) if raw.ndim == fused.ndim else raw

    def _change_logits_from_fused(self, fused: torch.Tensor, horizon_idx: Optional[int]) -> Optional[torch.Tensor]:
        if self.change_aux_head is None:
            return None
        if self.change_aux_horizon_heads is not None:
            if horizon_idx is None:
                raise ValueError("horizon_idx is required for per-horizon change auxiliary heads.")
            return self.change_aux_horizon_heads[int(horizon_idx)](fused)
        return self.change_aux_head(fused)

    def _prediction_positions(self, lengths: torch.Tensor, device: torch.device) -> Tuple[int, torch.Tensor, torch.Tensor]:
        start = max(self.calibration_steps, self.recent_steps - 1)
        max_horizon_steps = max(self.horizon_steps_list)
        max_forecast_t = int(torch.clamp(lengths.max() - max_horizon_steps, min=start).item())
        pred_steps = max(0, max_forecast_t - start)
        positions = start + torch.arange(pred_steps, device=device, dtype=torch.long)
        if self.multi_horizon:
            horizons = torch.tensor(self.horizon_steps_list, device=device, dtype=torch.long)
            mask = (positions.view(1, -1, 1) + horizons.view(1, 1, -1)) < lengths.view(-1, 1, 1)
        else:
            mask = (positions.view(1, -1) + self.horizon_steps) < lengths.view(-1, 1)
        return start, positions, mask

    def _anchor_indices(self, positions: torch.Tensor) -> Optional[torch.Tensor]:
        if self.anchor_mode == "none":
            if self.uses_window_start_fms:
                return (positions - self.recent_steps + 1).clamp_min(0)
            return None
        if self.anchor_mode == "calibration_end":
            return torch.full_like(positions, self.calibration_steps - 1)
        if self.anchor_mode == "recent_start_observed":
            return positions - self.recent_steps + 1
        anchor = torch.div(positions, self.anchor_interval_steps, rounding_mode="floor") * self.anchor_interval_steps
        minimum = torch.full_like(anchor, self.calibration_steps - 1)
        return torch.maximum(anchor, minimum).clamp_max(positions)

    def _gather_anchor_fms(self, fms: torch.Tensor, anchor_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if int(anchor_idx.max().detach().cpu()) >= fms.shape[1]:
            raise ValueError(
                f"anchor_mode={self.anchor_mode!r} needs FMS through index {int(anchor_idx.max())}, "
                f"but only {fms.shape[1]} steps were provided."
            )
        idx = anchor_idx.unsqueeze(0).expand(fms.shape[0], -1)
        if torch.isfinite(fms.gather(1, idx)).all():
            return fms.gather(1, idx), idx
        steps = torch.arange(fms.shape[1], dtype=torch.long, device=fms.device).unsqueeze(0)
        valid_idx = torch.where(torch.isfinite(fms), steps.expand_as(fms), torch.zeros_like(steps).expand_as(fms))
        latest_valid_idx = torch.cummax(valid_idx, dim=1).values
        actual_idx = latest_valid_idx.gather(1, idx)
        return fms.gather(1, actual_idx), actual_idx

    def _encode_calibration(self, head: torch.Tensor, fms: torch.Tensor) -> torch.Tensor:
        if head.shape[1] < self.calibration_steps:
            raise ValueError("head sequence shorter than calibration_steps")
        if fms.shape[1] < self.calibration_steps and self.fms_context_mode == "calibration_history":
            raise ValueError("FMS sequence shorter than calibration_steps")
        context_fms = calibration_context_fms(fms, self.calibration_steps, self.fms_context_mode)
        calib_seq = torch.cat([head[:, : self.calibration_steps], context_fms.unsqueeze(-1)], dim=-1)
        z = self.calibration_encoder(calib_seq)
        z = self.calibration_transformer(z)
        return self.calibration_pool(z)

    def _encode_recent(self, head: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if positions.numel() == 0:
            return head.new_zeros((head.shape[0], 0, self.d_model))
        recent_input = append_motion_features(head, self.motion_feature_mode)
        encoded = self.recent_encoder(recent_input)
        if self.recent_pool.mode == "last":
            return encoded.index_select(1, positions)
        window_idx = positions - self.recent_steps + 1
        if self.recent_pool.mode == "mean":
            cumsum = torch.cat([encoded.new_zeros((encoded.shape[0], 1, encoded.shape[2])), encoded.cumsum(dim=1)], dim=1)
            end = positions + 1
            sums = cumsum.index_select(1, end) - cumsum.index_select(1, window_idx)
            return sums / float(self.recent_steps)
        windows = encoded.unfold(dimension=1, size=self.recent_steps, step=1)
        windows = windows.permute(0, 1, 3, 2).contiguous()
        selected = windows.index_select(1, window_idx)
        bsz, pred_steps, recent_steps, dim = selected.shape
        pooled = self.recent_pool(selected.view(-1, recent_steps, dim))
        return pooled.view(bsz, pred_steps, self.d_model)

    def _encode_static(self, static: Optional[torch.Tensor], pred_steps: int, z_calib: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_static:
            return None
        if static is None:
            raise ValueError("LCSATCNFormer was created with use_static=True, but static tensor was not provided.")
        if static.ndim != 2 or static.shape[0] != z_calib.shape[0] or static.shape[1] != self.static_dim:
            raise ValueError(f"static must be [B,{self.static_dim}], got {static.shape}")
        if self.static_context_mode == "raw":
            return static.unsqueeze(1).expand(-1, pred_steps, -1)
        assert self.static_encoder is not None
        z_static = self.static_encoder(static)
        if self.static_projection is not None:
            z_static = self.static_projection(z_static)
        return z_static.unsqueeze(1).expand(-1, pred_steps, -1)

    def _fuse_single(
        self,
        z_calib: torch.Tensor,
        z_recent: torch.Tensor,
        z_anchor: Optional[torch.Tensor],
        z_static: Optional[torch.Tensor],
        horizon_seconds: torch.Tensor,
        horizon_idx: Optional[int] = None,
    ) -> torch.Tensor:
        pred_steps = z_recent.shape[1]
        parts = [
            z_calib.unsqueeze(1).expand(-1, pred_steps, -1),
            z_recent,
        ]
        if self.horizon_encoder is not None:
            parts.append(self.horizon_encoder(horizon_seconds))
        elif self.horizon_context_mode == "scalar":
            parts.append(horizon_seconds)
        if z_anchor is not None:
            parts.append(z_anchor)
        if z_static is not None:
            parts.append(z_static)
        fused = self.fusion(torch.cat(parts, dim=-1))
        return fused

    def _raw_from_head(
        self,
        fused: torch.Tensor,
        shared_head: nn.Module,
        horizon_heads: Optional[nn.ModuleList],
        horizon_idx: Optional[int],
    ) -> torch.Tensor:
        if horizon_heads is not None:
            if horizon_idx is None:
                raise ValueError("horizon_idx is required when per_horizon_heads=True.")
            return self._apply_output_head(horizon_heads[int(horizon_idx)], fused)
        return self._apply_output_head(shared_head, fused)

    def _predict_from_fused(
        self,
        fused: torch.Tensor,
        anchor_fms: Optional[torch.Tensor],
        horizon_idx: Optional[int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        level_raw = self._raw_from_head(fused, self.head, self.horizon_heads, horizon_idx)
        level_pred = torch.sigmoid(level_raw)
        aux: Dict[str, torch.Tensor] = {"future_level": level_pred}
        if self.forecast_head_mode == "level":
            return level_pred, aux
        if anchor_fms is None:
            raise ValueError(f"forecast_head_mode={self.forecast_head_mode!r} requires start-FMS anchor values.")
        assert self.delta_head is not None
        delta_raw = self._raw_from_head(fused, self.delta_head, self.delta_horizon_heads, horizon_idx)
        delta_pred = 0.5 * torch.tanh(delta_raw)
        delta_value = torch.clamp(anchor_fms + delta_pred, 0.0, 1.0)
        aux.update(
            {
                "future_delta_pred": delta_pred,
                "future_delta_value": delta_value,
                "future_delta_base": anchor_fms,
            }
        )
        if self.forecast_head_mode == "delta":
            return delta_value, aux
        if self.forecast_head_mode == "dual_average":
            return 0.5 * (level_pred + delta_value), aux
        assert self.forecast_head_mode == "dual_gated"
        assert self.gate_head is not None
        gate_raw = self._raw_from_head(fused, self.gate_head, self.gate_horizon_heads, horizon_idx)
        gate = torch.sigmoid(gate_raw)
        aux["future_gate"] = gate
        return gate * level_pred + (1.0 - gate) * delta_value, aux

    def forward(
        self,
        head: torch.Tensor,
        y_calib: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        static: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        assert head.ndim == 3, f"head must be [B,T,6], got {head.shape}"
        if head.shape[-1] != self.head_dim:
            raise ValueError(f"expected head_dim={self.head_dim}, got {head.shape[-1]}")
        bsz, steps, _ = head.shape
        device = head.device
        if lengths is None:
            lengths = torch.full((bsz,), steps, dtype=torch.long, device=device)
        else:
            lengths = lengths.to(device)
        if y_calib.ndim != 2:
            raise ValueError(f"FMS input must be [B,T] or [B,C], got {y_calib.shape}")
        fms = y_calib.to(device)
        if self.requires_full_fms and fms.shape[1] < steps:
            raise ValueError(
                f"fms_context_mode={self.fms_context_mode!r}, anchor_mode={self.anchor_mode!r} "
                "requires full FMS input through current time."
            )

        start, positions, mask = self._prediction_positions(lengths, device)
        pred_steps = int(positions.numel())
        if pred_steps == 0:
            empty = head.new_zeros((bsz, 0, self.output_dim)) if self.multi_horizon else head.new_zeros((bsz, 0))
            return {
                "future": empty,
                "mask": mask,
                "prediction_start": torch.tensor(start, device=device),
                "horizon_steps_list": torch.tensor(self.horizon_steps_list, dtype=torch.long, device=device),
                "use_static": torch.tensor(self.use_static, device=device),
            }

        z_calib = self._encode_calibration(head, fms)
        z_recent = self._encode_recent(head, positions)
        z_static = self._encode_static(static, pred_steps, z_calib)

        anchor_idx = self._anchor_indices(positions)
        anchor_fms = None
        z_anchor = None
        if anchor_idx is not None:
            anchor_fms, actual_anchor_idx = self._gather_anchor_fms(fms, anchor_idx)
            time_since = (positions.unsqueeze(0) - actual_anchor_idx).to(head.dtype) * self.sampling_interval
            anchor_input = torch.stack([anchor_fms, time_since / 120.0], dim=-1)
            if self.start_fms_context_mode == "encoded":
                assert self.anchor_encoder is not None
                z_anchor = self.anchor_encoder(anchor_input)
            elif self.start_fms_context_mode == "scalar":
                z_anchor = anchor_fms.unsqueeze(-1)
            else:
                z_anchor = anchor_input

        if self.multi_horizon:
            preds = []
            aux_by_key: Dict[str, List[torch.Tensor]] = {}
            distill_repr_values: List[torch.Tensor] = []
            for horizon_idx, horizon_seconds in enumerate(self.horizon_seconds_list):
                horizon_in = head.new_full((bsz, pred_steps, 1), float(horizon_seconds) / 60.0)
                fused = self._fuse_single(z_calib, z_recent, z_anchor, z_static, horizon_in, horizon_idx=horizon_idx)
                distill_repr_values.append(fused)
                pred_h, aux_h = self._predict_from_fused(fused, anchor_fms, horizon_idx)
                change_logits = self._change_logits_from_fused(fused, horizon_idx)
                if change_logits is not None:
                    aux_h["future_change_logits"] = change_logits
                preds.append(pred_h)
                for key, value in aux_h.items():
                    aux_by_key.setdefault(key, []).append(value)
            pred = torch.stack(preds, dim=-1)
            aux_out = {
                key: torch.stack(values, dim=2 if key == "future_change_logits" else -1)
                for key, values in aux_by_key.items()
            }
            aux_out["distill_repr"] = torch.stack(distill_repr_values, dim=2)
        else:
            horizon_in = head.new_full((bsz, pred_steps, 1), self.horizon_seconds / 60.0)
            fused = self._fuse_single(z_calib, z_recent, z_anchor, z_static, horizon_in)
            pred, aux_out = self._predict_from_fused(fused, anchor_fms, None)
            change_logits = self._change_logits_from_fused(fused, None)
            if change_logits is not None:
                aux_out["future_change_logits"] = change_logits
            aux_out["distill_repr"] = fused

        out: Dict[str, torch.Tensor] = {
            "future": pred,
            "mask": mask,
            "prediction_start": torch.tensor(start, device=device),
            "horizon_steps_list": torch.tensor(self.horizon_steps_list, dtype=torch.long, device=device),
            "use_static": torch.tensor(self.use_static, device=device),
            "z_calib_norm": z_calib.norm(dim=-1),
        }
        out.update(aux_out)
        if anchor_idx is not None:
            out["anchor_index"] = actual_anchor_idx
            out["anchor_fms"] = anchor_fms
        return out


class CalibInitStateForecaster(nn.Module):
    """Calibration-initialized latent state forecaster.

    FMS is visible only inside the calibration window. After calibration, head
    motion is processed as a causal stream and updates a latent sickness state.
    Post-calibration FMS labels are never read in ``forward``.
    """

    ANCHOR_MODES = {"none", "calibration_end"}

    def __init__(
        self,
        head_dim: int = 6,
        calibration_steps: int = 180,
        horizon_steps: int = 10,
        recent_steps: int = 20,
        sampling_interval: float = 0.5,
        horizon_seconds: Optional[float] = None,
        d_model: int = 96,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.05,
        calib_dilations: Sequence[int] = (1, 2, 4, 8, 16),
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 192,
        pooling: str = "mean",
        anchor_mode: str = "none",
        anchor_interval_seconds: float = 0.0,
        use_static: bool = False,
        static_dim: int = 4,
        static_hidden_dim: Optional[int] = None,
        static_dropout: float = 0.1,
        multi_horizon: bool = False,
        horizon_set: Optional[Sequence[float]] = None,
        per_horizon_heads: bool = True,
        forecast_head_mode: str = "level",
        delta_scale: float = 0.5,
        motion_feature_mode: str = "norm",
        stream_time_features: bool = False,
        stream_context_mode: str = "gru",
        calib_summary_features: bool = False,
        calibration_encoder_mode: str = "tcn_transformer",
        state_feedback_mode: str = "none",
        session_context_mode: str = "none",
        fms_context_mode: str = "calibration_history",
    ):
        super().__init__()
        if d_model % transformer_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by transformer_heads={transformer_heads}")
        anchor_mode = str(anchor_mode or "none").lower()
        if anchor_mode not in self.ANCHOR_MODES:
            raise ValueError(f"CalibInitStateForecaster anchor_mode must be one of {sorted(self.ANCHOR_MODES)}")
        fms_context_mode = normalize_fms_context_mode(fms_context_mode)
        if fms_context_mode not in {"calibration_history", "none"}:
            raise ValueError("CalibInitStateForecaster allows only calibration_history or none FMS context.")
        forecast_head_mode = str(forecast_head_mode or "level").lower()
        if forecast_head_mode not in {"level", "delta", "self_delta", "recent_start_delta", "rollin_start_delta"}:
            raise ValueError(
                "CalibInitStateForecaster supports forecast_head_mode level, delta, self_delta, "
                "recent_start_delta, or rollin_start_delta."
            )
        if forecast_head_mode == "delta" and anchor_mode != "calibration_end":
            raise ValueError("forecast_head_mode=delta requires anchor_mode=calibration_end.")

        self.head_dim = int(head_dim)
        self.calibration_steps = int(calibration_steps)
        self.horizon_steps = int(horizon_steps)
        self.recent_steps = int(recent_steps)
        self.sampling_interval = float(sampling_interval)
        self.horizon_seconds = float(horizon_seconds if horizon_seconds is not None else self.horizon_steps * self.sampling_interval)
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.pooling = pooling
        self.anchor_mode = anchor_mode
        self.anchor_interval_seconds = float(anchor_interval_seconds)
        self.fms_context_mode = fms_context_mode
        self.requires_full_fms = False
        self.use_static = bool(use_static)
        self.static_dim = int(static_dim)
        self.multi_horizon = bool(multi_horizon)
        self.per_horizon_heads = bool(per_horizon_heads)
        self.forecast_head_mode = forecast_head_mode
        self.delta_scale = float(delta_scale)
        self.motion_feature_mode = str(motion_feature_mode or "none").lower()
        self.motion_feature_dim = _motion_feature_dim(self.motion_feature_mode)
        self.stream_time_features = bool(stream_time_features)
        self.stream_time_feature_dim = 2 if self.stream_time_features else 0
        self.stream_context_mode = str(stream_context_mode or "gru").lower()
        if self.stream_context_mode not in {"gru", "gru_multiscale", "gru_tcn", "gru_tcn_multiscale"}:
            raise ValueError("stream_context_mode must be 'gru', 'gru_multiscale', 'gru_tcn', or 'gru_tcn_multiscale'.")
        self.calib_summary_features = bool(calib_summary_features)
        self.calibration_encoder_mode = str(calibration_encoder_mode or "tcn_transformer").lower()
        if self.calibration_encoder_mode not in {
            "tcn_transformer",
            "transformer",
            "transformer_cls",
            "deep_tcn",
            "deep_tcn_transformer",
        }:
            raise ValueError(
                "calibration_encoder_mode must be 'tcn_transformer', 'transformer', 'transformer_cls', "
                "'deep_tcn', or 'deep_tcn_transformer'."
            )
        self.state_feedback_mode = str(state_feedback_mode or "none").lower()
        if self.state_feedback_mode not in {"none", "predicted_current"}:
            raise ValueError("state_feedback_mode must be 'none' or 'predicted_current'.")
        self.state_feedback_dim = 1 if self.state_feedback_mode == "predicted_current" else 0
        self.session_context_mode = str(session_context_mode or "none").lower()
        if self.session_context_mode not in {"none", "summary"}:
            raise ValueError("session_context_mode must be 'none' or 'summary'.")
        self.session_summary_dim = 4
        self.horizon_set = [float(v) for v in horizon_set] if horizon_set else None
        if self.multi_horizon and not self.horizon_set:
            raise ValueError("multi_horizon=True requires horizon_set.")
        self.horizon_seconds_list = self.horizon_set if self.multi_horizon else [self.horizon_seconds]
        self.horizon_steps_list = [max(1, int(round(v / self.sampling_interval))) for v in self.horizon_seconds_list]
        self.output_dim = len(self.horizon_steps_list)

        calibration_input_dim = self.head_dim + 1
        if self.calibration_encoder_mode == "tcn_transformer":
            self.calibration_encoder = LCBranchTCN(
                input_dim=calibration_input_dim,
                d_model=self.d_model,
                dilations=calib_dilations,
                kernel_size=kernel_size,
                dropout=self.dropout,
            )
            self.calibration_cls_token = None
            self.calibration_position_embedding = None
        elif self.calibration_encoder_mode in {"deep_tcn", "deep_tcn_transformer"}:
            self.calibration_encoder = DeepTCNEncoder(
                input_dim=calibration_input_dim,
                hidden_dim=self.d_model,
                dilations=calib_dilations,
                kernel_size=kernel_size,
                dropout=self.dropout,
            )
            self.calibration_cls_token = None
            self.calibration_position_embedding = None
        else:
            self.calibration_encoder = nn.Sequential(
                nn.Linear(calibration_input_dim, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
                nn.Dropout(self.dropout),
            )
            cls_steps = 1 if self.calibration_encoder_mode == "transformer_cls" else 0
            self.calibration_cls_token = (
                nn.Parameter(torch.zeros(1, 1, self.d_model))
                if self.calibration_encoder_mode == "transformer_cls"
                else None
            )
            self.calibration_position_embedding = nn.Parameter(
                torch.zeros(1, int(self.calibration_steps) + cls_steps, self.d_model)
            )
            nn.init.normal_(self.calibration_position_embedding, mean=0.0, std=0.02)
            if self.calibration_cls_token is not None:
                nn.init.normal_(self.calibration_cls_token, mean=0.0, std=0.02)
        if self.calibration_encoder_mode == "deep_tcn":
            self.calibration_transformer = nn.Identity()
        else:
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=int(transformer_heads),
                dim_feedforward=int(transformer_ff_dim),
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.calibration_transformer = nn.TransformerEncoder(transformer_layer, num_layers=int(transformer_layers))
        self.calibration_pool = SequencePooling(self.d_model, pooling)
        self.init_state = nn.Sequential(nn.Linear(self.d_model, self.hidden_dim), nn.Tanh())
        self.calib_summary_encoder = (
            nn.Sequential(
                nn.Linear(11, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
                nn.Dropout(self.dropout),
            )
            if self.calib_summary_features
            else None
        )
        self.motion_input = nn.Sequential(
            nn.Linear(
                self.head_dim + self.motion_feature_dim + self.stream_time_feature_dim + self.state_feedback_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
        )
        self.stream_gru = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.stream_tcn = (
            nn.Sequential(
                TCNBlock(self.hidden_dim, dilation=1, dropout=self.dropout, kernel_size=kernel_size),
                TCNBlock(self.hidden_dim, dilation=2, dropout=self.dropout, kernel_size=kernel_size),
                TCNBlock(self.hidden_dim, dilation=4, dropout=self.dropout, kernel_size=kernel_size),
                TCNBlock(self.hidden_dim, dilation=8, dropout=self.dropout, kernel_size=kernel_size),
            )
            if self.stream_context_mode in {"gru_tcn", "gru_tcn_multiscale"}
            else None
        )
        self.multiscale_state_projection = (
            nn.Sequential(
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
                nn.Dropout(self.dropout),
            )
            if self.stream_context_mode in {"gru_multiscale", "gru_tcn_multiscale"}
            else None
        )
        hidden_static = int(static_hidden_dim if static_hidden_dim is not None else self.d_model)
        self.static_encoder = StaticEncoder(self.static_dim, hidden_static, static_dropout) if self.use_static else None
        self.static_projection = (
            nn.Sequential(nn.Linear(hidden_static, self.d_model), nn.GELU(), nn.LayerNorm(self.d_model))
            if self.use_static and hidden_static != self.d_model
            else None
        )
        self.session_summary_head = (
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.session_summary_dim),
            )
            if self.session_context_mode == "summary"
            else None
        )
        self.session_summary_encoder = (
            nn.Sequential(
                nn.Linear(self.session_summary_dim, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
                nn.Dropout(self.dropout),
            )
            if self.session_context_mode == "summary"
            else None
        )
        self.horizon_encoder = nn.Sequential(nn.Linear(1, self.d_model), nn.GELU(), nn.LayerNorm(self.d_model))
        static_context_dim = self.d_model if self.use_static else 0
        session_context_dim = self.d_model if self.session_context_mode == "summary" else 0
        fusion_dim = self.hidden_dim + self.d_model + self.d_model + static_context_dim + session_context_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.future_head = nn.Linear(self.d_model, 1)
        self.future_horizon_heads = (
            nn.ModuleList([nn.Linear(self.d_model, 1) for _ in self.horizon_seconds_list])
            if self.multi_horizon and self.per_horizon_heads
            else None
        )
        self.delta_head = (
            nn.Linear(self.d_model, 1)
            if self.forecast_head_mode in {"delta", "self_delta", "recent_start_delta", "rollin_start_delta"}
            else None
        )
        self.delta_horizon_heads = (
            nn.ModuleList([nn.Linear(self.d_model, 1) for _ in self.horizon_seconds_list])
            if self.forecast_head_mode in {"delta", "self_delta", "recent_start_delta", "rollin_start_delta"}
            and self.multi_horizon
            and self.per_horizon_heads
            else None
        )
        self.current_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 1),
        )
        self.recent_rf_steps = 1
        self.recent_rf_seconds = self.sampling_interval

    def _append_stream_time_features(self, motion: torch.Tensor, start: int) -> torch.Tensor:
        post_motion = motion[:, int(start) :]
        if not self.stream_time_features:
            return post_motion
        post_steps = int(post_motion.shape[1])
        positions = torch.arange(post_steps, dtype=motion.dtype, device=motion.device)
        absolute_minutes = (float(start) + positions) * self.sampling_interval / 60.0
        since_calib_minutes = positions * self.sampling_interval / 60.0
        time_features = torch.stack([absolute_minutes, since_calib_minutes], dim=-1).unsqueeze(0)
        time_features = time_features.expand(motion.shape[0], -1, -1)
        return torch.cat([post_motion, time_features], dim=-1)

    def _calibration_summary_features(self, head_calib: torch.Tensor, fms_calib: torch.Tensor) -> torch.Tensor:
        fms_first = fms_calib[:, 0]
        fms_last = fms_calib[:, -1]
        fms_mean = fms_calib.mean(dim=1)
        fms_std = fms_calib.std(dim=1, unbiased=False)
        fms_min = fms_calib.min(dim=1).values
        fms_max = fms_calib.max(dim=1).values
        fms_slope = (fms_last - fms_first) / max(float(self.calibration_steps - 1), 1.0)
        accel = head_calib[..., :3]
        gyro = head_calib[..., 3:6] if head_calib.shape[-1] >= 6 else head_calib[..., :3]
        accel_norm = torch.linalg.vector_norm(accel, dim=-1)
        gyro_norm = torch.linalg.vector_norm(gyro, dim=-1)
        return torch.stack(
            [
                fms_first,
                fms_last,
                fms_mean,
                fms_std,
                fms_min,
                fms_max,
                fms_slope,
                accel_norm.mean(dim=1),
                accel_norm.std(dim=1, unbiased=False),
                gyro_norm.mean(dim=1),
                gyro_norm.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )

    def _stream_context(self, state_seq: torch.Tensor) -> torch.Tensor:
        if self.stream_tcn is not None:
            state_seq = self.stream_tcn(state_seq)
        if self.multiscale_state_projection is None:
            return state_seq
        windows = [
            state_seq,
            _causal_rolling_mean(state_seq, max(1, int(round(5.0 / self.sampling_interval)))),
            _causal_rolling_mean(state_seq, max(1, int(round(15.0 / self.sampling_interval)))),
            _causal_rolling_mean(state_seq, max(1, int(round(30.0 / self.sampling_interval)))),
        ]
        return self.multiscale_state_projection(torch.cat(windows, dim=-1))

    def _encode_calibration(self, head: torch.Tensor, fms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if head.shape[1] < self.calibration_steps:
            raise ValueError("head sequence shorter than calibration_steps")
        context_fms = calibration_context_fms(fms, self.calibration_steps, self.fms_context_mode)
        calib_seq = torch.cat([head[:, : self.calibration_steps], context_fms.unsqueeze(-1)], dim=-1)
        z = self.calibration_encoder(calib_seq)
        if self.calibration_position_embedding is not None:
            if self.calibration_cls_token is not None:
                cls = self.calibration_cls_token.to(dtype=z.dtype, device=z.device).expand(z.shape[0], -1, -1)
                z = torch.cat([cls, z], dim=1)
            if z.shape[1] > self.calibration_position_embedding.shape[1]:
                raise ValueError(
                    f"calibration sequence length {z.shape[1]} exceeds positional embedding length "
                    f"{self.calibration_position_embedding.shape[1]}."
                )
            z = z + self.calibration_position_embedding[:, : z.shape[1]].to(dtype=z.dtype, device=z.device)
        z = self.calibration_transformer(z)
        if self.calibration_encoder_mode == "transformer_cls":
            z_calib = z[:, 0]
        else:
            z_calib = self.calibration_pool(z)
        if self.calib_summary_encoder is not None:
            summary = self._calibration_summary_features(head[:, : self.calibration_steps], context_fms)
            z_calib = z_calib + self.calib_summary_encoder(summary)
        return z_calib, context_fms[:, -1]

    def _run_stream(
        self,
        stream_motion: torch.Tensor,
        h0: torch.Tensor,
        z_calib: torch.Tensor,
        base_fms: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.state_feedback_mode == "none":
            stream = self.motion_input(stream_motion)
            state_seq, _ = self.stream_gru(stream, h0)
            return state_seq, None

        states: List[torch.Tensor] = []
        current_preds: List[torch.Tensor] = []
        hidden = h0
        prev_current = base_fms
        z_expand = z_calib
        for step_idx in range(stream_motion.shape[1]):
            feedback = prev_current.view(prev_current.shape[0], 1, 1)
            step_motion = torch.cat([stream_motion[:, step_idx : step_idx + 1], feedback], dim=-1)
            stream = self.motion_input(step_motion)
            out, hidden = self.stream_gru(stream, hidden)
            state = out[:, 0]
            current_raw = self.current_head(torch.cat([state, z_expand], dim=-1)).squeeze(-1)
            prev_current = torch.sigmoid(current_raw)
            states.append(state)
            current_preds.append(prev_current)
        return torch.stack(states, dim=1), torch.stack(current_preds, dim=1)

    def _encode_static(self, static: Optional[torch.Tensor], pred_steps: int, batch_size: int) -> Optional[torch.Tensor]:
        if not self.use_static:
            return None
        if static is None:
            raise ValueError("CalibInitStateForecaster was created with use_static=True, but static tensor was not provided.")
        if static.ndim != 2 or static.shape[0] != batch_size or static.shape[1] != self.static_dim:
            raise ValueError(f"static must be [B,{self.static_dim}], got {static.shape}")
        assert self.static_encoder is not None
        z_static = self.static_encoder(static)
        if self.static_projection is not None:
            z_static = self.static_projection(z_static)
        return z_static.unsqueeze(1).expand(-1, pred_steps, -1)

    def _future_from_fused(
        self,
        fused: torch.Tensor,
        base_fms: torch.Tensor,
        horizon_idx: Optional[int],
        current_base: Optional[torch.Tensor] = None,
        recent_start_base: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        level_raw = (
            self.future_horizon_heads[int(horizon_idx)](fused).squeeze(-1)
            if self.future_horizon_heads is not None and horizon_idx is not None
            else self.future_head(fused).squeeze(-1)
        )
        level_pred = torch.sigmoid(level_raw)
        if self.forecast_head_mode == "level":
            return level_pred, {"future_level": level_pred}
        assert self.delta_head is not None
        delta_raw = (
            self.delta_horizon_heads[int(horizon_idx)](fused).squeeze(-1)
            if self.delta_horizon_heads is not None and horizon_idx is not None
            else self.delta_head(fused).squeeze(-1)
        )
        delta_pred = self.delta_scale * torch.tanh(delta_raw)
        if self.forecast_head_mode == "self_delta":
            if current_base is None:
                raise ValueError("forecast_head_mode=self_delta requires current_base.")
            base = current_base
        elif self.forecast_head_mode in {"recent_start_delta", "rollin_start_delta"}:
            if recent_start_base is None:
                raise ValueError(f"forecast_head_mode={self.forecast_head_mode} requires recent_start_base.")
            base = recent_start_base
        else:
            base = base_fms.unsqueeze(1).expand_as(delta_pred)
        pred = torch.clamp(base + delta_pred, 0.0, 1.0)
        return pred, {
            "future_level": level_pred,
            "future_delta_pred": delta_pred,
            "future_delta_value": pred,
            "future_delta_base": base,
        }

    def forward(
        self,
        head: torch.Tensor,
        y_calib: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        static: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        assert head.ndim == 3, f"head must be [B,T,6], got {head.shape}"
        if head.shape[-1] != self.head_dim:
            raise ValueError(f"expected head_dim={self.head_dim}, got {head.shape[-1]}")
        bsz, steps, _ = head.shape
        device = head.device
        lengths = lengths.to(device) if lengths is not None else torch.full((bsz,), steps, dtype=torch.long, device=device)
        if y_calib.ndim != 2:
            raise ValueError(f"FMS input must be [B,T] or [B,C], got {y_calib.shape}")
        z_calib, base_fms = self._encode_calibration(head, y_calib.to(device))
        session_summary = None
        session_context = None
        if self.session_summary_head is not None:
            assert self.session_summary_encoder is not None
            session_summary = torch.sigmoid(self.session_summary_head(z_calib))
            session_context = self.session_summary_encoder(session_summary)

        start = self.calibration_steps
        max_horizon_steps = max(self.horizon_steps_list)
        max_forecast_t = int(torch.clamp(lengths.max() - max_horizon_steps, min=start).item())
        pred_steps = max(0, max_forecast_t - start)
        positions = start + torch.arange(pred_steps, device=device, dtype=torch.long)
        if self.multi_horizon:
            horizons = torch.tensor(self.horizon_steps_list, device=device, dtype=torch.long)
            mask = (positions.view(1, -1, 1) + horizons.view(1, 1, -1)) < lengths.view(-1, 1, 1)
        else:
            mask = (positions.view(1, -1) + self.horizon_steps) < lengths.view(-1, 1)
        if pred_steps == 0:
            empty = head.new_zeros((bsz, 0, self.output_dim)) if self.multi_horizon else head.new_zeros((bsz, 0))
            return {
                "future": empty,
                "mask": mask,
                "prediction_start": torch.tensor(start, device=device),
                "horizon_steps_list": torch.tensor(self.horizon_steps_list, dtype=torch.long, device=device),
                "use_static": torch.tensor(self.use_static, device=device),
            }

        motion = append_motion_features(head, self.motion_feature_mode)
        stream_motion = self._append_stream_time_features(motion, start)
        h0 = self.init_state(z_calib).unsqueeze(0)
        state_seq, feedback_current_pred = self._run_stream(stream_motion, h0, z_calib, base_fms)
        state_seq = self._stream_context(state_seq)
        state_at_positions = state_seq[:, :pred_steps]
        z_static = self._encode_static(static, pred_steps, bsz)
        z_session = session_context.unsqueeze(1).expand(-1, pred_steps, -1) if session_context is not None else None
        if feedback_current_pred is not None:
            full_current_pred = feedback_current_pred
        else:
            full_current_raw = self.current_head(
                torch.cat(
                    [state_seq, z_calib.unsqueeze(1).expand(-1, state_seq.shape[1], -1)],
                    dim=-1,
                )
            ).squeeze(-1)
            full_current_pred = torch.sigmoid(full_current_raw)
        current_pred = full_current_pred[:, :pred_steps]

        recent_start_base = None
        synthetic_anchor_idx = None
        synthetic_anchor_is_predicted = None
        if self.forecast_head_mode in {"recent_start_delta", "rollin_start_delta"}:
            synthetic_anchor_idx = (positions - self.recent_steps + 1).clamp_min(0).clamp_max(positions)

        horizon_fused_values: List[Tuple[int, torch.Tensor]] = []
        for horizon_idx, horizon_seconds in enumerate(self.horizon_seconds_list):
            horizon_in = head.new_full((bsz, pred_steps, 1), float(horizon_seconds) / 60.0)
            parts = [
                state_at_positions,
                z_calib.unsqueeze(1).expand(-1, pred_steps, -1),
                self.horizon_encoder(horizon_in),
            ]
            if z_static is not None:
                parts.append(z_static)
            if z_session is not None:
                parts.append(z_session)
            fused = self.fusion(torch.cat(parts, dim=-1))
            horizon_fused_values.append((horizon_idx, fused))
            if not self.multi_horizon:
                break

        if self.forecast_head_mode == "recent_start_delta":
            assert synthetic_anchor_idx is not None
            post_calib_anchor = synthetic_anchor_idx >= self.calibration_steps
            anchor_offsets = (synthetic_anchor_idx - self.calibration_steps).clamp_min(0)
            anchor_offsets = anchor_offsets.clamp_max(max(0, int(full_current_pred.shape[1]) - 1))
            predicted_anchor = full_current_pred.index_select(1, anchor_offsets)
            calibration_anchor = base_fms.unsqueeze(1).expand(-1, pred_steps)
            recent_start_base = torch.where(post_calib_anchor.view(1, -1), predicted_anchor, calibration_anchor)
            synthetic_anchor_is_predicted = post_calib_anchor.view(1, -1).expand(bsz, -1)
        elif self.forecast_head_mode == "rollin_start_delta":
            assert synthetic_anchor_idx is not None
            h5_idx = min(range(len(self.horizon_steps_list)), key=lambda idx: abs(self.horizon_seconds_list[idx] - 5.0))
            h5_steps = int(self.horizon_steps_list[h5_idx])
            h5_fused = horizon_fused_values[h5_idx][1] if self.multi_horizon else horizon_fused_values[0][1]
            h5_head = (
                self.future_horizon_heads[h5_idx]
                if self.future_horizon_heads is not None and self.multi_horizon
                else self.future_head
            )
            h5_level = torch.sigmoid(h5_head(h5_fused).squeeze(-1))
            rollin_source_idx = synthetic_anchor_idx - h5_steps
            rollin_offsets = rollin_source_idx - self.calibration_steps
            rollin_available = rollin_offsets >= 0
            safe_rollin_offsets = rollin_offsets.clamp_min(0).clamp_max(max(0, int(h5_level.shape[1]) - 1))
            predicted_anchor = h5_level.index_select(1, safe_rollin_offsets)
            calibration_idx = synthetic_anchor_idx.clamp_max(self.calibration_steps - 1)
            calibration_anchor = y_calib.to(device).index_select(1, calibration_idx)
            recent_start_base = torch.where(rollin_available.view(1, -1), predicted_anchor, calibration_anchor)
            synthetic_anchor_is_predicted = rollin_available.view(1, -1).expand(bsz, -1)

        preds = []
        aux_by_key: Dict[str, List[torch.Tensor]] = {}
        distill_repr_values: List[torch.Tensor] = []
        for horizon_idx, fused in horizon_fused_values:
            distill_repr_values.append(fused)
            pred_h, aux_h = self._future_from_fused(
                fused,
                base_fms,
                horizon_idx if self.multi_horizon else None,
                current_base=current_pred,
                recent_start_base=recent_start_base,
            )
            preds.append(pred_h)
            for key, value in aux_h.items():
                aux_by_key.setdefault(key, []).append(value)
        pred = torch.stack(preds, dim=-1) if self.multi_horizon else preds[0]
        out: Dict[str, torch.Tensor] = {
            "future": pred,
            "mask": mask,
            "prediction_start": torch.tensor(start, device=device),
            "horizon_steps_list": torch.tensor(self.horizon_steps_list, dtype=torch.long, device=device),
            "use_static": torch.tensor(self.use_static, device=device),
            "z_calib_norm": z_calib.norm(dim=-1),
            "state_norm": state_at_positions.norm(dim=-1),
            "current": current_pred,
            "calibration_end_fms": base_fms,
            "state_feedback_mode": self.state_feedback_mode,
            "session_context_mode": self.session_context_mode,
        }
        if recent_start_base is not None and synthetic_anchor_idx is not None:
            out["synthetic_anchor_fms"] = recent_start_base
            out["synthetic_anchor_index"] = synthetic_anchor_idx.view(1, -1).expand(bsz, -1)
            if synthetic_anchor_is_predicted is not None:
                out["synthetic_anchor_is_predicted"] = synthetic_anchor_is_predicted
        if session_summary is not None:
            out["session_summary"] = session_summary
        out["distill_repr"] = torch.stack(distill_repr_values, dim=2) if self.multi_horizon else distill_repr_values[0]
        for key, values in aux_by_key.items():
            out[key] = torch.stack(values, dim=-1) if self.multi_horizon else values[0]
        return out


class OnlineFMSRiskTracker(nn.Module):
    """Calibration-initialized current-FMS and rapid-rise/drop risk tracker.

    The forward path reads FMS only from the calibration window. Post-calibration
    state is updated causally from head motion and optional predicted-current
    feedback.
    """

    def __init__(
        self,
        head_dim: int = 6,
        calibration_steps: int = 180,
        horizon_steps: int = 20,
        recent_steps: int = 20,
        max_time_steps: int = 2048,
        sampling_interval: float = 0.5,
        horizon_seconds: Optional[float] = None,
        rise_horizon_steps: Optional[Sequence[int]] = None,
        rise_thresholds: Optional[Sequence[float]] = None,
        fall_horizon_steps: Optional[Sequence[int]] = None,
        fall_thresholds: Optional[Sequence[float]] = None,
        high_risk_horizon_steps: Optional[Sequence[int]] = None,
        high_risk_thresholds: Optional[Sequence[float]] = None,
        future_aux_horizon_steps: Optional[Sequence[int]] = None,
        d_model: int = 64,
        hidden_dim: int = 96,
        kernel_size: int = 3,
        dropout: float = 0.08,
        calib_dilations: Sequence[int] = (1, 2, 4, 8, 16),
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 128,
        pooling: str = "mean",
        use_static: bool = False,
        static_dim: int = 4,
        static_hidden_dim: Optional[int] = None,
        static_dropout: float = 0.1,
        motion_feature_mode: str = "norm",
        motion_stats_branch: bool = False,
        stream_time_features: bool = True,
        stream_context_mode: str = "gru_multiscale",
        stream_prepend_calibration: bool = False,
        stream_calib_condition_mode: str = "none",
        stream_calib_condition_strength: float = 0.1,
        calib_summary_features: bool = True,
        calibration_fusion_mode: str = "add",
        calibration_fusion_hidden_dim: Optional[int] = None,
        calibration_fusion_output_dim: Optional[int] = None,
        calibration_encoder_mode: str = "tcn_transformer",
        state_feedback_mode: str = "predicted_current",
        fms_context_mode: str = "calibration_history",
        ordinal_bins: Optional[Sequence[float]] = None,
        fms_combine_weight_ordinal: float = 0.6,
        current_head_mode: str = "basic",
        ordinal_head_mode: Optional[str] = None,
        current_delta_scale: float = 0.75,
        current_anchor_delta_growth_scale: float = 0.0,
        current_anchor_delta_growth_horizon_seconds: float = 90.0,
        current_anchor_delta_growth_power: float = 1.0,
        current_trajectory_offsets: Optional[Sequence[int]] = None,
        current_range_guard_low_threshold: float = 5.0,
        current_range_guard_temperature: float = 1.0,
        current_range_guard_floor: float = 0.10,
        current_range_guard_cap: float = 2.0,
        current_range_guard_cap_strength: float = 1.0,
        motion_encoder_context: str = "linear",
        motion_encoder_layers: int = 0,
        risk_head_enabled: bool = True,
        fall_risk_head_enabled: bool = False,
        high_risk_head_enabled: bool = False,
        risk_temporal_context: str = "none",
        risk_temporal_layers: int = 0,
        coarse_band_bins: Optional[Sequence[float]] = None,
        coarse_residual_head_enabled: bool = False,
        coarse_residual_range: float = 3.0,
        coarse_residual_combine_weight: float = 0.0,
        regime_head_enabled: bool = False,
        regime_class_count: int = 5,
        uncertainty_head_enabled: bool = False,
        uncertainty_min_log_sigma: float = -5.0,
        uncertainty_max_log_sigma: float = 1.0,
        deep_tcn_dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        calibration_tcn_adaptive_dilations: bool = False,
        calibration_tcn_max_padding_steps: int = 8,
        calibration_tcn_max_padding_fraction: float = 0.1,
        decoder_hidden_dim: Optional[int] = None,
        decoder_context_mode: str = "fused",
        decoder_temporal_context: str = "none",
        decoder_temporal_layers: int = 0,
        fds_enabled: bool = False,
        fds_min: float = 0.0,
        fds_max: float = 20.0,
        fds_bin_size: float = 1.0,
        fds_num_bins: int = 21,
        fds_kernel: str = "gaussian",
        fds_kernel_size: int = 5,
        fds_sigma: float = 2.0,
        fds_momentum: float = 0.9,
        fds_blend: float = 1.0,
        calib_fms_dropout: float = 0.0,
        calibration_end_fms_dropout: float = 0.0,
        current_session_affine_head_enabled: bool = False,
        current_session_affine_hidden_dim: Optional[int] = None,
        current_session_affine_scale_range: float = 0.25,
        current_session_affine_bias_range: float = 0.15,
        current_affine_head_enabled: bool = False,
        current_affine_hidden_dim: Optional[int] = None,
        current_affine_scale_range: float = 0.5,
        current_affine_bias_range: float = 0.25,
        current_binned_affine_head_enabled: bool = False,
        current_binned_affine_anchor_bins: Optional[Sequence[float]] = None,
        current_binned_affine_pred_bins: Optional[Sequence[float]] = None,
        current_binned_affine_time_bins: Optional[Sequence[float]] = None,
        current_binned_affine_scale_range: float = 1.5,
        current_binned_affine_bias_range: float = 0.5,
        calibration_residual_adapter_enabled: bool = False,
        calibration_residual_feature_dim: int = 0,
        calibration_residual_adapter_hidden_dim: Optional[int] = None,
        calibration_residual_adapter_mode: str = "mlp",
        calibration_residual_delta_range: float = 0.15,
        calibration_residual_decay_seconds: float = 120.0,
        calibration_residual_gate_low_threshold: float = 8.0,
        calibration_residual_gate_high_threshold: float = 10.0,
        calibration_residual_gate_anchor_threshold: float = 10.0,
        calibration_residual_gate_temperature: float = 1.0,
        calibration_summary_fusion_enabled: bool = False,
        calibration_summary_fusion_feature_dim: int = 0,
        calibration_summary_fusion_hidden_dim: Optional[int] = None,
        calibration_summary_fusion_mode: str = "additive_gated",
        calibration_summary_fusion_strength: float = 1.0,
        current_low_suppressor_enabled: bool = False,
        current_low_suppressor_hidden_dim: Optional[int] = None,
        current_low_suppressor_delta_range: float = 0.25,
        current_low_suppressor_gate_init_bias: float = -6.0,
        **_: Any,
    ):
        super().__init__()
        if d_model % transformer_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by transformer_heads={transformer_heads}")
        self.head_dim = int(head_dim)
        self.calibration_steps = int(calibration_steps)
        self.horizon_steps = int(horizon_steps)
        self.recent_steps = int(recent_steps)
        self.max_time_steps = int(max_time_steps)
        self.sampling_interval = float(sampling_interval)
        self.horizon_seconds = float(horizon_seconds if horizon_seconds is not None else self.horizon_steps * self.sampling_interval)
        self.rise_horizon_steps = [int(v) for v in (rise_horizon_steps or [self.horizon_steps])]
        self.rise_thresholds = [float(v) for v in (rise_thresholds or [2.0 for _ in self.rise_horizon_steps])]
        if len(self.rise_thresholds) != len(self.rise_horizon_steps):
            raise ValueError("rise_thresholds must have the same length as rise_horizon_steps.")
        self.fall_horizon_steps = [int(v) for v in (fall_horizon_steps or self.rise_horizon_steps)]
        self.fall_thresholds = [float(v) for v in (fall_thresholds or self.rise_thresholds)]
        if len(self.fall_thresholds) != len(self.fall_horizon_steps):
            raise ValueError("fall_thresholds must have the same length as fall_horizon_steps.")
        self.high_risk_horizon_steps = [int(v) for v in (high_risk_horizon_steps or [])]
        if any(v <= 0 for v in self.high_risk_horizon_steps):
            raise ValueError("high_risk_horizon_steps must contain only positive step counts.")
        self.high_risk_thresholds = [float(v) for v in (high_risk_thresholds or [])]
        if any(v < 0.0 or v > 20.0 for v in self.high_risk_thresholds):
            raise ValueError("high_risk_thresholds must be on the raw DenseFMS 0-20 scale.")
        if bool(high_risk_head_enabled) and (not self.high_risk_horizon_steps or not self.high_risk_thresholds):
            raise ValueError("high_risk_head_enabled requires non-empty high_risk_horizon_steps and high_risk_thresholds.")
        self.future_aux_horizon_steps = [int(v) for v in (future_aux_horizon_steps or [])]
        if any(v <= 0 for v in self.future_aux_horizon_steps):
            raise ValueError("future_aux_horizon_steps must contain only positive step counts.")
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.pooling = str(pooling)
        self.use_static = bool(use_static)
        self.static_dim = int(static_dim)
        self.motion_feature_mode = str(motion_feature_mode or "none").lower()
        self.motion_feature_dim = _motion_feature_dim(self.motion_feature_mode)
        self.motion_stats_branch = bool(motion_stats_branch)
        self.stream_time_features = bool(stream_time_features)
        self.stream_time_feature_dim = 2 if self.stream_time_features else 0
        self.stream_context_mode = str(stream_context_mode or "gru").lower()
        self.stream_prepend_calibration = bool(stream_prepend_calibration)
        self.stream_calib_condition_mode = str(stream_calib_condition_mode or "none").lower()
        if self.stream_calib_condition_mode not in {"none", "film"}:
            raise ValueError("stream_calib_condition_mode must be 'none' or 'film'.")
        self.stream_calib_condition_strength = max(0.0, float(stream_calib_condition_strength))
        if self.stream_context_mode not in {
            "gru",
            "gru_multiscale",
            "gru_tcn",
            "gru_tcn_multiscale",
            "deep_tcn",
            "deep_tcn_latent_gru",
            "transformer",
            "transformer_latent_gru",
        }:
            raise ValueError(
                "stream_context_mode must be 'gru', 'gru_multiscale', 'gru_tcn', 'gru_tcn_multiscale', "
                "'deep_tcn', 'deep_tcn_latent_gru', 'transformer', or 'transformer_latent_gru'."
            )
        self.calib_summary_features = bool(calib_summary_features)
        self.calibration_fusion_mode = str(calibration_fusion_mode or "add").lower()
        if self.calibration_fusion_mode not in {
            "add",
            "mean_last_summary_concat",
            "mean_last_gated_summary",
            "mean_last_attention_summary",
            "mean_last_event_attention_summary",
        }:
            raise ValueError(
                "calibration_fusion_mode must be one of 'add', 'mean_last_summary_concat', "
                "'mean_last_gated_summary', 'mean_last_attention_summary', or "
                "'mean_last_event_attention_summary'."
            )
        if self.calibration_fusion_mode == "add":
            self.calibration_repr_dim = self.d_model
        else:
            self.calibration_repr_dim = int(
                calibration_fusion_output_dim
                if calibration_fusion_output_dim is not None and int(calibration_fusion_output_dim) > 0
                else self.hidden_dim
            )
        self.calibration_encoder_mode = str(calibration_encoder_mode or "tcn_transformer").lower()
        if self.calibration_encoder_mode not in {
            "tcn_transformer",
            "transformer",
            "transformer_cls",
            "deep_tcn",
            "deep_tcn_transformer",
        }:
            raise ValueError(
                "calibration_encoder_mode must be 'tcn_transformer', 'transformer', 'transformer_cls', "
                "'deep_tcn', or 'deep_tcn_transformer'."
            )
        self.state_feedback_mode = str(state_feedback_mode or "none").lower()
        if self.state_feedback_mode not in {"none", "predicted_current"}:
            raise ValueError("state_feedback_mode must be 'none' or 'predicted_current'.")
        if self.stream_context_mode == "deep_tcn" and self.state_feedback_mode != "none":
            raise ValueError("stream_context_mode='deep_tcn' requires state_feedback_mode='none'.")
        if self.stream_context_mode == "transformer" and self.state_feedback_mode != "none":
            raise ValueError("stream_context_mode='transformer' requires state_feedback_mode='none'.")
        self.state_feedback_dim = 1 if self.state_feedback_mode == "predicted_current" else 0
        self.deep_tcn_dilations = [int(v) for v in deep_tcn_dilations]
        self.calibration_tcn_adaptive_dilations = bool(calibration_tcn_adaptive_dilations)
        self.calibration_tcn_max_padding_steps = max(0, int(calibration_tcn_max_padding_steps))
        self.calibration_tcn_max_padding_fraction = max(0.0, float(calibration_tcn_max_padding_fraction))
        if self.calibration_steps <= 0:
            self.calibration_deep_tcn_dilations = []
            self.calibration_tcn_rf_steps = 0
            self.calibration_tcn_pad_steps = 0
        elif self.calibration_encoder_mode in {"deep_tcn", "deep_tcn_transformer"}:
            if self.calibration_tcn_adaptive_dilations:
                (
                    self.calibration_deep_tcn_dilations,
                    self.calibration_tcn_rf_steps,
                    self.calibration_tcn_pad_steps,
                ) = adaptive_tcn_dilations_for_sequence(
                    self.deep_tcn_dilations,
                    self.calibration_steps,
                    kernel_size=kernel_size,
                    max_padding_steps=self.calibration_tcn_max_padding_steps,
                    max_padding_fraction=self.calibration_tcn_max_padding_fraction,
                )
            else:
                self.calibration_deep_tcn_dilations = list(self.deep_tcn_dilations)
                self.calibration_tcn_rf_steps = tcn_receptive_field_steps(
                    self.calibration_deep_tcn_dilations,
                    kernel_size=kernel_size,
                )
                self.calibration_tcn_pad_steps = max(0, self.calibration_tcn_rf_steps - self.calibration_steps)
        else:
            self.calibration_deep_tcn_dilations = []
            self.calibration_tcn_rf_steps = 0
            self.calibration_tcn_pad_steps = 0
        self.motion_encoder_context = str(motion_encoder_context or "linear").lower()
        if self.motion_encoder_context not in {"linear", "tcn"}:
            raise ValueError("motion_encoder_context must be 'linear' or 'tcn'.")
        self.motion_encoder_layers = max(0, int(motion_encoder_layers))
        self.fms_context_mode = normalize_fms_context_mode(fms_context_mode)
        if self.fms_context_mode not in {"calibration_history", "none"}:
            raise ValueError("OnlineFMSRiskTracker allows only calibration_history or none FMS context.")
        bins = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0] if ordinal_bins is None else [float(v) for v in ordinal_bins]
        if len(bins) < 2 or any(bins[i] >= bins[i + 1] for i in range(len(bins) - 1)):
            raise ValueError("ordinal_bins must be a strictly increasing sequence.")
        self.ordinal_bins_raw = bins
        self.register_buffer("ordinal_bins_norm", torch.tensor([v / 20.0 for v in bins], dtype=torch.float32))
        self.fms_combine_weight_ordinal = float(fms_combine_weight_ordinal)
        self.current_head_mode = normalize_current_head_mode(current_head_mode)
        self.ordinal_head_mode = normalize_ordinal_head_mode(ordinal_head_mode, self.current_head_mode)
        if self.calibration_steps <= 0:
            if self.fms_context_mode != "none":
                raise ValueError("calibration_steps=0 requires fms_context_mode='none'.")
        self.current_delta_scale = float(current_delta_scale)
        self.current_anchor_delta_growth_scale = max(0.0, float(current_anchor_delta_growth_scale))
        self.current_anchor_delta_growth_horizon_seconds = max(float(current_anchor_delta_growth_horizon_seconds), 1e-6)
        self.current_anchor_delta_growth_power = max(float(current_anchor_delta_growth_power), 1e-6)
        self.current_range_guard_low_threshold = float(current_range_guard_low_threshold)
        self.current_range_guard_temperature = max(float(current_range_guard_temperature), 1e-6)
        self.current_range_guard_floor = max(0.0, min(1.0, float(current_range_guard_floor)))
        self.current_range_guard_cap = max(0.0, float(current_range_guard_cap))
        self.current_range_guard_cap_strength = max(0.0, min(1.0, float(current_range_guard_cap_strength)))
        trajectory_offsets = [int(v) for v in (current_trajectory_offsets or [0, 5, 10, 20])]
        if 0 not in trajectory_offsets:
            trajectory_offsets = [0, *trajectory_offsets]
        if len(set(trajectory_offsets)) != len(trajectory_offsets):
            raise ValueError("current_trajectory_offsets must not contain duplicate offsets.")
        self.current_trajectory_offsets = trajectory_offsets
        self.current_trajectory_zero_index = self.current_trajectory_offsets.index(0)
        self.decoder_hidden_dim = None if decoder_hidden_dim is None else int(decoder_hidden_dim)
        if self.decoder_hidden_dim is not None and self.decoder_hidden_dim <= 0:
            self.decoder_hidden_dim = None
        self.decoder_context_mode = str(decoder_context_mode or "fused").lower()
        if self.decoder_context_mode not in {"fused", "state"}:
            raise ValueError("decoder_context_mode must be 'fused' or 'state'.")
        self.decoder_temporal_context = str(decoder_temporal_context or "none").lower()
        if self.decoder_temporal_context not in {"none", "tcn"}:
            raise ValueError("decoder_temporal_context must be 'none' or 'tcn'.")
        self.decoder_temporal_layers = max(0, int(decoder_temporal_layers))
        self.fds_enabled = bool(fds_enabled)
        self.fds_min = float(fds_min)
        self.fds_max = float(fds_max)
        self.fds_bin_size = float(fds_bin_size)
        self.fds_num_bins = int(fds_num_bins)
        self.fds_kernel = str(fds_kernel or "gaussian")
        self.fds_kernel_size = int(fds_kernel_size)
        self.fds_sigma = float(fds_sigma)
        self.fds_momentum = float(fds_momentum)
        self.fds_blend = float(fds_blend)
        self.risk_head_enabled = bool(risk_head_enabled)
        self.fall_risk_head_enabled = bool(fall_risk_head_enabled)
        self.high_risk_head_enabled = bool(high_risk_head_enabled)
        self.risk_temporal_context = str(risk_temporal_context or "none").lower()
        if self.risk_temporal_context not in {"none", "tcn"}:
            raise ValueError("risk_temporal_context must be 'none' or 'tcn'.")
        self.risk_temporal_layers = max(0, int(risk_temporal_layers))
        self.coarse_band_bins_raw = [float(v) for v in (coarse_band_bins or [])]
        if any(v <= 0.0 or v >= 20.0 for v in self.coarse_band_bins_raw) or any(
            self.coarse_band_bins_raw[i] >= self.coarse_band_bins_raw[i + 1]
            for i in range(len(self.coarse_band_bins_raw) - 1)
        ):
            raise ValueError("coarse_band_bins must be strictly increasing thresholds inside (0, 20).")
        self.register_buffer(
            "coarse_band_bins_norm",
            torch.tensor([v / 20.0 for v in self.coarse_band_bins_raw], dtype=torch.float32),
        )
        coarse_edges = [0.0, *self.coarse_band_bins_raw, 20.0] if self.coarse_band_bins_raw else []
        self.register_buffer(
            "coarse_band_centers_norm",
            torch.tensor(
                [0.5 * (coarse_edges[i] + coarse_edges[i + 1]) / 20.0 for i in range(max(0, len(coarse_edges) - 1))],
                dtype=torch.float32,
            ),
        )
        self.coarse_residual_head_enabled = bool(coarse_residual_head_enabled)
        if self.coarse_residual_head_enabled and not self.coarse_band_bins_raw:
            raise ValueError("coarse_residual_head_enabled requires non-empty coarse_band_bins.")
        self.coarse_residual_range = max(0.0, float(coarse_residual_range)) / 20.0
        self.coarse_residual_combine_weight = max(0.0, min(1.0, float(coarse_residual_combine_weight)))
        self.regime_head_enabled = bool(regime_head_enabled)
        self.regime_class_count = int(regime_class_count)
        if self.regime_class_count < 2:
            raise ValueError("regime_class_count must be >= 2.")
        self.uncertainty_head_enabled = bool(uncertainty_head_enabled)
        self.uncertainty_min_log_sigma = float(uncertainty_min_log_sigma)
        self.uncertainty_max_log_sigma = float(uncertainty_max_log_sigma)
        if self.uncertainty_min_log_sigma >= self.uncertainty_max_log_sigma:
            raise ValueError("uncertainty_min_log_sigma must be < uncertainty_max_log_sigma.")
        self.calib_fms_dropout = max(0.0, min(1.0, float(calib_fms_dropout)))
        self.calibration_end_fms_dropout = max(0.0, min(1.0, float(calibration_end_fms_dropout)))
        self.current_session_affine_head_enabled = bool(current_session_affine_head_enabled)
        self.current_session_affine_scale_range = max(0.0, float(current_session_affine_scale_range))
        self.current_session_affine_bias_range = max(0.0, float(current_session_affine_bias_range))
        self.current_affine_head_enabled = bool(current_affine_head_enabled)
        self.current_affine_scale_range = max(0.0, float(current_affine_scale_range))
        self.current_affine_bias_range = max(0.0, float(current_affine_bias_range))
        self.current_binned_affine_head_enabled = bool(current_binned_affine_head_enabled)
        self.current_binned_affine_anchor_bins = [float(v) for v in (current_binned_affine_anchor_bins or [5.0, 10.0])]
        self.current_binned_affine_pred_bins = [float(v) for v in (current_binned_affine_pred_bins or [5.0, 10.0])]
        self.current_binned_affine_time_bins = [float(v) for v in (current_binned_affine_time_bins or [160.0])]
        self.current_binned_affine_scale_range = max(0.0, float(current_binned_affine_scale_range))
        self.current_binned_affine_bias_range = max(0.0, float(current_binned_affine_bias_range))
        self.calibration_residual_adapter_enabled = bool(calibration_residual_adapter_enabled)
        self.calibration_residual_feature_dim = int(calibration_residual_feature_dim or 0)
        self.calibration_residual_adapter_mode = str(calibration_residual_adapter_mode or "mlp").lower()
        self.calibration_residual_delta_range = max(0.0, float(calibration_residual_delta_range))
        self.calibration_residual_decay_seconds = max(float(calibration_residual_decay_seconds), 1e-6)
        self.calibration_residual_gate_low_threshold = float(calibration_residual_gate_low_threshold)
        self.calibration_residual_gate_high_threshold = float(calibration_residual_gate_high_threshold)
        self.calibration_residual_gate_anchor_threshold = float(calibration_residual_gate_anchor_threshold)
        self.calibration_residual_gate_temperature = max(float(calibration_residual_gate_temperature), 1e-6)
        self.calibration_summary_fusion_enabled = bool(calibration_summary_fusion_enabled)
        self.calibration_summary_fusion_feature_dim = int(calibration_summary_fusion_feature_dim or 0)
        if self.calibration_summary_fusion_enabled and self.calibration_summary_fusion_feature_dim <= 0:
            self.calibration_summary_fusion_feature_dim = self.calibration_residual_feature_dim
        self.calibration_summary_fusion_mode = str(calibration_summary_fusion_mode or "additive_gated").lower()
        self.calibration_summary_fusion_strength = max(0.0, float(calibration_summary_fusion_strength))
        self.current_low_suppressor_enabled = bool(current_low_suppressor_enabled)
        self.current_low_suppressor_delta_range = max(0.0, float(current_low_suppressor_delta_range))
        self.current_low_suppressor_gate_init_bias = float(current_low_suppressor_gate_init_bias)
        self.requires_full_fms = False
        self.multi_horizon = False
        self.output_dim = 1

        calibration_input_dim = self.head_dim + 1
        if self.calibration_steps <= 0:
            self.calibration_encoder = nn.Identity()
            self.calibration_cls_token = None
            self.calibration_position_embedding = None
        elif self.calibration_encoder_mode == "tcn_transformer":
            self.calibration_encoder = LCBranchTCN(
                input_dim=calibration_input_dim,
                d_model=self.d_model,
                dilations=calib_dilations,
                kernel_size=kernel_size,
                dropout=self.dropout,
            )
            self.calibration_cls_token = None
            self.calibration_position_embedding = None
        elif self.calibration_encoder_mode in {"deep_tcn", "deep_tcn_transformer"}:
            self.calibration_encoder = DeepTCNEncoder(
                input_dim=calibration_input_dim,
                hidden_dim=self.d_model,
                dilations=self.calibration_deep_tcn_dilations,
                kernel_size=kernel_size,
                dropout=self.dropout,
            )
            self.calibration_cls_token = None
            self.calibration_position_embedding = None
        else:
            self.calibration_encoder = nn.Sequential(
                nn.Linear(calibration_input_dim, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
                nn.Dropout(self.dropout),
            )
            cls_steps = 1 if self.calibration_encoder_mode == "transformer_cls" else 0
            self.calibration_cls_token = (
                nn.Parameter(torch.zeros(1, 1, self.d_model))
                if self.calibration_encoder_mode == "transformer_cls"
                else None
            )
            self.calibration_position_embedding = nn.Parameter(
                torch.zeros(1, int(self.calibration_steps) + cls_steps, self.d_model)
            )
            nn.init.normal_(self.calibration_position_embedding, mean=0.0, std=0.02)
            if self.calibration_cls_token is not None:
                nn.init.normal_(self.calibration_cls_token, mean=0.0, std=0.02)
        if self.calibration_steps <= 0 or self.calibration_encoder_mode == "deep_tcn":
            self.calibration_transformer = nn.Identity()
        else:
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=int(transformer_heads),
                dim_feedforward=int(transformer_ff_dim),
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.calibration_transformer = nn.TransformerEncoder(transformer_layer, num_layers=int(transformer_layers))
        self.calibration_pool = SequencePooling(self.d_model, self.pooling)
        summary_dim = 8 + 2 * self.head_dim
        self.calib_summary_encoder = (
            nn.Sequential(
                nn.Linear(summary_dim, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
                nn.Dropout(self.dropout),
            )
            if self.calib_summary_features
            else None
        )
        self.calibration_attention_pool = (
            SequencePooling(self.d_model, "attention")
            if self.calibration_fusion_mode == "mean_last_attention_summary"
            else None
        )
        self.calibration_event_attention = (
            nn.Sequential(
                nn.Linear(self.d_model * 2 + 3, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, 1),
            )
            if self.calibration_fusion_mode == "mean_last_event_attention_summary"
            else None
        )
        self.calibration_summary_gate = (
            nn.Sequential(nn.Linear(self.d_model * 3, self.d_model), nn.Sigmoid())
            if self.calibration_fusion_mode == "mean_last_gated_summary"
            else None
        )
        fusion_input_dim = {
            "add": self.d_model,
            "mean_last_summary_concat": self.d_model * 3,
            "mean_last_gated_summary": self.d_model * 3,
            "mean_last_attention_summary": self.d_model * 4,
            "mean_last_event_attention_summary": self.d_model * 4,
        }[self.calibration_fusion_mode]
        fusion_hidden = int(
            calibration_fusion_hidden_dim
            if calibration_fusion_hidden_dim is not None and int(calibration_fusion_hidden_dim) > 0
            else max(fusion_input_dim, self.calibration_repr_dim)
        )
        self.calibration_fusion = (
            nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden),
                nn.GELU(),
                nn.LayerNorm(fusion_hidden),
                nn.Dropout(self.dropout),
                nn.Linear(fusion_hidden, self.calibration_repr_dim),
                nn.GELU(),
                nn.LayerNorm(self.calibration_repr_dim),
            )
            if self.calibration_fusion_mode != "add"
            else nn.Identity()
        )
        self.no_calibration_embedding = (
            nn.Parameter(torch.zeros(1, self.calibration_repr_dim))
            if self.calibration_steps <= 0
            else None
        )
        self.no_calibration_anchor_logit = (
            nn.Parameter(torch.zeros(1))
            if self.calibration_steps <= 0
            else None
        )
        if self.no_calibration_embedding is not None:
            nn.init.normal_(self.no_calibration_embedding, mean=0.0, std=0.02)
        self.init_state = nn.Sequential(nn.Linear(self.calibration_repr_dim, self.hidden_dim), nn.Tanh())
        stream_base_dim = self.head_dim + self.motion_feature_dim + self.stream_time_feature_dim
        self.motion_input = nn.Sequential(
            nn.Linear(
                stream_base_dim + self.state_feedback_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
        )
        if self.motion_encoder_context == "tcn" and self.motion_encoder_layers > 0:
            self.motion_stem_input = nn.Sequential(
                nn.Linear(stream_base_dim, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
                nn.Dropout(self.dropout),
            )
            self.motion_stem_blocks = nn.Sequential(
                *[
                    TCNBlock(
                        self.hidden_dim,
                        dilation=2**idx,
                        dropout=self.dropout,
                        kernel_size=kernel_size,
                    )
                    for idx in range(self.motion_encoder_layers)
                ]
            )
            self.motion_feedback_projection = (
                nn.Sequential(
                    nn.Linear(self.hidden_dim + self.state_feedback_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Dropout(self.dropout),
                )
                if self.state_feedback_dim > 0
                else None
            )
        else:
            self.motion_stem_input = None
            self.motion_stem_blocks = None
            self.motion_feedback_projection = None
        self.deep_tcn_stream = (
            DeepTCNEncoder(
                stream_base_dim,
                self.hidden_dim,
                dilations=self.deep_tcn_dilations,
                kernel_size=kernel_size,
                dropout=self.dropout,
            )
            if self.stream_context_mode in {"deep_tcn", "deep_tcn_latent_gru"}
            else None
        )
        self.stream_transformer_input = (
            nn.Sequential(
                nn.Linear(stream_base_dim, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
                nn.Dropout(self.dropout),
            )
            if self.stream_context_mode in {"transformer", "transformer_latent_gru"}
            else None
        )
        self.stream_transformer_position_embedding = (
            nn.Parameter(torch.zeros(1, self.max_time_steps, self.hidden_dim))
            if self.stream_context_mode in {"transformer", "transformer_latent_gru"}
            else None
        )
        if self.stream_transformer_position_embedding is not None:
            nn.init.normal_(self.stream_transformer_position_embedding, mean=0.0, std=0.02)
        if self.stream_context_mode in {"transformer", "transformer_latent_gru"}:
            stream_transformer_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(transformer_heads),
                dim_feedforward=int(transformer_ff_dim),
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.stream_transformer = nn.TransformerEncoder(
                stream_transformer_layer,
                num_layers=max(1, int(transformer_layers)),
            )
        else:
            self.stream_transformer = None
        self.stream_transformer_latent_gru = (
            nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
            if self.stream_context_mode == "transformer_latent_gru"
            else None
        )
        self.deep_tcn_latent_gru = (
            nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
            if self.stream_context_mode == "deep_tcn_latent_gru"
            else None
        )
        self.deep_tcn_feedback_projection = (
            nn.Sequential(
                nn.Linear(self.hidden_dim + 1, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
                nn.Dropout(self.dropout),
            )
            if self.stream_context_mode in {"deep_tcn_latent_gru", "transformer_latent_gru"}
            and self.state_feedback_mode == "predicted_current"
            else None
        )
        self.stream_calib_film = (
            nn.Sequential(
                nn.Linear(self.calibration_repr_dim, self.hidden_dim * 2),
                nn.Tanh(),
            )
            if self.stream_calib_condition_mode == "film"
            else None
        )
        self.stream_gru = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.feedback_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.calibration_repr_dim, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 1),
        )
        self.stream_tcn = (
            nn.Sequential(
                TCNBlock(self.hidden_dim, dilation=1, dropout=self.dropout, kernel_size=kernel_size),
                TCNBlock(self.hidden_dim, dilation=2, dropout=self.dropout, kernel_size=kernel_size),
                TCNBlock(self.hidden_dim, dilation=4, dropout=self.dropout, kernel_size=kernel_size),
                TCNBlock(self.hidden_dim, dilation=8, dropout=self.dropout, kernel_size=kernel_size),
            )
            if self.stream_context_mode in {"gru_tcn", "gru_tcn_multiscale"}
            else None
        )
        self.multiscale_state_projection = (
            nn.Sequential(
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
                nn.Dropout(self.dropout),
            )
            if self.stream_context_mode in {"gru_multiscale", "gru_tcn_multiscale"}
            else None
        )
        hidden_static = int(static_hidden_dim if static_hidden_dim is not None else self.d_model)
        self.static_encoder = (
            StaticEncoder(self.static_dim, hidden_static, static_dropout)
            if self.use_static and self.decoder_context_mode == "fused"
            else None
        )
        self.static_projection = (
            nn.Sequential(nn.Linear(hidden_static, self.d_model), nn.GELU(), nn.LayerNorm(self.d_model))
            if self.use_static and self.decoder_context_mode == "fused" and hidden_static != self.d_model
            else None
        )
        stats_dim = self.head_dim * 4 + 9
        self.motion_stats_encoder = (
            nn.Sequential(
                nn.Linear(stats_dim, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
                nn.Dropout(self.dropout),
            )
            if self.motion_stats_branch
            else None
        )
        if self.decoder_context_mode == "fused":
            fusion_dim = self.hidden_dim + self.calibration_repr_dim
            if self.motion_stats_branch:
                fusion_dim += self.d_model
            if self.use_static:
                fusion_dim += self.d_model
            self.decoder_feature_dim = self.d_model
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim, self.d_model * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * 2, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
            )
        else:
            fusion_dim = self.hidden_dim
            if self.motion_stats_branch:
                fusion_dim += self.d_model
            if self.use_static:
                fusion_dim += self.static_dim
            self.decoder_feature_dim = fusion_dim
            self.fusion = nn.Identity()
        self.decoder_temporal_blocks = (
            nn.Sequential(
                *[
                    TCNBlock(
                        self.decoder_feature_dim,
                        dilation=2**idx,
                        dropout=self.dropout,
                        kernel_size=kernel_size,
                    )
                    for idx in range(self.decoder_temporal_layers)
                ]
            )
            if self.decoder_temporal_context == "tcn" and self.decoder_temporal_layers > 0
            else None
        )
        self.calibration_summary_fusion = (
            CalibrationSummaryFeatureFusion(
                self.calibration_summary_fusion_feature_dim,
                self.decoder_feature_dim,
                calibration_summary_fusion_hidden_dim
                if calibration_summary_fusion_hidden_dim is not None
                else self.decoder_hidden_dim,
                self.dropout,
                mode=self.calibration_summary_fusion_mode,
                strength=self.calibration_summary_fusion_strength,
            )
            if self.calibration_summary_fusion_enabled
            else None
        )
        self.fds_module = (
            FeatureDistributionSmoothing(
                self.decoder_feature_dim,
                num_bins=self.fds_num_bins,
                min_value=self.fds_min,
                max_value=self.fds_max,
                bin_size=self.fds_bin_size,
                kernel=self.fds_kernel,
                kernel_size=self.fds_kernel_size,
                sigma=self.fds_sigma,
                momentum=self.fds_momentum,
                blend=self.fds_blend,
            )
            if self.fds_enabled
            else None
        )
        self.current_reg_head = make_current_regression_head(
            self.decoder_feature_dim,
            self.decoder_hidden_dim,
            self.dropout,
        )
        session_affine_input_dim = self.calibration_repr_dim
        if self.use_static:
            session_affine_input_dim += self.d_model if self.decoder_context_mode == "fused" else self.static_dim
        self.current_session_affine_head = (
            CurrentSessionAffineCalibrationHead(
                session_affine_input_dim,
                current_session_affine_hidden_dim
                if current_session_affine_hidden_dim is not None
                else self.decoder_hidden_dim,
                self.dropout,
                scale_range=self.current_session_affine_scale_range,
                bias_range=self.current_session_affine_bias_range,
            )
            if self.current_session_affine_head_enabled
            else None
        )
        affine_input_dim = self.decoder_feature_dim + self.calibration_repr_dim + 6
        self.current_affine_head = (
            CurrentAffineCalibrationHead(
                affine_input_dim,
                current_affine_hidden_dim if current_affine_hidden_dim is not None else self.decoder_hidden_dim,
                self.dropout,
                scale_range=self.current_affine_scale_range,
                bias_range=self.current_affine_bias_range,
            )
            if self.current_affine_head_enabled
            else None
        )
        self.current_binned_affine_head = (
            CurrentBinnedAffineCalibrationHead(
                anchor_bins=self.current_binned_affine_anchor_bins,
                pred_bins=self.current_binned_affine_pred_bins,
                time_bins=self.current_binned_affine_time_bins,
                scale_range=self.current_binned_affine_scale_range,
                bias_range=self.current_binned_affine_bias_range,
            )
            if self.current_binned_affine_head_enabled
            else None
        )
        self.calibration_residual_adapter = (
            CalibrationResidualAdapter(
                self.calibration_residual_feature_dim,
                calibration_residual_adapter_hidden_dim
                if calibration_residual_adapter_hidden_dim is not None
                else self.decoder_hidden_dim,
                self.dropout,
                mode=self.calibration_residual_adapter_mode,
                delta_range=self.calibration_residual_delta_range,
                decay_seconds=self.calibration_residual_decay_seconds,
                gate_low_threshold=self.calibration_residual_gate_low_threshold,
                gate_high_threshold=self.calibration_residual_gate_high_threshold,
                gate_anchor_threshold=self.calibration_residual_gate_anchor_threshold,
                gate_temperature=self.calibration_residual_gate_temperature,
            )
            if self.calibration_residual_adapter_enabled
            else None
        )
        low_suppressor_input_dim = self.decoder_feature_dim + self.calibration_repr_dim + 6
        self.current_low_suppressor_head = (
            CurrentLowFMSSuppressorHead(
                low_suppressor_input_dim,
                current_low_suppressor_hidden_dim
                if current_low_suppressor_hidden_dim is not None
                else self.decoder_hidden_dim,
                self.dropout,
                delta_range=self.current_low_suppressor_delta_range,
                gate_init_bias=self.current_low_suppressor_gate_init_bias,
            )
            if self.current_low_suppressor_enabled
            else None
        )
        risk_input_dim = current_head_risk_input_dim(self.decoder_feature_dim, self.current_head_mode)
        for attr_name in (
            "current_level_head",
            "current_delta_head",
            "current_gate_head",
            "session_drift_head",
            "current_residual_delta_head",
            "person_dynamic_head",
            "person_bias_head",
            "person_scale_head",
            "person_speed_head",
            "current_trajectory_head",
            "current_regime_gate_head",
            "current_regime_expert_head",
            "current_state_delta_head",
            "current_state_leak_head",
            "current_state_equilibrium_head",
            "current_range_delta_head",
            "current_range_level_head",
            "current_range_gate_head",
            "current_range_scale_head",
            "current_range_guard_head",
            "current_calib_prior_gate_head",
            "current_calib_prior_cap_head",
            "current_anchor_delta_head",
            "current_anchor_gate_head",
        ):
            setattr(self, attr_name, None)
        if self.current_head_mode == "dual_delta_gate":
            dual_heads = make_dual_delta_heads(
                self.decoder_feature_dim,
                self.calibration_repr_dim,
                self.decoder_hidden_dim,
                self.dropout,
            )
            self.current_level_head = dual_heads["current_level_head"]
            self.current_delta_head = dual_heads["current_delta_head"]
            self.current_gate_head = dual_heads["current_gate_head"]
            self.session_drift_head = dual_heads["session_drift_head"]
        elif self.current_head_mode == "residual_update":
            residual_heads = make_residual_update_heads(
                self.decoder_feature_dim,
                self.decoder_hidden_dim,
                self.dropout,
            )
            self.current_residual_delta_head = residual_heads["current_residual_delta_head"]
        elif self.current_head_mode == "person_prior":
            person_heads = make_person_prior_heads(
                self.decoder_feature_dim,
                self.calibration_repr_dim,
                self.decoder_hidden_dim,
                self.dropout,
            )
            self.person_dynamic_head = person_heads["person_dynamic_head"]
            self.person_bias_head = person_heads["person_bias_head"]
            self.person_scale_head = person_heads["person_scale_head"]
            self.person_speed_head = person_heads["person_speed_head"]
        elif self.current_head_mode == "trajectory_decoder":
            trajectory_heads = make_trajectory_decoder_heads(
                self.decoder_feature_dim,
                len(self.current_trajectory_offsets),
                self.decoder_hidden_dim,
                self.dropout,
            )
            self.current_trajectory_head = trajectory_heads["current_trajectory_head"]
        elif self.current_head_mode in {"regime_gated", "anchor_regime_gated"}:
            regime_input_dim = self.decoder_feature_dim + (4 if self.current_head_mode == "anchor_regime_gated" else 0)
            regime_heads = make_regime_gated_heads(
                regime_input_dim,
                self.regime_class_count,
                self.decoder_hidden_dim,
                self.dropout,
            )
            self.current_regime_gate_head = regime_heads["current_regime_gate_head"]
            self.current_regime_expert_head = regime_heads["current_regime_expert_head"]
        elif self.current_head_mode == "state_space_delta":
            state_heads = make_state_space_delta_heads(
                self.decoder_feature_dim,
                self.decoder_hidden_dim,
                self.dropout,
            )
            self.current_state_delta_head = state_heads["current_state_delta_head"]
            self.current_state_leak_head = state_heads["current_state_leak_head"]
            self.current_state_equilibrium_head = state_heads["current_state_equilibrium_head"]
        elif self.current_head_mode in {
            "range_scaled_delta",
            "guarded_range_scaled_delta",
            "calib_prior_range_scaled_delta",
            "calib_lowcap_range_scaled_delta",
        }:
            range_heads = make_range_scaled_delta_heads(
                self.decoder_feature_dim,
                self.calibration_repr_dim,
                self.decoder_hidden_dim,
                self.dropout,
                guarded=self.current_head_mode == "guarded_range_scaled_delta",
            )
            self.current_range_delta_head = range_heads["current_range_delta_head"]
            self.current_range_level_head = range_heads["current_range_level_head"]
            self.current_range_gate_head = range_heads["current_range_gate_head"]
            self.current_range_scale_head = range_heads["current_range_scale_head"]
            self.current_range_guard_head = range_heads.get("current_range_guard_head")
            if self.current_head_mode in {"calib_prior_range_scaled_delta", "calib_lowcap_range_scaled_delta"}:
                prior_heads = make_calibration_prior_heads(
                    self.decoder_feature_dim,
                    self.calibration_repr_dim,
                    self.decoder_hidden_dim,
                    self.dropout,
                )
                self.current_calib_prior_gate_head = prior_heads["current_calib_prior_gate_head"]
                self.current_calib_prior_cap_head = prior_heads["current_calib_prior_cap_head"]
        elif self.current_head_mode == "zero_anchor_mixture":
            anchor_heads = make_zero_anchor_mixture_heads(
                self.decoder_feature_dim,
                self.decoder_hidden_dim,
                self.dropout,
            )
            self.current_anchor_delta_head = anchor_heads["current_anchor_delta_head"]
            self.current_anchor_gate_head = anchor_heads["current_anchor_gate_head"]
        if self.risk_head_enabled or self.fall_risk_head_enabled or self.high_risk_head_enabled:
            if self.risk_temporal_context == "tcn" and self.risk_temporal_layers > 0:
                self.risk_context_projection = (
                    nn.Sequential(nn.Linear(risk_input_dim, self.d_model), nn.GELU(), nn.LayerNorm(self.d_model))
                    if risk_input_dim != self.d_model
                    else nn.Identity()
                )
                self.risk_temporal_blocks = nn.Sequential(
                    *[
                        TCNBlock(
                            self.d_model,
                            dilation=2**idx,
                            dropout=self.dropout,
                            kernel_size=kernel_size,
                        )
                        for idx in range(self.risk_temporal_layers)
                    ]
                )
                risk_head_dim = self.d_model
            else:
                self.risk_context_projection = None
                self.risk_temporal_blocks = None
                risk_head_dim = risk_input_dim
            self.risk_head_dim = int(risk_head_dim)
        else:
            self.risk_context_projection = None
            self.risk_temporal_blocks = None
            self.risk_head_dim = int(risk_input_dim)
        if self.risk_head_enabled:
            if self.current_head_mode == "dual_delta_gate":
                risk_hidden = self.decoder_hidden_dim or self.d_model
                self.risk_head = nn.Sequential(
                    nn.Linear(risk_head_dim, risk_hidden),
                    nn.GELU(),
                    nn.LayerNorm(risk_hidden),
                    nn.Dropout(self.dropout),
                    nn.Linear(risk_hidden, len(self.rise_horizon_steps)),
                )
            else:
                self.risk_head = _make_mlp_head(
                    risk_head_dim,
                    len(self.rise_horizon_steps),
                    self.decoder_hidden_dim,
                    self.dropout,
                )
        else:
            self.risk_head = None
        self.fall_risk_head = (
            _make_mlp_head(
                self.risk_head_dim,
                len(self.fall_horizon_steps),
                self.decoder_hidden_dim,
                self.dropout,
            )
            if self.fall_risk_head_enabled
            else None
        )
        high_risk_count = len(self.high_risk_horizon_steps) * len(self.high_risk_thresholds)
        self.high_risk_head = (
            _make_mlp_head(
                self.risk_head_dim,
                high_risk_count,
                self.decoder_hidden_dim,
                self.dropout,
            )
            if self.high_risk_head_enabled and high_risk_count > 0
            else None
        )
        self.ordinal_head = make_ordinal_head(
            self.decoder_feature_dim,
            len(self.ordinal_bins_raw),
            self.decoder_hidden_dim,
            self.dropout,
            self.ordinal_head_mode,
        )
        coarse_band_count = len(self.coarse_band_bins_raw) + 1 if self.coarse_band_bins_raw else 0
        self.coarse_band_head = (
            _make_mlp_head(self.decoder_feature_dim, coarse_band_count, self.decoder_hidden_dim, self.dropout)
            if coarse_band_count > 0
            else None
        )
        self.coarse_residual_head = (
            _make_mlp_head(self.decoder_feature_dim, coarse_band_count, self.decoder_hidden_dim, self.dropout)
            if self.coarse_residual_head_enabled and coarse_band_count > 0
            else None
        )
        self.regime_head = (
            _make_mlp_head(self.decoder_feature_dim, self.regime_class_count, self.decoder_hidden_dim, self.dropout)
            if self.regime_head_enabled
            else None
        )
        self.uncertainty_head = (
            _make_mlp_head(self.decoder_feature_dim, 1, self.decoder_hidden_dim, self.dropout)
            if self.uncertainty_head_enabled
            else None
        )
        future_aux_count = len(self.future_aux_horizon_steps)
        self.future_aux_head = (
            _make_mlp_head(self.decoder_feature_dim, future_aux_count, self.decoder_hidden_dim, self.dropout)
            if future_aux_count > 0
            else None
        )
        self.event_aux_head = (
            _make_mlp_head(self.decoder_feature_dim, future_aux_count * 3, self.decoder_hidden_dim, self.dropout)
            if future_aux_count > 0
            else None
        )
        self.recent_rf_steps = (
            tcn_receptive_field_steps(self.deep_tcn_dilations, kernel_size=kernel_size)
            if self.stream_context_mode in {"deep_tcn", "deep_tcn_latent_gru"}
            else self.recent_steps
        )
        self.recent_rf_seconds = self.recent_steps * self.sampling_interval
        if self.stream_context_mode in {"deep_tcn", "deep_tcn_latent_gru"}:
            self.recent_rf_seconds = self.recent_rf_steps * self.sampling_interval
        if self.stream_context_mode in {"transformer", "transformer_latent_gru"} and self.stream_prepend_calibration:
            self.recent_rf_steps = self.calibration_steps + self.recent_steps
            self.recent_rf_seconds = self.recent_rf_steps * self.sampling_interval

    def reset_fds_epoch_stats(self) -> None:
        if self.fds_module is not None:
            self.fds_module.reset_epoch_stats()

    def commit_fds_epoch_stats(self) -> Dict[str, float]:
        if self.fds_module is None:
            return {
                "fds_epoch_points": 0,
                "fds_bins_observed": 0,
                "fds_running_bins": 0,
                "fds_initialized": False,
            }
        return self.fds_module.commit_epoch_stats()

    def _stream_history_start(self) -> int:
        if self.stream_prepend_calibration and self.stream_context_mode in {"deep_tcn", "deep_tcn_latent_gru"}:
            return max(0, int(self.calibration_steps) - int(self.recent_rf_steps) + 1)
        if self.stream_prepend_calibration and self.stream_context_mode in {"transformer", "transformer_latent_gru"}:
            return 0
        return int(self.calibration_steps)

    def _append_stream_time_features(self, motion: torch.Tensor, start: int) -> torch.Tensor:
        start = int(start)
        post_motion = motion[:, start:]
        if not self.stream_time_features:
            return post_motion
        post_steps = int(post_motion.shape[1])
        positions = start + torch.arange(post_steps, dtype=motion.dtype, device=motion.device)
        absolute_minutes = positions * self.sampling_interval / 60.0
        since_calib_minutes = (positions - float(self.calibration_steps)) * self.sampling_interval / 60.0
        time_features = torch.stack([absolute_minutes, since_calib_minutes], dim=-1).unsqueeze(0)
        return torch.cat([post_motion, time_features.expand(motion.shape[0], -1, -1)], dim=-1)

    def _prediction_positions(self, lengths: torch.Tensor, device: torch.device) -> Tuple[int, torch.Tensor, torch.Tensor]:
        start = max(int(self.calibration_steps), int(self.recent_steps) - 1)
        max_current_t = int(torch.clamp(lengths.max(), min=start).item())
        pred_steps = max(0, max_current_t - start)
        positions = start + torch.arange(pred_steps, dtype=torch.long, device=device)
        mask = positions.view(1, -1) < lengths.view(-1, 1)
        return start, positions, mask

    def _encode_calibration(self, head: torch.Tensor, fms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if head.shape[1] < self.calibration_steps:
            raise ValueError("head sequence shorter than calibration_steps")
        if self.calibration_steps <= 0:
            if self.no_calibration_embedding is None or self.no_calibration_anchor_logit is None:
                raise ValueError("calibration_steps=0 requires no-calibration parameters.")
            z_calib = self.no_calibration_embedding.to(dtype=head.dtype, device=head.device).expand(head.shape[0], -1)
            base_value = torch.sigmoid(self.no_calibration_anchor_logit.to(dtype=head.dtype, device=head.device)).view(1)
            base_fms = base_value.expand(head.shape[0])
            return z_calib, base_fms, base_fms
        context_fms = calibration_context_fms(fms, self.calibration_steps, self.fms_context_mode)
        raw_base_fms = context_fms[:, -1]
        model_context_fms = context_fms
        if self.training and self.calib_fms_dropout > 0:
            keep = torch.rand_like(context_fms) >= self.calib_fms_dropout
            replacement = context_fms.mean(dim=1, keepdim=True).expand_as(context_fms)
            model_context_fms = torch.where(keep, context_fms, replacement)
        model_base_fms = raw_base_fms
        if self.training and self.calibration_end_fms_dropout > 0:
            keep_base = torch.rand((context_fms.shape[0],), dtype=torch.float32, device=context_fms.device) >= self.calibration_end_fms_dropout
            replacement_base = context_fms.mean(dim=1)
            model_base_fms = torch.where(keep_base, raw_base_fms, replacement_base)
        calib_seq = torch.cat([head[:, : self.calibration_steps], model_context_fms.unsqueeze(-1)], dim=-1)
        z = self.calibration_encoder(calib_seq)
        if self.calibration_position_embedding is not None:
            if self.calibration_cls_token is not None:
                cls = self.calibration_cls_token.to(dtype=z.dtype, device=z.device).expand(z.shape[0], -1, -1)
                z = torch.cat([cls, z], dim=1)
            if z.shape[1] > self.calibration_position_embedding.shape[1]:
                raise ValueError(
                    f"calibration sequence length {z.shape[1]} exceeds positional embedding length "
                    f"{self.calibration_position_embedding.shape[1]}."
                )
            z = z + self.calibration_position_embedding[:, : z.shape[1]].to(dtype=z.dtype, device=z.device)
        z = self.calibration_transformer(z)
        if self.calibration_encoder_mode == "transformer_cls":
            pooled = z[:, 0]
            sequence_z = z[:, 1:]
        else:
            pooled = self.calibration_pool(z)
            sequence_z = z
        if self.calib_summary_encoder is not None:
            z_summary = self.calib_summary_encoder(_calibration_summary_features(head, model_context_fms, self.calibration_steps))
        else:
            z_summary = torch.zeros_like(pooled)
        if self.calibration_fusion_mode == "add":
            z_calib = pooled
            if self.calib_summary_encoder is not None:
                z_calib = z_calib + z_summary
        else:
            z_mean = sequence_z.mean(dim=1)
            z_last = sequence_z[:, -1]
            if self.calibration_fusion_mode == "mean_last_gated_summary":
                assert self.calibration_summary_gate is not None
                gate = self.calibration_summary_gate(torch.cat([z_mean, z_last, z_summary], dim=-1))
                fusion_parts = [z_mean, z_last, gate * z_summary]
            elif self.calibration_fusion_mode == "mean_last_attention_summary":
                assert self.calibration_attention_pool is not None
                z_attn = self.calibration_attention_pool(sequence_z)
                fusion_parts = [z_mean, z_last, z_attn, z_summary]
            elif self.calibration_fusion_mode == "mean_last_event_attention_summary":
                assert self.calibration_event_attention is not None
                delta_z = torch.cat([sequence_z[:, :1].new_zeros(sequence_z[:, :1].shape), sequence_z[:, 1:] - sequence_z[:, :-1]], dim=1)
                fms_seq = model_context_fms[:, : sequence_z.shape[1]].to(dtype=sequence_z.dtype).unsqueeze(-1)
                delta_fms = torch.cat([fms_seq[:, :1].new_zeros(fms_seq[:, :1].shape), fms_seq[:, 1:] - fms_seq[:, :-1]], dim=1)
                time_t = torch.linspace(0.0, 1.0, int(sequence_z.shape[1]), dtype=sequence_z.dtype, device=sequence_z.device)
                time_t = time_t.view(1, -1, 1).expand(sequence_z.shape[0], -1, -1)
                score_in = torch.cat([sequence_z, delta_z, fms_seq, delta_fms, time_t], dim=-1)
                weights = torch.softmax(self.calibration_event_attention(score_in).squeeze(-1), dim=1)
                z_event = torch.sum(sequence_z * weights.unsqueeze(-1), dim=1)
                fusion_parts = [z_mean, z_last, z_event, z_summary]
            else:
                fusion_parts = [z_mean, z_last, z_summary]
            z_calib = self.calibration_fusion(torch.cat(fusion_parts, dim=-1))
        return z_calib, raw_base_fms, model_base_fms

    def _stream_context(self, state_seq: torch.Tensor) -> torch.Tensor:
        if self.stream_tcn is not None:
            state_seq = self.stream_tcn(state_seq)
        if self.multiscale_state_projection is None:
            return state_seq
        windows = [
            state_seq,
            _causal_rolling_mean(state_seq, max(1, int(round(5.0 / self.sampling_interval)))),
            _causal_rolling_mean(state_seq, max(1, int(round(15.0 / self.sampling_interval)))),
            _causal_rolling_mean(state_seq, max(1, int(round(30.0 / self.sampling_interval)))),
        ]
        return self.multiscale_state_projection(torch.cat(windows, dim=-1))

    def _encode_motion_stem(self, stream_motion: torch.Tensor) -> Optional[torch.Tensor]:
        if self.motion_stem_input is None or self.motion_stem_blocks is None:
            return None
        return self.motion_stem_blocks(self.motion_stem_input(stream_motion))

    def _run_stream(
        self,
        stream_motion: torch.Tensor,
        h0: torch.Tensor,
        z_calib: torch.Tensor,
        base_fms: torch.Tensor,
        update_start_offset: int = 0,
    ) -> torch.Tensor:
        if self.deep_tcn_stream is not None:
            deep_features = self.deep_tcn_stream(stream_motion)
            update_start_offset = max(0, int(update_start_offset))
            if update_start_offset > 0:
                deep_features = deep_features[:, update_start_offset:]
            if self.stream_calib_film is not None:
                scale_shift = self.stream_calib_film(z_calib).view(z_calib.shape[0], 1, 2, self.hidden_dim)
                scale = scale_shift[:, :, 0] * self.stream_calib_condition_strength
                shift = scale_shift[:, :, 1] * self.stream_calib_condition_strength
                deep_features = deep_features * (1.0 + scale) + shift
            if self.deep_tcn_latent_gru is not None:
                if self.deep_tcn_feedback_projection is not None:
                    states: List[torch.Tensor] = []
                    hidden = h0
                    prev_current = base_fms
                    for step_idx in range(deep_features.shape[1]):
                        feedback = prev_current.view(prev_current.shape[0], 1)
                        step = self.deep_tcn_feedback_projection(torch.cat([deep_features[:, step_idx], feedback], dim=-1))
                        out, hidden = self.deep_tcn_latent_gru(step.unsqueeze(1), hidden)
                        state = out[:, 0]
                        prev_current = torch.sigmoid(self.feedback_head(torch.cat([state, z_calib], dim=-1)).squeeze(-1))
                        states.append(state)
                    if not states:
                        return deep_features.new_zeros((deep_features.shape[0], 0, self.hidden_dim))
                    return torch.stack(states, dim=1)
                state_seq, _ = self.deep_tcn_latent_gru(deep_features, h0)
                return state_seq
            return deep_features
        if self.stream_transformer is not None:
            assert self.stream_transformer_input is not None
            features = self.stream_transformer_input(stream_motion)
            seq_len = int(features.shape[1])
            if self.stream_transformer_position_embedding is not None:
                if seq_len > self.stream_transformer_position_embedding.shape[1]:
                    raise ValueError(
                        f"stream sequence length {seq_len} exceeds max_time_steps="
                        f"{self.stream_transformer_position_embedding.shape[1]}."
                    )
                features = features + self.stream_transformer_position_embedding[:, :seq_len].to(
                    dtype=features.dtype,
                    device=features.device,
                )
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=features.device),
                diagonal=1,
            )
            features = self.stream_transformer(features, mask=causal_mask)
            update_start_offset = max(0, int(update_start_offset))
            if update_start_offset > 0:
                features = features[:, update_start_offset:]
            if self.stream_calib_film is not None:
                scale_shift = self.stream_calib_film(z_calib).view(z_calib.shape[0], 1, 2, self.hidden_dim)
                scale = scale_shift[:, :, 0] * self.stream_calib_condition_strength
                shift = scale_shift[:, :, 1] * self.stream_calib_condition_strength
                features = features * (1.0 + scale) + shift
            if self.stream_transformer_latent_gru is not None:
                if self.deep_tcn_feedback_projection is not None:
                    states: List[torch.Tensor] = []
                    hidden = h0
                    prev_current = base_fms
                    for step_idx in range(features.shape[1]):
                        feedback = prev_current.view(prev_current.shape[0], 1)
                        step = self.deep_tcn_feedback_projection(torch.cat([features[:, step_idx], feedback], dim=-1))
                        out, hidden = self.stream_transformer_latent_gru(step.unsqueeze(1), hidden)
                        state = out[:, 0]
                        prev_current = torch.sigmoid(self.feedback_head(torch.cat([state, z_calib], dim=-1)).squeeze(-1))
                        states.append(state)
                    if not states:
                        return features.new_zeros((features.shape[0], 0, self.hidden_dim))
                    return torch.stack(states, dim=1)
                state_seq, _ = self.stream_transformer_latent_gru(features, h0)
                return state_seq
            return features
        motion_stem = self._encode_motion_stem(stream_motion)
        if self.state_feedback_mode == "none":
            stream = motion_stem if motion_stem is not None else self.motion_input(stream_motion)
            state_seq, _ = self.stream_gru(stream, h0)
            return state_seq

        states: List[torch.Tensor] = []
        hidden = h0
        prev_current = base_fms
        for step_idx in range(stream_motion.shape[1]):
            feedback = prev_current.view(prev_current.shape[0], 1, 1)
            if motion_stem is not None:
                assert self.motion_feedback_projection is not None
                stream = self.motion_feedback_projection(torch.cat([motion_stem[:, step_idx : step_idx + 1], feedback], dim=-1))
            else:
                step_motion = torch.cat([stream_motion[:, step_idx : step_idx + 1], feedback], dim=-1)
                stream = self.motion_input(step_motion)
            out, hidden = self.stream_gru(stream, hidden)
            state = out[:, 0]
            prev_current = torch.sigmoid(self.feedback_head(torch.cat([state, z_calib], dim=-1)).squeeze(-1))
            states.append(state)
        if not states:
            return stream_motion.new_zeros((stream_motion.shape[0], 0, self.hidden_dim))
        return torch.stack(states, dim=1)

    def _encode_static(self, static: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
        if not self.use_static:
            return None
        if static is None:
            raise ValueError("OnlineFMSRiskTracker was created with use_static=True, but static tensor was not provided.")
        if static.ndim != 2 or static.shape[0] != batch_size or static.shape[1] != self.static_dim:
            raise ValueError(f"static must be [B,{self.static_dim}], got {static.shape}")
        if self.decoder_context_mode == "state":
            return static.to(dtype=torch.float32)
        assert self.static_encoder is not None
        z_static = self.static_encoder(static)
        if self.static_projection is not None:
            z_static = self.static_projection(z_static)
        return z_static

    def forward(
        self,
        head: torch.Tensor,
        y_calib: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        static: Optional[torch.Tensor] = None,
        fds_labels_raw: Optional[torch.Tensor] = None,
        fds_mask: Optional[torch.Tensor] = None,
        fds_update: bool = False,
        fds_apply: bool = False,
        calibration_residual_features: Optional[torch.Tensor] = None,
        calibration_residual_feature_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        assert head.ndim == 3, f"head must be [B,T,6], got {head.shape}"
        if head.shape[-1] != self.head_dim:
            raise ValueError(f"expected head_dim={self.head_dim}, got {head.shape[-1]}")
        bsz, steps, _ = head.shape
        device = head.device
        lengths = lengths.to(device) if lengths is not None else torch.full((bsz,), steps, dtype=torch.long, device=device)
        if y_calib.ndim != 2:
            raise ValueError(f"FMS input must be [B,C], got {y_calib.shape}")
        z_calib, base_fms, model_base_fms = self._encode_calibration(head, y_calib.to(device))
        start, positions, mask = self._prediction_positions(lengths, device)
        pred_steps = int(positions.numel())
        rise_h = torch.tensor(self.rise_horizon_steps, dtype=torch.long, device=device)
        rise_thr = torch.tensor(self.rise_thresholds, dtype=head.dtype, device=device)
        fall_h = torch.tensor(self.fall_horizon_steps, dtype=torch.long, device=device)
        fall_thr = torch.tensor(self.fall_thresholds, dtype=head.dtype, device=device)
        if pred_steps == 0:
            stream_history_start = self._stream_history_start()
            return {
                "current": head.new_zeros((bsz, 0)),
                "current_reg": head.new_zeros((bsz, 0)),
                "current_ordinal": head.new_zeros((bsz, 0)),
                "current_trajectory": head.new_zeros((bsz, 0, len(self.current_trajectory_offsets))),
                "current_trajectory_offsets": torch.tensor(self.current_trajectory_offsets, dtype=torch.long, device=device),
                "ordinal_logits": head.new_zeros((bsz, 0, len(self.ordinal_bins_raw))),
                "ordinal_probs": head.new_zeros((bsz, 0, len(self.ordinal_bins_raw))),
                "coarse_band_logits": head.new_zeros((bsz, 0, len(self.coarse_band_bins_raw) + 1 if self.coarse_band_bins_raw else 0)),
                "coarse_band_probs": head.new_zeros((bsz, 0, len(self.coarse_band_bins_raw) + 1 if self.coarse_band_bins_raw else 0)),
                "current_coarse_residual": head.new_zeros((bsz, 0)),
                "current_pre_coarse_residual": head.new_zeros((bsz, 0)),
                "coarse_residual_values": head.new_zeros((bsz, 0, len(self.coarse_band_bins_raw) + 1 if self.coarse_band_bins_raw else 0)),
                "regime_logits": head.new_zeros((bsz, 0, self.regime_class_count if self.regime_head_enabled else 0)),
                "regime_probs": head.new_zeros((bsz, 0, self.regime_class_count if self.regime_head_enabled else 0)),
                "current_log_sigma": head.new_zeros((bsz, 0)),
                "current_sigma": head.new_zeros((bsz, 0)),
                "current_pre_low_suppressor": head.new_zeros((bsz, 0)),
                "current_low_suppressor_correction": head.new_zeros((bsz, 0)),
                "current_low_suppressor_gate": head.new_zeros((bsz, 0)),
                "current_low_suppressor_gate_logits": head.new_zeros((bsz, 0)),
                "risk_logits": head.new_zeros((bsz, 0, len(self.rise_horizon_steps))),
                "risk_probs": head.new_zeros((bsz, 0, len(self.rise_horizon_steps))),
                "fall_risk_logits": head.new_zeros((bsz, 0, len(self.fall_horizon_steps))),
                "fall_risk_probs": head.new_zeros((bsz, 0, len(self.fall_horizon_steps))),
                "high_risk_logits": head.new_zeros((bsz, 0, len(self.high_risk_horizon_steps), len(self.high_risk_thresholds))),
                "high_risk_probs": head.new_zeros((bsz, 0, len(self.high_risk_horizon_steps), len(self.high_risk_thresholds))),
                "future_aux": head.new_zeros((bsz, 0, len(self.future_aux_horizon_steps))),
                "event_logits": head.new_zeros((bsz, 0, len(self.future_aux_horizon_steps), 3)),
                "event_probs": head.new_zeros((bsz, 0, len(self.future_aux_horizon_steps), 3)),
                "mask": mask,
                "prediction_start": torch.tensor(start, device=device),
                "rise_horizon_steps": rise_h,
                "fall_horizon_steps": fall_h,
                "high_risk_horizon_steps": torch.tensor(self.high_risk_horizon_steps, dtype=torch.long, device=device),
                "high_risk_thresholds": torch.tensor(self.high_risk_thresholds, dtype=head.dtype, device=device),
                "future_aux_horizon_steps": torch.tensor(self.future_aux_horizon_steps, dtype=torch.long, device=device),
                "rise_thresholds": rise_thr,
                "fall_thresholds": fall_thr,
                "ordinal_bins": self.ordinal_bins_norm.to(device),
                "coarse_band_bins": self.coarse_band_bins_norm.to(device),
                "coarse_band_centers": self.coarse_band_centers_norm.to(device),
                "use_static": torch.tensor(self.use_static, device=device),
                "stream_prepend_calibration": torch.tensor(self.stream_prepend_calibration, device=device),
                "stream_calib_condition_mode": self.stream_calib_condition_mode,
                "stream_calib_condition_strength": torch.tensor(self.stream_calib_condition_strength, device=device),
                "calibration_encoder_mode": self.calibration_encoder_mode,
                "calibration_fusion_mode": self.calibration_fusion_mode,
                "calibration_repr_dim": torch.tensor(self.calibration_repr_dim, device=device),
                "stream_history_start": torch.tensor(stream_history_start, device=device),
                "stream_update_start_offset": torch.tensor(int(self.calibration_steps) - int(stream_history_start), device=device),
                "current_head_mode": self.current_head_mode,
                "current_trajectory_zero_index": torch.tensor(self.current_trajectory_zero_index, device=device),
                "ordinal_head_mode": self.ordinal_head_mode,
                "decoder_context_mode": self.decoder_context_mode,
                "decoder_temporal_context": self.decoder_temporal_context,
                "decoder_temporal_layers": torch.tensor(self.decoder_temporal_layers, device=device),
                "risk_head_enabled": torch.tensor(self.risk_head_enabled, device=device),
                "fall_risk_head_enabled": torch.tensor(self.fall_risk_head_enabled, device=device),
                "high_risk_head_enabled": torch.tensor(self.high_risk_head_enabled, device=device),
                "regime_head_enabled": torch.tensor(self.regime_head_enabled, device=device),
                "uncertainty_head_enabled": torch.tensor(self.uncertainty_head_enabled, device=device),
                "calibration_summary_fusion_enabled": torch.tensor(
                    self.calibration_summary_fusion_enabled,
                    device=device,
                ),
                "calibration_summary_fusion_mode": self.calibration_summary_fusion_mode,
                "calibration_summary_fusion_feature_dim": torch.tensor(
                    self.calibration_summary_fusion_feature_dim,
                    device=device,
                ),
                "calibration_summary_fusion_strength": torch.tensor(
                    self.calibration_summary_fusion_strength,
                    device=device,
                ),
                "calibration_summary_fusion_gate": head.new_zeros((bsz, 0)),
                "calibration_summary_fusion_delta_norm": head.new_zeros((bsz, 0)),
                "current_low_suppressor_enabled": torch.tensor(self.current_low_suppressor_enabled, device=device),
                "fds_enabled": torch.tensor(self.fds_enabled, device=device),
                "fds_applied_points": torch.tensor(0, device=device),
                "fds_updated_points": torch.tensor(0, device=device),
            }

        motion = append_motion_features(head, self.motion_feature_mode)
        stream_history_start = self._stream_history_start()
        stream_update_start_offset = int(self.calibration_steps) - int(stream_history_start)
        stream_motion = self._append_stream_time_features(motion, stream_history_start)
        h0 = self.init_state(z_calib).unsqueeze(0)
        state_seq = self._run_stream(
            stream_motion,
            h0,
            z_calib,
            model_base_fms,
            update_start_offset=stream_update_start_offset,
        )
        state_seq = self._stream_context(state_seq)
        offsets = (positions - int(self.calibration_steps)).clamp_min(0)
        state_at_positions = state_seq.index_select(1, offsets)
        parts = [state_at_positions]
        if self.decoder_context_mode == "fused":
            parts.append(z_calib.unsqueeze(1).expand(-1, pred_steps, -1))
        if self.motion_stats_encoder is not None:
            motion_stats = _window_motion_stats(head, positions, self.recent_steps)
            parts.append(self.motion_stats_encoder(motion_stats))
        z_static = self._encode_static(static, bsz)
        if z_static is not None:
            parts.append(z_static.unsqueeze(1).expand(-1, pred_steps, -1))
        fused = self.fusion(torch.cat(parts, dim=-1))
        if self.decoder_temporal_blocks is not None:
            fused = self.decoder_temporal_blocks(fused)
        summary_fusion_gate = head.new_zeros((bsz, pred_steps))
        summary_fusion_delta_norm = head.new_zeros((bsz, pred_steps))
        if self.calibration_summary_fusion is not None:
            if calibration_residual_features is None:
                raise ValueError(
                    "calibration_residual_features must be provided when calibration_summary_fusion_enabled=true."
                )
            summary_features = calibration_residual_features.to(device=device, dtype=fused.dtype)
            if calibration_residual_feature_mask is not None:
                summary_mask = calibration_residual_feature_mask.to(device=device, dtype=fused.dtype)
                if summary_mask.shape != summary_features.shape:
                    raise ValueError(
                        "calibration_residual_feature_mask must match calibration_residual_features shape, "
                        f"got {summary_mask.shape} vs {summary_features.shape}."
                    )
                summary_features = torch.where(summary_mask > 0, summary_features, torch.zeros_like(summary_features))
            fused, summary_fusion_delta, summary_fusion_gate = self.calibration_summary_fusion(fused, summary_features)
            summary_fusion_delta_norm = summary_fusion_delta.norm(dim=-1)
        fds_applied_points = 0
        fds_updated_points = 0
        if (
            self.fds_module is not None
            and self.training
            and fds_labels_raw is not None
            and fds_mask is not None
        ):
            labels_raw = fds_labels_raw.to(device=device, dtype=fused.dtype)
            label_mask = fds_mask.to(device=device).bool() & mask.bool()
            if labels_raw.shape != fused.shape[:2]:
                raise ValueError(f"fds_labels_raw must be [B,{pred_steps}], got {labels_raw.shape}.")
            if label_mask.shape != fused.shape[:2]:
                raise ValueError(f"fds_mask must be [B,{pred_steps}], got {label_mask.shape}.")
            if bool(fds_update):
                fds_updated_points = self.fds_module.update_epoch_stats(fused.detach(), labels_raw.detach(), label_mask)
            if bool(fds_apply):
                fused, fds_applied_points = self.fds_module.apply(fused, labels_raw, label_mask)
        current_outputs = compute_current_head_outputs(
            fused=fused,
            z_calib=z_calib,
            model_base_fms=model_base_fms,
            ordinal_bins_norm=self.ordinal_bins_norm,
            current_head_mode=self.current_head_mode,
            ordinal_head_mode=self.ordinal_head_mode,
            current_delta_scale=self.current_delta_scale,
            current_anchor_delta_growth_scale=self.current_anchor_delta_growth_scale,
            current_anchor_delta_growth_horizon_seconds=self.current_anchor_delta_growth_horizon_seconds,
            current_anchor_delta_growth_power=self.current_anchor_delta_growth_power,
            sampling_interval=self.sampling_interval,
            calibration_steps=self.calibration_steps,
            positions=positions,
            fms_combine_weight_ordinal=self.fms_combine_weight_ordinal,
            current_reg_head=self.current_reg_head,
            ordinal_head=self.ordinal_head,
            current_level_head=self.current_level_head,
            current_delta_head=self.current_delta_head,
            current_gate_head=self.current_gate_head,
            session_drift_head=self.session_drift_head,
            current_residual_delta_head=self.current_residual_delta_head,
            person_dynamic_head=self.person_dynamic_head,
            person_bias_head=self.person_bias_head,
            person_scale_head=self.person_scale_head,
            person_speed_head=self.person_speed_head,
            current_trajectory_head=self.current_trajectory_head,
            current_trajectory_offsets=torch.tensor(self.current_trajectory_offsets, dtype=torch.long, device=device),
            current_trajectory_zero_index=self.current_trajectory_zero_index,
            current_regime_gate_head=self.current_regime_gate_head,
            current_regime_expert_head=self.current_regime_expert_head,
            current_state_delta_head=self.current_state_delta_head,
            current_state_leak_head=self.current_state_leak_head,
            current_state_equilibrium_head=self.current_state_equilibrium_head,
            current_range_delta_head=self.current_range_delta_head,
            current_range_level_head=self.current_range_level_head,
            current_range_gate_head=self.current_range_gate_head,
            current_range_scale_head=self.current_range_scale_head,
            current_range_guard_head=self.current_range_guard_head,
            current_calib_prior_gate_head=self.current_calib_prior_gate_head,
            current_calib_prior_cap_head=self.current_calib_prior_cap_head,
            current_anchor_delta_head=self.current_anchor_delta_head,
            current_anchor_gate_head=self.current_anchor_gate_head,
            current_range_guard_low_threshold=self.current_range_guard_low_threshold,
            current_range_guard_temperature=self.current_range_guard_temperature,
            current_range_guard_floor=self.current_range_guard_floor,
            current_range_guard_cap=self.current_range_guard_cap,
            current_range_guard_cap_strength=self.current_range_guard_cap_strength,
        )
        risk_context = current_outputs.pop("risk_context")
        current = current_outputs["current"]
        current_reg = current_outputs["current_reg"]
        current_ordinal = current_outputs["current_ordinal"]
        ordinal_logits = current_outputs["ordinal_logits"]
        ordinal_probs = current_outputs["ordinal_probs"]
        if self.current_session_affine_head is not None:
            current_pre_session_affine = current
            session_affine_parts = [z_calib]
            if z_static is not None:
                session_affine_parts.append(z_static)
            session_affine_features = torch.cat(session_affine_parts, dim=-1)
            current, session_affine_scale, session_affine_bias = self.current_session_affine_head(
                current_pre_session_affine,
                session_affine_features,
            )
            current_outputs["current"] = current
            current_outputs["current_pre_session_affine"] = current_pre_session_affine
            current_outputs["current_session_affine_scale"] = session_affine_scale
            current_outputs["current_session_affine_bias"] = session_affine_bias
        if self.current_affine_head is not None:
            current_pre_affine = current
            base_expanded = model_base_fms.view(bsz, 1).expand(-1, pred_steps)
            absolute_time = positions.to(dtype=fused.dtype).view(1, -1).expand(bsz, -1) * self.sampling_interval / 210.0
            since_calib_time = (
                (positions - int(self.calibration_steps)).clamp_min(0).to(dtype=fused.dtype).view(1, -1).expand(bsz, -1)
                * self.sampling_interval
                / 210.0
            )
            scalar_context = torch.stack(
                [
                    current,
                    current_reg,
                    current_ordinal,
                    base_expanded,
                    absolute_time,
                    since_calib_time,
                ],
                dim=-1,
            )
            affine_features = torch.cat(
                [
                    fused,
                    z_calib.unsqueeze(1).expand(-1, pred_steps, -1),
                    scalar_context,
                ],
                dim=-1,
            )
            current, affine_scale, affine_bias = self.current_affine_head(current_pre_affine, affine_features)
            current_outputs["current"] = current
            current_outputs["current_pre_affine"] = current_pre_affine
            current_outputs["current_affine_scale"] = affine_scale
            current_outputs["current_affine_bias"] = affine_bias
        if self.current_binned_affine_head is not None:
            current_pre_binned_affine = current
            current_time_seconds = positions.to(dtype=fused.dtype) * self.sampling_interval
            current, binned_scale, binned_bias, binned_bin = self.current_binned_affine_head(
                current_pre_binned_affine,
                model_base_fms,
                current_time_seconds,
            )
            current_outputs["current"] = current
            current_outputs["current_pre_binned_affine"] = current_pre_binned_affine
            current_outputs["current_binned_affine_scale"] = binned_scale
            current_outputs["current_binned_affine_bias"] = binned_bias
            current_outputs["current_binned_affine_bin"] = binned_bin
        if self.calibration_residual_adapter is not None:
            if calibration_residual_features is None:
                raise ValueError(
                    "calibration_residual_features must be provided when calibration_residual_adapter_enabled=true."
                )
            residual_features = calibration_residual_features.to(device=device, dtype=fused.dtype)
            if calibration_residual_feature_mask is not None:
                residual_mask = calibration_residual_feature_mask.to(device=device, dtype=fused.dtype)
                if residual_mask.shape != residual_features.shape:
                    raise ValueError(
                        "calibration_residual_feature_mask must match calibration_residual_features shape, "
                        f"got {residual_mask.shape} vs {residual_features.shape}."
                    )
                residual_features = torch.where(residual_mask > 0, residual_features, torch.zeros_like(residual_features))
            current_pre_residual_adapter = current
            current, residual_correction, residual_gate = self.calibration_residual_adapter(
                current_pre_residual_adapter,
                residual_features,
                positions,
                int(self.calibration_steps),
                float(self.sampling_interval),
                model_base_fms,
            )
            current_outputs["current"] = current
            current_outputs["current_pre_residual_adapter"] = current_pre_residual_adapter
            current_outputs["current_residual_adapter_correction"] = residual_correction
            current_outputs["current_residual_adapter_gate"] = residual_gate
        if self.current_low_suppressor_head is not None:
            current_pre_low_suppressor = current
            base_expanded = model_base_fms.view(bsz, 1).expand(-1, pred_steps)
            absolute_time = positions.to(dtype=fused.dtype).view(1, -1).expand(bsz, -1) * self.sampling_interval / 210.0
            since_calib_time = (
                (positions - int(self.calibration_steps)).clamp_min(0).to(dtype=fused.dtype).view(1, -1).expand(bsz, -1)
                * self.sampling_interval
                / 210.0
            )
            scalar_context = torch.stack(
                [
                    current,
                    current_reg,
                    current_ordinal,
                    base_expanded,
                    absolute_time,
                    since_calib_time,
                ],
                dim=-1,
            )
            low_suppressor_features = torch.cat(
                [
                    fused,
                    z_calib.unsqueeze(1).expand(-1, pred_steps, -1),
                    scalar_context,
                ],
                dim=-1,
            )
            current, low_correction, low_gate, low_gate_logits = self.current_low_suppressor_head(
                current_pre_low_suppressor,
                low_suppressor_features,
            )
            current_outputs["current"] = current
            current_outputs["current_pre_low_suppressor"] = current_pre_low_suppressor
            current_outputs["current_low_suppressor_correction"] = low_correction
            current_outputs["current_low_suppressor_gate"] = low_gate
            current_outputs["current_low_suppressor_gate_logits"] = low_gate_logits
        aux_current = {
            key: value
            for key, value in current_outputs.items()
            if key not in {"current", "current_reg", "current_ordinal", "ordinal_logits", "ordinal_probs"}
        }
        if self.risk_head_enabled:
            if self.risk_temporal_blocks is not None:
                assert self.risk_context_projection is not None
                risk_context = self.risk_temporal_blocks(self.risk_context_projection(risk_context))
            assert self.risk_head is not None
            risk_logits = self.risk_head(risk_context)
            risk_probs = torch.sigmoid(risk_logits)
        else:
            if (self.fall_risk_head_enabled or self.high_risk_head_enabled) and self.risk_temporal_blocks is not None:
                assert self.risk_context_projection is not None
                risk_context = self.risk_temporal_blocks(self.risk_context_projection(risk_context))
            risk_logits = head.new_zeros((bsz, pred_steps, len(self.rise_horizon_steps)))
            risk_probs = torch.zeros_like(risk_logits)
        if self.fall_risk_head_enabled:
            assert self.fall_risk_head is not None
            fall_risk_logits = self.fall_risk_head(risk_context)
            fall_risk_probs = torch.sigmoid(fall_risk_logits)
        else:
            fall_risk_logits = head.new_zeros((bsz, pred_steps, len(self.fall_horizon_steps)))
            fall_risk_probs = torch.zeros_like(fall_risk_logits)
        if self.high_risk_head is not None:
            high_risk_logits_flat = self.high_risk_head(risk_context)
            high_risk_logits = high_risk_logits_flat.view(
                bsz,
                pred_steps,
                len(self.high_risk_horizon_steps),
                len(self.high_risk_thresholds),
            )
            high_risk_probs = torch.sigmoid(high_risk_logits)
        else:
            high_risk_logits = head.new_zeros(
                (bsz, pred_steps, len(self.high_risk_horizon_steps), len(self.high_risk_thresholds))
            )
            high_risk_probs = torch.zeros_like(high_risk_logits)
        future_aux_count = len(self.future_aux_horizon_steps)
        if self.future_aux_head is not None and self.event_aux_head is not None:
            future_aux = torch.sigmoid(self.future_aux_head(fused))
            event_logits = self.event_aux_head(fused).view(bsz, pred_steps, future_aux_count, 3)
            event_probs = torch.softmax(event_logits, dim=-1)
        else:
            future_aux = head.new_zeros((bsz, pred_steps, 0))
            event_logits = head.new_zeros((bsz, pred_steps, 0, 3))
            event_probs = head.new_zeros((bsz, pred_steps, 0, 3))
        if self.coarse_band_head is not None:
            coarse_band_logits = self.coarse_band_head(fused)
            coarse_band_probs = torch.softmax(coarse_band_logits, dim=-1)
        else:
            coarse_band_logits = head.new_zeros((bsz, pred_steps, 0))
            coarse_band_probs = head.new_zeros((bsz, pred_steps, 0))
        current_pre_coarse_residual = current
        current_coarse_residual = head.new_zeros((bsz, pred_steps))
        coarse_residual_values = head.new_zeros((bsz, pred_steps, coarse_band_probs.shape[-1]))
        if self.coarse_residual_head is not None and coarse_band_probs.shape[-1] > 0:
            coarse_residual_raw = self.coarse_residual_head(fused)
            centers = self.coarse_band_centers_norm.to(device=device, dtype=fused.dtype).view(1, 1, -1)
            coarse_residual_values = torch.clamp(
                centers + self.coarse_residual_range * torch.tanh(coarse_residual_raw),
                0.0,
                1.0,
            )
            current_coarse_residual = (coarse_band_probs * coarse_residual_values).sum(dim=-1)
            if self.coarse_residual_combine_weight > 0.0:
                w_res = float(self.coarse_residual_combine_weight)
                current = torch.clamp((1.0 - w_res) * current + w_res * current_coarse_residual, 0.0, 1.0)
        if self.regime_head is not None:
            regime_logits = self.regime_head(fused)
            regime_probs = torch.softmax(regime_logits, dim=-1)
        else:
            regime_logits = head.new_zeros((bsz, pred_steps, 0))
            regime_probs = head.new_zeros((bsz, pred_steps, 0))
        if self.uncertainty_head is not None:
            current_log_sigma = self.uncertainty_head(fused).squeeze(-1).clamp(
                self.uncertainty_min_log_sigma,
                self.uncertainty_max_log_sigma,
            )
            current_sigma = torch.exp(current_log_sigma)
        else:
            current_log_sigma = head.new_zeros((bsz, pred_steps))
            current_sigma = head.new_zeros((bsz, pred_steps))
        out = {
            "current": current,
            "current_reg": current_reg,
            "current_ordinal": current_ordinal,
            "ordinal_logits": ordinal_logits,
            "ordinal_probs": ordinal_probs,
            "coarse_band_logits": coarse_band_logits,
            "coarse_band_probs": coarse_band_probs,
            "current_coarse_residual": current_coarse_residual,
            "current_pre_coarse_residual": current_pre_coarse_residual,
            "coarse_residual_values": coarse_residual_values,
            "regime_logits": regime_logits,
            "regime_probs": regime_probs,
            "current_log_sigma": current_log_sigma,
            "current_sigma": current_sigma,
            "risk_logits": risk_logits,
            "risk_probs": risk_probs,
            "fall_risk_logits": fall_risk_logits,
            "fall_risk_probs": fall_risk_probs,
            "high_risk_logits": high_risk_logits,
            "high_risk_probs": high_risk_probs,
            "future_aux": future_aux,
            "event_logits": event_logits,
            "event_probs": event_probs,
            "mask": mask,
            "prediction_start": torch.tensor(start, device=device),
            "rise_horizon_steps": rise_h,
            "fall_horizon_steps": fall_h,
            "high_risk_horizon_steps": torch.tensor(self.high_risk_horizon_steps, dtype=torch.long, device=device),
            "high_risk_thresholds": torch.tensor(self.high_risk_thresholds, dtype=head.dtype, device=device),
            "future_aux_horizon_steps": torch.tensor(self.future_aux_horizon_steps, dtype=torch.long, device=device),
            "rise_thresholds": rise_thr,
            "fall_thresholds": fall_thr,
            "ordinal_bins": self.ordinal_bins_norm.to(device),
            "coarse_band_bins": self.coarse_band_bins_norm.to(device),
            "coarse_band_centers": self.coarse_band_centers_norm.to(device),
            "use_static": torch.tensor(self.use_static, device=device),
            "z_calib_norm": z_calib.norm(dim=-1),
            "state_norm": state_at_positions.norm(dim=-1),
            "calibration_end_fms": base_fms,
            "model_anchor_fms": model_base_fms,
            "state_feedback_mode": self.state_feedback_mode,
            "motion_stats_branch": torch.tensor(self.motion_stats_branch, device=device),
            "motion_encoder_context": self.motion_encoder_context,
            "motion_encoder_layers": torch.tensor(self.motion_encoder_layers, device=device),
            "stream_context_mode": self.stream_context_mode,
            "stream_prepend_calibration": torch.tensor(self.stream_prepend_calibration, device=device),
            "stream_calib_condition_mode": self.stream_calib_condition_mode,
            "stream_calib_condition_strength": torch.tensor(self.stream_calib_condition_strength, device=device),
            "calibration_encoder_mode": self.calibration_encoder_mode,
            "calibration_fusion_mode": self.calibration_fusion_mode,
            "calibration_repr_dim": torch.tensor(self.calibration_repr_dim, device=device),
            "stream_history_start": torch.tensor(stream_history_start, device=device),
            "stream_update_start_offset": torch.tensor(stream_update_start_offset, device=device),
            "current_head_mode": self.current_head_mode,
            "current_anchor_delta_growth_scale": torch.tensor(
                self.current_anchor_delta_growth_scale,
                device=device,
            ),
            "current_anchor_delta_growth_horizon_seconds": torch.tensor(
                self.current_anchor_delta_growth_horizon_seconds,
                device=device,
            ),
            "current_anchor_delta_growth_power": torch.tensor(
                self.current_anchor_delta_growth_power,
                device=device,
            ),
            "current_trajectory_zero_index": torch.tensor(self.current_trajectory_zero_index, device=device),
            "ordinal_head_mode": self.ordinal_head_mode,
            "decoder_context_mode": self.decoder_context_mode,
            "decoder_temporal_context": self.decoder_temporal_context,
            "decoder_temporal_layers": torch.tensor(self.decoder_temporal_layers, device=device),
            "decoder_feature_dim": torch.tensor(self.decoder_feature_dim, device=device),
            "fds_enabled": torch.tensor(self.fds_enabled, device=device),
            "fds_applied_points": torch.tensor(fds_applied_points, device=device),
            "fds_updated_points": torch.tensor(fds_updated_points, device=device),
            "risk_temporal_context": self.risk_temporal_context,
            "risk_head_enabled": torch.tensor(self.risk_head_enabled, device=device),
            "fall_risk_head_enabled": torch.tensor(self.fall_risk_head_enabled, device=device),
            "high_risk_head_enabled": torch.tensor(self.high_risk_head_enabled, device=device),
            "regime_head_enabled": torch.tensor(self.regime_head_enabled, device=device),
            "regime_class_count": torch.tensor(self.regime_class_count, device=device),
            "uncertainty_head_enabled": torch.tensor(self.uncertainty_head_enabled, device=device),
            "uncertainty_min_log_sigma": torch.tensor(self.uncertainty_min_log_sigma, device=device),
            "uncertainty_max_log_sigma": torch.tensor(self.uncertainty_max_log_sigma, device=device),
            "deep_tcn_dilations": torch.tensor(self.deep_tcn_dilations, device=device),
            "calibration_deep_tcn_dilations": torch.tensor(self.calibration_deep_tcn_dilations, device=device),
            "calibration_tcn_adaptive_dilations": torch.tensor(
                self.calibration_tcn_adaptive_dilations,
                device=device,
            ),
            "calibration_tcn_rf_steps": torch.tensor(self.calibration_tcn_rf_steps, device=device),
            "calibration_tcn_pad_steps": torch.tensor(self.calibration_tcn_pad_steps, device=device),
            "decoder_hidden_dim": torch.tensor(self.decoder_hidden_dim or 0, device=device),
            "calib_fms_dropout": torch.tensor(self.calib_fms_dropout, device=device),
            "calibration_end_fms_dropout": torch.tensor(self.calibration_end_fms_dropout, device=device),
            "calibration_summary_fusion_enabled": torch.tensor(
                self.calibration_summary_fusion_enabled,
                device=device,
            ),
            "calibration_summary_fusion_mode": self.calibration_summary_fusion_mode,
            "calibration_summary_fusion_feature_dim": torch.tensor(
                self.calibration_summary_fusion_feature_dim,
                device=device,
            ),
            "calibration_summary_fusion_strength": torch.tensor(
                self.calibration_summary_fusion_strength,
                device=device,
            ),
            "calibration_summary_fusion_gate": summary_fusion_gate,
            "calibration_summary_fusion_delta_norm": summary_fusion_delta_norm,
            "calibration_residual_adapter_enabled": torch.tensor(self.calibration_residual_adapter_enabled, device=device),
            "calibration_residual_adapter_mode": self.calibration_residual_adapter_mode,
            "calibration_residual_feature_dim": torch.tensor(self.calibration_residual_feature_dim, device=device),
            "calibration_residual_delta_range": torch.tensor(self.calibration_residual_delta_range, device=device),
            "calibration_residual_decay_seconds": torch.tensor(self.calibration_residual_decay_seconds, device=device),
            "calibration_residual_gate_low_threshold": torch.tensor(
                self.calibration_residual_gate_low_threshold,
                device=device,
            ),
            "calibration_residual_gate_high_threshold": torch.tensor(
                self.calibration_residual_gate_high_threshold,
                device=device,
            ),
            "calibration_residual_gate_anchor_threshold": torch.tensor(
                self.calibration_residual_gate_anchor_threshold,
                device=device,
            ),
            "calibration_residual_gate_temperature": torch.tensor(
                self.calibration_residual_gate_temperature,
                device=device,
            ),
            "current_low_suppressor_enabled": torch.tensor(self.current_low_suppressor_enabled, device=device),
            "current_low_suppressor_delta_range": torch.tensor(self.current_low_suppressor_delta_range, device=device),
            "current_low_suppressor_gate_init_bias": torch.tensor(
                self.current_low_suppressor_gate_init_bias,
                device=device,
            ),
        }
        out.update(aux_current)
        return out


class StartOnlySequenceForecasterBase(nn.Module):
    ANCHOR_MODES = {"none", "calibration_end", "recent_start_observed", "sparse_observed"}

    def __init__(
        self,
        head_dim: int = 6,
        calibration_steps: int = 180,
        horizon_steps: int = 10,
        recent_steps: int = 60,
        sampling_interval: float = 0.5,
        horizon_seconds: Optional[float] = None,
        anchor_mode: str = "none",
        anchor_interval_seconds: float = 0.0,
        predict_delta_from_anchor: bool = False,
        use_static: bool = False,
        static_dim: int = 4,
        static_hidden_dim: Optional[int] = None,
        static_dropout: float = 0.1,
        multi_horizon: bool = False,
        horizon_set: Optional[Sequence[float]] = None,
        fms_context_mode: str = "start_only",
        d_model: int = 96,
        dropout: float = 0.05,
        per_horizon_heads: bool = True,
    ):
        super().__init__()
        anchor_mode = str(anchor_mode).lower()
        if anchor_mode not in self.ANCHOR_MODES:
            raise ValueError(f"anchor_mode must be one of {sorted(self.ANCHOR_MODES)}, got {anchor_mode!r}")
        fms_context_mode = normalize_fms_context_mode(fms_context_mode)
        uses_window_start_fms = _uses_window_start_fms(fms_context_mode)
        if predict_delta_from_anchor and anchor_mode == "none" and not uses_window_start_fms:
            raise ValueError("predict_delta_from_anchor requires an anchor mode or start_only context.")
        self.requires_full_fms = anchor_mode in {"sparse_observed", "recent_start_observed"} or uses_window_start_fms
        self.head_dim = int(head_dim)
        self.calibration_steps = int(calibration_steps)
        self.horizon_steps = int(horizon_steps)
        self.recent_steps = int(recent_steps)
        self.sampling_interval = float(sampling_interval)
        self.horizon_seconds = float(horizon_seconds if horizon_seconds is not None else self.horizon_steps * self.sampling_interval)
        self.anchor_mode = anchor_mode
        self.fms_context_mode = fms_context_mode
        self.uses_window_start_fms = uses_window_start_fms
        self.anchor_interval_seconds = float(anchor_interval_seconds)
        self.anchor_interval_steps = max(1, int(round(self.anchor_interval_seconds / self.sampling_interval)))
        self.predict_delta_from_anchor = bool(predict_delta_from_anchor)
        self.use_static = bool(use_static)
        self.static_dim = int(static_dim)
        self.multi_horizon = bool(multi_horizon)
        self.per_horizon_heads = bool(per_horizon_heads)
        self.horizon_set = [float(v) for v in horizon_set] if horizon_set else None
        if self.multi_horizon and not self.horizon_set:
            raise ValueError("multi_horizon=True requires horizon_set.")
        self.horizon_seconds_list = self.horizon_set if self.multi_horizon else [self.horizon_seconds]
        self.horizon_steps_list = [max(1, int(round(v / self.sampling_interval))) for v in self.horizon_seconds_list]
        self.output_dim = len(self.horizon_steps_list)
        self.d_model = int(d_model)
        self.dropout = float(dropout)
        hidden_static = int(static_hidden_dim if static_hidden_dim is not None else self.d_model)
        self.static_encoder = StaticEncoder(self.static_dim, hidden_static, static_dropout) if self.use_static else None
        self.static_projection = (
            nn.Sequential(nn.Linear(hidden_static, self.d_model), nn.GELU(), nn.LayerNorm(self.d_model))
            if self.use_static and hidden_static != self.d_model
            else None
        )
        self.anchor_encoder = (
            nn.Sequential(
                nn.Linear(2, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
            )
            if self.anchor_mode != "none" or self.uses_window_start_fms
            else None
        )
        self.horizon_encoder = nn.Sequential(nn.Linear(1, self.d_model), nn.GELU(), nn.LayerNorm(self.d_model))

    def _positions(self, lengths: torch.Tensor, device: torch.device) -> Tuple[int, torch.Tensor, torch.Tensor]:
        return _prediction_positions_for(
            lengths,
            self.calibration_steps,
            self.recent_steps,
            self.horizon_steps_list,
            device,
            self.multi_horizon,
        )

    def _anchor_context(
        self,
        fms: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        anchor_idx = _anchor_indices_for(
            positions,
            self.anchor_mode,
            self.calibration_steps,
            self.recent_steps,
            self.anchor_interval_steps,
        )
        if anchor_idx is None and self.uses_window_start_fms:
            anchor_idx = (positions - self.recent_steps + 1).clamp_min(0)
        if anchor_idx is None:
            return None, None, None
        anchor_label = self.anchor_mode if self.anchor_mode != "none" else "window_start_fms"
        anchor_fms, actual_anchor_idx = _gather_anchor_fms_safe(fms, anchor_idx, anchor_label)
        time_since = (positions.unsqueeze(0) - actual_anchor_idx).to(fms.dtype) * self.sampling_interval
        anchor_context = torch.stack([anchor_fms, time_since / 120.0], dim=-1)
        return anchor_context, anchor_fms, actual_anchor_idx

    def _encode_static(self, static: Optional[torch.Tensor], pred_steps: int, batch_size: int) -> Optional[torch.Tensor]:
        if not self.use_static:
            return None
        if static is None:
            raise ValueError(f"{self.__class__.__name__} was created with use_static=True, but static tensor was not provided.")
        if static.ndim != 2 or static.shape[0] != batch_size or static.shape[1] != self.static_dim:
            raise ValueError(f"static must be [B,{self.static_dim}], got {static.shape}")
        assert self.static_encoder is not None
        z_static = self.static_encoder(static)
        if self.static_projection is not None:
            z_static = self.static_projection(z_static)
        return z_static.unsqueeze(1).expand(-1, pred_steps, -1)

    def _empty_output(self, head: torch.Tensor, mask: torch.Tensor, start: int) -> Dict[str, torch.Tensor]:
        bsz = head.shape[0]
        empty = head.new_zeros((bsz, 0, self.output_dim)) if self.multi_horizon else head.new_zeros((bsz, 0))
        return {
            "future": empty,
            "mask": mask,
            "prediction_start": torch.tensor(start, device=head.device),
            "horizon_steps_list": torch.tensor(self.horizon_steps_list, dtype=torch.long, device=head.device),
            "use_static": torch.tensor(self.use_static, device=head.device),
        }

    def _prepare_common(
        self,
        head: torch.Tensor,
        y_calib: torch.Tensor,
        lengths: Optional[torch.Tensor],
        static: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        assert head.ndim == 3, f"head must be [B,T,{self.head_dim}], got {head.shape}"
        if head.shape[-1] != self.head_dim:
            raise ValueError(f"expected head_dim={self.head_dim}, got {head.shape[-1]}")
        bsz, steps, _ = head.shape
        device = head.device
        lengths = lengths.to(device) if lengths is not None else torch.full((bsz,), steps, dtype=torch.long, device=device)
        if y_calib.ndim != 2:
            raise ValueError(f"FMS input must be [B,T] or [B,C], got {y_calib.shape}")
        fms = y_calib.to(device)
        if self.requires_full_fms and fms.shape[1] < steps:
            raise ValueError(
                f"fms_context_mode={self.fms_context_mode!r}, anchor_mode={self.anchor_mode!r} "
                "requires full FMS input through current time."
            )
        start, positions, mask = self._positions(lengths, device)
        pred_steps = int(positions.numel())
        anchor_context, anchor_fms, actual_anchor_idx = self._anchor_context(fms, positions) if pred_steps else (None, None, None)
        z_anchor = self.anchor_encoder(anchor_context) if anchor_context is not None and self.anchor_encoder is not None else None
        z_static = self._encode_static(static, pred_steps, bsz)
        return {
            "fms": fms,
            "lengths": lengths,
            "start": start,
            "positions": positions,
            "mask": mask,
            "pred_steps": pred_steps,
            "anchor_fms": anchor_fms,
            "actual_anchor_idx": actual_anchor_idx,
            "z_anchor": z_anchor,
            "z_static": z_static,
        }

    def _finalize(self, raw_pred: torch.Tensor, common: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        anchor_fms = common["anchor_fms"]
        if self.predict_delta_from_anchor:
            assert anchor_fms is not None
            base = anchor_fms.unsqueeze(-1) if self.multi_horizon else anchor_fms
            pred = torch.clamp(base + 0.5 * torch.tanh(raw_pred), 0.0, 1.0)
        else:
            pred = torch.sigmoid(raw_pred)
        out: Dict[str, torch.Tensor] = {
            "future": pred,
            "mask": common["mask"],
            "prediction_start": torch.tensor(common["start"], device=pred.device),
            "horizon_steps_list": torch.tensor(self.horizon_steps_list, dtype=torch.long, device=pred.device),
            "use_static": torch.tensor(self.use_static, device=pred.device),
        }
        if common["actual_anchor_idx"] is not None and anchor_fms is not None:
            out["anchor_index"] = common["actual_anchor_idx"]
            out["anchor_fms"] = anchor_fms
        return out


class LCSACrossAttentionForecaster(StartOnlySequenceForecasterBase):
    """Start-only LC-SA variant where each forecast query cross-attends to calibration tokens."""

    def __init__(
        self,
        head_dim: int = 6,
        d_model: int = 96,
        kernel_size: int = 3,
        dropout: float = 0.05,
        calib_dilations: Sequence[int] = (1, 2, 4, 8, 16),
        recent_dilations: Sequence[int] | str = "auto",
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 192,
        pooling: str = "mean",
        **kwargs: Any,
    ):
        super().__init__(head_dim=head_dim, d_model=d_model, dropout=dropout, **kwargs)
        if self.d_model % int(transformer_heads) != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by transformer_heads={transformer_heads}")
        self.kernel_size = int(kernel_size)
        self.calib_dilations = [int(v) for v in calib_dilations]
        self.recent_dilations = resolve_dilations(recent_dilations, self.recent_steps * self.sampling_interval)
        self.calibration_encoder = LCBranchTCN(self.head_dim + 1, self.d_model, self.calib_dilations, self.kernel_size, self.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(transformer_heads),
            dim_feedforward=int(transformer_ff_dim),
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.calibration_transformer = nn.TransformerEncoder(layer, num_layers=int(transformer_layers))
        self.recent_encoder = LCBranchTCN(self.head_dim, self.d_model, self.recent_dilations, self.kernel_size, self.dropout)
        self.recent_pool = SequencePooling(self.d_model, pooling)
        self.cross_attn = nn.MultiheadAttention(self.d_model, int(transformer_heads), dropout=self.dropout, batch_first=True)
        self.cross_dropout = nn.Dropout(self.dropout)
        self.cross_norm = nn.LayerNorm(self.d_model)
        fusion_dim = self.d_model * (3 + (1 if self.anchor_encoder is not None else 0) + (1 if self.use_static else 0))
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.head = nn.Linear(self.d_model, 1)
        self.horizon_heads = (
            nn.ModuleList([nn.Linear(self.d_model, 1) for _ in self.horizon_steps_list])
            if self.multi_horizon and self.per_horizon_heads
            else None
        )
        self.recent_rf_steps = tcn_receptive_field_steps(self.recent_dilations, self.kernel_size)
        self.recent_rf_seconds = self.recent_rf_steps * self.sampling_interval

    def _calibration_tokens(self, head: torch.Tensor, fms: torch.Tensor) -> torch.Tensor:
        context_fms = calibration_context_fms(fms, self.calibration_steps, self.fms_context_mode)
        calib_seq = torch.cat([head[:, : self.calibration_steps], context_fms.unsqueeze(-1)], dim=-1)
        return self.calibration_transformer(self.calibration_encoder(calib_seq))

    def _recent_at(self, head: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        encoded = self.recent_encoder(head)
        if self.recent_pool.mode == "last":
            return encoded.index_select(1, positions)
        window_idx = positions - self.recent_steps + 1
        if self.recent_pool.mode == "mean":
            cumsum = torch.cat([encoded.new_zeros((encoded.shape[0], 1, encoded.shape[2])), encoded.cumsum(dim=1)], dim=1)
            end = positions + 1
            sums = cumsum.index_select(1, end) - cumsum.index_select(1, window_idx)
            return sums / float(self.recent_steps)
        windows = encoded.unfold(dimension=1, size=self.recent_steps, step=1).permute(0, 1, 3, 2).contiguous()
        selected = windows.index_select(1, window_idx)
        bsz, pred_steps, recent_steps, dim = selected.shape
        pooled = self.recent_pool(selected.view(-1, recent_steps, dim))
        return pooled.view(bsz, pred_steps, dim)

    def _predict_one(self, z_recent: torch.Tensor, calib_tokens: torch.Tensor, common: Dict[str, Any], horizon_seconds: float, horizon_idx: Optional[int]) -> torch.Tensor:
        bsz, pred_steps, _ = z_recent.shape
        horizon_in = z_recent.new_full((bsz, pred_steps, 1), float(horizon_seconds) / 60.0)
        z_horizon = self.horizon_encoder(horizon_in)
        query = z_recent + z_horizon
        if common["z_anchor"] is not None:
            query = query + common["z_anchor"]
        if common["z_static"] is not None:
            query = query + common["z_static"]
        attended, _ = self.cross_attn(query, calib_tokens, calib_tokens, need_weights=False)
        z_calib = self.cross_norm(query + self.cross_dropout(attended))
        parts = [z_calib, z_recent, z_horizon]
        if common["z_anchor"] is not None:
            parts.append(common["z_anchor"])
        if common["z_static"] is not None:
            parts.append(common["z_static"])
        fused = self.fusion(torch.cat(parts, dim=-1))
        if self.horizon_heads is not None:
            assert horizon_idx is not None
            return self.horizon_heads[int(horizon_idx)](fused).squeeze(-1)
        return self.head(fused).squeeze(-1)

    def forward(
        self,
        head: torch.Tensor,
        y_calib: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        static: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        common = self._prepare_common(head, y_calib, lengths, static)
        if common["pred_steps"] == 0:
            return self._empty_output(head, common["mask"], common["start"])
        calib_tokens = self._calibration_tokens(head, common["fms"])
        z_recent = self._recent_at(head, common["positions"])
        if self.multi_horizon:
            raw = torch.stack(
                [
                    self._predict_one(z_recent, calib_tokens, common, horizon, idx)
                    for idx, horizon in enumerate(self.horizon_seconds_list)
                ],
                dim=-1,
            )
        else:
            raw = self._predict_one(z_recent, calib_tokens, common, self.horizon_seconds, None)
        out = self._finalize(raw, common)
        out["z_recent_norm"] = z_recent.norm(dim=-1).mean(dim=1)
        return out


class GRUStateMixerForecaster(StartOnlySequenceForecasterBase):
    """GRU-window model with calibration summary, start-FMS context, and horizon-conditioned heads."""

    def __init__(
        self,
        head_dim: int = 6,
        d_model: int = 96,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        gru_layers: int = 1,
        **kwargs: Any,
    ):
        super().__init__(head_dim=head_dim, d_model=d_model, dropout=dropout, **kwargs)
        self.hidden_dim = int(hidden_dim)
        self.gru_layers = int(gru_layers)
        self.calibration_gru = nn.GRU(
            input_size=self.head_dim + 1,
            hidden_size=self.hidden_dim,
            num_layers=self.gru_layers,
            batch_first=True,
            dropout=self.dropout if self.gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.calibration_projection = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.d_model), nn.GELU(), nn.LayerNorm(self.d_model))
        self.recent_gru = nn.GRU(
            input_size=self.head_dim,
            hidden_size=self.d_model,
            num_layers=self.gru_layers,
            batch_first=True,
            dropout=self.dropout if self.gru_layers > 1 else 0.0,
        )
        fusion_dim = self.d_model * (3 + (1 if self.anchor_encoder is not None else 0) + (1 if self.use_static else 0))
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.head = nn.Linear(self.d_model, 1)
        self.horizon_heads = (
            nn.ModuleList([nn.Linear(self.d_model, 1) for _ in self.horizon_steps_list])
            if self.multi_horizon and self.per_horizon_heads
            else None
        )

    def _calibration_state(self, head: torch.Tensor, fms: torch.Tensor) -> torch.Tensor:
        context_fms = calibration_context_fms(fms, self.calibration_steps, self.fms_context_mode)
        calib_seq = torch.cat([head[:, : self.calibration_steps], context_fms.unsqueeze(-1)], dim=-1)
        _, h = self.calibration_gru(calib_seq)
        last = torch.cat([h[-2], h[-1]], dim=-1) if self.calibration_gru.bidirectional else h[-1]
        return self.calibration_projection(last)

    def _recent_at(self, head: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        windows = head.unfold(dimension=1, size=self.recent_steps, step=1).permute(0, 1, 3, 2).contiguous()
        selected = windows.index_select(1, positions - self.recent_steps + 1)
        bsz, pred_steps, recent_steps, head_dim = selected.shape
        flat = selected.view(-1, recent_steps, head_dim)
        chunks = []
        for start in range(0, flat.shape[0], 2048):
            _, h = self.recent_gru(flat[start : start + 2048])
            chunks.append(h[-1])
        return torch.cat(chunks, dim=0).view(bsz, pred_steps, self.d_model)

    def _predict_one(self, z_calib: torch.Tensor, z_recent: torch.Tensor, common: Dict[str, Any], horizon_seconds: float, horizon_idx: Optional[int]) -> torch.Tensor:
        bsz, pred_steps, _ = z_recent.shape
        horizon_in = z_recent.new_full((bsz, pred_steps, 1), float(horizon_seconds) / 60.0)
        z_horizon = self.horizon_encoder(horizon_in)
        parts = [z_calib.unsqueeze(1).expand(-1, pred_steps, -1), z_recent, z_horizon]
        if common["z_anchor"] is not None:
            parts.append(common["z_anchor"])
        if common["z_static"] is not None:
            parts.append(common["z_static"])
        fused = self.fusion(torch.cat(parts, dim=-1))
        if self.horizon_heads is not None:
            assert horizon_idx is not None
            return self.horizon_heads[int(horizon_idx)](fused).squeeze(-1)
        return self.head(fused).squeeze(-1)

    def forward(
        self,
        head: torch.Tensor,
        y_calib: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        static: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        common = self._prepare_common(head, y_calib, lengths, static)
        if common["pred_steps"] == 0:
            return self._empty_output(head, common["mask"], common["start"])
        z_calib = self._calibration_state(head, common["fms"])
        z_recent = self._recent_at(head, common["positions"])
        if self.multi_horizon:
            raw = torch.stack(
                [self._predict_one(z_calib, z_recent, common, horizon, idx) for idx, horizon in enumerate(self.horizon_seconds_list)],
                dim=-1,
            )
        else:
            raw = self._predict_one(z_calib, z_recent, common, self.horizon_seconds, None)
        out = self._finalize(raw, common)
        out["z_calib_norm"] = z_calib.norm(dim=-1)
        return out


class MotionConvMixerBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.depthwise = CausalConv1d(d_model, kernel_size, dilation=1)
        self.pointwise = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm1(x + self.dropout(F.gelu(y)))
        return self.norm2(x + self.dropout(self.pointwise(x)))


class MotionConvMixerForecaster(StartOnlySequenceForecasterBase):
    """Compact motion-first architecture using causal ConvMixer blocks and calibration summary features."""

    def __init__(
        self,
        head_dim: int = 6,
        d_model: int = 96,
        hidden_dim: int = 128,
        kernel_size: int = 5,
        dropout: float = 0.05,
        transformer_layers: int = 3,
        **kwargs: Any,
    ):
        super().__init__(head_dim=head_dim, d_model=d_model, dropout=dropout, **kwargs)
        self.summary_dim = 8 + self.head_dim * 2
        self.calibration_summary = nn.Sequential(
            nn.Linear(self.summary_dim, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(self.dropout),
            nn.Linear(int(hidden_dim), self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.motion_input = nn.Sequential(nn.Linear(self.head_dim, self.d_model), nn.GELU(), nn.LayerNorm(self.d_model))
        self.blocks = nn.Sequential(
            *[MotionConvMixerBlock(self.d_model, int(kernel_size), self.dropout) for _ in range(int(transformer_layers))]
        )
        self.recent_pool = SequencePooling(self.d_model, "mean")
        fusion_dim = self.d_model * (3 + (1 if self.anchor_encoder is not None else 0) + (1 if self.use_static else 0))
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.head = nn.Linear(self.d_model, 1)
        self.horizon_heads = (
            nn.ModuleList([nn.Linear(self.d_model, 1) for _ in self.horizon_steps_list])
            if self.multi_horizon and self.per_horizon_heads
            else None
        )
        self.recent_rf_steps = 1 + int(transformer_layers) * (int(kernel_size) - 1)
        self.recent_rf_seconds = self.recent_rf_steps * self.sampling_interval

    def _calibration_state(self, head: torch.Tensor, fms: torch.Tensor) -> torch.Tensor:
        context_fms = calibration_context_fms(fms, self.calibration_steps, self.fms_context_mode)
        return self.calibration_summary(_calibration_summary_features(head, context_fms, self.calibration_steps))

    def _recent_at(self, head: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        encoded = self.blocks(self.motion_input(head))
        window_idx = positions - self.recent_steps + 1
        cumsum = torch.cat([encoded.new_zeros((encoded.shape[0], 1, encoded.shape[2])), encoded.cumsum(dim=1)], dim=1)
        end = positions + 1
        sums = cumsum.index_select(1, end) - cumsum.index_select(1, window_idx)
        return sums / float(self.recent_steps)

    def _predict_one(self, z_calib: torch.Tensor, z_recent: torch.Tensor, common: Dict[str, Any], horizon_seconds: float, horizon_idx: Optional[int]) -> torch.Tensor:
        bsz, pred_steps, _ = z_recent.shape
        horizon_in = z_recent.new_full((bsz, pred_steps, 1), float(horizon_seconds) / 60.0)
        z_horizon = self.horizon_encoder(horizon_in)
        parts = [z_calib.unsqueeze(1).expand(-1, pred_steps, -1), z_recent, z_horizon]
        if common["z_anchor"] is not None:
            parts.append(common["z_anchor"])
        if common["z_static"] is not None:
            parts.append(common["z_static"])
        fused = self.fusion(torch.cat(parts, dim=-1))
        if self.horizon_heads is not None:
            assert horizon_idx is not None
            return self.horizon_heads[int(horizon_idx)](fused).squeeze(-1)
        return self.head(fused).squeeze(-1)

    def forward(
        self,
        head: torch.Tensor,
        y_calib: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        static: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        common = self._prepare_common(head, y_calib, lengths, static)
        if common["pred_steps"] == 0:
            return self._empty_output(head, common["mask"], common["start"])
        z_calib = self._calibration_state(head, common["fms"])
        z_recent = self._recent_at(head, common["positions"])
        if self.multi_horizon:
            raw = torch.stack(
                [self._predict_one(z_calib, z_recent, common, horizon, idx) for idx, horizon in enumerate(self.horizon_seconds_list)],
                dim=-1,
            )
        else:
            raw = self._predict_one(z_calib, z_recent, common, self.horizon_seconds, None)
        out = self._finalize(raw, common)
        out["z_calib_norm"] = z_calib.norm(dim=-1)
        return out


def build_model(model_name: str, **kwargs) -> nn.Module:
    name = model_name.lower()
    anchor_kwargs = {
        "head_dim": kwargs.get("head_dim", 6),
        "calibration_steps": kwargs.get("calibration_steps", 180),
        "horizon_steps": kwargs.get("horizon_steps", 10),
        "recent_steps": kwargs.get("recent_steps", 60),
        "sampling_interval": kwargs.get("sampling_interval", 0.5),
        "horizon_seconds": kwargs.get("horizon_seconds", None),
        "anchor_mode": kwargs.get("anchor_mode", "sparse_observed"),
        "anchor_interval_seconds": kwargs.get("anchor_interval_seconds", 60.0),
        "predict_delta_from_anchor": kwargs.get("predict_delta_from_anchor", True),
        "use_static": kwargs.get("use_static", False),
        "static_dim": kwargs.get("static_dim", 4),
        "dropout": kwargs.get("dropout", 0.1),
        "hidden_dim": kwargs.get("hidden_dim", kwargs.get("d_model", 128)),
        "mlp_layers": kwargs.get("mlp_layers", None),
        "multi_horizon": kwargs.get("multi_horizon", False),
        "horizon_set": kwargs.get("horizon_set", None),
        "delta_scale": kwargs.get("delta_scale", 0.5),
        "fms_context_mode": kwargs.get("fms_context_mode", "calibration_history"),
    }
    if name in {"anchor_delta_mlp", "anchordeltamlp"}:
        return AnchorDeltaMLP(**anchor_kwargs)
    if name in {"anchor_delta_gru", "anchordeltagru"}:
        return AnchorDeltaGRU(
            **anchor_kwargs,
            gru_layers=kwargs.get("gru_layers", kwargs.get("num_layers", 1)),
        )
    if name in {"recent_tcn_summary_calib", "recent_tcn_summary", "recenttcn_summarycalib"}:
        return RecentTCNSummaryCalib(
            **anchor_kwargs,
            d_model=kwargs.get("d_model", 64),
            kernel_size=kwargs.get("kernel_size", 3),
            recent_dilations=kwargs.get("recent_dilations", "auto"),
            pooling=kwargs.get("pooling", "mean"),
        )
    if name in {"gated_fusion", "gated_fusion_forecaster", "gatedfusion"}:
        return GatedFusionForecaster(
            **anchor_kwargs,
            d_model=kwargs.get("d_model", 64),
            kernel_size=kwargs.get("kernel_size", 3),
            recent_dilations=kwargs.get("recent_dilations", "auto"),
            pooling=kwargs.get("pooling", "mean"),
            branch_dropout=kwargs.get("branch_dropout", 0.0),
            anchor_dropout=kwargs.get("anchor_dropout", 0.0),
        )
    if name in {"lc_sa_tcnformer", "lc-sa-tcnformer", "lcsatcnformer"}:
        return LCSATCNFormer(
            head_dim=kwargs.get("head_dim", 6),
            calibration_steps=kwargs.get("calibration_steps", 180),
            horizon_steps=kwargs.get("horizon_steps", 10),
            recent_steps=kwargs.get("recent_steps", 60),
            sampling_interval=kwargs.get("sampling_interval", 0.5),
            horizon_seconds=kwargs.get("horizon_seconds", None),
            d_model=kwargs.get("d_model", 64),
            kernel_size=kwargs.get("kernel_size", 3),
            dropout=kwargs.get("dropout", 0.1),
            calib_dilations=kwargs.get("calib_dilations", (1, 2, 4, 8, 16)),
            recent_dilations=kwargs.get("recent_dilations", "auto"),
            transformer_layers=kwargs.get("transformer_layers", 1),
            transformer_heads=kwargs.get("transformer_heads", 4),
            transformer_ff_dim=kwargs.get("transformer_ff_dim", 128),
            pooling=kwargs.get("pooling", "mean"),
            anchor_mode=kwargs.get("anchor_mode", "calibration_end"),
            anchor_interval_seconds=kwargs.get("anchor_interval_seconds", 60.0),
            predict_delta_from_anchor=kwargs.get("predict_delta_from_anchor", False),
            use_static=kwargs.get("use_static", False),
            static_dim=kwargs.get("static_dim", 4),
            static_hidden_dim=kwargs.get("static_hidden_dim", None),
            static_dropout=kwargs.get("static_dropout", 0.1),
            multi_horizon=kwargs.get("multi_horizon", False),
            horizon_set=kwargs.get("horizon_set", None),
            per_horizon_heads=kwargs.get("per_horizon_heads", False),
            horizon_encoder_dim=kwargs.get("horizon_encoder_dim", None),
            horizon_context_mode=kwargs.get("horizon_context_mode", "encoded"),
            start_fms_context_mode=kwargs.get("start_fms_context_mode", "encoded"),
            static_context_mode=kwargs.get("static_context_mode", "encoded"),
            forecast_head_mode=kwargs.get("forecast_head_mode", "level"),
            horizon_head_mode=kwargs.get("horizon_head_mode", "linear"),
            horizon_head_hidden_dim=kwargs.get("horizon_head_hidden_dim", None),
            motion_feature_mode=kwargs.get("motion_feature_mode", "none"),
            change_aux_head=kwargs.get("change_aux_head", False),
            fms_context_mode=kwargs.get("fms_context_mode", "calibration_history"),
        )
    if name in {"calib_init_state_forecaster", "calib_init_gru_state_forecaster", "calibinitstate"}:
        return CalibInitStateForecaster(
            head_dim=kwargs.get("head_dim", 6),
            calibration_steps=kwargs.get("calibration_steps", 180),
            horizon_steps=kwargs.get("horizon_steps", 10),
            recent_steps=kwargs.get("recent_steps", 60),
            sampling_interval=kwargs.get("sampling_interval", 0.5),
            horizon_seconds=kwargs.get("horizon_seconds", None),
            d_model=kwargs.get("d_model", 96),
            hidden_dim=kwargs.get("hidden_dim", 128),
            kernel_size=kwargs.get("kernel_size", 3),
            dropout=kwargs.get("dropout", 0.05),
            calib_dilations=kwargs.get("calib_dilations", (1, 2, 4, 8, 16)),
            transformer_layers=kwargs.get("transformer_layers", 1),
            transformer_heads=kwargs.get("transformer_heads", 4),
            transformer_ff_dim=kwargs.get("transformer_ff_dim", 192),
            pooling=kwargs.get("pooling", "mean"),
            anchor_mode=kwargs.get("anchor_mode", "none"),
            anchor_interval_seconds=kwargs.get("anchor_interval_seconds", 0.0),
            use_static=kwargs.get("use_static", False),
            static_dim=kwargs.get("static_dim", 4),
            static_hidden_dim=kwargs.get("static_hidden_dim", None),
            static_dropout=kwargs.get("static_dropout", 0.1),
            multi_horizon=kwargs.get("multi_horizon", False),
            horizon_set=kwargs.get("horizon_set", None),
            per_horizon_heads=kwargs.get("per_horizon_heads", True),
            forecast_head_mode=kwargs.get("forecast_head_mode", "level"),
            delta_scale=kwargs.get("delta_scale", 0.5),
            motion_feature_mode=kwargs.get("motion_feature_mode", "norm"),
            stream_time_features=kwargs.get("stream_time_features", False),
            stream_context_mode=kwargs.get("stream_context_mode", "gru"),
            calib_summary_features=kwargs.get("calib_summary_features", False),
            state_feedback_mode=kwargs.get("state_feedback_mode", "none"),
            session_context_mode=kwargs.get("session_context_mode", "none"),
            fms_context_mode=kwargs.get("fms_context_mode", "calibration_history"),
        )
    if name in {"online_fms_risk_tracker", "online_risk_tracker", "online_current_risk"}:
        return OnlineFMSRiskTracker(
            head_dim=kwargs.get("head_dim", 6),
            calibration_steps=kwargs.get("calibration_steps", 180),
            horizon_steps=kwargs.get("horizon_steps", 20),
            recent_steps=kwargs.get("recent_steps", 20),
            sampling_interval=kwargs.get("sampling_interval", 0.5),
            horizon_seconds=kwargs.get("horizon_seconds", None),
            rise_horizon_steps=kwargs.get("rise_horizon_steps", None),
            rise_thresholds=kwargs.get("rise_thresholds", None),
            fall_horizon_steps=kwargs.get("fall_horizon_steps", None),
            fall_thresholds=kwargs.get("fall_thresholds", None),
            high_risk_horizon_steps=kwargs.get("high_risk_horizon_steps", None),
            high_risk_thresholds=kwargs.get("high_risk_thresholds", None),
            future_aux_horizon_steps=kwargs.get("future_aux_horizon_steps", None),
            d_model=kwargs.get("d_model", 64),
            hidden_dim=kwargs.get("hidden_dim", 96),
            kernel_size=kwargs.get("kernel_size", 3),
            dropout=kwargs.get("dropout", 0.08),
            calib_dilations=kwargs.get("calib_dilations", (1, 2, 4, 8, 16)),
            transformer_layers=kwargs.get("transformer_layers", 1),
            transformer_heads=kwargs.get("transformer_heads", 4),
            transformer_ff_dim=kwargs.get("transformer_ff_dim", 128),
            pooling=kwargs.get("pooling", "mean"),
            use_static=kwargs.get("use_static", False),
            static_dim=kwargs.get("static_dim", 4),
            static_hidden_dim=kwargs.get("static_hidden_dim", None),
            static_dropout=kwargs.get("static_dropout", 0.1),
            motion_feature_mode=kwargs.get("motion_feature_mode", "norm"),
            motion_stats_branch=kwargs.get("motion_stats_branch", False),
            stream_time_features=kwargs.get("stream_time_features", True),
            stream_context_mode=kwargs.get("stream_context_mode", "gru_multiscale"),
            stream_prepend_calibration=kwargs.get("stream_prepend_calibration", False),
            stream_calib_condition_mode=kwargs.get("stream_calib_condition_mode", "none"),
            stream_calib_condition_strength=kwargs.get("stream_calib_condition_strength", 0.1),
            calib_summary_features=kwargs.get("calib_summary_features", True),
            calibration_fusion_mode=kwargs.get("calibration_fusion_mode", "add"),
            calibration_fusion_hidden_dim=kwargs.get("calibration_fusion_hidden_dim", None),
            calibration_fusion_output_dim=kwargs.get("calibration_fusion_output_dim", None),
            calibration_encoder_mode=kwargs.get("calibration_encoder_mode", "tcn_transformer"),
            state_feedback_mode=kwargs.get("state_feedback_mode", "predicted_current"),
            fms_context_mode=kwargs.get("fms_context_mode", "calibration_history"),
            ordinal_bins=kwargs.get("ordinal_bins", None),
            fms_combine_weight_ordinal=kwargs.get("fms_combine_weight_ordinal", 0.6),
            current_head_mode=kwargs.get("current_head_mode", "basic"),
            ordinal_head_mode=kwargs.get("ordinal_head_mode", None),
            current_delta_scale=kwargs.get("current_delta_scale", 0.75),
            current_anchor_delta_growth_scale=kwargs.get("current_anchor_delta_growth_scale", 0.0),
            current_anchor_delta_growth_horizon_seconds=kwargs.get(
                "current_anchor_delta_growth_horizon_seconds",
                90.0,
            ),
            current_anchor_delta_growth_power=kwargs.get("current_anchor_delta_growth_power", 1.0),
            current_trajectory_offsets=kwargs.get("current_trajectory_offsets", None),
            motion_encoder_context=kwargs.get("motion_encoder_context", "linear"),
            motion_encoder_layers=kwargs.get("motion_encoder_layers", 0),
            risk_head_enabled=kwargs.get("risk_head_enabled", True),
            fall_risk_head_enabled=kwargs.get("fall_risk_head_enabled", False),
            high_risk_head_enabled=kwargs.get("high_risk_head_enabled", False),
            risk_temporal_context=kwargs.get("risk_temporal_context", "none"),
            risk_temporal_layers=kwargs.get("risk_temporal_layers", 0),
            coarse_band_bins=kwargs.get("coarse_band_bins", None),
            coarse_residual_head_enabled=kwargs.get("coarse_residual_head_enabled", False),
            coarse_residual_range=kwargs.get("coarse_residual_range", 3.0),
            coarse_residual_combine_weight=kwargs.get("coarse_residual_combine_weight", 0.0),
            regime_head_enabled=kwargs.get("regime_head_enabled", False),
            regime_class_count=kwargs.get("regime_class_count", 5),
            uncertainty_head_enabled=kwargs.get("uncertainty_head_enabled", False),
            uncertainty_min_log_sigma=kwargs.get("uncertainty_min_log_sigma", -5.0),
            uncertainty_max_log_sigma=kwargs.get("uncertainty_max_log_sigma", 1.0),
            deep_tcn_dilations=kwargs.get("deep_tcn_dilations", (1, 2, 4, 8, 16, 32)),
            calibration_tcn_adaptive_dilations=kwargs.get("calibration_tcn_adaptive_dilations", False),
            calibration_tcn_max_padding_steps=kwargs.get("calibration_tcn_max_padding_steps", 8),
            calibration_tcn_max_padding_fraction=kwargs.get("calibration_tcn_max_padding_fraction", 0.1),
            decoder_hidden_dim=kwargs.get("decoder_hidden_dim", None),
            decoder_context_mode=kwargs.get("decoder_context_mode", "fused"),
            decoder_temporal_context=kwargs.get("decoder_temporal_context", "none"),
            decoder_temporal_layers=kwargs.get("decoder_temporal_layers", 0),
            fds_enabled=kwargs.get("fds_enabled", False),
            fds_min=kwargs.get("fds_min", 0.0),
            fds_max=kwargs.get("fds_max", 20.0),
            fds_bin_size=kwargs.get("fds_bin_size", 1.0),
            fds_num_bins=kwargs.get("fds_num_bins", 21),
            fds_kernel=kwargs.get("fds_kernel", "gaussian"),
            fds_kernel_size=kwargs.get("fds_kernel_size", 5),
            fds_sigma=kwargs.get("fds_sigma", 2.0),
            fds_momentum=kwargs.get("fds_momentum", 0.9),
            fds_blend=kwargs.get("fds_blend", 1.0),
            calib_fms_dropout=kwargs.get("calib_fms_dropout", 0.0),
            calibration_end_fms_dropout=kwargs.get("calibration_end_fms_dropout", 0.0),
            current_session_affine_head_enabled=kwargs.get("current_session_affine_head_enabled", False),
            current_session_affine_hidden_dim=kwargs.get("current_session_affine_hidden_dim", None),
            current_session_affine_scale_range=kwargs.get("current_session_affine_scale_range", 0.25),
            current_session_affine_bias_range=kwargs.get("current_session_affine_bias_range", 0.15),
            current_affine_head_enabled=kwargs.get("current_affine_head_enabled", False),
            current_affine_hidden_dim=kwargs.get("current_affine_hidden_dim", None),
            current_affine_scale_range=kwargs.get("current_affine_scale_range", 0.5),
            current_affine_bias_range=kwargs.get("current_affine_bias_range", 0.25),
            current_binned_affine_head_enabled=kwargs.get("current_binned_affine_head_enabled", False),
            current_binned_affine_anchor_bins=kwargs.get("current_binned_affine_anchor_bins", None),
            current_binned_affine_pred_bins=kwargs.get("current_binned_affine_pred_bins", None),
            current_binned_affine_time_bins=kwargs.get("current_binned_affine_time_bins", None),
            current_binned_affine_scale_range=kwargs.get("current_binned_affine_scale_range", 1.5),
            current_binned_affine_bias_range=kwargs.get("current_binned_affine_bias_range", 0.5),
            calibration_residual_adapter_enabled=kwargs.get("calibration_residual_adapter_enabled", False),
            calibration_residual_feature_dim=kwargs.get("calibration_residual_feature_dim", 0),
            calibration_residual_adapter_hidden_dim=kwargs.get("calibration_residual_adapter_hidden_dim", None),
            calibration_residual_adapter_mode=kwargs.get("calibration_residual_adapter_mode", "mlp"),
            calibration_residual_delta_range=kwargs.get("calibration_residual_delta_range", 0.15),
            calibration_residual_decay_seconds=kwargs.get("calibration_residual_decay_seconds", 120.0),
            calibration_residual_gate_low_threshold=kwargs.get("calibration_residual_gate_low_threshold", 8.0),
            calibration_residual_gate_high_threshold=kwargs.get("calibration_residual_gate_high_threshold", 10.0),
            calibration_residual_gate_anchor_threshold=kwargs.get("calibration_residual_gate_anchor_threshold", 10.0),
            calibration_residual_gate_temperature=kwargs.get("calibration_residual_gate_temperature", 1.0),
            calibration_summary_fusion_enabled=kwargs.get("calibration_summary_fusion_enabled", False),
            calibration_summary_fusion_feature_dim=kwargs.get("calibration_summary_fusion_feature_dim", 0),
            calibration_summary_fusion_hidden_dim=kwargs.get("calibration_summary_fusion_hidden_dim", None),
            calibration_summary_fusion_mode=kwargs.get("calibration_summary_fusion_mode", "additive_gated"),
            calibration_summary_fusion_strength=kwargs.get("calibration_summary_fusion_strength", 1.0),
            current_low_suppressor_enabled=kwargs.get("current_low_suppressor_enabled", False),
            current_low_suppressor_hidden_dim=kwargs.get("current_low_suppressor_hidden_dim", None),
            current_low_suppressor_delta_range=kwargs.get("current_low_suppressor_delta_range", 0.25),
            current_low_suppressor_gate_init_bias=kwargs.get("current_low_suppressor_gate_init_bias", -6.0),
        )
    if name in {"lcsa_cross_attn", "lcsa_cross_attention", "lc-sa-cross-attn"}:
        return LCSACrossAttentionForecaster(
            head_dim=kwargs.get("head_dim", 6),
            calibration_steps=kwargs.get("calibration_steps", 180),
            horizon_steps=kwargs.get("horizon_steps", 10),
            recent_steps=kwargs.get("recent_steps", 60),
            sampling_interval=kwargs.get("sampling_interval", 0.5),
            horizon_seconds=kwargs.get("horizon_seconds", None),
            d_model=kwargs.get("d_model", 96),
            kernel_size=kwargs.get("kernel_size", 3),
            dropout=kwargs.get("dropout", 0.05),
            calib_dilations=kwargs.get("calib_dilations", (1, 2, 4, 8, 16)),
            recent_dilations=kwargs.get("recent_dilations", "auto"),
            transformer_layers=kwargs.get("transformer_layers", 1),
            transformer_heads=kwargs.get("transformer_heads", 4),
            transformer_ff_dim=kwargs.get("transformer_ff_dim", 192),
            pooling=kwargs.get("pooling", "mean"),
            anchor_mode=kwargs.get("anchor_mode", "none"),
            anchor_interval_seconds=kwargs.get("anchor_interval_seconds", 0.0),
            predict_delta_from_anchor=kwargs.get("predict_delta_from_anchor", False),
            use_static=kwargs.get("use_static", False),
            static_dim=kwargs.get("static_dim", 4),
            static_hidden_dim=kwargs.get("static_hidden_dim", None),
            static_dropout=kwargs.get("static_dropout", 0.1),
            multi_horizon=kwargs.get("multi_horizon", False),
            horizon_set=kwargs.get("horizon_set", None),
            per_horizon_heads=kwargs.get("per_horizon_heads", False),
            fms_context_mode=kwargs.get("fms_context_mode", "start_only"),
        )
    if name in {"gru_state_mixer", "start_gru_mixer", "gru-start-mixer"}:
        return GRUStateMixerForecaster(
            head_dim=kwargs.get("head_dim", 6),
            calibration_steps=kwargs.get("calibration_steps", 180),
            horizon_steps=kwargs.get("horizon_steps", 10),
            recent_steps=kwargs.get("recent_steps", 60),
            sampling_interval=kwargs.get("sampling_interval", 0.5),
            horizon_seconds=kwargs.get("horizon_seconds", None),
            d_model=kwargs.get("d_model", 96),
            hidden_dim=kwargs.get("hidden_dim", 128),
            dropout=kwargs.get("dropout", 0.05),
            gru_layers=kwargs.get("gru_layers", 1),
            anchor_mode=kwargs.get("anchor_mode", "none"),
            anchor_interval_seconds=kwargs.get("anchor_interval_seconds", 0.0),
            predict_delta_from_anchor=kwargs.get("predict_delta_from_anchor", False),
            use_static=kwargs.get("use_static", False),
            static_dim=kwargs.get("static_dim", 4),
            static_hidden_dim=kwargs.get("static_hidden_dim", None),
            static_dropout=kwargs.get("static_dropout", 0.1),
            multi_horizon=kwargs.get("multi_horizon", False),
            horizon_set=kwargs.get("horizon_set", None),
            per_horizon_heads=kwargs.get("per_horizon_heads", False),
            fms_context_mode=kwargs.get("fms_context_mode", "start_only"),
        )
    if name in {"motion_conv_mixer", "motion_convmixer", "conv_mixer_forecaster"}:
        return MotionConvMixerForecaster(
            head_dim=kwargs.get("head_dim", 6),
            calibration_steps=kwargs.get("calibration_steps", 180),
            horizon_steps=kwargs.get("horizon_steps", 10),
            recent_steps=kwargs.get("recent_steps", 60),
            sampling_interval=kwargs.get("sampling_interval", 0.5),
            horizon_seconds=kwargs.get("horizon_seconds", None),
            d_model=kwargs.get("d_model", 96),
            hidden_dim=kwargs.get("hidden_dim", 128),
            kernel_size=kwargs.get("kernel_size", 5),
            dropout=kwargs.get("dropout", 0.05),
            transformer_layers=kwargs.get("transformer_layers", 3),
            anchor_mode=kwargs.get("anchor_mode", "none"),
            anchor_interval_seconds=kwargs.get("anchor_interval_seconds", 0.0),
            predict_delta_from_anchor=kwargs.get("predict_delta_from_anchor", False),
            use_static=kwargs.get("use_static", False),
            static_dim=kwargs.get("static_dim", 4),
            static_hidden_dim=kwargs.get("static_hidden_dim", None),
            static_dropout=kwargs.get("static_dropout", 0.1),
            multi_horizon=kwargs.get("multi_horizon", False),
            horizon_set=kwargs.get("horizon_set", None),
            per_horizon_heads=kwargs.get("per_horizon_heads", False),
            fms_context_mode=kwargs.get("fms_context_mode", "start_only"),
        )
    if name in {"coff_lstm", "cofflstm"}:
        return COFFLSTM(
            head_dim=kwargs.get("head_dim", 6),
            calibration_steps=kwargs.get("calibration_steps", 60),
            horizon_steps=kwargs.get("horizon_steps", 10),
            recent_steps=kwargs.get("recent_steps", 20),
            sampling_interval=kwargs.get("sampling_interval", 0.5),
            delta_max=kwargs.get("delta_max", 0.1),
            no_film=kwargs.get("no_film", False),
            no_recent_encoder=kwargs.get("no_recent_encoder", False),
            use_legacy_multihead=kwargs.get("use_legacy_multihead", False),
            use_static=kwargs.get("use_static", False),
            static_dim=kwargs.get("static_dim", 4),
            static_hidden_dim=kwargs.get("static_hidden_dim", 64),
            static_dropout=kwargs.get("static_dropout", 0.1),
            recent_encoder=kwargs.get("recent_encoder", "tcn"),
            recent_attn_heads=kwargs.get("recent_attn_heads", 4),
            recent_attn_layers=kwargs.get("recent_attn_layers", 1),
            recent_attn_dropout=kwargs.get("recent_attn_dropout", 0.1),
            fms_context_mode=kwargs.get("fms_context_mode", "calibration_history"),
        )
    common = {
        "head_dim": kwargs.get("head_dim", 6),
        "calibration_steps": kwargs.get("calibration_steps", 60),
        "horizon_steps": kwargs.get("horizon_steps", 10),
        "recent_steps": kwargs.get("recent_steps", 20),
    }
    if name in {"recent10_tcn", "recent_tcn"}:
        return Recent10TCN(**common)
    if name in {"calib_only", "calibonly"}:
        return CalibOnly(
            head_dim=common["head_dim"],
            calibration_steps=common["calibration_steps"],
            horizon_steps=common["horizon_steps"],
            max_time_steps=kwargs.get("max_time_steps", 2048),
            fms_context_mode=kwargs.get("fms_context_mode", "calibration_history"),
        )
    raise ValueError(
        f"Unknown model '{model_name}'. Expected lc_sa_tcnformer, calib_init_state_forecaster, online_fms_risk_tracker, "
        "lcsa_cross_attn, gru_state_mixer, "
        "motion_conv_mixer, anchor_delta_mlp, anchor_delta_gru, recent_tcn_summary_calib, gated_fusion, "
        "coff_lstm, recent10_tcn, or calib_only."
    )
