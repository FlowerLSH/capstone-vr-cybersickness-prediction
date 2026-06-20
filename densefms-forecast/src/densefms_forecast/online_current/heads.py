"""Prediction heads for online-current FMS tracking.

The selected production path is intentionally simple: a basic regression head
plus a cumulative ordinal head, blended at the end. Legacy dual-delta and
paper-style variants stay here so the main tracker forward path does not have
to carry their details.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


CURRENT_HEAD_MODES = {
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
}
ORDINAL_HEAD_MODES = {"softmax", "cumulative", "clm", "coral", "corn"}


def _make_mlp_head(input_dim: int, output_dim: int, hidden_dim: Optional[int], dropout: float) -> nn.Module:
    if hidden_dim is None or int(hidden_dim) <= 0:
        return nn.Linear(input_dim, output_dim)
    hidden = int(hidden_dim)
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.GELU(),
        nn.LayerNorm(hidden),
        nn.Dropout(dropout),
        nn.Linear(hidden, output_dim),
    )


def _zero_initialize_last_linear(module: nn.Module, bias: float = 0.0) -> None:
    layers = list(module.modules())
    for layer in reversed(layers):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, float(bias))
            return


class CumulativeOrdinalHead(nn.Module):
    """Ordinal regression head with one severity score and ordered thresholds."""

    def __init__(self, input_dim: int, num_bins: int, hidden_dim: Optional[int], dropout: float):
        super().__init__()
        if int(num_bins) < 2:
            raise ValueError("CumulativeOrdinalHead requires at least two ordinal bins.")
        self.num_bins = int(num_bins)
        self.score_head = _make_mlp_head(input_dim, 1, hidden_dim, dropout)
        self.raw_threshold_deltas = nn.Parameter(torch.zeros(self.num_bins - 1))

    def thresholds(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        deltas = F.softplus(self.raw_threshold_deltas.to(device=device, dtype=dtype)) + 1e-4
        thresholds = torch.cumsum(deltas, dim=0)
        return thresholds - thresholds.mean()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        score = self.score_head(x).squeeze(-1)
        thresholds = self.thresholds(score.dtype, score.device)
        view_shape = (1,) * score.ndim + (-1,)
        binary_logits = score.unsqueeze(-1) - thresholds.view(view_shape)
        greater_probs = torch.sigmoid(binary_logits)
        if self.num_bins == 2:
            probs = torch.cat([1.0 - greater_probs, greater_probs], dim=-1)
        else:
            probs = torch.cat(
                [
                    1.0 - greater_probs[..., :1],
                    greater_probs[..., :-1] - greater_probs[..., 1:],
                    greater_probs[..., -1:],
                ],
                dim=-1,
            )
        probs = probs.clamp_min(1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return {
            "logits": probs.log(),
            "probs": probs,
            "binary_logits": binary_logits,
            "score": score,
            "thresholds": thresholds,
        }


class CornOrdinalHead(nn.Module):
    """CORN-style conditional ordinal head.

    Each binary classifier estimates P(y > k | y > k - 1). The unconditional
    survival probabilities are reconstructed with a cumulative product and then
    converted to a class PMF.
    """

    def __init__(self, input_dim: int, num_bins: int, hidden_dim: Optional[int], dropout: float):
        super().__init__()
        if int(num_bins) < 2:
            raise ValueError("CornOrdinalHead requires at least two ordinal bins.")
        self.num_bins = int(num_bins)
        self.binary_head = _make_mlp_head(input_dim, self.num_bins - 1, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        binary_logits = self.binary_head(x)
        conditional_probs = torch.sigmoid(binary_logits).clamp(1e-8, 1.0 - 1e-8)
        greater_probs = torch.cumprod(conditional_probs, dim=-1)
        if self.num_bins == 2:
            probs = torch.cat([1.0 - greater_probs, greater_probs], dim=-1)
        else:
            probs = torch.cat(
                [
                    1.0 - greater_probs[..., :1],
                    greater_probs[..., :-1] - greater_probs[..., 1:],
                    greater_probs[..., -1:],
                ],
                dim=-1,
            )
        probs = probs.clamp_min(1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return {
            "logits": probs.log(),
            "probs": probs,
            "binary_logits": binary_logits,
            "corn_conditional_probs": conditional_probs,
            "corn_greater_probs": greater_probs,
        }


def normalize_current_head_mode(mode: str | None) -> str:
    normalized = str(mode or "basic").lower()
    if normalized not in CURRENT_HEAD_MODES:
        raise ValueError(
            "current_head_mode must be one of: 'basic', 'dual_delta_gate', 'paper_ordreg', "
            "'residual_update', 'person_prior', 'trajectory_decoder', 'regime_gated', "
            "'anchor_regime_gated', 'state_space_delta', 'range_scaled_delta', 'guarded_range_scaled_delta', "
            "'calib_prior_range_scaled_delta', 'calib_lowcap_range_scaled_delta', or 'zero_anchor_mixture'."
        )
    return normalized


def default_ordinal_head_mode(current_head_mode: str) -> str:
    return "cumulative" if normalize_current_head_mode(current_head_mode) == "paper_ordreg" else "softmax"


def normalize_ordinal_head_mode(mode: str | None, current_head_mode: str) -> str:
    normalized = str(mode or default_ordinal_head_mode(current_head_mode)).lower()
    if normalized not in ORDINAL_HEAD_MODES:
        raise ValueError("ordinal_head_mode must be one of: softmax, cumulative, clm, coral, corn.")
    return normalized


def current_head_risk_input_dim(decoder_feature_dim: int, current_head_mode: str) -> int:
    extra = 4 if normalize_current_head_mode(current_head_mode) == "dual_delta_gate" else 0
    return int(decoder_feature_dim) + extra


def make_current_regression_head(
    decoder_feature_dim: int,
    decoder_hidden_dim: Optional[int],
    dropout: float,
) -> nn.Module:
    return _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout)


def make_dual_delta_heads(
    decoder_feature_dim: int,
    d_model: int,
    decoder_hidden_dim: Optional[int],
    dropout: float,
) -> Dict[str, nn.Module]:
    drift_hidden = decoder_hidden_dim or d_model
    return {
        "current_level_head": _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout),
        "current_delta_head": _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout),
        "current_gate_head": _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout),
        "session_drift_head": nn.Sequential(
            nn.Linear(d_model, drift_hidden),
            nn.GELU(),
            nn.LayerNorm(drift_hidden),
            nn.Dropout(dropout),
            nn.Linear(drift_hidden, 1),
        ),
    }


def make_residual_update_heads(
    decoder_feature_dim: int,
    decoder_hidden_dim: Optional[int],
    dropout: float,
) -> Dict[str, nn.Module]:
    return {
        "current_residual_delta_head": _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout),
    }


def make_person_prior_heads(
    decoder_feature_dim: int,
    d_model: int,
    decoder_hidden_dim: Optional[int],
    dropout: float,
) -> Dict[str, nn.Module]:
    prior_hidden = decoder_hidden_dim or d_model
    prior_head = lambda: nn.Sequential(
        nn.Linear(d_model, prior_hidden),
        nn.GELU(),
        nn.LayerNorm(prior_hidden),
        nn.Dropout(dropout),
        nn.Linear(prior_hidden, 1),
    )
    return {
        "person_dynamic_head": _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout),
        "person_bias_head": prior_head(),
        "person_scale_head": prior_head(),
        "person_speed_head": prior_head(),
    }


def make_trajectory_decoder_heads(
    decoder_feature_dim: int,
    trajectory_points: int,
    decoder_hidden_dim: Optional[int],
    dropout: float,
) -> Dict[str, nn.Module]:
    return {
        "current_trajectory_head": _make_mlp_head(
            decoder_feature_dim,
            int(trajectory_points),
            decoder_hidden_dim,
            dropout,
        ),
    }


def make_regime_gated_heads(
    decoder_feature_dim: int,
    expert_count: int,
    decoder_hidden_dim: Optional[int],
    dropout: float,
) -> Dict[str, nn.Module]:
    return {
        "current_regime_gate_head": _make_mlp_head(
            decoder_feature_dim,
            int(expert_count),
            decoder_hidden_dim,
            dropout,
        ),
        "current_regime_expert_head": _make_mlp_head(
            decoder_feature_dim,
            int(expert_count),
            decoder_hidden_dim,
            dropout,
        ),
    }


def make_state_space_delta_heads(
    decoder_feature_dim: int,
    decoder_hidden_dim: Optional[int],
    dropout: float,
) -> Dict[str, nn.Module]:
    return {
        "current_state_delta_head": _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout),
        "current_state_leak_head": _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout),
        "current_state_equilibrium_head": _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout),
    }


def make_range_scaled_delta_heads(
    decoder_feature_dim: int,
    d_model: int,
    decoder_hidden_dim: Optional[int],
    dropout: float,
    *,
    guarded: bool = False,
) -> Dict[str, nn.Module]:
    prior_hidden = decoder_hidden_dim or d_model
    heads = {
        "current_range_delta_head": _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout),
        "current_range_level_head": _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout),
        "current_range_gate_head": _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout),
        "current_range_scale_head": nn.Sequential(
            nn.Linear(d_model, prior_hidden),
            nn.GELU(),
            nn.LayerNorm(prior_hidden),
            nn.Dropout(dropout),
            nn.Linear(prior_hidden, 1),
        ),
    }
    if guarded:
        guard_head = _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout)
        _zero_initialize_last_linear(guard_head, bias=4.0)
        heads["current_range_guard_head"] = guard_head
    return heads


def make_calibration_prior_heads(
    decoder_feature_dim: int,
    d_model: int,
    decoder_hidden_dim: Optional[int],
    dropout: float,
) -> Dict[str, nn.Module]:
    cap_hidden = decoder_hidden_dim or d_model
    gate_head = _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout)
    cap_head = nn.Sequential(
        nn.Linear(d_model, cap_hidden),
        nn.GELU(),
        nn.LayerNorm(cap_hidden),
        nn.Dropout(dropout),
        nn.Linear(cap_hidden, 1),
    )
    _zero_initialize_last_linear(gate_head, bias=-6.0)
    _zero_initialize_last_linear(cap_head, bias=0.0)
    return {
        "current_calib_prior_gate_head": gate_head,
        "current_calib_prior_cap_head": cap_head,
    }


def make_zero_anchor_mixture_heads(
    decoder_feature_dim: int,
    decoder_hidden_dim: Optional[int],
    dropout: float,
) -> Dict[str, nn.Module]:
    delta_head = _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout)
    gate_head = _make_mlp_head(decoder_feature_dim, 1, decoder_hidden_dim, dropout)
    _zero_initialize_last_linear(delta_head, bias=0.0)
    _zero_initialize_last_linear(gate_head, bias=-6.0)
    return {
        "current_anchor_delta_head": delta_head,
        "current_anchor_gate_head": gate_head,
    }


def make_ordinal_head(
    decoder_feature_dim: int,
    num_bins: int,
    decoder_hidden_dim: Optional[int],
    dropout: float,
    ordinal_head_mode: str,
) -> nn.Module:
    if ordinal_head_mode in {"cumulative", "clm", "coral"}:
        return CumulativeOrdinalHead(decoder_feature_dim, num_bins, decoder_hidden_dim, dropout)
    if ordinal_head_mode == "corn":
        return CornOrdinalHead(decoder_feature_dim, num_bins, decoder_hidden_dim, dropout)
    return _make_mlp_head(decoder_feature_dim, num_bins, decoder_hidden_dim, dropout)


def compute_current_head_outputs(
    *,
    fused: torch.Tensor,
    z_calib: torch.Tensor,
    model_base_fms: torch.Tensor,
    ordinal_bins_norm: torch.Tensor,
    current_head_mode: str,
    ordinal_head_mode: str,
    current_delta_scale: float,
    fms_combine_weight_ordinal: float,
    current_reg_head: nn.Module,
    ordinal_head: nn.Module,
    current_level_head: Optional[nn.Module] = None,
    current_delta_head: Optional[nn.Module] = None,
    current_gate_head: Optional[nn.Module] = None,
    session_drift_head: Optional[nn.Module] = None,
    current_residual_delta_head: Optional[nn.Module] = None,
    person_dynamic_head: Optional[nn.Module] = None,
    person_bias_head: Optional[nn.Module] = None,
    person_scale_head: Optional[nn.Module] = None,
    person_speed_head: Optional[nn.Module] = None,
    current_trajectory_head: Optional[nn.Module] = None,
    current_trajectory_offsets: Optional[torch.Tensor] = None,
    current_trajectory_zero_index: int = 0,
    current_regime_gate_head: Optional[nn.Module] = None,
    current_regime_expert_head: Optional[nn.Module] = None,
    current_state_delta_head: Optional[nn.Module] = None,
    current_state_leak_head: Optional[nn.Module] = None,
    current_state_equilibrium_head: Optional[nn.Module] = None,
    current_range_delta_head: Optional[nn.Module] = None,
    current_range_level_head: Optional[nn.Module] = None,
    current_range_gate_head: Optional[nn.Module] = None,
    current_range_scale_head: Optional[nn.Module] = None,
    current_range_guard_head: Optional[nn.Module] = None,
    current_calib_prior_gate_head: Optional[nn.Module] = None,
    current_calib_prior_cap_head: Optional[nn.Module] = None,
    current_anchor_delta_head: Optional[nn.Module] = None,
    current_anchor_gate_head: Optional[nn.Module] = None,
    current_anchor_delta_growth_scale: float = 0.0,
    current_anchor_delta_growth_horizon_seconds: float = 90.0,
    current_anchor_delta_growth_power: float = 1.0,
    sampling_interval: float = 0.5,
    calibration_steps: int = 0,
    positions: Optional[torch.Tensor] = None,
    current_range_guard_low_threshold: float = 5.0,
    current_range_guard_temperature: float = 1.0,
    current_range_guard_floor: float = 0.10,
    current_range_guard_cap: float = 2.0,
    current_range_guard_cap_strength: float = 1.0,
) -> Dict[str, torch.Tensor]:
    mode = normalize_current_head_mode(current_head_mode)
    bsz, pred_steps, _ = fused.shape
    aux_current: Dict[str, torch.Tensor] = {}

    if ordinal_head_mode in {"cumulative", "clm", "coral", "corn"}:
        ordinal_out = ordinal_head(fused)
        if not isinstance(ordinal_out, dict):
            raise ValueError("cumulative/CLM/CORAL/CORN ordinal heads must return a dict.")
        ordinal_logits = ordinal_out["logits"]
        ordinal_probs = ordinal_out["probs"]
        aux_current.update(
            {
                "ordinal_binary_logits": ordinal_out["binary_logits"],
            }
        )
        if "score" in ordinal_out:
            aux_current["ordinal_score"] = ordinal_out["score"]
        if "thresholds" in ordinal_out:
            aux_current["ordinal_thresholds"] = ordinal_out["thresholds"]
        if "corn_conditional_probs" in ordinal_out:
            aux_current["ordinal_corn_conditional_probs"] = ordinal_out["corn_conditional_probs"]
        if "corn_greater_probs" in ordinal_out:
            aux_current["ordinal_corn_greater_probs"] = ordinal_out["corn_greater_probs"]
    else:
        ordinal_logits = ordinal_head(fused)
        ordinal_probs = torch.softmax(ordinal_logits, dim=-1)
    current_ordinal = (ordinal_probs * ordinal_bins_norm.to(fused.device).view(1, 1, -1)).sum(dim=-1)

    if mode == "dual_delta_gate":
        if (
            current_level_head is None
            or current_delta_head is None
            or current_gate_head is None
            or session_drift_head is None
        ):
            raise ValueError("dual_delta_gate requires level, delta, gate, and session-drift heads.")
        current_level = torch.sigmoid(current_level_head(fused).squeeze(-1))
        session_drift_prior = float(current_delta_scale) * torch.tanh(session_drift_head(z_calib).squeeze(-1))
        current_delta = float(current_delta_scale) * torch.tanh(current_delta_head(fused).squeeze(-1))
        current_delta_value = torch.clamp(
            model_base_fms.unsqueeze(1) + session_drift_prior.unsqueeze(1) + current_delta,
            0.0,
            1.0,
        )
        level_delta_gate = torch.sigmoid(current_gate_head(fused).squeeze(-1))
        current_reg = level_delta_gate * current_level + (1.0 - level_delta_gate) * current_delta_value
        risk_context = torch.cat(
            [
                fused,
                current_reg.unsqueeze(-1),
                current_delta_value.unsqueeze(-1),
                level_delta_gate.unsqueeze(-1),
                session_drift_prior.view(bsz, 1, 1).expand(-1, pred_steps, -1),
            ],
            dim=-1,
        )
        aux_current.update(
            {
                "current_level": current_level,
                "current_delta_value": current_delta_value,
                "current_delta_from_calib_end": current_delta,
                "current_level_delta_gate": level_delta_gate,
                "session_drift_prior": session_drift_prior,
            }
        )
    elif mode == "residual_update":
        if current_residual_delta_head is None:
            raise ValueError("residual_update requires current_residual_delta_head.")
        step_scale = max(float(current_delta_scale), 1e-8) / 20.0
        residual_step = step_scale * torch.tanh(current_residual_delta_head(fused).squeeze(-1))
        current_reg = torch.clamp(model_base_fms.unsqueeze(1) + residual_step.cumsum(dim=1), 0.0, 1.0)
        risk_context = fused
        aux_current.update(
            {
                "current_residual_step": residual_step,
                "current_residual_scale": fused.new_tensor(step_scale),
            }
        )
    elif mode == "person_prior":
        if (
            person_dynamic_head is None
            or person_bias_head is None
            or person_scale_head is None
            or person_speed_head is None
        ):
            raise ValueError("person_prior requires dynamic, bias, scale, and speed heads.")
        dynamic_signal = torch.tanh(person_dynamic_head(fused).squeeze(-1))
        prior_bias = torch.sigmoid(person_bias_head(z_calib).squeeze(-1))
        prior_scale = 1.5 * torch.sigmoid(person_scale_head(z_calib).squeeze(-1))
        response_speed = torch.sigmoid(person_speed_head(z_calib).squeeze(-1))
        raw_current = prior_bias.unsqueeze(1) + 0.5 * prior_scale.unsqueeze(1) * dynamic_signal
        level_current = torch.sigmoid(current_reg_head(fused).squeeze(-1))
        current_reg = torch.clamp(
            response_speed.unsqueeze(1) * raw_current + (1.0 - response_speed).unsqueeze(1) * level_current,
            0.0,
            1.0,
        )
        risk_context = fused
        aux_current.update(
            {
                "person_dynamic_signal": dynamic_signal,
                "person_prior_bias": prior_bias,
                "person_prior_scale": prior_scale,
                "person_response_speed": response_speed,
            }
        )
    elif mode == "trajectory_decoder":
        if current_trajectory_head is None:
            raise ValueError("trajectory_decoder requires current_trajectory_head.")
        trajectory = torch.sigmoid(current_trajectory_head(fused))
        if trajectory.ndim != 3 or trajectory.shape[:2] != fused.shape[:2]:
            raise ValueError(f"current_trajectory_head must return [B,P,K], got {trajectory.shape}.")
        zero_index = int(current_trajectory_zero_index)
        if zero_index < 0 or zero_index >= int(trajectory.shape[-1]):
            raise ValueError("current_trajectory_zero_index is outside the trajectory decoder output.")
        current_reg = trajectory[..., zero_index]
        risk_context = fused
        aux_current["current_trajectory"] = trajectory
        if current_trajectory_offsets is not None:
            aux_current["current_trajectory_offsets"] = current_trajectory_offsets.to(
                device=fused.device,
                dtype=torch.long,
            )
    elif mode in {"regime_gated", "anchor_regime_gated"}:
        if current_regime_gate_head is None or current_regime_expert_head is None:
            raise ValueError("regime_gated requires gate and expert heads.")
        regime_features = fused
        if mode == "anchor_regime_gated":
            if positions is None:
                raise ValueError("anchor_regime_gated requires positions.")
            elapsed = (positions.to(dtype=fused.dtype, device=fused.device) - float(calibration_steps)).clamp_min(0.0)
            elapsed = elapsed * max(float(sampling_interval), 1e-8) / 90.0
            elapsed = elapsed.view(1, pred_steps, 1).expand(bsz, -1, -1)
            base = model_base_fms.view(bsz, 1, 1).expand(-1, pred_steps, -1)
            anchor_low = torch.sigmoid((0.25 - base) / 0.05)
            regime_features = torch.cat([fused, base, current_ordinal.unsqueeze(-1), elapsed, anchor_low], dim=-1)
        gate_logits = current_regime_gate_head(regime_features)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        expert_values = torch.sigmoid(current_regime_expert_head(regime_features))
        if gate_logits.shape != expert_values.shape:
            raise ValueError("regime_gated gate and expert heads must have the same shape.")
        current_reg = (gate_probs * expert_values).sum(dim=-1)
        risk_context = fused
        aux_current.update(
            {
                "current_regime_gate_logits": gate_logits,
                "current_regime_gate_probs": gate_probs,
                "current_regime_expert_values": expert_values,
            }
        )
    elif mode == "state_space_delta":
        if current_state_delta_head is None or current_state_leak_head is None or current_state_equilibrium_head is None:
            raise ValueError("state_space_delta requires delta, leak, and equilibrium heads.")
        step_scale = max(float(current_delta_scale), 1e-8) / 20.0
        drive = step_scale * torch.tanh(current_state_delta_head(fused).squeeze(-1))
        leak = 0.25 * torch.sigmoid(current_state_leak_head(fused).squeeze(-1))
        equilibrium = torch.sigmoid(current_state_equilibrium_head(fused).squeeze(-1))
        prev = model_base_fms
        values = []
        for step_idx in range(pred_steps):
            prev = torch.clamp(
                prev + drive[:, step_idx] + leak[:, step_idx] * (equilibrium[:, step_idx] - prev),
                0.0,
                1.0,
            )
            values.append(prev)
        current_reg = torch.stack(values, dim=1) if values else fused.new_zeros((bsz, 0))
        risk_context = fused
        aux_current.update(
            {
                "current_state_drive": drive,
                "current_state_leak": leak,
                "current_state_equilibrium": equilibrium,
                "current_state_delta_scale": fused.new_tensor(step_scale),
            }
        )
    elif mode in {
        "range_scaled_delta",
        "guarded_range_scaled_delta",
        "calib_prior_range_scaled_delta",
        "calib_lowcap_range_scaled_delta",
    }:
        if (
            current_range_delta_head is None
            or current_range_level_head is None
            or current_range_gate_head is None
            or current_range_scale_head is None
        ):
            raise ValueError("range-scaled delta heads require delta, level, gate, and range-scale heads.")
        if mode == "guarded_range_scaled_delta" and current_range_guard_head is None:
            raise ValueError("guarded_range_scaled_delta requires current_range_guard_head.")
        step_scale = max(float(current_delta_scale), 1e-8) / 20.0
        delta_step = step_scale * torch.tanh(current_range_delta_head(fused).squeeze(-1))
        range_scale = 0.25 + 1.75 * torch.sigmoid(current_range_scale_head(z_calib).squeeze(-1))
        delta_value = torch.clamp(
            model_base_fms.unsqueeze(1) + range_scale.unsqueeze(1) * delta_step.cumsum(dim=1),
            0.0,
            1.0,
        )
        level_value = torch.sigmoid(current_range_level_head(fused).squeeze(-1))
        range_gate = torch.sigmoid(current_range_gate_head(fused).squeeze(-1))
        guarded_range_gate = range_gate
        range_guard_open = None
        range_guard_low_score = None
        range_guard_multiplier = None
        range_guard_cap_value = None
        if mode == "guarded_range_scaled_delta":
            low_threshold = fused.new_tensor(float(current_range_guard_low_threshold) / 20.0)
            temperature = fused.new_tensor(max(float(current_range_guard_temperature), 1e-6) / 20.0)
            floor = max(0.0, min(1.0, float(current_range_guard_floor)))
            cap_strength = max(0.0, min(1.0, float(current_range_guard_cap_strength)))
            anchor_low = torch.sigmoid((low_threshold - model_base_fms).unsqueeze(1) / temperature)
            ordinal_low = torch.sigmoid((low_threshold - current_ordinal) / temperature)
            range_guard_low_score = torch.maximum(anchor_low, ordinal_low).clamp(0.0, 1.0)
            range_guard_open = torch.sigmoid(current_range_guard_head(fused).squeeze(-1))
            range_guard_multiplier = floor + (1.0 - floor) * (1.0 - range_guard_low_score) * range_guard_open
            guarded_range_gate = range_gate * range_guard_multiplier
        mixed_value = guarded_range_gate * delta_value + (1.0 - guarded_range_gate) * level_value
        current_reg = mixed_value
        calib_prior_gate = None
        calib_prior_cap = None
        calib_prior_capped_value = None
        if mode == "guarded_range_scaled_delta":
            cap_delta = max(0.0, float(current_range_guard_cap)) / 20.0
            range_guard_cap_value = torch.clamp(model_base_fms.unsqueeze(1) + cap_delta, 0.0, 1.0)
            capped_level = torch.minimum(level_value, range_guard_cap_value)
            blend = range_guard_low_score * cap_strength
            current_reg = blend * capped_level + (1.0 - blend) * mixed_value
        if mode in {"calib_prior_range_scaled_delta", "calib_lowcap_range_scaled_delta"}:
            if current_calib_prior_gate_head is None or current_calib_prior_cap_head is None:
                raise ValueError(f"{mode} requires calibration prior gate and cap heads.")
            calib_prior_gate_logits = current_calib_prior_gate_head(fused).squeeze(-1)
            calib_prior_gate = torch.sigmoid(calib_prior_gate_logits)
            raw_calib_prior_cap = torch.sigmoid(current_calib_prior_cap_head(z_calib).squeeze(-1))
            if mode == "calib_lowcap_range_scaled_delta":
                calib_prior_cap_max = max(0.0, min(20.0, float(current_range_guard_cap))) / 20.0
                calib_prior_cap = raw_calib_prior_cap * calib_prior_cap_max
            else:
                calib_prior_cap = raw_calib_prior_cap
            calib_prior_capped_value = torch.minimum(current_reg, calib_prior_cap.unsqueeze(1))
            current_reg = calib_prior_gate * calib_prior_capped_value + (1.0 - calib_prior_gate) * current_reg
        risk_context = fused
        aux_current.update(
            {
                "current_range_delta_step": delta_step,
                "current_range_delta_value": delta_value,
                "current_range_level": level_value,
                "current_range_gate": range_gate,
                "current_range_effective_gate": guarded_range_gate,
                "current_range_scale": range_scale,
                "current_range_delta_scale": fused.new_tensor(step_scale),
            }
        )
        if mode in {"calib_prior_range_scaled_delta", "calib_lowcap_range_scaled_delta"}:
            aux_current.update(
                {
                    "current_calib_prior_gate": calib_prior_gate,
                    "current_calib_prior_gate_logits": calib_prior_gate_logits,
                    "current_calib_prior_cap": calib_prior_cap,
                    "current_calib_prior_capped_value": calib_prior_capped_value,
                }
            )
        if mode == "guarded_range_scaled_delta":
            aux_current.update(
                {
                    "current_range_guard_open": range_guard_open,
                    "current_range_guard_low_score": range_guard_low_score,
                    "current_range_guard_multiplier": range_guard_multiplier,
                    "current_range_guard_cap_value": range_guard_cap_value,
                }
            )
    elif mode == "zero_anchor_mixture":
        if current_anchor_delta_head is None or current_anchor_gate_head is None:
            raise ValueError("zero_anchor_mixture requires anchor delta and gate heads.")
        base_delta_range_raw = max(float(current_delta_scale), 1e-8)
        growth_raw = max(float(current_anchor_delta_growth_scale), 0.0)
        if growth_raw > 0.0:
            if positions is None:
                raise ValueError("zero_anchor_mixture delta growth requires positions.")
            if positions.ndim != 1 or int(positions.numel()) != pred_steps:
                raise ValueError(f"positions must be [T] matching pred_steps={pred_steps}, got {positions.shape}.")
            growth_horizon = max(float(current_anchor_delta_growth_horizon_seconds), 1e-6)
            power = max(float(current_anchor_delta_growth_power), 1e-6)
            elapsed_seconds = (positions.to(dtype=fused.dtype, device=fused.device) - float(calibration_steps)).clamp_min(0.0)
            elapsed_seconds = elapsed_seconds * max(float(sampling_interval), 1e-8)
            ratio = (elapsed_seconds / growth_horizon).clamp(0.0, 1.0).pow(power)
            delta_range = (base_delta_range_raw + growth_raw * ratio).view(1, pred_steps) / 20.0
        else:
            delta_range = fused.new_tensor(base_delta_range_raw / 20.0)
        level_value = torch.sigmoid(current_reg_head(fused).squeeze(-1))
        anchor_delta = delta_range * torch.tanh(current_anchor_delta_head(fused).squeeze(-1))
        anchor_value = torch.clamp(model_base_fms.unsqueeze(1) + anchor_delta, 0.0, 1.0)
        anchor_gate_logits = current_anchor_gate_head(fused).squeeze(-1)
        anchor_gate = torch.sigmoid(anchor_gate_logits)
        current_reg = anchor_gate * anchor_value + (1.0 - anchor_gate) * level_value
        risk_context = fused
        aux_current.update(
            {
                "current_anchor_level": level_value,
                "current_anchor_value": anchor_value,
                "current_anchor_delta": anchor_delta,
                "current_anchor_gate": anchor_gate,
                "current_anchor_gate_logits": anchor_gate_logits,
                "current_anchor_delta_range": (
                    delta_range.expand(bsz, pred_steps) if delta_range.ndim == 2 else delta_range
                ),
            }
        )
    else:
        current_reg = torch.sigmoid(current_reg_head(fused).squeeze(-1))
        risk_context = fused

    w_ord = max(0.0, min(1.0, float(fms_combine_weight_ordinal)))
    current = torch.clamp(w_ord * current_ordinal + (1.0 - w_ord) * current_reg, 0.0, 1.0)

    return {
        "current": current,
        "current_reg": current_reg,
        "current_ordinal": current_ordinal,
        "ordinal_logits": ordinal_logits,
        "ordinal_probs": ordinal_probs,
        "risk_context": risk_context,
        **aux_current,
    }
