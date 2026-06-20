"""Real-time-compatible DenseFMS forecasting simulation."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .data import session_from_csv, transform_sessions
from .model import COFFLSTM, OnlineFMSRiskTracker, build_model
from .utils import denormalize_fms, ensure_dir, normalize_time_config, seconds_to_steps, set_seed


ONLINE_CURRENT_RISK_MODEL_NAMES = {"online_fms_risk_tracker", "online_risk_tracker", "online_current_risk"}


def _scalar_from_tensor(value: torch.Tensor) -> float:
    return float(value.detach().cpu().reshape(-1)[0].item())


def _denormalize_scalar(value: float, scaler: Mapping[str, float]) -> float:
    return float(denormalize_fms(np.asarray([float(value)], dtype=np.float32), scaler)[0])


def _last_denormalized(
    outputs: Mapping[str, Any],
    key: str,
    scaler: Mapping[str, float],
    default: float = float("nan"),
) -> float:
    value = outputs.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[1] == 0:
        return default
    return _denormalize_scalar(_scalar_from_tensor(value[0, -1]), scaler)


def _last_tensor_value(outputs: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    value = outputs.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[1] == 0:
        return default
    return _scalar_from_tensor(value[0, -1])


def _session_static_values(session: Any) -> Dict[str, Any]:
    if session.static is None:
        return {}
    return dict(zip(session.static_feature_names or [], session.static.tolist()))


class OnlineCurrentRiskPrefixStreamer:
    """Prefix-only streaming wrapper for OnlineFMSRiskTracker inference.

    The wrapper is deliberately conservative: every prediction recomputes the
    model on the head-motion prefix observed so far and reads only the last
    current-FMS output. This is slower than a cached one-step implementation,
    but it is exactly causal and keeps existing checkpoints unchanged.
    """

    def __init__(
        self,
        model: OnlineFMSRiskTracker,
        fms_scaler: Mapping[str, float],
        calibration_steps: int,
        sampling_interval: float,
        device: torch.device,
        static_tensor: Optional[torch.Tensor] = None,
        rise_horizon_steps: Optional[Sequence[int]] = None,
        fall_horizon_steps: Optional[Sequence[int]] = None,
        high_fms_caution_threshold: float = 8.0,
        high_fms_warning_threshold: float = 12.0,
    ) -> None:
        self.model = model
        self.model.eval()
        self.fms_scaler = fms_scaler
        self.calibration_steps = int(calibration_steps)
        self.sampling_interval = float(sampling_interval)
        self.device = device
        self.static_tensor = static_tensor.to(device) if static_tensor is not None else None
        self.rise_horizon_steps = [
            int(v)
            for v in (
                rise_horizon_steps
                if rise_horizon_steps is not None
                else getattr(model, "rise_horizon_steps", [])
            )
        ]
        self.fall_horizon_steps = [
            int(v)
            for v in (
                fall_horizon_steps
                if fall_horizon_steps is not None
                else getattr(model, "fall_horizon_steps", [])
            )
        ]
        self.high_fms_caution_threshold = float(high_fms_caution_threshold)
        self.high_fms_warning_threshold = float(high_fms_warning_threshold)
        self._head_steps: List[torch.Tensor] = []
        self._calibration_fms: List[float] = []

    @property
    def observed_steps(self) -> int:
        return len(self._head_steps)

    @torch.no_grad()
    def push_normalized(
        self,
        head_step: np.ndarray | Sequence[float] | torch.Tensor,
        calibration_fms_norm: Optional[float] = None,
        target_fms_raw: Optional[float] = None,
        timestamp: Optional[float] = None,
        row_index: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Append one normalized head-motion sample and maybe return a prediction row.

        ``calibration_fms_norm`` is consumed only for the first calibration
        steps. After calibration, FMS may be supplied as ``target_fms_raw`` for
        display/evaluation, but it is never passed into the model.
        """
        head_arr = (
            head_step.detach().cpu().numpy()
            if isinstance(head_step, torch.Tensor)
            else np.asarray(head_step, dtype=np.float32)
        )
        head_arr = head_arr.astype(np.float32, copy=False).reshape(-1)
        if head_arr.shape != (int(getattr(self.model, "head_dim", 6)),):
            raise ValueError(f"head_step must have shape [{getattr(self.model, 'head_dim', 6)}], got {head_arr.shape}.")
        self._head_steps.append(torch.from_numpy(head_arr).to(self.device))
        current_index = int(row_index if row_index is not None else len(self._head_steps) - 1)

        if len(self._head_steps) <= self.calibration_steps:
            if calibration_fms_norm is None or not np.isfinite(float(calibration_fms_norm)):
                raise ValueError(
                    "Online-current streaming requires finite FMS labels for the initial calibration window."
                )
            self._calibration_fms.append(float(calibration_fms_norm))
            return None
        if len(self._calibration_fms) != self.calibration_steps:
            raise RuntimeError(
                f"Expected {self.calibration_steps} calibration FMS values, got {len(self._calibration_fms)}."
            )

        head = torch.stack(self._head_steps, dim=0).unsqueeze(0).to(dtype=torch.float32, device=self.device)
        y_calib = torch.tensor(self._calibration_fms, dtype=torch.float32, device=self.device).unsqueeze(0)
        lengths = torch.tensor([head.shape[1]], dtype=torch.long, device=self.device)
        model_kwargs: Dict[str, Any] = {}
        if bool(getattr(self.model, "use_static", False)):
            if self.static_tensor is None:
                raise ValueError("Checkpoint requires static features, but no static tensor was provided.")
            model_kwargs["static"] = self.static_tensor
        outputs = self.model(head, y_calib, lengths, **model_kwargs)
        current = outputs["current"]
        if current.shape[1] == 0:
            return None

        prediction_start = int(_scalar_from_tensor(outputs["prediction_start"]))
        prediction_index = int(prediction_start + current.shape[1] - 1)
        pred_norm = _scalar_from_tensor(current[0, -1])
        pred_fms = _denormalize_scalar(pred_norm, self.fms_scaler)
        target = float(target_fms_raw) if target_fms_raw is not None and np.isfinite(float(target_fms_raw)) else float("nan")
        abs_err = abs(pred_fms - target) if np.isfinite(target) else float("nan")
        row: Dict[str, Any] = {
            "streaming_mode": "prefix_recompute",
            "current_index": current_index,
            "prediction_index": prediction_index,
            "stream_step_number": current_index + 1,
            "current_timestamp": float(timestamp) if timestamp is not None and np.isfinite(float(timestamp)) else float("nan"),
            "observed_head_steps": int(head.shape[1]),
            "uses_head_motion_through_index": current_index,
            "uses_fms_through_index": int(self.calibration_steps) - 1,
            "post_calibration_fms_used_as_input": False,
            "calibration_steps": int(self.calibration_steps),
            "sampling_interval": float(self.sampling_interval),
            "prediction_start_index": prediction_start,
            "target_fms_now": target,
            "predicted_fms_now": pred_fms,
            "predicted_fms_now_normalized": pred_norm,
            "predicted_fms_regression": _last_denormalized(outputs, "current_reg", self.fms_scaler),
            "predicted_fms_ordinal": _last_denormalized(outputs, "current_ordinal", self.fms_scaler),
            "predicted_fms_sigma": _last_tensor_value(outputs, "current_sigma") * (
                float(self.fms_scaler["max"]) - float(self.fms_scaler["min"])
            ),
            "calibration_end_fms": (
                _denormalize_scalar(_scalar_from_tensor(outputs["calibration_end_fms"][0]), self.fms_scaler)
                if isinstance(outputs.get("calibration_end_fms"), torch.Tensor)
                else float("nan")
            ),
            "model_anchor_fms": (
                _denormalize_scalar(_scalar_from_tensor(outputs["model_anchor_fms"][0]), self.fms_scaler)
                if isinstance(outputs.get("model_anchor_fms"), torch.Tensor)
                else float("nan")
            ),
            "fms_absolute_error": abs_err,
            "alarm_caution": bool(pred_fms >= self.high_fms_caution_threshold),
            "alarm_warning_high_fms": bool(pred_fms >= self.high_fms_warning_threshold),
        }

        risk_probs = outputs.get("risk_probs")
        if isinstance(risk_probs, torch.Tensor) and risk_probs.ndim == 3 and risk_probs.shape[1] > 0:
            risk_last = risk_probs[0, -1].detach().cpu().numpy()
            for idx, horizon_steps in enumerate(self.rise_horizon_steps):
                horizon_seconds = float(horizon_steps) * self.sampling_interval
                row[f"p_rapid_rise_{horizon_seconds:g}s"] = (
                    float(risk_last[idx]) if idx < risk_last.shape[0] else float("nan")
                )
        fall_probs = outputs.get("fall_risk_probs")
        if isinstance(fall_probs, torch.Tensor) and fall_probs.ndim == 3 and fall_probs.shape[1] > 0:
            fall_last = fall_probs[0, -1].detach().cpu().numpy()
            for idx, horizon_steps in enumerate(self.fall_horizon_steps):
                horizon_seconds = float(horizon_steps) * self.sampling_interval
                row[f"p_rapid_drop_{horizon_seconds:g}s"] = (
                    float(fall_last[idx]) if idx < fall_last.shape[0] else float("nan")
                )
        high_probs = outputs.get("high_risk_probs")
        high_horizons = outputs.get("high_risk_horizon_steps")
        high_thresholds = outputs.get("high_risk_thresholds")
        if (
            isinstance(high_probs, torch.Tensor)
            and high_probs.ndim == 4
            and high_probs.shape[1] > 0
            and isinstance(high_horizons, torch.Tensor)
            and isinstance(high_thresholds, torch.Tensor)
        ):
            high_last = high_probs[0, -1].detach().cpu().numpy()
            horizons_np = high_horizons.detach().cpu().numpy().tolist()
            thresholds_np = high_thresholds.detach().cpu().numpy().tolist()
            for h_idx, horizon_steps in enumerate(horizons_np):
                horizon_seconds = float(horizon_steps) * self.sampling_interval
                for t_idx, threshold in enumerate(thresholds_np):
                    row[f"p_high_risk_{horizon_seconds:g}s_thr{float(threshold):g}"] = (
                        float(high_last[h_idx, t_idx])
                        if h_idx < high_last.shape[0] and t_idx < high_last.shape[1]
                        else float("nan")
                    )
        return row


def _load_realtime_context(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    set_seed(int(ckpt.get("config", {}).get("training", {}).get("seed", 42)))
    config = ckpt.get("config", {})
    normalize_time_config(config)
    if args.calibration_seconds is not None:
        config["data"]["calibration_seconds"] = float(args.calibration_seconds)
    if args.horizon_seconds is not None:
        config["data"]["horizon_seconds"] = float(args.horizon_seconds)
    if args.recent_window_seconds is not None:
        config["data"]["recent_window_seconds"] = float(args.recent_window_seconds)
        config["data"]["recent_seconds"] = float(args.recent_window_seconds)
    if args.static_features is not None:
        config["data"]["static_features"] = list(args.static_features)
    normalize_time_config(config)
    default_interval = float(config["data"].get("sampling_interval", ckpt["data_info"].get("sampling_interval", 0.5)))
    c_steps = seconds_to_steps(config["data"]["calibration_seconds"], default_interval, name="calibration_seconds")
    h_steps = seconds_to_steps(config["data"]["horizon_seconds"], default_interval, name="horizon_seconds")
    recent_steps = seconds_to_steps(config["data"]["recent_window_seconds"], default_interval, name="recent_window_seconds")
    model_kwargs = dict(ckpt["model_kwargs"])
    model_kwargs["calibration_steps"] = c_steps
    model_kwargs["horizon_steps"] = h_steps
    model_kwargs["recent_steps"] = recent_steps
    model_kwargs["sampling_interval"] = default_interval
    model_kwargs["horizon_seconds"] = float(config["data"]["horizon_seconds"])
    model = build_model(ckpt["model_name"], **model_kwargs).to(device)
    state_dict = dict(ckpt["model_state_dict"])
    if "coarse_band_centers_norm" not in state_dict and hasattr(model, "coarse_band_centers_norm"):
        state_dict["coarse_band_centers_norm"] = model.coarse_band_centers_norm.detach().clone()
    model.load_state_dict(state_dict)
    model.eval()

    max_session_points = args.max_session_points
    if max_session_points is None:
        max_session_points = config.get("data", {}).get("max_session_points")
    raw = session_from_csv(
        args.csv_path,
        ckpt["column_mapping"],
        default_sampling_interval=default_interval,
        max_session_points=int(max_session_points) if max_session_points is not None else None,
    )
    checkpoint_uses_static = bool(ckpt.get("model_kwargs", {}).get("use_static", False))
    if args.use_static and not checkpoint_uses_static:
        raise ValueError("--use_static was requested, but the checkpoint model was trained without static features.")
    allow_missing_static = bool(ckpt.get("config", {}).get("data", {}).get("allow_missing_static", False))
    session = transform_sessions(
        [raw],
        ckpt["scalers"],
        use_static=checkpoint_uses_static,
        static_features=config.get("data", {}).get("static_features"),
        allow_missing_static=allow_missing_static or args.allow_missing_static,
    )[0]
    return {
        "device": device,
        "ckpt": ckpt,
        "config": config,
        "default_interval": default_interval,
        "calibration_steps": c_steps,
        "horizon_steps": h_steps,
        "recent_steps": recent_steps,
        "model": model,
        "raw": raw,
        "session": session,
        "checkpoint_uses_static": checkpoint_uses_static,
    }


@torch.no_grad()
def _run_cofflstm_realtime(args: argparse.Namespace, context: Mapping[str, Any]) -> List[Dict[str, Any]]:
    device = context["device"]
    ckpt = context["ckpt"]
    config = context["config"]
    c_steps = int(context["calibration_steps"])
    h_steps = int(context["horizon_steps"])
    recent_steps = int(context["recent_steps"])
    model = context["model"]
    raw = context["raw"]
    session = context["session"]
    checkpoint_uses_static = bool(context["checkpoint_uses_static"])
    horizon_seconds = float(config["data"]["horizon_seconds"])
    if session.length < c_steps + h_steps + 1:
        raise ValueError(f"Session too short for realtime simulation: {session.length} rows.")
    if np.isnan(raw.fms[:c_steps]).any():
        raise ValueError("Calibration FMS contains missing values.")

    head = torch.from_numpy(session.head).float().to(device)
    y = torch.from_numpy(session.fms).float().to(device)
    z_calib = model.encode_calibration(head[:c_steps].unsqueeze(0), y[:c_steps].unsqueeze(0))
    static_tensor = None
    if checkpoint_uses_static:
        if session.static is None:
            raise ValueError("Static checkpoint requires static features, but this session has no static vector.")
        static_tensor = torch.from_numpy(session.static).float().unsqueeze(0).to(device)
    z_context, _ = model.make_context(z_calib, static_tensor)
    state = model.initial_state(z_context)

    ring = deque(maxlen=recent_steps)
    for i in range(c_steps - recent_steps, c_steps):
        if i < 0:
            ring.append(torch.zeros_like(head[0]))
        else:
            ring.append(head[i].detach().clone())

    rows: List[Dict[str, Any]] = []
    for t in range(c_steps, session.length):
        ring.append(head[t].detach().clone())
        if len(ring) < recent_steps:
            continue
        recent = torch.stack(list(ring), dim=0).unsqueeze(0)
        pred, state = model.step(head[t].unsqueeze(0), recent, state, z_context)
        pred_norm = pred.get("future", pred["y_future_hat"]).detach().cpu().numpy()
        pred_fms = float(denormalize_fms(pred_norm, ckpt["scalers"]["fms"])[0])
        gt = float(raw.fms[t + h_steps]) if t + h_steps < raw.length and np.isfinite(raw.fms[t + h_steps]) else np.nan
        abs_err = abs(pred_fms - gt) if np.isfinite(gt) else np.nan
        current_time = float(raw.time[t])
        static_values = dict(zip(session.static_feature_names or [], session.static.tolist())) if session.static is not None else {}
        gender_values = {
            name: value
            for name, value in static_values.items()
            if name.startswith("gender_")
        }
        row = {
            "use_static": checkpoint_uses_static,
            "age_original": raw.age,
            "age_z": static_values.get("age_z", np.nan),
            "mssq_original": raw.mssq,
            "mssq_z": static_values.get("mssq_z", np.nan),
            "gender_label": raw.gender,
            "gender_encoded": gender_values or None,
            "static_feature_names": session.static_feature_names,
            "row_index": t,
            "current_timestamp": current_time,
            "forecast_target_timestamp": current_time + horizon_seconds,
            "predicted_fms": pred_fms,
            "target_ground_truth_fms": gt,
            "absolute_error": abs_err,
        }
        rows.append(row)
        if args.print_rows:
            print(row)

    out_path = Path(args.output) if args.output else Path(args.checkpoint).resolve().parent / "realtime_predictions.csv"
    ensure_dir(out_path.parent)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved realtime predictions to {out_path}")
    return rows


@torch.no_grad()
def _run_online_current_risk_realtime(args: argparse.Namespace, context: Mapping[str, Any]) -> List[Dict[str, Any]]:
    device = context["device"]
    ckpt = context["ckpt"]
    config = context["config"]
    model = context["model"]
    raw = context["raw"]
    session = context["session"]
    c_steps = int(context["calibration_steps"])
    default_interval = float(context["default_interval"])
    checkpoint_uses_static = bool(context["checkpoint_uses_static"])
    if str(args.streaming_mode) != "prefix_recompute":
        raise ValueError("Online-current realtime currently supports only --streaming_mode prefix_recompute.")
    if getattr(model, "calibration_residual_adapter_enabled", False) or getattr(
        model,
        "calibration_summary_fusion_enabled",
        False,
    ):
        raise ValueError(
            "Online-current realtime does not support checkpoints that require offline "
            "calibration_residual_features. Use the selected_deeptcn_risk035_static4 baseline checkpoint."
        )
    if session.length < c_steps + 1:
        raise ValueError(f"Session too short for online-current streaming: {session.length} rows.")
    if np.isnan(raw.fms[:c_steps]).any():
        raise ValueError("Calibration FMS contains missing values.")

    static_tensor = None
    if checkpoint_uses_static:
        if session.static is None:
            raise ValueError("Static checkpoint requires static features, but this session has no static vector.")
        static_tensor = torch.from_numpy(session.static).float().unsqueeze(0).to(device)

    streamer = OnlineCurrentRiskPrefixStreamer(
        model=model,
        fms_scaler=ckpt["scalers"]["fms"],
        calibration_steps=c_steps,
        sampling_interval=default_interval,
        device=device,
        static_tensor=static_tensor,
        rise_horizon_steps=getattr(model, "rise_horizon_steps", []),
        fall_horizon_steps=getattr(model, "fall_horizon_steps", []),
        high_fms_caution_threshold=float(args.high_fms_caution_threshold),
        high_fms_warning_threshold=float(args.high_fms_warning_threshold),
    )
    static_values = _session_static_values(session)
    gender_values = {name: value for name, value in static_values.items() if name.startswith("gender_")}
    rows: List[Dict[str, Any]] = []
    finite_errors: List[float] = []
    for t in range(session.length):
        target_raw = float(raw.fms[t]) if t < raw.length and np.isfinite(raw.fms[t]) else float("nan")
        row = streamer.push_normalized(
            session.head[t],
            calibration_fms_norm=float(session.fms[t]) if t < c_steps else None,
            target_fms_raw=target_raw,
            timestamp=float(raw.time[t]) if t < raw.length else None,
            row_index=t,
        )
        if row is None:
            continue
        if np.isfinite(row["fms_absolute_error"]):
            finite_errors.append(float(row["fms_absolute_error"]))
        row.update(
            {
                "model_name": ckpt.get("model_name"),
                "use_static": checkpoint_uses_static,
                "participant_id": raw.participant_id,
                "session_id": raw.session_id,
                "source_file": raw.source_file,
                "age_original": raw.age,
                "age_z": static_values.get("age_z", np.nan),
                "mssq_original": raw.mssq,
                "mssq_z": static_values.get("mssq_z", np.nan),
                "gender_label": raw.gender,
                "gender_encoded": gender_values or None,
                "static_feature_names": session.static_feature_names,
                "calibration_seconds": float(config["data"]["calibration_seconds"]),
                "recent_window_seconds": float(config["data"]["recent_window_seconds"]),
                "session_length_steps": int(session.length),
                "cumulative_mae": float(np.mean(finite_errors)) if finite_errors else float("nan"),
            }
        )
        rows.append(row)
        if args.print_rows:
            print(row)

    out_path = Path(args.output) if args.output else Path(args.checkpoint).resolve().parent / "streaming_current_predictions.csv"
    ensure_dir(out_path.parent)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved online-current streaming predictions to {out_path}")
    return rows


@torch.no_grad()
def run_realtime(args: argparse.Namespace) -> List[Dict[str, Any]]:
    context = _load_realtime_context(args)
    model = context["model"]
    model_name = str(context["ckpt"].get("model_name", "")).lower()
    if isinstance(model, COFFLSTM):
        return _run_cofflstm_realtime(args, context)
    if isinstance(model, OnlineFMSRiskTracker) or model_name in ONLINE_CURRENT_RISK_MODEL_NAMES:
        return _run_online_current_risk_realtime(args, context)
    raise ValueError(
        "Real-time simulation supports COFFLSTM and online_fms_risk_tracker checkpoints. "
        f"Got model_name={context['ckpt'].get('model_name')!r}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate realtime DenseFMS forecasting from a CSV stream.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--print_rows", action="store_true")
    parser.add_argument("--use_static", action="store_true")
    parser.add_argument("--static_features", nargs="+", default=None)
    parser.add_argument("--allow_missing_static", action="store_true")
    parser.add_argument("--calibration_seconds", type=float, default=None)
    parser.add_argument("--horizon_seconds", type=float, default=None)
    parser.add_argument("--recent_window_seconds", type=float, default=None)
    parser.add_argument("--max_session_points", type=int, default=None)
    parser.add_argument("--streaming_mode", choices=["prefix_recompute"], default="prefix_recompute")
    parser.add_argument("--high_fms_caution_threshold", type=float, default=8.0)
    parser.add_argument("--high_fms_warning_threshold", type=float, default=12.0)
    args = parser.parse_args()
    run_realtime(args)


if __name__ == "__main__":
    main()
