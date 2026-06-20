"""HTTP bridge for the Unity RollerCoaster DenseFMS realtime demo.

Unity sends raw head-motion samples and raw FMS values. This bridge keeps the
PyTorch checkpoint in a separate Python process, normalizes incoming values with
the checkpoint scalers, and returns online predictions after calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch


DEFAULT_CODEX_REPO = Path(__file__).resolve().parents[3] / "densefms-forecast"
DEFAULT_CHECKPOINT = (
    DEFAULT_CODEX_REPO
    / "runs"
    / "risk_light_state_0521"
    / "state_headonly_pos0p5_thr12_seed42"
    / "best.pt"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    return value


def _normalize_gender(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in {"m", "male", "man", "boy"}:
        return "male"
    if token in {"f", "female", "woman", "girl"}:
        return "female"
    return "male"


def _require_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite number") from exc
    if not math.isfinite(out):
        raise ValueError(f"{key} must be finite")
    return out


def _require_float_any(payload: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        if key in payload:
            return _require_float(payload, key)
    joined = " or ".join(keys)
    raise ValueError(f"{joined} must be provided")


class DenseFMSModelContext:
    def __init__(self, checkpoint: Path, codex_repo: Path, device_name: str) -> None:
        self.checkpoint = checkpoint
        self.codex_repo = codex_repo
        if str(codex_repo) not in sys.path:
            sys.path.insert(0, str(codex_repo))

        from src.densefms_forecast.model import OnlineFMSRiskTracker, build_model
        from src.densefms_forecast.realtime import OnlineCurrentRiskPrefixStreamer

        self.online_cls = OnlineFMSRiskTracker
        self.streamer_cls = OnlineCurrentRiskPrefixStreamer
        self.device = torch.device(device_name)
        self.ckpt = torch.load(str(checkpoint), map_location=self.device, weights_only=True)
        self.model = build_model(self.ckpt["model_name"], **self.ckpt["model_kwargs"]).to(self.device)
        self.model.load_state_dict(self.ckpt["model_state_dict"])
        self.model.eval()
        self.scalers = self.ckpt["scalers"]
        self.calibration_steps = int(self.ckpt["model_kwargs"]["calibration_steps"])
        self.sampling_interval = float(self.ckpt["model_kwargs"]["sampling_interval"])
        self.head_mean = np.asarray(self.scalers["head"]["mean"], dtype=np.float32)
        self.head_std = np.asarray(self.scalers["head"]["std"], dtype=np.float32)
        self.head_std = np.maximum(self.head_std, 1e-8)
        self.fms_min = float(self.scalers["fms"]["min"])
        self.fms_max = float(self.scalers["fms"]["max"])
        self.fms_range = max(self.fms_max - self.fms_min, 1e-8)
        self.static_scaler = self.scalers.get("static", {})
        self.static_dim = int(self.ckpt["model_kwargs"].get("static_dim", 0))

    def normalize_head(self, raw_head: List[float]) -> np.ndarray:
        arr = np.asarray(raw_head, dtype=np.float32)
        if arr.shape != (6,):
            raise ValueError(f"head sample must contain 6 values, got {arr.shape}")
        return ((arr - self.head_mean) / self.head_std).astype(np.float32)

    def normalize_fms(self, fms_raw: float) -> float:
        return float(np.clip((float(fms_raw) - self.fms_min) / self.fms_range, 0.0, 1.0))

    def static_tensor(self, age: float, mssq: float, gender: str) -> torch.Tensor:
        scaler = self.static_scaler
        age_z = (float(age) - float(scaler["age_mean"])) / float(scaler["age_std"])
        mssq_z = (float(mssq) - float(scaler["mssq_mean"])) / float(scaler["mssq_std"])
        order = list(scaler.get("gender_category_order", ["male", "female"]))
        gender_norm = _normalize_gender(gender)
        values = [age_z, mssq_z]
        values.extend([1.0 if gender_norm == item else 0.0 for item in order])
        arr = np.asarray(values[: self.static_dim], dtype=np.float32)
        if arr.shape[0] != self.static_dim:
            raise ValueError(f"static vector has {arr.shape[0]} values, expected {self.static_dim}")
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)


@dataclass
class DemoSession:
    session_id: str
    age: float
    mssq: float
    gender: str
    streamer: Any
    sample_count: int = 0
    samples: List[Dict[str, Any]] = field(default_factory=list)
    predictions: List[Dict[str, Any]] = field(default_factory=list)


class BridgeState:
    def __init__(self, context: DenseFMSModelContext) -> None:
        self.context = context
        self.lock = threading.Lock()
        self.session: Optional[DemoSession] = None
        self.started_at = time.time()

    def start_session(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        age = _require_float(payload, "age")
        mssq = _require_float(payload, "mssq")
        gender = _normalize_gender(payload.get("gender", "male"))
        static_tensor = self.context.static_tensor(age, mssq, gender)
        streamer = self.context.streamer_cls(
            model=self.context.model,
            fms_scaler=self.context.scalers["fms"],
            calibration_steps=self.context.calibration_steps,
            sampling_interval=self.context.sampling_interval,
            device=self.context.device,
            static_tensor=static_tensor,
            rise_horizon_steps=getattr(self.context.model, "rise_horizon_steps", []),
            fall_horizon_steps=getattr(self.context.model, "fall_horizon_steps", []),
            high_fms_caution_threshold=8.0,
            high_fms_warning_threshold=12.0,
        )
        session_id = str(payload.get("session_id") or int(time.time()))
        with self.lock:
            self.session = DemoSession(
                session_id=session_id,
                age=age,
                mssq=mssq,
                gender=gender,
                streamer=streamer,
            )
        return {
            "ok": True,
            "status": "started",
            "session_id": session_id,
            "calibration_steps": self.context.calibration_steps,
            "sampling_interval": self.context.sampling_interval,
            "uses_static": True,
        }

    def step(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        with self.lock:
            session = self.session
        if session is None:
            raise RuntimeError("No active session. Call /session/start first.")

        step_index = int(_require_float(payload, "step_index"))
        timestamp = _require_float(payload, "timestamp")
        fms_raw = float(np.clip(_require_float(payload, "fms_raw"), 0.0, 20.0))
        raw_head = [
            _require_float_any(payload, "acc_x", "linear_velocity_x"),
            _require_float_any(payload, "acc_y", "linear_velocity_y"),
            _require_float_any(payload, "acc_z", "linear_velocity_z"),
            _require_float(payload, "angular_velocity_x"),
            _require_float(payload, "angular_velocity_y"),
            _require_float(payload, "angular_velocity_z"),
        ]
        head_norm = self.context.normalize_head(raw_head)
        fms_norm = self.context.normalize_fms(fms_raw)

        calibration_value = fms_norm if session.sample_count < self.context.calibration_steps else None
        row = session.streamer.push_normalized(
            head_norm,
            calibration_fms_norm=calibration_value,
            target_fms_raw=fms_raw,
            timestamp=timestamp,
            row_index=step_index,
        )

        sample = {
            "step_index": step_index,
            "timestamp": timestamp,
            "fms_raw": fms_raw,
            "acc_x": raw_head[0],
            "acc_y": raw_head[1],
            "acc_z": raw_head[2],
            "angular_velocity_x": raw_head[3],
            "angular_velocity_y": raw_head[4],
            "angular_velocity_z": raw_head[5],
        }
        session.samples.append(sample)
        session.sample_count += 1

        remaining = max(0, self.context.calibration_steps - session.sample_count)
        if row is None:
            return {
                "ok": True,
                "status": "calibrating",
                "sample_count": session.sample_count,
                "remaining_steps": remaining,
                "calibration_complete": remaining == 0,
                "has_prediction": False,
            }

        prediction = {
            "step_index": step_index,
            "timestamp": timestamp,
            "target_fms_now": float(row.get("target_fms_now", fms_raw)),
            "predicted_fms_now": float(row.get("predicted_fms_now", 0.0)),
            "predicted_fms_regression": float(row.get("predicted_fms_regression", 0.0)),
            "predicted_fms_ordinal": float(row.get("predicted_fms_ordinal", 0.0)),
            "fms_absolute_error": float(row.get("fms_absolute_error", 0.0)),
            "p_high_risk_20s_thr8": float(row.get("p_high_risk_20s_thr8", 0.0)),
            "p_high_risk_20s_thr12": float(row.get("p_high_risk_20s_thr12", 0.0)),
            "p_rapid_rise_10s": float(row.get("p_rapid_rise_10s", 0.0)),
            "p_rapid_rise_20s": float(row.get("p_rapid_rise_20s", 0.0)),
            "p_rapid_drop_10s": float(row.get("p_rapid_drop_10s", 0.0)),
            "p_rapid_drop_20s": float(row.get("p_rapid_drop_20s", 0.0)),
        }
        session.predictions.append(prediction)
        return {
            "ok": True,
            "status": "predicting",
            "sample_count": session.sample_count,
            "remaining_steps": 0,
            "calibration_complete": True,
            "has_prediction": True,
            "prediction_index": int(row.get("prediction_index", step_index)),
            **prediction,
        }

    def finish(self) -> Dict[str, Any]:
        with self.lock:
            session = self.session
            self.session = None
        if session is None:
            return {
                "ok": True,
                "status": "finished",
                "sample_count": 0,
                "prediction_count": 0,
                "has_metrics": False,
                "samples": [],
                "predictions": [],
            }

        errors = [
            float(row["fms_absolute_error"])
            for row in session.predictions
            if math.isfinite(float(row.get("fms_absolute_error", 0.0)))
        ]
        mae = float(np.mean(errors)) if errors else 0.0
        rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0
        last = session.predictions[-1] if session.predictions else {}
        return {
            "ok": True,
            "status": "finished",
            "session_id": session.session_id,
            "sample_count": session.sample_count,
            "prediction_count": len(session.predictions),
            "calibration_complete": session.sample_count >= self.context.calibration_steps,
            "has_metrics": bool(errors),
            "mae": mae,
            "rmse": rmse,
            "last_target_fms": float(last.get("target_fms_now", 0.0)),
            "last_predicted_fms": float(last.get("predicted_fms_now", 0.0)),
            "last_p_high_risk_20s_thr12": float(last.get("p_high_risk_20s_thr12", 0.0)),
            "samples": session.samples,
            "predictions": session.predictions,
        }


class BridgeHandler(BaseHTTPRequestHandler):
    state: BridgeState

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))
        sys.stdout.flush()

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(_json_safe(payload), separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_payload(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "status": "ok",
                    "model_name": self.state.context.ckpt.get("model_name", ""),
                    "checkpoint": str(self.state.context.checkpoint),
                    "calibration_steps": self.state.context.calibration_steps,
                    "sampling_interval": self.state.context.sampling_interval,
                    "uptime_seconds": time.time() - self.state.started_at,
                },
            )
            return
        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        try:
            payload = self._read_payload()
            if self.path == "/session/start":
                self._send_json(200, self.state.start_session(payload))
                return
            if self.path == "/session/step":
                self._send_json(200, self.state.step(payload))
                return
            if self.path == "/session/finish":
                self._send_json(200, self.state.finish())
                return
            self._send_json(404, {"ok": False, "error": "Not found"})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unity DenseFMS realtime HTTP bridge.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--codex_repo", default=str(DEFAULT_CODEX_REPO))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    codex_repo = Path(args.codex_repo)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not codex_repo.exists():
        raise FileNotFoundError(f"Codex repo not found: {codex_repo}")
    context = DenseFMSModelContext(checkpoint=checkpoint, codex_repo=codex_repo, device_name=args.device)
    BridgeHandler.state = BridgeState(context)
    server = ThreadingHTTPServer((args.host, int(args.port)), BridgeHandler)
    print(
        f"Unity DenseFMS bridge listening on http://{args.host}:{args.port} "
        f"checkpoint={checkpoint}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
