"""Evaluate a saved DenseFMS forecasting checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import (
    DenseFMSSessionDataset,
    apply_saved_split,
    collate_sessions,
    load_calibration_residual_features,
    load_raw_sessions,
    transform_sessions,
)
from .model import build_model
from .train import collect_online_current_risk_predictions, collect_predictions, save_prediction_csv, save_prediction_plots
from .utils import ensure_dir, load_json, normalize_time_config, save_json, seconds_to_steps, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DenseFMS forecasting checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--calibration_seconds", type=float, default=None)
    parser.add_argument("--horizon_seconds", type=float, default=None)
    parser.add_argument("--recent_window_seconds", type=float, default=None)
    parser.add_argument("--max_session_points", type=int, default=None)
    parser.add_argument("--calibration_residual_features_path", nargs="+", default=None)
    parser.add_argument("--common_eval_current_start", type=float, default=None)
    parser.add_argument("--common_eval_current_end", type=float, default=None)
    parser.add_argument("--common_eval_target_start", type=float, default=None)
    parser.add_argument("--common_eval_target_end", type=float, default=None)
    parser.add_argument("--common_eval_max_horizon_seconds", type=float, default=None)
    parser.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    set_seed(int(ckpt.get("config", {}).get("training", {}).get("seed", 42)))

    config = ckpt["config"]
    normalize_time_config(config)
    if args.calibration_seconds is not None:
        config["data"]["calibration_seconds"] = float(args.calibration_seconds)
    if args.horizon_seconds is not None:
        config["data"]["horizon_seconds"] = float(args.horizon_seconds)
    if args.recent_window_seconds is not None:
        config["data"]["recent_window_seconds"] = float(args.recent_window_seconds)
        config["data"]["recent_seconds"] = float(args.recent_window_seconds)
    if args.max_session_points is not None:
        config["data"]["max_session_points"] = int(args.max_session_points)
    normalize_time_config(config)
    data_cfg = config["data"]
    raw_sessions, _, data_info = load_raw_sessions(
        args.data_dir,
        mapping=ckpt.get("column_mapping"),
        calibration_seconds=float(data_cfg["calibration_seconds"]),
        horizon_seconds=float(data_cfg["horizon_seconds"]),
        default_sampling_interval=float(data_cfg.get("sampling_interval", data_cfg.get("default_sampling_interval", 0.5))),
        max_session_points=data_cfg.get("max_session_points"),
    )
    split_info = load_json(args.split_file) if args.split_file else ckpt["split_info"]
    split_raw = apply_saved_split(raw_sessions, split_info)
    if args.split == "all":
        selected_raw = raw_sessions
    else:
        selected_raw = split_raw.get(args.split, [])
    if not selected_raw:
        raise RuntimeError(f"No sessions available for split '{args.split}'.")
    use_static = bool(ckpt.get("model_kwargs", {}).get("use_static", False))
    allow_missing_static = bool(config.get("data", {}).get("allow_missing_static", False))
    residual_path = args.calibration_residual_features_path or config.get("data", {}).get("calibration_residual_features_path")
    residual_feature_map = None
    residual_feature_names = None
    if residual_path:
        residual_feature_map, residual_feature_names, _ = load_calibration_residual_features(residual_path)
    residual_required = bool(
        ckpt.get("model_kwargs", {}).get("calibration_residual_adapter_enabled", False)
        or ckpt.get("model_kwargs", {}).get("calibration_summary_fusion_enabled", False)
    )
    sessions = transform_sessions(
        selected_raw,
        ckpt["scalers"],
        use_static=use_static,
        static_features=config.get("data", {}).get("static_features"),
        allow_missing_static=allow_missing_static,
        head_channel_mode=config.get("data", {}).get("head_channel_mode", "all"),
        calibration_residual_feature_map=residual_feature_map,
        calibration_residual_feature_names=residual_feature_names,
        require_calibration_residual_features=residual_required,
    )
    loader = DataLoader(
        DenseFMSSessionDataset(sessions),
        batch_size=args.batch_size,
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
    run_dir = Path(args.checkpoint).resolve().parent
    task_mode = str(config.get("task", {}).get("mode", config.get("task_mode", "future_forecast"))).lower()
    if task_mode == "online_current_risk":
        future_aux_horizon_seconds = [float(v) for v in config.get("task", {}).get("future_aux_horizon_seconds", [])]
        future_aux_horizon_steps = [
            seconds_to_steps(value, float(data_info["sampling_interval"]), name="future_aux_horizon_seconds")
            for value in future_aux_horizon_seconds
        ]
        high_risk_horizon_seconds = [float(v) for v in config.get("task", {}).get("high_risk_horizon_seconds", [])]
        high_risk_horizon_steps = [
            seconds_to_steps(value, float(data_info["sampling_interval"]), name="high_risk_horizon_seconds")
            for value in high_risk_horizon_seconds
        ]
        high_risk_label_mode = str(config.get("task", {}).get("high_risk_label_mode", "future_any"))
        high_risk_onset_past_steps = seconds_to_steps(
            float(config.get("task", {}).get("high_risk_onset_past_seconds", 0.0)),
            float(data_info["sampling_interval"]),
            name="high_risk_onset_past_seconds",
            allow_zero=True,
        )
        result = collect_online_current_risk_predictions(
            model,
            loader,
            device,
            int(data_info["calibration_steps"]),
            ckpt["scalers"]["fms"],
            rise_horizon_steps=[int(v) for v in model_kwargs.get("rise_horizon_steps", [int(data_info["horizon_steps"])])],
            rise_thresholds=[float(v) for v in config.get("task", {}).get("rise_thresholds", model_kwargs.get("rise_thresholds", [2.0]))],
            ordinal_bins=[float(v) for v in model_kwargs.get("ordinal_bins", [0, 2, 4, 6, 8, 10, 12, 15, 20])],
            fall_horizon_steps=[
                int(v)
                for v in model_kwargs.get(
                    "fall_horizon_steps",
                    model_kwargs.get("rise_horizon_steps", [int(data_info["horizon_steps"])]),
                )
            ],
            fall_thresholds=[
                float(v)
                for v in config.get(
                    "task",
                    {},
                ).get("fall_thresholds", model_kwargs.get("fall_thresholds", config.get("task", {}).get("rise_thresholds", [2.0])))
            ],
            high_risk_horizon_steps=high_risk_horizon_steps,
            high_risk_thresholds=[
                float(v) for v in config.get("task", {}).get("high_risk_thresholds", model_kwargs.get("high_risk_thresholds", []))
            ],
            high_risk_label_mode=high_risk_label_mode,
            high_risk_onset_past_steps=high_risk_onset_past_steps,
            high_fms_caution_threshold=float(config.get("evaluation", {}).get("high_fms_caution_threshold", 8.0)),
            high_fms_warning_threshold=float(config.get("evaluation", {}).get("high_fms_warning_threshold", 12.0)),
            rapid_rise_probability_threshold=float(config.get("evaluation", {}).get("rapid_rise_probability_threshold", 0.5)),
            rapid_drop_probability_threshold=float(
                config.get("evaluation", {}).get(
                    "rapid_drop_probability_threshold",
                    config.get("evaluation", {}).get("rapid_rise_probability_threshold", 0.5),
                )
            ),
            final_warning_mode=str(config.get("evaluation", {}).get("final_warning_mode", "high_or_rapid")),
            use_static=use_static,
            calibration_seconds=float(data_cfg["calibration_seconds"]),
            recent_window_seconds=float(data_cfg["recent_window_seconds"]),
            sampling_interval=float(data_info["sampling_interval"]),
            recent_window_steps=int(model_kwargs["recent_steps"]),
            run_name=run_dir.name,
            model_name=ckpt["model_name"],
            split_name=args.split,
            future_aux_horizon_steps=future_aux_horizon_steps,
        )
    else:
        result = collect_predictions(
            model,
            loader,
            device,
            int(data_info["calibration_steps"]),
            int(data_info["horizon_steps"]),
            ckpt["scalers"]["fms"],
            high_fms_threshold=float(config.get("evaluation", {}).get("high_fms_threshold", 7.0)),
            use_static=use_static,
            calibration_seconds=float(data_cfg["calibration_seconds"]),
            horizon_seconds=float(data_cfg["horizon_seconds"]),
            recent_window_seconds=float(data_cfg["recent_window_seconds"]),
            common_eval_current_start=args.common_eval_current_start,
            common_eval_current_end=args.common_eval_current_end,
            common_eval_target_start=args.common_eval_target_start,
            common_eval_target_end=args.common_eval_target_end,
            common_eval_max_horizon_seconds=args.common_eval_max_horizon_seconds,
            sampling_interval=float(data_info["sampling_interval"]),
            recent_window_steps=int(model_kwargs["recent_steps"]),
            run_name=run_dir.name,
            model_name=ckpt["model_name"],
            split_name=args.split,
            anchor_mode=str(model_kwargs.get("anchor_mode", "none" if ckpt["model_name"] != "lc_sa_tcnformer" else "calibration_end")),
            anchor_interval_seconds=float(model_kwargs.get("anchor_interval_seconds", 60.0)),
            fms_context_mode=str(model_kwargs.get("fms_context_mode", "calibration_history")),
            is_upper_bound_anchor=str(model_kwargs.get("anchor_mode", "")) == "recent_start_observed",
        )
    out_dir = ensure_dir(run_dir / f"eval_{args.split}")
    loss_mode = ckpt.get("loss", config.get("loss", {})).get("mode", "unknown_loss")
    save_prediction_plots(result["series"], out_dir / "plots", f"{args.split}_{loss_mode}")
    if args.save_predictions:
        save_prediction_csv(result["prediction_records"], out_dir / f"{args.split}_predictions.csv")
    payload = {"checkpoint": args.checkpoint, "split": args.split, "split_file": args.split_file, "metrics": result["metrics"]}
    save_json(out_dir / "metrics.json", payload)
    print(payload["metrics"])
    print(f"Saved evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
