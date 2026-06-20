"""Lightweight sanity tests for DenseFMS sequence losses and leakage guards."""

from __future__ import annotations

import inspect
import json
import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.densefms_forecast.losses import FutureSequenceLoss
from src.densefms_forecast.model import append_motion_features, build_model, calibration_context_fms
from src.densefms_forecast.train import (
    build_participant_balanced_session_weights,
    build_lds_weight_info,
    collect_online_current_risk_predictions,
    collect_predictions,
    compute_online_current_goal_selection_metrics,
    compute_online_current_risk_loss,
    compute_online_current_risk_targets,
    compute_loss,
    compute_session_summary_targets,
    compute_teacher_current_distillation_loss,
    compute_teacher_delta_distillation_loss,
    compute_teacher_future_distillation_loss,
    compute_teacher_repr_distillation_loss,
    load_teacher_model,
    validate_teacher_compatibility,
)
from src.densefms_forecast.data import (
    DenseFMSSession,
    DenseFMSSessionDataset,
    apply_head_channel_mode,
    collate_sessions,
    current_sequence_times,
    fit_scalers,
    fit_static_scaler,
    future_sequence_targets,
    infer_column_mapping,
    load_raw_sessions,
    normalize_gender,
    parse_scenario_from_filename,
    static_vector_for_session,
    static_feature_dim,
    transform_sessions,
    valid_prediction_mask,
)
from scripts.run_calibration_horizon_sweep import common_window_spec
from scripts.run_densefms_optimization import summarize_run
from scripts.run_lc_sa_tcnformer_full_search import build_stage1, train_cmd
from scripts.run_densefms_long_target_search import build_stage2 as build_long_stage2, train_cmd as long_target_train_cmd
from scripts.run_online_current_integrated_improvement import PLAN_CANDIDATES, _command as integrated_improvement_command
from scripts.run_zero_anchor_ablation_0515 import (
    BASE_CONFIG as ZERO_ANCHOR_ABLATION_BASE_CONFIG,
    BASE_SPLIT as ZERO_ANCHOR_ABLATION_BASE_SPLIT,
    RUNS_DIR as ZERO_ANCHOR_ABLATION_RUNS_DIR,
    build_commands as zero_anchor_ablation_build_commands,
    build_specs as zero_anchor_ablation_build_specs,
)
from scripts.run_online_current_remaining_experiments import (
    REMAINING_CANDIDATES,
    _command as remaining_experiment_command,
    _pretrain_command as remaining_pretrain_command,
)
from scripts.run_online_current_calsummary_earlyfusion_experiments import (
    RECIPES as CALSUMMARY_EARLYFUSION_RECIPES,
    _command as calsummary_earlyfusion_command,
    _is_past_cutoff as calsummary_earlyfusion_is_past_cutoff,
    _time_status as calsummary_earlyfusion_time_status,
)
from scripts.evaluate_online_current_calsummary_earlyfusion_gates import evaluate as evaluate_calsummary_earlyfusion_gates
from scripts.prepare_online_current_test_promotion import (
    build_eval_command as build_test_promotion_eval_command,
    select_candidates as select_test_promotion_candidates,
)
from scripts.audit_online_current_goal_status import audit_row as audit_online_current_goal_row
from scripts.run_online_current_goal_resume_pipeline import (
    build_cutoff_status as build_goal_resume_cutoff_status,
    build_step_commands as build_goal_resume_pipeline_steps,
)
from scripts.build_online_current_goal_state_manifest import build_manifest as build_goal_state_manifest
from scripts.verify_online_current_goal_handoff import verify as verify_goal_handoff
from scripts.audit_online_current_c4_feasibility import audit as audit_c4_feasibility
from scripts.analyze_online_current_c4_metric_alternatives import analyze as analyze_c4_metric_alternatives
from scripts.verify_online_current_earlyfusion_readiness import verify_command as verify_earlyfusion_readiness_command
from scripts.summarize_online_current_goal_metrics import _row as summarize_goal_metric_row
from src.densefms_forecast.realtime import OnlineCurrentRiskPrefixStreamer
from src.densefms_forecast.utils import compute_sequence_analysis_metrics, load_config, normalize_time_config, seconds_to_steps


def assert_close(a: torch.Tensor, b: torch.Tensor, name: str, atol: float = 1e-7) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=0):
        raise AssertionError(f"{name} mismatch: {a.item()} vs {b.item()}")


def test_loss_equivalence() -> None:
    pred = torch.tensor([[0.1, 0.2, 0.4, 0.7]])
    true = torch.tensor([[0.0, 0.3, 0.5, 0.8]])
    mask = torch.tensor([[True, True, True, True]])
    level = FutureSequenceLoss("level_only", trend_weight=0.0)
    trend_zero = FutureSequenceLoss("level_trend_raw", trend_weight=0.0)
    loss_a, _ = level(pred, true, mask)
    loss_b, _ = trend_zero(pred, true, mask)
    assert_close(loss_a, loss_b, "level_trend_raw with weight 0 must equal level_only")


def test_low_target_weight_changes_level_loss() -> None:
    pred = torch.tensor([[0.4, 0.4, 0.9]])
    true = torch.tensor([[0.0, 0.2, 0.9]])
    mask = torch.tensor([[True, True, True]])
    base = FutureSequenceLoss("level_only", loss_type="mae", low_target_weight=0.0)
    weighted = FutureSequenceLoss("level_only", loss_type="mae", low_target_weight=2.0, low_target_threshold=0.2)
    loss_base, parts_base = base(pred, true, mask)
    loss_weighted, parts_weighted = weighted(pred, true, mask)
    if not loss_weighted > loss_base:
        raise AssertionError(f"Low-target weighting should increase this low-bin loss: {loss_base} vs {loss_weighted}")
    if parts_weighted["low_target_weight"] != 2.0 or parts_base["low_target_weight"] != 0.0:
        raise AssertionError("Low-target loss part logging is incorrect")


def test_trend_loss_correctness_and_level_distinction() -> None:
    true = torch.tensor([[5.0, 5.0, 6.0, 6.0]])
    pred = torch.tensor([[7.0, 7.0, 8.0, 8.0]])
    mask = torch.tensor([[True, True, True, True]])
    loss_fn = FutureSequenceLoss("level_trend_raw", trend_weight=0.1)
    _, parts = loss_fn(pred, true, mask)
    if abs(parts["loss_trend"]) > 1e-7:
        raise AssertionError(f"Expected zero trend loss, got {parts['loss_trend']}")
    if parts["loss_level"] <= 0:
        raise AssertionError("Expected non-zero level loss when absolute levels differ")


def test_mask_correctness() -> None:
    true = torch.tensor([[0.0, 0.0, 0.0]])
    pred = torch.tensor([[0.0, 100.0, 0.0]])
    mask = torch.tensor([[True, False, True]])
    loss_fn = FutureSequenceLoss("level_trend_raw", trend_weight=1.0)
    _, parts = loss_fn(pred, true, mask)
    if parts["trend_points"] != 0 or abs(parts["loss_trend"]) > 1e-7:
        raise AssertionError("Trend loss crossed an invalid padded/missing position")


def test_no_legacy_multihead_loss() -> None:
    outputs = {
        "future": torch.tensor([[0.2, 0.3, 0.4]]),
        "mask": torch.tensor([[True, True, True]]),
    }
    fms = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4]])
    lengths = torch.tensor([5])
    loss_fn = FutureSequenceLoss("level_only", trend_weight=0.0)
    _, parts = compute_loss(outputs, fms, lengths, calibration_steps=0, horizon_steps=2, loss_fn=loss_fn)
    forbidden = {"now", "delta", "future"}
    if forbidden & set(parts):
        raise AssertionError(f"Legacy loss terms leaked into logs: {forbidden & set(parts)}")
    for required in ("loss_total", "loss_level", "loss_trend"):
        if required not in parts:
            raise AssertionError(f"Missing sequence loss log key: {required}")


def test_leakage_forward_signature_and_post_calib_independence() -> None:
    sig = inspect.signature(build_model("coff_lstm").forward)
    if list(sig.parameters) != ["head", "y_calib", "lengths", "static"]:
        raise AssertionError(f"Unexpected model forward signature: {sig}")
    torch.manual_seed(0)
    model = build_model(
        "coff_lstm",
        head_dim=6,
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=2,
        use_legacy_multihead=False,
    )
    model.eval()
    head = torch.randn(1, 10, 6)
    y_calib = torch.rand(1, 4)
    lengths = torch.tensor([10])
    with torch.no_grad():
        out_a = model(head, y_calib, lengths)["future"]
        out_b = model(head, y_calib, lengths)["future"]
    assert_close(out_a, out_b, "model output should depend on head+y_calib only", atol=1e-6)


def test_static_off_compatibility() -> None:
    model = build_model("coff_lstm", calibration_steps=4, horizon_steps=2, recent_steps=2, use_static=False)
    head = torch.randn(2, 10, 6)
    y_calib = torch.rand(2, 4)
    lengths = torch.tensor([10, 9])
    out = model(head, y_calib, lengths)
    if "future" not in out or out["future"].shape[0] != 2:
        raise AssertionError("Static-off model forward failed without static tensor")


def test_static_on_requirement_and_shape() -> None:
    model = build_model("coff_lstm", calibration_steps=4, horizon_steps=2, recent_steps=2, use_static=True)
    head = torch.randn(2, 10, 6)
    y_calib = torch.rand(2, 4)
    lengths = torch.tensor([10, 9])
    try:
        model(head, y_calib, lengths)
    except ValueError as exc:
        if "static tensor" not in str(exc):
            raise
    else:
        raise AssertionError("Static-on model should fail when static tensor is missing")
    try:
        model(head, y_calib, lengths, static=torch.randn(2, 3))
    except ValueError as exc:
        if "static must be" not in str(exc):
            raise
    else:
        raise AssertionError("Static-on model should fail for wrong static shape")
    out = model(head, y_calib, lengths, static=torch.randn(2, 4))
    if not out["use_static"]:
        raise AssertionError("Static-on output did not mark use_static=True")


def test_static_full_model_forward() -> None:
    model = build_model("coff_lstm", calibration_steps=4, horizon_steps=2, recent_steps=2, use_static=True, static_dim=5)
    head = torch.randn(2, 10, 6)
    y_calib = torch.rand(2, 4)
    lengths = torch.tensor([10, 9])
    out = model(head, y_calib, lengths, static=torch.randn(2, 5))
    if "future" not in out or out["future"].shape[0] != 2:
        raise AssertionError("Static full model forward failed")


def _dummy_session(age, gender, mssq=10.0):
    return DenseFMSSession(
        head=torch.zeros(6, 6).numpy(),
        fms=torch.zeros(6).numpy(),
        time=torch.arange(6).numpy(),
        participant_id=None,
        session_id=f"s_{age}_{gender}",
        source_file="dummy.csv",
        age=age,
        gender=gender,
        mssq=mssq,
    )


def test_participant_balanced_session_weights_equalize_participant_totals() -> None:
    sessions = []
    for idx, participant_id in enumerate(["PA", "PA", "PA", "PB"]):
        session = _dummy_session(20.0 + idx, "male")
        session.participant_id = participant_id
        session.session_id = f"{participant_id}_{idx}"
        sessions.append(session)
    weights = build_participant_balanced_session_weights(sessions).numpy()
    total_pa = float(weights[:3].sum())
    total_pb = float(weights[3])
    if abs(total_pa - total_pb) > 1e-8:
        raise AssertionError(f"Participant-balanced weights should equalize totals: PA={total_pa}, PB={total_pb}")
    if not np.allclose(weights[:3], weights[0]):
        raise AssertionError(f"Same-participant sessions should have equal weights: {weights}")
    fallback_sessions = [_dummy_session(20.0, "male"), _dummy_session(21.0, "female")]
    fallback_weights = build_participant_balanced_session_weights(fallback_sessions).numpy()
    if not np.allclose(fallback_weights, np.ones_like(fallback_weights)):
        raise AssertionError(f"Unique fallback sessions should keep uniform weights: {fallback_weights}")


def test_static_scaler_train_only_and_gender_encoding() -> None:
    train = [_dummy_session(20.0, "male"), _dummy_session(40.0, "female")]
    scaler = fit_static_scaler(train)
    if abs(scaler["age_mean"] - 30.0) > 1e-7 or abs(scaler["age_std"] - 10.0) > 1e-7:
        raise AssertionError(f"Unexpected train-only age scaler: {scaler}")
    val_vec = static_vector_for_session(_dummy_session(50.0, "female"), scaler)
    if abs(float(val_vec[0]) - 2.0) > 1e-7:
        raise AssertionError("Val/test static transform did not use train mean/std")
    if val_vec.tolist()[1:] != [0.0, 1.0, 0.0]:
        raise AssertionError(f"Female one-hot incorrect: {val_vec}")
    if normalize_gender("M") != "male" or normalize_gender("woman") != "female":
        raise AssertionError("Case-insensitive gender normalization failed")
    unknown_vec = static_vector_for_session(_dummy_session(None, "other"), scaler, allow_missing_static=True)
    if unknown_vec.tolist()[1:] != [0.0, 0.0, 1.0]:
        raise AssertionError(f"Unknown gender one-hot incorrect: {unknown_vec}")
    binary_scaler = fit_static_scaler(train, gender_encoding="binary2")
    if binary_scaler["static_feature_names"] != ["age_z", "gender_male", "gender_female"]:
        raise AssertionError(f"2D gender feature names incorrect: {binary_scaler}")
    binary_female_vec = static_vector_for_session(_dummy_session(50.0, "female"), binary_scaler)
    if binary_female_vec.tolist()[1:] != [0.0, 1.0]:
        raise AssertionError(f"2D female one-hot incorrect: {binary_female_vec}")
    binary_unknown_vec = static_vector_for_session(_dummy_session(None, "other"), binary_scaler, allow_missing_static=True)
    if binary_unknown_vec.tolist()[1:] != [0.0, 0.0]:
        raise AssertionError(f"2D unknown gender should encode as all zeros: {binary_unknown_vec}")


def test_mssq_column_inference_and_static_dim() -> None:
    columns = [
        "timestamp",
        "fms",
        "acc_x",
        "acc_y",
        "acc_z",
        "angular_velocity_x",
        "angular_velocity_y",
        "angular_velocity_z",
        "gender",
        "mssq_total",
        "age",
    ]
    mapping = infer_column_mapping(columns)
    if mapping["static"]["mssq_column"] != "mssq_total":
        raise AssertionError(f"MSSQ column was not inferred: {mapping['static']}")
    if static_feature_dim(["age", "gender"]) != 4:
        raise AssertionError("age/gender static_dim should be 4")
    if static_feature_dim(["age", "gender", "mssq"]) != 5:
        raise AssertionError("age/gender/mssq static_dim should be 5")
    if static_feature_dim(["age", "gender"], gender_encoding="binary2") != 3:
        raise AssertionError("age/gender static_dim should be 3 with 2D gender")
    if static_feature_dim(["age", "gender", "mssq"], gender_encoding="binary2") != 4:
        raise AssertionError("age/gender/mssq static_dim should be 4 with 2D gender")
    binary_full_scaler = fit_static_scaler(
        [_dummy_session(20.0, "male", mssq=10.0), _dummy_session(40.0, "female", mssq=30.0)],
        static_features=["age", "mssq", "gender"],
        gender_encoding="binary2",
    )
    if binary_full_scaler["static_feature_names"] != ["age_z", "mssq_z", "gender_male", "gender_female"]:
        raise AssertionError(f"4D static feature names incorrect: {binary_full_scaler}")
    binary_full_vec = static_vector_for_session(
        _dummy_session(30.0, "female", mssq=20.0),
        binary_full_scaler,
        static_features=["age", "mssq", "gender"],
    )
    if binary_full_vec.tolist() != [0.0, 0.0, 0.0, 1.0]:
        raise AssertionError(f"4D static vector should be normalized age/MSSQ plus 2D gender: {binary_full_vec}")


def test_mssq_scaler_and_missing_behavior() -> None:
    train = [_dummy_session(20.0, "male", mssq=10.0), _dummy_session(40.0, "female", mssq=30.0)]
    scaler = fit_static_scaler(train, static_features=["age", "gender", "mssq"])
    if abs(scaler["mssq_mean"] - 20.0) > 1e-7 or abs(scaler["mssq_std"] - 10.0) > 1e-7:
        raise AssertionError(f"Unexpected train-only MSSQ scaler: {scaler}")
    val_vec = static_vector_for_session(_dummy_session(50.0, "female", mssq=40.0), scaler)
    if abs(float(val_vec[1]) - 2.0) > 1e-7:
        raise AssertionError("Val/test MSSQ transform did not use train mean/std")
    missing = _dummy_session(50.0, "female", mssq=None)
    try:
        static_vector_for_session(missing, scaler)
    except ValueError as exc:
        if "MSSQ" not in str(exc):
            raise
    else:
        raise AssertionError("Missing MSSQ should fail when allow_missing_static=False")
    imputed = static_vector_for_session(missing, scaler, allow_missing_static=True)
    if abs(float(imputed[1])) > 1e-7:
        raise AssertionError("Missing MSSQ should impute train mean, producing mssq_z=0")


def test_scenario_static_feature_encoding() -> None:
    if parse_scenario_from_filename("PA001_reverse_optical_flow_high_density_1_01_01_pm.csv") != "reverse_optical_flow_high_density":
        raise AssertionError("Scenario parser did not identify high-density reverse optical flow")
    if parse_scenario_from_filename("PA001_ROF_foward_whiteline_1_01_01_pm.csv") != "rof_forward_whiteline":
        raise AssertionError("Scenario parser should normalize ROF/foward whiteline aliases")
    train = [
        DenseFMSSession(
            head=np.zeros((6, 6), dtype=np.float32),
            fms=np.zeros(6, dtype=np.float32),
            time=np.arange(6, dtype=np.float32),
            participant_id="P1",
            session_id="scenario_base",
            source_file="PA001_Base.csv",
            age=20.0,
            gender="male",
            mssq=10.0,
        ),
        DenseFMSSession(
            head=np.zeros((6, 6), dtype=np.float32),
            fms=np.zeros(6, dtype=np.float32),
            time=np.arange(6, dtype=np.float32),
            participant_id="P2",
            session_id="scenario_high_density",
            source_file="PA002_reverse_optical_flow_high_density.csv",
            age=40.0,
            gender="female",
            mssq=30.0,
        ),
    ]
    scaler = fit_static_scaler(train, static_features=["age", "mssq", "gender", "scenario"], gender_encoding="binary2")
    if scaler["static_dim"] != static_feature_dim(["age", "mssq", "gender", "scenario"], gender_encoding="binary2"):
        raise AssertionError(f"Scenario static_dim mismatch: {scaler}")
    vec = static_vector_for_session(train[1], scaler)
    names = scaler["static_feature_names"]
    scenario_cols = [idx for idx, name in enumerate(names) if name.startswith("scenario_")]
    if len(scenario_cols) != 9:
        raise AssertionError(f"Expected 9 scenario one-hot columns, got {names}")
    active = [names[idx] for idx in scenario_cols if abs(float(vec[idx]) - 1.0) < 1e-7]
    if active != ["scenario_reverse_optical_flow_high_density"]:
        raise AssertionError(f"Scenario one-hot encoded wrong category: active={active}, vec={vec}, names={names}")


def test_fms_fixed_0_20_normalization() -> None:
    sess = DenseFMSSession(
        head=np.zeros((4, 6), dtype=np.float32),
        fms=np.asarray([5.0, 10.0, 15.0, 20.0], dtype=np.float32),
        time=np.arange(4, dtype=np.float32),
        participant_id="P1",
        session_id="s_fms_scale",
        source_file="dummy.csv",
    )
    scalers = fit_scalers([sess], calibration_steps=1, horizon_steps=1)
    fms_scaler = scalers["fms"]
    if fms_scaler["min"] != 0.0 or fms_scaler["max"] != 20.0:
        raise AssertionError(f"FMS scaler must use fixed DenseFMS 0-20 range, got {fms_scaler}")
    transformed = transform_sessions([sess], scalers)[0]
    expected = np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    if not np.allclose(transformed.fms, expected, atol=1e-7):
        raise AssertionError(f"FMS normalization should divide by fixed 20-point scale: {transformed.fms}")


def test_thresholded_trend_plateau_small_wiggle() -> None:
    metrics = compute_sequence_analysis_metrics(
        [5.1, 5.2, 5.1, 5.3],
        [5.0, 5.0, 5.0, 5.0],
        eps_fms=0.5,
    )
    if metrics["trend_sign_accuracy_raw_exact"] >= 1.0:
        raise AssertionError("Raw exact trend sign accuracy should penalize continuous wiggle on plateaus")
    if abs(metrics["trend_stationary_accuracy_1step_eps0.5"] - 1.0) > 1e-7:
        raise AssertionError("Thresholded stationary accuracy should ignore small plateau wiggles")
    if abs(metrics["trend_acc_1step_eps0.5"] - 1.0) > 1e-7:
        raise AssertionError("Thresholded trend accuracy should be perfect for small plateau wiggles")


def test_thresholded_trend_true_up_movement() -> None:
    metrics = compute_sequence_analysis_metrics(
        [5.0, 5.0, 5.7],
        [5.0, 5.0, 6.0],
        eps_fms=0.5,
    )
    if abs(metrics["trend_moving_accuracy_1step_eps0.5"] - 1.0) > 1e-7:
        raise AssertionError("Expected thresholded up movement to match")
    if abs(metrics["trend_up_f1_1step_eps0.5"] - 1.0) > 1e-7:
        raise AssertionError("Expected up-class F1 to be perfect for matched up movement")


def test_thresholded_trend_false_movement() -> None:
    metrics = compute_sequence_analysis_metrics(
        [5.0, 6.0, 5.0],
        [5.0, 5.0, 5.0],
        eps_fms=0.5,
    )
    if metrics["trend_stationary_accuracy_1step_eps0.5"] != 0.0:
        raise AssertionError("False movement should reduce stationary accuracy")
    if metrics["change_f1_1step_eps0.5"] != 0.0 or metrics["change_precision_1step_eps0.5"] != 0.0:
        raise AssertionError("False movement should create change false positives with zero change F1")


def test_thresholded_trend_multistep() -> None:
    metrics = compute_sequence_analysis_metrics(
        [5.0, 5.1, 5.2, 5.1, 5.9],
        [5.0, 5.0, 5.0, 5.0, 6.0],
        eps_fms=0.5,
    )
    if abs(metrics["trend_acc_2s_eps0.5"] - 1.0) > 1e-7:
        raise AssertionError("Expected 2-second thresholded trend to match the up movement")
    if abs(metrics["trend_up_f1_2s_eps0.5"] - 1.0) > 1e-7:
        raise AssertionError("Expected 2-second up-class F1 to match the up movement")


def test_thresholded_trend_mask_correctness() -> None:
    metrics = compute_sequence_analysis_metrics(
        [[0.0, 0.0, 100.0]],
        [[0.0, 0.0, 100.0]],
        valid_mask=[[True, True, False]],
        eps_fms=0.5,
    )
    if abs(metrics["derivative_mae_all"]) > 1e-7:
        raise AssertionError("Trend metrics crossed into an invalid/padded position")
    if abs(metrics["trend_acc_1step_eps0.5"] - 1.0) > 1e-7:
        raise AssertionError("Only the valid adjacent stationary diff should be counted")


def test_seconds_to_steps_conversion() -> None:
    if seconds_to_steps(30.0, 0.5, name="calibration_seconds") != 60:
        raise AssertionError("30s should convert to 60 steps at 0.5s")
    if seconds_to_steps(0.0, 0.5, name="calibration_seconds", allow_zero=True) != 0:
        raise AssertionError("0s calibration should convert to 0 steps when allow_zero=True")
    if seconds_to_steps(5.0, 0.5, name="horizon_seconds") != 10:
        raise AssertionError("5s should convert to 10 steps at 0.5s")
    if seconds_to_steps(2.5, 0.5, name="horizon_seconds") != 5:
        raise AssertionError("2.5s should convert to 5 steps at 0.5s")


def test_head_channel_mode_masks_excluded_channels_after_normalization() -> None:
    raw_head = np.asarray(
        [
            [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            [2.0, 3.0, 4.0, 11.0, 21.0, 31.0],
            [3.0, 4.0, 5.0, 12.0, 22.0, 32.0],
            [4.0, 5.0, 6.0, 13.0, 23.0, 33.0],
            [5.0, 6.0, 7.0, 14.0, 24.0, 34.0],
            [6.0, 7.0, 8.0, 15.0, 25.0, 35.0],
        ],
        dtype=np.float32,
    )
    direct_linear = apply_head_channel_mode(raw_head, "linear_only")
    direct_angular = apply_head_channel_mode(raw_head, "angular_only")
    if direct_linear.shape != raw_head.shape or direct_angular.shape != raw_head.shape:
        raise AssertionError("head_channel_mode must preserve the 6D motion shape.")
    if not np.allclose(direct_linear[:, 3:], 0.0) or not np.allclose(direct_angular[:, :3], 0.0):
        raise AssertionError("Direct head channel masking did not zero the excluded channels.")

    session = DenseFMSSession(
        head=raw_head,
        fms=np.arange(6, dtype=np.float32),
        time=np.arange(6, dtype=np.float32) * 0.5,
        participant_id="PA",
        session_id="head_channel_fixture",
        source_file="head_channel_fixture.csv",
        fms_raw=np.arange(6, dtype=np.float32),
        head_raw=raw_head.copy(),
    )
    scalers = fit_scalers([session], calibration_steps=1, horizon_steps=1)
    all_head = transform_sessions([session], scalers, head_channel_mode="all")[0].head
    linear_head = transform_sessions([session], scalers, head_channel_mode="linear_only")[0].head
    angular_head = transform_sessions([session], scalers, head_channel_mode="angular_only")[0].head
    if not np.allclose(linear_head[:, :3], all_head[:, :3]) or not np.allclose(linear_head[:, 3:], 0.0):
        raise AssertionError("linear_only should keep normalized acceleration and zero angular channels.")
    if not np.allclose(angular_head[:, 3:], all_head[:, 3:]) or not np.allclose(angular_head[:, :3], 0.0):
        raise AssertionError("angular_only should keep normalized angular velocity and zero acceleration channels.")


def test_online_current_goal_selection_metrics_use_original_low_bin() -> None:
    metrics = compute_online_current_goal_selection_metrics(
        y_true=[0.0, 1.5, 2.0, 4.0, 12.0],
        y_pred=[3.0, 5.0, 10.0, 4.0, 12.0],
        mae=2.9,
        r2=0.68,
        high8_f1=0.9,
        high12_f1=0.7,
    )
    low_strict = metrics["low_fms"]["0_2"]
    low_inclusive = metrics["low_fms"]["0_2_inclusive"]
    if low_strict["n"] != 2 or abs(low_strict["signed_bias"] - 3.25) > 1e-9:
        raise AssertionError(f"Strict [0,2) low-bin metrics are wrong: {low_strict}")
    if low_inclusive["n"] != 3 or low_inclusive["signed_bias"] <= low_strict["signed_bias"]:
        raise AssertionError(f"Inclusive <=2 helper bin should remain distinct: {low_inclusive}")
    if metrics["goal_composite"]["strict120"] <= 2.9:
        raise AssertionError(f"Composite score should penalize low-bias/R2/high12 deficits: {metrics}")


def test_online_current_goal_summary_reports_strict_low_bin() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "predictions.csv"
        path.write_text(
            "target_fms_now,predicted_fms_now\n"
            "0.0,3.0\n"
            "1.5,5.0\n"
            "2.0,10.0\n"
            "12.0,12.0\n",
            encoding="utf-8",
        )
        row = summarize_goal_metric_row(
            "fixture",
            path,
            pred_column="predicted_fms_now",
            low_bin_max=2.0,
            thresholds=[8.0, 12.0],
        )
    if row["strict_low_n"] != 2 or abs(row["strict_low_signed_bias"] - 3.25) > 1e-9:
        raise AssertionError(f"Strict original [0,2) low-bin summary is wrong: {row}")
    if row["low_n"] != 3 or row["low_signed_bias"] <= row["strict_low_signed_bias"]:
        raise AssertionError(f"Inclusive helper low-bin should remain separate from strict low-bin: {row}")


def test_online_current_recovery_low_suppressor_gate_target() -> None:
    fms = torch.tensor(
        [
            [5.0, 5.0, 0.0, 1.0, 4.0, 10.0],
            [0.0, 0.0, 0.0, 1.0, 4.0, 10.0],
        ],
        dtype=torch.float32,
    ) / 20.0
    outputs = {
        "current": torch.full((2, 4), 4.0 / 20.0),
        "mask": torch.ones((2, 4), dtype=torch.bool),
        "prediction_start": torch.tensor(2),
        "current_low_suppressor_gate_logits": torch.zeros((2, 4), dtype=torch.float32),
        "model_anchor_fms": torch.tensor([8.0, 0.0], dtype=torch.float32) / 20.0,
    }
    common_kwargs = dict(
        fms=fms,
        lengths=torch.tensor([6, 6]),
        fms_scaler={"min": 0.0, "max": 20.0},
        rise_horizon_steps=[],
        rise_thresholds=[],
        ordinal_bins=list(range(21)),
        risk_loss_weight=0.0,
        smoothness_weight=0.0,
        current_reg_aux_weight=0.0,
        ordinal_loss_weight=0.0,
        low_suppressor_gate_loss_weight=1.0,
        low_suppressor_threshold=2.0,
        low_suppressor_gate_pos_weight=3.0,
    )
    low_loss, low_parts = compute_online_current_risk_loss(
        outputs,
        low_suppressor_gate_target_mode="low",
        **common_kwargs,
    )
    recovery_loss, recovery_parts = compute_online_current_risk_loss(
        outputs,
        low_suppressor_gate_target_mode="recovery_low",
        low_suppressor_anchor_threshold=5.0,
        **common_kwargs,
    )
    if not torch.isfinite(low_loss) or not torch.isfinite(recovery_loss):
        raise AssertionError("Recovery low suppressor target produced non-finite loss.")
    if not recovery_parts["loss_low_suppressor_gate"] < low_parts["loss_low_suppressor_gate"]:
        raise AssertionError(
            "recovery_low should have fewer positive gate targets than low mode under this fixture: "
            f"low={low_parts} recovery={recovery_parts}"
        )


def test_target_shift_variable_horizon() -> None:
    fms = torch.arange(20, dtype=torch.float32).unsqueeze(0)
    lengths = torch.tensor([20])
    target_5s, mask_5s = future_sequence_targets(fms, lengths, calibration_steps=4, horizon_steps=10)
    target_2p5s, mask_2p5s = future_sequence_targets(fms, lengths, calibration_steps=4, horizon_steps=5)
    if not torch.isclose(target_5s[0, 0], fms[0, 14]):
        raise AssertionError("5s horizon should target t+10")
    if not torch.isclose(target_2p5s[0, 0], fms[0, 9]):
        raise AssertionError("2.5s horizon should target t+5")
    if int(mask_5s.sum()) != 6 or int(mask_2p5s.sum()) != 11:
        raise AssertionError("Unexpected valid target counts for variable horizons")


def test_calibration_input_variable_length_no_post_calib_fms() -> None:
    sig = inspect.signature(build_model("coff_lstm", calibration_steps=120, horizon_steps=2, recent_steps=2).forward)
    if list(sig.parameters) != ["head", "y_calib", "lengths", "static"]:
        raise AssertionError("Model forward should only accept calibration FMS, not full post-calibration FMS")
    model = build_model("coff_lstm", calibration_steps=120, horizon_steps=2, recent_steps=2)
    head = torch.randn(1, 130, 6)
    y_calib = torch.rand(1, 120)
    out = model(head, y_calib, torch.tensor([130]))
    if out["future"].shape[1] != 8:
        raise AssertionError("Variable calibration length produced wrong prediction count")


def test_recent_window_no_future_leakage_when_calib_shorter_than_recent() -> None:
    torch.manual_seed(123)
    model = build_model("coff_lstm", calibration_steps=2, horizon_steps=2, recent_steps=5)
    model.eval()
    head_a = torch.randn(1, 12, 6)
    head_b = head_a.clone()
    head_b[:, 3:, :] += 1000.0
    y_calib = torch.rand(1, 2)
    lengths = torch.tensor([12])
    with torch.no_grad():
        pred_a = model(head_a, y_calib, lengths)["future"][:, 0]
        pred_b = model(head_b, y_calib, lengths)["future"][:, 0]
    if not torch.allclose(pred_a, pred_b, atol=1e-6):
        raise AssertionError("Recent window for first prediction leaked samples after current t")


def test_recent_transformer_forward_and_mask_shape() -> None:
    torch.manual_seed(7)
    model = build_model("coff_lstm", calibration_steps=4, horizon_steps=2, recent_steps=5, recent_encoder="transformer")
    head = torch.randn(2, 12, 6)
    y_calib = torch.rand(2, 4)
    lengths = torch.tensor([12, 10])
    out = model(head, y_calib, lengths)
    if out["future"].shape != out["mask"].shape:
        raise AssertionError("Transformer recent encoder future/mask shapes differ")
    if out["future"].shape[0] != 2 or out["future"].shape[1] != 6:
        raise AssertionError(f"Unexpected transformer recent output shape: {out['future'].shape}")


def test_tcn_transformer_prediction_masks_match() -> None:
    head = torch.randn(2, 12, 6)
    y_calib = torch.rand(2, 4)
    lengths = torch.tensor([12, 10])
    tcn = build_model("coff_lstm", calibration_steps=4, horizon_steps=2, recent_steps=5, recent_encoder="tcn")
    attn = build_model("coff_lstm", calibration_steps=4, horizon_steps=2, recent_steps=5, recent_encoder="transformer")
    out_tcn = tcn(head, y_calib, lengths)
    out_attn = attn(head, y_calib, lengths)
    if out_tcn["mask"].shape != out_attn["mask"].shape or not torch.equal(out_tcn["mask"], out_attn["mask"]):
        raise AssertionError("TCN and transformer recent encoders should use identical prediction masks")


def test_recent_transformer_no_future_leakage_when_calib_shorter_than_recent() -> None:
    torch.manual_seed(1234)
    model = build_model("coff_lstm", calibration_steps=2, horizon_steps=2, recent_steps=5, recent_encoder="transformer")
    model.eval()
    head_a = torch.randn(1, 12, 6)
    head_b = head_a.clone()
    head_b[:, 3:, :] -= 1000.0
    y_calib = torch.rand(1, 2)
    lengths = torch.tensor([12])
    with torch.no_grad():
        pred_a = model(head_a, y_calib, lengths)["future"][:, 0]
        pred_b = model(head_b, y_calib, lengths)["future"][:, 0]
    if not torch.allclose(pred_a, pred_b, atol=1e-6):
        raise AssertionError("Transformer recent encoder leaked samples after current t")


def test_lc_sa_tcnformer_forward_shape_and_start_index() -> None:
    torch.manual_seed(11)
    model = build_model(
        "lc_sa_tcnformer",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=6,
        sampling_interval=0.5,
        horizon_seconds=1.0,
        anchor_mode="calibration_end",
        d_model=32,
        transformer_heads=4,
        calib_dilations=[1, 2],
        recent_dilations=[1, 2],
    )
    head = torch.randn(2, 14, 6)
    fms = torch.rand(2, 14)
    lengths = torch.tensor([14, 12])
    out = model(head, fms[:, :4], lengths)
    if int(out["prediction_start"].item()) != 5:
        raise AssertionError(f"LC-SA prediction_start should be max(C,W-1)=5, got {out['prediction_start']}")
    if out["future"].shape != out["mask"].shape or out["future"].shape != (2, 7):
        raise AssertionError(f"Unexpected LC-SA output/mask shape: {out['future'].shape}, {out['mask'].shape}")


def test_lc_sa_target_shift_uses_prediction_start() -> None:
    outputs = {
        "future": torch.full((1, 3), 0.5),
        "mask": torch.tensor([[True, True, True]]),
        "prediction_start": torch.tensor(5),
    }
    fms = torch.arange(12, dtype=torch.float32).unsqueeze(0) / 20.0
    lengths = torch.tensor([12])
    loss_fn = FutureSequenceLoss("level_only")
    _, parts = compute_loss(outputs, fms, lengths, calibration_steps=4, horizon_steps=2, loss_fn=loss_fn)
    targets, _ = future_sequence_targets(fms, lengths, calibration_steps=4, horizon_steps=2, max_pred_steps=3, prediction_start_steps=5)
    expected, _ = loss_fn(outputs["future"], targets, outputs["mask"])
    if parts["valid_points"] != 3:
        raise AssertionError(f"Expected 3 valid points, got {parts['valid_points']}")
    actual, _ = compute_loss(outputs, fms, lengths, calibration_steps=4, horizon_steps=2, loss_fn=loss_fn)
    if not torch.allclose(actual, expected):
        raise AssertionError("compute_loss did not honor model prediction_start for target shift")


def test_lc_sa_anchor_policy_correctness() -> None:
    fms = torch.arange(20, dtype=torch.float32).unsqueeze(0) / 20.0
    head = torch.zeros(1, 20, 6)
    lengths = torch.tensor([20])
    model = build_model(
        "lc_sa_tcnformer",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=4,
        sampling_interval=0.5,
        anchor_mode="sparse_observed",
        anchor_interval_seconds=3.0,
        d_model=32,
        transformer_heads=4,
        calib_dilations=[1],
        recent_dilations=[1],
    )
    out = model(head, fms, lengths)
    anchor = out["anchor_index"][0]
    positions = int(out["prediction_start"].item()) + torch.arange(anchor.numel())
    if not bool(torch.all(anchor <= positions)):
        raise AssertionError("LC-SA sparse anchors must not be after current index")
    if int(anchor[0].item()) != 3:
        raise AssertionError(f"First sparse anchor should clamp to calibration end index 3, got {anchor[0]}")


def test_lc_sa_sparse_anchor_uses_latest_finite_observation() -> None:
    fms = torch.arange(20, dtype=torch.float32).unsqueeze(0) / 20.0
    fms[0, 6] = float("nan")
    head = torch.zeros(1, 20, 6)
    lengths = torch.tensor([20])
    model = build_model(
        "lc_sa_tcnformer",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=4,
        sampling_interval=0.5,
        anchor_mode="sparse_observed",
        anchor_interval_seconds=3.0,
        d_model=32,
        transformer_heads=4,
        calib_dilations=[1],
        recent_dilations=[1],
    )
    out = model(head, fms, lengths)
    if not torch.isfinite(out["future"]).all():
        raise AssertionError("Sparse anchor with a missing scheduled FMS produced non-finite predictions")
    positions = int(out["prediction_start"].item()) + torch.arange(out["anchor_index"].shape[1])
    scheduled_at_six = torch.where(positions == 6)[0]
    if scheduled_at_six.numel() and int(out["anchor_index"][0, int(scheduled_at_six[0])].item()) != 5:
        raise AssertionError("Sparse anchor should fall back to latest finite FMS at or before scheduled anchor")


def test_fms_context_mode_tensor_policy() -> None:
    fms = torch.tensor([[0.2, 0.4, 0.6, 0.8, 1.0]], dtype=torch.float32)
    none_ctx = calibration_context_fms(fms, 4, "none")
    start_ctx = calibration_context_fms(fms, 4, "start_only")
    sparse_ctx = calibration_context_fms(fms, 4, "sparse_anchor")
    hist_ctx = calibration_context_fms(fms, 4, "calibration_history")
    if not torch.equal(none_ctx, torch.zeros_like(none_ctx)):
        raise AssertionError("fms_context_mode=none should hide all calibration FMS values")
    if not torch.equal(hist_ctx, fms[:, :4]):
        raise AssertionError("calibration_history should expose the dense calibration FMS sequence")
    if not torch.equal(start_ctx, hist_ctx) or not torch.equal(sparse_ctx, hist_ctx):
        raise AssertionError("start_only and sparse_anchor should keep calibration FMS history separate from rolling anchor context")


def test_lc_sa_start_only_uses_window_start_fms() -> None:
    torch.manual_seed(21)
    model = build_model(
        "lc_sa_tcnformer",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=4,
        sampling_interval=0.5,
        anchor_mode="none",
        fms_context_mode="start_only",
        d_model=32,
        transformer_heads=4,
        calib_dilations=[1],
        recent_dilations=[1],
    )
    model.eval()
    head = torch.randn(1, 16, 6)
    fms = torch.linspace(0.1, 0.9, 16).unsqueeze(0)
    lengths = torch.tensor([16])
    try:
        model(head, fms[:, :4], lengths)
    except ValueError as exc:
        if "requires full FMS" not in str(exc):
            raise
    else:
        raise AssertionError("start_only should require full FMS so it can read the recent-window start FMS")
    with torch.no_grad():
        out = model(head, fms, lengths)
    if "anchor_index" not in out or "anchor_fms" not in out:
        raise AssertionError("start_only should report recent-window start anchor indices and FMS")
    positions = int(out["prediction_start"].item()) + torch.arange(out["anchor_index"].shape[1])
    expected_idx = positions - model.recent_steps + 1
    if not torch.equal(out["anchor_index"][0].cpu(), expected_idx.cpu()):
        raise AssertionError("start_only anchor_index should be the start index of each recent motion window")
    expected_fms = fms[0, expected_idx].cpu()
    if not torch.allclose(out["anchor_fms"][0].cpu(), expected_fms):
        raise AssertionError("start_only anchor_fms should gather FMS at the recent-window start index")
    fms_future_changed = fms.clone()
    fms_future_changed[:, int(expected_idx.max().item()) + 1 :] = torch.linspace(0.95, 0.05, fms_future_changed.shape[1] - int(expected_idx.max().item()) - 1)
    with torch.no_grad():
        out_future_changed = model(head, fms_future_changed, lengths)
    if not torch.allclose(out["future"], out_future_changed["future"], atol=1e-6):
        raise AssertionError("start_only should not use FMS after the latest recent-window start anchor")


def test_prediction_csv_start_fms_fallback_metadata() -> None:
    raw_fms = np.arange(10, dtype=np.float32)
    raw_fms[4] = np.nan
    norm_fms = raw_fms / 20.0
    session = DenseFMSSession(
        head=np.zeros((10, 6), dtype=np.float32),
        fms=norm_fms.astype(np.float32),
        time=np.arange(10, dtype=np.float32) * 0.5,
        participant_id="PA_SYN",
        session_id="synthetic_start_fms_fallback",
        source_file="synthetic_start_fms_fallback.csv",
        fms_raw=raw_fms,
        head_raw=np.zeros((10, 6), dtype=np.float32),
    )
    loader = DataLoader(DenseFMSSessionDataset([session]), batch_size=1, collate_fn=collate_sessions)
    model = build_model(
        "lc_sa_tcnformer",
        head_dim=6,
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=1,
        sampling_interval=0.5,
        anchor_mode="none",
        fms_context_mode="start_only",
        d_model=16,
        transformer_heads=4,
        transformer_ff_dim=32,
    )
    result = collect_predictions(
        model,
        loader,
        torch.device("cpu"),
        calibration_steps=4,
        horizon_steps=2,
        fms_scaler={"min": 0.0, "max": 20.0},
        calibration_seconds=2.0,
        horizon_seconds=1.0,
        recent_window_seconds=0.5,
        recent_window_steps=1,
        sampling_interval=0.5,
        anchor_mode="none",
        fms_context_mode="start_only",
        split_name="val",
    )
    first = result["prediction_records"][0]
    if first["nominal_start_index"] != 4:
        raise AssertionError(f"Expected nominal_start_index=4, got {first['nominal_start_index']}")
    if first["start_fms_index"] != 3 or first["anchor_index"] != 3:
        raise AssertionError(f"Expected fallback start/anchor index 3, got {first}")
    if first["anchor_is_fallback"] is not True:
        raise AssertionError(f"Expected anchor_is_fallback=True, got {first['anchor_is_fallback']}")
    if abs(float(first["start_fms_value"]) - 3.0) > 1e-7:
        raise AssertionError(f"Expected fallback start_fms_value=3.0, got {first['start_fms_value']}")


def test_sparse_anchor_mode_requires_full_fms_and_uses_calibration_history() -> None:
    torch.manual_seed(22)
    model = build_model(
        "lc_sa_tcnformer",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=4,
        sampling_interval=0.5,
        anchor_mode="sparse_observed",
        anchor_interval_seconds=3.0,
        fms_context_mode="sparse_anchor",
        d_model=32,
        transformer_heads=4,
        calib_dilations=[1],
        recent_dilations=[1],
    )
    model.eval()
    head = torch.randn(1, 16, 6)
    fms = torch.linspace(0.1, 0.9, 16).unsqueeze(0)
    try:
        model(head, fms[:, :4], torch.tensor([16]))
    except ValueError as exc:
        if "requires full FMS" not in str(exc):
            raise
    else:
        raise AssertionError("sparse_anchor mode should require full FMS for sparse observed anchors")
    out = model(head, fms, torch.tensor([16]))
    if "anchor_index" not in out:
        raise AssertionError("sparse_anchor diagnostic should report anchor indices")
    sparse_ctx = calibration_context_fms(fms, 4, "sparse_anchor")
    if not torch.allclose(sparse_ctx, fms[:, :4]):
        raise AssertionError("sparse_anchor calibration context should keep the full calibration FMS history")


def test_lc_sa_no_recent_motion_future_leakage() -> None:
    torch.manual_seed(12)
    model = build_model(
        "lc_sa_tcnformer",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=6,
        sampling_interval=0.5,
        anchor_mode="calibration_end",
        d_model=32,
        transformer_heads=4,
        calib_dilations=[1, 2],
        recent_dilations=[1, 2],
    )
    model.eval()
    head_a = torch.randn(1, 14, 6)
    head_b = head_a.clone()
    # First prediction is at t=5, so changing t>=6 must not change it.
    head_b[:, 6:, :] += 1000.0
    fms = torch.rand(1, 14)
    lengths = torch.tensor([14])
    with torch.no_grad():
        pred_a = model(head_a, fms[:, :4], lengths)["future"][:, 0]
        pred_b = model(head_b, fms[:, :4], lengths)["future"][:, 0]
    if not torch.allclose(pred_a, pred_b, atol=1e-6):
        raise AssertionError("LC-SA recent encoder leaked motion after current time")


def test_lc_sa_dynamic_dilation_schedule_and_rf() -> None:
    model = build_model(
        "lc_sa_tcnformer",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=60,
        sampling_interval=0.5,
        anchor_mode="none",
        d_model=32,
        transformer_heads=4,
        calib_dilations=[1],
        recent_dilations="auto",
    )
    if model.recent_dilations != [1, 2, 4, 8]:
        raise AssertionError(f"30s recent window should use [1,2,4,8], got {model.recent_dilations}")
    if model.recent_rf_steps != 61:
        raise AssertionError(f"Unexpected receptive field for kernel=3 and [1,2,4,8]: {model.recent_rf_steps}")


def test_new_long_target_model_family_forward_shapes() -> None:
    torch.manual_seed(31)
    head = torch.randn(2, 48, 6)
    fms = torch.rand(2, 48)
    lengths = torch.tensor([48, 45])
    static = torch.randn(2, 5)
    for model_name in ["anchor_delta_mlp", "anchor_delta_gru", "recent_tcn_summary_calib", "gated_fusion"]:
        model = build_model(
            model_name,
            head_dim=6,
            calibration_steps=8,
            horizon_steps=4,
            recent_steps=10,
            sampling_interval=0.5,
            horizon_seconds=2.0,
            anchor_mode="sparse_observed",
            anchor_interval_seconds=5.0,
            predict_delta_from_anchor=True,
            use_static=True,
            static_dim=5,
            d_model=32,
            hidden_dim=32,
            transformer_heads=4,
            calib_dilations=[1],
            recent_dilations=[1],
        )
        out = model(head, fms, lengths, static=static)
        if out["future"].shape != out["mask"].shape or out["future"].shape[0] != 2:
            raise AssertionError(f"{model_name} future/mask shape mismatch: {out['future'].shape}, {out['mask'].shape}")
        if int(out["prediction_start"].item()) != 9:
            raise AssertionError(f"{model_name} prediction_start should be 9, got {out['prediction_start']}")
        if "anchor_index" not in out or not bool(torch.all(out["anchor_index"] <= (9 + torch.arange(out["anchor_index"].shape[1])))):
            raise AssertionError(f"{model_name} sparse anchor index must be <= current prediction index")


def test_new_long_target_multi_horizon_forward_shape() -> None:
    torch.manual_seed(32)
    model = build_model(
        "anchor_delta_mlp",
        head_dim=6,
        calibration_steps=8,
        horizon_steps=2,
        recent_steps=10,
        sampling_interval=0.5,
        horizon_seconds=1.0,
        anchor_mode="sparse_observed",
        anchor_interval_seconds=5.0,
        predict_delta_from_anchor=True,
        use_static=True,
        static_dim=5,
        hidden_dim=32,
        multi_horizon=True,
        horizon_set=[1.0, 2.5, 5.0],
    )
    head = torch.randn(2, 48, 6)
    fms = torch.rand(2, 48)
    lengths = torch.tensor([48, 45])
    out = model(head, fms, lengths, static=torch.randn(2, 5))
    if out["future"].ndim != 3 or out["future"].shape[-1] != 3:
        raise AssertionError(f"Multi-horizon output should be [B,T,3], got {out['future'].shape}")
    if not torch.equal(out["horizon_steps_list"], torch.tensor([2, 5, 10])):
        raise AssertionError(f"Unexpected multi-horizon steps: {out['horizon_steps_list']}")


def test_goal_dynamic_heads_dual_head_and_motion_features() -> None:
    torch.manual_seed(321)
    model = build_model(
        "lc_sa_tcnformer",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=5,
        sampling_interval=0.5,
        fms_context_mode="start_only",
        anchor_mode="none",
        multi_horizon=True,
        horizon_set=[5.0, 10.0, 15.0],
        per_horizon_heads=True,
        d_model=32,
        transformer_heads=4,
        transformer_ff_dim=64,
        forecast_head_mode="dual_gated",
        horizon_head_mode="h15_residual",
        horizon_head_hidden_dim=32,
        motion_feature_mode="norm_delta_energy",
    )
    head = torch.randn(2, 80, 6)
    fms = torch.linspace(0.1, 0.9, 80).repeat(2, 1)
    lengths = torch.tensor([80, 72])
    out = model(head, fms, lengths)
    if out["future"].ndim != 3 or out["future"].shape[-1] != 3:
        raise AssertionError(f"Dual-head multi-horizon output shape wrong: {out['future'].shape}")
    for key in ("future_level", "future_delta_pred", "future_delta_value", "future_delta_base", "future_gate"):
        if key not in out:
            raise AssertionError(f"Dual-head output missing {key}")
        expected = out["future"].shape
        if out[key].shape != expected:
            raise AssertionError(f"{key} shape {out[key].shape} != {expected}")
    if getattr(model, "motion_feature_mode") != "norm_delta_energy":
        raise AssertionError("motion_feature_mode was not preserved on the model")


def test_predict_delta_from_anchor_backcompat_sets_delta_head() -> None:
    model = build_model(
        "lc_sa_tcnformer",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=5,
        sampling_interval=0.5,
        fms_context_mode="start_only",
        anchor_mode="none",
        predict_delta_from_anchor=True,
        d_model=32,
        transformer_heads=4,
        transformer_ff_dim=64,
    )
    if getattr(model, "forecast_head_mode") != "delta":
        raise AssertionError("predict_delta_from_anchor should keep backward-compatible delta behavior")


def test_lc_sa_calibration_end_delta_uses_calibration_only_fms() -> None:
    torch.manual_seed(909)
    model = build_model(
        "lc_sa_tcnformer",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=5,
        sampling_interval=0.5,
        fms_context_mode="calibration_history",
        anchor_mode="calibration_end",
        anchor_interval_seconds=0.0,
        multi_horizon=True,
        horizon_set=[5.0, 10.0, 15.0],
        per_horizon_heads=True,
        d_model=32,
        transformer_heads=4,
        transformer_ff_dim=64,
        forecast_head_mode="delta",
        motion_feature_mode="norm",
    )
    if getattr(model, "requires_full_fms"):
        raise AssertionError("calibration_end delta must not require post-calibration FMS input")
    model.eval()
    head = torch.randn(2, 36, 6)
    fms_full = torch.rand(2, 36)
    fms_changed_after_calib = fms_full.clone()
    fms_changed_after_calib[:, 4:] = 1.0 - fms_changed_after_calib[:, 4:]
    lengths = torch.tensor([36, 31])
    with torch.no_grad():
        out_calib_only = model(head, fms_full[:, :4], lengths)
        out_full_a = model(head, fms_full, lengths)
        out_full_b = model(head, fms_changed_after_calib, lengths)
        head_future_changed = head.clone()
        head_future_changed[:, 5:, :] += 1000.0
        out_future_changed = model(head_future_changed, fms_full[:, :4], lengths)
    if not torch.allclose(out_calib_only["future"], out_full_a["future"], atol=1e-6):
        raise AssertionError("calibration_end delta changed when passing full FMS instead of calibration slice")
    if not torch.allclose(out_full_a["future"], out_full_b["future"], atol=1e-6):
        raise AssertionError("calibration_end delta leaked post-calibration FMS values")
    if "anchor_index" not in out_full_a or not torch.equal(out_full_a["anchor_index"], torch.full_like(out_full_a["anchor_index"], 3)):
        raise AssertionError(f"calibration_end anchor index should stay at calibration_steps-1, got {out_full_a.get('anchor_index')}")


def test_calib_init_state_forecaster_uses_only_calibration_fms() -> None:
    torch.manual_seed(910)
    model = build_model(
        "calib_init_state_forecaster",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=5,
        sampling_interval=0.5,
        fms_context_mode="calibration_history",
        anchor_mode="calibration_end",
        anchor_interval_seconds=0.0,
        multi_horizon=True,
        horizon_set=[5.0, 10.0, 15.0],
        per_horizon_heads=True,
        d_model=32,
        hidden_dim=32,
        transformer_heads=4,
        transformer_ff_dim=64,
        forecast_head_mode="delta",
        motion_feature_mode="norm",
    )
    if getattr(model, "requires_full_fms"):
        raise AssertionError("calib-init state forecaster must not require post-calibration FMS input")
    model.eval()
    head = torch.randn(2, 40, 6)
    fms_full = torch.rand(2, 40)
    fms_changed_after_calib = fms_full.clone()
    fms_changed_after_calib[:, 4:] = 1.0 - fms_changed_after_calib[:, 4:]
    lengths = torch.tensor([40, 35])
    with torch.no_grad():
        out_calib_only = model(head, fms_full[:, :4], lengths)
        out_full_a = model(head, fms_full, lengths)
        out_full_b = model(head, fms_changed_after_calib, lengths)
        head_future_changed = head.clone()
        head_future_changed[:, 5:, :] += 1000.0
        out_future_changed = model(head_future_changed, fms_full[:, :4], lengths)
    if out_calib_only["future"].shape[-1] != 3 or out_calib_only["future"].shape[:2] != out_calib_only["mask"].shape[:2]:
        raise AssertionError(f"Unexpected calib-init future/mask shapes: {out_calib_only['future'].shape}, {out_calib_only['mask'].shape}")
    if out_calib_only["current"].shape != out_calib_only["future"].shape[:2]:
        raise AssertionError(f"Unexpected current auxiliary shape: {out_calib_only['current'].shape}")
    if not torch.allclose(out_calib_only["future"], out_full_a["future"], atol=1e-6):
        raise AssertionError("calib-init predictions changed when full FMS was passed instead of calibration slice")
    if not torch.allclose(out_full_a["future"], out_full_b["future"], atol=1e-6):
        raise AssertionError("calib-init forecaster leaked post-calibration FMS values")
    if not torch.allclose(out_calib_only["future"][:, 0], out_future_changed["future"][:, 0], atol=1e-6):
        raise AssertionError("calib-init forecaster leaked motion after the first current index")
    if "anchor_fms" in out_full_a or "start_fms_value" in out_full_a:
        raise AssertionError("calib-init forecaster should not expose post-calibration FMS anchors")
    fms = torch.rand(2, 40)
    loss_fn = FutureSequenceLoss("level_only")
    _, parts = compute_loss(
        out_calib_only,
        fms,
        lengths,
        calibration_steps=4,
        horizon_steps=2,
        loss_fn=loss_fn,
        current_aux_weight=0.2,
    )
    if parts["loss_current_aux"] <= 0 or parts["current_aux_points"] <= 0:
        raise AssertionError(f"current auxiliary loss did not run: {parts}")


def test_calib_init_self_delta_uses_predicted_current_not_real_post_fms() -> None:
    torch.manual_seed(911)
    model = build_model(
        "calib_init_state_forecaster",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=5,
        sampling_interval=0.5,
        fms_context_mode="calibration_history",
        anchor_mode="none",
        anchor_interval_seconds=0.0,
        multi_horizon=True,
        horizon_set=[5.0, 10.0, 15.0],
        per_horizon_heads=True,
        d_model=32,
        hidden_dim=32,
        transformer_heads=4,
        transformer_ff_dim=64,
        forecast_head_mode="self_delta",
        delta_scale=1.0,
        motion_feature_mode="norm",
    )
    if getattr(model, "requires_full_fms"):
        raise AssertionError("self_delta must not require full FMS input")
    model.eval()
    head = torch.randn(2, 40, 6)
    fms_full = torch.rand(2, 40)
    fms_changed_after_calib = fms_full.clone()
    fms_changed_after_calib[:, 4:] = 1.0 - fms_changed_after_calib[:, 4:]
    lengths = torch.tensor([40, 35])
    with torch.no_grad():
        out_a = model(head, fms_full[:, :4], lengths)
        out_b = model(head, fms_full, lengths)
        out_c = model(head, fms_changed_after_calib, lengths)
    if not torch.allclose(out_a["future"], out_b["future"], atol=1e-6):
        raise AssertionError("self_delta changed when full FMS was passed instead of calibration slice")
    if not torch.allclose(out_b["future"], out_c["future"], atol=1e-6):
        raise AssertionError("self_delta leaked real post-calibration FMS values")
    if out_a["future_delta_base"].shape != out_a["current"].shape + (3,):
        raise AssertionError(f"self_delta base should be per-horizon predicted-current base, got {out_a['future_delta_base'].shape}")
    loss_fn = FutureSequenceLoss("level_only")
    _, parts = compute_loss(
        out_a,
        fms_full,
        lengths,
        calibration_steps=4,
        horizon_steps=2,
        loss_fn=loss_fn,
        dual_aux_beta=0.2,
        current_aux_weight=0.5,
        current_delta_aux_weight=0.2,
    )
    if parts["loss_dual_delta"] <= 0 or parts["loss_current_aux"] <= 0 or parts["loss_current_delta_aux"] <= 0:
        raise AssertionError(f"self_delta auxiliary losses did not run: {parts}")


def test_calib_init_recent_start_delta_uses_synthetic_anchor_not_real_fms() -> None:
    torch.manual_seed(913)
    model = build_model(
        "calib_init_state_forecaster",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=5,
        sampling_interval=0.5,
        fms_context_mode="calibration_history",
        anchor_mode="none",
        anchor_interval_seconds=0.0,
        multi_horizon=True,
        horizon_set=[1.0, 2.0, 3.0],
        per_horizon_heads=True,
        d_model=32,
        hidden_dim=32,
        transformer_heads=4,
        transformer_ff_dim=64,
        forecast_head_mode="recent_start_delta",
        delta_scale=1.0,
        motion_feature_mode="norm",
    )
    if getattr(model, "requires_full_fms"):
        raise AssertionError("recent_start_delta must not require full FMS input")
    model.eval()
    head = torch.randn(2, 30, 6)
    fms_full = torch.rand(2, 30)
    fms_changed_after_calib = fms_full.clone()
    fms_changed_after_calib[:, 4:] = 1.0 - fms_changed_after_calib[:, 4:]
    lengths = torch.tensor([30, 25])
    with torch.no_grad():
        out_calib = model(head, fms_full[:, :4], lengths)
        out_full = model(head, fms_full, lengths)
        out_changed = model(head, fms_changed_after_calib, lengths)
    if "synthetic_anchor_fms" not in out_calib or "synthetic_anchor_index" not in out_calib:
        raise AssertionError("recent_start_delta should expose synthetic anchor diagnostics")
    if "anchor_fms" in out_calib:
        raise AssertionError("recent_start_delta should not expose real FMS anchors")
    if not torch.allclose(out_calib["future"], out_full["future"], atol=1e-6):
        raise AssertionError("recent_start_delta changed when full FMS was passed instead of calibration slice")
    if not torch.allclose(out_full["future"], out_changed["future"], atol=1e-6):
        raise AssertionError("recent_start_delta leaked real post-calibration FMS values")
    if out_calib["future_delta_base"].shape != out_calib["future"].shape:
        raise AssertionError(
            f"recent_start_delta base should align with future shape, got {out_calib['future_delta_base'].shape}"
        )
    if not bool(out_calib["synthetic_anchor_is_predicted"].any().item()):
        raise AssertionError("recent_start_delta should use predicted anchors after the calibration boundary")
    loss_fn = FutureSequenceLoss("level_only")
    _, parts = compute_loss(
        out_calib,
        fms_full,
        lengths,
        calibration_steps=4,
        horizon_steps=2,
        loss_fn=loss_fn,
        dual_aux_beta=0.2,
        current_aux_weight=0.5,
        current_delta_aux_weight=0.2,
    )
    if parts["loss_dual_delta"] <= 0 or parts["loss_current_aux"] <= 0 or parts["loss_current_delta_aux"] <= 0:
        raise AssertionError(f"recent_start_delta auxiliary losses did not run: {parts}")


def test_calib_init_rollin_start_delta_uses_lagged_predictions_not_real_fms() -> None:
    torch.manual_seed(914)
    model = build_model(
        "calib_init_state_forecaster",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=5,
        sampling_interval=0.5,
        fms_context_mode="calibration_history",
        anchor_mode="none",
        anchor_interval_seconds=0.0,
        multi_horizon=True,
        horizon_set=[1.0, 2.0, 3.0],
        per_horizon_heads=True,
        d_model=32,
        hidden_dim=32,
        transformer_heads=4,
        transformer_ff_dim=64,
        forecast_head_mode="rollin_start_delta",
        delta_scale=1.0,
        motion_feature_mode="norm",
    )
    if getattr(model, "requires_full_fms"):
        raise AssertionError("rollin_start_delta must not require full FMS input")
    model.eval()
    head = torch.randn(2, 34, 6)
    fms_full = torch.rand(2, 34)
    fms_changed_after_calib = fms_full.clone()
    fms_changed_after_calib[:, 4:] = 1.0 - fms_changed_after_calib[:, 4:]
    lengths = torch.tensor([34, 30])
    with torch.no_grad():
        out_calib = model(head, fms_full[:, :4], lengths)
        out_full = model(head, fms_full, lengths)
        out_changed = model(head, fms_changed_after_calib, lengths)
    if "synthetic_anchor_fms" not in out_calib or "future_level" not in out_calib:
        raise AssertionError("rollin_start_delta should expose synthetic anchor and level-tape diagnostics")
    if "anchor_fms" in out_calib:
        raise AssertionError("rollin_start_delta should not expose real FMS anchors")
    if not torch.allclose(out_calib["future"], out_full["future"], atol=1e-6):
        raise AssertionError("rollin_start_delta changed when full FMS was passed instead of calibration slice")
    if not torch.allclose(out_full["future"], out_changed["future"], atol=1e-6):
        raise AssertionError("rollin_start_delta leaked real post-calibration FMS values")
    if out_calib["future_delta_base"].shape != out_calib["future"].shape:
        raise AssertionError(
            f"rollin_start_delta base should align with future shape, got {out_calib['future_delta_base'].shape}"
        )
    if not bool(out_calib["synthetic_anchor_is_predicted"].any().item()):
        raise AssertionError("rollin_start_delta should use lagged h-step predictions after warmup")
    loss_fn = FutureSequenceLoss("level_only")
    _, parts = compute_loss(
        out_calib,
        fms_full,
        lengths,
        calibration_steps=4,
        horizon_steps=2,
        loss_fn=loss_fn,
        dual_aux_alpha=0.2,
        dual_aux_beta=0.2,
        current_aux_weight=0.5,
        current_delta_aux_weight=0.2,
    )
    if (
        parts["loss_dual_level"] <= 0
        or parts["loss_dual_delta"] <= 0
        or parts["loss_current_aux"] <= 0
        or parts["loss_current_delta_aux"] <= 0
    ):
        raise AssertionError(f"rollin_start_delta auxiliary losses did not run: {parts}")


def test_calib_init_session_summary_context_is_calibration_only() -> None:
    torch.manual_seed(912)
    model = build_model(
        "calib_init_state_forecaster",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=5,
        sampling_interval=0.5,
        fms_context_mode="calibration_history",
        anchor_mode="none",
        anchor_interval_seconds=0.0,
        multi_horizon=True,
        horizon_set=[5.0, 10.0, 15.0],
        per_horizon_heads=True,
        d_model=32,
        hidden_dim=32,
        transformer_heads=4,
        transformer_ff_dim=64,
        forecast_head_mode="self_delta",
        delta_scale=1.0,
        motion_feature_mode="norm",
        session_context_mode="summary",
    )
    if getattr(model, "requires_full_fms"):
        raise AssertionError("session summary context must remain deployable without full FMS input")
    model.eval()
    head = torch.randn(2, 40, 6)
    fms_full = torch.rand(2, 40)
    fms_changed_after_calib = fms_full.clone()
    fms_changed_after_calib[:, 4:] = 1.0 - fms_changed_after_calib[:, 4:]
    lengths = torch.tensor([40, 35])
    with torch.no_grad():
        out_a = model(head, fms_full[:, :4], lengths)
        out_b = model(head, fms_full, lengths)
        out_c = model(head, fms_changed_after_calib, lengths)
    if "session_summary" not in out_a or out_a["session_summary"].shape != (2, 4):
        raise AssertionError(f"session summary output shape is wrong: {out_a.get('session_summary')}")
    if not torch.allclose(out_a["future"], out_b["future"], atol=1e-6):
        raise AssertionError("session summary context changed when passing full FMS instead of calibration slice")
    if not torch.allclose(out_b["future"], out_c["future"], atol=1e-6):
        raise AssertionError("session summary context leaked post-calibration FMS values")
    target, mask = compute_session_summary_targets(fms_full, lengths, calibration_steps=4)
    if target.shape != (2, 4) or not mask.all():
        raise AssertionError(f"session summary target generation failed: {target}, {mask}")
    loss_fn = FutureSequenceLoss("level_only")
    _, parts = compute_loss(
        out_a,
        fms_full,
        lengths,
        calibration_steps=4,
        horizon_steps=2,
        loss_fn=loss_fn,
        current_aux_weight=0.5,
        session_aux_weight=0.3,
        session_aux_loss_type="smooth_l1",
    )
    if parts["loss_session_aux"] <= 0 or parts["session_aux_points"] != 8:
        raise AssertionError(f"session auxiliary loss did not run: {parts}")


def test_teacher_future_distillation_with_start_only_teacher_keeps_student_deployable() -> None:
    torch.manual_seed(913)
    student = build_model(
        "calib_init_state_forecaster",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=5,
        sampling_interval=0.5,
        fms_context_mode="calibration_history",
        anchor_mode="none",
        anchor_interval_seconds=0.0,
        multi_horizon=True,
        horizon_set=[1.0, 2.0],
        per_horizon_heads=True,
        d_model=32,
        hidden_dim=32,
        transformer_heads=4,
        transformer_ff_dim=64,
        forecast_head_mode="self_delta",
        motion_feature_mode="norm",
    )
    teacher = build_model(
        "lc_sa_tcnformer",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=5,
        sampling_interval=0.5,
        horizon_seconds=1.0,
        anchor_mode="none",
        fms_context_mode="start_only",
        multi_horizon=True,
        horizon_set=[1.0, 2.0],
        per_horizon_heads=True,
        d_model=32,
        transformer_heads=4,
        transformer_ff_dim=64,
        calib_dilations=[1, 2],
        recent_dilations=[1, 2],
        forecast_head_mode="delta",
    )
    if getattr(student, "requires_full_fms"):
        raise AssertionError("Distilled student must remain deployable without post-calibration FMS.")
    if not getattr(teacher, "requires_full_fms"):
        raise AssertionError("start_only teacher should require full FMS during privileged training.")
    student.eval()
    teacher.eval()
    head = torch.randn(2, 24, 6)
    fms = torch.linspace(0.05, 0.95, 24).unsqueeze(0).repeat(2, 1)
    fms[1] = torch.flip(fms[1], dims=[0])
    lengths = torch.tensor([24, 22])
    fms_changed_after_calib = fms.clone()
    fms_changed_after_calib[:, 4:] = 1.0 - fms_changed_after_calib[:, 4:]
    with torch.no_grad():
        student_out = student(head, fms[:, :4], lengths)
        student_out_full_changed = student(head, fms_changed_after_calib, lengths)
        teacher_out = teacher(head, fms, lengths)
        teacher_out_changed = teacher(head, fms_changed_after_calib, lengths)
    if not torch.allclose(student_out["future"], student_out_full_changed["future"], atol=1e-6):
        raise AssertionError("Distillation student leaked post-calibration FMS before teacher loss.")
    if torch.allclose(teacher_out["future"], teacher_out_changed["future"], atol=1e-6):
        raise AssertionError("start_only teacher did not react to changed recent-window start FMS values.")
    distill_loss, parts = compute_teacher_future_distillation_loss(
        student_out,
        teacher_out,
        horizon_steps=2,
        loss_type="smooth_l1",
    )
    if not torch.isfinite(distill_loss) or parts["teacher_distill_points"] <= 0:
        raise AssertionError(f"teacher distillation loss did not produce valid points: loss={distill_loss}, parts={parts}")
    delta_loss, delta_parts = compute_teacher_delta_distillation_loss(
        student_out,
        teacher_out,
        horizon_steps=2,
        loss_type="smooth_l1",
    )
    if not torch.isfinite(delta_loss) or delta_parts["teacher_delta_distill_points"] <= 0:
        raise AssertionError(
            f"teacher delta distillation loss did not produce valid points: loss={delta_loss}, parts={delta_parts}"
        )
    if student_out["distill_repr"].ndim != 4 or teacher_out["distill_repr"].ndim != 4:
        raise AssertionError(
            f"multi-horizon distill_repr should be [B,P,H,D], got "
            f"student={student_out['distill_repr'].shape}, teacher={teacher_out['distill_repr'].shape}"
        )
    projector = torch.nn.Linear(student_out["distill_repr"].shape[-1], teacher_out["distill_repr"].shape[-1])
    repr_loss, repr_parts = compute_teacher_repr_distillation_loss(
        student_out,
        teacher_out,
        horizon_steps=2,
        student_projector=projector,
        loss_type="smooth_l1",
    )
    if not torch.isfinite(repr_loss) or repr_parts["teacher_repr_distill_points"] <= 0:
        raise AssertionError(
            f"teacher representation distillation loss did not produce valid points: "
            f"loss={repr_loss}, parts={repr_parts}"
        )


def test_teacher_checkpoint_loader_and_compatibility_guard() -> None:
    model_kwargs = {
        "calibration_steps": 4,
        "horizon_steps": 2,
        "recent_steps": 5,
        "sampling_interval": 0.5,
        "horizon_seconds": 1.0,
        "anchor_mode": "none",
        "fms_context_mode": "start_only",
        "d_model": 32,
        "transformer_heads": 4,
        "transformer_ff_dim": 64,
        "calib_dilations": [1, 2],
        "recent_dilations": [1, 2],
    }
    teacher = build_model("lc_sa_tcnformer", **model_kwargs)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "teacher.pt"
        torch.save(
            {
                "model_state_dict": teacher.state_dict(),
                "model_name": "lc_sa_tcnformer",
                "model_kwargs": model_kwargs,
            },
            path,
        )
        loaded, info = load_teacher_model(path, torch.device("cpu"))
    if not getattr(loaded, "requires_full_fms"):
        raise AssertionError("Loaded start_only teacher should preserve requires_full_fms=True.")
    validate_teacher_compatibility(info, dict(model_kwargs))
    try:
        validate_teacher_compatibility(info, {**model_kwargs, "calibration_steps": 5})
    except ValueError as exc:
        if "calibration_steps" not in str(exc):
            raise
    else:
        raise AssertionError("Teacher compatibility guard should reject calibration_steps mismatch.")


def test_calib_init_stream_time_multiscale_no_future_leakage() -> None:
    torch.manual_seed(912)
    model = build_model(
        "calib_init_state_forecaster",
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=5,
        sampling_interval=0.5,
        fms_context_mode="calibration_history",
        anchor_mode="none",
        anchor_interval_seconds=0.0,
        multi_horizon=True,
        horizon_set=[5.0, 10.0, 15.0],
        per_horizon_heads=True,
        d_model=32,
        hidden_dim=32,
        transformer_heads=4,
        transformer_ff_dim=64,
        forecast_head_mode="self_delta",
        delta_scale=1.0,
        motion_feature_mode="norm_delta_energy",
        stream_time_features=True,
        stream_context_mode="gru_tcn_multiscale",
        calib_summary_features=True,
        state_feedback_mode="predicted_current",
    )
    if getattr(model, "requires_full_fms"):
        raise AssertionError("stream time/multiscale calib-init model must not require full FMS input")
    model.eval()
    head_a = torch.randn(1, 40, 6)
    head_b = head_a.clone()
    # First prediction is at current index 4, so changing motion after index 4
    # must not affect that prediction.
    head_b[:, 5:, :] += 1000.0
    fms_full = torch.rand(1, 40)
    fms_changed_after_calib = fms_full.clone()
    fms_changed_after_calib[:, 4:] = 1.0 - fms_changed_after_calib[:, 4:]
    lengths = torch.tensor([40])
    with torch.no_grad():
        out_a = model(head_a, fms_full[:, :4], lengths)
        out_b = model(head_b, fms_full[:, :4], lengths)
        out_c = model(head_a, fms_changed_after_calib, lengths)
    if out_a["future"].shape[-1] != 3:
        raise AssertionError(f"Expected three horizon outputs, got {out_a['future'].shape}")
    if not torch.allclose(out_a["future"][:, 0], out_b["future"][:, 0], atol=1e-6):
        raise AssertionError("stream time/TCN/multiscale branch leaked motion after the current index")
    if not torch.allclose(out_a["future"], out_c["future"], atol=1e-6):
        raise AssertionError("stream time/TCN/multiscale branch leaked post-calibration FMS values")


def test_new_long_target_recent_motion_no_future_leakage() -> None:
    torch.manual_seed(33)
    model = build_model(
        "anchor_delta_mlp",
        head_dim=6,
        calibration_steps=4,
        horizon_steps=2,
        recent_steps=6,
        sampling_interval=0.5,
        horizon_seconds=1.0,
        anchor_mode="calibration_end",
        predict_delta_from_anchor=True,
        use_static=False,
        hidden_dim=32,
    )
    model.eval()
    head_a = torch.randn(1, 18, 6)
    head_b = head_a.clone()
    # First prediction is at t=5, so changing motion after t=5 must not affect it.
    head_b[:, 6:, :] += 1000.0
    fms = torch.rand(1, 18)
    with torch.no_grad():
        pred_a = model(head_a, fms[:, :4], torch.tensor([18]))["future"][:, 0]
        pred_b = model(head_b, fms[:, :4], torch.tensor([18]))["future"][:, 0]
    if not torch.allclose(pred_a, pred_b, atol=1e-6):
        raise AssertionError("New AnchorDeltaMLP recent window leaked motion after current time")


def test_lc_sa_full_search_dry_run_command_generation() -> None:
    runs = build_stage1(max_runs=1)
    args = argparse.Namespace(
        data_dir="./DenseFMS/Dataset",
        config="configs/lc_sa_tcnformer.yaml",
        output_dir="./runs/lc_sa_tcnformer_full_search",
        split_file="./artifacts/densefms_split_seed42.json",
        seed=42,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_workers=0,
        skip_existing=True,
        device=None,
    )
    cmd = train_cmd(args, runs[0], epochs=1, patience=1, all_runs=runs)
    required = {"--model", "lc_sa_tcnformer", "--no_test_eval", "--anchor_mode", "--fms_context_mode"}
    if not required.issubset(set(cmd)):
        raise AssertionError(f"LC-SA full-search command missing required args: {cmd}")
    if cmd[cmd.index("--fms_context_mode") + 1] != "start_only":
        raise AssertionError(f"LC-SA full-search main command should default to start_only context: {cmd}")


def test_long_target_search_dry_run_command_generation() -> None:
    runs = build_long_stage2(reduced=True)
    args = argparse.Namespace(
        data_dir="./DenseFMS/Dataset",
        config="configs/lc_sa_tcnformer.yaml",
        output_dir="./runs/densefms_long_target_search",
        split_file="./artifacts/densefms_split_seed42.json",
        seed=42,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_workers=0,
        skip_existing=True,
        device=None,
    )
    item = next(run for run in runs if run["model"] == "anchor_delta_mlp")
    cmd = long_target_train_cmd(args, item, epochs=1, patience=1, all_runs=runs)
    required = {"--model", "anchor_delta_mlp", "--no_test_eval", "--anchor_mode", "--fms_context_mode", "--hidden_dim"}
    if not required.issubset(set(cmd)):
        raise AssertionError(f"Long target command missing required args: {cmd}")
    if cmd[cmd.index("--fms_context_mode") + 1] != "start_only":
        raise AssertionError(f"Long target main command should default to start_only context: {cmd}")
    if cmd[cmd.index("--anchor_mode") + 1] != "none":
        raise AssertionError(f"Long target main command should not use sparse anchors: {cmd}")


def test_online_current_integrated_improvement_dry_run_command_generation() -> None:
    args = argparse.Namespace(
        python=sys.executable,
        data_dir="DenseFMS/Dataset",
        base_config="configs/online_current/selected_fds_static4.yaml",
        runs_dir="runs/online_fms_current_tracking_0509_integrated",
        split_file=(
            "runs/online_fms_current_tracking_0508/"
            "deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json"
        ),
    )
    commands = [integrated_improvement_command(args, candidate) for candidate in PLAN_CANDIDATES]
    if len(commands) < 8:
        raise AssertionError(f"Integrated improvement plan generated too few commands: {len(commands)}")
    for cmd in commands:
        required = {"--config", "--model", "--run_name", "--runs_dir", "--split_file", "--no_test_eval", "--skip_existing"}
        missing = required - set(cmd)
        if missing:
            raise AssertionError(f"Integrated improvement command missing {missing}: {cmd}")
        if "online_fms_risk_tracker" not in cmd:
            raise AssertionError(f"Integrated improvement command must use online risk/current model: {cmd}")
    flat = [" ".join(cmd) for cmd in commands]
    if not any("--future_aux_horizon_seconds 5.0 10.0 15.0" in item for item in flat):
        raise AssertionError(f"Future auxiliary candidate command missing: {flat}")
    if not any("--motion_feature_mode causal_dynamics_v1" in item for item in flat):
        raise AssertionError(f"Causal dynamics feature-bank candidate command missing: {flat}")


def test_zero_anchor_ablation_dry_run_command_generation() -> None:
    args = argparse.Namespace(
        python="python3",
        data_dir="DenseFMS/Dataset",
        config=ZERO_ANCHOR_ABLATION_BASE_CONFIG,
        split_file=ZERO_ANCHOR_ABLATION_BASE_SPLIT,
        runs_dir=ZERO_ANCHOR_ABLATION_RUNS_DIR,
        device="cpu",
    )
    specs = zero_anchor_ablation_build_specs(reuse_existing_120=True)
    commands = []
    for spec in specs:
        commands.extend(zero_anchor_ablation_build_commands(args, spec))
    if len(specs) != 9 or len(commands) != 13:
        raise AssertionError(f"Zero-anchor ablation should generate 9 specs and 13 commands, got {len(specs)}, {len(commands)}")
    flat = [" ".join(str(part) for part in item["command"]) for item in commands]
    if not any("--calibration_seconds 0" in item and "--fms_context_mode none" in item for item in flat):
        raise AssertionError(f"0s calibration baseline must disable FMS context: {flat}")
    if not any("--head_channel_mode linear_only" in item for item in flat):
        raise AssertionError(f"linear_only ablation command missing: {flat}")
    if not any("--head_channel_mode angular_only" in item for item in flat):
        raise AssertionError(f"angular_only ablation command missing: {flat}")
    if not any("--no_static" in item for item in flat):
        raise AssertionError(f"no_static ablation command missing: {flat}")
    finetunes = [item for item in commands if item["stage"] == "zero_anchor_finetune"]
    if len(finetunes) != 6:
        raise AssertionError(f"Expected six non-reused zero-anchor fine-tune commands, got {len(finetunes)}")
    for item in finetunes:
        cmd = item["command"]
        required = {"--init_checkpoint", "--current_head_mode", "zero_anchor_mixture", "--no_test_eval", "--skip_existing"}
        if not required.issubset(set(cmd)):
            raise AssertionError(f"Zero-anchor fine-tune command missing required args: {cmd}")


def test_valid_prediction_mask_count() -> None:
    mask = valid_prediction_mask(torch.tensor([20]), total_len=20, calibration_steps=4, horizon_steps=3)
    if int(mask.sum()) != 13:
        raise AssertionError(f"Expected 13 valid prediction positions, got {int(mask.sum())}")


def test_max_session_points_caps_target_extent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "PA999_Base.csv"
        rows = []
        for i in range(500):
            rows.append(
                ",".join(
                    [
                        f"{i * 0.5:.1f}",
                        str(float(i % 21)),
                        "0.1",
                        "0.2",
                        "0.3",
                        "0.4",
                        "0.5",
                        "0.6",
                        "m",
                        "12",
                        "25",
                    ]
                )
            )
        path.write_text("\n".join(rows), encoding="utf-8")

        sessions, _mapping, info = load_raw_sessions(
            tmp,
            calibration_seconds=90.0,
            horizon_seconds=15.0,
            max_session_points=420,
        )
        if len(sessions) != 1:
            raise AssertionError(f"Expected one session, got {len(sessions)}")
        sess = sessions[0]
        if sess.length != 420:
            raise AssertionError(f"Expected loaded length 420, got {sess.length}")
        if sess.original_length != 500:
            raise AssertionError(f"Expected original length 500, got {sess.original_length}")
        if info["truncated_session_count"] != 1:
            raise AssertionError(f"Expected one truncated session, got {info['truncated_session_count']}")
        horizon_steps = int(info["horizon_steps"])
        max_current_index = sess.length - horizon_steps - 1
        max_target_index = max_current_index + horizon_steps
        if max_target_index != 419:
            raise AssertionError(f"Expected final target index 419, got {max_target_index}")


def test_current_sequence_times_and_common_window_spec() -> None:
    times = torch.arange(20, dtype=torch.float32).unsqueeze(0) * 0.5
    current, mask = current_sequence_times(times, torch.tensor([20]), calibration_steps=4, horizon_steps=5)
    if not torch.isclose(current[0, 0], torch.tensor(2.0)):
        raise AssertionError("First current time should be calibration_steps * sampling_interval")
    if int(mask.sum()) != 11:
        raise AssertionError("Unexpected current time mask count")
    calib_spec = common_window_spec([30.0, 60.0, 90.0], [5.0])
    if calib_spec["target_start"] != 95.0:
        raise AssertionError(f"Calibration sweep target start wrong: {calib_spec}")
    horizon_spec = common_window_spec([30.0], [1.0, 2.5, 5.0, 10.0, 15.0])
    if horizon_spec["current_start"] != 30.0 or horizon_spec["max_horizon_seconds"] != 15.0:
        raise AssertionError(f"Horizon sweep common window wrong: {horizon_spec}")
    grid_spec = common_window_spec([30.0, 60.0, 90.0], [2.5, 5.0, 10.0])
    if grid_spec["current_start"] != 90.0 or grid_spec["max_horizon_seconds"] != 10.0 or grid_spec["target_start"] is not None:
        raise AssertionError(f"Grid common window wrong: {grid_spec}")


def test_time_config_backward_compatibility() -> None:
    cfg = {"data": {"default_sampling_interval": 0.5, "recent_seconds": 10}}
    normalize_time_config(cfg)
    if cfg["data"]["sampling_interval"] != 0.5 or cfg["data"]["recent_window_seconds"] != 10.0:
        raise AssertionError(f"Legacy time config aliases failed: {cfg}")
    if cfg["data"]["calibration_seconds"] != 30.0 or cfg["data"]["horizon_seconds"] != 5.0:
        raise AssertionError("Default calibration/horizon seconds changed")
    if cfg["data"]["max_session_points"] != 420:
        raise AssertionError(f"Default max_session_points must be 420: {cfg}")
    cfg_cap = {"data": {"max_session_points": 999}}
    normalize_time_config(cfg_cap)
    if cfg_cap["data"]["max_session_points"] != 420:
        raise AssertionError(f"max_session_points must be capped at 420: {cfg_cap}")


def test_config_2x2() -> None:
    files = {
        "configs/coff_lstm_no_static_level.yaml": (False, "level_only"),
        "configs/coff_lstm_no_static_trend.yaml": (False, "level_trend_raw"),
        "configs/coff_lstm_static_level.yaml": (True, "level_only"),
        "configs/coff_lstm_static_trend.yaml": (True, "level_trend_raw"),
    }
    comparable = []
    for path, (use_static, loss_mode) in files.items():
        cfg = load_config(path)
        if bool(cfg["model"]["use_static"]) != use_static or bool(cfg["data"]["use_static"]) != use_static:
            raise AssertionError(f"use_static mismatch in {path}")
        if cfg["loss"]["mode"] != loss_mode:
            raise AssertionError(f"loss mode mismatch in {path}")
        comparable.append((cfg["data"]["calibration_seconds"], cfg["data"]["horizon_seconds"], cfg["training"]["seed"], cfg["training"]["batch_size"]))
    if len(set(comparable)) != 1:
        raise AssertionError("2x2 configs differ in shared preprocessing/training settings")


def test_config_static_full() -> None:
    cfg = load_config("configs/coff_lstm_static_full_level.yaml")
    if cfg["data"]["static_features"] != ["age", "gender", "mssq"]:
        raise AssertionError("Full static config must use age/gender/mssq")
    if int(cfg["model"]["static_dim"]) != 5 or static_feature_dim(cfg["data"]["static_features"]) != 5:
        raise AssertionError("Full static config must resolve to static_dim=5")
    if cfg["loss"]["mode"] != "level_only":
        raise AssertionError("Full static level config should use level_only")
    if cfg["model"].get("recent_encoder") != "tcn":
        raise AssertionError("Full static config should default to TCN recent encoder")


def test_online_current_risk_target_shift_and_rise_labels() -> None:
    fms_raw = torch.tensor([[0.0, 1.0, 2.0, 3.0, 2.0, 6.0, 6.0]]) / 20.0
    lengths = torch.tensor([7])
    targets = compute_online_current_risk_targets(
        fms_raw,
        lengths,
        prediction_start=2,
        pred_steps=3,
        rise_horizon_steps=[2],
        rise_thresholds_normalized=[2.0 / 20.0],
        ordinal_bins_normalized=[v / 20.0 for v in [0, 2, 4, 6, 8, 10, 12, 15, 20]],
        fall_horizon_steps=[2],
        fall_thresholds_normalized=[1.0 / 20.0],
        future_horizon_steps=[1, 2],
        event_delta_threshold_normalized=1.0 / 20.0,
    )
    expected_current = torch.tensor([[2.0, 3.0, 2.0]]) / 20.0
    if not torch.allclose(targets["current"], expected_current):
        raise AssertionError(f"Online current targets use wrong current index: {targets['current']}")
    expected_rise = torch.tensor([[[0.0], [1.0], [1.0]]])
    if not torch.equal(targets["rise_labels"], expected_rise):
        raise AssertionError(f"Rapid-rise labels are misaligned: {targets['rise_labels']}")
    if not targets["rise_mask"].all():
        raise AssertionError("Rapid-rise labels unexpectedly invalid for fully available windows")
    expected_fall = torch.tensor([[[0.0], [1.0], [0.0]]])
    if not torch.equal(targets["fall_labels"], expected_fall):
        raise AssertionError(f"Rapid-drop labels are misaligned: {targets['fall_labels']}")
    if not targets["fall_mask"].all():
        raise AssertionError("Rapid-drop labels unexpectedly invalid for fully available windows")
    expected_future = torch.tensor([[[3.0, 2.0], [2.0, 6.0], [6.0, 6.0]]]) / 20.0
    if not torch.allclose(targets["future"], expected_future):
        raise AssertionError(f"Future auxiliary targets are misaligned: {targets['future']}")
    expected_delta = torch.tensor([[[1.0, 0.0], [-1.0, 3.0], [4.0, 4.0]]]) / 20.0
    if not torch.allclose(targets["future_delta"], expected_delta):
        raise AssertionError(f"Future-delta auxiliary targets are misaligned: {targets['future_delta']}")
    expected_events = torch.tensor([[[2, 1], [0, 2], [2, 2]]], dtype=torch.long)
    if not torch.equal(targets["event_labels"], expected_events):
        raise AssertionError(f"Event auxiliary labels are misaligned: {targets['event_labels']}")
    if not targets["future_mask"].all():
        raise AssertionError("Future auxiliary labels unexpectedly invalid for fully available windows")


def test_online_current_high_risk_onset_labels_exclude_plateau() -> None:
    fms_raw = torch.tensor([[10.0, 10.0, 10.0, 11.0, 12.0, 13.0, 13.0, 13.0, 10.0, 10.0]]) / 20.0
    lengths = torch.tensor([10])
    targets = compute_online_current_risk_targets(
        fms_raw,
        lengths,
        prediction_start=2,
        pred_steps=5,
        rise_horizon_steps=[],
        rise_thresholds_normalized=[],
        ordinal_bins_normalized=[v / 20.0 for v in [0, 2, 4, 6, 8, 10, 12, 15, 20]],
        high_risk_horizon_steps=[3],
        high_risk_thresholds_normalized=[12.0 / 20.0],
        high_risk_label_mode="onset",
        high_risk_onset_past_steps=2,
    )
    labels = targets["high_risk_labels"][0, :, 0, 0]
    mask = targets["high_risk_mask"][0, :, 0, 0]
    expected_labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    expected_mask = torch.tensor([True, True, False, False, False])
    if not torch.equal(labels, expected_labels):
        raise AssertionError(f"Onset high-risk labels are wrong: {labels}")
    if not torch.equal(mask, expected_mask):
        raise AssertionError(f"Onset high-risk mask should exclude high plateau: {mask}")


def test_online_current_high_risk_current_or_future_labels_include_plateau() -> None:
    fms_raw = torch.tensor([[10.0, 10.0, 12.0, 13.0, 11.0, 10.0, 9.0, 13.0]]) / 20.0
    lengths = torch.tensor([8])
    targets = compute_online_current_risk_targets(
        fms_raw,
        lengths,
        prediction_start=1,
        pred_steps=4,
        rise_horizon_steps=[],
        rise_thresholds_normalized=[],
        ordinal_bins_normalized=[v / 20.0 for v in [0, 2, 4, 6, 8, 10, 12, 15, 20]],
        high_risk_horizon_steps=[2],
        high_risk_thresholds_normalized=[12.0 / 20.0],
        high_risk_label_mode="current_or_future",
    )
    labels = targets["high_risk_labels"][0, :, 0, 0]
    mask = targets["high_risk_mask"][0, :, 0, 0]
    expected_labels = torch.tensor([1.0, 1.0, 1.0, 0.0])
    expected_mask = torch.tensor([True, True, True, True])
    if not torch.equal(labels, expected_labels):
        raise AssertionError(f"Current-or-future high-risk labels are wrong: {labels}")
    if not torch.equal(mask, expected_mask):
        raise AssertionError(f"Current-or-future high-risk mask is wrong: {mask}")


def test_online_current_risk_model_forward_shape_and_calibration_only_fms() -> None:
    torch.manual_seed(7)
    model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=4,
        rise_horizon_steps=[2, 4],
        rise_thresholds=[2.0, 3.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_time_features=True,
        stream_context_mode="gru",
        state_feedback_mode="predicted_current",
        motion_stats_branch=True,
        motion_encoder_context="tcn",
        motion_encoder_layers=2,
        current_head_mode="dual_delta_gate",
        fall_risk_head_enabled=True,
    )
    model.eval()
    head = torch.randn(2, 12, 6)
    fms_full = torch.rand(2, 12)
    lengths = torch.tensor([12, 10])
    with torch.no_grad():
        out_a = model(head, fms_full[:, :4], lengths)
        fms_changed = fms_full.clone()
        fms_changed[:, 4:] = 1.0 - fms_changed[:, 4:]
        out_b = model(head, fms_changed, lengths)
    if out_a["current"].shape != (2, 8):
        raise AssertionError(f"Unexpected online current shape: {out_a['current'].shape}")
    if out_a["risk_probs"].shape != (2, 8, 2):
        raise AssertionError(f"Unexpected online risk shape: {out_a['risk_probs'].shape}")
    if out_a["fall_risk_probs"].shape != (2, 8, 2):
        raise AssertionError(f"Unexpected online fall-risk shape: {out_a['fall_risk_probs'].shape}")
    if out_a["ordinal_probs"].shape != (2, 8, 9):
        raise AssertionError(f"Unexpected online ordinal shape: {out_a['ordinal_probs'].shape}")
    for key in ("current_level", "current_delta_value", "current_level_delta_gate", "session_drift_prior"):
        if key not in out_a:
            raise AssertionError(f"dual_delta_gate online tracker did not return {key}")
    if out_a["session_drift_prior"].shape != (2,):
        raise AssertionError(f"Unexpected session drift shape: {out_a['session_drift_prior'].shape}")
    if out_a["motion_encoder_context"] != "tcn" or int(out_a["motion_encoder_layers"].item()) != 2:
        raise AssertionError("Online tracker did not report the configured motion encoder stem")
    if not torch.allclose(out_a["current"], out_b["current"], atol=1e-6):
        raise AssertionError("Online tracker output changed when only post-calibration FMS input was modified")
    if getattr(model, "requires_full_fms", True):
        raise AssertionError("Online tracker must not require full post-calibration FMS input")


def test_online_current_risk_can_disable_risk_head() -> None:
    torch.manual_seed(71)
    model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=4,
        rise_horizon_steps=[2, 4],
        rise_thresholds=[2.0, 3.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_time_features=True,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2, 4],
        state_feedback_mode="none",
        motion_feature_mode="norm_delta_energy",
        current_head_mode="basic",
        ordinal_head_mode="cumulative",
        risk_head_enabled=False,
        risk_temporal_context="tcn",
        risk_temporal_layers=2,
        decoder_hidden_dim=48,
    )
    if getattr(model, "risk_head", None) is not None:
        raise AssertionError("risk_head_enabled=False should not instantiate a trainable risk head.")
    if getattr(model, "risk_temporal_blocks", None) is not None:
        raise AssertionError("risk_head_enabled=False should not instantiate risk temporal blocks.")
    if getattr(model, "fall_risk_head", None) is not None:
        raise AssertionError("fall_risk_head_enabled=False should not instantiate a trainable fall-risk head.")
    model.eval()
    head = torch.randn(2, 12, 6)
    fms_calib = torch.rand(2, 4)
    lengths = torch.tensor([12, 10])
    with torch.no_grad():
        out = model(head, fms_calib, lengths)
    if bool(out["risk_head_enabled"].detach().cpu().item()):
        raise AssertionError("risk_head_enabled metadata should be false.")
    if out["risk_logits"].shape != (2, 8, 2) or out["risk_probs"].shape != (2, 8, 2):
        raise AssertionError(f"Disabled risk head should keep output shape placeholders: {out['risk_probs'].shape}")
    if out["fall_risk_logits"].shape != (2, 8, 2) or out["fall_risk_probs"].shape != (2, 8, 2):
        raise AssertionError(f"Disabled fall-risk head should keep output shape placeholders: {out['fall_risk_probs'].shape}")
    if not torch.equal(out["risk_logits"], torch.zeros_like(out["risk_logits"])):
        raise AssertionError("Disabled risk head logits should be zero placeholders.")
    if not torch.equal(out["risk_probs"], torch.zeros_like(out["risk_probs"])):
        raise AssertionError("Disabled risk head probabilities should be zero placeholders.")
    if not torch.equal(out["fall_risk_probs"], torch.zeros_like(out["fall_risk_probs"])):
        raise AssertionError("Disabled fall-risk head probabilities should be zero placeholders.")


def test_online_current_risk_deep_tcn_forward_shape_and_cap_config() -> None:
    cfg = load_config("configs/online_fms_current_tracker_deep_tcn_ordreg_420.yaml")
    normalize_time_config(cfg)
    if cfg["data"]["max_session_points"] != 420:
        raise AssertionError("DeepTCN config must keep max_session_points=420")
    if cfg["model"]["stream_context_mode"] != "deep_tcn" or cfg["model"]["state_feedback_mode"] != "none":
        raise AssertionError("DeepTCN online config must use causal DeepTCN stream without recurrent feedback")
    large_cfg = load_config("configs/online_fms_current_tracker_deep_tcn_ordreg_420_large.yaml")
    normalize_time_config(large_cfg)
    if large_cfg["data"]["max_session_points"] != 420 or int(large_cfg["model"].get("decoder_hidden_dim", 0)) <= 0:
        raise AssertionError("Large DeepTCN config must keep the 420 cap and enable decoder MLP heads")
    latent_cfg = load_config("configs/online_fms_current_tracker_deep_tcn_latent_gru_420_large_calib240.yaml")
    normalize_time_config(latent_cfg)
    if not bool(latent_cfg["model"].get("stream_prepend_calibration", False)):
        raise AssertionError("MAE latent-GRU config must prepend calibration history for DeepTCN stream prewarm")
    if latent_cfg["model"].get("deep_tcn_dilations") != [1, 2, 4, 8, 16]:
        raise AssertionError("MAE latent-GRU config must keep DeepTCN dilations capped at 16")
    paper_cfg = load_config("configs/online_fms_current_tracker_deep_tcn_latent_gru_420_large_calib240_paper_ordreg.yaml")
    normalize_time_config(paper_cfg)
    if paper_cfg["model"].get("current_head_mode") != "paper_ordreg" or paper_cfg["model"].get("ordinal_head_mode") != "cumulative":
        raise AssertionError("Paper-style DeepTCN latent-GRU config must use the direct regression + cumulative ordinal decoder")
    if paper_cfg["loss"].get("ordinal_loss_mode") != "cumulative_bce":
        raise AssertionError("Paper-style DeepTCN latent-GRU config must enable cumulative ordinal loss")
    cumulative_cfg = load_config("configs/online_fms_current_tracker_deep_tcn_latent_gru_420_large_calib240_direct_cumulative_bins0to20.yaml")
    normalize_time_config(cumulative_cfg)
    if cumulative_cfg["model"].get("ordinal_head_mode") != "cumulative" or cumulative_cfg["loss"].get("ordinal_loss_mode") != "cumulative_bce":
        raise AssertionError("0..20 direct cumulative config must use cumulative ordinal head and cumulative BCE")
    if cumulative_cfg["model"].get("ordinal_bins") != list(range(21)):
        raise AssertionError("0..20 direct cumulative config must keep every integer FMS class")
    model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=4,
        rise_horizon_steps=[2, 4],
        rise_thresholds=[2.0, 3.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_time_features=True,
        stream_context_mode="deep_tcn",
        deep_tcn_dilations=[1, 2, 4],
        state_feedback_mode="none",
        motion_feature_mode="norm_delta_energy",
        current_head_mode="dual_delta_gate",
        fms_combine_weight_ordinal=0.35,
        decoder_hidden_dim=48,
    )
    model.eval()
    head = torch.randn(2, 12, 6)
    fms_full = torch.rand(2, 12)
    lengths = torch.tensor([12, 10])
    with torch.no_grad():
        out = model(head, fms_full[:, :4], lengths)
    if out["current"].shape != (2, 8) or out["ordinal_probs"].shape != (2, 8, 9):
        raise AssertionError(f"Unexpected DeepTCN online output shapes: {out['current'].shape}, {out['ordinal_probs'].shape}")
    if not torch.isfinite(out["current"]).all():
        raise AssertionError("DeepTCN online current predictions contain non-finite values")
    if getattr(model, "recent_rf_steps") != 29:
        raise AssertionError(f"DeepTCN receptive field not recorded as expected: {getattr(model, 'recent_rf_steps')}")
    latent_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=4,
        rise_horizon_steps=[2, 4],
        rise_thresholds=[2.0, 3.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_time_features=True,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2, 4],
        state_feedback_mode="none",
        motion_feature_mode="norm_delta_energy",
        current_head_mode="dual_delta_gate",
        fms_combine_weight_ordinal=0.35,
        decoder_hidden_dim=48,
    )
    latent_model.eval()
    with torch.no_grad():
        latent_out = latent_model(head, fms_full[:, :4], lengths)
    if latent_out["current"].shape != (2, 8) or latent_out["ordinal_probs"].shape != (2, 8, 9):
        raise AssertionError(
            f"Unexpected latent-GRU DeepTCN output shapes: {latent_out['current'].shape}, {latent_out['ordinal_probs'].shape}"
        )
    if not torch.isfinite(latent_out["current"]).all():
        raise AssertionError("Latent-GRU DeepTCN current predictions contain non-finite values")
    if getattr(latent_model, "recent_rf_steps") != 29:
        raise AssertionError(f"Latent-GRU DeepTCN receptive field not recorded as expected: {getattr(latent_model, 'recent_rf_steps')}")
    if latent_out["stream_context_mode"] != "deep_tcn_latent_gru":
        raise AssertionError("Latent-GRU DeepTCN mode metadata was not preserved")
    paper_decoder_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=4,
        rise_horizon_steps=[2, 4],
        rise_thresholds=[2.0, 3.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_time_features=True,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2, 4],
        state_feedback_mode="none",
        motion_feature_mode="norm_delta_energy",
        current_head_mode="paper_ordreg",
        ordinal_head_mode="cumulative",
        fms_combine_weight_ordinal=0.6,
        decoder_hidden_dim=48,
    )
    paper_decoder_model.eval()
    with torch.no_grad():
        paper_out = paper_decoder_model(head, fms_full[:, :4], lengths)
    if paper_out["current"].shape != (2, 8) or paper_out["ordinal_binary_logits"].shape != (2, 8, 8):
        raise AssertionError(
            "Paper-style ordinal/regression decoder produced unexpected shapes: "
            f"{paper_out['current'].shape}, {paper_out['ordinal_binary_logits'].shape}"
        )
    if paper_out["current_head_mode"] != "paper_ordreg" or paper_out["ordinal_head_mode"] != "cumulative":
        raise AssertionError("Paper-style decoder metadata was not preserved")
    if any(key in paper_out for key in ("current_delta_value", "current_level_delta_gate", "session_drift_prior")):
        raise AssertionError("Paper-style decoder must not expose anchor/delta gate heads")
    expected_paper_current = 0.6 * paper_out["current_ordinal"] + 0.4 * paper_out["current_reg"]
    if not torch.allclose(paper_out["current"], expected_paper_current.clamp(0.0, 1.0), atol=1e-6):
        raise AssertionError("Paper-style decoder did not combine ordinal and direct regression predictions as configured")
    if not torch.all(torch.diff(paper_out["ordinal_thresholds"]) > 0):
        raise AssertionError("Cumulative ordinal thresholds must remain ordered")
    direct_cumulative_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=4,
        rise_horizon_steps=[2, 4],
        rise_thresholds=[2.0, 3.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_time_features=True,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2, 4],
        state_feedback_mode="none",
        motion_feature_mode="norm_delta_energy",
        current_head_mode="basic",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        fms_combine_weight_ordinal=0.2,
        decoder_hidden_dim=48,
    )
    direct_cumulative_model.eval()
    with torch.no_grad():
        direct_cumulative_out = direct_cumulative_model(head, fms_full[:, :4], lengths)
    if direct_cumulative_out["ordinal_binary_logits"].shape != (2, 8, 20):
        raise AssertionError(
            f"0..20 cumulative ordinal head should expose 20 ordered threshold logits, got {direct_cumulative_out['ordinal_binary_logits'].shape}"
        )
    if direct_cumulative_out["ordinal_probs"].shape != (2, 8, 21):
        raise AssertionError(f"0..20 cumulative ordinal probabilities have wrong shape: {direct_cumulative_out['ordinal_probs'].shape}")
    state_decoder_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=4,
        rise_horizon_steps=[2, 4],
        rise_thresholds=[2.0, 3.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_time_features=True,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2, 4],
        state_feedback_mode="none",
        motion_feature_mode="norm_delta_energy",
        current_head_mode="basic",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        fms_combine_weight_ordinal=0.2,
        decoder_context_mode="state",
        decoder_hidden_dim=48,
    )
    state_decoder_model.eval()
    with torch.no_grad():
        state_decoder_out = state_decoder_model(head, fms_full[:, :4], lengths)
    if state_decoder_out["current"].shape != (2, 8) or state_decoder_out["ordinal_binary_logits"].shape != (2, 8, 20):
        raise AssertionError("State-only decoder output shapes are wrong")
    if state_decoder_out["decoder_context_mode"] != "state":
        raise AssertionError("State-only decoder metadata was not preserved")
    if int(state_decoder_out["decoder_feature_dim"].item()) != 40:
        raise AssertionError(f"State-only decoder should expose hidden_dim=40, got {state_decoder_out['decoder_feature_dim']}")
    if not isinstance(state_decoder_model.fusion, torch.nn.Identity):
        raise AssertionError("State-only decoder should bypass the fusion MLP")
    if state_decoder_model.static_encoder is not None:
        raise AssertionError("Static encoder should stay unused when static is disabled")
    state_static_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=4,
        rise_horizon_steps=[2, 4],
        rise_thresholds=[2.0, 3.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_time_features=True,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2, 4],
        state_feedback_mode="none",
        motion_feature_mode="norm_delta_energy",
        current_head_mode="basic",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        fms_combine_weight_ordinal=0.2,
        use_static=True,
        static_dim=4,
        decoder_context_mode="state",
        decoder_hidden_dim=48,
    )
    state_static_model.eval()
    static_4d = torch.randn(2, 4)
    with torch.no_grad():
        state_static_out = state_static_model(head, fms_full[:, :4], lengths, static=static_4d)
    if int(state_static_out["decoder_feature_dim"].item()) != 44:
        raise AssertionError(f"State+static decoder should expose hidden_dim+static_dim=44, got {state_static_out['decoder_feature_dim']}")
    if state_static_model.static_encoder is not None or not isinstance(state_static_model.fusion, torch.nn.Identity):
        raise AssertionError("State+static decoder should concatenate raw static features without encoded static fusion")
    if not bool(state_static_out["use_static"].item()):
        raise AssertionError("State+static decoder did not preserve use_static metadata")
    decoder_tcn_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=4,
        rise_horizon_steps=[2, 4],
        rise_thresholds=[2.0, 3.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_time_features=True,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2, 4],
        state_feedback_mode="none",
        motion_feature_mode="norm_delta_energy",
        current_head_mode="basic",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        fms_combine_weight_ordinal=0.2,
        use_static=True,
        static_dim=4,
        decoder_context_mode="state",
        decoder_hidden_dim=48,
        decoder_temporal_context="tcn",
        decoder_temporal_layers=2,
    )
    decoder_tcn_model.eval()
    with torch.no_grad():
        decoder_tcn_out = decoder_tcn_model(head, fms_full[:, :4], lengths, static=static_4d)
    if decoder_tcn_out["current"].shape != (2, 8) or decoder_tcn_out["risk_probs"].shape != (2, 8, 2):
        raise AssertionError("Decoder TCN produced wrong online current/risk shapes")
    if decoder_tcn_out["decoder_temporal_context"] != "tcn" or int(decoder_tcn_out["decoder_temporal_layers"].item()) != 2:
        raise AssertionError("Decoder TCN metadata was not preserved")
    head_future_changed = head.clone()
    head_future_changed[:, 9:] += 100.0
    with torch.no_grad():
        causal_a = decoder_tcn_model(head, fms_full[:, :4], lengths, static=static_4d)
        causal_b = decoder_tcn_model(head_future_changed, fms_full[:, :4], lengths, static=static_4d)
    if not torch.allclose(causal_a["current"][:, :5], causal_b["current"][:, :5], atol=1e-6):
        raise AssertionError("Decoder TCN leaked future motion into earlier current outputs")
    if not torch.allclose(causal_a["risk_probs"][:, :5], causal_b["risk_probs"][:, :5], atol=1e-6):
        raise AssertionError("Decoder TCN leaked future motion into earlier risk outputs")
    calib_film_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=4,
        rise_horizon_steps=[2, 4],
        rise_thresholds=[2.0, 3.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_time_features=True,
        stream_context_mode="deep_tcn_latent_gru",
        stream_prepend_calibration=True,
        stream_calib_condition_mode="film",
        stream_calib_condition_strength=0.2,
        deep_tcn_dilations=[1, 2, 4],
        state_feedback_mode="none",
        motion_feature_mode="norm_delta_energy",
        current_head_mode="basic",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        fms_combine_weight_ordinal=0.2,
        use_static=True,
        static_dim=4,
        decoder_context_mode="state",
        decoder_hidden_dim=48,
    )
    calib_film_model.eval()
    with torch.no_grad():
        film_out = calib_film_model(head, fms_full[:, :4], lengths, static=static_4d)
    if film_out["current"].shape != (2, 8) or not torch.isfinite(film_out["current"]).all():
        raise AssertionError("Calibration-FiLM stream conditioning produced invalid current predictions")
    if film_out["stream_calib_condition_mode"] != "film" or abs(float(film_out["stream_calib_condition_strength"].item()) - 0.2) > 1e-7:
        raise AssertionError("Calibration-FiLM stream conditioning metadata was not preserved")
    with torch.no_grad():
        film_a = calib_film_model(head, fms_full[:, :4], lengths, static=static_4d)
        film_b = calib_film_model(head_future_changed, fms_full[:, :4], lengths, static=static_4d)
    if not torch.allclose(film_a["current"][:, :5], film_b["current"][:, :5], atol=1e-6):
        raise AssertionError("Calibration-FiLM stream conditioning leaked future motion into earlier current outputs")
    fds_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=4,
        rise_horizon_steps=[2, 4],
        rise_thresholds=[2.0, 3.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_time_features=True,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2, 4],
        state_feedback_mode="none",
        motion_feature_mode="norm_delta_energy",
        current_head_mode="basic",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        fms_combine_weight_ordinal=0.2,
        use_static=True,
        static_dim=4,
        decoder_context_mode="state",
        decoder_hidden_dim=48,
        fds_enabled=True,
        fds_min=0.0,
        fds_max=20.0,
        fds_bin_size=1.0,
        fds_kernel_size=5,
        fds_sigma=2.0,
        fds_start_update=1,
        fds_start_smooth=2,
    )
    fds_labels_raw = fms_full[:, 4:12] * 20.0
    fds_mask = torch.arange(8).view(1, -1) < (lengths - 4).view(-1, 1)
    fds_model.train()
    fds_out_epoch1 = fds_model(
        head,
        fms_full[:, :4],
        lengths,
        static=static_4d,
        fds_labels_raw=fds_labels_raw,
        fds_mask=fds_mask,
        fds_update=True,
        fds_apply=True,
    )
    if int(fds_out_epoch1["fds_updated_points"].item()) != int(fds_mask.sum().item()):
        raise AssertionError("FDS did not collect the expected number of training feature points")
    if int(fds_out_epoch1["fds_applied_points"].item()) != 0:
        raise AssertionError("FDS should not apply feature smoothing before running stats are initialized")
    fds_summary = fds_model.commit_fds_epoch_stats()
    if not fds_summary["fds_initialized"] or fds_summary["fds_running_bins"] <= 0:
        raise AssertionError("FDS running statistics were not initialized after commit")
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "fds_state.pt"
        torch.save(fds_model.state_dict(), state_path)
        fds_reloaded = build_model(
            "online_fms_risk_tracker",
            head_dim=6,
            calibration_steps=4,
            recent_steps=3,
            horizon_steps=4,
            rise_horizon_steps=[2, 4],
            rise_thresholds=[2.0, 3.0],
            d_model=32,
            hidden_dim=40,
            transformer_heads=4,
            transformer_ff_dim=64,
            stream_time_features=True,
            stream_context_mode="deep_tcn_latent_gru",
            deep_tcn_dilations=[1, 2, 4],
            state_feedback_mode="none",
            motion_feature_mode="norm_delta_energy",
            current_head_mode="basic",
            ordinal_head_mode="cumulative",
            ordinal_bins=list(range(21)),
            fms_combine_weight_ordinal=0.2,
            use_static=True,
            static_dim=4,
            decoder_context_mode="state",
            decoder_hidden_dim=48,
            fds_enabled=True,
            fds_min=0.0,
            fds_max=20.0,
            fds_bin_size=1.0,
            fds_kernel_size=5,
            fds_sigma=2.0,
        )
        fds_reloaded.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False), strict=True)
        if not bool(fds_reloaded.fds_module.initialized.item()):
            raise AssertionError("FDS checkpoint round-trip lost initialized running statistics")
        if int((fds_reloaded.fds_module.running_count > 0).sum().item()) != int(fds_summary["fds_running_bins"]):
            raise AssertionError("FDS checkpoint round-trip changed the number of initialized bins")
    fds_model.reset_fds_epoch_stats()
    fds_out_epoch2 = fds_model(
        head,
        fms_full[:, :4],
        lengths,
        static=static_4d,
        fds_labels_raw=fds_labels_raw,
        fds_mask=fds_mask,
        fds_update=True,
        fds_apply=True,
    )
    if int(fds_out_epoch2["fds_applied_points"].item()) <= 0:
        raise AssertionError("FDS did not apply smoothing after running stats were initialized")
    if fds_out_epoch2["current"].shape != (2, 8) or not torch.isfinite(fds_out_epoch2["current"]).all():
        raise AssertionError("FDS-enabled state decoder produced invalid current predictions")
    fds_model.eval()
    with torch.no_grad():
        fds_eval_out = fds_model(
            head,
            fms_full[:, :4],
            lengths,
            static=static_4d,
            fds_labels_raw=fds_labels_raw,
            fds_mask=fds_mask,
            fds_update=True,
            fds_apply=True,
        )
    if int(fds_eval_out["fds_updated_points"].item()) != 0 or int(fds_eval_out["fds_applied_points"].item()) != 0:
        raise AssertionError("FDS must stay inactive during eval/inference even if labels are accidentally passed")
    prewarm_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=20,
        recent_steps=3,
        horizon_steps=2,
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_time_features=True,
        stream_context_mode="deep_tcn_latent_gru",
        stream_prepend_calibration=True,
        deep_tcn_dilations=[1, 2],
        state_feedback_mode="none",
        motion_feature_mode="norm_delta_energy",
        current_head_mode="dual_delta_gate",
        fms_combine_weight_ordinal=0.35,
        decoder_hidden_dim=48,
    )
    prewarm_model.eval()
    prewarm_head = torch.randn(2, 26, 6)
    prewarm_fms = torch.rand(2, 20)
    prewarm_lengths = torch.tensor([26, 24])
    with torch.no_grad():
        prewarm_out = prewarm_model(prewarm_head, prewarm_fms, prewarm_lengths)
    if getattr(prewarm_model, "recent_rf_steps") != 13:
        raise AssertionError(f"Prewarm DeepTCN receptive field should account for two convs per block, got {prewarm_model.recent_rf_steps}")
    if int(prewarm_out["stream_history_start"].item()) != 8 or int(prewarm_out["stream_update_start_offset"].item()) != 12:
        raise AssertionError("DeepTCN stream prewarm did not use the calibration tail needed to fill the causal receptive field")
    if not bool(prewarm_out["stream_prepend_calibration"].item()):
        raise AssertionError("DeepTCN stream prewarm metadata was not preserved")
    feedback_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=4,
        rise_horizon_steps=[2, 4],
        rise_thresholds=[2.0, 3.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_time_features=True,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2, 4],
        state_feedback_mode="predicted_current",
        motion_feature_mode="norm_delta_energy",
        current_head_mode="dual_delta_gate",
        fms_combine_weight_ordinal=0.35,
        decoder_hidden_dim=48,
    )
    feedback_model.eval()
    with torch.no_grad():
        feedback_out = feedback_model(head, fms_full[:, :4], lengths)
    if feedback_out["current"].shape != (2, 8) or feedback_out["ordinal_probs"].shape != (2, 8, 9):
        raise AssertionError(
            f"Unexpected feedback latent-GRU DeepTCN output shapes: {feedback_out['current'].shape}, {feedback_out['ordinal_probs'].shape}"
        )
    if not torch.isfinite(feedback_out["current"]).all():
        raise AssertionError("Feedback latent-GRU DeepTCN current predictions contain non-finite values")
    if feedback_out["state_feedback_mode"] != "predicted_current":
        raise AssertionError("Feedback latent-GRU DeepTCN feedback metadata was not preserved")
    dropout_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=4,
        rise_horizon_steps=[2, 4],
        rise_thresholds=[2.0, 3.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_time_features=True,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2, 4],
        state_feedback_mode="none",
        motion_feature_mode="norm_delta_energy",
        current_head_mode="dual_delta_gate",
        fms_combine_weight_ordinal=0.35,
        decoder_hidden_dim=48,
        calib_fms_dropout=1.0,
        calibration_end_fms_dropout=1.0,
    )
    dropout_model.train()
    fms_pattern = torch.tensor([[0.0, 0.2, 0.4, 1.0], [1.0, 0.8, 0.6, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        dropout_out = dropout_model(head, fms_pattern, lengths)
    expected_raw_anchor = fms_pattern[:, -1]
    expected_model_anchor = fms_pattern.mean(dim=1)
    if not torch.allclose(dropout_out["calibration_end_fms"], expected_raw_anchor):
        raise AssertionError("Anchor dropout must preserve raw calibration_end_fms metadata for loss weighting")
    if not torch.allclose(dropout_out["model_anchor_fms"], expected_model_anchor):
        raise AssertionError("Anchor dropout must replace the decoder anchor with calibration mean during training")


def test_online_current_risk_zero_calibration_motion_only_forward() -> None:
    torch.manual_seed(20)
    model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=0,
        horizon_steps=2,
        recent_steps=3,
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_context_mode="deep_tcn_latent_gru",
        stream_prepend_calibration=True,
        deep_tcn_dilations=[1, 2],
        fms_context_mode="none",
        state_feedback_mode="none",
        current_head_mode="zero_anchor_mixture",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        decoder_context_mode="state",
        decoder_hidden_dim=48,
    )
    out = model(torch.randn(2, 16, 6), torch.zeros(2, 0), torch.tensor([16, 15]))
    if out["current"].shape != (2, 14):
        raise AssertionError(f"zero-calibration online current shape failed: {out['current'].shape}")
    if int(out["stream_history_start"]) != 0:
        raise AssertionError("zero-calibration stream should start from the first motion step.")
    if out["current_anchor_value"].shape != out["current"].shape:
        raise AssertionError("zero-calibration zero-anchor head did not produce anchor values.")
    if not torch.isfinite(out["calibration_end_fms"]).all():
        raise AssertionError("zero-calibration learned anchor metadata must be finite.")
    if getattr(model, "calibration_tcn_rf_steps") != 0 or getattr(model, "calibration_deep_tcn_dilations") != []:
        raise AssertionError("zero-calibration should not instantiate an active calibration TCN receptive field.")


def test_online_current_risk_adaptive_calibration_tcn_dilation() -> None:
    torch.manual_seed(21)
    short_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=60,
        horizon_steps=2,
        recent_steps=3,
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        calibration_encoder_mode="deep_tcn",
        stream_context_mode="deep_tcn_latent_gru",
        stream_prepend_calibration=True,
        deep_tcn_dilations=[1, 2, 4, 8, 16],
        calibration_tcn_adaptive_dilations=True,
        current_head_mode="zero_anchor_mixture",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        decoder_context_mode="state",
        decoder_hidden_dim=48,
    )
    if getattr(short_model, "calibration_deep_tcn_dilations") != [1, 2, 4, 8]:
        raise AssertionError(
            "Short calibration should drop only the too-large final dilation: "
            f"{getattr(short_model, 'calibration_deep_tcn_dilations')}"
        )
    if getattr(short_model, "calibration_tcn_rf_steps") != 61 or getattr(short_model, "calibration_tcn_pad_steps") != 1:
        raise AssertionError("Short calibration RF should be 61 with one step of causal padding.")
    out = short_model(torch.randn(2, 96, 6), torch.rand(2, 60), torch.tensor([96, 95]))
    if out["current"].shape != (2, 36):
        raise AssertionError(f"adaptive calibration TCN forward shape failed: {out['current'].shape}")

    near_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=120,
        horizon_steps=2,
        recent_steps=3,
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        calibration_encoder_mode="deep_tcn",
        stream_context_mode="deep_tcn_latent_gru",
        stream_prepend_calibration=True,
        deep_tcn_dilations=[1, 2, 4, 8, 16],
        calibration_tcn_adaptive_dilations=True,
        current_head_mode="zero_anchor_mixture",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        decoder_context_mode="state",
        decoder_hidden_dim=48,
    )
    if getattr(near_model, "calibration_deep_tcn_dilations") != [1, 2, 4, 8, 16]:
        raise AssertionError("Near-full calibration should keep base dilations and rely on causal padding.")
    if getattr(near_model, "calibration_tcn_rf_steps") != 125 or getattr(near_model, "calibration_tcn_pad_steps") != 5:
        raise AssertionError("Near-full calibration RF should be 125 with five causal padding steps.")


def test_online_current_risk_recent_motion_no_future_leakage() -> None:
    torch.manual_seed(9)
    model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=2,
        rise_horizon_steps=[2],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_time_features=False,
        stream_context_mode="gru",
        state_feedback_mode="none",
        motion_stats_branch=True,
        motion_encoder_context="tcn",
        motion_encoder_layers=2,
        current_head_mode="dual_delta_gate",
        risk_temporal_context="tcn",
        risk_temporal_layers=2,
    )
    model.eval()
    head = torch.randn(1, 12, 6)
    fms = torch.rand(1, 4)
    lengths = torch.tensor([12])
    head_changed = head.clone()
    head_changed[:, 9:] += 100.0
    with torch.no_grad():
        out_a = model(head, fms, lengths)
        out_b = model(head_changed, fms, lengths)
    if not torch.allclose(out_a["current"][:, :5], out_b["current"][:, :5], atol=1e-6):
        raise AssertionError("Online tracker current outputs before a changed future motion step were affected")
    if not torch.allclose(out_a["risk_probs"][:, :5], out_b["risk_probs"][:, :5], atol=1e-6):
        raise AssertionError("Online tracker temporal risk head leaked future motion into earlier risk outputs")


def test_online_current_risk_prefix_streamer_matches_full_forward() -> None:
    torch.manual_seed(19)
    calibration_steps = 4
    total_steps = 12
    model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=calibration_steps,
        recent_steps=3,
        horizon_steps=2,
        rise_horizon_steps=[1, 2],
        rise_thresholds=[1.0, 2.0],
        d_model=32,
        hidden_dim=40,
        kernel_size=3,
        dropout=0.0,
        calib_dilations=[1, 2],
        calibration_encoder_mode="deep_tcn",
        stream_context_mode="deep_tcn_latent_gru",
        stream_prepend_calibration=True,
        stream_time_features=True,
        deep_tcn_dilations=[1, 2],
        state_feedback_mode="none",
        motion_feature_mode="causal_dynamics_v1",
        current_head_mode="basic",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        fms_combine_weight_ordinal=0.15,
        decoder_context_mode="state",
        decoder_hidden_dim=48,
        use_static=True,
        static_dim=4,
        risk_head_enabled=True,
        fall_risk_head_enabled=True,
    )
    model.eval()
    head = torch.randn(1, total_steps, 6)
    fms = torch.linspace(0.05, 0.95, total_steps).view(1, total_steps)
    static = torch.tensor([[0.1, -0.2, 1.0, 0.0]], dtype=torch.float32)
    lengths = torch.tensor([total_steps])
    fms_scaler = {"min": 0.0, "max": 20.0}
    with torch.no_grad():
        full_out = model(head, fms[:, :calibration_steps], lengths, static=static)
    streamer = OnlineCurrentRiskPrefixStreamer(
        model=model,
        fms_scaler=fms_scaler,
        calibration_steps=calibration_steps,
        sampling_interval=0.5,
        device=torch.device("cpu"),
        static_tensor=static,
        rise_horizon_steps=[1, 2],
        fall_horizon_steps=[1, 2],
    )
    rows = []
    for t in range(total_steps):
        row = streamer.push_normalized(
            head[0, t],
            calibration_fms_norm=float(fms[0, t]) if t < calibration_steps else None,
            target_fms_raw=float(fms[0, t] * 20.0),
            timestamp=float(t) * 0.5,
            row_index=t,
        )
        if row is None:
            continue
        rows.append(row)
        idx = t - calibration_steps
        expected = float(full_out["current"][0, idx].item() * 20.0)
        if abs(float(row["predicted_fms_now"]) - expected) > 1e-5:
            raise AssertionError(
                f"Prefix streamer prediction mismatch at t={t}: {row['predicted_fms_now']} vs {expected}"
            )
        if int(row["prediction_index"]) != t:
            raise AssertionError(f"Prefix streamer returned wrong prediction index: {row['prediction_index']} vs {t}")
        if row["post_calibration_fms_used_as_input"]:
            raise AssertionError("Prefix streamer must not use post-calibration FMS as model input.")
    if len(rows) != total_steps - calibration_steps:
        raise AssertionError(f"Expected {total_steps - calibration_steps} streaming rows, got {len(rows)}.")
    if not all(f"p_rapid_rise_{h * 0.5:g}s" in rows[-1] for h in [1, 2]):
        raise AssertionError(f"Prefix streamer did not emit rapid-rise probabilities: {rows[-1]}")


def test_online_current_risk_loss_smoke_and_anchor_policy() -> None:
    torch.manual_seed(11)
    model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=2,
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        state_feedback_mode="predicted_current",
    )
    head = torch.randn(2, 12, 6)
    fms = torch.rand(2, 12)
    lengths = torch.tensor([12, 11])
    out = model(head, fms[:, :4], lengths)
    loss, parts = compute_online_current_risk_loss(
        out,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        ordinal_bins=[0, 2, 4, 6, 8, 10, 12, 15, 20],
    )
    if not torch.isfinite(loss) or parts["valid_points"] <= 0 or parts["risk_points"] <= 0:
        raise AssertionError(f"Online current-risk loss smoke failed: loss={loss}, parts={parts}")
    paper_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=2,
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        state_feedback_mode="none",
        current_head_mode="paper_ordreg",
        ordinal_head_mode="cumulative",
    )
    paper_out = paper_model(head, fms[:, :4], lengths)
    paper_loss, paper_parts = compute_online_current_risk_loss(
        paper_out,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        ordinal_bins=[0, 2, 4, 6, 8, 10, 12, 15, 20],
        ordinal_loss_mode="cumulative_bce",
    )
    if not torch.isfinite(paper_loss) or paper_parts["ordinal_points"] <= 0:
        raise AssertionError(f"Paper-style cumulative ordinal loss smoke failed: loss={paper_loss}, parts={paper_parts}")
    if paper_parts["ordinal_loss_mode"] != "cumulative_bce":
        raise AssertionError(f"Paper-style cumulative ordinal loss mode was not reported: {paper_parts}")
    dummy_outputs = {
        "current": torch.zeros(1, 3),
        "mask": torch.ones(1, 3, dtype=torch.bool),
        "prediction_start": torch.tensor(1),
    }
    dummy_fms = torch.tensor([[0.0, 0.5, 0.5, 0.5, 0.0]])
    dummy_lengths = torch.tensor([5])
    smooth_loss, smooth_parts = compute_online_current_risk_loss(
        dummy_outputs,
        dummy_fms,
        dummy_lengths,
        {"min": 0.0, "max": 1.0},
        rise_horizon_steps=[1],
        rise_thresholds=[0.25],
        ordinal_bins=[0.0, 0.5, 1.0],
        current_reg_aux_weight=0.0,
        ordinal_loss_weight=0.0,
        risk_loss_weight=0.0,
        smoothness_weight=0.0,
        loss_type="smooth_l1",
    )
    mae_loss, mae_parts = compute_online_current_risk_loss(
        dummy_outputs,
        dummy_fms,
        dummy_lengths,
        {"min": 0.0, "max": 1.0},
        rise_horizon_steps=[1],
        rise_thresholds=[0.25],
        ordinal_bins=[0.0, 0.5, 1.0],
        current_reg_aux_weight=0.0,
        ordinal_loss_weight=0.0,
        risk_loss_weight=0.0,
        smoothness_weight=0.0,
        loss_type="mae",
    )
    if smooth_parts["valid_points"] != mae_parts["valid_points"]:
        raise AssertionError(f"Online current-risk loss_type comparison used mismatched masks: {smooth_parts}, {mae_parts}")
    if not mae_loss > smooth_loss:
        raise AssertionError(f"Online current-risk MAE loss did not differ from SmoothL1: {mae_loss} <= {smooth_loss}")
    anchor_outputs = {
        "current": torch.tensor([[1.0, 1.0, 1.0]]),
        "current_reg": torch.tensor([[1.0, 1.0, 1.0]]),
        "mask": torch.ones(1, 3, dtype=torch.bool),
        "prediction_start": torch.tensor(1),
        "calibration_end_fms": torch.tensor([1.0]),
    }
    anchor_dummy_fms = torch.tensor([[0.0, 0.9, 0.5, 0.0, 0.0]])
    plain_anchor_loss, plain_anchor_parts = compute_online_current_risk_loss(
        anchor_outputs,
        anchor_dummy_fms,
        dummy_lengths,
        {"min": 0.0, "max": 1.0},
        rise_horizon_steps=[1],
        rise_thresholds=[0.25],
        ordinal_bins=[0.0, 0.5, 1.0],
        current_reg_aux_weight=0.0,
        ordinal_loss_weight=0.0,
        risk_loss_weight=0.0,
        smoothness_weight=0.0,
        loss_type="mae",
    )
    weighted_anchor_loss, weighted_anchor_parts = compute_online_current_risk_loss(
        anchor_outputs,
        anchor_dummy_fms,
        dummy_lengths,
        {"min": 0.0, "max": 1.0},
        rise_horizon_steps=[1],
        rise_thresholds=[0.25],
        ordinal_bins=[0.0, 0.5, 1.0],
        current_reg_aux_weight=0.0,
        ordinal_loss_weight=0.0,
        risk_loss_weight=0.0,
        smoothness_weight=0.0,
        loss_type="mae",
        anchor_break_weight=2.0,
        anchor_break_threshold=0.25,
        anchor_break_max_weight=3.0,
    )
    if weighted_anchor_parts["anchor_break_points"] <= 0 or weighted_anchor_parts["anchor_break_mean_weight"] <= 1.0:
        raise AssertionError(f"Anchor-break weighting did not activate: {weighted_anchor_parts}")
    if not weighted_anchor_loss > plain_anchor_loss:
        raise AssertionError(f"Anchor-break weighted loss should exceed plain loss: {weighted_anchor_loss} <= {plain_anchor_loss}")
    if "anchor_index" in out:
        raise AssertionError("Online tracker should not expose a post-calibration observed-FMS anchor policy")


def test_online_current_risk_lds_weighting_train_targets_only() -> None:
    train_sessions = []
    for idx, raw_values in enumerate(
        [
            [0, 0, 5, 5, 5, 5, 5, 5],
            [0, 0, 5, 5, 5, 5, 18, 5],
        ]
    ):
        raw = np.asarray(raw_values, dtype=np.float32)
        train_sessions.append(
            DenseFMSSession(
                head=np.zeros((len(raw), 6), dtype=np.float32),
                fms=(raw / 20.0).astype(np.float32),
                fms_raw=raw,
                time=np.arange(len(raw), dtype=np.float32) * 0.5,
                participant_id=f"P{idx}",
                session_id=f"lds_train_{idx}",
                source_file=f"lds_train_{idx}.csv",
            )
        )
    lds_info = build_lds_weight_info(
        train_sessions,
        prediction_start=2,
        fms_scaler={"min": 0.0, "max": 20.0},
        min_value=0.0,
        max_value=20.0,
        bin_size=1.0,
        kernel="gaussian",
        kernel_size=5,
        sigma=2.0,
        gamma=0.5,
        weight_min=0.5,
        weight_max=3.0,
    )
    weights = lds_info["weights"]
    if lds_info["train_target_count"] != 12:
        raise AssertionError(f"LDS should use only post-start train targets, got {lds_info['train_target_count']}")
    val_like_raw = np.full(8, 20.0, dtype=np.float32)
    val_like_session = DenseFMSSession(
        head=np.zeros((len(val_like_raw), 6), dtype=np.float32),
        fms=(val_like_raw / 20.0).astype(np.float32),
        fms_raw=val_like_raw,
        time=np.arange(len(val_like_raw), dtype=np.float32) * 0.5,
        participant_id="PV",
        session_id="lds_val_like_not_used",
        source_file="lds_val_like_not_used.csv",
    )
    if val_like_session.length <= 2:
        raise AssertionError("Invalid LDS validation-like leakage fixture")
    if lds_info["empirical_density"][20] != 0:
        raise AssertionError("LDS train-only density should not include held-out-like FMS=20 targets")
    if not weights[18] > weights[5]:
        raise AssertionError(f"Rare high-FMS LDS bin should be weighted above dense mid bin: w18={weights[18]}, w5={weights[5]}")
    outputs = {
        "current": torch.full((1, 4), 5.0 / 20.0),
        "mask": torch.ones(1, 4, dtype=torch.bool),
        "prediction_start": torch.tensor(2),
    }
    fms = torch.tensor([[0.0, 0.0, 5.0, 5.0, 18.0, 18.0]], dtype=torch.float32) / 20.0
    lengths = torch.tensor([6])
    plain_loss, plain_parts = compute_online_current_risk_loss(
        outputs,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[1],
        rise_thresholds=[2.0],
        ordinal_bins=[0.0, 5.0, 18.0, 20.0],
        current_reg_aux_weight=0.0,
        ordinal_loss_weight=0.0,
        risk_loss_weight=0.0,
        smoothness_weight=0.0,
        loss_type="mae",
    )
    weighted_loss, weighted_parts = compute_online_current_risk_loss(
        outputs,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[1],
        rise_thresholds=[2.0],
        ordinal_bins=[0.0, 5.0, 18.0, 20.0],
        current_reg_aux_weight=0.0,
        ordinal_loss_weight=0.0,
        risk_loss_weight=0.0,
        smoothness_weight=0.0,
        loss_type="mae",
        lds_weight_table=torch.tensor(weights, dtype=torch.float32),
        lds_min=0.0,
        lds_bin_size=1.0,
    )
    if not weighted_parts["lds_weighting"] or weighted_parts["lds_points"] != plain_parts["valid_points"]:
        raise AssertionError(f"LDS weighting metadata was not reported correctly: {weighted_parts}")
    if not weighted_loss > plain_loss:
        raise AssertionError(f"LDS should emphasize the rare high-FMS errors in this fixture: {weighted_loss} <= {plain_loss}")
    aux_outputs = {
        "current": torch.full((1, 4), 5.0 / 20.0),
        "current_reg": torch.full((1, 4), 5.0 / 20.0),
        "mask": torch.ones(1, 4, dtype=torch.bool),
        "prediction_start": torch.tensor(2),
    }
    plain_aux_loss, plain_aux_parts = compute_online_current_risk_loss(
        aux_outputs,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[1],
        rise_thresholds=[2.0],
        ordinal_bins=[0.0, 5.0, 18.0, 20.0],
        current_reg_aux_weight=1.0,
        ordinal_loss_weight=0.0,
        risk_loss_weight=0.0,
        smoothness_weight=0.0,
        loss_type="mae",
    )
    weighted_aux_loss, weighted_aux_parts = compute_online_current_risk_loss(
        aux_outputs,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[1],
        rise_thresholds=[2.0],
        ordinal_bins=[0.0, 5.0, 18.0, 20.0],
        current_reg_aux_weight=1.0,
        ordinal_loss_weight=0.0,
        risk_loss_weight=0.0,
        smoothness_weight=0.0,
        loss_type="mae",
        lds_weight_table=torch.tensor(weights, dtype=torch.float32),
        lds_min=0.0,
        lds_bin_size=1.0,
    )
    if not weighted_aux_parts["loss_current_reg_aux"] > plain_aux_parts["loss_current_reg_aux"]:
        raise AssertionError(
            f"LDS should also weight current_reg aux loss: {weighted_aux_parts['loss_current_reg_aux']} "
            f"<= {plain_aux_parts['loss_current_reg_aux']}"
        )
    if not weighted_aux_loss > plain_aux_loss:
        raise AssertionError(f"LDS aux total loss should exceed plain aux total loss: {weighted_aux_loss} <= {plain_aux_loss}")


def test_online_current_risk_transition_weighting_targets_only() -> None:
    outputs = {
        "current": torch.tensor([[4.0, 4.0, 4.0, 4.0]], dtype=torch.float32) / 20.0,
        "current_reg": torch.tensor([[4.0, 4.0, 4.0, 4.0]], dtype=torch.float32) / 20.0,
        "mask": torch.ones(1, 4, dtype=torch.bool),
        "prediction_start": torch.tensor(1),
    }
    fms = torch.tensor([[0.0, 12.0, 12.0, 4.0, 4.0, 4.0]], dtype=torch.float32) / 20.0
    lengths = torch.tensor([6])
    plain_loss, plain_parts = compute_online_current_risk_loss(
        outputs,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        ordinal_bins=[0.0, 4.0, 8.0, 12.0, 20.0],
        current_reg_aux_weight=1.0,
        ordinal_loss_weight=0.0,
        risk_loss_weight=0.0,
        smoothness_weight=0.0,
        loss_type="mae",
    )
    weighted_loss, weighted_parts = compute_online_current_risk_loss(
        outputs,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        ordinal_bins=[0.0, 4.0, 8.0, 12.0, 20.0],
        current_reg_aux_weight=1.0,
        ordinal_loss_weight=0.0,
        risk_loss_weight=0.0,
        smoothness_weight=0.0,
        loss_type="mae",
        transition_weighting=True,
        transition_horizon_steps=[2],
        transition_drop_threshold=2.0,
        transition_recovery_threshold=3.0,
        transition_high_threshold=8.0,
        transition_low_threshold=5.0,
        transition_drop_weight=2.0,
        transition_recovery_weight=3.0,
        transition_rise_weight=1.0,
        transition_max_weight=3.0,
    )
    if not weighted_parts["transition_weighting"]:
        raise AssertionError(f"Transition weighting metadata did not activate: {weighted_parts}")
    if weighted_parts["transition_recovery_points"] != 2 or weighted_parts["transition_drop_points"] != 2:
        raise AssertionError(f"Expected two high-to-low transition points: {weighted_parts}")
    if weighted_parts["transition_rise_points"] != 0:
        raise AssertionError(f"Recovery fixture should not count rise events: {weighted_parts}")
    if weighted_parts["transition_max_observed_weight"] != 3.0:
        raise AssertionError(f"Recovery weight should be the maximum observed event weight: {weighted_parts}")
    if not weighted_parts["loss_current_reg_aux"] > plain_parts["loss_current_reg_aux"]:
        raise AssertionError(
            f"Transition weighting should emphasize current_reg recovery errors: "
            f"{weighted_parts['loss_current_reg_aux']} <= {plain_parts['loss_current_reg_aux']}"
        )
    if not weighted_loss > plain_loss:
        raise AssertionError(f"Transition-weighted loss should exceed plain loss in this fixture: {weighted_loss} <= {plain_loss}")


def test_online_current_risk_trajectory_shape_loss() -> None:
    outputs = {
        "current": torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "mask": torch.ones(1, 4, dtype=torch.bool),
        "prediction_start": torch.tensor(1),
    }
    fms = torch.tensor([[0.0, 0.0, 0.0, 20.0, 20.0]], dtype=torch.float32) / 20.0
    lengths = torch.tensor([5])
    plain_loss, _plain_parts = compute_online_current_risk_loss(
        outputs,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[1],
        rise_thresholds=[2.0],
        ordinal_bins=[0.0, 10.0, 20.0],
        current_reg_aux_weight=0.0,
        ordinal_loss_weight=0.0,
        risk_loss_weight=0.0,
        smoothness_weight=0.0,
        loss_type="mae",
    )
    shaped_loss, shaped_parts = compute_online_current_risk_loss(
        outputs,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[1],
        rise_thresholds=[2.0],
        ordinal_bins=[0.0, 10.0, 20.0],
        current_reg_aux_weight=0.0,
        ordinal_loss_weight=0.0,
        risk_loss_weight=0.0,
        smoothness_weight=0.0,
        loss_type="mae",
        trajectory_loss_weight=1.0,
        trajectory_delta_steps=[1, 2],
        trajectory_delta_weight=1.0,
        trajectory_centered_weight=0.5,
        trajectory_range_weight=0.2,
        trajectory_loss_type="mae",
        trajectory_min_points=2,
    )
    if shaped_parts["loss_trajectory"] <= 0:
        raise AssertionError(f"Trajectory shape loss should activate on a flat prediction: {shaped_parts}")
    if shaped_parts["trajectory_delta_points"] != 5 or shaped_parts["trajectory_centered_points"] != 4 or shaped_parts["trajectory_range_points"] != 1:
        raise AssertionError(f"Trajectory shape metadata has wrong point counts: {shaped_parts}")
    if not shaped_loss > plain_loss:
        raise AssertionError(f"Trajectory-shaped loss should exceed plain loss in this fixture: {shaped_loss} <= {plain_loss}")
    perfect_outputs = {
        "current": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
        "mask": torch.ones(1, 4, dtype=torch.bool),
        "prediction_start": torch.tensor(1),
    }
    perfect_loss, perfect_parts = compute_online_current_risk_loss(
        perfect_outputs,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[1],
        rise_thresholds=[2.0],
        ordinal_bins=[0.0, 10.0, 20.0],
        current_reg_aux_weight=0.0,
        ordinal_loss_weight=0.0,
        risk_loss_weight=0.0,
        smoothness_weight=0.0,
        loss_type="mae",
        trajectory_loss_weight=1.0,
        trajectory_delta_steps=[1, 2],
        trajectory_delta_weight=1.0,
        trajectory_centered_weight=0.5,
        trajectory_range_weight=0.2,
        trajectory_loss_type="mae",
        trajectory_min_points=2,
    )
    if perfect_parts["loss_trajectory"] > 1e-8 or float(perfect_loss.detach().cpu()) > 1e-8:
        raise AssertionError(f"Trajectory shape loss should be zero for a perfect trajectory: {perfect_loss}, {perfect_parts}")


def test_online_current_risk_teacher_current_distillation_alignment() -> None:
    student_outputs = {
        "current": torch.tensor([[0.0, 0.2, 0.5, 0.7]], dtype=torch.float32),
        "mask": torch.tensor([[True, True, True, False]]),
        "prediction_start": torch.tensor(2),
    }
    teacher_outputs = {
        "current": torch.tensor([[9.0, 0.1, 0.4, 0.9, 1.0]], dtype=torch.float32),
        "mask": torch.tensor([[True, True, True, False, True]]),
        "prediction_start": torch.tensor(1),
    }
    loss, parts = compute_teacher_current_distillation_loss(
        student_outputs,
        teacher_outputs,
        loss_type="mae",
    )
    expected = torch.tensor((0.1 + 0.2) / 2.0, dtype=torch.float32)
    if not torch.allclose(loss, expected, atol=1e-6):
        raise AssertionError(f"Current teacher distillation misaligned predictions: {loss} != {expected}")
    if parts["teacher_distill_points"] != 2 or parts["teacher_distill_target"] != "current":
        raise AssertionError(f"Unexpected current teacher distillation metadata: {parts}")

    no_overlap_loss, no_overlap_parts = compute_teacher_current_distillation_loss(
        {
            "current": torch.tensor([[0.0, 0.2]], dtype=torch.float32),
            "mask": torch.ones(1, 2, dtype=torch.bool),
            "prediction_start": torch.tensor(10),
        },
        teacher_outputs,
        loss_type="mae",
    )
    if float(no_overlap_loss.detach().cpu()) != 0.0 or no_overlap_parts["teacher_distill_points"] != 0:
        raise AssertionError(f"No-overlap distillation should produce zero loss and zero points: {no_overlap_parts}")


def test_online_current_risk_final_warning_rapid_only_excludes_high_fms() -> None:
    class DummyOnlineRiskModel(torch.nn.Module):
        requires_full_fms = False

        def forward(self, head, y_calib, lengths, static=None):
            pred_steps = int(head.shape[1]) - 2
            return {
                "current": head.new_full((head.shape[0], pred_steps), 0.8),
                "mask": torch.ones((head.shape[0], pred_steps), dtype=torch.bool, device=head.device),
                "prediction_start": torch.tensor(2, device=head.device),
                "risk_probs": head.new_zeros((head.shape[0], pred_steps, 1)),
                "ordinal_probs": head.new_zeros((head.shape[0], pred_steps, 9)),
            }

    raw_fms = np.full(8, 15.0, dtype=np.float32)
    session = DenseFMSSession(
        head=np.zeros((8, 6), dtype=np.float32),
        fms=(raw_fms / 20.0).astype(np.float32),
        fms_raw=raw_fms,
        time=np.arange(8, dtype=np.float32) * 0.5,
        participant_id="P",
        session_id="rapid_only_dummy",
        source_file="dummy.csv",
    )
    loader = DataLoader(DenseFMSSessionDataset([session]), batch_size=1, collate_fn=collate_sessions)
    result = collect_online_current_risk_predictions(
        DummyOnlineRiskModel(),
        loader,
        torch.device("cpu"),
        calibration_steps=2,
        fms_scaler={"min": 0.0, "max": 20.0},
        rise_horizon_steps=[1],
        rise_thresholds=[2.0],
        ordinal_bins=[0, 2, 4, 6, 8, 10, 12, 15, 20],
        high_fms_warning_threshold=12.0,
        rapid_rise_probability_threshold=0.5,
        final_warning_mode="rapid_rise_only",
        split_name="test",
    )
    records = result["prediction_records"]
    valid_records = [row for row in records if row["rapid_rise_valid_0.5s"]]
    if not valid_records:
        raise AssertionError("Dummy rapid-only warning test produced no valid records")
    if not all(row["alarm_warning_high_fms"] for row in valid_records):
        raise AssertionError("Dummy model should produce high-FMS diagnostic alarms")
    if any(row["final_warning"] for row in valid_records):
        raise AssertionError("final_warning_mode=rapid_rise_only must exclude high-FMS diagnostic alarms")
    if result["metrics"]["final_warning_mode"] != "rapid_rise_only":
        raise AssertionError("final warning mode was not recorded in metrics")


def test_online_current_risk_future_delta_event_auxiliary_paths() -> None:
    torch.manual_seed(13)
    model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=2,
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        future_aux_horizon_steps=[1, 2],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2],
        state_feedback_mode="none",
        current_head_mode="basic",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        decoder_context_mode="state",
        decoder_hidden_dim=48,
        fall_risk_head_enabled=True,
    )
    model.eval()
    head = torch.randn(2, 12, 6)
    fms = torch.rand(2, 12)
    lengths = torch.tensor([12, 11])
    with torch.no_grad():
        out = model(head, fms[:, :4], lengths)
    if out["future_aux"].shape != (2, 8, 2):
        raise AssertionError(f"Future auxiliary head has wrong shape: {out['future_aux'].shape}")
    if out["event_logits"].shape != (2, 8, 2, 3) or out["event_probs"].shape != (2, 8, 2, 3):
        raise AssertionError(f"Event auxiliary head has wrong shape: {out['event_logits'].shape}, {out['event_probs'].shape}")
    if out["fall_risk_probs"].shape != (2, 8, 1):
        raise AssertionError(f"Fall-risk head has wrong shape: {out['fall_risk_probs'].shape}")
    loss, parts = compute_online_current_risk_loss(
        out,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        ordinal_bins=list(range(21)),
        ordinal_loss_mode="cumulative_bce",
        fall_horizon_steps=[2],
        fall_thresholds=[2.0],
        fall_loss_weight=0.05,
        future_aux_horizon_steps=[1, 2],
        future_aux_loss_weight=0.2,
        delta_aux_loss_weight=0.3,
        event_aux_loss_weight=0.1,
        event_delta_threshold=1.0,
    )
    if (
        not torch.isfinite(loss)
        or parts["future_aux_points"] <= 0
        or parts["delta_aux_points"] <= 0
        or parts["event_aux_points"] <= 0
        or parts["fall_risk_points"] <= 0
    ):
        raise AssertionError(f"Future/delta/event auxiliary loss path failed: loss={loss}, parts={parts}")

    raw_fms = np.asarray([0.0, 1.0, 2.0, 3.0, 2.0, 6.0, 6.0, 7.0, 8.0, 7.0, 9.0, 10.0], dtype=np.float32)
    session = DenseFMSSession(
        head=np.zeros((len(raw_fms), 6), dtype=np.float32),
        fms=(raw_fms / 20.0).astype(np.float32),
        fms_raw=raw_fms,
        time=np.arange(len(raw_fms), dtype=np.float32) * 0.5,
        participant_id="P",
        session_id="future_aux_dummy",
        source_file="future_aux_dummy.csv",
    )
    loader = DataLoader(DenseFMSSessionDataset([session]), batch_size=1, collate_fn=collate_sessions)
    result = collect_online_current_risk_predictions(
        model,
        loader,
        torch.device("cpu"),
        calibration_steps=4,
        fms_scaler={"min": 0.0, "max": 20.0},
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        ordinal_bins=list(range(21)),
        fall_horizon_steps=[2],
        fall_thresholds=[2.0],
        sampling_interval=0.5,
        future_aux_horizon_steps=[1, 2],
        split_name="sanity",
    )
    metrics = result["metrics"]
    for label in ("0.5s", "1s"):
        if label not in metrics["future_aux"] or metrics["future_aux"][label]["n"] <= 0:
            raise AssertionError(f"Missing future auxiliary regression metrics for {label}: {metrics['future_aux']}")
        if label not in metrics["delta_aux"] or "corr" not in metrics["delta_aux"][label]:
            raise AssertionError(f"Missing delta auxiliary metrics for {label}: {metrics['delta_aux']}")
        if label not in metrics["event_aux"] or metrics["event_aux"][label]["n"] <= 0:
            raise AssertionError(f"Missing event auxiliary metrics for {label}: {metrics['event_aux']}")
    if "1s" not in metrics["rapid_drop"] or metrics["rapid_drop"]["1s"]["n"] <= 0:
        raise AssertionError(f"Missing rapid-drop metrics: {metrics.get('rapid_drop')}")
    valid_records = [row for row in result["prediction_records"] if row["future_aux_valid_0.5s"]]
    if not valid_records:
        raise AssertionError("Future auxiliary prediction collection produced no valid records")
    first = valid_records[0]
    for key in ("future_aux_pred_0.5s", "delta_aux_pred_0.5s", "event_aux_pred_0.5s", "p_rapid_drop_1s"):
        if first[key] is None:
            raise AssertionError(f"Future auxiliary prediction CSV field {key} was not populated: {first}")


def test_online_current_causal_dynamics_feature_bank() -> None:
    torch.manual_seed(17)
    head = torch.randn(1, 12, 6)
    changed_future = head.clone()
    changed_future[:, 7:] = changed_future[:, 7:] + 100.0
    features = append_motion_features(head, "causal_dynamics_v1")
    changed_features = append_motion_features(changed_future, "causal_dynamics_v1")
    if features.shape != (1, 12, 24):
        raise AssertionError(f"causal_dynamics_v1 should append 18 features to 6D motion, got {features.shape}")
    if not torch.isfinite(features).all():
        raise AssertionError("causal_dynamics_v1 produced non-finite values")
    if not torch.allclose(features[:, :7], changed_features[:, :7], atol=1e-6, rtol=0):
        raise AssertionError("causal_dynamics_v1 changed earlier feature rows after editing future motion")

    model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=2,
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        future_aux_horizon_steps=[1],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2],
        state_feedback_mode="none",
        current_head_mode="basic",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        decoder_context_mode="state",
        decoder_hidden_dim=48,
        motion_feature_mode="causal_dynamics_v1",
    )
    out = model(torch.randn(2, 12, 6), torch.rand(2, 4), torch.tensor([12, 11]))
    if out["current"].shape != (2, 8) or out["future_aux"].shape != (2, 8, 1):
        raise AssertionError(f"causal_dynamics_v1 model forward shape failed: {out['current'].shape}, {out['future_aux'].shape}")


def test_online_current_remaining_heads_and_multitimescale_features() -> None:
    torch.manual_seed(18)
    head = torch.randn(1, 140, 6)
    changed_future = head.clone()
    changed_future[:, 80:] = changed_future[:, 80:] - 100.0
    features = append_motion_features(head, "multi_timescale_v1")
    changed_features = append_motion_features(changed_future, "multi_timescale_v1")
    if features.shape != (1, 140, 22):
        raise AssertionError(f"multi_timescale_v1 should append 16 features to 6D motion, got {features.shape}")
    if not torch.isfinite(features).all():
        raise AssertionError("multi_timescale_v1 produced non-finite values")
    if not torch.allclose(features[:, :80], changed_features[:, :80], atol=1e-6, rtol=0):
        raise AssertionError("multi_timescale_v1 changed earlier feature rows after editing future motion")

    fms = torch.rand(2, 16)
    lengths = torch.tensor([16, 15])
    trajectory_decoder_out = None
    head_specs = [
        ("residual_update", "current_residual_step", {}),
        ("person_prior", "person_response_speed", {}),
        ("trajectory_decoder", "current_trajectory", {"current_trajectory_offsets": [0, 2, 4]}),
        ("regime_gated", "current_regime_gate_probs", {"regime_class_count": 5}),
        ("anchor_regime_gated", "current_regime_gate_probs", {"regime_class_count": 5}),
        ("state_space_delta", "current_state_drive", {"current_delta_scale": 1.2}),
        ("range_scaled_delta", "current_range_scale", {"current_delta_scale": 1.2}),
        ("calib_prior_range_scaled_delta", "current_calib_prior_gate", {"current_delta_scale": 1.2}),
        (
            "calib_lowcap_range_scaled_delta",
            "current_calib_prior_gate",
            {"current_delta_scale": 1.2, "current_range_guard_cap": 6.0},
        ),
        (
            "guarded_range_scaled_delta",
            "current_range_guard_low_score",
            {
                "current_delta_scale": 1.2,
                "current_range_guard_low_threshold": 5.0,
                "current_range_guard_temperature": 1.0,
                "current_range_guard_floor": 0.2,
                "current_range_guard_cap": 2.0,
            },
        ),
        (
            "zero_anchor_mixture",
            "current_anchor_gate",
            {
                "current_delta_scale": 2.0,
                "current_anchor_delta_growth_scale": 2.0,
                "current_anchor_delta_growth_horizon_seconds": 2.0,
            },
        ),
    ]
    for mode, expected_key, extra_kwargs in head_specs:
        model = build_model(
            "online_fms_risk_tracker",
            head_dim=6,
            calibration_steps=4,
            recent_steps=3,
            horizon_steps=2,
            rise_horizon_steps=[2],
            rise_thresholds=[2.0],
            d_model=32,
            hidden_dim=40,
            transformer_heads=4,
            transformer_ff_dim=64,
            stream_context_mode="deep_tcn_latent_gru",
            deep_tcn_dilations=[1, 2],
            state_feedback_mode="none",
            current_head_mode=mode,
            ordinal_head_mode="cumulative",
            ordinal_bins=list(range(21)),
            decoder_context_mode="state",
            decoder_hidden_dim=48,
            **extra_kwargs,
        )
        out = model(torch.randn(2, 16, 6), fms[:, :4], lengths)
        if out["current"].shape != (2, 12) or expected_key not in out:
            raise AssertionError(f"{mode} current head forward failed: keys={sorted(out)} shape={out['current'].shape}")
        if mode == "zero_anchor_mixture":
            if out["current_anchor_value"].shape != out["current"].shape:
                raise AssertionError("zero_anchor_mixture anchor value shape failed.")
            if torch.any(out["current_anchor_gate"] < 0.0) or torch.any(out["current_anchor_gate"] > 1.0):
                raise AssertionError("zero_anchor_mixture gate must stay in [0, 1].")
            delta_range = out["current_anchor_delta_range"]
            if delta_range.shape != out["current"].shape:
                raise AssertionError(f"zero_anchor_mixture delta range shape failed: {delta_range.shape}")
            if not torch.all(delta_range[:, -1] > delta_range[:, 0]):
                raise AssertionError("zero_anchor_mixture dynamic delta range should grow over time.")
            anchor_loss, anchor_parts = compute_online_current_risk_loss(
                out,
                fms,
                lengths,
                {"min": 0.0, "max": 20.0},
                rise_horizon_steps=[2],
                rise_thresholds=[2.0],
                ordinal_bins=list(range(21)),
                low_suppressor_gate_loss_weight=0.1,
            )
            if not torch.isfinite(anchor_loss) or anchor_parts["low_suppressor_gate_points"] <= 0:
                raise AssertionError(f"zero_anchor_mixture gate loss path failed: loss={anchor_loss}, parts={anchor_parts}")
            high_anchor_loss, high_anchor_parts = compute_online_current_risk_loss(
                out,
                fms,
                lengths,
                {"min": 0.0, "max": 20.0},
                rise_horizon_steps=[2],
                rise_thresholds=[2.0],
                ordinal_bins=list(range(21)),
                anchor_gate_loss_weight=0.1,
                anchor_gate_threshold=6.0,
                anchor_gate_pos_weight=2.0,
            )
            if not torch.isfinite(high_anchor_loss) or high_anchor_parts["anchor_gate_points"] <= 0:
                raise AssertionError(
                    f"zero_anchor_mixture high anchor gate loss path failed: loss={high_anchor_loss}, parts={high_anchor_parts}"
                )
        if mode == "trajectory_decoder":
            if out["current_trajectory"].shape != (2, 12, 3):
                raise AssertionError(f"trajectory decoder shape failed: {out['current_trajectory'].shape}")
            trajectory_decoder_out = out
        if mode == "guarded_range_scaled_delta":
            for key in ["current_range_guard_open", "current_range_guard_multiplier", "current_range_effective_gate"]:
                if key not in out or out[key].shape != out["current"].shape:
                    raise AssertionError(f"guarded range head missing or wrong shape for {key}")
            if torch.any(out["current_range_guard_low_score"] < 0.0) or torch.any(out["current_range_guard_low_score"] > 1.0):
                raise AssertionError("guarded range low score must stay in [0, 1].")
            if torch.any(out["current_range_effective_gate"] > out["current_range_gate"] + 1e-6):
                raise AssertionError("guarded range effective gate should not exceed the base range gate.")
        if mode in {"calib_prior_range_scaled_delta", "calib_lowcap_range_scaled_delta"}:
            if out["current_calib_prior_cap"].shape != (2,):
                raise AssertionError(f"calibration-prior cap shape failed: {out['current_calib_prior_cap'].shape}")
            if torch.any(out["current_calib_prior_gate"] < 0.0) or torch.any(out["current_calib_prior_gate"] > 1.0):
                raise AssertionError("calibration-prior gate must stay in [0, 1].")
            if torch.any(out["current_calib_prior_cap"] < 0.0) or torch.any(out["current_calib_prior_cap"] > 1.0):
                raise AssertionError("calibration-prior cap must stay in [0, 1].")
            if mode == "calib_lowcap_range_scaled_delta" and torch.any(out["current_calib_prior_cap"] > 6.0 / 20.0 + 1e-6):
                raise AssertionError("low-cap calibration-prior cap must obey current_range_guard_cap.")
    if trajectory_decoder_out is None:
        raise AssertionError("trajectory decoder sanity output was not produced")
    trajectory_loss, trajectory_parts = compute_online_current_risk_loss(
        trajectory_decoder_out,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        ordinal_bins=list(range(21)),
        trajectory_decoder_loss_weight=0.5,
    )
    if not torch.isfinite(trajectory_loss) or trajectory_parts["trajectory_decoder_points"] <= 0:
        raise AssertionError(f"Trajectory decoder loss path failed: loss={trajectory_loss}, parts={trajectory_parts}")

    aux_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=2,
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2],
        state_feedback_mode="none",
        current_head_mode="basic",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        decoder_context_mode="state",
        decoder_hidden_dim=48,
        coarse_band_bins=[5.0, 10.0, 15.0],
        regime_head_enabled=True,
        regime_class_count=5,
        uncertainty_head_enabled=True,
    )
    out = aux_model(torch.randn(2, 16, 6), fms[:, :4], lengths)
    if out["coarse_band_logits"].shape != (2, 12, 4) or out["regime_logits"].shape != (2, 12, 5):
        raise AssertionError("Coarse band or regime head shape is wrong")
    if out["current_sigma"].shape != out["current"].shape or not torch.isfinite(out["current_sigma"]).all():
        raise AssertionError("Uncertainty sigma head shape/value is wrong")
    loss, parts = compute_online_current_risk_loss(
        out,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        ordinal_bins=list(range(21)),
        ordinal_loss_mode="cumulative_bce",
        coarse_band_loss_weight=0.1,
        regime_loss_weight=0.1,
        uncertainty_loss_weight=0.1,
    )
    if (
        not torch.isfinite(loss)
        or parts["coarse_band_points"] <= 0
        or parts["regime_points"] <= 0
        or parts["uncertainty_points"] <= 0
    ):
        raise AssertionError(f"Remaining auxiliary loss paths failed: loss={loss}, parts={parts}")

    affine_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=2,
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2],
        state_feedback_mode="none",
        current_head_mode="range_scaled_delta",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        decoder_context_mode="state",
        decoder_hidden_dim=48,
        current_session_affine_head_enabled=True,
        current_session_affine_hidden_dim=32,
        current_session_affine_scale_range=0.25,
        current_session_affine_bias_range=0.15,
        current_affine_head_enabled=True,
        current_affine_hidden_dim=32,
        current_affine_scale_range=0.5,
        current_affine_bias_range=0.25,
        current_binned_affine_head_enabled=True,
        current_binned_affine_anchor_bins=[5.0, 10.0],
        current_binned_affine_pred_bins=[5.0, 10.0],
        current_binned_affine_time_bins=[6.0],
        current_binned_affine_scale_range=1.5,
        current_binned_affine_bias_range=0.5,
    )
    affine_out = affine_model(torch.randn(2, 16, 6), fms[:, :4], lengths)
    if "current_pre_session_affine" not in affine_out or affine_out["current_pre_session_affine"].shape != affine_out["current"].shape:
        raise AssertionError(f"Session affine calibration head missing pre-correction output: keys={sorted(affine_out)}")
    if affine_out.get("current_session_affine_scale", torch.empty(0)).shape != (2, 1):
        raise AssertionError("Session affine calibration scale should be one value per session.")
    if affine_out.get("current_session_affine_bias", torch.empty(0)).shape != (2, 1):
        raise AssertionError("Session affine calibration bias should be one value per session.")
    for key in ("current_pre_affine", "current_affine_scale", "current_affine_bias"):
        if key not in affine_out or affine_out[key].shape != affine_out["current"].shape:
            raise AssertionError(f"Affine calibration head missing {key}: keys={sorted(affine_out)}")
    for key in (
        "current_pre_binned_affine",
        "current_binned_affine_scale",
        "current_binned_affine_bias",
        "current_binned_affine_bin",
    ):
        if key not in affine_out or affine_out[key].shape != affine_out["current"].shape:
            raise AssertionError(f"Binned affine calibration head missing {key}: keys={sorted(affine_out)}")
    if not torch.allclose(affine_out["current"], affine_out["current_pre_affine"], atol=1e-6, rtol=0):
        raise AssertionError("Affine calibration head must be identity-initialized.")
    if not torch.allclose(affine_out["current"], affine_out["current_pre_session_affine"], atol=1e-6, rtol=0):
        raise AssertionError("Session affine calibration head must be identity-initialized.")
    if not torch.allclose(
        affine_out["current_session_affine_scale"],
        torch.ones_like(affine_out["current_session_affine_scale"]),
        atol=1e-6,
        rtol=0,
    ):
        raise AssertionError("Session affine calibration scale should initialize to 1.")
    if not torch.allclose(
        affine_out["current_session_affine_bias"],
        torch.zeros_like(affine_out["current_session_affine_bias"]),
        atol=1e-6,
        rtol=0,
    ):
        raise AssertionError("Session affine calibration bias should initialize to 0.")
    if not torch.allclose(affine_out["current_affine_scale"], torch.ones_like(affine_out["current_affine_scale"]), atol=1e-6, rtol=0):
        raise AssertionError("Affine calibration scale should initialize to 1.")
    if not torch.allclose(affine_out["current_affine_bias"], torch.zeros_like(affine_out["current_affine_bias"]), atol=1e-6, rtol=0):
        raise AssertionError("Affine calibration bias should initialize to 0.")
    if not torch.allclose(
        affine_out["current_binned_affine_scale"],
        torch.ones_like(affine_out["current_binned_affine_scale"]),
        atol=1e-6,
        rtol=0,
    ):
        raise AssertionError("Binned affine calibration scale should initialize to 1.")
    if not torch.allclose(
        affine_out["current_binned_affine_bias"],
        torch.zeros_like(affine_out["current_binned_affine_bias"]),
        atol=1e-6,
        rtol=0,
    ):
        raise AssertionError("Binned affine calibration bias should initialize to 0.")
    affine_loss, affine_parts = compute_online_current_risk_loss(
        affine_out,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        ordinal_bins=list(range(21)),
        session_affine_scale_regularization_weight=0.1,
        session_affine_bias_regularization_weight=0.1,
    )
    if (
        not torch.isfinite(affine_loss)
        or affine_parts["session_affine_scale_points"] <= 0
        or affine_parts["session_affine_bias_points"] <= 0
    ):
        raise AssertionError(f"Session affine regularization path failed: loss={affine_loss}, parts={affine_parts}")

    suppressor_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=2,
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2],
        state_feedback_mode="none",
        current_head_mode="basic",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        decoder_context_mode="state",
        decoder_hidden_dim=48,
        current_low_suppressor_enabled=True,
        current_low_suppressor_hidden_dim=32,
        current_low_suppressor_delta_range=0.25,
    )
    suppressor_out = suppressor_model(torch.randn(2, 16, 6), fms[:, :4], lengths)
    for key in (
        "current_pre_low_suppressor",
        "current_low_suppressor_correction",
        "current_low_suppressor_gate",
        "current_low_suppressor_gate_logits",
    ):
        if key not in suppressor_out or suppressor_out[key].shape != suppressor_out["current"].shape:
            raise AssertionError(f"Low-FMS suppressor missing {key}: keys={sorted(suppressor_out)}")
    if not torch.isfinite(suppressor_out["current_low_suppressor_gate"]).all():
        raise AssertionError("Low-FMS suppressor gate produced non-finite values.")
    if torch.any(suppressor_out["current_low_suppressor_correction"] < 0):
        raise AssertionError("Low-FMS suppressor correction must be non-negative before subtraction.")
    if torch.any(suppressor_out["current"] > suppressor_out["current_pre_low_suppressor"] + 1e-6):
        raise AssertionError("Low-FMS suppressor must be negative-only.")
    suppressor_loss, suppressor_parts = compute_online_current_risk_loss(
        suppressor_out,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        ordinal_bins=list(range(21)),
        low_suppressor_gate_loss_weight=0.1,
        low_suppressor_correction_regularization_weight=0.1,
    )
    if (
        not torch.isfinite(suppressor_loss)
        or suppressor_parts["low_suppressor_gate_points"] <= 0
        or suppressor_parts["low_suppressor_correction_points"] <= 0
    ):
        raise AssertionError(f"Low-FMS suppressor loss path failed: loss={suppressor_loss}, parts={suppressor_parts}")

    residual_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=2,
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2],
        state_feedback_mode="none",
        current_head_mode="basic",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        decoder_context_mode="state",
        decoder_hidden_dim=48,
        calibration_residual_adapter_enabled=True,
        calibration_residual_feature_dim=13,
        calibration_residual_adapter_hidden_dim=24,
        calibration_residual_adapter_mode="mlp_decay_high_gate",
        calibration_residual_delta_range=0.15,
        calibration_residual_decay_seconds=120.0,
        calibration_residual_gate_low_threshold=8.0,
        calibration_residual_gate_high_threshold=10.0,
        calibration_residual_gate_anchor_threshold=10.0,
        calibration_residual_gate_temperature=1.0,
    )
    try:
        residual_model(torch.randn(2, 16, 6), fms[:, :4], lengths)
    except ValueError as exc:
        if "calibration_residual_features" not in str(exc):
            raise
    else:
        raise AssertionError("Residual adapter model should require calibration_residual_features.")
    residual_features = torch.zeros(2, 13)
    residual_out = residual_model(
        torch.randn(2, 16, 6),
        fms[:, :4],
        lengths,
        calibration_residual_features=residual_features,
    )
    if residual_out["current_pre_residual_adapter"].shape != residual_out["current"].shape:
        raise AssertionError("Residual adapter did not expose pre-adapter current output.")
    if residual_out["current_residual_adapter_correction"].shape != residual_out["current"].shape:
        raise AssertionError("Residual adapter correction shape is wrong.")
    if residual_out["current_residual_adapter_gate"].shape != residual_out["current"].shape:
        raise AssertionError("Residual adapter high-regime gate shape is wrong.")
    residual_gate = residual_out["current_residual_adapter_gate"]
    if not torch.isfinite(residual_gate).all() or torch.any(residual_gate < 0.0) or torch.any(residual_gate > 1.0):
        raise AssertionError("Residual adapter high-regime gate must be finite and in [0, 1].")
    if not torch.allclose(residual_out["current"], residual_out["current_pre_residual_adapter"], atol=1e-6, rtol=0):
        raise AssertionError("MLP residual adapter must be identity-initialized.")
    residual_loss, residual_parts = compute_online_current_risk_loss(
        residual_out,
        fms,
        lengths,
        {"min": 0.0, "max": 20.0},
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        ordinal_bins=list(range(21)),
        calibration_residual_regularization_weight=0.1,
        low_overprediction_weight=0.1,
        high_underprediction_weight=0.1,
    )
    if not torch.isfinite(residual_loss) or residual_parts["calibration_residual_reg_points"] <= 0:
        raise AssertionError(f"Residual adapter regularization path failed: loss={residual_loss}, parts={residual_parts}")

    mean_decay_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=2,
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2],
        state_feedback_mode="none",
        current_head_mode="basic",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        decoder_context_mode="state",
        decoder_hidden_dim=48,
        calibration_residual_adapter_enabled=True,
        calibration_residual_feature_dim=13,
        calibration_residual_adapter_mode="mean_decay",
        calibration_residual_delta_range=0.15,
        calibration_residual_decay_seconds=120.0,
    )
    mean_features = torch.zeros(2, 13)
    mean_features[:, 0] = 0.05
    mean_out = mean_decay_model(
        torch.randn(2, 16, 6),
        fms[:, :4],
        lengths,
        calibration_residual_features=mean_features,
    )
    if not torch.isfinite(mean_out["current"]).all() or torch.abs(mean_out["current_residual_adapter_correction"]).max() <= 0:
        raise AssertionError("Mean-decay residual adapter did not apply a finite correction.")

    summary_fusion_model = build_model(
        "online_fms_risk_tracker",
        head_dim=6,
        calibration_steps=4,
        recent_steps=3,
        horizon_steps=2,
        rise_horizon_steps=[2],
        rise_thresholds=[2.0],
        d_model=32,
        hidden_dim=40,
        transformer_heads=4,
        transformer_ff_dim=64,
        stream_context_mode="deep_tcn_latent_gru",
        deep_tcn_dilations=[1, 2],
        state_feedback_mode="none",
        current_head_mode="basic",
        ordinal_head_mode="cumulative",
        ordinal_bins=list(range(21)),
        decoder_context_mode="state",
        decoder_hidden_dim=48,
        calibration_summary_fusion_enabled=True,
        calibration_summary_fusion_feature_dim=13,
        calibration_summary_fusion_hidden_dim=24,
        calibration_summary_fusion_mode="additive_gated",
        calibration_summary_fusion_strength=1.0,
    )
    try:
        summary_fusion_model(torch.randn(2, 16, 6), fms[:, :4], lengths)
    except ValueError as exc:
        if "calibration_residual_features" not in str(exc):
            raise
    else:
        raise AssertionError("Calibration-summary fusion model should require calibration_residual_features.")
    summary_features = torch.randn(2, 13)
    summary_mask = torch.ones_like(summary_features)
    summary_out = summary_fusion_model(
        torch.randn(2, 16, 6),
        fms[:, :4],
        lengths,
        calibration_residual_features=summary_features,
        calibration_residual_feature_mask=summary_mask,
    )
    if summary_out["calibration_summary_fusion_gate"].shape != summary_out["current"].shape:
        raise AssertionError("Calibration-summary fusion gate shape is wrong.")
    if summary_out["calibration_summary_fusion_delta_norm"].shape != summary_out["current"].shape:
        raise AssertionError("Calibration-summary fusion delta norm shape is wrong.")
    if not torch.isfinite(summary_out["current"]).all():
        raise AssertionError("Calibration-summary fusion produced non-finite current predictions.")
    if torch.abs(summary_out["calibration_summary_fusion_delta_norm"]).max() > 1e-6:
        raise AssertionError("Calibration-summary fusion should be identity-initialized.")


def test_online_current_calibration_transformer_modes() -> None:
    torch.manual_seed(19)
    head = torch.randn(2, 18, 6)
    fms = torch.rand(2, 18)
    changed_after_calib = fms.clone()
    changed_after_calib[:, 4:] = 1.0 - changed_after_calib[:, 4:]
    lengths = torch.tensor([18, 17])
    for mode, pooling, fusion_mode, expected_calib_dim in (
        ("transformer", "attention", "add", 32),
        ("transformer_cls", "mean", "add", 32),
        ("deep_tcn", "attention", "add", 32),
        ("deep_tcn_transformer", "mean", "add", 32),
        ("deep_tcn", "mean", "mean_last_summary_concat", 40),
        ("deep_tcn", "mean", "mean_last_gated_summary", 40),
        ("deep_tcn", "mean", "mean_last_attention_summary", 40),
        ("deep_tcn", "mean", "mean_last_event_attention_summary", 40),
    ):
        model = build_model(
            "online_fms_risk_tracker",
            head_dim=6,
            calibration_steps=4,
            recent_steps=3,
            horizon_steps=2,
            rise_horizon_steps=[2],
            rise_thresholds=[2.0],
            d_model=32,
            hidden_dim=40,
            transformer_layers=2,
            transformer_heads=4,
            transformer_ff_dim=64,
            pooling=pooling,
            stream_context_mode="deep_tcn_latent_gru",
            deep_tcn_dilations=[1, 2],
            state_feedback_mode="none",
            current_head_mode="basic",
            ordinal_head_mode="cumulative",
            ordinal_bins=list(range(21)),
            decoder_context_mode="state",
            decoder_hidden_dim=48,
            calibration_encoder_mode=mode,
            calibration_fusion_mode=fusion_mode,
            calib_summary_features=True,
        )
        model.eval()
        with torch.no_grad():
            out_full = model(head, fms, lengths)
            out_changed = model(head, changed_after_calib, lengths)
            out_calib_only = model(head, fms[:, :4], lengths)
        if out_full["current"].shape != (2, 14):
            raise AssertionError(f"{mode} calibration encoder current shape failed: {out_full['current'].shape}")
        if out_full["calibration_encoder_mode"] != mode:
            raise AssertionError(f"{mode} calibration encoder did not report its mode")
        if out_full["calibration_fusion_mode"] != fusion_mode:
            raise AssertionError(f"{fusion_mode} calibration fusion mode was not reported")
        if int(out_full["calibration_repr_dim"].detach().cpu()) != expected_calib_dim:
            raise AssertionError(f"{fusion_mode} calibration repr dim mismatch: {out_full['calibration_repr_dim']}")
        if not torch.allclose(out_full["current"], out_changed["current"], atol=1e-6):
            raise AssertionError(f"{mode}/{fusion_mode} calibration encoder leaked post-calibration FMS")
        if not torch.allclose(out_full["current"], out_calib_only["current"], atol=1e-6):
            raise AssertionError(f"{mode}/{fusion_mode} calibration encoder changed when passing calibration-only FMS")


def test_online_current_remaining_experiments_dry_run_command_generation() -> None:
    args = argparse.Namespace(
        python=sys.executable,
        data_dir="DenseFMS/Dataset",
        base_config="configs/online_current/selected_fds_static4.yaml",
        runs_dir="runs/online_fms_current_tracking_0509_remaining",
        split_file=(
            "runs/online_fms_current_tracking_0508/"
            "deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json"
        ),
        pretrain_runs_dir="runs/online_fms_current_tracking_0509_remaining/motion_pretrain",
        pretrain_run_name="motion_energy_causal_dynamics_v1_seed42",
        pretrain_epochs=30,
        pretrain_patience=5,
        batch_size=48,
    )
    pretrain = remaining_pretrain_command(args)
    commands = [remaining_experiment_command(args, candidate) for candidate in REMAINING_CANDIDATES]
    if len(commands) < 10:
        raise AssertionError(f"Remaining experiment plan generated too few commands: {len(commands)}")
    if "scripts/pretrain_online_current_motion_encoder.py" not in pretrain:
        raise AssertionError(f"Remaining pretrain command is wrong: {pretrain}")
    flat = [" ".join(cmd) for cmd in commands]
    required_snippets = [
        "--motion_feature_mode multi_timescale_v1",
        "--current_head_mode person_prior",
        "--current_head_mode residual_update",
        "--coarse_band_bins 5.0 10.0 15.0",
        "--motion_pretrain_checkpoint",
        "--static_features age mssq gender scenario",
        "--regime_head_enabled --regime_loss_weight",
        "--uncertainty_head_enabled --uncertainty_loss_weight",
    ]
    for snippet in required_snippets:
        if not any(snippet in item for item in flat):
            raise AssertionError(f"Remaining experiment dry-run command missing snippet {snippet}: {flat}")


def test_online_current_calsummary_earlyfusion_dry_run_command_generation() -> None:
    args = argparse.Namespace(
        python=sys.executable,
        data_dir="DenseFMS/Dataset",
        base_config="runs/head_redesign_ablation_0513/range_scaled_delta2_120_seed42/config_snapshot.json",
        init_checkpoint="runs/head_redesign_ablation_0513/range_scaled_delta2_120_seed42/best.pt",
        calibration_summary_features=(
            "reports/overnight_current_fms_goal_0514_120s/calibration_summary_features_train_val.json"
        ),
        runs_dir="runs/overnight_current_fms_goal_0514_120s",
        split_file=None,
        selection_metric="goal_composite.strict120",
        learning_rate=1e-3,
        batch_size=48,
        epochs=50,
        patience=8,
        seed=42,
        max_session_points=420,
        smoke=False,
        smoke_limit_sessions=12,
        smoke_max_train_batches=2,
        smoke_max_eval_batches=1,
        smoke_epochs=1,
        smoke_patience=1,
    )
    commands = [calsummary_earlyfusion_command(args, recipe) for recipe in CALSUMMARY_EARLYFUSION_RECIPES]
    if len(commands) != 2:
        raise AssertionError(f"Expected two calibration-summary early-fusion commands, got {len(commands)}.")
    flat = [" ".join(cmd) for cmd in commands]
    required_snippets = [
        "--calibration_residual_features_path "
        "reports/overnight_current_fms_goal_0514_120s/calibration_summary_features_train_val.json",
        "--require_calibration_residual_features",
        "--calibration_summary_fusion_enabled",
        "--calibration_summary_fusion_mode additive_gated",
        "--selection_metric goal_composite.strict120",
        "--max_session_points 420",
        "--no_test_eval",
    ]
    for snippet in required_snippets:
        if not all(snippet in item for item in flat):
            raise AssertionError(f"Early-fusion dry-run command missing snippet {snippet}: {flat}")
    if not any("--low_overprediction_weight 0.02" in item for item in flat):
        raise AssertionError(f"Weak low-penalty early-fusion candidate is missing: {flat}")
    if any("--split_file" in cmd for cmd in commands):
        raise AssertionError(f"Split file should be omitted when no split_file is provided: {commands}")


def test_online_current_calsummary_earlyfusion_cutoff_guard() -> None:
    if calsummary_earlyfusion_is_past_cutoff(datetime(2026, 5, 14, 11, 59, 59), 12):
        raise AssertionError("Cutoff guard should allow runs before 12:00.")
    if not calsummary_earlyfusion_is_past_cutoff(datetime(2026, 5, 14, 12, 0, 0), 12):
        raise AssertionError("Cutoff guard should block runs at 12:00.")
    if not calsummary_earlyfusion_is_past_cutoff(datetime(2026, 5, 14, 19, 0, 0), 12):
        raise AssertionError("Cutoff guard should block runs after 12:00.")
    args = argparse.Namespace(cutoff_hour=12, disable_cutoff_guard=False, execute=True)
    blocked = calsummary_earlyfusion_time_status(args, now=datetime(2026, 5, 14, 19, 0, 0).astimezone())
    if not blocked["cutoff_passed"] or blocked["execute_allowed_by_time"]:
        raise AssertionError(f"Time status should block execute after cutoff: {blocked}")
    args.disable_cutoff_guard = True
    allowed = calsummary_earlyfusion_time_status(args, now=datetime(2026, 5, 14, 19, 0, 0).astimezone())
    if not allowed["cutoff_passed"] or not allowed["execute_allowed_by_time"]:
        raise AssertionError(f"Disabling guard should be visible in time status: {allowed}")


def test_online_current_calsummary_earlyfusion_gate_requires_strict_low() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.csv"
        good = Path(tmp) / "good.csv"
        bad.write_text(
            "target_fms_now,predicted_fms_now\n"
            "0.0,4.0\n"
            "1.0,4.0\n"
            "8.0,8.0\n"
            "12.0,12.0\n",
            encoding="utf-8",
        )
        good.write_text(
            "target_fms_now,predicted_fms_now\n"
            "0.0,1.0\n"
            "1.0,2.0\n"
            "8.0,8.0\n"
            "12.0,12.0\n",
            encoding="utf-8",
        )
        args = argparse.Namespace(
            inputs=[f"bad={bad}", f"good={good}"],
            pred_column="predicted_fms_now",
            thresholds=[8.0, 12.0],
            mae_max=2.0,
            r2_min=-10.0,
            strict_low_bias_max=2.5,
            high12_f1_min=0.5,
            required_pass_count=2,
            require_strict_low_gate=True,
        )
        rows = evaluate_calsummary_earlyfusion_gates(args)
    lookup = {row["label"]: row for row in rows}
    if lookup["bad"]["gate_pass_count"] < 2 or lookup["bad"]["test_candidate"]:
        raise AssertionError(f"Strict-low failure must block test candidate status: {lookup['bad']}")
    if not lookup["good"]["gate_strict_low_bias"] or not lookup["good"]["test_candidate"]:
        raise AssertionError(f"Strict-low passing candidate should be allowed by gates: {lookup['good']}")


def test_online_current_test_promotion_requires_validation_gate() -> None:
    rows = [
        {
            "label": "bad_mae_only",
            "path": r"runs\overnight_current_fms_goal_0514_120s\bad\val_predictions.csv",
            "status": "ok",
            "mae": "1.60",
            "r2": "0.72",
            "strict_low_signed_bias": "4.00",
            "high12_f1": "0.80",
            "gate_pass_count": "3",
            "test_candidate": "False",
        },
        {
            "label": "good_gate",
            "path": r"runs\overnight_current_fms_goal_0514_120s\good\val_predictions.csv",
            "status": "ok",
            "mae": "1.68",
            "r2": "0.71",
            "strict_low_signed_bias": "2.20",
            "high12_f1": "0.79",
            "gate_pass_count": "4",
            "test_candidate": "True",
        },
    ]
    selected = select_test_promotion_candidates(rows, max_candidates=1)
    if len(selected) != 1 or selected[0]["label"] != "good_gate":
        raise AssertionError(f"Promotion should select only validation-gated candidates: {selected}")
    args = argparse.Namespace(
        python=sys.executable,
        data_dir="DenseFMS/Dataset",
        split="test",
        batch_size=48,
        max_session_points=420,
        calibration_residual_features_path=(
            "reports/overnight_current_fms_goal_0514_120s/calibration_summary_features_train_val.json"
        ),
    )
    command = build_test_promotion_eval_command(args, selected[0])
    flat = " ".join(command).replace("\\", "/")
    required_snippets = [
        "-m src.densefms_forecast.evaluate",
        "--checkpoint runs/overnight_current_fms_goal_0514_120s/good/best.pt",
        "--split test",
        "--split_file runs/overnight_current_fms_goal_0514_120s/good/split.json",
        "--max_session_points 420",
        "--save_predictions",
    ]
    for snippet in required_snippets:
        if snippet not in flat:
            raise AssertionError(f"Promotion command missing {snippet}: {flat}")


def test_online_current_goal_status_audit_counts_real_criteria() -> None:
    current = audit_online_current_goal_row(
        {
            "label": "final_equal4_anchor_guard",
            "mae": "2.0011",
            "r2": "0.6195",
            "strict_low_signed_bias": "3.3103",
            "high8_f1": "0.8720",
            "high12_f1": "0.6698",
        }
    )
    if current["pass_count"] != 0 or current["goal_complete"]:
        raise AssertionError(f"Current best should remain incomplete under strict criteria: {current}")
    c4 = current["criteria"]["C4_HIGH8_HIGH12_F1_RELATIVE"]
    if c4["possible_under_f1"]:
        raise AssertionError(f"High8 +25% F1 should be impossible from current baseline: {c4}")

    good = audit_online_current_goal_row(
        {
            "label": "hypothetical_good",
            "mae": "1.70",
            "r2": "0.80",
            "strict_low_signed_bias": "2.20",
            "high8_f1": "0.90",
            "high12_f1": "0.90",
        },
        high8_baseline_f1=0.60,
        high12_baseline_f1=0.60,
    )
    if good["pass_count"] != 4 or not good["goal_complete"]:
        raise AssertionError(f"Hypothetical good row should complete the goal: {good}")


def test_online_current_goal_resume_pipeline_never_runs_test() -> None:
    args = argparse.Namespace(
        python=sys.executable,
        run_safe_steps=False,
        execute_validation=False,
        cutoff_hour=12,
        disable_cutoff_guard=False,
    )
    steps = build_goal_resume_pipeline_steps(args)
    if not steps:
        raise AssertionError("Resume pipeline should contain planned steps.")
    if any(step["runs_test"] for step in steps):
        raise AssertionError(f"Resume pipeline must not run original test: {steps}")
    validation = steps[0]
    if validation["may_train"]:
        raise AssertionError(f"Default resume pipeline should not train: {validation}")
    flat = [" ".join(str(part) for part in step["command"]) for step in steps]
    required = [
        "scripts/run_online_current_calsummary_earlyfusion_experiments.py",
        "scripts/evaluate_online_current_calsummary_earlyfusion_gates.py",
        "scripts/prepare_online_current_test_promotion.py",
        "scripts/audit_online_current_goal_status.py",
    ]
    for snippet in required:
        if not any(snippet in item for item in flat):
            raise AssertionError(f"Resume pipeline missing step {snippet}: {flat}")

    args.execute_validation = True
    training_steps = build_goal_resume_pipeline_steps(args)
    if not training_steps[0]["may_train"]:
        raise AssertionError(f"--execute_validation should mark only validation step as trainable: {training_steps}")
    if any(step["runs_test"] for step in training_steps):
        raise AssertionError(f"Even execute_validation mode must not run test: {training_steps}")
    after_cutoff = build_goal_resume_cutoff_status(args, now=datetime(2026, 5, 14, 13, 0, 0))
    if not after_cutoff["cutoff_passed"] or not after_cutoff["validation_training_blocked_by_cutoff"]:
        raise AssertionError(f"Execute-validation mode should be blocked after cutoff: {after_cutoff}")
    args.disable_cutoff_guard = True
    disabled = build_goal_resume_cutoff_status(args, now=datetime(2026, 5, 14, 13, 0, 0))
    if disabled["validation_training_blocked_by_cutoff"] or not disabled["validation_training_allowed_by_time"]:
        raise AssertionError(f"Disabling cutoff guard should allow explicit validation execution: {disabled}")


def test_online_current_goal_state_manifest_collects_status() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        goal_json = root / "goal.json"
        gate_csv = root / "gate.csv"
        promotion_csv = root / "promotion.csv"
        pipeline_json = root / "pipeline.json"
        c4_json = root / "c4.json"
        readiness_json = root / "readiness.json"
        goal_json.write_text(
            json.dumps({"label": "best", "pass_count": 0, "goal_complete": False, "criteria": {}}),
            encoding="utf-8",
        )
        gate_csv.write_text(
            "label,status,test_candidate\n"
            "baseline,ok,False\n"
            "candidate,ok,True\n",
            encoding="utf-8",
        )
        promotion_csv.write_text(
            "label,command\n"
            "candidate,python -m src.densefms_forecast.evaluate --split test\n",
            encoding="utf-8",
        )
        pipeline_json.write_text(
            json.dumps(
                {
                    "time_status": {
                        "cutoff_passed": True,
                        "validation_training_allowed_by_time": False,
                    },
                    "results": [{"name": "gate", "status": "ok"}, {"name": "audit", "status": "ok"}],
                }
            ),
            encoding="utf-8",
        )
        c4_json.write_text(
            json.dumps(
                {
                    "both_threshold_feasible": {"f1": False, "false_positive_rate": True},
                    "both_threshold_pass": {"f1": False, "false_positive_rate": False},
                }
            ),
            encoding="utf-8",
        )
        readiness_json.write_text(
            json.dumps(
                {
                    "command_count": 2,
                    "ok": True,
                    "time_status": {
                        "time_check_present": True,
                        "execute_allowed_by_time": False,
                    },
                    "commands": [
                        {
                            "checks": {"config_calibration_seconds_120": True},
                            "paths": {"config_calibration_seconds": 120.0},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            goal_status_json=str(goal_json),
            validation_gate_csv=str(gate_csv),
            test_promotion_csv=str(promotion_csv),
            resume_pipeline_json=str(pipeline_json),
            c4_feasibility_json=str(c4_json),
            earlyfusion_readiness_json=str(readiness_json),
        )
        manifest = build_goal_state_manifest(args)
    if manifest["current_goal_audit"]["pass_count"] != 0 or manifest["current_goal_audit"]["goal_complete"]:
        raise AssertionError(f"Manifest goal status is wrong: {manifest}")
    if manifest["validation_gate"]["test_candidate_count"] != 1:
        raise AssertionError(f"Manifest should count one validation-gated candidate: {manifest}")
    if manifest["test_promotion"]["command_count"] != 1:
        raise AssertionError(f"Manifest should count one promotion command: {manifest}")
    if not manifest["resume_pipeline"]["all_ok"]:
        raise AssertionError(f"Manifest should summarize successful safe pipeline: {manifest}")
    if manifest["resume_pipeline"]["time_status"]["validation_training_allowed_by_time"]:
        raise AssertionError(f"Manifest should preserve cutoff time status: {manifest}")
    if manifest["c4_feasibility"]["f1_feasible"] or not manifest["c4_feasibility"]["false_positive_rate_feasible"]:
        raise AssertionError(f"Manifest should summarize C4 feasibility: {manifest}")
    if "c4_metric_decision" not in manifest["artifacts"]:
        raise AssertionError(f"Manifest should track the C4 metric decision artifact: {manifest}")
    if "c4_metric_decision_json" not in manifest["artifacts"]:
        raise AssertionError(f"Manifest should track the C4 metric decision JSON artifact: {manifest}")
    if not manifest["earlyfusion_readiness"]["ok"] or manifest["earlyfusion_readiness"]["command_count"] != 2:
        raise AssertionError(f"Manifest should summarize early-fusion readiness: {manifest}")
    if not manifest["earlyfusion_readiness"]["time_status"]["time_check_present"]:
        raise AssertionError(f"Manifest should preserve early-fusion command Time Check: {manifest}")
    if "completion_blocker" not in manifest["artifacts"]:
        raise AssertionError(f"Manifest should track the completion blocker artifact: {manifest}")
    if "completion_blocker_json" not in manifest["artifacts"]:
        raise AssertionError(f"Manifest should track the completion blocker JSON artifact: {manifest}")
    if "completion_blocker_verification" not in manifest["artifacts"]:
        raise AssertionError(f"Manifest should track the completion blocker verification artifact: {manifest}")
    if "completion_blocker_verification_json" not in manifest["artifacts"]:
        raise AssertionError(f"Manifest should track the completion blocker verification JSON artifact: {manifest}")
    if "prompt_to_artifact_checklist" not in manifest["artifacts"]:
        raise AssertionError(f"Manifest should track the prompt-to-artifact checklist artifact: {manifest}")
    if "prompt_to_artifact_checklist_json" not in manifest["artifacts"]:
        raise AssertionError(f"Manifest should track the prompt-to-artifact checklist JSON artifact: {manifest}")
    if "prompt_to_artifact_checklist_verification" not in manifest["artifacts"]:
        raise AssertionError(f"Manifest should track the prompt-to-artifact checklist verification artifact: {manifest}")
    if "prompt_to_artifact_checklist_verification_json" not in manifest["artifacts"]:
        raise AssertionError(f"Manifest should track the prompt-to-artifact checklist verification JSON artifact: {manifest}")
    if "cutoff_lock" not in manifest["artifacts"]:
        raise AssertionError(f"Manifest should track the cutoff lock artifact: {manifest}")
    if "cutoff_lock_json" not in manifest["artifacts"]:
        raise AssertionError(f"Manifest should track the cutoff lock JSON artifact: {manifest}")
    if "post_cutoff_process_audit" not in manifest["artifacts"]:
        raise AssertionError(f"Manifest should track the post-cutoff process audit artifact: {manifest}")
    if "post_cutoff_process_audit_json" not in manifest["artifacts"]:
        raise AssertionError(f"Manifest should track the post-cutoff process audit JSON artifact: {manifest}")
    if "next_experiment_plan" not in manifest["artifacts"]:
        raise AssertionError(f"Manifest should track the next experiment plan artifact: {manifest}")


def test_online_current_goal_handoff_verifier_checks_consistency() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest_json = root / "manifest.json"
        audit_json = root / "audit.json"
        gate_csv = root / "gate.csv"
        promotion_csv = root / "promotion.csv"
        pipeline_json = root / "pipeline.json"
        c4_json = root / "c4.json"
        c4_decision = root / "c4_decision.md"
        c4_decision_json = root / "c4_decision.json"
        c4_alternatives_json = root / "c4_alternatives.json"
        readiness_json = root / "readiness.json"
        readme = root / "README.md"
        completion_blocker = root / "completion_blocker.md"
        completion_blocker_json = root / "completion_blocker.json"
        completion_blocker_verification = root / "completion_blocker_verification.md"
        completion_blocker_verification_json = root / "completion_blocker_verification.json"
        prompt_checklist = root / "prompt_checklist.csv"
        prompt_checklist_json = root / "prompt_checklist.json"
        prompt_checklist_verification = root / "prompt_checklist_verification.md"
        prompt_checklist_verification_json = root / "prompt_checklist_verification.json"
        four_candidate_evidence = root / "four_candidate_evidence.md"
        four_candidate_evidence_json = root / "four_candidate_evidence.json"
        cutoff_lock = root / "cutoff_lock.md"
        cutoff_lock_json = root / "cutoff_lock.json"
        process_audit = root / "process_audit.md"
        process_audit_json = root / "process_audit.json"
        next_plan = root / "next_plan.md"
        manifest_json.write_text(
            json.dumps(
                {
                    "current_goal_audit": {"goal_complete": False, "pass_count": 0},
                    "validation_gate": {"test_candidate_count": 0},
                    "test_promotion": {"command_count": 0},
                    "resume_pipeline": {
                        "all_ok": True,
                        "time_status": {
                            "cutoff_passed": True,
                            "validation_training_allowed_by_time": False,
                        },
                    },
                    "c4_feasibility": {"f1_feasible": False},
                    "c4_metric_alternatives": {"any_both_threshold_pass": False},
                    "earlyfusion_readiness": {
                        "ok": True,
                        "time_status": {
                            "time_check_present": True,
                            "execute_allowed_by_time": False,
                        },
                    },
                    "artifacts": {
                        "c4_metric_decision": {
                            "exists": True,
                            "path": str(c4_decision),
                        },
                        "c4_metric_decision_json": {
                            "exists": True,
                            "path": str(c4_decision_json),
                        },
                        "completion_blocker": {
                            "exists": True,
                            "path": str(completion_blocker),
                        },
                        "completion_blocker_json": {
                            "exists": True,
                            "path": str(completion_blocker_json),
                        },
                        "completion_blocker_verification": {
                            "exists": True,
                            "path": str(completion_blocker_verification),
                        },
                        "completion_blocker_verification_json": {
                            "exists": True,
                            "path": str(completion_blocker_verification_json),
                        },
                        "prompt_to_artifact_checklist": {
                            "exists": True,
                            "path": str(prompt_checklist),
                        },
                        "prompt_to_artifact_checklist_json": {
                            "exists": True,
                            "path": str(prompt_checklist_json),
                        },
                        "prompt_to_artifact_checklist_verification": {
                            "exists": True,
                            "path": str(prompt_checklist_verification),
                        },
                        "prompt_to_artifact_checklist_verification_json": {
                            "exists": True,
                            "path": str(prompt_checklist_verification_json),
                        },
                        "four_candidate_evidence": {
                            "exists": True,
                            "path": str(four_candidate_evidence),
                        },
                        "four_candidate_evidence_json": {
                            "exists": True,
                            "path": str(four_candidate_evidence_json),
                        },
                        "cutoff_lock": {
                            "exists": True,
                            "path": str(cutoff_lock),
                        },
                        "cutoff_lock_json": {
                            "exists": True,
                            "path": str(cutoff_lock_json),
                        },
                        "post_cutoff_process_audit": {
                            "exists": True,
                            "path": str(process_audit),
                        },
                        "post_cutoff_process_audit_json": {
                            "exists": True,
                            "path": str(process_audit_json),
                        },
                        "next_experiment_plan": {
                            "exists": True,
                            "path": str(next_plan),
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        audit_json.write_text(json.dumps({"goal_complete": False, "pass_count": 0}), encoding="utf-8")
        gate_csv.write_text("label,test_candidate\nbaseline,False\n", encoding="utf-8")
        promotion_csv.write_text("label,command\n", encoding="utf-8")
        pipeline_json.write_text(
            json.dumps(
                {
                    "time_status": {
                        "cutoff_passed": True,
                        "validation_training_allowed_by_time": False,
                    },
                    "steps": [{"name": "safe", "runs_test": False}],
                    "results": [{"name": "safe", "status": "ok"}],
                }
            ),
            encoding="utf-8",
        )
        c4_json.write_text(json.dumps({"both_threshold_feasible": {"f1": False}}), encoding="utf-8")
        c4_decision.write_text("Do not use F1 +25% for high8.\n", encoding="utf-8")
        c4_decision_json.write_text(
            json.dumps(
                {
                    "goal_complete": False,
                    "c4_status": "unresolved_requires_user_confirmation",
                    "f1_relative_25_percent_targets": {
                        "high8_f1_target": 1.09,
                        "high8_feasible": False,
                    },
                    "decision": {
                        "do_not_claim_c4_pass_without_user_confirmed_metric": True,
                        "recommended_primary_c4": "high8/high12 false-positive-rate 25% reduction with recall non-degradation",
                        "recommended_secondary_c4": "high8/high12 false-negative-rate 25% reduction with precision non-degradation",
                    },
                }
            ),
            encoding="utf-8",
        )
        c4_alternatives_json.write_text(
            json.dumps({"summaries": [{"pass_both_high8_high12": False}]}),
            encoding="utf-8",
        )
        readiness_json.write_text(
            json.dumps(
                {
                    "ok": True,
                    "time_status": {
                        "time_check_present": True,
                        "execute_allowed_by_time": False,
                    },
                    "commands": [
                        {
                            "checks": {"config_calibration_seconds_120": True},
                            "paths": {"config_calibration_seconds": 120.0},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        readme.write_text(
            "reports/overnight_current_fms_goal_0514_120s/goal_state_manifest/goal_state_manifest.md\n"
            "reports/overnight_current_fms_goal_0514_120s/c4_feasibility_audit/c4_feasibility_audit.md\n"
            "reports/overnight_current_fms_goal_0514_120s/c4_metric_alternatives/c4_metric_alternatives.md\n"
            "reports/overnight_current_fms_goal_0514_120s/earlyfusion_readiness/earlyfusion_readiness.md\n",
            encoding="utf-8",
        )
        completion_blocker.write_text(
            "Do not call `update_goal`.\n",
            encoding="utf-8",
        )
        completion_blocker_json.write_text(
            json.dumps(
                {
                    "goal_complete": False,
                    "update_goal_allowed": False,
                    "pass_count": 0,
                    "required_pass_count": 3,
                    "criteria": {
                        "test_mae_le_1_8": {"value": 2.0011},
                        "test_r2_ge_0_75": {"value": 0.6195},
                        "strict_original_low_fms_0_2_signed_bias_le_2_5": {"value": 3.3103},
                        "high8_high12_plus_25_percent": {
                            "high8_f1": 0.8720,
                            "high12_f1": 0.6698,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        completion_blocker_verification.write_text("overall ok: `True`\n", encoding="utf-8")
        completion_blocker_verification_json.write_text(
            json.dumps(
                {
                    "ok": True,
                    "observed": {
                        "goal_complete": False,
                        "update_goal_allowed": False,
                        "pass_count": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        prompt_checklist.write_text(
            "requirement,status\n"
            "achieve 3 of 4 test criteria,not_achieved\n"
            "C1 test MAE <= 1.8,failed\n"
            "C2 test R2 >= 0.75,failed\n"
            "C3 strict original FMS 0-2 signed bias <= +2.5,failed\n"
            "C4 high8/high12 >=25% improvement,failed\n"
            "do not mark goal complete,blocked\n",
            encoding="utf-8",
        )
        prompt_checklist_json.write_text(
            json.dumps(
                {
                    "goal_complete": False,
                    "pass_count": 0,
                    "required_pass_count": 3,
                }
            ),
            encoding="utf-8",
        )
        prompt_checklist_verification.write_text("overall ok: `True`\n", encoding="utf-8")
        prompt_checklist_verification_json.write_text(
            json.dumps(
                {
                    "ok": True,
                    "observed": {
                        "goal_complete": False,
                        "pass_count": 0,
                        "missing_requirements": [],
                        "missing_evidence": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        four_candidate_evidence.write_text("overall ok: `True`\n", encoding="utf-8")
        four_candidate_evidence_json.write_text(
            json.dumps(
                {
                    "ok": True,
                    "required_member_columns": [
                        "member_pred_selected_risk035",
                        "member_pred_risk045",
                        "member_pred_zero_anchor",
                        "member_pred_range_scaled",
                    ],
                    "checks": {
                        "train_val_test_present": True,
                        "all_csvs_ok": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        cutoff_lock.write_text(
            "pass count | 0/4\n"
            "2.0011\n"
            "0.6195\n"
            "+3.3103\n"
            "0.8720\n"
            "0.6698\n"
            "Do not start new full training after the cutoff.\n"
            "Do not execute original test from the resume pipeline.\n"
            "Do not call update_goal; the objective is not achieved.\n",
            encoding="utf-8",
        )
        cutoff_lock_json.write_text(
            json.dumps(
                {
                    "goal_complete": False,
                    "pass_count": 0,
                    "cutoff": {
                        "new_full_training_allowed": False,
                        "original_test_allowed": False,
                    },
                    "metrics": {
                        "test_mae": 2.0011,
                        "test_r2": 0.6195,
                        "strict_original_low_fms_0_2_signed_bias": 3.3103,
                        "high8_f1": 0.8720,
                        "high12_f1": 0.6698,
                    },
                }
            ),
            encoding="utf-8",
        )
        process_audit.write_text(
            "The active goal remains incomplete.\n",
            encoding="utf-8",
        )
        process_audit_json.write_text(
            json.dumps(
                {
                    "cutoff_passed": True,
                    "ok": True,
                    "observed": {
                        "sandbox_ps_training_processes": 0,
                        "windows_tasklist_python_processes": 0,
                        "windows_tasklist_training_processes": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        next_plan.write_text(
            "### Reopen Protocol\n"
            "현재는 사용자가 명시적으로 full training 재개를 지시한 뒤에만 수행한다.\n"
            "Run validation-only early-fusion first.\n"
            "validation-only command 2개를 실행한다.\n"
            "Run original test only if exactly one final-test command is emitted.\n"
            "final-report-only로 1회 실행한다.\n"
            "C4 metric needs clarification before claiming progress.\n",
            encoding="utf-8",
        )
        readme.write_text(
            readme.read_text(encoding="utf-8")
            + "reports/overnight_current_fms_goal_0514_120s_c4_metric_decision.md\n",
            encoding="utf-8",
        )
        readme.write_text(
            readme.read_text(encoding="utf-8")
            + "reports/overnight_current_fms_goal_0514_120s_completion_blocker.md\n",
            encoding="utf-8",
        )
        readme.write_text(
            readme.read_text(encoding="utf-8")
            + "reports/overnight_current_fms_goal_0514_120s/completion_blocker_verification/verification.md\n",
            encoding="utf-8",
        )
        readme.write_text(
            readme.read_text(encoding="utf-8")
            + "reports/overnight_current_fms_goal_0514_120s/prompt_to_artifact_checklist/checklist.csv\n",
            encoding="utf-8",
        )
        readme.write_text(
            readme.read_text(encoding="utf-8")
            + "reports/overnight_current_fms_goal_0514_120s/prompt_to_artifact_checklist/verification.md\n",
            encoding="utf-8",
        )
        readme.write_text(
            readme.read_text(encoding="utf-8")
            + "reports/overnight_current_fms_goal_0514_120s/four_candidate_evidence/four_candidate_evidence.md\n",
            encoding="utf-8",
        )
        readme.write_text(
            readme.read_text(encoding="utf-8")
            + "reports/overnight_current_fms_goal_0514_120s_cutoff_lock.md\n",
            encoding="utf-8",
        )
        readme.write_text(
            readme.read_text(encoding="utf-8")
            + "reports/overnight_current_fms_goal_0514_120s_cutoff_lock.json\n",
            encoding="utf-8",
        )
        readme.write_text(
            readme.read_text(encoding="utf-8")
            + "reports/overnight_current_fms_goal_0514_120s/post_cutoff_process_audit/process_audit.md\n",
            encoding="utf-8",
        )
        readme.write_text(
            readme.read_text(encoding="utf-8")
            + "reports/overnight_current_fms_goal_0514_120s_next_experiment_plan.md\n",
            encoding="utf-8",
        )
        readme.write_text(
            readme.read_text(encoding="utf-8")
            + "검증 범위: cutoff lock, next experiment plan, C1-C4 metric records.\n",
            encoding="utf-8",
        )
        args = argparse.Namespace(
            manifest_json=str(manifest_json),
            audit_json=str(audit_json),
            gate_csv=str(gate_csv),
            promotion_csv=str(promotion_csv),
            pipeline_json=str(pipeline_json),
            c4_feasibility_json=str(c4_json),
            c4_metric_decision=str(c4_decision),
            c4_metric_decision_json=str(c4_decision_json),
            c4_metric_alternatives_json=str(c4_alternatives_json),
            earlyfusion_readiness_json=str(readiness_json),
            readme=str(readme),
            completion_blocker=str(completion_blocker),
            completion_blocker_json=str(completion_blocker_json),
            completion_blocker_verification=str(completion_blocker_verification),
            completion_blocker_verification_json=str(completion_blocker_verification_json),
            prompt_checklist=str(prompt_checklist),
            prompt_checklist_json=str(prompt_checklist_json),
            prompt_checklist_verification=str(prompt_checklist_verification),
            prompt_checklist_verification_json=str(prompt_checklist_verification_json),
            four_candidate_evidence=str(four_candidate_evidence),
            four_candidate_evidence_json=str(four_candidate_evidence_json),
            cutoff_lock=str(cutoff_lock),
            cutoff_lock_json=str(cutoff_lock_json),
            process_audit=str(process_audit),
            process_audit_json=str(process_audit_json),
            next_experiment_plan=str(next_plan),
        )
        result = verify_goal_handoff(args)
    if not result["ok"]:
        raise AssertionError(f"Consistent handoff fixture should pass: {result}")


def test_online_current_c4_feasibility_flags_impossible_f1() -> None:
    rows = [
        {
            "label": "baseline",
            "high8_precision": "0.80",
            "high8_recall": "0.90",
            "high8_f1": "0.872",
            "high8_false_positive_rate": "0.20",
            "high8_false_negative_rate": "0.10",
            "high12_precision": "0.70",
            "high12_recall": "0.70",
            "high12_f1": "0.70",
            "high12_false_positive_rate": "0.10",
            "high12_false_negative_rate": "0.30",
        },
        {
            "label": "candidate",
            "high8_precision": "0.90",
            "high8_recall": "0.95",
            "high8_f1": "0.90",
            "high8_false_positive_rate": "0.10",
            "high8_false_negative_rate": "0.05",
            "high12_precision": "0.90",
            "high12_recall": "0.90",
            "high12_f1": "0.90",
            "high12_false_positive_rate": "0.05",
            "high12_false_negative_rate": "0.10",
        },
    ]
    result = audit_c4_feasibility(
        rows,
        baseline_label="baseline",
        candidate_label="candidate",
        improvement_factor=1.25,
    )
    if result["both_threshold_feasible"]["f1"]:
        raise AssertionError(f"High8 F1 +25% should be infeasible when target exceeds 1: {result}")
    if not result["both_threshold_pass"]["false_positive_rate"]:
        raise AssertionError(f"Candidate should pass 25% FPR reduction for both thresholds: {result}")
    if not result["both_threshold_pass"]["false_negative_rate"]:
        raise AssertionError(f"Candidate should pass 25% FNR reduction for both thresholds: {result}")


def test_online_current_c4_metric_alternatives_compares_existing_predictions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pred_base = root / "baseline_predictions.csv"
        pred_candidate = root / "candidate_predictions.csv"
        pred_base.write_text(
            "target_fms_now,predicted_fms_now\n"
            "0,0.1\n"
            "8,0.2\n"
            "12,0.3\n"
            "14,0.4\n",
            encoding="utf-8",
        )
        pred_candidate.write_text(
            "target_fms_now,predicted_fms_now\n"
            "0,0.1\n"
            "8,8.0\n"
            "12,12.0\n"
            "14,14.0\n",
            encoding="utf-8",
        )
        metrics_path = root / "goal_metrics.csv"
        fields = [
            "label",
            "path",
            "pred_column",
            "high8_precision",
            "high8_recall",
            "high8_f1",
            "high8_false_positive_rate",
            "high8_false_negative_rate",
            "high12_precision",
            "high12_recall",
            "high12_f1",
            "high12_false_positive_rate",
            "high12_false_negative_rate",
        ]
        rows = [
            {
                "label": "baseline",
                "path": str(pred_base),
                "pred_column": "predicted_fms_now",
                "high8_precision": "0.90",
                "high8_recall": "0.90",
                "high8_f1": "0.90",
                "high8_false_positive_rate": "0.40",
                "high8_false_negative_rate": "0.20",
                "high12_precision": "0.60",
                "high12_recall": "0.60",
                "high12_f1": "0.60",
                "high12_false_positive_rate": "0.20",
                "high12_false_negative_rate": "0.30",
            },
            {
                "label": "candidate",
                "path": str(pred_candidate),
                "pred_column": "predicted_fms_now",
                "high8_precision": "0.95",
                "high8_recall": "0.95",
                "high8_f1": "0.95",
                "high8_false_positive_rate": "0.20",
                "high8_false_negative_rate": "0.10",
                "high12_precision": "0.80",
                "high12_recall": "0.80",
                "high12_f1": "0.80",
                "high12_false_positive_rate": "0.10",
                "high12_false_negative_rate": "0.20",
            },
        ]
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            import csv

            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        result = analyze_c4_metric_alternatives(
            metrics_path,
            metrics_path,
            val_baseline_label="baseline",
            test_baseline_label="baseline",
            improvement_factor=1.25,
        )
    summaries = result["summaries"]
    val_candidate = {
        row["metric_family"]: row
        for row in summaries
        if row["split"] == "validation" and row["candidate_label"] == "candidate"
    }
    if val_candidate["f1"]["feasible_both_high8_high12"]:
        raise AssertionError("F1 +25% should be infeasible because high8 baseline F1 is already 0.90.")
    if not val_candidate["false_positive_rate"]["pass_both_high8_high12"]:
        raise AssertionError("Candidate should pass both-threshold FPR reduction.")
    if not val_candidate["false_negative_rate"]["pass_both_high8_high12"]:
        raise AssertionError("Candidate should pass both-threshold FNR reduction.")
    if not np.isfinite(float(val_candidate["auprc"]["high8_candidate_value"])):
        raise AssertionError("AUPRC should be computed from prediction CSVs.")


def test_online_current_earlyfusion_readiness_checks_safety_flags() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / "data"
        data_dir.mkdir()
        config = root / "config.json"
        checkpoint = root / "best.pt"
        features = root / "features.json"
        config.write_text(json.dumps({"data": {"calibration_seconds": 120.0}}), encoding="utf-8")
        for path in (checkpoint, features):
            path.write_text("{}", encoding="utf-8")
        data_s = data_dir.as_posix()
        config_s = config.as_posix()
        checkpoint_s = checkpoint.as_posix()
        features_s = features.as_posix()
        safe_command = (
            f"{sys.executable} -m src.densefms_forecast.train --data_dir {data_s} "
            f"--config {config_s} --init_checkpoint {checkpoint_s} "
            f"--calibration_residual_features_path {features_s} --require_calibration_residual_features "
            "--calibration_summary_fusion_enabled --selection_metric goal_composite.strict120 "
            "--max_session_points 420 --no_test_eval"
        )
        safe = verify_earlyfusion_readiness_command({"run_name": "safe", "command": safe_command})
        if not safe["ok"]:
            raise AssertionError(f"Safe early-fusion command should pass readiness: {safe}")

        unsafe_command = safe_command.replace(" --no_test_eval", "") + " --split test"
        unsafe = verify_earlyfusion_readiness_command({"run_name": "unsafe", "command": unsafe_command})
        if unsafe["ok"]:
            raise AssertionError(f"Unsafe command should fail readiness: {unsafe}")
        if unsafe["checks"]["uses_no_test_eval"] or unsafe["checks"]["does_not_request_test_split"]:
            raise AssertionError(f"Readiness should catch test-safety violations: {unsafe}")

        bad_config = root / "config90.json"
        bad_config.write_text(json.dumps({"data": {"calibration_seconds": 90.0}}), encoding="utf-8")
        wrong_calib_command = safe_command.replace(config_s, bad_config.as_posix())
        wrong_calib = verify_earlyfusion_readiness_command({"run_name": "wrong_calib", "command": wrong_calib_command})
        if wrong_calib["ok"] or wrong_calib["checks"]["config_calibration_seconds_120"]:
            raise AssertionError(f"Readiness should reject non-120s calibration config: {wrong_calib}")


def test_optimization_runner_partial_summary() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "opt_stage2_transformer_full_static_level_calib90_h5_rw10"
        run_dir.mkdir(parents=True)
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "status": "interrupted",
                    "last_completed_epoch": 3,
                    "best_epoch_so_far": 2,
                    "best_val_MAE_so_far": 2.5,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "stage": "stage2",
                    "condition": "full_static",
                    "recent_encoder": "transformer",
                    "calibration_seconds": 90,
                    "horizon_seconds": 5,
                    "recent_window_seconds": 10,
                    "loss_mode": "level_only",
                }
            ),
            encoding="utf-8",
        )
        row = summarize_run(run_dir)
        if row["status"] != "interrupted" or row["best_val_MAE_so_far"] != 2.5:
            raise AssertionError(f"Partial summary failed: {row}")
        if row["recent_encoder"] != "transformer" or row["condition"] != "full_static":
            raise AssertionError(f"Partial run config summary failed: {row}")


def main() -> None:
    tests = [
        test_loss_equivalence,
        test_low_target_weight_changes_level_loss,
        test_trend_loss_correctness_and_level_distinction,
        test_mask_correctness,
        test_no_legacy_multihead_loss,
        test_leakage_forward_signature_and_post_calib_independence,
        test_static_off_compatibility,
        test_static_on_requirement_and_shape,
        test_static_full_model_forward,
        test_participant_balanced_session_weights_equalize_participant_totals,
        test_static_scaler_train_only_and_gender_encoding,
        test_mssq_column_inference_and_static_dim,
        test_mssq_scaler_and_missing_behavior,
        test_scenario_static_feature_encoding,
        test_fms_fixed_0_20_normalization,
        test_thresholded_trend_plateau_small_wiggle,
        test_thresholded_trend_true_up_movement,
        test_thresholded_trend_false_movement,
        test_thresholded_trend_multistep,
        test_thresholded_trend_mask_correctness,
        test_seconds_to_steps_conversion,
        test_head_channel_mode_masks_excluded_channels_after_normalization,
        test_online_current_goal_selection_metrics_use_original_low_bin,
        test_online_current_goal_summary_reports_strict_low_bin,
        test_online_current_recovery_low_suppressor_gate_target,
        test_target_shift_variable_horizon,
        test_calibration_input_variable_length_no_post_calib_fms,
        test_recent_window_no_future_leakage_when_calib_shorter_than_recent,
        test_recent_transformer_forward_and_mask_shape,
        test_tcn_transformer_prediction_masks_match,
        test_recent_transformer_no_future_leakage_when_calib_shorter_than_recent,
        test_lc_sa_tcnformer_forward_shape_and_start_index,
        test_lc_sa_target_shift_uses_prediction_start,
        test_lc_sa_anchor_policy_correctness,
        test_lc_sa_sparse_anchor_uses_latest_finite_observation,
        test_fms_context_mode_tensor_policy,
        test_lc_sa_start_only_uses_window_start_fms,
        test_prediction_csv_start_fms_fallback_metadata,
        test_sparse_anchor_mode_requires_full_fms_and_uses_calibration_history,
        test_lc_sa_no_recent_motion_future_leakage,
        test_lc_sa_dynamic_dilation_schedule_and_rf,
        test_new_long_target_model_family_forward_shapes,
        test_new_long_target_multi_horizon_forward_shape,
        test_goal_dynamic_heads_dual_head_and_motion_features,
        test_predict_delta_from_anchor_backcompat_sets_delta_head,
        test_lc_sa_calibration_end_delta_uses_calibration_only_fms,
        test_calib_init_state_forecaster_uses_only_calibration_fms,
        test_calib_init_self_delta_uses_predicted_current_not_real_post_fms,
        test_calib_init_recent_start_delta_uses_synthetic_anchor_not_real_fms,
        test_calib_init_rollin_start_delta_uses_lagged_predictions_not_real_fms,
        test_calib_init_session_summary_context_is_calibration_only,
        test_teacher_future_distillation_with_start_only_teacher_keeps_student_deployable,
        test_teacher_checkpoint_loader_and_compatibility_guard,
        test_calib_init_stream_time_multiscale_no_future_leakage,
        test_new_long_target_recent_motion_no_future_leakage,
        test_lc_sa_full_search_dry_run_command_generation,
        test_long_target_search_dry_run_command_generation,
        test_online_current_integrated_improvement_dry_run_command_generation,
        test_zero_anchor_ablation_dry_run_command_generation,
        test_valid_prediction_mask_count,
        test_max_session_points_caps_target_extent,
        test_current_sequence_times_and_common_window_spec,
        test_time_config_backward_compatibility,
        test_config_2x2,
        test_config_static_full,
        test_online_current_risk_target_shift_and_rise_labels,
        test_online_current_high_risk_onset_labels_exclude_plateau,
        test_online_current_high_risk_current_or_future_labels_include_plateau,
        test_online_current_risk_model_forward_shape_and_calibration_only_fms,
        test_online_current_risk_can_disable_risk_head,
        test_online_current_risk_deep_tcn_forward_shape_and_cap_config,
        test_online_current_risk_zero_calibration_motion_only_forward,
        test_online_current_risk_adaptive_calibration_tcn_dilation,
        test_online_current_risk_recent_motion_no_future_leakage,
        test_online_current_risk_prefix_streamer_matches_full_forward,
        test_online_current_risk_loss_smoke_and_anchor_policy,
        test_online_current_risk_lds_weighting_train_targets_only,
        test_online_current_risk_transition_weighting_targets_only,
        test_online_current_risk_trajectory_shape_loss,
        test_online_current_risk_teacher_current_distillation_alignment,
        test_online_current_risk_final_warning_rapid_only_excludes_high_fms,
        test_online_current_risk_future_delta_event_auxiliary_paths,
        test_online_current_causal_dynamics_feature_bank,
        test_online_current_remaining_heads_and_multitimescale_features,
        test_online_current_calibration_transformer_modes,
        test_online_current_remaining_experiments_dry_run_command_generation,
        test_online_current_calsummary_earlyfusion_dry_run_command_generation,
        test_online_current_calsummary_earlyfusion_cutoff_guard,
        test_online_current_calsummary_earlyfusion_gate_requires_strict_low,
        test_online_current_test_promotion_requires_validation_gate,
        test_online_current_goal_status_audit_counts_real_criteria,
        test_online_current_goal_resume_pipeline_never_runs_test,
        test_online_current_goal_state_manifest_collects_status,
        test_online_current_goal_handoff_verifier_checks_consistency,
        test_online_current_c4_feasibility_flags_impossible_f1,
        test_online_current_c4_metric_alternatives_compares_existing_predictions,
        test_online_current_earlyfusion_readiness_checks_safety_flags,
        test_optimization_runner_partial_summary,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
