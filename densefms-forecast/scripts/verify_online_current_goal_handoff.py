"""Verify consistency of strict 120s goal handoff artifacts.

This verifier reads existing reports only. It does not train, run validation,
or evaluate the original test set.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping


DEFAULT_BASE_DIR = Path("reports/overnight_current_fms_goal_0514_120s")
DEFAULT_MANIFEST_JSON = DEFAULT_BASE_DIR / "goal_state_manifest/goal_state_manifest.json"
DEFAULT_AUDIT_JSON = DEFAULT_BASE_DIR / "goal_status_audit/goal_status_audit.json"
DEFAULT_GATE_CSV = DEFAULT_BASE_DIR / "calsummary_earlyfusion_gate_eval/validation_gate_metrics.csv"
DEFAULT_PROMOTION_CSV = DEFAULT_BASE_DIR / "test_promotion_commands/test_promotion_commands.csv"
DEFAULT_PIPELINE_JSON = DEFAULT_BASE_DIR / "resume_pipeline/resume_pipeline.json"
DEFAULT_C4_JSON = DEFAULT_BASE_DIR / "c4_feasibility_audit/c4_feasibility_audit.json"
DEFAULT_C4_METRIC_DECISION = Path("reports/overnight_current_fms_goal_0514_120s_c4_metric_decision.md")
DEFAULT_C4_METRIC_DECISION_JSON = Path("reports/overnight_current_fms_goal_0514_120s_c4_metric_decision.json")
DEFAULT_C4_ALTERNATIVES_JSON = DEFAULT_BASE_DIR / "c4_metric_alternatives/c4_metric_alternatives.json"
DEFAULT_READINESS_JSON = DEFAULT_BASE_DIR / "earlyfusion_readiness/earlyfusion_readiness.json"
DEFAULT_README = Path("reports/README.md")
DEFAULT_COMPLETION_BLOCKER = Path("reports/overnight_current_fms_goal_0514_120s_completion_blocker.md")
DEFAULT_COMPLETION_BLOCKER_JSON = Path("reports/overnight_current_fms_goal_0514_120s_completion_blocker.json")
DEFAULT_COMPLETION_BLOCKER_VERIFICATION = DEFAULT_BASE_DIR / "completion_blocker_verification/verification.md"
DEFAULT_COMPLETION_BLOCKER_VERIFICATION_JSON = DEFAULT_BASE_DIR / "completion_blocker_verification/verification.json"
DEFAULT_PROMPT_CHECKLIST = DEFAULT_BASE_DIR / "prompt_to_artifact_checklist/checklist.csv"
DEFAULT_PROMPT_CHECKLIST_JSON = DEFAULT_BASE_DIR / "prompt_to_artifact_checklist/checklist.json"
DEFAULT_PROMPT_CHECKLIST_VERIFICATION = DEFAULT_BASE_DIR / "prompt_to_artifact_checklist/verification.md"
DEFAULT_PROMPT_CHECKLIST_VERIFICATION_JSON = DEFAULT_BASE_DIR / "prompt_to_artifact_checklist/verification.json"
DEFAULT_FOUR_CANDIDATE_EVIDENCE = DEFAULT_BASE_DIR / "four_candidate_evidence/four_candidate_evidence.md"
DEFAULT_FOUR_CANDIDATE_EVIDENCE_JSON = DEFAULT_BASE_DIR / "four_candidate_evidence/four_candidate_evidence.json"
DEFAULT_CUTOFF_LOCK = Path("reports/overnight_current_fms_goal_0514_120s_cutoff_lock.md")
DEFAULT_CUTOFF_LOCK_JSON = Path("reports/overnight_current_fms_goal_0514_120s_cutoff_lock.json")
DEFAULT_PROCESS_AUDIT = DEFAULT_BASE_DIR / "post_cutoff_process_audit/process_audit.md"
DEFAULT_PROCESS_AUDIT_JSON = DEFAULT_BASE_DIR / "post_cutoff_process_audit/process_audit.json"
DEFAULT_NEXT_EXPERIMENT_PLAN = Path("reports/overnight_current_fms_goal_0514_120s_next_experiment_plan.md")
DEFAULT_OUT_DIR = DEFAULT_BASE_DIR / "handoff_verification"


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z %z")


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = _read_json(Path(args.manifest_json))
    audit = _read_json(Path(args.audit_json))
    gate_rows = _read_csv(Path(args.gate_csv))
    promotion_rows = _read_csv(Path(args.promotion_csv))
    pipeline = _read_json(Path(args.pipeline_json))
    c4 = _read_json(Path(args.c4_feasibility_json))
    c4_decision_path = Path(getattr(args, "c4_metric_decision", DEFAULT_C4_METRIC_DECISION))
    c4_decision_json_path = Path(getattr(args, "c4_metric_decision_json", DEFAULT_C4_METRIC_DECISION_JSON))
    c4_decision_text = c4_decision_path.read_text(encoding="utf-8") if c4_decision_path.exists() else ""
    c4_decision = _read_json(c4_decision_json_path)
    c4_alternatives = _read_json(
        Path(getattr(args, "c4_metric_alternatives_json", DEFAULT_C4_ALTERNATIVES_JSON))
    )
    readiness = _read_json(Path(args.earlyfusion_readiness_json))
    readiness_commands = readiness.get("commands", [])
    readme_text = Path(args.readme).read_text(encoding="utf-8") if Path(args.readme).exists() else ""
    completion_blocker_path = Path(getattr(args, "completion_blocker", DEFAULT_COMPLETION_BLOCKER))
    completion_blocker_json_path = Path(getattr(args, "completion_blocker_json", DEFAULT_COMPLETION_BLOCKER_JSON))
    completion_blocker_verification_path = Path(
        getattr(args, "completion_blocker_verification", DEFAULT_COMPLETION_BLOCKER_VERIFICATION)
    )
    completion_blocker_verification_json_path = Path(
        getattr(args, "completion_blocker_verification_json", DEFAULT_COMPLETION_BLOCKER_VERIFICATION_JSON)
    )
    prompt_checklist_path = Path(getattr(args, "prompt_checklist", DEFAULT_PROMPT_CHECKLIST))
    prompt_checklist_json_path = Path(getattr(args, "prompt_checklist_json", DEFAULT_PROMPT_CHECKLIST_JSON))
    prompt_checklist_verification_path = Path(
        getattr(args, "prompt_checklist_verification", DEFAULT_PROMPT_CHECKLIST_VERIFICATION)
    )
    prompt_checklist_verification_json_path = Path(
        getattr(args, "prompt_checklist_verification_json", DEFAULT_PROMPT_CHECKLIST_VERIFICATION_JSON)
    )
    four_candidate_evidence_path = Path(getattr(args, "four_candidate_evidence", DEFAULT_FOUR_CANDIDATE_EVIDENCE))
    four_candidate_evidence_json_path = Path(
        getattr(args, "four_candidate_evidence_json", DEFAULT_FOUR_CANDIDATE_EVIDENCE_JSON)
    )
    cutoff_lock_path = Path(getattr(args, "cutoff_lock", DEFAULT_CUTOFF_LOCK))
    cutoff_lock_json_path = Path(getattr(args, "cutoff_lock_json", DEFAULT_CUTOFF_LOCK_JSON))
    process_audit_path = Path(getattr(args, "process_audit", DEFAULT_PROCESS_AUDIT))
    process_audit_json_path = Path(getattr(args, "process_audit_json", DEFAULT_PROCESS_AUDIT_JSON))
    next_plan_path = Path(getattr(args, "next_experiment_plan", DEFAULT_NEXT_EXPERIMENT_PLAN))
    completion_blocker_text = completion_blocker_path.read_text(encoding="utf-8") if completion_blocker_path.exists() else ""
    completion_blocker_json = _read_json(completion_blocker_json_path)
    completion_blocker_verification_json = _read_json(completion_blocker_verification_json_path)
    prompt_checklist_text = prompt_checklist_path.read_text(encoding="utf-8") if prompt_checklist_path.exists() else ""
    prompt_checklist_json = _read_json(prompt_checklist_json_path)
    prompt_checklist_verification_json = _read_json(prompt_checklist_verification_json_path)
    four_candidate_evidence_json = _read_json(four_candidate_evidence_json_path)
    cutoff_lock_text = cutoff_lock_path.read_text(encoding="utf-8") if cutoff_lock_path.exists() else ""
    cutoff_lock_json = _read_json(cutoff_lock_json_path)
    process_audit_text = process_audit_path.read_text(encoding="utf-8") if process_audit_path.exists() else ""
    process_audit_json = _read_json(process_audit_json_path)
    next_plan_text = next_plan_path.read_text(encoding="utf-8") if next_plan_path.exists() else ""

    manifest_goal = manifest.get("current_goal_audit", {}) if isinstance(manifest, dict) else {}
    manifest_gate = manifest.get("validation_gate", {}) if isinstance(manifest, dict) else {}
    manifest_promotion = manifest.get("test_promotion", {}) if isinstance(manifest, dict) else {}
    manifest_pipeline = manifest.get("resume_pipeline", {}) if isinstance(manifest, dict) else {}
    manifest_c4 = manifest.get("c4_feasibility", {}) if isinstance(manifest, dict) else {}
    manifest_c4_alternatives = manifest.get("c4_metric_alternatives", {}) if isinstance(manifest, dict) else {}
    manifest_readiness = manifest.get("earlyfusion_readiness", {}) if isinstance(manifest, dict) else {}
    manifest_readiness_time_status = manifest_readiness.get("time_status", {}) if isinstance(manifest_readiness, dict) else {}
    manifest_artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    manifest_c4_decision = manifest_artifacts.get("c4_metric_decision", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_c4_decision_json = manifest_artifacts.get("c4_metric_decision_json", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_completion_blocker = manifest_artifacts.get("completion_blocker", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_completion_blocker_json = manifest_artifacts.get("completion_blocker_json", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_completion_blocker_verification = manifest_artifacts.get("completion_blocker_verification", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_completion_blocker_verification_json = manifest_artifacts.get("completion_blocker_verification_json", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_prompt_checklist = manifest_artifacts.get("prompt_to_artifact_checklist", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_prompt_checklist_json = manifest_artifacts.get("prompt_to_artifact_checklist_json", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_prompt_checklist_verification = manifest_artifacts.get("prompt_to_artifact_checklist_verification", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_prompt_checklist_verification_json = manifest_artifacts.get("prompt_to_artifact_checklist_verification_json", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_four_candidate_evidence = manifest_artifacts.get("four_candidate_evidence", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_four_candidate_evidence_json = manifest_artifacts.get("four_candidate_evidence_json", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_cutoff_lock = manifest_artifacts.get("cutoff_lock", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_cutoff_lock_json = manifest_artifacts.get("cutoff_lock_json", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_process_audit = manifest_artifacts.get("post_cutoff_process_audit", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_process_audit_json = manifest_artifacts.get("post_cutoff_process_audit_json", {}) if isinstance(manifest_artifacts, dict) else {}
    manifest_next_plan = manifest_artifacts.get("next_experiment_plan", {}) if isinstance(manifest_artifacts, dict) else {}
    gate_candidate_count = sum(1 for row in gate_rows if _truthy(row.get("test_candidate", False)))
    promotion_command_count = sum(1 for row in promotion_rows if str(row.get("command", "")).strip())
    pipeline_results = pipeline.get("results", []) if isinstance(pipeline, dict) else []
    pipeline_steps = pipeline.get("steps", []) if isinstance(pipeline, dict) else []
    pipeline_time_status = pipeline.get("time_status", {}) if isinstance(pipeline, dict) else {}
    c4_alt_summaries = c4_alternatives.get("summaries", []) if isinstance(c4_alternatives, dict) else []
    c4_alt_any_pass = any(_truthy(row.get("pass_both_high8_high12", False)) for row in c4_alt_summaries)

    checks = {
        "manifest_exists": Path(args.manifest_json).exists(),
        "audit_exists": Path(args.audit_json).exists(),
        "gate_csv_exists": Path(args.gate_csv).exists(),
        "promotion_csv_exists": Path(args.promotion_csv).exists(),
        "pipeline_json_exists": Path(args.pipeline_json).exists(),
        "c4_json_exists": Path(args.c4_feasibility_json).exists(),
        "c4_metric_decision_exists": c4_decision_path.exists(),
        "c4_metric_decision_json_exists": c4_decision_json_path.exists(),
        "c4_alternatives_json_exists": Path(getattr(args, "c4_metric_alternatives_json", DEFAULT_C4_ALTERNATIVES_JSON)).exists(),
        "readiness_json_exists": Path(args.earlyfusion_readiness_json).exists(),
        "completion_blocker_exists": completion_blocker_path.exists(),
        "completion_blocker_json_exists": completion_blocker_json_path.exists(),
        "completion_blocker_verification_exists": completion_blocker_verification_path.exists(),
        "completion_blocker_verification_json_exists": completion_blocker_verification_json_path.exists(),
        "prompt_checklist_exists": prompt_checklist_path.exists(),
        "prompt_checklist_json_exists": prompt_checklist_json_path.exists(),
        "prompt_checklist_verification_exists": prompt_checklist_verification_path.exists(),
        "prompt_checklist_verification_json_exists": prompt_checklist_verification_json_path.exists(),
        "four_candidate_evidence_exists": four_candidate_evidence_path.exists(),
        "four_candidate_evidence_json_exists": four_candidate_evidence_json_path.exists(),
        "cutoff_lock_exists": cutoff_lock_path.exists(),
        "cutoff_lock_json_exists": cutoff_lock_json_path.exists(),
        "process_audit_exists": process_audit_path.exists(),
        "process_audit_json_exists": process_audit_json_path.exists(),
        "next_experiment_plan_exists": next_plan_path.exists(),
        "audit_manifest_goal_complete_match": manifest_goal.get("goal_complete") == audit.get("goal_complete"),
        "audit_manifest_pass_count_match": manifest_goal.get("pass_count") == audit.get("pass_count"),
        "gate_candidate_count_match": int(manifest_gate.get("test_candidate_count", -1)) == int(gate_candidate_count),
        "promotion_command_count_match": int(manifest_promotion.get("command_count", -1)) == int(promotion_command_count),
        "pipeline_all_ok_match": bool(manifest_pipeline.get("all_ok")) == (
            bool(pipeline_results) and all(row.get("status") == "ok" for row in pipeline_results)
        ),
        "pipeline_time_status_match": manifest_pipeline.get("time_status", {}) == pipeline_time_status,
        "c4_manifest_f1_feasible_match": bool(manifest_c4.get("f1_feasible")) == bool(c4.get("both_threshold_feasible", {}).get("f1", False)),
        "c4_alternatives_any_pass_match": bool(manifest_c4_alternatives.get("any_both_threshold_pass")) == bool(c4_alt_any_pass),
        "readiness_manifest_ok_match": bool(manifest_readiness.get("ok")) == bool(readiness.get("ok", False)),
        "readiness_time_status_match": manifest_readiness_time_status == readiness.get("time_status", {}),
        "readiness_commands_confirm_120s_calibration": (
            bool(readiness_commands)
            and all(
                row.get("checks", {}).get("config_calibration_seconds_120") is True
                and row.get("paths", {}).get("config_calibration_seconds") == 120.0
                for row in readiness_commands
            )
        ),
        "manifest_tracks_c4_metric_decision": bool(manifest_c4_decision.get("exists", False)) == c4_decision_path.exists(),
        "manifest_tracks_c4_metric_decision_json": bool(manifest_c4_decision_json.get("exists", False)) == c4_decision_json_path.exists(),
        "manifest_tracks_completion_blocker": bool(manifest_completion_blocker.get("exists", False)) == completion_blocker_path.exists(),
        "manifest_tracks_completion_blocker_json": bool(manifest_completion_blocker_json.get("exists", False)) == completion_blocker_json_path.exists(),
        "manifest_tracks_completion_blocker_verification": bool(manifest_completion_blocker_verification.get("exists", False)) == completion_blocker_verification_path.exists(),
        "manifest_tracks_completion_blocker_verification_json": bool(manifest_completion_blocker_verification_json.get("exists", False)) == completion_blocker_verification_json_path.exists(),
        "manifest_tracks_prompt_checklist": bool(manifest_prompt_checklist.get("exists", False)) == prompt_checklist_path.exists(),
        "manifest_tracks_prompt_checklist_json": bool(manifest_prompt_checklist_json.get("exists", False)) == prompt_checklist_json_path.exists(),
        "manifest_tracks_prompt_checklist_verification": bool(manifest_prompt_checklist_verification.get("exists", False)) == prompt_checklist_verification_path.exists(),
        "manifest_tracks_prompt_checklist_verification_json": bool(manifest_prompt_checklist_verification_json.get("exists", False)) == prompt_checklist_verification_json_path.exists(),
        "manifest_tracks_four_candidate_evidence": bool(manifest_four_candidate_evidence.get("exists", False)) == four_candidate_evidence_path.exists(),
        "manifest_tracks_four_candidate_evidence_json": bool(manifest_four_candidate_evidence_json.get("exists", False)) == four_candidate_evidence_json_path.exists(),
        "manifest_tracks_cutoff_lock": bool(manifest_cutoff_lock.get("exists", False)) == cutoff_lock_path.exists(),
        "manifest_tracks_cutoff_lock_json": bool(manifest_cutoff_lock_json.get("exists", False)) == cutoff_lock_json_path.exists(),
        "manifest_tracks_process_audit": bool(manifest_process_audit.get("exists", False)) == process_audit_path.exists(),
        "manifest_tracks_process_audit_json": bool(manifest_process_audit_json.get("exists", False)) == process_audit_json_path.exists(),
        "manifest_tracks_next_experiment_plan": bool(manifest_next_plan.get("exists", False)) == next_plan_path.exists(),
        "c4_decision_marks_unresolved": (
            c4_decision.get("goal_complete") is False
            and c4_decision.get("c4_status") == "unresolved_requires_user_confirmation"
            and c4_decision.get("decision", {}).get("do_not_claim_c4_pass_without_user_confirmed_metric") is True
        ),
        "c4_decision_rejects_high8_f1_plus25": (
            c4_decision.get("f1_relative_25_percent_targets", {}).get("high8_f1_target") == 1.09
            and c4_decision.get("f1_relative_25_percent_targets", {}).get("high8_feasible") is False
            and "Do not use F1 +25% for high8" in c4_decision_text
        ),
        "c4_decision_recommends_feasible_metrics": (
            "false-positive-rate 25% reduction" in c4_decision.get("decision", {}).get("recommended_primary_c4", "")
            and "false-negative-rate 25% reduction" in c4_decision.get("decision", {}).get("recommended_secondary_c4", "")
        ),
        "completion_blocker_marks_update_goal_disallowed": (
            completion_blocker_json.get("goal_complete") is False
            and completion_blocker_json.get("update_goal_allowed") is False
            and completion_blocker_json.get("pass_count") == 0
            and completion_blocker_json.get("required_pass_count") == 3
            and "Do not call `update_goal`" in completion_blocker_text
        ),
        "completion_blocker_records_current_metrics": (
            completion_blocker_json.get("criteria", {}).get("test_mae_le_1_8", {}).get("value") == 2.0011
            and completion_blocker_json.get("criteria", {}).get("test_r2_ge_0_75", {}).get("value") == 0.6195
            and completion_blocker_json.get("criteria", {}).get("strict_original_low_fms_0_2_signed_bias_le_2_5", {}).get("value") == 3.3103
            and completion_blocker_json.get("criteria", {}).get("high8_high12_plus_25_percent", {}).get("high8_f1") == 0.8720
            and completion_blocker_json.get("criteria", {}).get("high8_high12_plus_25_percent", {}).get("high12_f1") == 0.6698
        ),
        "completion_blocker_verification_ok": (
            completion_blocker_verification_json.get("ok") is True
            and completion_blocker_verification_json.get("observed", {}).get("goal_complete") is False
            and completion_blocker_verification_json.get("observed", {}).get("update_goal_allowed") is False
            and completion_blocker_verification_json.get("observed", {}).get("pass_count") == 0
        ),
        "prompt_checklist_marks_goal_not_achieved": (
            prompt_checklist_json.get("goal_complete") is False
            and prompt_checklist_json.get("pass_count") == 0
            and prompt_checklist_json.get("required_pass_count") == 3
            and "achieve 3 of 4 test criteria,not_achieved" in prompt_checklist_text
        ),
        "prompt_checklist_covers_objective_criteria": all(
            token in prompt_checklist_text
            for token in [
                "C1 test MAE <= 1.8",
                "C2 test R2 >= 0.75",
                "C3 strict original FMS 0-2 signed bias <= +2.5",
                "C4 high8/high12 >=25% improvement",
                "do not mark goal complete",
            ]
        ),
        "prompt_checklist_verification_ok": (
            prompt_checklist_verification_json.get("ok") is True
            and prompt_checklist_verification_json.get("observed", {}).get("goal_complete") is False
            and prompt_checklist_verification_json.get("observed", {}).get("pass_count") == 0
            and prompt_checklist_verification_json.get("observed", {}).get("missing_requirements") == []
            and prompt_checklist_verification_json.get("observed", {}).get("missing_evidence") == []
        ),
        "four_candidate_evidence_ok": (
            four_candidate_evidence_json.get("ok") is True
            and set(four_candidate_evidence_json.get("required_member_columns", []))
            == {
                "member_pred_selected_risk035",
                "member_pred_risk045",
                "member_pred_zero_anchor",
                "member_pred_range_scaled",
            }
            and four_candidate_evidence_json.get("checks", {}).get("train_val_test_present") is True
            and four_candidate_evidence_json.get("checks", {}).get("all_csvs_ok") is True
        ),
        "cutoff_lock_blocks_training_after_cutoff": "Do not start new full training after the cutoff" in cutoff_lock_text,
        "cutoff_lock_blocks_original_test": "Do not execute original test" in cutoff_lock_text,
        "cutoff_lock_marks_goal_incomplete": "objective is not achieved" in cutoff_lock_text,
        "cutoff_lock_records_zero_pass_count": "pass count | 0/4" in cutoff_lock_text,
        "cutoff_lock_records_current_mae": "2.0011" in cutoff_lock_text,
        "cutoff_lock_records_current_r2": "0.6195" in cutoff_lock_text,
        "cutoff_lock_records_current_low_bias": "+3.3103" in cutoff_lock_text,
        "cutoff_lock_records_current_high8_f1": "0.8720" in cutoff_lock_text,
        "cutoff_lock_records_current_high12_f1": "0.6698" in cutoff_lock_text,
        "cutoff_lock_json_marks_goal_incomplete": cutoff_lock_json.get("goal_complete") is False,
        "cutoff_lock_json_records_zero_pass_count": cutoff_lock_json.get("pass_count") == 0,
        "cutoff_lock_json_blocks_training": cutoff_lock_json.get("cutoff", {}).get("new_full_training_allowed") is False,
        "cutoff_lock_json_blocks_original_test": cutoff_lock_json.get("cutoff", {}).get("original_test_allowed") is False,
        "cutoff_lock_json_records_current_metrics": (
            cutoff_lock_json.get("metrics", {}).get("test_mae") == 2.0011
            and cutoff_lock_json.get("metrics", {}).get("test_r2") == 0.6195
            and cutoff_lock_json.get("metrics", {}).get("strict_original_low_fms_0_2_signed_bias") == 3.3103
            and cutoff_lock_json.get("metrics", {}).get("high8_f1") == 0.8720
            and cutoff_lock_json.get("metrics", {}).get("high12_f1") == 0.6698
        ),
        "process_audit_marks_cutoff_passed": process_audit_json.get("cutoff_passed") is True,
        "process_audit_reports_no_training_processes": (
            process_audit_json.get("ok") is True
            and process_audit_json.get("observed", {}).get("sandbox_ps_training_processes") == 0
            and process_audit_json.get("observed", {}).get("windows_tasklist_python_processes") == 0
            and process_audit_json.get("observed", {}).get("windows_tasklist_training_processes") == 0
            and "active goal remains incomplete" in process_audit_text
        ),
        "next_plan_uses_validation_only_resume": "validation-only" in next_plan_text,
        "next_plan_requires_test_promotion_gate": "exactly one final-test command" in next_plan_text,
        "next_plan_documents_reopen_protocol": (
            "Reopen Protocol" in next_plan_text
            and "사용자가 명시적으로 full training 재개" in next_plan_text
            and "validation-only command 2개" in next_plan_text
            and "final-report-only로 1회 실행" in next_plan_text
        ),
        "next_plan_flags_c4_metric_clarification": "C4" in next_plan_text and "metric" in next_plan_text.lower(),
        "pipeline_never_runs_test": all(not _truthy(step.get("runs_test", False)) for step in pipeline_steps),
        "readme_points_to_manifest": "goal_state_manifest/goal_state_manifest.md" in readme_text,
        "readme_points_to_c4": "c4_feasibility_audit/c4_feasibility_audit.md" in readme_text,
        "readme_points_to_c4_decision": "overnight_current_fms_goal_0514_120s_c4_metric_decision.md" in readme_text,
        "readme_points_to_c4_alternatives": "c4_metric_alternatives/c4_metric_alternatives.md" in readme_text,
        "readme_points_to_readiness": "earlyfusion_readiness/earlyfusion_readiness.md" in readme_text,
        "readme_points_to_completion_blocker": "overnight_current_fms_goal_0514_120s_completion_blocker.md" in readme_text,
        "readme_points_to_completion_blocker_verification": "completion_blocker_verification/verification.md" in readme_text,
        "readme_points_to_prompt_checklist": "prompt_to_artifact_checklist/checklist.csv" in readme_text,
        "readme_points_to_prompt_checklist_verification": "prompt_to_artifact_checklist/verification.md" in readme_text,
        "readme_points_to_four_candidate_evidence": "four_candidate_evidence/four_candidate_evidence.md" in readme_text,
        "readme_points_to_cutoff_lock": "overnight_current_fms_goal_0514_120s_cutoff_lock.md" in readme_text,
        "readme_points_to_cutoff_lock_json": "overnight_current_fms_goal_0514_120s_cutoff_lock.json" in readme_text,
        "readme_points_to_process_audit": "post_cutoff_process_audit/process_audit.md" in readme_text,
        "readme_points_to_next_experiment_plan": "overnight_current_fms_goal_0514_120s_next_experiment_plan.md" in readme_text,
        "readme_documents_handoff_scope": (
            "검증 범위" in readme_text
            and "cutoff lock" in readme_text
            and "next experiment plan" in readme_text
            and "C1-C4" in readme_text
        ),
    }
    return {
        "created_at": _now(),
        "ok": all(bool(value) for value in checks.values()),
        "checks": checks,
        "observed": {
            "audit_goal_complete": audit.get("goal_complete"),
            "audit_pass_count": audit.get("pass_count"),
            "gate_candidate_count": gate_candidate_count,
            "promotion_command_count": promotion_command_count,
            "pipeline_result_count": len(pipeline_results),
            "pipeline_cutoff_passed": pipeline_time_status.get("cutoff_passed"),
            "pipeline_validation_training_allowed_by_time": pipeline_time_status.get("validation_training_allowed_by_time"),
            "c4_f1_feasible": c4.get("both_threshold_feasible", {}).get("f1"),
            "c4_decision_status": c4_decision.get("c4_status"),
            "c4_alternatives_any_pass": c4_alt_any_pass,
            "earlyfusion_readiness_ok": readiness.get("ok"),
            "earlyfusion_command_time_check_present": readiness.get("time_status", {}).get("time_check_present"),
            "earlyfusion_execute_allowed_by_time": readiness.get("time_status", {}).get("execute_allowed_by_time"),
            "earlyfusion_commands_confirm_120s_calibration": (
                all(
                    row.get("checks", {}).get("config_calibration_seconds_120") is True
                    and row.get("paths", {}).get("config_calibration_seconds") == 120.0
                    for row in readiness_commands
                )
                if readiness_commands
                else False
            ),
            "completion_blocker_exists": completion_blocker_path.exists(),
            "completion_blocker_update_goal_allowed": completion_blocker_json.get("update_goal_allowed"),
            "completion_blocker_verification_ok": completion_blocker_verification_json.get("ok"),
            "prompt_checklist_exists": prompt_checklist_path.exists(),
            "prompt_checklist_goal_complete": prompt_checklist_json.get("goal_complete"),
            "prompt_checklist_verification_ok": prompt_checklist_verification_json.get("ok"),
            "four_candidate_evidence_ok": four_candidate_evidence_json.get("ok"),
            "cutoff_lock_exists": cutoff_lock_path.exists(),
            "cutoff_lock_json_exists": cutoff_lock_json_path.exists(),
            "process_audit_exists": process_audit_path.exists(),
            "next_experiment_plan_exists": next_plan_path.exists(),
            "cutoff_lock_blocks_training_after_cutoff": "Do not start new full training after the cutoff" in cutoff_lock_text,
            "cutoff_lock_records_zero_pass_count": "pass count | 0/4" in cutoff_lock_text,
            "cutoff_lock_records_current_high8_f1": "0.8720" in cutoff_lock_text,
            "cutoff_lock_records_current_high12_f1": "0.6698" in cutoff_lock_text,
            "cutoff_lock_json_blocks_training": cutoff_lock_json.get("cutoff", {}).get("new_full_training_allowed") is False,
            "process_audit_no_training_processes": process_audit_json.get("ok") is True,
            "next_plan_requires_test_promotion_gate": "exactly one final-test command" in next_plan_text,
            "next_plan_documents_reopen_protocol": "Reopen Protocol" in next_plan_text,
        },
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Strict 120s Goal Handoff Verification",
        "",
        f"작성일: {result.get('created_at', '')}",
        "",
        f"- overall ok: `{result.get('ok')}`",
        "",
        "## Observed",
        "",
    ]
    for key, value in result.get("observed", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks", "", "| check | pass |", "|---|---|"])
    for key, value in result.get("checks", {}).items():
        lines.append(f"| {key} | {bool(value)} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "If `overall ok` is false, inspect handoff artifacts before resuming experiments.",
            "This verifier does not imply the research goal is complete.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_json", default=str(DEFAULT_MANIFEST_JSON))
    parser.add_argument("--audit_json", default=str(DEFAULT_AUDIT_JSON))
    parser.add_argument("--gate_csv", default=str(DEFAULT_GATE_CSV))
    parser.add_argument("--promotion_csv", default=str(DEFAULT_PROMOTION_CSV))
    parser.add_argument("--pipeline_json", default=str(DEFAULT_PIPELINE_JSON))
    parser.add_argument("--c4_feasibility_json", default=str(DEFAULT_C4_JSON))
    parser.add_argument("--c4_metric_decision", default=str(DEFAULT_C4_METRIC_DECISION))
    parser.add_argument("--c4_metric_decision_json", default=str(DEFAULT_C4_METRIC_DECISION_JSON))
    parser.add_argument("--c4_metric_alternatives_json", default=str(DEFAULT_C4_ALTERNATIVES_JSON))
    parser.add_argument("--earlyfusion_readiness_json", default=str(DEFAULT_READINESS_JSON))
    parser.add_argument("--readme", default=str(DEFAULT_README))
    parser.add_argument("--completion_blocker", default=str(DEFAULT_COMPLETION_BLOCKER))
    parser.add_argument("--completion_blocker_json", default=str(DEFAULT_COMPLETION_BLOCKER_JSON))
    parser.add_argument("--completion_blocker_verification", default=str(DEFAULT_COMPLETION_BLOCKER_VERIFICATION))
    parser.add_argument("--completion_blocker_verification_json", default=str(DEFAULT_COMPLETION_BLOCKER_VERIFICATION_JSON))
    parser.add_argument("--prompt_checklist", default=str(DEFAULT_PROMPT_CHECKLIST))
    parser.add_argument("--prompt_checklist_json", default=str(DEFAULT_PROMPT_CHECKLIST_JSON))
    parser.add_argument("--prompt_checklist_verification", default=str(DEFAULT_PROMPT_CHECKLIST_VERIFICATION))
    parser.add_argument("--prompt_checklist_verification_json", default=str(DEFAULT_PROMPT_CHECKLIST_VERIFICATION_JSON))
    parser.add_argument("--four_candidate_evidence", default=str(DEFAULT_FOUR_CANDIDATE_EVIDENCE))
    parser.add_argument("--four_candidate_evidence_json", default=str(DEFAULT_FOUR_CANDIDATE_EVIDENCE_JSON))
    parser.add_argument("--cutoff_lock", default=str(DEFAULT_CUTOFF_LOCK))
    parser.add_argument("--cutoff_lock_json", default=str(DEFAULT_CUTOFF_LOCK_JSON))
    parser.add_argument("--process_audit", default=str(DEFAULT_PROCESS_AUDIT))
    parser.add_argument("--process_audit_json", default=str(DEFAULT_PROCESS_AUDIT_JSON))
    parser.add_argument("--next_experiment_plan", default=str(DEFAULT_NEXT_EXPERIMENT_PLAN))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--fail_on_error", action="store_true")
    args = parser.parse_args()

    result = verify(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "handoff_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_report(out_dir / "handoff_verification.md", result)
    print(json.dumps(result, indent=2))
    if args.fail_on_error and not result["ok"]:
        raise SystemExit("Handoff verification failed.")


if __name__ == "__main__":
    main()
