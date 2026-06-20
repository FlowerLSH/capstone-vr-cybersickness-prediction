"""Build a manifest for the strict 120s online-current goal state.

The manifest is a compact handoff artifact. It reads existing reports only; it
does not train, evaluate the test set, or modify model outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


DEFAULT_BASE_DIR = Path("reports/overnight_current_fms_goal_0514_120s")
DEFAULT_OUT_DIR = DEFAULT_BASE_DIR / "goal_state_manifest"


KEY_ARTIFACTS = {
    "final_report": "reports/overnight_current_fms_goal_0514_120s_final_report.md",
    "completion_audit": "reports/overnight_current_fms_goal_0514_120s_completion_audit.md",
    "completion_blocker": "reports/overnight_current_fms_goal_0514_120s_completion_blocker.md",
    "completion_blocker_json": "reports/overnight_current_fms_goal_0514_120s_completion_blocker.json",
    "completion_blocker_verification": "reports/overnight_current_fms_goal_0514_120s/completion_blocker_verification/verification.md",
    "completion_blocker_verification_json": "reports/overnight_current_fms_goal_0514_120s/completion_blocker_verification/verification.json",
    "prompt_to_artifact_checklist": "reports/overnight_current_fms_goal_0514_120s/prompt_to_artifact_checklist/checklist.csv",
    "prompt_to_artifact_checklist_json": "reports/overnight_current_fms_goal_0514_120s/prompt_to_artifact_checklist/checklist.json",
    "prompt_to_artifact_checklist_verification": "reports/overnight_current_fms_goal_0514_120s/prompt_to_artifact_checklist/verification.md",
    "prompt_to_artifact_checklist_verification_json": "reports/overnight_current_fms_goal_0514_120s/prompt_to_artifact_checklist/verification.json",
    "four_candidate_evidence": "reports/overnight_current_fms_goal_0514_120s/four_candidate_evidence/four_candidate_evidence.md",
    "four_candidate_evidence_json": "reports/overnight_current_fms_goal_0514_120s/four_candidate_evidence/four_candidate_evidence.json",
    "resume_dashboard": "reports/overnight_current_fms_goal_0514_120s_resume_dashboard.md",
    "cutoff_lock": "reports/overnight_current_fms_goal_0514_120s_cutoff_lock.md",
    "cutoff_lock_json": "reports/overnight_current_fms_goal_0514_120s_cutoff_lock.json",
    "post_cutoff_process_audit": "reports/overnight_current_fms_goal_0514_120s/post_cutoff_process_audit/process_audit.md",
    "post_cutoff_process_audit_json": "reports/overnight_current_fms_goal_0514_120s/post_cutoff_process_audit/process_audit.json",
    "next_experiment_plan": "reports/overnight_current_fms_goal_0514_120s_next_experiment_plan.md",
    "attempt_log": "ONLINE_CURRENT_ATTEMPT_LOG.md",
    "goal_status_audit_md": "reports/overnight_current_fms_goal_0514_120s/goal_status_audit/goal_status_audit.md",
    "goal_status_audit_json": "reports/overnight_current_fms_goal_0514_120s/goal_status_audit/goal_status_audit.json",
    "validation_gate_report": "reports/overnight_current_fms_goal_0514_120s/calsummary_earlyfusion_gate_eval/validation_gate_report.md",
    "validation_gate_csv": "reports/overnight_current_fms_goal_0514_120s/calsummary_earlyfusion_gate_eval/validation_gate_metrics.csv",
    "test_promotion_report": "reports/overnight_current_fms_goal_0514_120s/test_promotion_commands/test_promotion_commands.md",
    "test_promotion_csv": "reports/overnight_current_fms_goal_0514_120s/test_promotion_commands/test_promotion_commands.csv",
    "resume_pipeline_report": "reports/overnight_current_fms_goal_0514_120s/resume_pipeline/resume_pipeline.md",
    "resume_pipeline_json": "reports/overnight_current_fms_goal_0514_120s/resume_pipeline/resume_pipeline.json",
    "earlyfusion_commands": "reports/overnight_current_fms_goal_0514_120s/calsummary_earlyfusion_commands/commands.md",
    "c4_feasibility_report": "reports/overnight_current_fms_goal_0514_120s/c4_feasibility_audit/c4_feasibility_audit.md",
    "c4_feasibility_json": "reports/overnight_current_fms_goal_0514_120s/c4_feasibility_audit/c4_feasibility_audit.json",
    "c4_metric_decision": "reports/overnight_current_fms_goal_0514_120s_c4_metric_decision.md",
    "c4_metric_decision_json": "reports/overnight_current_fms_goal_0514_120s_c4_metric_decision.json",
    "c4_metric_alternatives_report": "reports/overnight_current_fms_goal_0514_120s/c4_metric_alternatives/c4_metric_alternatives.md",
    "c4_metric_alternatives_json": "reports/overnight_current_fms_goal_0514_120s/c4_metric_alternatives/c4_metric_alternatives.json",
    "earlyfusion_readiness_report": "reports/overnight_current_fms_goal_0514_120s/earlyfusion_readiness/earlyfusion_readiness.md",
    "earlyfusion_readiness_json": "reports/overnight_current_fms_goal_0514_120s/earlyfusion_readiness/earlyfusion_readiness.json",
}


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z %z")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def _artifact_status(paths: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for name, raw_path in paths.items():
        path = Path(raw_path)
        rows[name] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
    return rows


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    artifacts = _artifact_status(KEY_ARTIFACTS)
    goal_audit = _read_json(Path(args.goal_status_json), {})
    gate_rows = _read_csv(Path(args.validation_gate_csv))
    promotion_rows = _read_csv(Path(args.test_promotion_csv))
    pipeline = _read_json(Path(args.resume_pipeline_json), {})
    c4 = _read_json(Path(args.c4_feasibility_json), {})
    c4_alternatives = _read_json(
        Path(getattr(args, "c4_metric_alternatives_json", DEFAULT_BASE_DIR / "c4_metric_alternatives/c4_metric_alternatives.json")),
        {},
    )
    readiness = _read_json(Path(args.earlyfusion_readiness_json), {})
    gate_candidates = [row for row in gate_rows if _truthy(row.get("test_candidate", False))]
    pipeline_results = pipeline.get("results", []) if isinstance(pipeline, dict) else []
    pipeline_time_status = pipeline.get("time_status", {}) if isinstance(pipeline, dict) else {}
    c4_feasible = c4.get("both_threshold_feasible", {}) if isinstance(c4, dict) else {}
    c4_pass = c4.get("both_threshold_pass", {}) if isinstance(c4, dict) else {}
    c4_alt_summaries = c4_alternatives.get("summaries", []) if isinstance(c4_alternatives, dict) else []
    c4_alt_passes = [row for row in c4_alt_summaries if _truthy(row.get("pass_both_high8_high12", False))]
    manifest = {
        "created_at": _now(),
        "objective": {
            "calibration_seconds": 120,
            "required_pass_count": 3,
            "criteria": [
                "test MAE <= 1.8",
                "test R2 >= 0.75",
                "strict original 0<=FMS<2 signed bias <= +2.5",
                "high8/high12 prediction >= 25% improvement over current baseline",
            ],
        },
        "current_goal_audit": {
            "label": goal_audit.get("label"),
            "pass_count": goal_audit.get("pass_count"),
            "goal_complete": goal_audit.get("goal_complete"),
            "criteria": goal_audit.get("criteria", {}),
        },
        "validation_gate": {
            "row_count": len(gate_rows),
            "test_candidate_count": len(gate_candidates),
            "test_candidates": [row.get("label", "") for row in gate_candidates],
        },
        "test_promotion": {
            "command_count": len([row for row in promotion_rows if row.get("command")]),
            "commands_present": any(row.get("command") for row in promotion_rows),
        },
        "resume_pipeline": {
            "result_count": len(pipeline_results),
            "all_ok": bool(pipeline_results) and all(row.get("status") == "ok" for row in pipeline_results),
            "time_status": pipeline_time_status,
            "results": pipeline_results,
        },
        "c4_feasibility": {
            "precision_feasible": bool(c4_feasible.get("precision", False)),
            "recall_feasible": bool(c4_feasible.get("recall", False)),
            "f1_feasible": bool(c4_feasible.get("f1", False)),
            "false_positive_rate_feasible": bool(c4_feasible.get("false_positive_rate", False)),
            "false_negative_rate_feasible": bool(c4_feasible.get("false_negative_rate", False)),
            "any_metric_pass": any(bool(value) for value in c4_pass.values()),
        },
        "c4_metric_alternatives": {
            "summary_count": len(c4_alt_summaries),
            "any_both_threshold_pass": bool(c4_alt_passes),
            "passing_rows": [
                {
                    "split": row.get("split"),
                    "candidate_label": row.get("candidate_label"),
                    "metric_family": row.get("metric_family"),
                }
                for row in c4_alt_passes
            ],
        },
        "earlyfusion_readiness": {
            "command_count": int(readiness.get("command_count", 0)) if isinstance(readiness, dict) else 0,
            "ok": bool(readiness.get("ok", False)) if isinstance(readiness, dict) else False,
            "time_status": readiness.get("time_status", {}) if isinstance(readiness, dict) else {},
        },
        "artifacts": artifacts,
        "next_action": (
            "Do not call update_goal. If training is explicitly reopened before cutoff, run validation-only "
            "early-fusion; then gate; then prepare test promotion; run original test only if exactly one "
            "validation-gated command is present."
        ),
    }
    return manifest


def _write_markdown(path: Path, manifest: Mapping[str, Any]) -> None:
    audit = manifest.get("current_goal_audit", {})
    gate = manifest.get("validation_gate", {})
    promotion = manifest.get("test_promotion", {})
    pipeline = manifest.get("resume_pipeline", {})
    c4 = manifest.get("c4_feasibility", {})
    c4_alternatives = manifest.get("c4_metric_alternatives", {})
    readiness = manifest.get("earlyfusion_readiness", {})
    artifacts = manifest.get("artifacts", {})
    lines = [
        "# Strict 120s Goal State Manifest",
        "",
        f"작성일: {manifest.get('created_at', '')}",
        "",
        "## Current Status",
        "",
        f"- label: `{audit.get('label')}`",
        f"- pass count: `{audit.get('pass_count')}/4`",
        f"- goal complete: `{audit.get('goal_complete')}`",
        "",
        "## Gate And Promotion",
        "",
        f"- validation gate rows: `{gate.get('row_count')}`",
        f"- validation-gated test candidates: `{gate.get('test_candidate_count')}`",
        f"- promotion command count: `{promotion.get('command_count')}`",
        "",
        "## Resume Pipeline",
        "",
        f"- result count: `{pipeline.get('result_count')}`",
        f"- all ok: `{pipeline.get('all_ok')}`",
        f"- cutoff passed: `{pipeline.get('time_status', {}).get('cutoff_passed')}`",
        f"- validation training allowed by time: `{pipeline.get('time_status', {}).get('validation_training_allowed_by_time')}`",
        "",
        "## C4 Feasibility",
        "",
        f"- precision feasible: `{c4.get('precision_feasible')}`",
        f"- recall feasible: `{c4.get('recall_feasible')}`",
        f"- F1 feasible: `{c4.get('f1_feasible')}`",
        f"- FPR reduction feasible: `{c4.get('false_positive_rate_feasible')}`",
        f"- FNR reduction feasible: `{c4.get('false_negative_rate_feasible')}`",
        f"- any current metric pass: `{c4.get('any_metric_pass')}`",
        "",
        "## C4 Metric Alternatives",
        "",
        f"- summary count: `{c4_alternatives.get('summary_count')}`",
        f"- any high8+high12 +25% pass: `{c4_alternatives.get('any_both_threshold_pass')}`",
        "",
        "## Early-Fusion Readiness",
        "",
        f"- command count: `{readiness.get('command_count')}`",
        f"- ok: `{readiness.get('ok')}`",
        f"- command report time check present: `{readiness.get('time_status', {}).get('time_check_present')}`",
        f"- execute allowed by time: `{readiness.get('time_status', {}).get('execute_allowed_by_time')}`",
        "",
        "## Key Artifacts",
        "",
        "| name | exists | path |",
        "|---|---|---|",
    ]
    for name, row in artifacts.items():
        lines.append(f"| {name} | {row.get('exists')} | `{row.get('path')}` |")
    lines.extend(
        [
            "",
            "## Next Action",
            "",
            str(manifest.get("next_action", "")),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal_status_json", default=str(DEFAULT_BASE_DIR / "goal_status_audit/goal_status_audit.json"))
    parser.add_argument(
        "--validation_gate_csv",
        default=str(DEFAULT_BASE_DIR / "calsummary_earlyfusion_gate_eval/validation_gate_metrics.csv"),
    )
    parser.add_argument(
        "--test_promotion_csv",
        default=str(DEFAULT_BASE_DIR / "test_promotion_commands/test_promotion_commands.csv"),
    )
    parser.add_argument("--resume_pipeline_json", default=str(DEFAULT_BASE_DIR / "resume_pipeline/resume_pipeline.json"))
    parser.add_argument("--c4_feasibility_json", default=str(DEFAULT_BASE_DIR / "c4_feasibility_audit/c4_feasibility_audit.json"))
    parser.add_argument("--c4_metric_alternatives_json", default=str(DEFAULT_BASE_DIR / "c4_metric_alternatives/c4_metric_alternatives.json"))
    parser.add_argument("--earlyfusion_readiness_json", default=str(DEFAULT_BASE_DIR / "earlyfusion_readiness/earlyfusion_readiness.json"))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    manifest = build_manifest(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "goal_state_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_markdown(out_dir / "goal_state_manifest.md", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
