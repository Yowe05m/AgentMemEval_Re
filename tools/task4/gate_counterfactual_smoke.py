"""Fail-closed admission gate for the real-service counterfactual-memory smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from agentmemeval.evaluation.degeneracy import (
    evaluate_behavior_health,
    evaluate_execution_health,
)
from agentmemeval.prompts.decision import BASE_SYSTEM_PROMPT, PROMPT_TEMPLATE_VERSION
from agentmemeval.prompts.experience_update import EXPERIENCE_UPDATE_PROMPT

READY_STATUS = "ready_to_start_campaign_p_v7_pilot"
REQUIRED_ARTIFACTS = (
    "resolved_config.yaml",
    "manifest.json",
    "events.jsonl",
    "hand_summaries.jsonl",
    "metrics.json",
    "protocol_audit.json",
    "checkpoint_generalization.json",
    "report.md",
    "experiment_result.json",
)
BEHAVIOR_THRESHOLDS = {
    "min_vpip": 0.02,
    "max_fold_rate": 0.98,
    "min_voluntary_participation_hands": 1,
    "max_all_in_hand_rate": 0.5,
    "max_bust_hand_rate": 0.5,
    "max_single_hand_reward_activity_share": 0.75,
    "single_hand_reward_activity_diagnostic_only": True,
    "max_empty_retrieval_rate": 0.98,
    "max_structural_signature_share": 0.95,
}
UNCONDITIONAL_TERMS = ("总是", "必然", "优先果断", "应优先在", "无论何种")
INSUFFICIENT_EVIDENCE_PATTERN = re.compile(
    r"证据不足|样本(?:量)?不足|缺乏.+数据|无法(?:直接)?验证|无.+数据"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    audit = build_gate(
        Path(args.run_dir).resolve(),
        expected_code_sha=str(args.expected_code_sha),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), **audit}, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == READY_STATUS else 2


def build_gate(run_dir: Path, *, expected_code_sha: str) -> dict[str, Any]:
    blockers: list[str] = []
    missing = [
        name
        for name in REQUIRED_ARTIFACTS
        if not (run_dir / name).is_file() or (run_dir / name).stat().st_size < 1
    ]
    if missing:
        blockers.append(f"missing required artifacts: {missing}")
        return _audit(run_dir, expected_code_sha, blockers)

    config = _read_yaml(run_dir / "resolved_config.yaml")
    manifest = _read_json(run_dir / "manifest.json")
    metrics = _read_json(run_dir / "metrics.json")
    protocol = _read_json(run_dir / "protocol_audit.json")
    events = _read_jsonl(run_dir / "events.jsonl")
    hands = _read_jsonl(run_dir / "hand_summaries.jsonl")
    experiment = dict(config.get("experiment", {}))
    metadata = dict(manifest.get("metadata", {}))
    code = dict(metadata.get("code", {}))
    prompts = dict(metadata.get("prompts", {}))

    if code.get("commit") != expected_code_sha:
        blockers.append(f"code SHA mismatch: {code.get('commit')}")
    if code.get("dirty") is not False:
        blockers.append("run was created from a dirty worktree")
    if experiment.get("run_mode") != "smoke":
        blockers.append(f"run_mode is not smoke: {experiment.get('run_mode')}")
    if "not_for_main_table" not in str(experiment.get("protocol_label", "")):
        blockers.append("protocol label does not preserve not_for_main_table status")

    expected_prompts = {
        "decision_version": PROMPT_TEMPLATE_VERSION,
        "decision_system_sha256": _text_sha256(BASE_SYSTEM_PROMPT),
        "experience_update_sha256": _text_sha256(EXPERIENCE_UPDATE_PROMPT),
    }
    for key, expected in expected_prompts.items():
        if prompts.get(key) != expected:
            blockers.append(f"{key} mismatch: {prompts.get(key)}")

    execution = evaluate_execution_health(hands, metrics)
    if execution.get("valid") is not True:
        blockers.append("execution health is invalid")
    recorded_execution = protocol.get("execution_health", {})
    if not isinstance(recorded_execution, dict) or recorded_execution.get("valid") is not True:
        blockers.append("protocol audit execution health is invalid")
    action_fallback_count = sum(bool(event.get("fallback_used")) for event in events)
    service_error_count = sum(
        isinstance(event.get("llm"), dict) and bool(event["llm"].get("error"))
        for event in events
    )
    if action_fallback_count:
        blockers.append(f"action fallback events present: {action_fallback_count}")
    if service_error_count:
        blockers.append(f"LLM service error events present: {service_error_count}")

    evaluation_targets = protocol.get("evaluation_target_ids", [])
    if not isinstance(evaluation_targets, list) or not evaluation_targets:
        blockers.append("protocol audit lacks evaluation_target_ids")
        evaluation_targets = []
    behavior = evaluate_behavior_health(
        metrics,
        {
            "behavior_threshold_status": "frozen",
            "behavior_thresholds": BEHAVIOR_THRESHOLDS,
        },
        [str(value) for value in evaluation_targets],
    )
    if behavior.get("status") != "passed":
        blockers.append("one or more evaluation targets fail behavior hard gates")

    revision_audit = _revision_audit(run_dir / "memory_snapshots")
    blockers.extend(revision_audit["blockers"])
    evidence = {
        "required_artifact_sha256": {
            name: _sha256(run_dir / name) for name in REQUIRED_ARTIFACTS
        },
        "prompt_identity": prompts,
        "expected_prompt_identity": expected_prompts,
        "execution_health": execution,
        "recorded_execution_health": recorded_execution,
        "action_fallback_count": action_fallback_count,
        "service_error_count": service_error_count,
        "behavior_health": behavior,
        "revision_audit": revision_audit,
    }
    return _audit(run_dir, expected_code_sha, blockers, evidence)


def _revision_audit(root: Path) -> dict[str, Any]:
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*_final.json")):
        snapshot = _read_json(path)
        mechanism = str(snapshot.get("mechanism", ""))
        payload = dict(snapshot.get("payload", {}))
        if mechanism == "expr":
            expr = payload
        elif mechanism in {"fact_expr_sync", "fact_expr_async"}:
            expr = dict(payload.get("expr", {}))
        else:
            continue
        revisions = list(expr.get("revision_log", []))
        violations: list[dict[str, Any]] = []
        if mechanism in {"expr", "fact_expr_sync"} and revisions:
            first = dict(revisions[0])
            if first.get("keep") is not True:
                violations.append(
                    {
                        "revision_index": 1,
                        "reason": "first single-hand revision did not keep prior document",
                    }
                )
        for index, raw in enumerate(revisions, start=1):
            revision = dict(raw)
            new_md = str(revision.get("new_md", ""))
            terms = [term for term in UNCONDITIONAL_TERMS if term in new_md]
            if terms:
                violations.append(
                    {
                        "revision_index": index,
                        "reason": "unconditional prescription terms",
                        "terms": terms,
                    }
                )
            evidence_note = " ".join(
                [
                    str(revision.get("calibration_note", "")),
                    str(revision.get("self_check", "")),
                ]
            )
            if (
                revision.get("keep") is not True
                and INSUFFICIENT_EVIDENCE_PATTERN.search(evidence_note)
            ):
                violations.append(
                    {
                        "revision_index": index,
                        "reason": (
                            "revision changed policy while self-audit reports "
                            "insufficient evidence"
                        ),
                    }
                )
        row = {
            "path": str(path),
            "mechanism": mechanism,
            "revision_count": len(revisions),
            "violation_count": len(violations),
            "violations": violations,
        }
        rows.append(row)
        if violations:
            blockers.append(f"{path.name} has {len(violations)} revision violations")
    if len(rows) != 6:
        blockers.append(f"expected 6 final experiential snapshots, found {len(rows)}")
    return {
        "experiential_snapshot_count": len(rows),
        "snapshots": rows,
        "blockers": blockers,
    }


def _audit(
    run_dir: Path,
    expected_code_sha: str,
    blockers: list[str],
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "task4_counterfactual_smoke_gate_v1",
        "run_dir": str(run_dir),
        "expected_code_sha": expected_code_sha,
        "evidence": evidence or {},
        "blockers": blockers,
        "status": READY_STATUS if not blockers else "no_go",
    }


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
