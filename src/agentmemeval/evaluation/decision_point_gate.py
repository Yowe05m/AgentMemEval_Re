"""Fail-closed gate for the TASK4 V8 decision-point true-service smoke."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

FACTUAL_MECHANISMS = {"fact", "fact_expr_sync", "fact_expr_async"}
EXPECTED_RETRIEVAL_UNIT = "decision_point_max_v1"
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


def build_decision_point_smoke_gate(
    run_dir: str | Path,
    *,
    expected_code_sha: str,
    expected_train_hands: int,
    expected_test_hands: int,
) -> dict[str, Any]:
    """Audit complete artifacts, execution invariants, and decision-point evidence."""

    root = Path(run_dir).resolve()
    blockers: list[str] = []
    missing = [
        name
        for name in REQUIRED_ARTIFACTS
        if not (root / name).is_file() or (root / name).stat().st_size < 1
    ]
    if missing:
        blockers.append(f"missing required artifacts: {missing}")
        return _result(root, expected_code_sha, blockers, {})

    config = _read_yaml(root / "resolved_config.yaml")
    manifest = _read_json(root / "manifest.json")
    protocol = _read_json(root / "protocol_audit.json")
    events = _read_jsonl(root / "events.jsonl")
    hands = _read_jsonl(root / "hand_summaries.jsonl")

    code = dict(dict(manifest.get("metadata", {})).get("code", {}))
    if code.get("commit") != expected_code_sha:
        blockers.append(f"code SHA mismatch: {code.get('commit')}")
    if code.get("dirty") is not False:
        blockers.append("run was created from a dirty worktree")

    agent = dict(config.get("agent", {}))
    if agent.get("retrieval_unit") != EXPECTED_RETRIEVAL_UNIT:
        blockers.append(f"resolved retrieval_unit mismatch: {agent.get('retrieval_unit')}")
    experiment = dict(config.get("experiment", {}))
    if int(experiment.get("train_hands", -1)) != expected_train_hands:
        blockers.append("resolved train_hands mismatch")
    if int(experiment.get("test_hands", -1)) != expected_test_hands:
        blockers.append("resolved test_hands mismatch")

    execution = dict(protocol.get("execution_health", {}))
    if execution.get("valid") is not True:
        blockers.append("execution health is invalid")
    for field in (
        "fallback_count",
        "memory_revision_fallback_count",
        "reward_conservation_violation_count",
        "stack_conservation_violation_count",
    ):
        if int(execution.get(field, -1)) != 0:
            blockers.append(f"execution {field} is not zero: {execution.get(field)}")

    hand_audit = _audit_hands(hands, expected_train_hands, expected_test_hands)
    blockers.extend(hand_audit["blockers"])
    event_audit = _audit_events(events)
    blockers.extend(event_audit["blockers"])
    snapshot_audit = _audit_snapshots(root / "memory_snapshots", config)
    blockers.extend(snapshot_audit["blockers"])

    evidence = {
        "required_artifact_sha256": {
            name: _sha256(root / name) for name in REQUIRED_ARTIFACTS
        },
        "execution_health": execution,
        "hand_audit": hand_audit,
        "event_audit": event_audit,
        "snapshot_audit": snapshot_audit,
    }
    return _result(root, expected_code_sha, blockers, evidence)


def _audit_hands(
    hands: list[dict[str, Any]], expected_train: int, expected_test: int
) -> dict[str, Any]:
    blockers: list[str] = []
    stages = Counter(str(row.get("stage", "")) for row in hands)
    if stages.get("train", 0) != expected_train:
        blockers.append(f"train hand count mismatch: {stages.get('train', 0)}")
    if stages.get("test", 0) != expected_test:
        blockers.append(f"test hand count mismatch: {stages.get('test', 0)}")
    ids = [str(row.get("hand_id", "")) for row in hands]
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if key and count > 1)
    if len(ids) != len(set(ids)) or any(not key for key in ids):
        blockers.append("hand IDs are missing or duplicated")
    test_updates = [
        str(row.get("hand_id", ""))
        for row in hands
        if row.get("stage") == "test" and row.get("memory_updated") is not False
    ]
    if test_updates:
        blockers.append(f"test hands updated memory: {test_updates}")
    reward_violations: list[str] = []
    stack_violations: list[str] = []
    negative_stacks: list[str] = []
    for row in hands:
        hand_id = str(row.get("hand_id", ""))
        rewards = dict(row.get("rewards", {}))
        starting = dict(row.get("starting_stacks", {}))
        final = dict(row.get("final_stacks", {}))
        if sum(float(value) for value in rewards.values()) != 0:
            reward_violations.append(hand_id)
        if sum(float(value) for value in starting.values()) != sum(
            float(value) for value in final.values()
        ):
            stack_violations.append(hand_id)
        if any(float(value) < 0 for value in final.values()):
            negative_stacks.append(hand_id)
    if reward_violations:
        blockers.append(f"reward conservation violations: {reward_violations}")
    if stack_violations:
        blockers.append(f"stack conservation violations: {stack_violations}")
    if negative_stacks:
        blockers.append(f"negative final stacks: {negative_stacks}")
    return {
        "stage_counts": dict(stages),
        "duplicate_hand_ids": duplicate_ids,
        "test_memory_update_hand_ids": test_updates,
        "reward_conservation_violation_hand_ids": reward_violations,
        "stack_conservation_violation_hand_ids": stack_violations,
        "negative_stack_hand_ids": negative_stacks,
        "blockers": blockers,
    }


def _audit_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    action_fallbacks = sum(bool(event.get("fallback_used")) for event in events)
    service_errors = sum(
        isinstance(event.get("llm"), dict) and bool(event["llm"].get("error"))
        for event in events
    )
    scored = 0
    malformed: list[str] = []
    for line_number, event in enumerate(events, 1):
        context = event.get("memory_context")
        if not isinstance(context, dict):
            continue
        metadata = context.get("metadata")
        if not isinstance(metadata, dict):
            continue
        for score in metadata.get("retrieval_scores", []):
            if not isinstance(score, dict):
                malformed.append(f"line {line_number}: non-object score")
                continue
            scored += 1
            if (
                score.get("retrieval_unit") != EXPECTED_RETRIEVAL_UNIT
                or not isinstance(score.get("matched_decision_index"), int)
                or not str(score.get("matched_phase", "")).strip()
            ):
                malformed.append(f"line {line_number}: incomplete matched decision")
    if action_fallbacks:
        blockers.append(f"action fallback events present: {action_fallbacks}")
    if service_errors:
        blockers.append(f"LLM service error events present: {service_errors}")
    if scored < 1:
        blockers.append("no scored decision-point retrieval was observed")
    if malformed:
        blockers.append(f"malformed decision-point retrieval scores: {malformed[:20]}")
    return {
        "event_count": len(events),
        "action_fallback_count": action_fallbacks,
        "service_error_count": service_errors,
        "retrieval_score_count": scored,
        "malformed_retrieval_scores": malformed,
        "blockers": blockers,
    }


def _audit_snapshots(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    roster = list(dict(config.get("experiment", {})).get("agent_roster", []))
    expected = sum(
        isinstance(row, dict) and str(row.get("mechanism")) in FACTUAL_MECHANISMS
        for row in roster
    )
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    record_count = 0
    decision_count = 0
    for path in sorted(root.glob("*_final.json")):
        snapshot = _read_json(path)
        mechanism = str(snapshot.get("mechanism", ""))
        if mechanism not in FACTUAL_MECHANISMS:
            continue
        payload = dict(snapshot.get("payload", {}))
        fact = payload if mechanism == "fact" else dict(payload.get("fact", {}))
        records = list(fact.get("records", []))
        row_decisions = 0
        malformed = 0
        for record in records:
            source = dict(record.get("source", {})) if isinstance(record, dict) else {}
            decisions = source.get("decisions", [])
            if not isinstance(decisions, list):
                malformed += 1
                continue
            for decision in decisions:
                row_decisions += 1
                if (
                    not isinstance(decision, dict)
                    or not str(decision.get("retrieval_query", "")).strip()
                    or not isinstance(decision.get("features"), list)
                ):
                    malformed += 1
        rows.append(
            {
                "path": str(path),
                "mechanism": mechanism,
                "schema_version": fact.get("schema_version"),
                "retrieval_unit": fact.get("retrieval_unit"),
                "record_count": len(records),
                "decision_view_count": row_decisions,
                "malformed_decision_view_count": malformed,
            }
        )
        record_count += len(records)
        decision_count += row_decisions
        if int(fact.get("schema_version", 0)) < 6:
            blockers.append(f"{path.name} factual snapshot schema is older than 6")
        if fact.get("retrieval_unit") != EXPECTED_RETRIEVAL_UNIT:
            blockers.append(f"{path.name} retrieval_unit mismatch")
        if malformed:
            blockers.append(f"{path.name} has malformed decision views: {malformed}")
    if len(rows) != expected:
        blockers.append(f"expected {expected} final factual snapshots, found {len(rows)}")
    if record_count < 1 or decision_count < 1:
        blockers.append("final snapshots contain no versioned decision evidence")
    return {
        "expected_factual_snapshot_count": expected,
        "factual_snapshot_count": len(rows),
        "record_count": record_count,
        "decision_view_count": decision_count,
        "snapshots": rows,
        "blockers": blockers,
    }


def _result(
    run_dir: Path,
    expected_code_sha: str,
    blockers: list[str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "task4_decision_point_smoke_gate_v1",
        "classification": "engineering_smoke_not_for_paper",
        "run_dir": str(run_dir),
        "expected_code_sha": expected_code_sha,
        "evidence": evidence,
        "blockers": blockers,
        "status": "ready_to_start_v8_calibration_pilot" if not blockers else "no_go",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
