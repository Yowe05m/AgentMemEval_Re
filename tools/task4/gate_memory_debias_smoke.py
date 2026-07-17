"""Fail-closed gate for the real-service V5 factual-memory repair smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

EXPECTED_PROMPT_VERSION = "2026-07-17-v5-nonnormative-memory"
FACTUAL_MECHANISMS = {"fact", "fact_expr_sync", "fact_expr_async"}
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
    return 0 if audit["status"] == "ready_to_start_v5_pilot" else 2


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
    metadata = dict(manifest.get("metadata", {}))
    code = dict(metadata.get("code", {}))
    prompts = dict(metadata.get("prompts", {}))
    agent_config = dict(config.get("agent", {}))

    if code.get("commit") != expected_code_sha:
        blockers.append(f"code SHA mismatch: {code.get('commit')}")
    if code.get("dirty") is not False:
        blockers.append("run was created from a dirty worktree")
    if prompts.get("decision_version") != EXPECTED_PROMPT_VERSION:
        blockers.append(
            f"decision prompt version mismatch: {prompts.get('decision_version')}"
        )
    if agent_config.get("reject_single_preflop_fold") is not True:
        blockers.append("resolved config does not enable single-preflop-fold rejection")

    execution = dict(protocol.get("execution_health", {}))
    if execution.get("valid") is not True:
        blockers.append("execution health is invalid")
    events = _read_jsonl(run_dir / "events.jsonl")
    action_fallbacks = sum(bool(event.get("fallback_used")) for event in events)
    service_errors = sum(
        isinstance(event.get("llm"), dict) and bool(event["llm"].get("error"))
        for event in events
    )
    if action_fallbacks:
        blockers.append(f"action fallback events present: {action_fallbacks}")
    if service_errors:
        blockers.append(f"LLM service error events present: {service_errors}")

    snapshot_audit = _snapshot_audit(run_dir / "memory_snapshots")
    blockers.extend(snapshot_audit["blockers"])
    behavior_audit = _systematic_collapse_audit(metrics)
    blockers.extend(behavior_audit["blockers"])

    evidence = {
        "required_artifact_sha256": {
            name: _sha256(run_dir / name) for name in REQUIRED_ARTIFACTS
        },
        "execution_health": execution,
        "event_count": len(events),
        "action_fallback_count": action_fallbacks,
        "service_error_count": service_errors,
        "snapshot_audit": snapshot_audit,
        "systematic_collapse_audit": behavior_audit,
        "prompt_version": prompts.get("decision_version"),
        "resolved_reject_single_preflop_fold": agent_config.get(
            "reject_single_preflop_fold"
        ),
    }
    return _audit(run_dir, expected_code_sha, blockers, evidence)


def _snapshot_audit(root: Path) -> dict[str, Any]:
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*_final.json")):
        snapshot = _read_json(path)
        mechanism = str(snapshot.get("mechanism", ""))
        if mechanism not in FACTUAL_MECHANISMS:
            continue
        payload = dict(snapshot.get("payload", {}))
        fact = payload if mechanism == "fact" else dict(payload.get("fact", {}))
        records = list(fact.get("records", []))
        intent_count = 0
        for record in records:
            source = dict(record.get("source", {})) if isinstance(record, dict) else {}
            decisions = source.get("decisions", [])
            if isinstance(decisions, list):
                intent_count += sum(
                    isinstance(decision, dict) and "intent" in decision
                    for decision in decisions
                )
        row = {
            "path": str(path),
            "mechanism": mechanism,
            "schema_version": fact.get("schema_version"),
            "reject_single_preflop_fold": fact.get("reject_single_preflop_fold"),
            "record_count": len(records),
            "persisted_decision_intent_count": intent_count,
            "single_preflop_fold_rejection_count": dict(
                fact.get("admission_counts", {})
            ).get("reason:single_preflop_fold_low_information", 0),
        }
        rows.append(row)
        if int(row["schema_version"] or 0) < 5:
            blockers.append(f"{path.name} factual snapshot schema is older than 5")
        if row["reject_single_preflop_fold"] is not True:
            blockers.append(f"{path.name} does not persist repair admission policy")
        if intent_count:
            blockers.append(f"{path.name} persists model reason intent as factual evidence")
    if len(rows) != 6:
        blockers.append(f"expected 6 final factual snapshots, found {len(rows)}")
    return {"factual_snapshot_count": len(rows), "snapshots": rows, "blockers": blockers}


def _systematic_collapse_audit(metrics: dict[str, Any]) -> dict[str, Any]:
    primary = dict(metrics.get("primary_metrics", {}))
    stage_per_agent = dict(primary.get("stage_per_agent", {}))
    stage_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for stage, per_agent in sorted(stage_per_agent.items()):
        if not isinstance(per_agent, dict):
            continue
        factual = [
            (str(agent_id), values)
            for agent_id, values in sorted(per_agent.items())
            if str(agent_id).startswith(("fact_", "sync_", "async_"))
            and isinstance(values, dict)
        ]
        collapsed = [
            {
                "agent_id": agent_id,
                "vpip": float(values.get("vpip", 0.0)),
                "fold_rate": float(values.get("fold_rate", 0.0)),
            }
            for agent_id, values in factual
            if float(values.get("vpip", 0.0)) < 0.02
            or float(values.get("fold_rate", 0.0)) > 0.98
        ]
        stage_rows.append(
            {
                "stage": str(stage),
                "factual_agent_count": len(factual),
                "collapsed_agent_count": len(collapsed),
                "collapsed_agents": collapsed,
            }
        )
        if factual and len(collapsed) * 2 >= len(factual):
            blockers.append(
                f"{stage} shows systematic factual-memory collapse: "
                f"{len(collapsed)}/{len(factual)} agents"
            )
    if not stage_rows:
        blockers.append("metrics contain no stage-level behavior evidence")
    return {
        "definition": "at least half of factual-memory agents violate VPIP/fold hard caps",
        "stages": stage_rows,
        "blockers": blockers,
    }


def _audit(
    run_dir: Path,
    expected_code_sha: str,
    blockers: list[str],
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "task4_memory_debias_smoke_gate_v1",
        "run_dir": str(run_dir),
        "expected_code_sha": expected_code_sha,
        "evidence": evidence or {},
        "blockers": blockers,
        "status": "ready_to_start_v5_pilot" if not blockers else "no_go",
    }


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
