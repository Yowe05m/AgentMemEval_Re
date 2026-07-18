"""Build an immutable, fail-closed Campaign P gate before Campaign E starts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from agentmemeval.evaluation.pilot import calibrate_behavior_thresholds

REQUIRED_ARTIFACTS = (
    "resolved_config.yaml",
    "manifest.json",
    "hand_summaries.jsonl",
    "metrics.json",
    "protocol_audit.json",
    "checkpoint_generalization.json",
    "report.md",
    "experiment_result.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--expected-max-model-len", type=int, required=True)
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir).resolve()
    output = Path(args.output).resolve()
    audit = build_gate(
        campaign_dir,
        expected_code_sha=str(args.expected_code_sha),
        expected_max_model_len=int(args.expected_max_model_len),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), **audit}, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "ready_to_start_campaign_e" else 2


def build_gate(
    campaign_dir: Path,
    *,
    expected_code_sha: str,
    expected_max_model_len: int,
) -> dict[str, Any]:
    manifest_path = campaign_dir / "campaign_manifest.json"
    state_path = campaign_dir / "state.tsv"
    manifest = _read_json(manifest_path)
    campaign = manifest.get("campaign", {})
    conditions = campaign.get("conditions") or [
        {"condition_id": "mixed_table", "target_mechanism": "mixed"}
    ]
    seeds = campaign.get("seeds", [])
    expected = len(conditions) * len(seeds)
    with state_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    completed = [row for row in rows if row.get("status") == "complete"]
    failed = [row for row in rows if row.get("status") == "failed"]
    identities = [
        (str(row.get("condition_id", "")), int(row.get("seed", 0)))
        for row in completed
    ]
    duplicates = sorted(
        identity for identity in set(identities) if identities.count(identity) > 1
    )
    blockers: list[str] = []
    if len(completed) != expected:
        blockers.append(f"complete matrix mismatch: {len(completed)}/{expected}")
    if failed:
        blockers.append(f"failed state rows present: {len(failed)}")
    if duplicates:
        blockers.append(f"duplicate complete matrix units: {duplicates}")

    metrics_list: list[dict[str, Any]] = []
    evaluation_target_ids_by_run: list[list[str]] = []
    leaf_evidence: list[dict[str, Any]] = []
    runtime_identities: list[dict[str, Any]] = []
    for row in completed:
        run_dir = Path(str(row["run_dir"]))
        missing = [
            name
            for name in REQUIRED_ARTIFACTS
            if not (run_dir / name).is_file() or (run_dir / name).stat().st_size < 1
        ]
        if missing:
            blockers.append(f"{row['run_id']} missing artifacts: {missing}")
            continue
        metrics = _read_json(run_dir / "metrics.json")
        protocol = _read_json(run_dir / "protocol_audit.json")
        run_manifest = _read_json(run_dir / "manifest.json")
        metrics_list.append(metrics)
        evaluation_targets = protocol.get("evaluation_target_ids", [])
        if not isinstance(evaluation_targets, list) or not evaluation_targets:
            blockers.append(f"{row['run_id']} lacks evaluation_target_ids")
            evaluation_target_ids_by_run.append([])
        else:
            evaluation_target_ids_by_run.append(
                [str(value) for value in evaluation_targets]
            )
        execution = protocol.get("execution_health", {})
        if not isinstance(execution, dict) or execution.get("valid") is not True:
            blockers.append(f"{row['run_id']} execution health invalid")
        revision_fallbacks = _revision_fallback_count(metrics)
        if revision_fallbacks:
            blockers.append(
                f"{row['run_id']} deterministic experience revision fallbacks: "
                f"{revision_fallbacks}"
            )
        runtime = _runtime_identity(run_manifest)
        runtime_identities.append(runtime)
        observed_code = dict(runtime.get("code", {})).get("commit")
        if str(observed_code) != expected_code_sha:
            blockers.append(
                f"{row['run_id']} code SHA mismatch: {observed_code}"
            )
        if dict(runtime.get("code", {})).get("dirty") is not False:
            blockers.append(f"{row['run_id']} was created from a dirty worktree")
        service = json.loads(str(runtime.get("service", "{}")))
        observed_context = dict(service.get("service_startup_parameters", {})).get(
            "max_model_len"
        )
        if int(observed_context or 0) != expected_max_model_len:
            blockers.append(
                f"{row['run_id']} max_model_len mismatch: {observed_context}"
            )
        leaf_evidence.append(
            {
                "condition_id": row["condition_id"],
                "seed": int(row["seed"]),
                "attempt": int(row["attempt"]),
                "run_id": row["run_id"],
                "run_dir": str(run_dir),
                "execution_health": execution,
                "memory_revision_fallback_count": revision_fallbacks,
                "sha256": {
                    name: _sha256(run_dir / name)
                    for name in REQUIRED_ARTIFACTS
                },
            }
        )

    canonical_runtime = {
        json.dumps(identity, ensure_ascii=False, sort_keys=True)
        for identity in runtime_identities
    }
    if len(canonical_runtime) != 1:
        blockers.append(
            f"runtime identity is not homogeneous: {len(canonical_runtime)} identities"
        )
    behavior = calibrate_behavior_thresholds(
        metrics_list,
        evaluation_target_ids_by_run,
    )
    if behavior.get("status") != "frozen":
        blockers.extend(str(item) for item in behavior.get("blockers", []))
        if not behavior.get("blockers"):
            blockers.append(f"behavior gate status is {behavior.get('status')}")

    return {
        "schema_version": "task4_campaign_p_before_e_gate_v2",
        "campaign_dir": str(campaign_dir),
        "campaign_id": campaign.get("campaign_id"),
        "expected_matrix_units": expected,
        "completed_matrix_units": len(completed),
        "failed_state_rows": len(failed),
        "ignored_noncomplete_state_rows": len(rows) - len(completed),
        "expected_code_sha": expected_code_sha,
        "expected_max_model_len": expected_max_model_len,
        "runtime_identity": runtime_identities[0] if len(canonical_runtime) == 1 else None,
        "behavior_freeze_preview": behavior,
        "campaign_manifest_sha256": _sha256(manifest_path),
        "state_tsv_sha256": _sha256(state_path),
        "leaf_evidence": sorted(
            leaf_evidence, key=lambda item: (item["condition_id"], item["seed"])
        ),
        "blockers": blockers,
        "status": "ready_to_start_campaign_e" if not blockers else "no_go",
    }


def _runtime_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = manifest.get("metadata", {})
    return {
        "code": metadata.get("code", {}),
        "gpu": metadata.get("gpu", {}),
        "model_service_runtime": metadata.get("model_service_runtime", {}),
        "model": metadata.get("model", {}),
        "service": json.dumps(
            metadata.get("service", {}), ensure_ascii=False, sort_keys=True
        ),
        "embedding": json.dumps(
            metadata.get("embedding", {}), ensure_ascii=False, sort_keys=True
        ),
        "prompts": metadata.get("prompts", {}),
    }


def _revision_fallback_count(metrics: dict[str, Any]) -> int:
    primary = metrics.get("primary_metrics", {})
    stage_per_agent = primary.get("stage_per_agent", {})
    tables = (
        list(stage_per_agent.values())
        if isinstance(stage_per_agent, dict) and stage_per_agent
        else [primary.get("per_agent", {})]
    )
    maximum_by_agent: dict[str, int] = {}
    for per_agent in tables:
        if not isinstance(per_agent, dict):
            continue
        for agent_id, values in per_agent.items():
            memory = values.get("memory", {}) if isinstance(values, dict) else {}
            count = int(memory.get("revision_fallback_count", 0))
            key = str(agent_id)
            maximum_by_agent[key] = max(maximum_by_agent.get(key, 0), count)
    return sum(maximum_by_agent.values())


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
