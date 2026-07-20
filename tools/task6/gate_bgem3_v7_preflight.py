"""Fail-closed TASK6 gate for the 528 BGE-M3 V7 real-service preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from agentmemeval.evaluation.degeneracy import evaluate_execution_health

READY_STATUS = "ready_to_start_task6_bgem3_campaign_p_v7"
EXPECTED_PROMPTS = {
    "decision_version": "2026-07-19-v6-counterfactual-calibrated-memory",
    "decision_system_sha256": (
        "9cd2f157225e14bfee9113c3af01a2ff4fff839aeb68dcfd8f11740bd8647800"
    ),
    "experience_update_sha256": (
        "7788fa2f85adca9710cf20f2fc95769db1b2b93ee60f9a5236a430b87d4ad382"
    ),
}
EXPECTED_BGE = {
    "model": "BAAI/bge-m3",
    "revision": "5617a9f61b028005a4858fdac845db406aefb181",
    "weights_hash": "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38",
    "tokenizer_revision": "5617a9f61b028005a4858fdac845db406aefb181",
    "query_policy": "raw_symmetric_no_instruction",
    "cache_schema_version": "bgem3_native_document_repr_v1",
}
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
    parser.add_argument("--counterfactual-gate", required=True)
    parser.add_argument("--bge-health-before", required=True)
    parser.add_argument("--bge-health-after", required=True)
    parser.add_argument("--bge-score-smoke", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = build_gate(
        run_dir=Path(args.run_dir).resolve(),
        counterfactual_gate=Path(args.counterfactual_gate).resolve(),
        health_before=Path(args.bge_health_before).resolve(),
        health_after=Path(args.bge_health_after).resolve(),
        score_smoke=Path(args.bge_score_smoke).resolve(),
        expected_code_sha=str(args.expected_code_sha),
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), **audit}, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == READY_STATUS else 2


def build_gate(
    *,
    run_dir: Path,
    counterfactual_gate: Path,
    health_before: Path,
    health_after: Path,
    score_smoke: Path,
    expected_code_sha: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    missing = [
        name
        for name in REQUIRED_ARTIFACTS
        if not (run_dir / name).is_file() or (run_dir / name).stat().st_size < 1
    ]
    if missing:
        blockers.append(f"missing required artifacts: {missing}")
        return _result(run_dir, expected_code_sha, blockers, {})

    manifest = _read_json(run_dir / "manifest.json")
    metrics = _read_json(run_dir / "metrics.json")
    events = _read_jsonl(run_dir / "events.jsonl")
    hands = _read_jsonl(run_dir / "hand_summaries.jsonl")
    counterfactual = _read_json(counterfactual_gate)
    before = _read_json(health_before)
    after = _read_json(health_after)
    score = _read_json(score_smoke)

    metadata = dict(manifest.get("metadata", {}))
    code = dict(metadata.get("code", {}))
    prompts = dict(metadata.get("prompts", {}))
    embedding = dict(metadata.get("embedding", {}))
    protocol = dict(metadata.get("protocol", {}))
    if code != {"commit": expected_code_sha, "dirty": False}:
        blockers.append(f"code identity mismatch: {code}")
    for key, expected in EXPECTED_PROMPTS.items():
        if prompts.get(key) != expected:
            blockers.append(f"prompt identity mismatch for {key}: {prompts.get(key)}")
    if counterfactual.get("status") != "ready_to_start_campaign_p_v7_pilot":
        blockers.append("counterfactual behavior/execution gate did not pass")

    expected_embedding_fields = {
        "name": EXPECTED_BGE["model"],
        "revision": EXPECTED_BGE["revision"],
        "weights_hash": EXPECTED_BGE["weights_hash"],
        "tokenizer_revision": EXPECTED_BGE["tokenizer_revision"],
        "query_policy": EXPECTED_BGE["query_policy"],
        "cache_schema_version": EXPECTED_BGE["cache_schema_version"],
    }
    for key, expected in expected_embedding_fields.items():
        if embedding.get(key) != expected:
            blockers.append(f"embedding identity mismatch for {key}: {embedding.get(key)}")
    if embedding.get("backend") != "bgem3_hybrid_http":
        blockers.append(f"unexpected embedding backend: {embedding.get('backend')}")
    if embedding.get("hybrid_weights") != [0.4, 0.2, 0.4]:
        blockers.append(f"unexpected hybrid weights: {embedding.get('hybrid_weights')}")
    if embedding.get("candidate_depth") != 1000:
        blockers.append("candidate_depth is not frozen at 1000")
    if embedding.get("colbert_rerank_depth") != 1000:
        blockers.append("colbert_rerank_depth is not frozen at 1000")
    if embedding.get("final_top_k_policy") != "agent_roster_top_k":
        blockers.append("final top-k policy mismatch")
    if protocol.get("behavior_threshold_status") != "frozen":
        blockers.append("behavior thresholds are not frozen in manifest")
    if not protocol.get("behavior_threshold_sha256"):
        blockers.append("behavior threshold hash is missing from manifest")

    train = [hand for hand in hands if hand.get("stage") == "train"]
    test = [hand for hand in hands if hand.get("stage") == "test"]
    if len(hands) != 60 or len(train) != 20 or len(test) != 40:
        blockers.append(
            f"preflight hand budget mismatch: total={len(hands)} train={len(train)} "
            f"test={len(test)}"
        )
    if any(hand.get("memory_updated") is not True for hand in train):
        blockers.append("one or more train hands did not update memory")
    if any(hand.get("memory_updated") is not False for hand in test):
        blockers.append("one or more test hands updated memory")

    execution = evaluate_execution_health(hands, metrics)
    if execution.get("valid") is not True:
        blockers.append("execution health is invalid")
    fallback_count = sum(bool(event.get("fallback_used")) for event in events)
    llm_error_count = sum(
        isinstance(event.get("llm"), dict) and bool(event["llm"].get("error"))
        for event in events
    )
    if fallback_count:
        blockers.append(f"action fallback count is nonzero: {fallback_count}")
    if llm_error_count:
        blockers.append(f"LLM/service error count is nonzero: {llm_error_count}")

    for key, expected in EXPECTED_BGE.items():
        if after.get(key) != expected:
            blockers.append(f"BGE health identity mismatch for {key}: {after.get(key)}")
    counter_deltas = {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in (
            "request_count",
            "scored_document_count",
            "cache_hit_count",
            "cache_miss_count",
            "cache_replacement_count",
        )
    }
    if counter_deltas["request_count"] <= 0 or counter_deltas["scored_document_count"] <= 0:
        blockers.append(f"BGE scoring counters did not increase: {counter_deltas}")
    if counter_deltas["cache_miss_count"] < 0 or counter_deltas["cache_hit_count"] < 0:
        blockers.append(f"BGE cache counters regressed: {counter_deltas}")
    if int(after.get("cache_entries", -1)) > int(after.get("cache_capacity", -1)):
        blockers.append("BGE cache exceeded its frozen capacity")

    smoke_scores = score.get("scores")
    if not isinstance(smoke_scores, list) or not smoke_scores:
        blockers.append("BGE score smoke has no score rows")
    else:
        for index, row in enumerate(smoke_scores):
            if not isinstance(row, dict):
                blockers.append(f"BGE score row {index} is malformed")
                continue
            try:
                values = [float(row[key]) for key in ("combined", "dense", "sparse", "colbert")]
            except (KeyError, TypeError, ValueError):
                blockers.append(f"BGE score row {index} has schema drift")
                continue
            if not all(math.isfinite(value) for value in values):
                blockers.append(f"BGE score row {index} contains non-finite values")

    evidence = {
        "required_artifact_sha256": {
            name: _sha256(run_dir / name) for name in REQUIRED_ARTIFACTS
        },
        "prompt_identity": prompts,
        "embedding_identity": embedding,
        "behavior_threshold_identity": {
            key: protocol.get(key)
            for key in (
                "behavior_threshold_status",
                "behavior_thresholds",
                "behavior_threshold_sha256",
                "behavior_threshold_source",
                "behavior_threshold_freeze_label",
            )
        },
        "hand_counts": {"total": len(hands), "train": len(train), "test": len(test)},
        "execution_health": execution,
        "action_fallback_count": fallback_count,
        "llm_error_count": llm_error_count,
        "bge_counter_deltas": counter_deltas,
        "bge_health_after": after,
        "counterfactual_gate_sha256": _sha256(counterfactual_gate),
        "bge_score_smoke_sha256": _sha256(score_smoke),
    }
    return _result(run_dir, expected_code_sha, blockers, evidence)


def _result(
    run_dir: Path,
    expected_code_sha: str,
    blockers: list[str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": READY_STATUS if not blockers else "blocked",
        "run_dir": str(run_dir),
        "expected_code_sha": expected_code_sha,
        "blockers": blockers,
        "evidence": evidence,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSONL objects: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
