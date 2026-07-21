"""Frozen TASK8 protocol primitives shared by local and multi-worker runners."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from agentmemeval.core.domain import MemorySnapshot
from agentmemeval.core.errors import ConfigError
from agentmemeval.core.seeds import derive_seed

SCHEDULE_SCHEMA_VERSION = "task8-heldout-schedule-v1"
ISOLATION_SCHEMA_VERSION = "task8-memory-isolation-v1"


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize an identity object deterministically."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def derive_formal_hand_seed(
    root_seed: int,
    phase: str,
    checkpoint_hand: int,
    table_id: str,
    hand_index: int,
    rng_stream: str,
) -> int:
    """Derive CRN without mechanism, target, agent, worker, path, or cache identity."""

    if phase not in {"source", "heldout"}:
        raise ConfigError(f"未知 Formal RNG phase：{phase}")
    if checkpoint_hand < 0 or hand_index < 0:
        raise ConfigError("checkpoint_hand 与 hand_index 不能为负数")
    if not table_id or not rng_stream:
        raise ConfigError("table_id 与 rng_stream 不能为空")
    return derive_seed(
        int(root_seed),
        "task8_formal_crn_v1",
        phase,
        int(checkpoint_hand),
        str(table_id),
        int(hand_index),
        str(rng_stream),
    )


def build_heldout_schedule_manifest(
    *,
    root_seed: int,
    checkpoint_set: list[int],
    table_set: list[str],
    hands_by_checkpoint: dict[int, int],
    table_size: int,
    roster_identity: str | dict[str, str],
) -> dict[str, Any]:
    """Build a standalone, hashable H01/H02/H03 schedule."""

    if checkpoint_set != sorted(set(checkpoint_set)) or not checkpoint_set:
        raise ConfigError("schedule checkpoint_set 必须严格递增且非空")
    if len(table_set) != len(set(table_set)) or not table_set:
        raise ConfigError("schedule table_set 必须非空且无重复")
    if table_size < 2 or not roster_identity:
        raise ConfigError("schedule 需要 table_size>=2 和 roster_identity")
    roster_identities = (
        {table_id: str(roster_identity[table_id]) for table_id in table_set}
        if isinstance(roster_identity, dict)
        else {table_id: str(roster_identity) for table_id in table_set}
    )
    if set(roster_identities) != set(table_set) or any(
        not value for value in roster_identities.values()
    ):
        raise ConfigError("schedule roster identities 必须完整覆盖 held-out tables")
    rows: list[dict[str, Any]] = []
    for checkpoint_hand in checkpoint_set:
        hands = int(hands_by_checkpoint.get(checkpoint_hand, 0))
        if hands < 0:
            raise ConfigError("schedule hands_by_checkpoint 不能为负数")
        for table_offset, table_id in enumerate(table_set):
            for hand_index in range(hands):
                rows.append(
                    {
                        "phase": "heldout",
                        "checkpoint_hand": checkpoint_hand,
                        "table_id": table_id,
                        "roster_identity": roster_identities[table_id],
                        "hand_index": hand_index,
                        "hand_number": hand_index + 1,
                        "dealer_index": (hand_index + table_offset) % table_size,
                        "deal_and_opponent_seed": derive_formal_hand_seed(
                            root_seed,
                            "heldout",
                            checkpoint_hand,
                            table_id,
                            hand_index,
                            "deal_and_opponent",
                        ),
                    }
                )
    body = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "root_seed": int(root_seed),
        "checkpoint_set": list(checkpoint_set),
        "heldout_table_set": list(table_set),
        "table_size": int(table_size),
        "roster_identities": roster_identities,
        "source_namespace": "task8/source/v1",
        "heldout_namespace": "task8/heldout/v1",
        "rows": rows,
    }
    return {**body, "schedule_sha256": sha256_json(body)}


def verify_schedule_manifest(manifest: dict[str, Any]) -> str:
    """Fail closed when a frozen schedule is incomplete or has been altered."""

    supplied = str(manifest.get("schedule_sha256", ""))
    body = {key: value for key, value in manifest.items() if key != "schedule_sha256"}
    if body.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
        raise ConfigError("held-out schedule schema_version 不受支持")
    actual = sha256_json(body)
    if not supplied or supplied != actual:
        raise ConfigError("held-out schedule SHA-256 不匹配")
    if body.get("source_namespace") == body.get("heldout_namespace"):
        raise ConfigError("source 与 held-out RNG namespace 必须隔离")
    return actual


def clone_memory_branches(snapshot: MemorySnapshot) -> dict[str, MemorySnapshot]:
    """Create isolated Frozen, Online, and whole-memory Without branches."""

    frozen = copy.deepcopy(snapshot)
    online = copy.deepcopy(snapshot)
    without = _without_memory_snapshot(snapshot)
    branches = {"Frozen": frozen, "Online": online, "Without": without}
    payload_ids = [id(branch.payload) for branch in branches.values()]
    if len(payload_ids) != len(set(payload_ids)):
        raise ConfigError("memory branches 意外共享可写 payload")
    return branches


def build_clone_audit(
    parent: MemorySnapshot, branches: dict[str, MemorySnapshot]
) -> dict[str, Any]:
    parent_sha = sha256_json(parent.to_dict())
    return {
        "schema_version": ISOLATION_SCHEMA_VERSION,
        "parent_checkpoint_sha256": parent_sha,
        "branches": {
            name: {
                "snapshot_sha256": sha256_json(snapshot.to_dict()),
                "transform": "identity_deepcopy" if name != "Without" else "whole_memory_removal",
            }
            for name, snapshot in sorted(branches.items())
        },
    }


def _without_memory_snapshot(snapshot: MemorySnapshot) -> MemorySnapshot:
    mechanism = snapshot.mechanism
    payload = copy.deepcopy(snapshot.payload)
    if mechanism == "no_memory":
        payload = {}
    elif mechanism == "fact":
        payload = _clear_fact_payload(payload)
    elif mechanism == "expr":
        payload = _clear_expr_payload(payload)
    elif mechanism == "fact_expr_sync":
        _require_nested_payload(payload, mechanism)
        payload["fact"] = _clear_fact_payload(dict(payload["fact"]))
        payload["expr"] = _clear_expr_payload(dict(payload["expr"]))
    elif mechanism == "fact_expr_async":
        _require_nested_payload(payload, mechanism)
        payload["fact"] = _clear_fact_payload(dict(payload["fact"]))
        payload["expr"] = _clear_expr_payload(dict(payload["expr"]))
        for key, empty in (
            ("sweep_log", []),
            ("evidence_review_queue", []),
            ("fact_state", {}),
            ("skipped_trajectory_hand_ids", []),
        ):
            payload[key] = empty
        payload["hand_counter"] = 0
        payload["eligible_hand_counter"] = 0
    else:
        raise ConfigError(
            f"pending_review: 无法无歧义移除机制 {mechanism!r} 的整份记忆"
        )
    return MemorySnapshot(
        mechanism=mechanism,
        agent_id=snapshot.agent_id,
        scope=snapshot.scope,
        payload=payload,
    )


def _require_nested_payload(payload: dict[str, Any], mechanism: str) -> None:
    if not isinstance(payload.get("fact"), dict) or not isinstance(payload.get("expr"), dict):
        raise ConfigError(f"pending_review: {mechanism} snapshot 缺少 fact/expr 子快照")


def _clear_fact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload["records"] = []
    payload["admission_log"] = []
    payload["admission_counts"] = {}
    payload["last_admission_status"] = None
    payload["last_admission_reasons"] = []
    payload["retrieval_request_count"] = 0
    payload["empty_retrieval_count"] = 0
    payload["retrieval_below_threshold_count"] = 0
    payload["retrieval_duplicate_excluded_count"] = 0
    payload["retrieval_audit_log"] = []
    payload["next_record_index"] = 0
    return payload


def _clear_expr_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # Empty history restores the mechanism's freshly constructed baseline document.
    payload["history"] = []
    payload["revision_log"] = []
    payload["skipped_trajectory_hand_ids"] = []
    return payload
