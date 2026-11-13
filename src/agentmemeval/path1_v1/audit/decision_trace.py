"""Rebuildable decision traces with an explicit AgentObservation visibility boundary."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from agentmemeval.path1_v1.core.domain import (
    AgentObservation,
    ExperienceDocument,
    FactualMemoryRecord,
    LegalAction,
    LegalActionSet,
    MemoryContext,
    MemoryScope,
    PlayerPublicState,
)
from agentmemeval.path1_v1.environment.raise_sizing import RaiseSizingPlan
from agentmemeval.path1_v1.prompts.decision import (
    PROMPT_TEMPLATE_VERSION,
    render_experience_section,
    render_facts_section,
    render_memory_section,
    render_system_prompt,
    render_user_prompt,
)

DECISION_TRACE_SCHEMA_VERSION = "decision_trace_v1"
SAFE_OBSERVATION_SCHEMA_VERSION = "agent_observation_visible_v1"
DECISION_SOURCE_DIRECTORY = "dps"

_OBSERVATION_FIELDS = (
    "agent_id",
    "table_id",
    "hand_id",
    "phase",
    "seat",
    "hole_cards",
    "community_cards",
    "pot",
    "current_bet",
    "to_call",
    "players",
    "action_history",
    "legal_actions",
    "seed",
    "dealer_seat",
    "small_blind_seat",
    "big_blind_seat",
    "small_blind_amount",
    "big_blind_amount",
)
_PUBLIC_ACTION_FIELDS = (
    "event",
    "table_id",
    "hand_id",
    "phase",
    "agent_id",
    "action_type",
    "amount",
    "committed",
    "to_call",
    "pot_before",
    "pot_after",
    "effective_raise",
    "last_raise_size",
)


class DecisionTraceError(ValueError):
    """Raised when an audit trace cannot reproduce the exact live decision prompt."""


@dataclass(frozen=True, slots=True)
class RebuiltDecisionPrompt:
    """Exact, verified prompt inputs reconstructed from one persisted action event."""

    observation: AgentObservation
    memory_context: MemoryContext
    raise_sizing: RaiseSizingPlan
    system_prompt: str
    user_prompt: str
    verification: dict[str, bool]


def sha256_text(text: str) -> str:
    """Hash text exactly as UTF-8 without line-ending or whitespace normalization."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: object) -> str:
    """Hash a JSON-compatible value with deterministic keys and separators."""

    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(rendered)


def safe_observation_payload(observation: AgentObservation) -> dict[str, object]:
    """Serialize only the observation contract and allowlisted public action fields.

    ``AgentObservation.hole_cards`` is the observing Agent's own hand. Opponent
    player entries intentionally use ``PlayerPublicState``, which has no card field.
    Unknown nested action fields are discarded rather than trusted as public data.
    """

    payload = observation.to_dict()
    safe_actions: list[dict[str, object]] = []
    for raw_event in observation.action_history:
        safe_actions.append(
            {key: raw_event[key] for key in _PUBLIC_ACTION_FIELDS if key in raw_event}
        )
    payload["action_history"] = safe_actions
    return {field: payload[field] for field in _OBSERVATION_FIELDS}


def build_and_persist_decision_trace(
    *,
    run_root: str | Path,
    observation: AgentObservation,
    memory_context: MemoryContext,
    raise_sizing: RaiseSizingPlan,
    decision_index: int,
    expected_prompt: Mapping[str, object] | None = None,
    experience_reference: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one immutable trace and retain overwritten experience bodies by SHA."""

    if decision_index < 0:
        raise DecisionTraceError("decision_index must be non-negative")
    safe_payload = safe_observation_payload(observation)
    replay_observation = _observation_from_payload(safe_payload)
    system_prompt = render_system_prompt(memory_context)
    user_prompt = render_user_prompt(
        observation,
        memory_context,
        raise_sizing=raise_sizing,
    )
    safe_user_prompt = render_user_prompt(
        replay_observation,
        memory_context,
        raise_sizing=raise_sizing,
    )
    if safe_user_prompt != user_prompt:
        raise DecisionTraceError(
            "safe observation projection changed the live user prompt; "
            "extend the public-action allowlist before recording this decision"
        )

    prompt_fingerprint = {
        "template_version": PROMPT_TEMPLATE_VERSION,
        "system_sha256": sha256_text(system_prompt),
        "user_sha256": sha256_text(user_prompt),
        "system_chars": len(system_prompt),
        "user_chars": len(user_prompt),
    }
    if expected_prompt is not None:
        for key, value in prompt_fingerprint.items():
            if expected_prompt.get(key) != value:
                raise DecisionTraceError(
                    f"live prompt metadata mismatch for {key}: "
                    f"expected={expected_prompt.get(key)!r}, rebuilt={value!r}"
                )

    facts_text = render_facts_section(memory_context)
    experience_text = render_experience_section(memory_context)
    memory_text = render_memory_section(memory_context)
    facts_fingerprint = _text_fingerprint(facts_text)
    experience_fingerprint = _text_fingerprint(experience_text)
    memory_fingerprint = _text_fingerprint(memory_text)

    fact_ids = [fact.record_id for fact in memory_context.facts]
    experience_payload = _experience_trace_payload(
        run_root=run_root,
        context=memory_context,
        rendered_fingerprint=experience_fingerprint,
        experience_reference=experience_reference,
    )
    observation_sha256 = sha256_json(safe_payload)
    decision_key = sha256_json(
        {
            "agent_id": observation.agent_id,
            "table_id": observation.table_id,
            "hand_id": observation.hand_id,
            "decision_index": decision_index,
            "observation_sha256": observation_sha256,
        }
    )
    return {
        "schema_version": DECISION_TRACE_SCHEMA_VERSION,
        "decision_key": decision_key,
        "decision_index": decision_index,
        "decision_index_scope": "agent_within_hand_zero_based",
        "observation": {
            "schema_version": SAFE_OBSERVATION_SCHEMA_VERSION,
            "visibility_boundary": "AgentObservation",
            "sha256": observation_sha256,
            "payload": safe_payload,
        },
        "memory": {
            "facts": {**facts_fingerprint, "fact_ids": fact_ids},
            "experience": experience_payload,
            "section": memory_fingerprint,
        },
        "prompt": prompt_fingerprint,
    }


def rebuild_decision_prompt(
    event: Mapping[str, object],
    *,
    run_root: str | Path,
) -> RebuiltDecisionPrompt:
    """Rebuild and strictly verify both prompts from one persisted action event."""

    trace = _required_mapping(event.get("decision_trace"), "decision_trace")
    if trace.get("schema_version") != DECISION_TRACE_SCHEMA_VERSION:
        raise DecisionTraceError("unsupported or missing decision trace schema")
    observation_trace = _required_mapping(trace.get("observation"), "trace observation")
    safe_payload = _required_mapping(
        observation_trace.get("payload"),
        "safe observation payload",
    )
    observation = _observation_from_payload(safe_payload)
    observation_matches = sha256_json(safe_payload) == observation_trace.get("sha256")

    compact_context = _required_mapping(event.get("memory_context"), "memory_context")
    memory_trace = _required_mapping(trace.get("memory"), "trace memory")
    context = _memory_context_from_compact(
        compact_context,
        memory_trace,
        run_root=run_root,
    )
    raise_sizing_payload = _required_mapping(event.get("raise_sizing"), "raise_sizing")
    raise_sizing = _raise_sizing_from_payload(raise_sizing_payload)

    facts_text = render_facts_section(context)
    experience_text = render_experience_section(context)
    memory_text = render_memory_section(context)
    facts_trace = _required_mapping(memory_trace.get("facts"), "facts trace")
    experience_trace = _required_mapping(
        memory_trace.get("experience"),
        "experience trace",
    )
    section_trace = _required_mapping(memory_trace.get("section"), "memory section trace")
    system_prompt = render_system_prompt(context)
    user_prompt = render_user_prompt(observation, context, raise_sizing=raise_sizing)
    prompt_trace = _required_mapping(trace.get("prompt"), "trace prompt")
    event_prompt = _required_mapping(event.get("prompt"), "event prompt")
    checks = {
        "observation_sha256": observation_matches,
        "fact_ids": list(facts_trace.get("fact_ids", []))
        == [fact.record_id for fact in context.facts],
        "facts_section": _matches_text_fingerprint(facts_trace, facts_text),
        "experience_section": _matches_text_fingerprint(
            experience_trace,
            experience_text,
        ),
        "memory_section": _matches_text_fingerprint(section_trace, memory_text),
        "template_version": (
            prompt_trace.get("template_version") == PROMPT_TEMPLATE_VERSION
            and event_prompt.get("template_version") == PROMPT_TEMPLATE_VERSION
        ),
        "system_sha256": (
            prompt_trace.get("system_sha256") == sha256_text(system_prompt)
            and event_prompt.get("system_sha256") == sha256_text(system_prompt)
        ),
        "user_sha256": (
            prompt_trace.get("user_sha256") == sha256_text(user_prompt)
            and event_prompt.get("user_sha256") == sha256_text(user_prompt)
        ),
        "system_chars": (
            prompt_trace.get("system_chars") == len(system_prompt)
            and event_prompt.get("system_chars") == len(system_prompt)
        ),
        "user_chars": (
            prompt_trace.get("user_chars") == len(user_prompt)
            and event_prompt.get("user_chars") == len(user_prompt)
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise DecisionTraceError("decision prompt roundtrip failed: " + ", ".join(sorted(failed)))
    return RebuiltDecisionPrompt(
        observation=observation,
        memory_context=context,
        raise_sizing=raise_sizing,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        verification=checks,
    )


def _experience_trace_payload(
    *,
    run_root: str | Path,
    context: MemoryContext,
    rendered_fingerprint: dict[str, object],
    experience_reference: Mapping[str, object] | None,
) -> dict[str, object]:
    experience = context.experience
    if experience is None:
        return {
            **rendered_fingerprint,
            "version": None,
            "body_sha256": None,
            "body_chars": 0,
            "body_source": None,
            "canonical_path": None,
        }
    body = str(experience.body)
    body_sha256 = sha256_text(body)
    reference = dict(experience_reference or {})
    if reference:
        expected = {
            "version": experience.version,
            "sha256": body_sha256,
            "chars": len(body),
        }
        for key, value in expected.items():
            if reference.get(key) != value:
                raise DecisionTraceError(
                    f"experience reference mismatch for {key}: "
                    f"reference={reference.get(key)!r}, context={value!r}"
                )
    return {
        **rendered_fingerprint,
        "version": int(experience.version),
        "body_sha256": body_sha256,
        "body_chars": len(body),
        "body_source": _persist_text_source(run_root, "experience_body", body),
        "canonical_path": reference.get("path"),
    }


def _text_fingerprint(text: str) -> dict[str, object]:
    return {"sha256": sha256_text(text), "chars": len(text)}


def _matches_text_fingerprint(reference: Mapping[str, object], text: str) -> bool:
    return reference.get("sha256") == sha256_text(text) and reference.get("chars") == len(text)


def _persist_text_source(
    run_root: str | Path,
    source_kind: str,
    text: str,
) -> dict[str, object]:
    root = Path(run_root).resolve()
    digest = sha256_text(text)
    compact_kind = {"experience_body": "e"}.get(source_kind, source_kind)
    relative_path = Path(DECISION_SOURCE_DIRECTORY) / compact_kind / f"{digest[:24]}.txt"
    path = (root / relative_path).resolve()
    _assert_inside_root(path, root)
    io_path = _platform_io_path(path)
    for attempt in range(2):
        try:
            io_path.parent.mkdir(parents=True, exist_ok=True)
            with io_path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
            break
        except FileNotFoundError:
            if attempt:
                raise
        except FileExistsError:
            persisted = io_path.read_text(encoding="utf-8")
            if persisted != text:
                raise DecisionTraceError(f"content-addressed source collision at {path}") from None
            break
    return {
        "sha256": digest,
        "chars": len(text),
        "source_path": relative_path.as_posix(),
    }


def _read_text_source(
    run_root: str | Path,
    reference: Mapping[str, object],
) -> str:
    root = Path(run_root).resolve()
    raw_path = reference.get("source_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise DecisionTraceError("decision trace source_path is missing")
    path = (root / Path(raw_path)).resolve()
    _assert_inside_root(path, root)
    io_path = _platform_io_path(path)
    if not io_path.is_file():
        raise DecisionTraceError(f"decision trace source is missing: {raw_path}")
    text = io_path.read_text(encoding="utf-8")
    if sha256_text(text) != reference.get("sha256"):
        raise DecisionTraceError(f"decision trace source SHA mismatch: {raw_path}")
    if len(text) != reference.get("chars"):
        raise DecisionTraceError(f"decision trace source char count mismatch: {raw_path}")
    return text


def _memory_context_from_compact(
    payload: Mapping[str, object],
    trace: Mapping[str, object],
    *,
    run_root: str | Path,
) -> MemoryContext:
    raw_facts = payload.get("facts", [])
    if not isinstance(raw_facts, list):
        raise DecisionTraceError("memory_context.facts must be an array")
    facts = [_fact_from_payload(_required_mapping(item, "fact")) for item in raw_facts]
    metadata = dict(_required_mapping(payload.get("metadata", {}), "memory metadata"))
    compact_experience = payload.get("experience")
    experience: ExperienceDocument | None = None
    if compact_experience is not None:
        reference = _required_mapping(compact_experience, "experience reference")
        experience_trace = _required_mapping(trace.get("experience"), "experience trace")
        body_source = experience_trace.get("body_source")
        if body_source is None:
            body = _resolve_legacy_experience_body(run_root, reference)
        else:
            body = _read_text_source(
                run_root,
                _required_mapping(body_source, "experience body source"),
            )
        expected_sha = str(reference.get("sha256", ""))
        if sha256_text(body) != expected_sha:
            raise DecisionTraceError("resolved experience body does not match event reference")
        if len(body) != reference.get("chars"):
            raise DecisionTraceError("resolved experience body char count does not match")
        if experience_trace.get("body_sha256") != expected_sha:
            raise DecisionTraceError("trace experience body SHA diverges from event reference")
        if experience_trace.get("version") != reference.get("version"):
            raise DecisionTraceError("trace experience version diverges from event reference")
        experience = ExperienceDocument(
            version=int(reference["version"]),
            body=body,
            source_hand_ids=[],
            updated_at="",
            scope=cast(MemoryScope, str(metadata.get("scope", "per_agent"))),
        )
    return MemoryContext(
        facts=facts,
        experience=experience,
        metadata=metadata,
    )


def _resolve_legacy_experience_body(
    run_root: str | Path,
    reference: Mapping[str, object],
) -> str:
    root = Path(run_root).resolve()
    raw_path = reference.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise DecisionTraceError("experience body source and canonical path are both missing")
    canonical = (root / Path(raw_path)).resolve()
    _assert_inside_root(canonical, root)
    expected_sha = str(reference.get("sha256", ""))
    if canonical.is_file():
        body = canonical.read_text(encoding="utf-8")
        if sha256_text(body) == expected_sha:
            return body
    log_path = canonical.parent / "experience_log.jsonl"
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            for field in ("old_md", "new_md"):
                candidate = row.get(field)
                if isinstance(candidate, str) and sha256_text(candidate) == expected_sha:
                    return candidate
    raise DecisionTraceError("referenced historical experience body could not be resolved")


def _fact_from_payload(payload: Mapping[str, object]) -> FactualMemoryRecord:
    raw_features = payload.get("features", [])
    if not isinstance(raw_features, list):
        raise DecisionTraceError("fact.features must be an array")
    return FactualMemoryRecord(
        record_id=str(payload["record_id"]),
        agent_id=str(payload["agent_id"]),
        table_id=str(payload["table_id"]),
        hand_id=str(payload["hand_id"]),
        scope=cast(MemoryScope, str(payload["scope"])),
        state_summary=str(payload.get("state_summary", "")),
        action_summary=str(payload.get("action_summary", "")),
        final_reward=int(payload.get("final_reward", 0)),
        features=[str(item) for item in raw_features],
        source=dict(_required_mapping(payload.get("source", {}), "fact source")),
        created_at=str(payload.get("created_at", "")),
    )


def _observation_from_payload(payload: Mapping[str, object]) -> AgentObservation:
    if set(payload) != set(_OBSERVATION_FIELDS):
        missing = sorted(set(_OBSERVATION_FIELDS) - set(payload))
        extra = sorted(set(payload) - set(_OBSERVATION_FIELDS))
        raise DecisionTraceError(
            f"safe observation schema mismatch; missing={missing}, extra={extra}"
        )
    raw_players = payload["players"]
    raw_history = payload["action_history"]
    raw_legal = _required_mapping(payload["legal_actions"], "legal_actions")
    raw_actions = raw_legal.get("actions")
    if not isinstance(raw_players, list) or not isinstance(raw_history, list):
        raise DecisionTraceError("players and action_history must be arrays")
    if not isinstance(raw_actions, list):
        raise DecisionTraceError("legal_actions.actions must be an array")
    players = [
        PlayerPublicState(**dict(_required_mapping(item, "public player"))) for item in raw_players
    ]
    legal_actions = LegalActionSet(
        actions=[
            LegalAction(**dict(_required_mapping(item, "legal action"))) for item in raw_actions
        ]
    )
    return AgentObservation(
        agent_id=str(payload["agent_id"]),
        table_id=str(payload["table_id"]),
        hand_id=str(payload["hand_id"]),
        phase=str(payload["phase"]),
        seat=int(payload["seat"]),
        hole_cards=[str(card) for card in cast(list[object], payload["hole_cards"])],
        community_cards=[str(card) for card in cast(list[object], payload["community_cards"])],
        pot=int(payload["pot"]),
        current_bet=int(payload["current_bet"]),
        to_call=int(payload["to_call"]),
        players=players,
        action_history=[dict(_required_mapping(item, "public action")) for item in raw_history],
        legal_actions=legal_actions,
        seed=int(payload["seed"]),
        dealer_seat=_optional_int(payload["dealer_seat"]),
        small_blind_seat=_optional_int(payload["small_blind_seat"]),
        big_blind_seat=_optional_int(payload["big_blind_seat"]),
        small_blind_amount=_optional_int(payload["small_blind_amount"]),
        big_blind_amount=_optional_int(payload["big_blind_amount"]),
    )


def _raise_sizing_from_payload(payload: Mapping[str, object]) -> RaiseSizingPlan:
    raw_allowed = payload.get("allowed_amounts")
    if raw_allowed is not None and not isinstance(raw_allowed, list):
        raise DecisionTraceError("raise_sizing.allowed_amounts must be an array or null")
    return RaiseSizingPlan(
        policy=str(payload["policy"]),
        allowed_amounts=(
            tuple(int(item) for item in raw_allowed) if isinstance(raw_allowed, list) else None
        ),
        pot_after_call=int(payload["pot_after_call"]),
        stack_to_pot_ratio=float(payload["stack_to_pot_ratio"]),
        native_min_amount=_optional_int(payload.get("native_min_amount")),
        native_max_amount=_optional_int(payload.get("native_max_amount")),
    )


def _required_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DecisionTraceError(f"{label} must be an object")
    return value


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _assert_inside_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DecisionTraceError("decision trace path escaped the run root") from exc


def _platform_io_path(path: Path) -> Path:
    """Use Windows extended paths so long append-only run IDs remain writable."""

    if os.name != "nt":
        return path
    raw = str(path)
    if raw.startswith("\\\\?\\"):
        return path
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw[2:])
    return Path("\\\\?\\" + raw)
