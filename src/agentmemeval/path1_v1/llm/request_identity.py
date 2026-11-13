"""Episode-bound Path1 HTTP identities and independent 63-bit sampling streams.

Sending a seed is observable client behavior, not proof that a backend honors
it. Runtime admission must separately verify the deployed sampler and range.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from agentmemeval.path1_v1.llm.terminal_policy import LogicalRequestBudget, Path1TerminalError

SAMPLER_ALGORITHM = "path1-request-sha256-63bit-v1"


def _bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _slug(value: str) -> None:
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise Path1TerminalError("configuration", "invalid_request_identity_component")


class SamplerSeedLedger:
    def __init__(self, *, forbidden_seeds: set[int] | None = None) -> None:
        self.forbidden = set(forbidden_seeds or ())
        if any(type(value) is not int or not 0 <= value < 2**63 for value in self.forbidden):
            raise Path1TerminalError("configuration", "invalid_forbidden_sampler_seed")
        self.entries: dict[str, int] = {}
        self.owners: dict[int, str] = {}

    def derive(self, labels: Mapping[str, Any]) -> int:
        key = _bytes(dict(labels)).decode("utf-8")
        seed = int(_sha([SAMPLER_ALGORITHM, dict(labels)])[:16], 16) % 2**63
        if seed in self.forbidden or (seed in self.owners and self.owners[seed] != key):
            raise Path1TerminalError("state", "sampler_seed_collision_no_redraw")
        if key in self.entries and self.entries[key] != seed:
            raise Path1TerminalError("state", "sampler_derivation_changed")
        self.entries[key] = seed
        self.owners[seed] = key
        return seed


@dataclass(frozen=True)
class EpisodeRequestContext:
    campaign_id: str
    phase: str
    artifact_id: str
    seed_namespace: str
    root_seed: int
    stage: str
    episode_index: int
    arm: str
    actor_id: str
    client_seed: int

    def __post_init__(self) -> None:
        for value in (self.campaign_id, self.artifact_id, self.seed_namespace, self.actor_id):
            _slug(value)
        if self.phase not in {"E1", "E2", "E3", "offline"}:
            raise Path1TerminalError("configuration", "invalid_request_phase")
        if (self.stage, self.arm) not in {
            ("formation", "formation"),
            ("evaluation", "retained"),
            ("evaluation", "removed"),
        }:
            raise Path1TerminalError("configuration", "invalid_request_stage_arm")
        for value in (self.root_seed, self.episode_index, self.client_seed):
            if type(value) is not int or value < 0:
                raise Path1TerminalError("configuration", "invalid_request_seed_or_episode")
        if self.client_seed >= 2**63:
            raise Path1TerminalError("configuration", "invalid_client_seed_range")


class RequestIdentityBinding:
    """One fresh actor client, with separate action and updater logical clocks."""

    def __init__(
        self,
        context: EpisodeRequestContext,
        ledger: SamplerSeedLedger,
        *,
        before_request: Callable[[], None] | None = None,
    ) -> None:
        self.context = context
        self.ledger = ledger
        self.before_request = before_request
        self.logical_counts = {"action": 0, "revision": 0}
        self.logical: dict[str, tuple[str, int]] = {}
        self.issued: set[tuple[str, int]] = set()
        self.records: list[dict[str, Any]] = []

    def issue(self, request: Any, budget: LogicalRequestBudget, stage: str) -> dict[str, Any]:
        if self.before_request is not None:
            self.before_request()
        context = self.context
        expected_table = f"{context.artifact_id}_{context.stage}_{context.episode_index:04d}"
        if (
            request.observation.agent_id != context.actor_id
            or request.observation.table_id != expected_table
        ):
            raise Path1TerminalError("state", "request_actor_or_episode_mismatch")
        if any(key in request.metadata for key in ("sampler_seed", "path1_request_identity")):
            raise Path1TerminalError("configuration", "metadata_cannot_override_request_identity")
        if "seed" in request.provider_config:
            raise Path1TerminalError("configuration", "request_cannot_override_sampler_seed")
        purpose = budget.purpose
        if purpose == "revision" and (
            context.actor_id != "agent_00" or context.stage != "formation"
        ):
            raise Path1TerminalError("state", "updater_request_outside_formation_hero")
        logical_id = budget.logical_request_id
        if logical_id not in self.logical:
            if stage != "primary":
                raise Path1TerminalError("state", "unbound_repair_request")
            self.logical[logical_id] = (purpose, self.logical_counts[purpose])
            self.logical_counts[purpose] += 1
        bound_purpose, logical_index = self.logical[logical_id]
        if bound_purpose != purpose:
            raise Path1TerminalError("state", "logical_request_purpose_changed")
        request_index = 0 if stage == "primary" else 1
        key = (logical_id, request_index)
        if key in self.issued or (request_index == 1 and (logical_id, 0) not in self.issued):
            raise Path1TerminalError("state", "duplicate_or_unbound_http_request")
        role = (
            "hero_revision"
            if purpose == "revision"
            else ("hero_action" if context.actor_id == "agent_00" else "opponent_action")
        )
        labels = {
            **asdict(context),
            "role": role,
            "logical_index": logical_index,
            "request_index": request_index,
            "request_kind": "primary" if request_index == 0 else "contract_repair",
        }
        seed = self.ledger.derive(labels)
        identity = {
            "schema": "path1-http-request-v1",
            **labels,
            "sampler_algorithm": SAMPLER_ALGORITHM,
            "sampler_seed": seed,
            "request_id": _sha(labels),
            "logical_request_id": logical_id,
            "repair_stage": stage,
            "backend_seed_execution_verified": False,
        }
        self.issued.add(key)
        self.records.append(identity)
        return dict(identity)


def require_path1_request_identity(client: object, expected: EpisodeRequestContext) -> None:
    from agentmemeval.path1_v1.llm.providers.openai_compatible import OpenAICompatibleClient
    from agentmemeval.path1_v1.llm.terminal_policy import require_path1_terminal_policy

    require_path1_terminal_policy(client)
    if (
        type(client) is not OpenAICompatibleClient
        or type(client._request_identity_binding) is not RequestIdentityBinding
    ):
        raise Path1TerminalError("configuration", "request_identity_not_bound")
    if client._request_identity_binding.context != expected:
        raise Path1TerminalError("state", "request_identity_context_mismatch")
