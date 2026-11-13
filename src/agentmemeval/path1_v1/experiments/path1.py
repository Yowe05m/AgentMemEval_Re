"""Episode lifecycle primitives for Path1; not a scientific admission entry point.

Workers must still validate the campaign/runtime contract and bind provider audit
and sampler settings. This module never infers admission from a successful run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentmemeval.path1_v1.agents.base import LLMDecisionAgent
from agentmemeval.path1_v1.agents.llm_agent import build_agent
from agentmemeval.path1_v1.core.domain import HandResult, MemorySnapshot
from agentmemeval.path1_v1.core.protocols import LLMClient
from agentmemeval.path1_v1.environment.holdem_adapter import DIRECT_RNG_ALGORITHM
from agentmemeval.path1_v1.experiments.table_play import run_single_hand
from agentmemeval.path1_v1.memory.experiential import ExperientialMemory
from agentmemeval.path1_v1.storage.artifacts import ArtifactManager

HERO_ID = "agent_00"
ARMS = ("retained", "removed")
READ_TELEMETRY = {"context_build_count", "last_injected_experience_version"}
PACKAGE_FIELDS = {"history", "revision_log", "skipped_trajectory_hand_ids"}
CONFIG_FIELDS = {
    "window_size", "max_chars", "prompt_token_budget", "revision_prompt_window_size",
    "update_period", "revision_strategy", "revision_model", "semantics_mode",
    "revision_contract", "document_contract", "failure_policy", "semantic_review_mode",
    "semantic_reviewer_model", "semantic_proposer_model_family", "semantic_reviewer_model_family",
    "revision_recap_version", "trajectory_schema_version",
}
ClientFactory = Callable[[str, int], LLMClient]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _new_json(path: Path, value: Any) -> None:
    # Publish only after successful write/fsync/close. Hard-link creation is
    # atomic and refuses an existing destination on both supported platforms.
    # Preserve the staged file as evidence; never delete pre-existing outputs.
    staging = path.parent / "_publication_staging"
    staging.mkdir(exist_ok=True)
    temporary = staging / f"{path.name}.{uuid4().hex}.pending"
    with temporary.open("xb") as handle:
        handle.write(_canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)


def _append_json(path: Path, value: Any) -> None:
    with path.open("ab") as handle:
        handle.write(_canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _component(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Path1 identifiers require safe ASCII path components")
    return value


@dataclass(frozen=True)
class Path1Design:
    campaign_id: str
    artifact_id: str
    phase: str
    root_seed: int
    seed_namespace: str = "legacy"
    formation_episodes: int = 6
    formation_episode_hands: int = 50
    evaluation_episode_hands: int = 50
    table_size: int = 8
    starting_stack: int = 1000
    small_blind: int = 1
    big_blind: int = 2
    max_raises_per_street: int = 4

    def __post_init__(self) -> None:
        _component(self.campaign_id)
        _component(self.artifact_id)
        _component(self.seed_namespace)
        if self.phase not in {"offline", "E1", "E2", "E3"}:
            raise ValueError("unknown Path1 phase")
        for field, value in asdict(self).items():
            if field in {"campaign_id", "artifact_id", "phase", "seed_namespace"}:
                continue
            if type(value) is not int or value < (0 if field == "root_seed" else 1):
                raise ValueError(f"{field} must be a valid integer")
        if self.table_size != 8 or (self.small_blind, self.big_blind) != (1, 2):
            raise ValueError("Path1 source requires eight seats and 1/2 blinds")
        if self.starting_stack != 1000 or self.max_raises_per_street != 4:
            raise ValueError("Path1 source stack/raise contract mismatch")
        if self.phase in {"E2", "E3"} and (
            self.formation_episodes, self.formation_episode_hands, self.evaluation_episode_hands
        ) != (6, 50, 50):
            raise ValueError("research lifecycle must be 6x50 formation and 50-hand evaluation")


class SeedLedger:
    """Label-derived streams; collisions fail closed, never silently share a stream."""

    def __init__(self, design: Path1Design, forbidden_seeds: set[int] | None = None):
        self.namespace = {
            "algorithm": "path1-sha256-63bit-v2",
            "campaign": design.campaign_id, "artifact": design.artifact_id,
            "phase": design.phase, "root_seed": design.root_seed,
            "seed_namespace": design.seed_namespace,
        }
        self.entries: dict[str, int] = {}
        self.owners: dict[int, str] = {}
        self.forbidden = set(forbidden_seeds or ())
        if any(type(value) is not int or not 0 <= value < 2**63 for value in self.forbidden):
            raise ValueError("invalid forbidden Path1 seed")

    def get(self, *labels: str | int) -> int:
        key = _canonical(list(labels)).decode()
        if any(type(label) not in {str, int} for label in labels):
            raise ValueError("Path1 seed labels must be exact strings or integers")
        value = int(_digest([self.namespace, list(labels)])[:16], 16) % 2**63
        if value in self.forbidden or (value in self.owners and self.owners[value] != key):
            raise ValueError("seed collision; freeze a new namespace before execution")
        self.owners[value] = key
        self.entries[key] = value
        return value


def transform_snapshot(parent: MemorySnapshot, *, remove: bool) -> MemorySnapshot:
    """Reset read telemetry and remove the whole package only in the removed arm.

    Empty history means restore the same Expr controller's initialization, not
    substitute NullMemory. Unknown fields require review rather than leaking.
    """
    if parent.mechanism != "expr" or parent.agent_id != HERO_ID or parent.scope != "per_agent":
        raise ValueError("Path1 requires the fixed hero's per-agent Expr snapshot")
    if set(parent.payload) != CONFIG_FIELDS | PACKAGE_FIELDS | READ_TELEMETRY:
        raise ValueError("unreviewed or missing Expr snapshot fields")
    snapshot = copy.deepcopy(parent)
    snapshot.payload["context_build_count"] = 0
    snapshot.payload["last_injected_experience_version"] = None
    if remove:
        for field in PACKAGE_FIELDS:
            snapshot.payload[field] = []
    return snapshot


def _frozen_state(agent: LLMDecisionAgent) -> dict[str, Any]:
    payload = copy.deepcopy(agent.snapshot_memory().payload)
    for field in READ_TELEMETRY:
        payload.pop(field)
    return payload


def _validate_hero_config(config: dict[str, Any]) -> None:
    if config.get("mechanism") != "expr" or config.get("memory_scope", "per_agent") != "per_agent":
        raise ValueError("Path1 uses one per-agent Expr controller in both arms")
    if config.get("experience_semantic_review_mode", "disabled") != "disabled":
        raise ValueError("Path1 has no separate semantic reviewer in this contract")


class _Session:
    def __init__(self, output: Path, design: Path1Design, hero_config: dict[str, Any],
                 model: str, client_factory: ClientFactory, forbidden_seeds: set[int] | None):
        _validate_hero_config(hero_config)
        self.output = output
        self.design = design
        self.hero_config = copy.deepcopy(hero_config)
        self.model = model
        self.client_factory = client_factory
        self.seeds = SeedLedger(design, forbidden_seeds)
        self.clients: list[LLMClient] = []  # Keep identities alive to detect reuse.
        self.episodes: list[dict[str, Any]] = []
        self.settled_hands = 0
        output.mkdir(parents=True, exist_ok=False)
        _new_json(output / "path1_manifest.json", {
            "schema": "path1-lifecycle-v1", "design": asdict(design),
            "hero_config": self.hero_config, "model": model,
            "admission": "not_admitted_by_lifecycle_library",
            "provider_seed_binding": "requires_worker_and_backend_verification",
            "source_config_sha256": _digest([asdict(design), hero_config, model]),
        })

    def agents(self, episode: int, arm: str, stage: str, artifacts: ArtifactManager,
               snapshot: MemorySnapshot | None) -> dict[str, LLMDecisionAgent]:
        agents = {}
        for agent_id in [HERO_ID, *[f"opponent_{i:02d}" for i in range(1, 8)]]:
            seed = self.seeds.get(stage, episode, arm, "client", agent_id)
            from agentmemeval.path1_v1.llm.request_identity import (
                EpisodeRequestContext,
                require_path1_request_identity,
            )

            request_context = EpisodeRequestContext(
                campaign_id=self.design.campaign_id, phase=self.design.phase,
                artifact_id=self.design.artifact_id, seed_namespace=self.design.seed_namespace,
                root_seed=self.design.root_seed, stage=stage, episode_index=episode,
                arm=arm, actor_id=agent_id, client_seed=seed,
            )
            episode_factory = getattr(self.client_factory, "create_episode_client", None)
            client = (
                episode_factory(request_context, artifacts.run_dir / "provider_exchanges.jsonl")
                if episode_factory is not None else self.client_factory(agent_id, seed)
            )
            if self.design.phase != "offline":
                from agentmemeval.path1_v1.llm.terminal_policy import require_path1_terminal_policy

                require_path1_terminal_policy(client)
                require_path1_request_identity(client, request_context)
            if any(client is previous for previous in self.clients):
                raise ValueError("client_factory reused a client across actors/episodes/arms")
            self.clients.append(client)
            config = self.hero_config if agent_id == HERO_ID else {
                "mechanism": "no_memory", "memory_scope": "per_agent",
                "raise_sizing_policy": self.hero_config.get(
                    "raise_sizing_policy", "native_no_limit",
                ),
            }
            agent = build_agent(agent_id, config, client, self.model)
            if agent_id == HERO_ID and snapshot is not None:
                agent.restore_memory(transform_snapshot(snapshot, remove=arm == "removed"))
            agent.set_memory_frozen(stage == "evaluation")
            agent.bind_memory_persistence(artifacts.agent_memory_directory(agent_id),
                                          reference_root=artifacts.run_dir)
            agents[agent_id] = agent
        memory = agents[HERO_ID].memory
        if not isinstance(memory, ExperientialMemory) or memory.trajectories:
            raise ValueError("fresh episode did not reset the hero trajectory window")
        return agents

    def episode(self, index: int, arm: str, stage: str, snapshot: MemorySnapshot | None,
                hero_seat: int) -> tuple[MemorySnapshot, dict[str, Any]]:
        d = self.design
        name = f"{stage}_{index:04d}_{arm}"
        count = d.formation_episode_hands if stage == "formation" else d.evaluation_episode_hands
        artifacts = ArtifactManager(self.output, name, {
            "experiment": {"scenario": "path1", "seed": d.root_seed},
            "provider": {"provider": "worker_client_factory", "model": self.model},
            "agent": self.hero_config,
        })
        agents = self.agents(index, arm, stage, artifacts, snapshot)
        agent_ids = [f"opponent_{i:02d}" for i in range(1, 8)]
        agent_ids.insert(hero_seat, HERO_ID)
        stacks = dict.fromkeys(agent_ids, d.starting_stack)
        rewards = dict.fromkeys(agent_ids, 0)
        rebuys = dict.fromkeys(agent_ids, 0)
        settlement_count_before = self.settled_hands
        frozen_before = _frozen_state(agents[HERO_ID])
        before_snapshot = agents[HERO_ID].snapshot_memory()
        _new_json(artifacts.run_dir / "initial_snapshot.json", before_snapshot.to_dict())
        identity = {"campaign_id": d.campaign_id, "phase": d.phase,
                    "artifact_id": d.artifact_id, "episode_index": index,
                    "arm": arm, "hero_seat": hero_seat, "stage": stage}
        artifacts.log_event({"event": "episode_start", **identity, "stacks": dict(stacks),
                             "trajectory_count": 0, "updater_enabled": stage == "formation"})
        for hand in range(count):
            before = dict(stacks)
            seed = self.seeds.get(stage, index, "deal", hand)
            # Different policies may produce different subsequent stacks and
            # observations. Pair only the exogenous deal and episode initial state.
            def settled(result: HandResult, hand_index: int = hand,
                        starting: dict[str, int] = before, deal_seed: int = seed) -> None:
                if (set(result.rewards) != set(agent_ids)
                        or set(result.final_stacks) != set(agent_ids)):
                    raise ValueError("settlement roster mismatch")
                if any(type(v) is not int or v < 0 for v in result.final_stacks.values()):
                    raise ValueError("invalid settlement stack")
                if any(type(v) is not int for v in result.rewards.values()):
                    raise ValueError("invalid reward type")
                if sum(result.rewards.values()) != 0 or any(
                    result.final_stacks[a] - starting[a] != result.rewards[a] for a in agent_ids
                ):
                    raise ValueError("cash flow does not conserve chips")
                _append_json(self.output / "settled_hands.jsonl", {
                    **identity, "hand_index": hand_index, "hand_id": result.hand_id,
                    "deal_seed": deal_seed, "actual_rng_seed": deal_seed,
                    "dealer_index": hand_index % d.table_size,
                    "rewards": result.rewards, "starting_stacks": starting,
                    "final_stacks": result.final_stacks, "memory_update": "not_yet_attempted",
                })
                self.settled_hands += 1
                for a in agent_ids:
                    rewards[a] += result.rewards[a]

            transition = {**identity, "hand_index": hand, "updater_enabled": stage == "formation"}
            try:
                run_single_hand(
                    agents=agents, table_id=f"{d.artifact_id}_{stage}_{index:04d}",
                    agent_ids=agent_ids, stacks=stacks, seed=seed,
                    direct_rng_seed=seed, stage=stage,
                    small_blind=d.small_blind, big_blind=d.big_blind,
                    max_raises_per_street=d.max_raises_per_street,
                    update_memory=stage == "formation", artifacts=artifacts,
                    dealer_index=hand % d.table_size, hand_number=hand + 1,
                    hand_metadata=identity, on_settled=settled,
                )
            except BaseException as exc:
                _append_json(self.output / "hand_lifecycle.jsonl", {
                    **transition, "status": "failed", "failure_type": type(exc).__name__,
                    "failure_stage": "inspect_settlement_and_provider_evidence",
                })
                raise
            _append_json(self.output / "hand_lifecycle.jsonl", {
                **transition, "status": "returned_after_update_boundary",
            })
            formation_hand = index * d.formation_episode_hands + hand + 1
            if stage == "formation" and formation_hand in {30, 90, 150, 300}:
                _new_json(self.output / f"checkpoint_hand_{formation_hand:04d}.json",
                          agents[HERO_ID].snapshot_memory().to_dict())
            for agent_id, stack in list(stacks.items()):
                if stack == 0:
                    stacks[agent_id] = d.starting_stack
                    rebuys[agent_id] += d.starting_stack
                    artifacts.log_event({"event": "rebuy", **identity, "after_hand": hand + 1,
                                         "agent_id": agent_id, "stack_added": d.starting_stack})
        if any(stacks[a] - d.starting_stack - rebuys[a] != rewards[a] for a in agent_ids):
            raise ValueError("episode rebuy accounting mismatch")
        if self.settled_hands - settlement_count_before != count:
            raise ValueError("episode settlement count mismatch")
        if stage == "evaluation" and frozen_before != _frozen_state(agents[HERO_ID]):
            raise ValueError("frozen hero package mutated during evaluation")
        if stage == "evaluation" and agents[HERO_ID].memory.trajectories:
            raise ValueError("frozen evaluation collected trajectories")
        result_snapshot = agents[HERO_ID].snapshot_memory()
        _new_json(artifacts.run_dir / "final_snapshot.json", result_snapshot.to_dict())
        rows = artifacts.hand_summaries.read_all()
        if len(rows) != count:
            raise ValueError("missing completed hand summaries")
        if d.phase != "offline" and any(
            type(row.get("actual_rng_seed")) is not int
            or row["actual_rng_seed"] != self.seeds.get(stage, index, "deal", hand)
            or row.get("actual_rng_algorithm") != DIRECT_RNG_ALGORITHM
            for hand, row in enumerate(rows)
        ):
            raise ValueError("actual environment RNG binding is missing or differs")
        receipt = {**identity, "status": "complete", "hands": count, "rewards": rewards,
                   "rebuys": rebuys, "final_stacks": stacks,
                   "hero_net_bb100": rewards[HERO_ID] * 100 / (d.big_blind * count),
                   "deals": [row["deal_sha256"] for row in rows],
                   "deal_seeds": [row["seed"] for row in rows],
                   "actual_rng_seeds": [row.get("actual_rng_seed") for row in rows],
                   "initial_stacks": rows[0]["starting_stacks"],
                   "controller": type(agents[HERO_ID]).__name__,
                   "mechanism": agents[HERO_ID].snapshot_memory().mechanism,
                   "allowed_frozen_changes": sorted(READ_TELEMETRY),
                   "final_snapshot_sha256": _digest(result_snapshot.to_dict())}
        _new_json(artifacts.run_dir / "episode_receipt.json", receipt)
        self.episodes.append(receipt)
        return result_snapshot, receipt

    def terminal(self, status: str, **extra: Any) -> dict[str, Any]:
        receipt = {"status": status, "artifact_id": self.design.artifact_id,
                   "settled_hands": self.settled_hands, "completed_episodes": len(self.episodes),
                   "primary_analysis_admitted": False, **extra}
        _new_json(self.output / "seed_manifest.json", {
            "namespace": self.seeds.namespace, "streams": self.seeds.entries,
            "environment_rng_binding": "direct_rng_seed_without_secondary_hash",
            "actual_rng_algorithm": DIRECT_RNG_ALGORITHM,
            "backend_binding_verified": False,
        })
        _new_json(self.output / "terminal_receipt.json", receipt)
        return receipt


def run_formation(*, output_dir: Path, design: Path1Design, hero_config: dict[str, Any],
                  model: str, client_factory: ClientFactory,
                  forbidden_seeds: set[int] | None = None) -> tuple[MemorySnapshot, dict[str, Any]]:
    """Form one artifact; no best-checkpoint selection or successful-attempt retry."""
    session = _Session(
        Path(output_dir), design, hero_config, model, client_factory, forbidden_seeds,
    )
    parent = None
    try:
        for index in range(design.formation_episodes):
            parent, _ = session.episode(index, "formation", "formation", parent, 0)
        assert parent is not None
        _new_json(session.output / "artifact_snapshot.json", parent.to_dict())
    except BaseException as exc:
        session.terminal("incomplete", failure_type=type(exc).__name__,
                         primary_estimate=None, retry_eligibility="not_established")
        raise
    receipt = session.terminal("complete", selection="last_scheduled_update_before_reset",
                               snapshot_sha256=_digest(parent.to_dict()),
                               expected_hands=design.formation_episodes
                               * design.formation_episode_hands)
    return parent, receipt


def run_evaluation_pair(*, output_dir: Path, design: Path1Design,
                        hero_config: dict[str, Any], model: str, client_factory: ClientFactory,
                        snapshot: MemorySnapshot, episode_index: int,
                        arm_order: tuple[str, str] = ARMS,
                        forbidden_seeds: set[int] | None = None) -> dict[str, Any]:
    """Run both arms of one complete pair; statistical aggregation is separate."""
    if type(episode_index) is not int or episode_index < 0 or set(arm_order) != set(ARMS):
        raise ValueError("invalid pair identity/order")
    if len(arm_order) != 2:
        raise ValueError("each pair must contain each arm once")
    parent_hash = _digest(snapshot.to_dict())
    transform_snapshot(snapshot, remove=False)  # Validate before creating any output.
    session = _Session(
        Path(output_dir), design, hero_config, model, client_factory, forbidden_seeds,
    )
    rows = {}
    try:
        for arm in arm_order:
            _, rows[arm] = session.episode(episode_index, arm, "evaluation", snapshot,
                                           episode_index % design.table_size)
        first, second = rows["retained"], rows["removed"]
        for field in (
            "deals", "deal_seeds", "actual_rng_seeds", "initial_stacks", "hero_seat", "hands",
        ):
            if first[field] != second[field]:
                raise ValueError(f"evaluation exogenous pairing mismatch: {field}")
        if _digest(snapshot.to_dict()) != parent_hash:
            raise ValueError("evaluation mutated its source artifact")
    except BaseException as exc:
        session.terminal("incomplete", failure_type=type(exc).__name__,
                         primary_estimate=None, retry_eligibility="not_established")
        raise
    return session.terminal("complete", episode_index=episode_index,
                            snapshot_sha256=parent_hash,
                            paired_difference_bb100=first["hero_net_bb100"]
                            - second["hero_net_bb100"], arms=rows)
