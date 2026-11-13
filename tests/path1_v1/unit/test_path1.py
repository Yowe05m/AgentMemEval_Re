from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from agentmemeval.path1_v1.agents.llm_agent import build_agent
from agentmemeval.path1_v1.core.domain import ActionDecision, HandResult
from agentmemeval.path1_v1.experiments import path1
from agentmemeval.path1_v1.memory.experiential import ExperientialMemory

HERO_CONFIG = {
    "mechanism": "expr", "memory_scope": "per_agent", "window_size": 30,
    "experience_revision_prompt_window_size": 2,
    "experience_revision_strategy": "llm", "experience_update_period": 1,
    "experience_semantics_mode": "quality_filtered_v2",
}


class LegalClient:
    provider = "offline_fixture"
    model = "fixture"

    def generate_structured(self, request, schema):
        assert schema is ActionDecision
        legal = request.observation.legal_actions.types()
        action = "check" if "check" in legal else "call" if "call" in legal else "fold"
        return ActionDecision(action, reason_summary="fixture", intent="fixture")


def factory(_agent_id, _seed):
    return LegalClient()


def design(**overrides):
    return path1.Path1Design("test_campaign", "artifact_00", "offline", 24511, **overrides)


def snapshot():
    return build_agent(path1.HERO_ID, HERO_CONFIG, LegalClient(), "fixture").snapshot_memory()


def synthetic_hand(**kwargs):
    agents, ids, stacks = kwargs["agents"], kwargs["agent_ids"], kwargs["stacks"]
    before = dict(stacks)
    hand_id = f'{kwargs["table_id"]}_{kwargs["hand_number"]}'
    rewards = dict.fromkeys(ids, 0)
    # One intentionally busted hero per episode verifies rebuy is excluded from reward.
    if kwargs["hand_number"] == 1:
        rewards[path1.HERO_ID] = -stacks[path1.HERO_ID]
        rewards["opponent_01"] = stacks[path1.HERO_ID]
    for key in ids:
        stacks[key] += rewards[key]
    result = HandResult(kwargs["table_id"], hand_id, rewards, dict(stacks), [], [], {})
    kwargs["on_settled"](result)
    hero = agents[path1.HERO_ID]
    if kwargs["update_memory"]:
        # Deliberate lifecycle fixture; not a real learning result.
        doc = copy.deepcopy(hero.memory.current)
        doc.version += 1
        doc.source_hand_ids = [hand_id]
        hero.memory.history.append(doc)
        hero.memory.trajectories.append(hand_id)
        hero.memory._atomic_write_experience(doc.body)
    kwargs["artifacts"].log_hand({
        "seed": kwargs["seed"], "deal_sha256": str(kwargs["seed"]),
        "starting_stacks": before, "final_stacks": dict(stacks),
    })
    return result


def test_full_300_hand_schedule_resets_transients_but_carries_final_memory(tmp_path, monkeypatch):
    instances = []
    starts = []

    def checked_hand(**kwargs):
        if kwargs["hand_number"] == 1:
            hero = kwargs["agents"][path1.HERO_ID]
            instances.append(kwargs["agents"])
            starts.append((hero.memory.current.version, list(hero.memory.trajectories)))
            assert set(kwargs["stacks"].values()) == {1000}
        return synthetic_hand(**kwargs)

    monkeypatch.setattr(path1, "run_single_hand", checked_hand)
    parent, receipt = path1.run_formation(
        output_dir=tmp_path / "formation", design=design(), hero_config=HERO_CONFIG,
        model="fixture", client_factory=factory,
    )
    assert starts == [(1 + 50 * i, []) for i in range(6)]
    assert parent.payload["history"][-1]["version"] == 301
    assert receipt["expected_hands"] == receipt["settled_hands"] == 300
    assert receipt["selection"] == "last_scheduled_update_before_reset"
    assert not receipt["primary_analysis_admitted"]
    for hand in (30, 90, 150, 300):
        checkpoint = json.loads((tmp_path / "formation" / f"checkpoint_hand_{hand:04d}.json")
                                .read_text(encoding="utf-8"))
        assert checkpoint["payload"]["history"][-1]["version"] == hand + 1
    assert len({id(group[path1.HERO_ID].memory) for group in instances}) == 6
    assert len({id(agent.llm_client) for group in instances for agent in group.values()}) == 48
    for index in range(6):
        ep = json.loads((tmp_path / "formation" / f"formation_{index:04d}_formation"
                         / "episode_receipt.json").read_text(encoding="utf-8"))
        assert ep["rewards"][path1.HERO_ID] == -1000
        assert ep["rebuys"][path1.HERO_ID] == 1000
        assert ep["hero_net_bb100"] == -1000


def test_removal_clears_history_catalog_and_diagnostics_without_changing_controller():
    parent = snapshot()
    parent.payload["history"][-1]["metadata"]["rule_catalog"] = [{"rule": "SECRET_SENTINEL"}]
    parent.payload["revision_log"] = [{"diagnostic": "SECRET_SENTINEL"}]
    parent.payload["skipped_trajectory_hand_ids"] = ["SECRET_SENTINEL"]
    original = copy.deepcopy(parent.to_dict())
    removed = path1.transform_snapshot(parent, remove=True)
    retained = path1.transform_snapshot(parent, remove=False)
    assert "SECRET_SENTINEL" not in json.dumps(removed.to_dict())
    assert "SECRET_SENTINEL" in json.dumps(retained.to_dict())
    assert parent.to_dict() == original
    restored = build_agent(path1.HERO_ID, HERO_CONFIG, LegalClient(), "fixture")
    restored.restore_memory(removed)
    assert restored.snapshot_memory().mechanism == "expr"
    assert restored.memory.current.version == 1
    assert not restored.memory.current.source_hand_ids
    parent.payload["unexpected_cache"] = {"text": "SECRET_SENTINEL"}
    with pytest.raises(ValueError, match="unreviewed"):
        path1.transform_snapshot(parent, remove=True)


def test_real_engine_pair_uses_same_actual_deals_and_is_arm_order_independent(tmp_path):
    parent = snapshot()
    original = copy.deepcopy(parent.to_dict())
    d = design(evaluation_episode_hands=2)
    results = []
    for index, order in enumerate((path1.ARMS, tuple(reversed(path1.ARMS)))):
        results.append(path1.run_evaluation_pair(
            output_dir=tmp_path / f"pair_{index}", design=d, hero_config=HERO_CONFIG,
            model="fixture", client_factory=factory, snapshot=parent,
            episode_index=3, arm_order=order,
        ))
    assert parent.to_dict() == original
    assert results[0]["settled_hands"] == 4
    for arm in path1.ARMS:
        assert results[0]["arms"][arm]["deals"] == results[1]["arms"][arm]["deals"]
        assert results[0]["arms"][arm]["hero_seat"] == 3
        assert results[0]["arms"][arm]["mechanism"] == "expr"
    assert results[0]["paired_difference_bb100"] == results[1]["paired_difference_bb100"] == 0


def test_pairing_allows_endogenous_stacks_to_differ(tmp_path, monkeypatch):
    original = synthetic_hand

    def divergent_hand(**kwargs):
        if kwargs["hand_metadata"]["arm"] == "removed":
            # Preserve the exogenous hand key while selecting a different policy outcome.
            kwargs["hand_number"] += 2
        return original(**kwargs)

    monkeypatch.setattr(path1, "run_single_hand", divergent_hand)
    result = path1.run_evaluation_pair(
        output_dir=tmp_path / "pair", design=design(evaluation_episode_hands=2),
        hero_config=HERO_CONFIG, model="fixture", client_factory=factory,
        snapshot=snapshot(), episode_index=0,
    )
    assert result["arms"]["retained"]["final_stacks"] != result["arms"]["removed"]["final_stacks"]
    assert result["paired_difference_bb100"] == -25000


def test_update_failure_preserves_settled_cash_flow_and_stops_unit(tmp_path, monkeypatch):
    def bad_update(_self, _trajectory, **_kwargs):
        raise RuntimeError("updater failed")

    monkeypatch.setattr(ExperientialMemory, "on_hand_finished", bad_update)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="updater failed"):
        path1.run_formation(
            output_dir=output, design=design(formation_episodes=2, formation_episode_hands=2),
            hero_config=HERO_CONFIG, model="fixture", client_factory=factory,
        )
    receipt = json.loads((output / "terminal_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "incomplete"
    assert receipt["primary_estimate"] is None
    assert receipt["settled_hands"] == 1
    assert receipt["completed_episodes"] == 0
    settled = (output / "settled_hands.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(settled) == 1
    assert sum(json.loads(settled[0])["rewards"].values()) == 0
    assert not (output / "artifact_snapshot.json").exists()


def test_action_failure_does_not_complete_or_retry(tmp_path):
    class BrokenClient:
        def generate_structured(self, *_args):
            raise RuntimeError("unparseable action after provider repair")

    output = tmp_path / "failed_action"
    with pytest.raises(RuntimeError, match="unparseable"):
        path1.run_formation(
            output_dir=output, design=design(formation_episodes=1, formation_episode_hands=1),
            hero_config=HERO_CONFIG, model="fixture",
            client_factory=lambda _agent, _seed: BrokenClient(),
        )
    receipt = json.loads((output / "terminal_receipt.json").read_text(encoding="utf-8"))
    assert receipt["settled_hands"] == 0 and receipt["primary_estimate"] is None
    assert receipt["retry_eligibility"] == "not_established"


def test_shared_client_and_existing_output_are_rejected(tmp_path):
    shared = LegalClient()
    arguments = dict(output_dir=tmp_path / "claimed", design=design(),
                     hero_config=HERO_CONFIG, model="fixture",
                     client_factory=lambda _agent, _seed: shared)
    with pytest.raises(ValueError, match="reused"):
        path1.run_formation(**arguments)
    before = (tmp_path / "claimed" / "terminal_receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        path1.run_formation(**arguments)
    assert (tmp_path / "claimed" / "terminal_receipt.json").read_bytes() == before


def test_seeds_separate_artifacts_phases_roles_and_reject_overlap():
    d = design()
    a = path1.SeedLedger(d)
    seed = a.get("evaluation", 0, "deal", 0)
    assert a.get("evaluation", 0, "deal", 0) == seed
    values = {seed, a.get("formation", 0, "deal", 0), a.get("evaluation", 0, "client", 0),
              path1.SeedLedger(replace(d, artifact_id="artifact_01")).get(
                  "evaluation", 0, "deal", 0),
              path1.SeedLedger(replace(d, phase="E1")).get("evaluation", 0, "deal", 0)}
    assert len(values) == 5
    with pytest.raises(ValueError, match="collision"):
        path1.SeedLedger(d, forbidden_seeds={seed}).get("evaluation", 0, "deal", 0)


@pytest.mark.parametrize("changes", [{"root_seed": True}, {"formation_episodes": 0},
                                    {"campaign_id": "../escape"}, {"table_size": 4}])
def test_invalid_design_rejected(changes):
    with pytest.raises(ValueError):
        replace(design(), **changes)


def test_research_design_rejects_shortened_formation():
    with pytest.raises(ValueError, match="6x50"):
        replace(design(formation_episode_hands=6), phase="E2")


def test_opponents_use_same_frozen_raise_sizing_policy(tmp_path, monkeypatch):
    def check(**kwargs):
        assert {agent.raise_sizing_policy for agent in kwargs["agents"].values()} == {
            "local_discrete",
        }
        return synthetic_hand(**kwargs)

    monkeypatch.setattr(path1, "run_single_hand", check)
    path1.run_formation(
        output_dir=tmp_path / "formation", design=design(
            formation_episodes=1, formation_episode_hands=1,
        ), hero_config={**HERO_CONFIG, "raise_sizing_policy": "local_discrete"},
        model="fixture", client_factory=factory,
    )


def test_json_publication_failure_after_write_has_no_final_complete_receipt(tmp_path, monkeypatch):
    def fail_sync(_fd):
        raise OSError("injected fsync failure after complete JSON write")

    monkeypatch.setattr(path1.os, "fsync", fail_sync)
    final = tmp_path / "terminal_receipt.json"
    with pytest.raises(OSError, match="injected fsync"):
        path1._new_json(final, {"status": "complete"})
    assert not final.exists()
    pending = list((tmp_path / "_publication_staging").iterdir())
    assert len(pending) == 1
    assert json.loads(pending[0].read_text(encoding="utf-8"))["status"] == "complete"


def test_new_json_never_overwrites_existing_final_receipt(tmp_path):
    final = tmp_path / "terminal_receipt.json"
    path1._new_json(final, {"status": "incomplete"})
    with pytest.raises(FileExistsError):
        path1._new_json(final, {"status": "complete"})
    assert json.loads(final.read_text(encoding="utf-8"))["status"] == "incomplete"


def test_live_phase_rejects_unverified_provider_before_any_request(tmp_path):
    from agentmemeval.path1_v1.llm.terminal_policy import Path1TerminalError

    output = tmp_path / "unverified_live"
    with pytest.raises(Path1TerminalError, match="unverified_provider_implementation"):
        path1.run_formation(
            output_dir=output,
            design=replace(design(formation_episodes=2, formation_episode_hands=6), phase="E1"),
            hero_config=HERO_CONFIG, model="fixture", client_factory=factory,
        )
    receipt = json.loads((output / "terminal_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "incomplete" and receipt["settled_hands"] == 0
