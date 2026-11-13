from __future__ import annotations

import copy
import json
from dataclasses import asdict

import pytest

from agentmemeval.path1_v1.agents.llm_agent import build_agent
from agentmemeval.path1_v1.analysis.path1_evidence import (
    EvidenceError,
    audit_evaluation_pair,
    file_inventory,
)
from agentmemeval.path1_v1.core.domain import ActionDecision
from agentmemeval.path1_v1.experiments.path1 import Path1Design, run_evaluation_pair


class LegalClient:
    provider = "offline_fixture"
    model = "fixture"

    def generate_structured(self, request, schema):
        assert schema is ActionDecision
        legal = request.observation.legal_actions.types()
        action = "check" if "check" in legal else "call" if "call" in legal else "fold"
        return ActionDecision(action, reason_summary="fixture", intent="fixture")


@pytest.fixture
def evidence(tmp_path):
    config = {
        "mechanism": "expr",
        "window_size": 30,
        "experience_revision_prompt_window_size": 2,
        "experience_revision_strategy": "llm",
    }
    agent = build_agent("agent_00", config, LegalClient(), "fixture")
    snapshot = agent.snapshot_memory()
    source = tmp_path / "source_snapshot.json"
    source.write_text(json.dumps(snapshot.to_dict()), encoding="utf-8")
    design = Path1Design(
        "audit_fixture", "artifact_00", "offline", 11999, evaluation_episode_hands=2
    )
    root = tmp_path / "pair"
    run_evaluation_pair(
        output_dir=root,
        design=design,
        hero_config=config,
        model="fixture",
        client_factory=lambda _agent, _seed: LegalClient(),
        snapshot=snapshot,
        episode_index=0,
    )
    return root, source, asdict(design)


def audit(evidence, **kwargs):
    root, source, design = evidence
    return audit_evaluation_pair(
        root, source_snapshot_path=source, expected_design=design, **kwargs
    )


def change_json(path, transform):
    value = json.loads(path.read_text(encoding="utf-8"))
    transform(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def change_jsonl(path, transform):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    transform(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_recomputes_real_engine_pair_without_granting_scientific_admission(evidence):
    before = file_inventory(evidence[0])
    result = audit(evidence, expected_inventory=before)
    assert result["lifecycle_consistent"] and result["recovery_inventory_verified"]
    assert not result["scientific_admission"]
    assert result["hands_per_arm"] == 2
    assert result["difference_bb100"] == 0 and result["exact_difference"] == [0, 1]
    assert file_inventory(evidence[0]) == before
    assert not audit(evidence)["recovery_inventory_verified"]


@pytest.mark.parametrize("status", ["partial", "incomplete", "failed"])
def test_incomplete_unit_never_yields_complete_case_estimate(evidence, status):
    change_json(evidence[0] / "terminal_receipt.json", lambda data: data.update(status=status))
    with pytest.raises(EvidenceError, match="incomplete"):
        audit(evidence)


def test_fabricated_summary_mean_rejected(evidence):
    change_json(
        evidence[0] / "terminal_receipt.json", lambda data: data.update(paired_difference_bb100=999)
    )
    with pytest.raises(EvidenceError, match="recomputed cash flow"):
        audit(evidence)


def test_missing_settled_hand_rejected(evidence):
    change_jsonl(evidence[0] / "settled_hands.jsonl", lambda rows: rows.pop())
    with pytest.raises(EvidenceError, match="denominator"):
        audit(evidence)


def test_reward_and_stack_inconsistency_rejected(evidence):
    path = evidence[0] / "evaluation_0000_retained/hand_summaries.jsonl"
    change_jsonl(path, lambda rows: rows[0]["rewards"].update(agent_00=12345))
    with pytest.raises(EvidenceError, match="conserve chips"):
        audit(evidence)


def test_rebuy_event_cannot_inflate_net_reward(evidence):
    root, _, design = evidence
    event = {
        "event": "rebuy",
        "campaign_id": design["campaign_id"],
        "phase": design["phase"],
        "artifact_id": design["artifact_id"],
        "episode_index": 0,
        "arm": "retained",
        "hero_seat": 0,
        "stage": "evaluation",
        "after_hand": 1,
        "agent_id": "agent_00",
        "stack_added": 1000,
    }
    with (root / "evaluation_0000_retained/events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    with pytest.raises(EvidenceError, match="rebuy event"):
        audit(evidence)


def test_same_seed_claim_with_different_actual_deal_rejected(evidence):
    root = evidence[0]
    fake = "a" * 64
    change_jsonl(
        root / "evaluation_0000_removed/hand_summaries.jsonl",
        lambda rows: rows[0].update(deal_sha256=fake),
    )
    change_json(
        root / "evaluation_0000_removed/episode_receipt.json",
        lambda value: value["deals"].__setitem__(0, fake),
    )
    change_json(
        root / "terminal_receipt.json",
        lambda value: value["arms"]["removed"]["deals"].__setitem__(0, fake),
    )
    with pytest.raises(EvidenceError, match="independent RNG replay"):
        audit(evidence)


@pytest.mark.parametrize("field", ["campaign", "artifact", "phase", "root_seed", "seed_namespace"])
def test_seed_manifest_namespace_is_bound_to_expected_design(evidence, field):
    change_json(
        evidence[0] / "seed_manifest.json",
        lambda value: value["namespace"].update({field: "another_namespace"}),
    )
    with pytest.raises(EvidenceError, match="seed namespace"):
        audit(evidence)


@pytest.mark.parametrize("change", ["missing", "extra", "float", "value"])
def test_seed_stream_inventory_is_independently_rederived(evidence, change):
    def alter(value):
        streams = value["streams"]
        key = next(iter(streams))
        if change == "missing":
            streams.pop(key)
        elif change == "extra":
            streams['["unexpected"]'] = 7
        elif change == "float":
            streams[key] = float(streams[key])
        else:
            streams[key] += 1

    change_json(evidence[0] / "seed_manifest.json", alter)
    with pytest.raises(EvidenceError, match="seed stream ledger"):
        audit(evidence)


@pytest.mark.parametrize("field,value", [
    ("actual_rng_seed", 7),
    ("actual_rng_seed", True),
    ("actual_rng_algorithm", "legacy-sha256-table-id-int32-v1"),
])
def test_raw_engine_rng_identity_cannot_be_self_reported_as_legacy(evidence, field, value):
    change_jsonl(
        evidence[0] / "evaluation_0000_retained/hand_summaries.jsonl",
        lambda rows: rows[0].update({field: value}),
    )
    with pytest.raises(EvidenceError, match="engine RNG|invalid integer"):
        audit(evidence)


def test_two_arms_with_same_forged_deal_still_fail_independent_replay(evidence):
    root = evidence[0]
    fake = "a" * 64
    for arm in ("retained", "removed"):
        change_jsonl(
            root / f"evaluation_0000_{arm}/hand_summaries.jsonl",
            lambda rows: rows[0].update(deal_sha256=fake),
        )
        change_json(
            root / f"evaluation_0000_{arm}/episode_receipt.json",
            lambda value: value["deals"].__setitem__(0, fake),
        )
        change_json(
            root / "terminal_receipt.json",
            lambda value, arm=arm: value["arms"][arm]["deals"].__setitem__(0, fake),
        )
    with pytest.raises(EvidenceError, match="independent RNG replay"):
        audit(evidence)


def test_stale_source_and_cross_campaign_rejected(evidence):
    root, source, design = evidence
    changed = copy.deepcopy(design)
    changed["campaign_id"] = "another_campaign"
    with pytest.raises(EvidenceError, match="unexpected design"):
        audit_evaluation_pair(root, source_snapshot_path=source, expected_design=changed)
    change_json(source, lambda value: value["payload"]["history"][0].update(body="tampered"))
    with pytest.raises(EvidenceError, match="source snapshot identity"):
        audit(evidence)


def test_updater_enabled_in_frozen_arm_rejected(evidence):
    change_jsonl(
        evidence[0] / "hand_lifecycle.jsonl", lambda rows: rows[0].update(updater_enabled=True)
    )
    with pytest.raises(EvidenceError, match="update boundary"):
        audit(evidence)


def test_snapshot_mutation_rejected_before_aggregate(evidence):
    change_json(
        evidence[0] / "evaluation_0000_retained/final_snapshot.json",
        lambda value: value["payload"].update(revision_log=[{"unauthorized": True}]),
    )
    with pytest.raises(EvidenceError, match="frozen package"):
        audit(evidence)


def test_recovery_inventory_rejects_extra_or_changed_file(evidence):
    inventory = file_inventory(evidence[0])
    (evidence[0] / "unlisted.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(EvidenceError, match="recovery inventory"):
        audit(evidence, expected_inventory=inventory)


@pytest.mark.parametrize("tail", [b'{"unfinished":', b"\n"])
def test_incomplete_or_blank_jsonl_tail_rejected(evidence, tail):
    with (evidence[0] / "settled_hands.jsonl").open("ab") as handle:
        handle.write(tail)
    with pytest.raises(EvidenceError, match="JSON"):
        audit(evidence)


def test_duplicate_json_keys_rejected(evidence):
    (evidence[0] / "terminal_receipt.json").write_text(
        '{"status":"complete","status":"failed"}',
        encoding="utf-8",
    )
    with pytest.raises(EvidenceError, match="duplicate JSON"):
        audit(evidence)


def test_bool_episode_identity_and_nonfinite_json_rejected(evidence):
    path = evidence[0] / "settled_hands.jsonl"
    before = path.read_bytes()
    change_jsonl(path, lambda rows: rows[0].update(episode_index=False))
    with pytest.raises(EvidenceError, match="identity"):
        audit(evidence)
    path.write_bytes(before)
    (evidence[0] / "terminal_receipt.json").write_text('{"number":1e999}', encoding="utf-8")
    with pytest.raises(EvidenceError, match="non-finite"):
        audit(evidence)


def test_known_cashflow_fixture_excludes_legitimate_rebuy_from_gain(evidence):
    # Synthetic accounting fixture, not a claimed replay of these poker actions.
    root, _, design = evidence
    arm_dir = root / "evaluation_0000_retained"
    rows = [
        json.loads(line)
        for line in (arm_dir / "hand_summaries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ids = rows[0]["agent_ids"]
    zero = dict.fromkeys(ids, 0)
    reward = {**zero, "agent_00": -1000, "opponent_01": 1000}
    ended = {**dict.fromkeys(ids, 1000), "agent_00": 0, "opponent_01": 2000}
    replenished = {**ended, "agent_00": 1000}
    rows[0].update(rewards=reward, final_stacks=ended)
    rows[1].update(starting_stacks=replenished, rewards=zero, final_stacks=replenished)
    change_jsonl(
        arm_dir / "hand_summaries.jsonl", lambda target: target.__setitem__(slice(None), rows)
    )

    def settled(target):
        for item in target:
            if item["arm"] == "retained":
                row = rows[item["hand_index"]]
                for key in ("starting_stacks", "rewards", "final_stacks"):
                    item[key] = row[key]

    change_jsonl(root / "settled_hands.jsonl", settled)
    rebuy = {
        "event": "rebuy",
        "campaign_id": design["campaign_id"],
        "phase": design["phase"],
        "artifact_id": design["artifact_id"],
        "episode_index": 0,
        "arm": "retained",
        "hero_seat": 0,
        "stage": "evaluation",
        "after_hand": 1,
        "agent_id": "agent_00",
        "stack_added": 1000,
    }
    with (arm_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rebuy) + "\n")
    change_json(
        arm_dir / "episode_receipt.json",
        lambda item: item.update(
            rewards=reward,
            rebuys={**zero, "agent_00": 1000},
            final_stacks=replenished,
            hero_net_bb100=-25000.0,
        ),
    )
    receipt = json.loads((arm_dir / "episode_receipt.json").read_text(encoding="utf-8"))

    def terminal(item):
        item["arms"]["retained"] = receipt
        item["paired_difference_bb100"] = -25000.0 - item["arms"]["removed"]["hero_net_bb100"]

    change_json(root / "terminal_receipt.json", terminal)
    result = audit(evidence)
    assert result["net_chips"]["retained"] == -1000
    assert result["difference_bb100"] == (-1000 - result["net_chips"]["removed"]) * 25


@pytest.mark.parametrize(
    "field", ["settled_reward", "receipt_reward", "lifecycle_index", "receipt_seed"]
)
def test_integer_ledgers_reject_equal_valued_float_or_bool(evidence, field):
    root = evidence[0]
    if field == "settled_reward":
        change_jsonl(
            root / "settled_hands.jsonl",
            lambda rows: rows[0]["rewards"].update(
                agent_00=float(rows[0]["rewards"]["agent_00"]),
            ),
        )
    elif field == "lifecycle_index":
        change_jsonl(root / "hand_lifecycle.jsonl", lambda rows: rows[0].update(hand_index=False))
    else:
        path = root / "evaluation_0000_retained/episode_receipt.json"

        def alter(item):
            if field == "receipt_reward":
                item["rewards"]["agent_00"] = float(item["rewards"]["agent_00"])
            else:
                item["deal_seeds"][0] = float(item["deal_seeds"][0])

        change_json(path, alter)
        value = json.loads(path.read_text(encoding="utf-8"))
        change_json(
            root / "terminal_receipt.json", lambda item: item["arms"].update(retained=value)
        )
    with pytest.raises(EvidenceError, match="invalid integer"):
        audit(evidence)


def test_null_hand_identifier_is_not_accepted_as_a_join_key(evidence):
    root = evidence[0]
    change_jsonl(
        root / "evaluation_0000_retained/hand_summaries.jsonl",
        lambda rows: rows[0].update(hand_id=None),
    )
    change_jsonl(root / "settled_hands.jsonl", lambda rows: rows[0].update(hand_id=None))
    with pytest.raises(EvidenceError, match="logical hand ID"):
        audit(evidence)
