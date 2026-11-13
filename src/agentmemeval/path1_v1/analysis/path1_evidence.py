"""Read-only cash-flow and paired-lifecycle audit, independent of the runner.

Passing these checks is not scientific admission: provider/runtime verification,
complete cohort accounting, external provenance and formal statistics are separate.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from fractions import Fraction
from pathlib import Path
from typing import Any

from agentmemeval.path1_v1.memory.experiential import INITIAL_EXPERIENCE

DIRECT_RNG = "python-random-direct-int63-v1"
DERIVED_SEEDS = "path1-sha256-63bit-v2"

HERO = "agent_00"
ARMS = ("retained", "removed")
TELEMETRY = {"context_build_count", "last_injected_experience_version"}
PACKAGE = {"history", "revision_log", "skipped_trajectory_hand_ids"}


class EvidenceError(ValueError):
    """Missing, inconsistent or corrupted evidence; never convert into zero gain."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        _require(key not in value, "duplicate JSON object key")
        value[key] = item
    return value


def _bad_constant(_value: str) -> None:
    raise EvidenceError("non-finite JSON number")


def _decode(data: bytes) -> Any:
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_object_pairs, parse_constant=_bad_constant
        )
        _finite_numbers(value)
        return value
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("invalid UTF-8 JSON evidence") from exc


def _finite_numbers(value: Any) -> None:
    if type(value) is float:
        _require(math.isfinite(value), "non-finite JSON number")
    elif type(value) is list:
        for item in value:
            _finite_numbers(item)
    elif type(value) is dict:
        for item in value.values():
            _finite_numbers(item)


def _hash_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _same_value(actual: Any, expected: Any) -> bool:
    # Python equality treats True/1/1.0 as interchangeable; evidence does not.
    return _hash_json(actual) == _hash_json(expected)


def _read(path: Path) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"missing/linked evidence: {path.name}")
    return path.read_bytes()


def _json(path: Path) -> dict[str, Any]:
    value = _decode(_read(path))
    _require(type(value) is dict, "JSON evidence must be an object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    data = _read(path)
    _require(not data or data.endswith(b"\n"), "incomplete JSONL tail")
    values = [_decode(line) for line in data.splitlines()]
    _require(all(type(value) is dict for value in values), "JSONL rows must be objects")
    return values


def _integer(value: Any, minimum: int | None = None) -> int:
    _require(type(value) is int and (minimum is None or value >= minimum), "invalid integer")
    return value


def _money(value: Any, ids: set[str], *, nonnegative: bool) -> dict[str, int]:
    _require(type(value) is dict and set(value) == ids, "cash-flow roster mismatch")
    for amount in value.values():
        _integer(amount, 0 if nonnegative else None)
    return value


def _same_number(actual: Any, expected: Fraction) -> None:
    _require(
        type(actual) in (int, float)
        and math.isfinite(actual)
        and math.isclose(actual, float(expected), abs_tol=1e-12, rel_tol=1e-12),
        "reported BB/100 differs from independently recomputed cash flow",
    )


def _assigned_pair_streams(design: dict, episode: int, manifest: dict) -> list[int]:
    """Independently reconstruct v2 labels; do not trust the runner's ledger."""
    namespace = {
        "algorithm": DERIVED_SEEDS,
        "campaign": design["campaign_id"],
        "artifact": design["artifact_id"],
        "phase": design["phase"],
        "root_seed": _integer(design["root_seed"], 0),
        "seed_namespace": design["seed_namespace"],
    }
    _require(_same_value(manifest.get("namespace"), namespace), "seed namespace differs")
    expected = {}

    def derive(labels: list) -> int:
        key = json.dumps(labels, sort_keys=True, separators=(",", ":"), allow_nan=False)
        value = int(_hash_json([namespace, labels])[:16], 16) % 2**63
        expected[key] = value
        return value

    deals = [
        derive(["evaluation", episode, "deal", hand])
        for hand in range(_integer(design["evaluation_episode_hands"], 1))
    ]
    for arm in ARMS:
        for actor in [HERO, *[f"opponent_{i:02d}" for i in range(1, 8)]]:
            derive(["evaluation", episode, arm, "client", actor])
    _require(len(set(expected.values())) == len(expected), "assigned seed collision")
    _require(_same_value(manifest.get("streams"), expected), "seed stream ledger differs")
    return deals


def _deal_digest(seed: int, dealer: int) -> str:
    # This is an independent replay of the frozen deal contract, not a call to
    # the runner or HoldemEnvironment. Engine records alone cannot prove binding.
    deck = [rank + suit for rank in "23456789TJQKA" for suit in "cdhs"]
    random.Random(seed).shuffle(deck)
    hole_cards = [[deck.pop(), deck.pop()] for _ in range(8)]
    return _hash_json({
        "hole_cards_by_seat": hole_cards,
        "remaining_deck": deck,
        "dealer_index": dealer,
    })


def file_inventory(root: str | Path) -> list[dict[str, Any]]:
    """Inventory every file, including preserved publication stages; do not modify it."""
    root = Path(root)
    _require(root.is_dir() and not root.is_symlink(), "run root must be a real directory")
    records = []
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), "run inventory contains a symlink")
        if path.is_file():
            data = path.read_bytes()
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return records


def _identity(row: dict[str, Any], design: dict[str, Any], episode: int, arm: str) -> None:
    expected = {
        "campaign_id": design["campaign_id"],
        "phase": design["phase"],
        "artifact_id": design["artifact_id"],
        "episode_index": episode,
        "arm": arm,
        "hero_seat": episode % 8,
        "stage": "evaluation",
    }
    _require(
        all(
            type(row.get(key)) is type(value) and row[key] == value
            for key, value in expected.items()
        ),
        "evidence identity mismatch",
    )


def _snapshot_check(initial: dict, final: dict, parent: dict, arm: str) -> None:
    for value in (initial, final, parent):
        _require(
            value.get("mechanism") == "expr"
            and value.get("agent_id") == HERO
            and value.get("scope") == "per_agent",
            "wrong snapshot controller",
        )
        _require(type(value.get("payload")) is dict, "invalid snapshot payload")
    before, after, source = initial["payload"], final["payload"], parent["payload"]
    _require(set(before) == set(after) == set(source), "snapshot field set changed")
    _require(
        _integer(before.get("context_build_count"), 0) == 0
        and before.get("last_injected_experience_version") is None,
        "episode inherited stale read telemetry",
    )
    _require(
        _same_value(
            {k: v for k, v in before.items() if k not in TELEMETRY},
            {k: v for k, v in after.items() if k not in TELEMETRY},
        ),
        "frozen package/config mutated",
    )
    _require(
        _same_value(
            {k: v for k, v in before.items() if k not in TELEMETRY | PACKAGE},
            {k: v for k, v in source.items() if k not in TELEMETRY | PACKAGE},
        ),
        "controller configuration differs between source and evaluation",
    )
    if arm == "retained":
        _require(
            all(_same_value(before[k], source[k]) for k in PACKAGE), "retained package differs"
        )
    else:
        history = before.get("history")
        _require(type(history) is list and len(history) == 1, "removed history was not reset")
        doc = history[0]
        _require(
            doc.get("body") == INITIAL_EXPERIENCE
            and type(doc.get("version")) is int
            and doc.get("version") == 1
            and doc.get("source_hand_ids") == [],
            "removed document leaked learned content",
        )
        _require(
            before.get("revision_log") == [] and before.get("skipped_trajectory_hand_ids") == [],
            "removed package leaked diagnostics",
        )
        allowed_metadata = {
            "reason",
            "structure",
            "document_contract_version",
            "semantics_mode",
            "revision_contract",
            "failure_policy",
            "semantic_review_mode",
            "semantic_reviewer_model",
        }
        _require(
            type(doc.get("metadata")) is dict
            and set(doc["metadata"]) == allowed_metadata
            and doc["metadata"].get("reason") == "initial",
            "removed metadata leaked state",
        )


def _audit_arm(
    root: Path,
    design: dict,
    parent: dict,
    terminal: dict,
    episode: int,
    arm: str,
    settlements: list[dict],
    lifecycle: list[dict],
    assigned_deals: list[int],
) -> dict[str, Any]:
    directory = root / f"evaluation_{episode:04d}_{arm}"
    _require(directory.is_dir() and not directory.is_symlink(), "missing/linked arm directory")
    receipt = _json(directory / "episode_receipt.json")
    _require(_same_value(receipt, terminal["arms"][arm]), "terminal and arm receipt disagree")
    _identity(receipt, design, episode, arm)
    _require(
        receipt.get("status") == "complete"
        and receipt.get("mechanism") == "expr"
        and receipt.get("controller") == "LLMDecisionAgent",
        "invalid arm completion",
    )
    count = _integer(design["evaluation_episode_hands"], 1)
    _require(_integer(receipt.get("hands"), 1) == count, "receipt hand count mismatch")
    hands = _jsonl(directory / "hand_summaries.jsonl")
    events = _jsonl(directory / "events.jsonl")
    _require(len(hands) == len(settlements) == len(lifecycle) == count, "incomplete hand ledger")
    ids_order = [f"opponent_{i:02d}" for i in range(1, 8)]
    ids_order.insert(episode % 8, HERO)
    ids = set(ids_order)
    prior = dict.fromkeys(ids, _integer(design["starting_stack"], 1))
    total_reward, total_rebuy = dict.fromkeys(ids, 0), dict.fromkeys(ids, 0)
    rebuy_events = [e for e in events if e.get("event") == "rebuy"]
    expected_rebuys = []
    deals, seeds, hand_ids = [], [], set()
    for index, (hand, settled, life) in enumerate(zip(hands, settlements, lifecycle, strict=True)):
        for row in (hand, settled, life):
            _identity(row, design, episode, arm)
        _require(
            hand.get("agent_ids") == ids_order
            and _integer(hand.get("hand_number"), 1) == index + 1,
            "hand roster/order mismatch",
        )
        _require(
            hand.get("table_id") == f"{design['artifact_id']}_evaluation_{episode:04d}",
            "logical table identity mismatch",
        )
        expected_hand_id = (
            f"{design['artifact_id']}_evaluation_{episode:04d}"
            f"-h{index + 1}-{_integer(hand.get('seed'), 0)}"
        )
        _require(
            type(hand.get("hand_id")) is str and hand["hand_id"] == expected_hand_id,
            "invalid logical hand ID",
        )
        _require(hand.get("hand_id") not in hand_ids, "duplicate hand ID")
        hand_ids.add(hand["hand_id"])
        _require(
            _integer(hand.get("dealer_index"), 0) == index % 8
            and hand.get("memory_updated") is False,
            "dealer/update lifecycle mismatch",
        )
        _require(
            _integer(life.get("hand_index"), 0) == _integer(settled.get("hand_index"), 0) == index
            and life.get("status") == "returned_after_update_boundary"
            and life.get("updater_enabled") is False,
            "failed or missing update boundary",
        )
        _require(
            settled.get("hand_id") == hand["hand_id"]
            and _integer(settled.get("deal_seed"), 0) == hand["seed"]
            and _integer(settled.get("actual_rng_seed"), 0) == hand["seed"]
            and _integer(settled.get("dealer_index"), 0) == index % 8,
            "settlement join mismatch",
        )
        starting = _money(hand.get("starting_stacks"), ids, nonnegative=True)
        ending = _money(hand.get("final_stacks"), ids, nonnegative=True)
        reward = _money(hand.get("rewards"), ids, nonnegative=False)
        for field in ("starting_stacks", "final_stacks", "rewards"):
            _money(settled.get(field), ids, nonnegative=field != "rewards")
        _require(starting == prior, "episode stack continuity mismatch")
        _require(
            sum(reward.values()) == 0 and all(ending[a] - starting[a] == reward[a] for a in ids),
            "hand cash flow does not conserve chips",
        )
        _require(
            all(
                settled.get(key) == hand[key]
                for key in ("starting_stacks", "final_stacks", "rewards")
            ),
            "settled/raw cash flow differs",
        )
        prior = dict(ending)
        for actor in ids:
            total_reward[actor] += reward[actor]
            if ending[actor] == 0:
                prior[actor] = design["starting_stack"]
                total_rebuy[actor] += design["starting_stack"]
                expected_rebuys.append((index + 1, actor, design["starting_stack"]))
        deal = hand.get("deal_sha256")
        _require(
            type(deal) is str and len(deal) == 64 and all(c in "0123456789abcdef" for c in deal),
            "invalid deal digest",
        )
        deals.append(deal)
        seeds.append(_integer(hand.get("seed"), 0))
        _require(
            _integer(hand.get("actual_rng_seed"), 0) == hand["seed"] < 2**63
            and hand.get("actual_rng_algorithm") == DIRECT_RNG,
            "actual engine RNG differs from assigned Path1 stream",
        )
        _require(hand["seed"] == assigned_deals[index], "raw seed differs from assigned stream")
        _require(
            deal == _deal_digest(assigned_deals[index], index % 8),
            "actual deal differs from independent RNG replay",
        )
    actual_rebuys = []
    for event in rebuy_events:
        _identity(event, design, episode, arm)
        _require(event.get("agent_id") in ids, "unknown rebuy actor")
        actual_rebuys.append(
            (
                _integer(event.get("after_hand"), 1),
                event["agent_id"],
                _integer(event.get("stack_added"), 1),
            )
        )
    _require(sorted(actual_rebuys) == sorted(expected_rebuys), "missing/extra/wrong rebuy event")
    for field in ("rewards", "rebuys", "final_stacks", "initial_stacks"):
        _money(receipt.get(field), ids, nonnegative=field != "rewards")
    _require(type(receipt.get("deal_seeds")) is list, "invalid receipt seed list")
    for seed in receipt["deal_seeds"]:
        _integer(seed, 0)
    _require(
        _same_value(receipt.get("actual_rng_seeds"), assigned_deals),
        "receipt actual RNG ledger differs",
    )
    _require(
        receipt.get("rewards") == total_reward
        and receipt.get("rebuys") == total_rebuy
        and receipt.get("final_stacks") == prior,
        "arm aggregates disagree with raw cash flow",
    )
    _require(
        all(prior[a] - design["starting_stack"] - total_rebuy[a] == total_reward[a] for a in ids),
        "rebuy incorrectly included in net reward",
    )
    _require(
        receipt.get("initial_stacks") == hands[0]["starting_stacks"]
        and receipt.get("deals") == deals
        and receipt.get("deal_seeds") == seeds,
        "reported initial state/deals differ from raw hands",
    )
    initial, final = (
        _json(directory / "initial_snapshot.json"),
        _json(directory / "final_snapshot.json"),
    )
    _snapshot_check(initial, final, parent, arm)
    _require(
        _hash_json(final) == receipt.get("final_snapshot_sha256"), "final snapshot hash mismatch"
    )
    score = Fraction(total_reward[HERO] * 100, _integer(design["big_blind"], 1) * count)
    _same_number(receipt.get("hero_net_bb100"), score)
    return {
        "net_reward_chips": total_reward[HERO],
        "hands": count,
        "score": score,
        "deals": deals,
        "seeds": seeds,
        "initial_stacks": hands[0]["starting_stacks"],
    }


def audit_evaluation_pair(
    root: str | Path,
    *,
    source_snapshot_path: str | Path,
    expected_design: dict[str, Any],
    expected_inventory: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Require both complete arms and recompute from raw hands, never summary means.

    Inventory is only verified if the caller supplies a trusted recovery manifest.
    This routine does not discover or pick successful attempts from a directory.
    """
    root = Path(root)
    before_inventory = file_inventory(root)
    if expected_inventory is not None:
        _require(_same_value(before_inventory, expected_inventory), "recovery inventory differs")
    manifest, terminal = _json(root / "path1_manifest.json"), _json(root / "terminal_receipt.json")
    _require(
        manifest.get("schema") == "path1-lifecycle-v1"
        and _hash_json(manifest.get("design")) == _hash_json(expected_design),
        "unexpected design/manifest",
    )
    design = manifest["design"]
    seed_manifest = _json(root / "seed_manifest.json")
    namespace = seed_manifest.get("namespace")
    _require(
        type(namespace) is dict
        and namespace.get("algorithm") == DERIVED_SEEDS
        and seed_manifest.get("actual_rng_algorithm") == DIRECT_RNG,
        "unknown or unbound actual RNG algorithm",
    )
    _require(
        _integer(design.get("table_size"), 1) == 8
        and _integer(design.get("big_blind"), 1) == 2
        and _integer(design.get("small_blind"), 1) == 1
        and _integer(design.get("starting_stack"), 1) == 1000,
        "source protocol differs",
    )
    _require(
        manifest.get("source_config_sha256")
        == _hash_json([design, manifest["hero_config"], manifest["model"]]),
        "source config digest mismatch",
    )
    _require(
        terminal.get("status") == "complete"
        and terminal.get("artifact_id") == design["artifact_id"],
        "incomplete/wrong artifact cannot yield a paired estimate",
    )
    _require(terminal.get("primary_analysis_admitted") is False, "library cannot grant admission")
    episode = _integer(terminal.get("episode_index"), 0)
    assigned_deals = _assigned_pair_streams(design, episode, seed_manifest)
    _require(
        type(terminal.get("arms")) is dict and set(terminal["arms"]) == set(ARMS),
        "pair must contain both arms exactly once",
    )
    _require(
        _integer(terminal.get("completed_episodes"), 0) == 2, "pair has incomplete episode count"
    )
    parent = _json(Path(source_snapshot_path))
    parent_hash = _hash_json(parent)
    _require(parent_hash == terminal.get("snapshot_sha256"), "source snapshot identity mismatch")
    expected_dirs = {f"evaluation_{episode:04d}_{arm}" for arm in ARMS}
    _require(
        {p.name for p in root.iterdir() if p.is_dir() and p.name != "_publication_staging"}
        == expected_dirs,
        "unexpected/missing episode directories",
    )
    settlements = _jsonl(root / "settled_hands.jsonl")
    lifecycle = _jsonl(root / "hand_lifecycle.jsonl")
    total = 2 * _integer(design["evaluation_episode_hands"], 1)
    _require(
        len(settlements) == len(lifecycle) == _integer(terminal.get("settled_hands"), 0) == total,
        "pair denominator mismatch",
    )
    results = {}
    for arm in ARMS:
        results[arm] = _audit_arm(
            root,
            design,
            parent,
            terminal,
            episode,
            arm,
            [row for row in settlements if row.get("arm") == arm],
            [row for row in lifecycle if row.get("arm") == arm],
            assigned_deals,
        )
    retained, removed = results["retained"], results["removed"]
    for field in ("deals", "seeds", "initial_stacks"):
        _require(retained[field] == removed[field], f"exogenous pairing mismatch: {field}")
    difference = retained["score"] - removed["score"]
    _same_number(terminal.get("paired_difference_bb100"), difference)
    _require(file_inventory(root) == before_inventory, "evidence changed during audit")
    _require(
        _hash_json(_json(Path(source_snapshot_path))) == parent_hash,
        "source snapshot changed during audit",
    )
    return {
        "schema": "path1-pair-recomputation-v1",
        "lifecycle_consistent": True,
        "scientific_admission": False,
        "recovery_inventory_verified": expected_inventory is not None,
        "campaign_id": design["campaign_id"],
        "phase": design["phase"],
        "artifact_id": design["artifact_id"],
        "episode_index": episode,
        "hero_seat": episode % 8,
        "snapshot_sha256": parent_hash,
        "hands_per_arm": retained["hands"],
        "difference_bb100": float(difference),
        "exact_difference": [difference.numerator, difference.denominator],
        "net_chips": {arm: result["net_reward_chips"] for arm, result in results.items()},
        "input_inventory_sha256": _hash_json(before_inventory),
        "scope": (
            "raw cash flow, v2 RNG/deal replay, pairing, frozen snapshots; "
            "provider/admission separate"
        ),
    }
