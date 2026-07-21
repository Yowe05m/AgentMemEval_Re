"""Outcome-blind retrieval relevance sampling and human-label audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentmemeval.evaluation.resource_audit import (
    select_latest_completed_state_rows,
)

ALLOWED_LABELS = {"relevant", "irrelevant", "uncertain"}
REQUIRED_SOURCE_DESIGNS = {"mixed_table", "target_vs_seven_no_memory"}
REVIEW_POLICY = {
    "minimum_rows": 200,
    "minimum_decisive_selected_rows": 30,
    "minimum_wilson_precision_lower_95": 0.60,
    "maximum_projected_empty_rate": 0.50,
    "maximum_uncertain_rate": 0.25,
    "selection_rule": "lowest_candidate_meeting_all_constraints_to_preserve_coverage",
}


def build_relevance_review_pack(
    campaign_dirs: list[str | Path],
    *,
    sample_size: int = 240,
    sample_seed: int = 20260717,
    review_schema_version: str = "v1",
) -> dict[str, Any]:
    """Build a deterministic, score-blind review sample from completed pilot leaves."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    schema_version = _review_schema_id(review_schema_version)
    pairs: list[dict[str, Any]] = []
    query_max_scores: list[float] = []
    sources: list[dict[str, Any]] = []
    for raw_dir in campaign_dirs:
        campaign_dir = Path(raw_dir).resolve()
        manifest_path = campaign_dir / "campaign_manifest.json"
        state_path = campaign_dir / "state.tsv"
        manifest = _read_json(manifest_path)
        campaign_id = str(manifest.get("campaign_id") or campaign_dir.name)
        campaign = manifest.get("campaign", {})
        if not isinstance(campaign, dict):
            raise ValueError(f"{campaign_id} manifest lacks campaign specification")
        design = str(campaign.get("design", ""))
        seeds = campaign.get("seeds", [])
        conditions = campaign.get("conditions") or [
            {"condition_id": "mixed_table", "target_mechanism": "mixed"}
        ]
        if not isinstance(seeds, list) or not isinstance(conditions, list):
            raise ValueError(f"{campaign_id} manifest has invalid campaign matrix")
        expected = len(seeds) * len(conditions)
        with state_path.open("r", encoding="utf-8", newline="") as handle:
            states = list(csv.DictReader(handle, delimiter="\t"))
        completed, state_selection = select_latest_completed_state_rows(states)
        if expected < 1 or len(completed) != expected:
            raise ValueError(
                f"{campaign_id} matrix is incomplete: {len(completed)}/{expected}"
            )
        event_sources: list[dict[str, Any]] = []
        for state in completed:
            run_id = str(state["run_id"])
            run_dir = campaign_dir / "runs" / run_id
            if not run_dir.is_dir():
                run_dir = Path(str(state["run_dir"]))
            events_path = run_dir / "events.jsonl"
            event_hash = _sha256(events_path)
            event_sources.append(
                {
                    "condition_id": str(state.get("condition_id", "")),
                    "seed": int(state["seed"]),
                    "attempt": int(state["attempt"]),
                    "run_id": run_id,
                    "events_sha256": event_hash,
                }
            )
            with events_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    context = event.get("memory_context")
                    if not isinstance(context, dict):
                        continue
                    metadata = context.get("metadata")
                    facts = context.get("facts")
                    if not isinstance(metadata, dict) or not isinstance(facts, list):
                        continue
                    scores = {
                        str(item.get("record_id")): item
                        for item in metadata.get("retrieval_scores", [])
                        if isinstance(item, dict) and item.get("record_id") is not None
                    }
                    observed_scores = [
                        float(item["score"])
                        for item in scores.values()
                        if item.get("score") is not None
                    ]
                    if observed_scores:
                        query_max_scores.append(max(observed_scores))
                    for fact in facts:
                        if not isinstance(fact, dict):
                            continue
                        record_id = str(fact.get("record_id", ""))
                        score = scores.get(record_id)
                        if not record_id or not isinstance(score, dict):
                            continue
                        pair_key = (
                            f"{campaign_id}|{run_id}|{line_number}|{record_id}"
                        )
                        pair = {
                                "pair_key": pair_key,
                                "campaign_id": campaign_id,
                                "run_id": run_id,
                                "seed": int(state["seed"]),
                                "event_line": line_number,
                                "events_sha256": event_hash,
                                "agent_id": str(event.get("agent_id", "")),
                                "mechanism": str(metadata.get("mechanism", "unknown")),
                                "stage": str(event.get("stage", "unknown")),
                                "phase": str(event.get("phase", "unknown")),
                                "query": str(metadata.get("query", "")),
                                "record_id": record_id,
                                "record": {
                                    "state_summary": str(fact.get("state_summary", "")),
                                    "action_summary": str(fact.get("action_summary", "")),
                                    "features": list(fact.get("features", [])),
                                },
                                "score": float(score["score"]),
                                "semantic": _optional_float(score.get("semantic")),
                                "feature": _optional_float(score.get("feature")),
                                "salience": _optional_float(score.get("salience")),
                            }
                        if schema_version.endswith("_v2"):
                            pair.update(_matched_decision_evidence(fact, score, pair_key))
                        pairs.append(pair)
        sources.append(
            {
                "campaign_id": campaign_id,
                "campaign_dir": str(campaign_dir),
                "design": design,
                "campaign_manifest_sha256": _sha256(manifest_path),
                "state_tsv_sha256": _sha256(state_path),
                "expected_state_rows": expected,
                "completed_state_rows": len(completed),
                "matrix_complete": len(completed) == expected,
                "state_selection": state_selection,
                "event_sources": sorted(
                    event_sources,
                    key=lambda item: (
                        item["condition_id"],
                        item["seed"],
                        item["attempt"],
                    ),
                ),
            }
        )
    observed_designs = {str(source["design"]) for source in sources}
    if len(sources) != 2 or observed_designs != REQUIRED_SOURCE_DESIGNS:
        raise ValueError(
            "retrieval review requires exactly one complete Campaign P and E source: "
            f"count={len(sources)}, "
            f"designs={sorted(observed_designs)}/{sorted(REQUIRED_SOURCE_DESIGNS)}"
        )
    if not pairs:
        raise ValueError("completed pilot leaves contain no query-record retrieval pairs")
    boundaries = [_quantile([item["score"] for item in pairs], q) for q in (0.25, 0.5, 0.75)]
    strata: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        score_bin = _score_bin(pair["score"], boundaries)
        pair["score_bin"] = score_bin
        strata[(pair["mechanism"], pair["stage"], pair["phase"], score_bin)].append(pair)
    for values in strata.values():
        values.sort(key=lambda item: _rank(sample_seed, item["pair_key"]))
    selected: list[dict[str, Any]] = []
    offsets = {key: 0 for key in strata}
    ordered_keys = sorted(strata)
    while len(selected) < min(sample_size, len(pairs)):
        progressed = False
        for key in ordered_keys:
            index = offsets[key]
            if index < len(strata[key]):
                selected.append(strata[key][index])
                offsets[key] += 1
                progressed = True
                if len(selected) >= min(sample_size, len(pairs)):
                    break
        if not progressed:
            break
    scores = [item["score"] for item in pairs]
    candidate_quantiles = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    candidates = sorted({_quantile(scores, q) for q in candidate_quantiles})
    keyed_rows = []
    blind_rows = []
    for index, pair in enumerate(selected, 1):
        row_id = f"RR{index:04d}_{hashlib.sha256(pair['pair_key'].encode()).hexdigest()[:12]}"
        keyed_rows.append({"row_id": row_id, **pair})
        blind_rows.append(_blind_row(row_id, pair, schema_version))
    return {
        "schema_version": schema_version,
        "status": "pending_independent_human_labels",
        "sample_seed": sample_seed,
        "requested_sample_size": sample_size,
        "sampled_row_count": len(keyed_rows),
        "population_pair_count": len(pairs),
        "population_query_count": len(query_max_scores),
        "score_bin_boundaries": boundaries,
        "candidate_thresholds": candidates,
        "query_max_scores": query_max_scores,
        "policy": REVIEW_POLICY,
        "sources": sources,
        "blind_rows": blind_rows,
        "keyed_rows": keyed_rows,
    }


def audit_relevance_labels(pack: dict[str, Any], labels: list[dict[str, str]]) -> dict[str, Any]:
    """Validate independent human labels and apply the preregistered threshold rule."""

    blockers: list[str] = []
    keyed_rows = pack.get("keyed_rows", [])
    if (
        not isinstance(keyed_rows, list)
        or not all(
            isinstance(row, dict) and str(row.get("row_id", "")).strip()
            for row in keyed_rows
        )
    ):
        blockers.append("review pack keyed rows are invalid")
        keyed_rows = []
    row_ids = [str(row["row_id"]) for row in keyed_rows]
    if len(set(row_ids)) != len(row_ids):
        blockers.append("review pack contains duplicate keyed row IDs")
    rows = {str(row["row_id"]): row for row in keyed_rows}
    schema_version = str(pack.get("schema_version", ""))
    if schema_version not in {
        "task4_retrieval_relevance_review_pack_v1",
        "task4_retrieval_relevance_review_pack_v2",
    }:
        blockers.append("review pack schema is invalid")
    if pack.get("status") != "pending_independent_human_labels":
        blockers.append("review pack status is invalid")
    if pack.get("policy") != REVIEW_POLICY:
        blockers.append("review pack policy is missing or altered")
    sources = pack.get("sources", [])
    source_designs = (
        {str(source.get("design", "")) for source in sources}
        if isinstance(sources, list)
        and all(isinstance(source, dict) for source in sources)
        else set()
    )
    if (
        not isinstance(sources, list)
        or len(sources) != 2
        or source_designs != REQUIRED_SOURCE_DESIGNS
    ):
        blockers.append("review pack does not bind complete Campaign P/E designs")
    if (
        not isinstance(sources, list)
        or len(sources) != 2
        or any(not _source_evidence_complete(source) for source in sources)
    ):
        blockers.append("review pack source matrix or event evidence is incomplete")
    blind_rows = pack.get("blind_rows", [])
    expected_blind_rows = [
        _blind_row(str(row["row_id"]), row, schema_version)
        for row in keyed_rows
    ]
    if blind_rows != expected_blind_rows:
        blockers.append("blind review rows do not match the keyed-row projection")
    source_rebuild_verified = False
    source_rebuild_content_sha256 = None
    if (
        isinstance(sources, list)
        and len(sources) == 2
        and all(_source_evidence_complete(source) for source in sources)
    ):
        try:
            rebuilt = build_relevance_review_pack(
                [str(source["campaign_dir"]) for source in sources],
                sample_size=int(pack.get("requested_sample_size", 0)),
                sample_seed=int(pack.get("sample_seed", 0)),
                review_schema_version=schema_version,
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"review pack source rebuild failed: {type(exc).__name__}")
        else:
            source_rebuild_content_sha256 = _json_sha256(rebuilt)
            source_rebuild_verified = rebuilt == pack
            if not source_rebuild_verified:
                blockers.append("review pack differs from deterministic source rebuild")
    if len(rows) < int(REVIEW_POLICY["minimum_rows"]):
        blockers.append(f"review sample too small: {len(rows)}")
    label_map: dict[str, dict[str, str]] = {}
    for label in labels:
        row_id = str(label.get("row_id", ""))
        if row_id in label_map:
            blockers.append(f"duplicate label row: {row_id}")
            continue
        if row_id not in rows:
            blockers.append(f"unknown label row: {row_id}")
            continue
        value = str(label.get("label", "")).strip().lower()
        if value not in ALLOWED_LABELS:
            blockers.append(f"invalid label for {row_id}: {value}")
        if str(label.get("reviewer_type", "")).strip().lower() != "human":
            blockers.append(f"{row_id} is not declared as human-reviewed")
        if not str(label.get("reviewer_id", "")).strip():
            blockers.append(f"{row_id} has no reviewer_id")
        label_map[row_id] = {**label, "label": value}
    missing = sorted(set(rows) - set(label_map))
    if missing:
        blockers.append(f"missing labels: {len(missing)}")
    uncertain = sum(item.get("label") == "uncertain" for item in label_map.values())
    uncertain_rate = uncertain / len(rows) if rows else 1.0
    if uncertain_rate > float(REVIEW_POLICY["maximum_uncertain_rate"]):
        blockers.append(f"uncertain rate too high: {uncertain_rate:.6f}")
    evaluations = []
    query_max = [float(value) for value in pack.get("query_max_scores", [])]
    for threshold in pack.get("candidate_thresholds", []):
        threshold = float(threshold)
        selected = [
            (row, label_map.get(row_id, {}).get("label"))
            for row_id, row in rows.items()
            if float(row["score"]) >= threshold
        ]
        relevant = sum(label == "relevant" for _row, label in selected)
        irrelevant = sum(label == "irrelevant" for _row, label in selected)
        decisive = relevant + irrelevant
        precision = relevant / decisive if decisive else 0.0
        lower = _wilson_lower(relevant, decisive)
        empty_rate = (
            sum(value < threshold for value in query_max) / len(query_max)
            if query_max
            else 1.0
        )
        eligible = (
            decisive >= int(REVIEW_POLICY["minimum_decisive_selected_rows"])
            and lower >= float(REVIEW_POLICY["minimum_wilson_precision_lower_95"])
            and empty_rate <= float(REVIEW_POLICY["maximum_projected_empty_rate"])
        )
        evaluations.append(
            {
                "threshold": threshold,
                "selected_sample_rows": len(selected),
                "decisive_selected_rows": decisive,
                "relevant": relevant,
                "irrelevant": irrelevant,
                "precision": precision,
                "wilson_precision_lower_95": lower,
                "projected_empty_retrieval_rate": empty_rate,
                "eligible": eligible,
            }
        )
    eligible = [item for item in evaluations if item["eligible"]]
    selected_threshold = min((item["threshold"] for item in eligible), default=None)
    if not eligible:
        blockers.append("no candidate threshold satisfies the preregistered review policy")
    return {
        "schema_version": "task4_retrieval_relevance_audit_v2",
        "review_status": "human_labels_verified" if not blockers else "blocked",
        "review_pack_schema_version": schema_version,
        "retrieval_threshold_status": "frozen" if not blockers else "blocked",
        "minimum_retrieval_score": selected_threshold,
        "policy": REVIEW_POLICY,
        "sampled_row_count": len(rows),
        "labeled_row_count": len(label_map),
        "uncertain_rate": uncertain_rate,
        "candidate_evaluations": evaluations,
        "review_pack_content_sha256": _json_sha256(pack),
        "review_policy_sha256": _json_sha256(REVIEW_POLICY),
        "source_campaign_count": len(sources) if isinstance(sources, list) else 0,
        "source_designs": sorted(source_designs),
        "source_evidence": sources if isinstance(sources, list) else [],
        "source_rebuild_verified": source_rebuild_verified,
        "source_rebuild_content_sha256": source_rebuild_content_sha256,
        "blockers": blockers,
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _blind_record(record: dict[str, Any]) -> dict[str, Any]:
    outcome_prefixes = (
        "hand_outcome:",
        "showdown_visible_agent_ids:",
        "summary:",
    )
    state_lines = str(record.get("state_summary", "")).splitlines()
    outcome_blind_state = "\n".join(
        line
        for line in state_lines
        if not line.strip().lower().startswith(outcome_prefixes)
    )
    return {
        "state_summary": outcome_blind_state,
        "action_summary": str(record.get("action_summary", "")),
        "features": list(record.get("features", [])),
    }


def _blind_row(
    row_id: str,
    pair: dict[str, Any],
    schema_version: str = "task4_retrieval_relevance_review_pack_v1",
) -> dict[str, Any]:
    record = pair.get("record", {})
    if not isinstance(record, dict):
        record = {}
    blind = {
        "row_id": row_id,
        "mechanism": str(pair.get("mechanism", "")),
        "stage": str(pair.get("stage", "")),
        "phase": str(pair.get("phase", "")),
        "query": str(pair.get("query", "")),
        "record": _blind_record(record),
    }
    if schema_version.endswith("_v2"):
        blind["matched_decision"] = _blind_decision(pair.get("matched_decision"))
        blind["matched_phase"] = str(pair.get("matched_phase", ""))
        blind["retrieval_unit"] = str(pair.get("retrieval_unit", ""))
    return blind


def _review_schema_id(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "v1": "task4_retrieval_relevance_review_pack_v1",
        "task4_retrieval_relevance_review_pack_v1": (
            "task4_retrieval_relevance_review_pack_v1"
        ),
        "v2": "task4_retrieval_relevance_review_pack_v2",
        "task4_retrieval_relevance_review_pack_v2": (
            "task4_retrieval_relevance_review_pack_v2"
        ),
    }
    if normalized not in aliases:
        raise ValueError(f"unknown review schema version: {value}")
    return aliases[normalized]


def _matched_decision_evidence(
    fact: dict[str, Any], score: dict[str, Any], pair_key: str
) -> dict[str, Any]:
    if str(score.get("retrieval_unit", "")) != "decision_point_max_v1":
        raise ValueError(f"V2 review pair lacks decision-point retrieval identity: {pair_key}")
    raw_index = score.get("matched_decision_index")
    if raw_index is None:
        raise ValueError(f"V2 review pair lacks matched decision index: {pair_key}")
    index = int(raw_index)
    source = fact.get("source", {})
    decisions = source.get("decisions", []) if isinstance(source, dict) else []
    if not isinstance(decisions, list) or index < 0 or index >= len(decisions):
        raise ValueError(f"V2 review pair matched decision is unavailable: {pair_key}")
    decision = decisions[index]
    if not isinstance(decision, dict):
        raise ValueError(f"V2 review pair matched decision is malformed: {pair_key}")
    phase = str(score.get("matched_phase", ""))
    if not phase or phase != str(decision.get("phase", "")):
        raise ValueError(f"V2 review pair matched phase is inconsistent: {pair_key}")
    return {
        "retrieval_unit": "decision_point_max_v1",
        "matched_decision_index": index,
        "matched_phase": phase,
        "matched_decision": dict(decision),
    }


def _blind_decision(value: object) -> dict[str, object]:
    decision = value if isinstance(value, dict) else {}
    allowed = (
        "phase",
        "board",
        "hole",
        "pot_before",
        "to_call",
        "action_type",
        "retrieval_query",
        "features",
    )
    return {key: decision.get(key) for key in allowed}


def _score_bin(value: float, boundaries: list[float]) -> str:
    if value <= boundaries[0]:
        return "q1"
    if value <= boundaries[1]:
        return "q2"
    if value <= boundaries[2]:
        return "q3"
    return "q4"


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total < 1:
        return 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (centre - margin) / denominator)


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


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_evidence_complete(source: Any) -> bool:
    if not isinstance(source, dict):
        return False
    if source.get("matrix_complete") is not True:
        return False
    if not str(source.get("campaign_dir", "")).strip():
        return False
    try:
        expected = int(source.get("expected_state_rows", -1))
        completed = int(source.get("completed_state_rows", -2))
    except (TypeError, ValueError):
        return False
    if expected < 1 or completed != expected:
        return False
    if not _is_sha256(source.get("campaign_manifest_sha256")):
        return False
    if not _is_sha256(source.get("state_tsv_sha256")):
        return False
    event_sources = source.get("event_sources")
    return (
        isinstance(event_sources, list)
        and len(event_sources) == expected
        and all(
            isinstance(event, dict)
            and bool(str(event.get("run_id", "")).strip())
            and _is_sha256(event.get("events_sha256"))
            for event in event_sources
        )
    )


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)
