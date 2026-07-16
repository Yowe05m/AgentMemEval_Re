"""Outcome-blind retrieval relevance sampling and human-label audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ALLOWED_LABELS = {"relevant", "irrelevant", "uncertain"}
REVIEW_POLICY = {
    "minimum_rows": 200,
    "minimum_decisive_selected_rows": 30,
    "minimum_wilson_precision_lower_95": 0.60,
    "maximum_projected_empty_rate": 0.50,
    "maximum_uncertain_rate": 0.25,
    "selection_rule": "lowest_candidate_meeting_all_constraints_to_preserve_coverage",
}


def build_relevance_review_pack(
    campaign_dirs: list[str | Path], *, sample_size: int = 240, sample_seed: int = 20260717
) -> dict[str, Any]:
    """Build a deterministic, score-blind review sample from completed pilot leaves."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    pairs: list[dict[str, Any]] = []
    query_max_scores: list[float] = []
    sources: list[dict[str, Any]] = []
    for raw_dir in campaign_dirs:
        campaign_dir = Path(raw_dir).resolve()
        manifest_path = campaign_dir / "campaign_manifest.json"
        state_path = campaign_dir / "state.tsv"
        manifest = _read_json(manifest_path)
        campaign_id = str(manifest.get("campaign_id") or campaign_dir.name)
        with state_path.open("r", encoding="utf-8", newline="") as handle:
            states = list(csv.DictReader(handle, delimiter="\t"))
        completed = [row for row in states if row.get("status") == "complete"]
        sources.append(
            {
                "campaign_id": campaign_id,
                "campaign_manifest_sha256": _sha256(manifest_path),
                "state_tsv_sha256": _sha256(state_path),
                "completed_state_rows": len(completed),
            }
        )
        for state in completed:
            run_id = str(state["run_id"])
            run_dir = campaign_dir / "runs" / run_id
            if not run_dir.is_dir():
                run_dir = Path(str(state["run_dir"]))
            events_path = run_dir / "events.jsonl"
            event_hash = _sha256(events_path)
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
                        pairs.append(
                            {
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
        blind_rows.append(
            {
                "row_id": row_id,
                "mechanism": pair["mechanism"],
                "stage": pair["stage"],
                "phase": pair["phase"],
                "query": pair["query"],
                "record": pair["record"],
            }
        )
    return {
        "schema_version": "task4_retrieval_relevance_review_pack_v1",
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

    rows = {str(row["row_id"]): row for row in pack.get("keyed_rows", [])}
    blockers: list[str] = []
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
        "schema_version": "task4_retrieval_relevance_audit_v1",
        "review_status": "human_labels_verified" if not blockers else "blocked",
        "retrieval_threshold_status": "frozen" if not blockers else "blocked",
        "minimum_retrieval_score": selected_threshold,
        "policy": REVIEW_POLICY,
        "sampled_row_count": len(rows),
        "labeled_row_count": len(label_map),
        "uncertain_rate": uncertain_rate,
        "candidate_evaluations": evaluations,
        "blockers": blockers,
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


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
