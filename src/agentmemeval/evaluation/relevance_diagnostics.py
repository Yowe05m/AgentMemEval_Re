"""Post-hoc diagnostics for a completed retrieval relevance review.

The output is development evidence only.  It must never be used to re-label the
same review sample as an independent validation set.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from agentmemeval.evaluation.relevance_review import ALLOWED_LABELS

_QUERY_RE = re.compile(
    r"phase=(?P<phase>\w+).*?hole=(?P<hole>\[[^\]]*\]).*?"
    r"pot=(?P<pot>-?\d+).*?to_call=(?P<to_call>-?\d+).*?players=(?P<players>\d+)"
)


def analyze_relevance_diagnostics(
    pack: dict[str, Any], labels: list[dict[str, str]]
) -> dict[str, Any]:
    """Explain a completed review without changing its labels or policy."""

    keyed_rows = pack.get("keyed_rows")
    if not isinstance(keyed_rows, list) or not keyed_rows:
        raise ValueError("review pack has no keyed rows")
    rows_by_id = {
        str(row.get("row_id", "")).strip(): row
        for row in keyed_rows
        if isinstance(row, dict) and str(row.get("row_id", "")).strip()
    }
    if len(rows_by_id) != len(keyed_rows):
        raise ValueError("review pack row IDs are missing or duplicated")

    labels_by_id: dict[str, str] = {}
    for raw in labels:
        row_id = str(raw.get("row_id", "")).strip()
        label = str(raw.get("label", "")).strip().lower()
        if row_id in labels_by_id:
            raise ValueError(f"duplicate label row: {row_id}")
        if row_id not in rows_by_id:
            raise ValueError(f"unknown label row: {row_id}")
        if label not in ALLOWED_LABELS:
            raise ValueError(f"invalid label for {row_id}: {label}")
        labels_by_id[row_id] = label
    missing = sorted(set(rows_by_id) - set(labels_by_id))
    if missing:
        raise ValueError(f"missing labels: {len(missing)}")

    observations = []
    for row_id, row in rows_by_id.items():
        compatibility = _compatibility(row)
        observations.append(
            {
                "row_id": row_id,
                "label": labels_by_id[row_id],
                "mechanism": str(row.get("mechanism", "unknown")),
                "stage": str(row.get("stage", "unknown")),
                "query_phase": str(row.get("phase", "unknown")),
                "score_bin": str(row.get("score_bin", "unknown")),
                "score": _number(row.get("score")),
                "semantic": _number(row.get("semantic")),
                "feature": _number(row.get("feature")),
                "salience": _number(row.get("salience")),
                **compatibility,
            }
        )

    relevant = [item for item in observations if item["label"] == "relevant"]
    irrelevant = [item for item in observations if item["label"] == "irrelevant"]
    uncertain = [item for item in observations if item["label"] == "uncertain"]
    score_fields = ("score", "semantic", "feature", "salience")
    separability = {
        field: {
            "auc_relevant_over_irrelevant": _pairwise_auc(relevant, irrelevant, field),
            "by_label": {
                label: _numeric_summary(
                    [item[field] for item in observations if item["label"] == label]
                )
                for label in sorted(ALLOWED_LABELS)
            },
        }
        for field in score_fields
    }

    compatibility_fields = (
        "phase_match",
        "players_match",
        "pot_bucket_match",
        "to_call_bucket_match",
        "hole_pair_status_match",
        "hole_suited_status_match",
    )
    compatibility = {
        field: _boolean_strata(observations, field) for field in compatibility_fields
    }
    label_counts = Counter(item["label"] for item in observations)
    high_score_irrelevant = sorted(
        irrelevant,
        key=lambda item: (
            -math.inf if item["score"] is None else float(item["score"]),
            item["row_id"],
        ),
        reverse=True,
    )[:20]
    findings = _findings(
        label_counts=label_counts,
        separability=separability,
        compatibility=compatibility,
    )
    return {
        "schema_version": "task4_retrieval_relevance_diagnostics_v1",
        "classification": "post_hoc_development_only",
        "paper_validation_eligible": False,
        "reuse_as_independent_validation_prohibited": True,
        "sampled_row_count": len(observations),
        "label_counts": dict(sorted(label_counts.items())),
        "relevant_rate_excluding_uncertain": (
            len(relevant) / (len(relevant) + len(irrelevant))
            if relevant or irrelevant
            else None
        ),
        "uncertain_rate": len(uncertain) / len(observations),
        "score_separability": separability,
        "strata": {
            field: _categorical_strata(observations, field)
            for field in ("mechanism", "stage", "query_phase", "score_bin")
        },
        "compatibility": compatibility,
        "findings": findings,
        "relevant_rows": [_row_view(item) for item in relevant],
        "highest_score_irrelevant_rows": [
            _row_view(item) for item in high_score_irrelevant
        ],
    }


def _compatibility(row: dict[str, Any]) -> dict[str, object]:
    match = _QUERY_RE.search(str(row.get("query", "")))
    if match is None:
        raise ValueError(f"cannot parse review query for {row.get('row_id')}")
    try:
        hole = ast.literal_eval(match.group("hole"))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"cannot parse hole cards for {row.get('row_id')}") from exc
    if not isinstance(hole, list):
        raise ValueError(f"hole cards are not a list for {row.get('row_id')}")
    features = [str(value) for value in row.get("record", {}).get("features", [])]
    feature_values = defaultdict(list)
    for feature in features:
        name, separator, value = feature.partition(":")
        if separator:
            feature_values[name].append(value)

    query_phase = match.group("phase")
    query_players = match.group("players")
    query_pot_bucket = _bucket(int(match.group("pot")))
    query_to_call_bucket = _bucket(int(match.group("to_call")))
    query_pair = len(hole) == 2 and str(hole[0])[0] == str(hole[1])[0]
    query_suited = len(hole) == 2 and str(hole[0])[1] == str(hole[1])[1]
    return {
        "record_phase": _only(feature_values["phase"]),
        "record_players": _only(feature_values["players"]),
        "record_pot_bucket": _only(feature_values["pot"]),
        "record_to_call_bucket": _only(feature_values["to_call"]),
        "phase_match": query_phase in feature_values["phase"],
        "players_match": query_players in feature_values["players"],
        "pot_bucket_match": query_pot_bucket in feature_values["pot"],
        "to_call_bucket_match": query_to_call_bucket in feature_values["to_call"],
        "hole_pair_status_match": query_pair == ("hole_pair" in features),
        "hole_suited_status_match": query_suited == ("hole_suited" in features),
    }


def _only(values: list[str]) -> str | None:
    return values[0] if len(values) == 1 else None


def _bucket(value: int) -> str:
    if value <= 0:
        return "zero"
    if value <= 2:
        return "small"
    if value <= 8:
        return "medium"
    return "large"


def _number(value: object) -> float | None:
    return None if value is None else float(value)


def _numeric_summary(values: list[float | None]) -> dict[str, float | int | None]:
    observed = [float(value) for value in values if value is not None]
    if not observed:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(observed),
        "min": min(observed),
        "median": median(observed),
        "mean": mean(observed),
        "max": max(observed),
    }


def _pairwise_auc(
    relevant: list[dict[str, object]],
    irrelevant: list[dict[str, object]],
    field: str,
) -> float | None:
    positives = [float(item[field]) for item in relevant if item[field] is not None]
    negatives = [float(item[field]) for item in irrelevant if item[field] is not None]
    if not positives or not negatives:
        return None
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            favorable += float(positive > negative) + 0.5 * float(positive == negative)
    return favorable / (len(positives) * len(negatives))


def _counts(rows: list[dict[str, object]]) -> dict[str, object]:
    counts = Counter(str(row["label"]) for row in rows)
    decisive = counts["relevant"] + counts["irrelevant"]
    return {
        "rows": len(rows),
        "relevant": counts["relevant"],
        "irrelevant": counts["irrelevant"],
        "uncertain": counts["uncertain"],
        "precision_excluding_uncertain": (
            counts["relevant"] / decisive if decisive else None
        ),
    }


def _categorical_strata(
    observations: list[dict[str, object]], field: str
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in observations:
        grouped[str(item[field])].append(item)
    return [{field: key, **_counts(grouped[key])} for key in sorted(grouped)]


def _boolean_strata(
    observations: list[dict[str, object]], field: str
) -> list[dict[str, object]]:
    return [
        {field: value, **_counts([item for item in observations if item[field] is value])}
        for value in (False, True)
    ]


def _row_view(item: dict[str, object]) -> dict[str, object]:
    return {
        key: item[key]
        for key in (
            "row_id",
            "label",
            "mechanism",
            "stage",
            "query_phase",
            "record_phase",
            "score_bin",
            "score",
            "semantic",
            "feature",
            "phase_match",
            "players_match",
            "pot_bucket_match",
            "to_call_bucket_match",
            "hole_pair_status_match",
            "hole_suited_status_match",
        )
    }


def _findings(
    *,
    label_counts: Counter[str],
    separability: dict[str, dict[str, object]],
    compatibility: dict[str, list[dict[str, object]]],
) -> list[str]:
    findings = []
    if label_counts["relevant"] < 30:
        findings.append("fewer_than_30_relevant_rows_in_entire_review_sample")
    for field in ("score", "semantic", "feature"):
        auc = separability[field]["auc_relevant_over_irrelevant"]
        if auc is not None and float(auc) <= 0.55:
            findings.append(f"{field}_does_not_separate_relevance_auc_le_0_55")
    phase = {bool(item["phase_match"]): item for item in compatibility["phase_match"]}
    if phase[False]["rows"]:
        findings.append("review_contains_query_record_phase_mismatches")
    return findings
