"""Fail-closed contract for an independent experience strategy reviewer.

The main experience mechanism remains responsible for proposing one complete
five-section Markdown document.  This module binds a reviewer response to that
exact candidate and evidence bundle, checks a finite issue vocabulary, and
permits only small mechanical corrections before deterministic validation is
replayed.  It deliberately does not generate strategy or estimate poker EV.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from agentmemeval.path1_v1.core.model_identity import canonical_model_family

SEMANTIC_REVIEW_SCHEMA_VERSION = "strategy_semantic_review_v2"
SEMANTIC_REVIEW_POLICY_VERSION = "2026-07-27-independent-strategy-review-v2"

SEMANTIC_REVIEW_SYSTEM_PROMPT = """/no_think
You are an independent semantic reviewer of poker strategy experience, not the
proposal model or proposal source. The candidate is immutable. Return only
schema-conforming JSON; do not rewrite Markdown or expose a reasoning trace.

accept means only that bound evidence and reference cards show no blocking
contradiction; it does not prove optimality. Interpret human poker semantics,
not engine tokens: blinds are forced bets, the first voluntary preflop
investment is an open-raise, and a postflop investment after a check is a bet
even if the engine encodes it as raise. Value requires worse calls, bluffing
requires better folds, and protection requires equity denial.

Frequent checks describe coverage, not optimality. CHECK_COPY_BIAS is permitted
only when allowed_issue_codes authorizes it and a mandatory host finding binds
observed_check_share >= 0.65 plus omitted proactive branches. RC-CHECK alone
cannot authorize it. Every issue identifies a candidate claim and cites bound
evidence, a reference card, or a host finding.

Copy all bound hashes exactly. acknowledged_finding_ids must match the contract.
Issue IDs are unique and ordered I001, I002, ... A bullet issue copies its
target_bullet_id and uses an exact claim substring or an empty excerpt.
Document and calibration issues use target_bullet_id=null. accept requires no
issues or repairs; reject requires at least one issue and no repairs. repair may
contain at most two allowlisted mechanical repairs covering all issues. Any
CHECK_COPY_BIAS issue forces rejection. Use only identifiers supplied by the
request and output one JSON object."""

EXPERIENCE_SECTIONS = (
    "Starting Hands",
    "Postflop Play",
    "Turn and River",
    "Opponent Adaptation",
    "Bet Sizing and Position",
)

SEMANTIC_ISSUE_CODES = (
    "STATE_ACTION_MISMATCH",
    "ACTION_PURPOSE_MISMATCH",
    "CHECK_COPY_BIAS",
    "REVERSAL_DIRECTION_ERROR",
    "SINGLE_HAND_OUTCOME_PROOF",
    "UNOBSERVED_COUNTERFACTUAL",
    "EV_PREMISE_MISSING",
    "EVIDENCE_SCOPE_OVERREACH",
    "OPPONENT_TYPE_CONFLATION",
    "STATIC_FACT_AS_STRATEGY",
)
_SEMANTIC_ISSUE_CODE_SET = frozenset(SEMANTIC_ISSUE_CODES)

MECHANICAL_REPAIR_OPERATIONS = (
    "replace_action_term",
    "delete_clause",
    "delete_bullet",
    "neutralize_calibration_note",
)
_MECHANICAL_REPAIR_OPERATION_SET = frozenset(MECHANICAL_REPAIR_OPERATIONS)

MAX_MECHANICAL_REPAIRS = 2
MAX_REPAIRED_RULE_FRACTION = 0.20
SEMANTIC_REVIEW_MAX_TRAJECTORY_HANDS = 5
NEUTRAL_CALIBRATION_NOTE = (
    "The selected action and outcome were recorded; one hand cannot identify counterfactual EV."
)

_REFERENCE_CARD_ITEMS = (
    (
        "RC-ACTION-STATE",
        "Use human poker terms: check requires to_call=0; blinds are forced bets; the first "
        "voluntary preflop investment is an open-raise; a postflop investment after a check "
        "is a bet even when the engine token is raise. fold, call, or raise requires an "
        "outstanding bet. A future raise branch is compatible with a checked-to state.",
    ),
    (
        "RC-HU-ORDER",
        "Heads-up, BTN is SB and acts before BB preflop, so BTN cannot first face a BB limp. "
        "Postflop, BB acts first and BTN acts second.",
    ),
    (
        "RC-VALUE",
        "A value bet or raise requires enough worse hands to continue. Frequent check-folds "
        "provide fold equity, not value payment.",
    ),
    (
        "RC-BLUFF",
        "A bluff requires enough better hands to fold. A pure bluff investing B to win pot P "
        "breaks even at fold rate B/(P+B), ignoring equity and rake.",
    ),
    (
        "RC-PROTECTION",
        "Protection comes from betting hands with meaningful equity out of the pot. It cannot "
        "justify a call or replace the comparison between lost value and denied equity.",
    ),
    (
        "RC-CHECK",
        "Checking may realize equity, induce bluffs, or avoid a negative-EV bet; frequency is "
        "not optimality evidence. An out-of-position check does not guarantee a free card. "
        "RC-CHECK cannot authorize CHECK_COPY_BIAS without request permission and a host finding.",
    ),
    (
        "RC-CALL-RAISE",
        "With call cost C and contestable pot P before calling, simplified required equity is "
        "C/(P+C). More opponent bluffs usually increase call EV.",
    ),
    (
        "RC-REVERSAL",
        "A valid reversal lowers the original action's EV or raises a named alternative's EV. "
        "A condition supporting the original action or unrelated to it is not a reversal.",
    ),
    (
        "RC-OUTCOME-EVIDENCE",
        "A hand reward combines action, card variance, and opponents. Unselected-action returns "
        "are unobserved; one hand can form a tentative hypothesis but cannot prove a strategy.",
    ),
    (
        "RC-OPPONENT-TYPE",
        "A check-fold range may be attacked for fold equity; a check-call range calls for a "
        "stronger value range. Do not merge them into one archetype and response.",
    ),
)
REFERENCE_CARD_IDS = tuple(item[0] for item in _REFERENCE_CARD_ITEMS)
_REFERENCE_CARD_ID_SET = frozenset(REFERENCE_CARD_IDS)

_HASH_RE = re.compile(r"^[a-f0-9]{64}$")
_ISSUE_ID_RE = re.compile(r"^I[0-9]{3}$")
_BULLET_ID_RE = re.compile(r"^B[0-9]{3}$")
_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^(?P<prefix>\s*[-*]\s+)(?P<body>.+?)\s*$")

_ALLOWED_ACTION_TERM_REPLACEMENTS = frozenset(
    {
        ("raise", "bet"),
        ("Raise", "Bet"),
        ("RAISE", "BET"),
    }
)

_REPAIRABLE_DELETE_CLAUSE_CODES = frozenset(
    {
        "ACTION_PURPOSE_MISMATCH",
        "REVERSAL_DIRECTION_ERROR",
        "SINGLE_HAND_OUTCOME_PROOF",
        "UNOBSERVED_COUNTERFACTUAL",
        "EV_PREMISE_MISSING",
        "EVIDENCE_SCOPE_OVERREACH",
    }
)
_CALIBRATION_REPAIR_CODES = frozenset({"SINGLE_HAND_OUTCOME_PROOF", "UNOBSERVED_COUNTERFACTUAL"})
_NON_REPAIRABLE_ISSUE_CODES = frozenset({"CHECK_COPY_BIAS"})
_CHECK_COPY_BIAS_MIN_SHARE = 0.65
_CHECK_COPY_BIAS_FINDING_CODES = frozenset(
    {"behavior_check_copy_bias", "behavior_no_proactive_strategy_branch"}
)

SEMANTIC_REVIEW_FAILURE_CODES = (
    "non_contract_failure",
    "closed_object_violation",
    "schema_version_mismatch",
    "bound_hash_mismatch",
    "model_family_not_independent",
    "finding_acknowledgement_mismatch",
    "mandatory_finding_not_represented",
    "unknown_source_finding_id",
    "duplicate_issue_id",
    "issue_id_sequence_invalid",
    "issue_target_invalid",
    "issue_excerpt_not_bound",
    "unbound_evidence_reference",
    "unknown_reference_card",
    "issue_code_not_evidence_authorized",
    "issue_support_missing",
    "verdict_shape_violation",
    "repair_contract_violation",
    "repair_budget_violation",
    "replay_contract_violation",
    "unclassified_contract_violation",
)
_SEMANTIC_REVIEW_FAILURE_CODE_SET = frozenset(SEMANTIC_REVIEW_FAILURE_CODES)


class SemanticReviewContractError(ValueError):
    """Raised when a reviewer payload cannot be trusted or replayed."""


@dataclass(frozen=True, slots=True)
class ExperienceBullet:
    """One single-line Markdown bullet bound to its source location."""

    bullet_id: str
    section: str
    index: int
    line_index: int
    prefix: str
    text: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SemanticReviewContext:
    """Immutable identity envelope supplied by the host, not by the model."""

    candidate_md: str
    calibration_note: str
    candidate_sha256: str
    calibration_note_sha256: str
    evidence_bundle_sha256: str
    reference_card_sha256: str
    proposer_model_family: str
    reviewer_model_family: str
    independence_passed: bool
    mandatory_finding_ids: tuple[str, ...]
    allowed_evidence_refs: tuple[str, ...]
    allowed_issue_codes: tuple[str, ...]
    check_copy_bias_finding_ids: tuple[str, ...]
    bullets: tuple[ExperienceBullet, ...]


@dataclass(frozen=True, slots=True)
class SemanticIssue:
    issue_id: str
    code: str
    scope: str
    target_bullet_id: str | None
    section: str | None
    bullet_index: int | None
    bullet_sha256: str | None
    claim_excerpt: str
    evidence_refs: tuple[str, ...]
    reference_card_ids: tuple[str, ...]
    source_finding_ids: tuple[str, ...]
    diagnosis: str


@dataclass(frozen=True, slots=True)
class MechanicalRepair:
    op: str
    issue_ids: tuple[str, ...]
    target_bullet_id: str | None
    section: str | None
    bullet_index: int | None
    original_sha256: str | None
    from_text: str | None
    to_text: str | None


@dataclass(frozen=True, slots=True)
class SemanticReview:
    schema_version: str
    candidate_sha256: str
    evidence_bundle_sha256: str
    reference_card_sha256: str
    verdict: str
    acknowledged_finding_ids: tuple[str, ...]
    issues: tuple[SemanticIssue, ...]
    repairs: tuple[MechanicalRepair, ...]
    proposer_model_family: str
    reviewer_model_family: str
    independence_passed: bool


@dataclass(frozen=True, slots=True)
class SemanticReviewReplay:
    """Deterministic replay result used by a future integration layer."""

    status: str
    installable: bool
    document_md: str
    calibration_note: str
    document_sha256: str
    calibration_note_sha256: str
    applied_repair_count: int
    validation_findings: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class SemanticReviewRequestMaterial:
    """Provider-neutral, request-bound material shared by production and probes."""

    system_prompt: str
    user_prompt: str
    response_schema: str
    response_json_schema: dict[str, object]
    metadata: dict[str, object]


def _review_json_schema() -> dict[str, object]:
    nullable_bullet_id: dict[str, object] = {
        "type": ["string", "null"],
        "pattern": "^B[0-9]{3}$",
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": SEMANTIC_REVIEW_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "candidate_sha256",
            "evidence_bundle_sha256",
            "reference_card_sha256",
            "verdict",
            "acknowledged_finding_ids",
            "issues",
            "repairs",
        ],
        "properties": {
            "schema_version": {"const": SEMANTIC_REVIEW_SCHEMA_VERSION},
            "candidate_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "evidence_bundle_sha256": {
                "type": "string",
                "pattern": "^[a-f0-9]{64}$",
            },
            "reference_card_sha256": {
                "type": "string",
                "pattern": "^[a-f0-9]{64}$",
            },
            "verdict": {"enum": ["accept", "repair", "reject"]},
            "acknowledged_finding_ids": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "maxLength": 96},
            },
            "issues": {
                "type": "array",
                "maxItems": 20,
                "items": {"$ref": "#/$defs/issue"},
            },
            "repairs": {
                "type": "array",
                "maxItems": MAX_MECHANICAL_REPAIRS,
                "items": {"$ref": "#/$defs/repair"},
            },
        },
        "allOf": [
            {
                "if": {
                    "properties": {"verdict": {"const": "accept"}},
                    "required": ["verdict"],
                },
                "then": {
                    "properties": {
                        "issues": {"maxItems": 0},
                        "repairs": {"maxItems": 0},
                    }
                },
            },
            {
                "if": {
                    "properties": {"verdict": {"const": "reject"}},
                    "required": ["verdict"],
                },
                "then": {
                    "properties": {
                        "issues": {"minItems": 1},
                        "repairs": {"maxItems": 0},
                    }
                },
            },
            {
                "if": {
                    "properties": {"verdict": {"const": "repair"}},
                    "required": ["verdict"],
                },
                "then": {
                    "properties": {
                        "issues": {"minItems": 1},
                        "repairs": {
                            "minItems": 1,
                            "maxItems": MAX_MECHANICAL_REPAIRS,
                        },
                    }
                },
            },
        ],
        "$defs": {
            "issue": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "issue_id",
                    "code",
                    "scope",
                    "target_bullet_id",
                    "claim_excerpt",
                    "evidence_refs",
                    "reference_card_ids",
                    "source_finding_ids",
                    "diagnosis",
                ],
                "properties": {
                    "issue_id": {"type": "string", "pattern": "^I[0-9]{3}$"},
                    "code": {"enum": list(SEMANTIC_ISSUE_CODES)},
                    "scope": {"enum": ["document", "bullet", "calibration_note"]},
                    "target_bullet_id": nullable_bullet_id,
                    "claim_excerpt": {"type": "string", "maxLength": 240},
                    "evidence_refs": {
                        "type": "array",
                        "maxItems": 12,
                        "uniqueItems": True,
                        "items": {"type": "string", "maxLength": 128},
                    },
                    "reference_card_ids": {
                        "type": "array",
                        "maxItems": 6,
                        "uniqueItems": True,
                        "items": {"enum": list(REFERENCE_CARD_IDS)},
                    },
                    "source_finding_ids": {
                        "type": "array",
                        "maxItems": 20,
                        "uniqueItems": True,
                        "items": {"type": "string", "maxLength": 96},
                    },
                    "diagnosis": {"type": "string", "minLength": 1, "maxLength": 300},
                },
                "allOf": [
                    {
                        "if": {
                            "properties": {"scope": {"const": "bullet"}},
                            "required": ["scope"],
                        },
                        "then": {
                            "properties": {
                                "target_bullet_id": {
                                    "type": "string",
                                    "pattern": "^B[0-9]{3}$",
                                }
                            }
                        },
                        "else": {"properties": {"target_bullet_id": {"const": None}}},
                    },
                    {
                        "if": {
                            "properties": {"scope": {"const": "document"}},
                            "required": ["scope"],
                        },
                        "then": {"properties": {"claim_excerpt": {"const": ""}}},
                    },
                ],
            },
            "repair": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "op",
                    "issue_ids",
                    "target_bullet_id",
                    "from_text",
                    "to_text",
                ],
                "properties": {
                    "op": {"enum": list(MECHANICAL_REPAIR_OPERATIONS)},
                    "issue_ids": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "pattern": "^I[0-9]{3}$"},
                    },
                    "target_bullet_id": nullable_bullet_id,
                    "from_text": {"type": ["string", "null"], "maxLength": 240},
                    "to_text": {"type": ["string", "null"], "maxLength": 240},
                },
                "allOf": [
                    {
                        "if": {
                            "properties": {"op": {"const": "neutralize_calibration_note"}},
                            "required": ["op"],
                        },
                        "then": {"properties": {"target_bullet_id": {"const": None}}},
                        "else": {
                            "properties": {
                                "target_bullet_id": {
                                    "type": "string",
                                    "pattern": "^B[0-9]{3}$",
                                }
                            }
                        },
                    }
                ],
            },
        },
    }


STRATEGY_SEMANTIC_REVIEW_JSON_SCHEMA = _review_json_schema()


def semantic_review_json_schema(
    context: SemanticReviewContext | None = None,
) -> dict[str, object]:
    """Return a provider schema, optionally bound to one immutable review context."""

    schema = copy.deepcopy(STRATEGY_SEMANTIC_REVIEW_JSON_SCHEMA)
    if context is None:
        return schema
    properties = schema["properties"]
    assert isinstance(properties, dict)
    properties["candidate_sha256"] = {"const": context.candidate_sha256}
    properties["evidence_bundle_sha256"] = {"const": context.evidence_bundle_sha256}
    properties["reference_card_sha256"] = {"const": context.reference_card_sha256}
    properties["acknowledged_finding_ids"] = {"const": list(context.mandatory_finding_ids)}
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    issue = definitions["issue"]
    assert isinstance(issue, dict)
    issue_properties = issue["properties"]
    assert isinstance(issue_properties, dict)
    issue_properties["code"] = {"enum": list(context.allowed_issue_codes)}
    target_bullet_id = issue_properties["target_bullet_id"]
    assert isinstance(target_bullet_id, dict)
    target_bullet_id["enum"] = [
        *(bullet.bullet_id for bullet in context.bullets),
        None,
    ]
    evidence_refs = issue_properties["evidence_refs"]
    assert isinstance(evidence_refs, dict)
    if context.allowed_evidence_refs:
        evidence_refs["items"] = {"enum": list(context.allowed_evidence_refs)}
    else:
        evidence_refs["maxItems"] = 0
    source_ids = issue_properties["source_finding_ids"]
    assert isinstance(source_ids, dict)
    if context.mandatory_finding_ids:
        source_ids["items"] = {"enum": list(context.mandatory_finding_ids)}
    else:
        source_ids["maxItems"] = 0
    if context.check_copy_bias_finding_ids:
        issue_rules = issue["allOf"]
        assert isinstance(issue_rules, list)
        issue_rules.append(
            {
                "if": {
                    "properties": {"code": {"const": "CHECK_COPY_BIAS"}},
                    "required": ["code"],
                },
                "then": {
                    "properties": {
                        "scope": {"const": "document"},
                        "target_bullet_id": {"const": None},
                        "claim_excerpt": {"const": ""},
                        "source_finding_ids": {
                            "type": "array",
                            "minItems": 1,
                            "contains": {"enum": list(context.check_copy_bias_finding_ids)},
                        },
                    }
                },
            }
        )
    repair = definitions["repair"]
    assert isinstance(repair, dict)
    repair_properties = repair["properties"]
    assert isinstance(repair_properties, dict)
    repair_target = repair_properties["target_bullet_id"]
    assert isinstance(repair_target, dict)
    repair_target["enum"] = [
        *(bullet.bullet_id for bullet in context.bullets),
        None,
    ]
    return schema


def semantic_review_contract_failure_code(exc: BaseException) -> str:
    """Map contract text to a finite, persistence-safe diagnostic code."""

    if not isinstance(exc, SemanticReviewContractError):
        return "non_contract_failure"
    message = str(exc)
    checks = (
        ("unsupported semantic review schema", "schema_version_mismatch"),
        ("hash", "bound_hash_mismatch"),
        ("non-independent", "model_family_not_independent"),
        ("own candidate", "model_family_not_independent"),
        ("acknowledged_finding_ids", "finding_acknowledgement_mismatch"),
        ("mandatory deterministic findings", "mandatory_finding_not_represented"),
        ("source finding IDs outside", "unknown_source_finding_id"),
        ("issue_id values must be unique", "duplicate_issue_id"),
        ("issue_id values must be I001", "issue_id_sequence_invalid"),
        ("issue excerpt", "issue_excerpt_not_bound"),
        ("target_bullet_id", "issue_target_invalid"),
        ("outside the bound bundle", "unbound_evidence_reference"),
        ("unknown reference card", "unknown_reference_card"),
        ("not authorized by bound evidence", "issue_code_not_evidence_authorized"),
        ("needs evidence", "issue_support_missing"),
        ("verdict", "verdict_shape_violation"),
        ("twenty percent", "repair_budget_violation"),
        ("more than two", "repair_budget_violation"),
        ("replay", "replay_contract_violation"),
        ("repair", "repair_contract_violation"),
        ("keys mismatch", "closed_object_violation"),
    )
    for fragment, code in checks:
        if fragment in message:
            return code
    return "unclassified_contract_violation"


def semantic_review_system_prompt() -> str:
    """Return the single reviewer policy shared by production and true-provider probes."""

    return SEMANTIC_REVIEW_SYSTEM_PROMPT


def build_semantic_review_request_material(
    *,
    context: SemanticReviewContext,
    evidence_bundle: object,
) -> SemanticReviewRequestMaterial:
    """Build the one canonical production/probe prompt and bound response contract."""

    if canonical_json_sha256(evidence_bundle) != context.evidence_bundle_sha256:
        raise SemanticReviewContractError(
            "request evidence hash does not match the bound review context"
        )
    bullet_inventory = [
        {
            "bullet_id": bullet.bullet_id,
            "section": bullet.section,
            "bullet_index": bullet.index,
            "bullet_sha256": bullet.sha256,
        }
        for bullet in context.bullets
    ]
    request_contract = {
        "schema_version": SEMANTIC_REVIEW_SCHEMA_VERSION,
        "policy_version": SEMANTIC_REVIEW_POLICY_VERSION,
        "candidate_sha256": context.candidate_sha256,
        "evidence_bundle_sha256": context.evidence_bundle_sha256,
        "reference_card_sha256": context.reference_card_sha256,
        "proposer_model_family": context.proposer_model_family,
        "reviewer_model_family": context.reviewer_model_family,
        "independence_passed": context.independence_passed,
        "mandatory_finding_ids": list(context.mandatory_finding_ids),
        "allowed_evidence_refs": list(context.allowed_evidence_refs),
        "finite_issue_codes": list(SEMANTIC_ISSUE_CODES),
        "allowed_issue_codes": list(context.allowed_issue_codes),
        "check_copy_bias_gate": {
            "authorized": bool(context.check_copy_bias_finding_ids),
            "minimum_observed_check_share": _CHECK_COPY_BIAS_MIN_SHARE,
            "qualifying_host_finding_ids": list(context.check_copy_bias_finding_ids),
            "reference_card_alone_is_sufficient": False,
        },
        "mechanical_repair_operations": list(MECHANICAL_REPAIR_OPERATIONS),
        "rules": [
            "accept requires issues=[] and repairs=[]",
            "reject requires at least one issue and repairs=[]",
            "repair permits at most two issue-linked mechanical operations",
            "an issue code may be emitted only when listed in allowed_issue_codes",
            "CHECK_COPY_BIAS additionally requires its qualifying host finding id",
            "every mandatory finding id must appear in source_finding_ids",
            "bullet issues and bullet repairs target one bound bullet_id",
        ],
    }
    user_prompt = "\n\n".join(
        (
            "Review the bound candidate using poker EV, bound evidence, and reference "
            "cards; do not guess host intent.",
            "Review identity and hash contract:\n"
            + json.dumps(
                request_contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "Candidate five-section experience document:\n" + context.candidate_md,
            "Candidate bullet inventory:\n"
            + json.dumps(
                bullet_inventory,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "Read-only evidence bundle:\n"
            + json.dumps(
                evidence_bundle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "Fixed poker EV and action-semantics reference cards:\n"
            + json.dumps(
                reference_card_bundle(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            f"Return the verdict strictly under the {SEMANTIC_REVIEW_SCHEMA_VERSION} JSON Schema.",
        )
    )
    response_schema = semantic_review_json_schema(context)
    metadata = {
        "response_schema": SEMANTIC_REVIEW_SCHEMA_VERSION,
        "response_json_schema": response_schema,
        "semantic_review_policy_version": SEMANTIC_REVIEW_POLICY_VERSION,
        "candidate_sha256": context.candidate_sha256,
        "evidence_bundle_sha256": context.evidence_bundle_sha256,
        "reference_card_sha256": context.reference_card_sha256,
        "proposer_model_family": context.proposer_model_family,
        "reviewer_model_family": context.reviewer_model_family,
        "mandatory_finding_ids": list(context.mandatory_finding_ids),
        "allowed_evidence_refs": list(context.allowed_evidence_refs),
        "allowed_issue_codes": list(context.allowed_issue_codes),
        "check_copy_bias_finding_ids": list(context.check_copy_bias_finding_ids),
        "bullet_ids": [bullet.bullet_id for bullet in context.bullets],
    }
    if isinstance(evidence_bundle, Mapping):
        trajectory_window = evidence_bundle.get("trajectory_window")
        if isinstance(trajectory_window, Mapping):
            included = trajectory_window.get("included_hand_count")
            total = trajectory_window.get("total_hand_count")
            if type(included) is int and type(total) is int:
                metadata["evidence_hand_count_included"] = included
                metadata["evidence_hand_count_total"] = total
    return SemanticReviewRequestMaterial(
        system_prompt=semantic_review_system_prompt(),
        user_prompt=user_prompt,
        response_schema=SEMANTIC_REVIEW_SCHEMA_VERSION,
        response_json_schema=response_schema,
        metadata=metadata,
    )


def build_semantic_review_evidence_view(
    *,
    trajectory_recaps: Sequence[Mapping[str, object]],
    behavior_digest: Mapping[str, object],
    fact_evidence: Sequence[Mapping[str, object]],
    fact_groups: Mapping[str, Sequence[str]],
    deterministic_findings: Sequence[Mapping[str, object]],
    calibration_note: str,
    current_experience: Mapping[str, object],
) -> dict[str, object]:
    """Build the deterministic 32K-safe reviewer view without truncating facts/gates."""

    if not isinstance(calibration_note, str):
        raise SemanticReviewContractError("calibration_note must be a string")
    total_hand_count = len(trajectory_recaps)
    included = list(trajectory_recaps[-SEMANTIC_REVIEW_MAX_TRAJECTORY_HANDS:])
    view: dict[str, object] = {
        "schema_version": "experience_semantic_evidence_v2_bounded",
        "trajectory_window": {
            "selection": "most_recent",
            "limit": SEMANTIC_REVIEW_MAX_TRAJECTORY_HANDS,
            "included_hand_count": len(included),
            "total_hand_count": total_hand_count,
        },
        "trajectory_recaps": copy.deepcopy(included),
        "behavior_digest": copy.deepcopy(dict(behavior_digest)),
        "fact_evidence": copy.deepcopy(list(fact_evidence)),
        "fact_groups": copy.deepcopy(
            {str(key): list(values) for key, values in fact_groups.items()}
        ),
        "deterministic_findings": copy.deepcopy(list(deterministic_findings)),
        "calibration_note": calibration_note,
        "current_experience": copy.deepcopy(dict(current_experience)),
    }
    canonical_json_sha256(view)
    return view


def reference_card_bundle() -> dict[str, str]:
    """Return the complete, versioned logic card required by every review call."""

    return dict(_REFERENCE_CARD_ITEMS)


def sha256_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("sha256_text requires str")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash JSON data independently of mapping insertion order."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SemanticReviewContractError("evidence bundle must be strict JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()


def reference_card_sha256() -> str:
    return canonical_json_sha256(reference_card_bundle())


def extract_experience_bullets(candidate_md: str) -> tuple[ExperienceBullet, ...]:
    """Extract repairable single-line bullets from the five fixed chapters."""

    if not isinstance(candidate_md, str):
        raise TypeError("candidate_md must be str")
    current_section: str | None = None
    section_counts: dict[str, int] = {section: 0 for section in EXPERIENCE_SECTIONS}
    bullets: list[ExperienceBullet] = []
    for line_index, line in enumerate(candidate_md.splitlines()):
        heading = _HEADING_RE.match(line)
        if heading:
            name = heading.group(1).strip()
            current_section = name if name in section_counts else None
            continue
        if current_section is None:
            continue
        bullet = _BULLET_RE.match(line)
        if bullet is None:
            continue
        text = bullet.group("body")
        section_counts[current_section] += 1
        bullets.append(
            ExperienceBullet(
                bullet_id=f"B{len(bullets) + 1:03d}",
                section=current_section,
                index=section_counts[current_section],
                line_index=line_index,
                prefix=bullet.group("prefix"),
                text=text,
                sha256=sha256_text(text),
            )
        )
    return tuple(bullets)


def _bound_check_copy_bias_finding_ids(
    evidence_bundle: object,
    *,
    mandatory_finding_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Authorize CHECK_COPY_BIAS only from a complete, request-bound evidence gate."""

    if not isinstance(evidence_bundle, Mapping):
        return ()
    behavior_digest = evidence_bundle.get("behavior_digest")
    if not isinstance(behavior_digest, Mapping):
        return ()
    observed_share = behavior_digest.get("observed_check_share")
    if (
        isinstance(observed_share, bool)
        or not isinstance(observed_share, (int, float))
        or not math.isfinite(float(observed_share))
        or float(observed_share) < _CHECK_COPY_BIAS_MIN_SHARE
        or float(observed_share) > 1.0
    ):
        return ()

    active_branches = evidence_bundle.get("active_branches")
    bundle_marks_omission = (
        isinstance(active_branches, Mapping)
        and active_branches.get("candidate_omits_active_branches") is True
    )
    raw_findings = evidence_bundle.get("deterministic_findings")
    if not isinstance(raw_findings, Sequence) or isinstance(raw_findings, (str, bytes, bytearray)):
        return ()
    mandatory = set(mandatory_finding_ids)
    qualifying: list[str] = []
    for raw in raw_findings:
        if not isinstance(raw, Mapping):
            continue
        finding_id = str(raw.get("finding_id", ""))
        if finding_id not in mandatory:
            continue
        if str(raw.get("code", "")) not in _CHECK_COPY_BIAS_FINDING_CODES:
            continue
        if str(raw.get("semantic_issue_hint", "")) != "CHECK_COPY_BIAS":
            continue
        finding_marks_omission = raw.get("candidate_omits_active_branches") is True
        if bundle_marks_omission or finding_marks_omission:
            qualifying.append(finding_id)
    return tuple(sorted(set(qualifying)))


def bind_semantic_review_context(
    *,
    candidate_md: str,
    evidence_bundle: object,
    proposer_model_family: str,
    reviewer_model_family: str,
    calibration_note: str = "",
    mandatory_finding_ids: Iterable[str] = (),
    allowed_evidence_refs: Iterable[str] = (),
) -> SemanticReviewContext:
    """Bind hashes and identities before any reviewer response is accepted."""

    if not isinstance(candidate_md, str) or not candidate_md.strip():
        raise SemanticReviewContractError("candidate document must be non-empty")
    if not isinstance(calibration_note, str):
        raise SemanticReviewContractError("calibration_note must be a string")
    proposer = canonical_model_family(proposer_model_family)
    reviewer = canonical_model_family(reviewer_model_family)
    findings = _validated_string_tuple(
        list(mandatory_finding_ids),
        field="mandatory_finding_ids",
        max_items=100,
        max_chars=96,
    )
    evidence_refs = _validated_string_tuple(
        list(allowed_evidence_refs),
        field="allowed_evidence_refs",
        max_items=500,
        max_chars=128,
    )
    sorted_findings = tuple(sorted(findings))
    check_copy_bias_finding_ids = _bound_check_copy_bias_finding_ids(
        evidence_bundle,
        mandatory_finding_ids=sorted_findings,
    )
    allowed_issue_codes = tuple(
        code
        for code in SEMANTIC_ISSUE_CODES
        if code != "CHECK_COPY_BIAS" or check_copy_bias_finding_ids
    )
    return SemanticReviewContext(
        candidate_md=candidate_md,
        calibration_note=calibration_note,
        candidate_sha256=sha256_text(candidate_md),
        calibration_note_sha256=sha256_text(calibration_note),
        evidence_bundle_sha256=canonical_json_sha256(evidence_bundle),
        reference_card_sha256=reference_card_sha256(),
        proposer_model_family=proposer,
        reviewer_model_family=reviewer,
        independence_passed=proposer != reviewer,
        mandatory_finding_ids=sorted_findings,
        allowed_evidence_refs=tuple(sorted(evidence_refs)),
        allowed_issue_codes=allowed_issue_codes,
        check_copy_bias_finding_ids=check_copy_bias_finding_ids,
        bullets=extract_experience_bullets(candidate_md),
    )


def validate_semantic_review_payload(
    payload: Mapping[str, object],
    *,
    context: SemanticReviewContext,
) -> SemanticReview:
    """Validate one structured reviewer response against its bound context."""

    if not isinstance(payload, Mapping):
        raise SemanticReviewContractError("review payload must be an object")
    expected_keys = {
        "schema_version",
        "candidate_sha256",
        "evidence_bundle_sha256",
        "reference_card_sha256",
        "verdict",
        "acknowledged_finding_ids",
        "issues",
        "repairs",
    }
    _require_exact_keys(payload, expected_keys, field="review payload")
    if payload["schema_version"] != SEMANTIC_REVIEW_SCHEMA_VERSION:
        raise SemanticReviewContractError("unsupported semantic review schema version")
    _require_bound_hash(payload, "candidate_sha256", context.candidate_sha256)
    _require_bound_hash(
        payload,
        "evidence_bundle_sha256",
        context.evidence_bundle_sha256,
    )
    _require_bound_hash(
        payload,
        "reference_card_sha256",
        context.reference_card_sha256,
    )

    verdict = payload["verdict"]
    if verdict not in {"accept", "repair", "reject"}:
        raise SemanticReviewContractError("verdict must be accept, repair, or reject")
    if verdict in {"accept", "repair"} and not context.independence_passed:
        raise SemanticReviewContractError(
            "the proposer model family cannot accept or repair its own candidate"
        )

    acknowledged = _validated_string_tuple(
        payload["acknowledged_finding_ids"],
        field="acknowledged_finding_ids",
        max_items=100,
        max_chars=96,
    )
    if acknowledged != context.mandatory_finding_ids:
        raise SemanticReviewContractError(
            "acknowledged_finding_ids must exactly match host findings"
        )

    raw_issues = _require_list(payload["issues"], field="issues", max_items=20)
    issues = tuple(_parse_issue(item, context=context) for item in raw_issues)
    issue_ids = [item.issue_id for item in issues]
    if len(issue_ids) != len(set(issue_ids)):
        raise SemanticReviewContractError("issue_id values must be unique")
    expected_issue_ids = [f"I{index:03d}" for index in range(1, len(issues) + 1)]
    if issue_ids != expected_issue_ids:
        raise SemanticReviewContractError(
            "issue_id values must be I001, I002, ... in response order"
        )
    represented_findings = {
        finding_id for issue in issues for finding_id in issue.source_finding_ids
    }
    missing_findings = set(context.mandatory_finding_ids) - represented_findings
    if missing_findings:
        missing = ", ".join(sorted(missing_findings))
        raise SemanticReviewContractError(
            f"mandatory deterministic findings were not represented: {missing}"
        )
    if verdict == "repair" and any(issue.code in _NON_REPAIRABLE_ISSUE_CODES for issue in issues):
        raise SemanticReviewContractError(
            "document-wide behavior-copy bias requires rejection, not local repair"
        )

    raw_repairs = _require_list(
        payload["repairs"],
        field="repairs",
        max_items=MAX_MECHANICAL_REPAIRS,
    )
    repairs = tuple(_parse_repair(item, context=context, issues=issues) for item in raw_repairs)
    _validate_verdict_shape(verdict, issues=issues, repairs=repairs)
    _validate_repair_budget(context=context, repairs=repairs)
    if verdict == "repair":
        covered = {issue_id for repair in repairs for issue_id in repair.issue_ids}
        if covered != set(issue_ids):
            raise SemanticReviewContractError(
                "repair verdict must mechanically address every reported issue"
            )

    return SemanticReview(
        schema_version=SEMANTIC_REVIEW_SCHEMA_VERSION,
        candidate_sha256=context.candidate_sha256,
        evidence_bundle_sha256=context.evidence_bundle_sha256,
        reference_card_sha256=context.reference_card_sha256,
        verdict=str(verdict),
        acknowledged_finding_ids=tuple(acknowledged),
        issues=issues,
        repairs=repairs,
        proposer_model_family=context.proposer_model_family,
        reviewer_model_family=context.reviewer_model_family,
        independence_passed=context.independence_passed,
    )


def replay_semantic_review(
    review: SemanticReview,
    *,
    context: SemanticReviewContext,
    deterministic_validator: Callable[[str], Sequence[object] | Iterable[object]],
) -> SemanticReviewReplay:
    """Apply bounded repairs and replay deterministic validators without an LLM call."""

    if not isinstance(review, SemanticReview):
        raise TypeError("review must be SemanticReview")
    if not callable(deterministic_validator):
        raise TypeError("deterministic_validator must be callable")
    if review.candidate_sha256 != context.candidate_sha256:
        raise SemanticReviewContractError("replay candidate hash mismatch")
    if review.evidence_bundle_sha256 != context.evidence_bundle_sha256:
        raise SemanticReviewContractError("replay evidence hash mismatch")
    if review.reference_card_sha256 != context.reference_card_sha256:
        raise SemanticReviewContractError("replay reference-card hash mismatch")
    if review.verdict == "reject":
        return SemanticReviewReplay(
            status="rejected_by_reviewer",
            installable=False,
            document_md=context.candidate_md,
            calibration_note=context.calibration_note,
            document_sha256=context.candidate_sha256,
            calibration_note_sha256=context.calibration_note_sha256,
            applied_repair_count=0,
            validation_findings=(),
        )
    if not review.independence_passed or not context.independence_passed:
        raise SemanticReviewContractError("non-independent review cannot be installed")

    repaired_document, repaired_calibration = _apply_mechanical_repairs(
        context=context,
        repairs=review.repairs,
    )
    try:
        raw_findings = deterministic_validator(repaired_document)
        if raw_findings is None:
            raise TypeError("validator returned None")
        findings = tuple(_normalize_replay_finding(item) for item in raw_findings)
    except Exception as exc:
        raise SemanticReviewContractError("deterministic replay failed") from exc
    installable = not findings
    if installable:
        status = "accepted" if review.verdict == "accept" else "repaired"
    else:
        status = "rejected_after_replay"
    return SemanticReviewReplay(
        status=status,
        installable=installable,
        document_md=repaired_document,
        calibration_note=repaired_calibration,
        document_sha256=sha256_text(repaired_document),
        calibration_note_sha256=sha256_text(repaired_calibration),
        applied_repair_count=len(review.repairs),
        validation_findings=findings,
    )


def _parse_issue(
    raw: object,
    *,
    context: SemanticReviewContext,
) -> SemanticIssue:
    if not isinstance(raw, Mapping):
        raise SemanticReviewContractError("each issue must be an object")
    keys = {
        "issue_id",
        "code",
        "scope",
        "target_bullet_id",
        "claim_excerpt",
        "evidence_refs",
        "reference_card_ids",
        "source_finding_ids",
        "diagnosis",
    }
    _require_exact_keys(raw, keys, field="issue")
    issue_id = _require_string(raw["issue_id"], field="issue_id", max_chars=4)
    if not _ISSUE_ID_RE.fullmatch(issue_id):
        raise SemanticReviewContractError("issue_id must match I000")
    code = _require_string(raw["code"], field="issue code", max_chars=64)
    if code not in _SEMANTIC_ISSUE_CODE_SET:
        raise SemanticReviewContractError(f"unknown semantic issue code: {code}")
    if code not in context.allowed_issue_codes:
        raise SemanticReviewContractError(
            f"semantic issue code is not authorized by bound evidence: {code}"
        )
    scope = _require_string(raw["scope"], field="issue scope", max_chars=32)
    if scope not in {"document", "bullet", "calibration_note"}:
        raise SemanticReviewContractError("invalid issue scope")
    target_bullet_id = _nullable_bullet_id(raw["target_bullet_id"])
    excerpt = _require_string(raw["claim_excerpt"], field="claim_excerpt", max_chars=240)
    evidence_refs = _validated_string_tuple(
        raw["evidence_refs"],
        field="evidence_refs",
        max_items=12,
        max_chars=128,
    )
    unknown_evidence_refs = set(evidence_refs) - set(context.allowed_evidence_refs)
    if unknown_evidence_refs:
        raise SemanticReviewContractError(
            "issue cited evidence outside the bound bundle: "
            + ", ".join(sorted(unknown_evidence_refs))
        )
    card_ids = _validated_string_tuple(
        raw["reference_card_ids"],
        field="reference_card_ids",
        max_items=6,
        max_chars=64,
    )
    unknown_cards = set(card_ids) - _REFERENCE_CARD_ID_SET
    if unknown_cards:
        raise SemanticReviewContractError(
            f"unknown reference card IDs: {', '.join(sorted(unknown_cards))}"
        )
    source_ids = _validated_string_tuple(
        raw["source_finding_ids"],
        field="source_finding_ids",
        max_items=20,
        max_chars=96,
    )
    unknown_source_ids = set(source_ids) - set(context.mandatory_finding_ids)
    if unknown_source_ids:
        raise SemanticReviewContractError(
            "issue cited source finding IDs outside the bound contract"
        )
    if code == "CHECK_COPY_BIAS":
        if scope != "document":
            raise SemanticReviewContractError("CHECK_COPY_BIAS must be a document issue")
        if not set(source_ids).intersection(context.check_copy_bias_finding_ids):
            raise SemanticReviewContractError(
                "CHECK_COPY_BIAS is not authorized by bound evidence: "
                "qualifying host finding required"
            )
    diagnosis = _require_string(raw["diagnosis"], field="diagnosis", max_chars=300)
    if not diagnosis:
        raise SemanticReviewContractError("diagnosis must be non-empty")
    if not evidence_refs and not card_ids and not source_ids:
        raise SemanticReviewContractError("every issue needs evidence, a logic card, or a gate")

    section: str | None = None
    bullet_index: int | None = None
    bullet_sha: str | None = None
    if scope == "bullet":
        if target_bullet_id is None:
            raise SemanticReviewContractError("bullet issue requires target_bullet_id")
        bullet = _find_bullet_by_id(context, target_bullet_id)
        section = bullet.section
        bullet_index = bullet.index
        bullet_sha = bullet.sha256
        if excerpt and excerpt not in bullet.text:
            raise SemanticReviewContractError("issue excerpt is not in its target bullet")
    elif scope == "calibration_note":
        if target_bullet_id is not None:
            raise SemanticReviewContractError("calibration issue target_bullet_id must be null")
        if excerpt and excerpt not in context.calibration_note:
            raise SemanticReviewContractError("calibration excerpt is not in the bound note")
    else:
        if target_bullet_id is not None:
            raise SemanticReviewContractError("document issue target_bullet_id must be null")
        if excerpt:
            raise SemanticReviewContractError("document issue excerpt must be empty")

    return SemanticIssue(
        issue_id=issue_id,
        code=code,
        scope=scope,
        target_bullet_id=target_bullet_id,
        section=section,
        bullet_index=bullet_index,
        bullet_sha256=bullet_sha,
        claim_excerpt=excerpt,
        evidence_refs=evidence_refs,
        reference_card_ids=card_ids,
        source_finding_ids=source_ids,
        diagnosis=diagnosis,
    )


def _parse_repair(
    raw: object,
    *,
    context: SemanticReviewContext,
    issues: tuple[SemanticIssue, ...],
) -> MechanicalRepair:
    if not isinstance(raw, Mapping):
        raise SemanticReviewContractError("each repair must be an object")
    keys = {
        "op",
        "issue_ids",
        "target_bullet_id",
        "from_text",
        "to_text",
    }
    _require_exact_keys(raw, keys, field="repair")
    op = _require_string(raw["op"], field="repair op", max_chars=40)
    if op not in _MECHANICAL_REPAIR_OPERATION_SET:
        raise SemanticReviewContractError(f"unknown mechanical repair operation: {op}")
    issue_ids = _validated_string_tuple(
        raw["issue_ids"],
        field="repair issue_ids",
        max_items=20,
        max_chars=4,
    )
    if not issue_ids:
        raise SemanticReviewContractError("repair must link at least one issue")
    issue_by_id = {issue.issue_id: issue for issue in issues}
    if any(issue_id not in issue_by_id for issue_id in issue_ids):
        raise SemanticReviewContractError("repair references an unknown issue_id")
    linked = tuple(issue_by_id[issue_id] for issue_id in issue_ids)
    target_bullet_id = _nullable_bullet_id(raw["target_bullet_id"])
    from_text = _nullable_string(raw["from_text"], field="from_text", max_chars=240)
    to_text = _nullable_string(raw["to_text"], field="to_text", max_chars=240)

    section: str | None = None
    bullet_index: int | None = None
    original_sha: str | None = None
    if op == "neutralize_calibration_note":
        if target_bullet_id is not None:
            raise SemanticReviewContractError("calibration repair target_bullet_id must be null")
        if from_text is not None or to_text is not None:
            raise SemanticReviewContractError("calibration neutralization uses fixed host text")
        if any(
            issue.scope != "calibration_note" or issue.code not in _CALIBRATION_REPAIR_CODES
            for issue in linked
        ):
            raise SemanticReviewContractError("linked issues do not permit calibration repair")
    else:
        if target_bullet_id is None:
            raise SemanticReviewContractError("bullet repair requires target_bullet_id")
        bullet = _find_bullet_by_id(context, target_bullet_id)
        section = bullet.section
        bullet_index = bullet.index
        original_sha = bullet.sha256
        if any(
            issue.scope != "bullet" or issue.target_bullet_id != target_bullet_id
            for issue in linked
        ):
            raise SemanticReviewContractError("repair target differs from its linked issues")
        if op == "replace_action_term":
            if (from_text, to_text) not in _ALLOWED_ACTION_TERM_REPLACEMENTS:
                raise SemanticReviewContractError("action-term replacement is not allowlisted")
            if any(issue.code != "STATE_ACTION_MISMATCH" for issue in linked):
                raise SemanticReviewContractError(
                    "action-term replacement requires a state/action issue"
                )
            if bullet.text.count(str(from_text)) != 1:
                raise SemanticReviewContractError(
                    "action term must occur exactly once in the target bullet"
                )
        elif op == "delete_clause":
            if not from_text or to_text not in {None, ""}:
                raise SemanticReviewContractError("delete_clause requires exact from_text")
            if bullet.text.count(from_text) != 1:
                raise SemanticReviewContractError(
                    "deleted clause must occur exactly once in the target bullet"
                )
            if any(issue.code not in _REPAIRABLE_DELETE_CLAUSE_CODES for issue in linked):
                raise SemanticReviewContractError("linked issue does not permit clause deletion")
        elif op == "delete_bullet":
            if from_text is not None or to_text is not None:
                raise SemanticReviewContractError("delete_bullet takes no replacement text")

    return MechanicalRepair(
        op=op,
        issue_ids=issue_ids,
        target_bullet_id=target_bullet_id,
        section=section,
        bullet_index=bullet_index,
        original_sha256=original_sha,
        from_text=from_text,
        to_text=to_text,
    )


def _validate_verdict_shape(
    verdict: object,
    *,
    issues: tuple[SemanticIssue, ...],
    repairs: tuple[MechanicalRepair, ...],
) -> None:
    if verdict == "accept" and (issues or repairs):
        raise SemanticReviewContractError("accept requires empty issues and repairs")
    if verdict == "repair" and (not issues or not repairs):
        raise SemanticReviewContractError("repair requires issues and mechanical repairs")
    if verdict == "reject" and (not issues or repairs):
        raise SemanticReviewContractError("reject requires issues and forbids repairs")


def _validate_repair_budget(
    *,
    context: SemanticReviewContext,
    repairs: tuple[MechanicalRepair, ...],
) -> None:
    if len(repairs) > MAX_MECHANICAL_REPAIRS:
        raise SemanticReviewContractError("more than two mechanical repairs are forbidden")
    bullet_targets = [
        repair.target_bullet_id for repair in repairs if repair.op != "neutralize_calibration_note"
    ]
    if len(bullet_targets) != len(set(bullet_targets)):
        raise SemanticReviewContractError("a bullet may be repaired only once")
    if not bullet_targets:
        return
    total_rules = len(context.bullets)
    allowed = min(
        MAX_MECHANICAL_REPAIRS,
        max(1, math.floor(total_rules * MAX_REPAIRED_RULE_FRACTION)),
    )
    if len(bullet_targets) > allowed:
        raise SemanticReviewContractError(
            "mechanical repairs exceed twenty percent of candidate rules"
        )


def _apply_mechanical_repairs(
    *,
    context: SemanticReviewContext,
    repairs: tuple[MechanicalRepair, ...],
) -> tuple[str, str]:
    lines = context.candidate_md.splitlines()
    trailing_newline = context.candidate_md.endswith(("\n", "\r"))
    calibration = context.calibration_note
    bullet_repairs: list[tuple[int, ExperienceBullet, MechanicalRepair]] = []
    for repair in repairs:
        if repair.op == "neutralize_calibration_note":
            calibration = NEUTRAL_CALIBRATION_NOTE
            continue
        if repair.target_bullet_id is None:
            raise SemanticReviewContractError("replay encountered an incomplete bullet target")
        bullet = _find_bullet_by_id(context, repair.target_bullet_id)
        bullet_repairs.append((bullet.line_index, bullet, repair))

    for line_index, bullet, repair in sorted(bullet_repairs, reverse=True):
        if repair.op == "delete_bullet":
            del lines[line_index]
            continue
        if repair.from_text is None:
            raise SemanticReviewContractError("replay repair is missing from_text")
        if repair.op == "replace_action_term":
            if repair.to_text is None:
                raise SemanticReviewContractError("action replacement is missing to_text")
            new_text = bullet.text.replace(repair.from_text, repair.to_text, 1)
        elif repair.op == "delete_clause":
            new_text = bullet.text.replace(repair.from_text, "", 1).strip()
            if not new_text:
                raise SemanticReviewContractError("clause deletion emptied its bullet")
        else:
            raise SemanticReviewContractError(f"unsupported replay operation: {repair.op}")
        lines[line_index] = f"{bullet.prefix}{new_text}"

    document = "\n".join(lines)
    if trailing_newline:
        document += "\n"
    return document, calibration


def _normalize_replay_finding(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return {str(key): item for key, item in converted.items()}
    return {"message": str(value)}


def _find_bullet(
    context: SemanticReviewContext,
    *,
    section: str,
    index: int,
) -> ExperienceBullet:
    for bullet in context.bullets:
        if bullet.section == section and bullet.index == index:
            return bullet
    raise SemanticReviewContractError(f"candidate does not contain bullet {section}#{index}")


def _find_bullet_by_id(
    context: SemanticReviewContext,
    bullet_id: str,
) -> ExperienceBullet:
    for bullet in context.bullets:
        if bullet.bullet_id == bullet_id:
            return bullet
    raise SemanticReviewContractError(f"candidate does not contain target_bullet_id {bullet_id}")


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    field: str,
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise SemanticReviewContractError(f"{field} keys mismatch; missing={missing}, extra={extra}")


def _require_bound_hash(
    payload: Mapping[str, object],
    field: str,
    expected: str,
) -> None:
    value = payload[field]
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise SemanticReviewContractError(f"{field} must be a lowercase SHA-256")
    if value != expected:
        raise SemanticReviewContractError(f"{field} does not match bound input")


def _require_list(value: object, *, field: str, max_items: int) -> list[object]:
    if not isinstance(value, list):
        raise SemanticReviewContractError(f"{field} must be an array")
    if len(value) > max_items:
        raise SemanticReviewContractError(f"{field} exceeds its item limit")
    return value


def _validated_string_tuple(
    value: object,
    *,
    field: str,
    max_items: int,
    max_chars: int,
) -> tuple[str, ...]:
    raw = list(value) if not isinstance(value, list) and isinstance(value, tuple) else value
    items = _require_list(raw, field=field, max_items=max_items)
    result = tuple(_require_string(item, field=field, max_chars=max_chars) for item in items)
    if len(result) != len(set(result)):
        raise SemanticReviewContractError(f"{field} values must be unique")
    return result


def _require_string(value: object, *, field: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise SemanticReviewContractError(f"{field} must be a string")
    if len(value) > max_chars:
        raise SemanticReviewContractError(f"{field} exceeds {max_chars} characters")
    return value


def _nullable_string(value: object, *, field: str, max_chars: int) -> str | None:
    if value is None:
        return None
    return _require_string(value, field=field, max_chars=max_chars)


def _nullable_bullet_id(value: object) -> str | None:
    if value is None:
        return None
    bullet_id = _require_string(value, field="target_bullet_id", max_chars=4)
    if not _BULLET_ID_RE.fullmatch(bullet_id):
        raise SemanticReviewContractError("target_bullet_id must match B000 or be null")
    return bullet_id


def _nullable_section(value: object) -> str | None:
    if value is None:
        return None
    section = _require_string(value, field="section", max_chars=20)
    if section not in EXPERIENCE_SECTIONS:
        raise SemanticReviewContractError(f"unknown experience section: {section}")
    return section


def _nullable_positive_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticReviewContractError(f"{field} must be a positive integer or null")
    return value


def _nullable_hash(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise SemanticReviewContractError(f"{field} must be a lowercase SHA-256 or null")
    return value


__all__ = [
    "EXPERIENCE_SECTIONS",
    "MAX_MECHANICAL_REPAIRS",
    "MAX_REPAIRED_RULE_FRACTION",
    "MECHANICAL_REPAIR_OPERATIONS",
    "NEUTRAL_CALIBRATION_NOTE",
    "REFERENCE_CARD_IDS",
    "SEMANTIC_ISSUE_CODES",
    "SEMANTIC_REVIEW_FAILURE_CODES",
    "SEMANTIC_REVIEW_MAX_TRAJECTORY_HANDS",
    "SEMANTIC_REVIEW_POLICY_VERSION",
    "SEMANTIC_REVIEW_SCHEMA_VERSION",
    "STRATEGY_SEMANTIC_REVIEW_JSON_SCHEMA",
    "ExperienceBullet",
    "MechanicalRepair",
    "SemanticIssue",
    "SemanticReview",
    "SemanticReviewContext",
    "SemanticReviewContractError",
    "SemanticReviewReplay",
    "SemanticReviewRequestMaterial",
    "bind_semantic_review_context",
    "build_semantic_review_request_material",
    "build_semantic_review_evidence_view",
    "canonical_json_sha256",
    "canonical_model_family",
    "extract_experience_bullets",
    "reference_card_bundle",
    "reference_card_sha256",
    "replay_semantic_review",
    "semantic_review_contract_failure_code",
    "semantic_review_json_schema",
    "semantic_review_system_prompt",
    "sha256_text",
    "validate_semantic_review_payload",
]
