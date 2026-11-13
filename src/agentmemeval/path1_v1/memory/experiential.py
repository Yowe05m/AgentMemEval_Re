from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path

from agentmemeval.path1_v1.core.domain import (
    AgentObservation,
    DecisionEvent,
    ExperienceDocument,
    HandTrajectory,
    MemoryContext,
    MemoryScope,
    MemorySnapshot,
    utc_now_iso,
)
from agentmemeval.path1_v1.core.poker_position import public_position_label
from agentmemeval.path1_v1.core.protocols import LLMClient
from agentmemeval.path1_v1.environment.decision_facts import (
    build_decision_facts,
    chips_to_big_blinds,
)
from agentmemeval.path1_v1.llm.schemas import LLMRequest
from agentmemeval.path1_v1.llm.terminal_policy import (
    PATH1_TERMINAL_POLICY,
    Path1ResponseContractError,
    Path1TerminalError,
    active_path1_budget,
    path1_logical_request,
    require_path1_terminal_policy,
    terminal_from_exception,
)
from agentmemeval.path1_v1.memory.base import trajectory_quality
from agentmemeval.path1_v1.memory.experience_semantic_review import (
    SEMANTIC_REVIEW_POLICY_VERSION,
    SemanticReviewContractError,
    bind_semantic_review_context,
    build_semantic_review_evidence_view,
    build_semantic_review_request_material,
    canonical_model_family,
    extract_experience_bullets,
    replay_semantic_review,
    semantic_review_contract_failure_code,
    validate_semantic_review_payload,
)
from agentmemeval.path1_v1.memory.poker_experience_validation import (
    recommended_action_streets,
    validate_poker_experience_document,
)
from agentmemeval.path1_v1.prompts.experience_update import (
    EXPERIENCE_ASYNC_FACT_CLASSIFICATION_PROMPT,
    EXPERIENCE_ASYNC_TRANSFERABLE_FACT_CLASSIFICATION_PROMPT,
    EXPERIENCE_MAIN_REFLECTION_PROMPT,
    EXPERIENCE_PAPER_ASYNC_FACT_CLASSIFICATION_PROMPT,
    EXPERIENCE_PAPER_OBJECTIVE_COMPATIBILITY_ADDENDUM,
    EXPERIENCE_PAPER_STRATEGY_REVIEW_CARD,
    EXPERIENCE_PAPER_UPDATE_PROMPT,
    EXPERIENCE_PATCH_UPDATE_PROMPT,
    EXPERIENCE_SECTION_HEADINGS,
    EXPERIENCE_TITLE,
    EXPERIENCE_TRANSFERABLE_QUALITY_PROMPT,
    EXPERIENCE_UPDATE_PROMPT,
)
from agentmemeval.path1_v1.storage.jsonl_store import JsonlStore

EXPERIENCE_FULL_DOCUMENT_REVISION_SCHEMA_VERSION = "experience_revision_v1"
EXPERIENCE_PATCH_REVISION_SCHEMA_VERSION = "experience_revision_v2"
# Backward-compatible public aliases retained for callers that explicitly use the
# non-paper structured patch fixture.
EXPERIENCE_REVISION_SCHEMA_VERSION = EXPERIENCE_PATCH_REVISION_SCHEMA_VERSION
EXPERIENCE_LEGACY_FULL_DOCUMENT_PROMPT_VERSION = (
    "2026-07-26-v7-paper-full-document-evidence-authority"
)
EXPERIENCE_FULL_DOCUMENT_PROMPT_VERSION = "2026-07-30-paper-en-full-document-v1"
EXPERIENCE_PAPER_FULL_DOCUMENT_PROMPT_VERSION = "2026-07-30-paper-en-main-v1"
EXPERIENCE_REVISION_RECAP_VERSION = "paper_main_recap_compact_v3"
EXPERIENCE_PAPER_RECAP_RENDER_VERSION = "main_summarize_hand_text_v1"
EXPERIENCE_BEHAVIOR_DIGEST_VERSION = "paper_behavior_digest_v1"
EXPERIENCE_SEMANTIC_REVIEW_PROMPT_VERSION = SEMANTIC_REVIEW_POLICY_VERSION
EXPERIENCE_NEUTRALIZED_CALIBRATION_NOTE = (
    "The committed action and net reward were recorded; one hand cannot identify counterfactual EV."
)
EXPERIENCE_TRAJECTORY_SCHEMA_VERSION = "experience_trajectory_v2_explicit_intent"
EXPERIENCE_PATCH_PROMPT_VERSION = "2026-07-30-paper-en-patch-v1"
EXPERIENCE_PROMPT_VERSION = EXPERIENCE_PATCH_PROMPT_VERSION
EXPERIENCE_LEGACY_DOCUMENT_CONTRACT_VERSION = "five_section_actionable_v1"
EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION = "transferable_rulebook_v2"
EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION = "paper_five_section_v1"
# Backward-compatible alias retained for callers and historical fixtures that
# did not yet select the versioned transferable quality gate.
EXPERIENCE_DOCUMENT_CONTRACT_VERSION = EXPERIENCE_LEGACY_DOCUMENT_CONTRACT_VERSION
EXPERIENCE_MAX_RULES_PER_SECTION = 16
EXPERIENCE_MAX_RULES_TOTAL = 64
# Main remains the semantic mother implementation.  This local safety extension
# only prevents a mature paper-style document from growing without first
# consolidating; it never prevents an inherited oversized document from being
# repaired, merged, or reduced.
EXPERIENCE_PAPER_CONSOLIDATION_MAX_RULES_PER_SECTION = 16
EXPERIENCE_PAPER_CONSOLIDATION_MAX_RULES_TOTAL = 64
EXPERIENCE_REQUIRED_HEADINGS = tuple(f"## {heading}" for heading in EXPERIENCE_SECTION_HEADINGS)
EXPERIENCE_SECTION_LABELS = tuple(
    heading.removeprefix("## ") for heading in EXPERIENCE_REQUIRED_HEADINGS
)
EXPERIENCE_EMPTY_SECTION = "(empty)"
EXPERIENCE_ACCEPTED_EMPTY_SECTIONS = (
    EXPERIENCE_EMPTY_SECTION,
    "(no validated rules)",
)
_PAPER_NUMBERED_BULLET_RE = re.compile(r"^\s*\d{1,2}[.)]\s+(.+?)\s*$")
EXPERIENCE_REVISION_SYSTEM_TEMPLATE = (
    "Maintain a persistent long-term no-limit hold'em experience document. Prefer incremental "
    "content edits, but keep=false must return the complete five-section Markdown document, not "
    "a patch. Use only the supplied compact recaps and fact evidence. Do not include identities, "
    "hand or fact IDs, seats, hidden information, or single-hand counterfactual claims. Preserve "
    "valid rules and express visible conditions, actions or sizing, and invalidators. Objective "
    "cards, actions, stacks, and deterministic_decision_facts override "
    "declared_intent_not_ground_truth. Follow the requested JSON contract exactly. With keep=true, "
    "new_md is empty or unchanged; with keep=false, new_md begins with '# My Experience' and "
    "contains only the five fixed sections in order. The document is limited to {max_chars} "
    "characters; calibration_note and self_check are limited to 240 characters each. Output no "
    "explanation, fence, or reasoning trace."
)
EXPERIENCE_PATCH_REVISION_SYSTEM_TEMPLATE = (
    "Maintain the playbook through structured patches; code renders Markdown. Use only supplied "
    "trajectory and fact evidence. A trajectory proves an observed state, action, and outcome, "
    "not an unselected action. One hand can create only a tentative hypothesis. keep requires "
    "patches=[]; patch changes only named rules. Sections must use the five fixed English names. "
    "Every add/revise patch needs observable when conditions, explicit guidance, and invalidators. "
    "A model prior uses basis=model_prior and status=tentative. Keep trajectory and fact IDs in "
    "their dedicated fields. Output operation, patches, fact_assessments, calibration_note, and "
    "self_check. The rendered document is limited to {max_chars} characters and auxiliary notes "
    "to 240 characters each. Output no reasoning trace."
)

EXPERIENCE_TRANSFERABLE_REVISION_SYSTEM_TEMPLATE = (
    EXPERIENCE_REVISION_SYSTEM_TEMPLATE
    + " Apply transferable_rulebook_v2: merge duplicates before editing; allow at most 16 "
    "rules per section and 64 total. Exclude dealt suits, hand-by-hand examples, recent returns, "
    "and sample narratives. Poker terms, positions, hand classes, and action order must agree."
)

EXPERIENCE_PAPER_REVISION_SYSTEM_TEMPLATE = EXPERIENCE_MAIN_REFLECTION_PROMPT

INITIAL_EXPERIENCE = """# My Experience

## Starting Hands
(empty)

## Postflop Play
(empty)

## Turn and River
(empty)

## Opponent Adaptation
(empty)

## Bet Sizing and Position
(empty)
"""


class ExperientialMemory:
    name = "expr"

    def __init__(
        self,
        agent_id: str,
        scope: MemoryScope = "per_agent",
        window_size: int = 8,
        max_chars: int = 1600,
        prompt_token_budget: int = 32768,
        revision_prompt_window_size: int | None = None,
        update_period: int = 1,
        revision_strategy: str = "deterministic",
        llm_client: LLMClient | None = None,
        model: str = "",
        semantics_mode: str = "quality_filtered_v2",
        revision_contract: str | None = None,
        document_contract: str = EXPERIENCE_LEGACY_DOCUMENT_CONTRACT_VERSION,
        failure_policy: str = "fail_closed_keep",
        semantic_review_mode: str = "disabled",
        semantic_reviewer_client: LLMClient | None = None,
        semantic_reviewer_model: str = "",
    ) -> None:

        self.agent_id = agent_id
        self.scope: MemoryScope = scope
        self.window_size = window_size
        self.max_chars = max_chars
        if type(prompt_token_budget) is not int or prompt_token_budget < 1:
            raise ValueError("experience prompt_token_budget must be a positive integer")
        self.prompt_token_budget = prompt_token_budget
        if revision_prompt_window_size is not None and (
            type(revision_prompt_window_size) is not int
            or revision_prompt_window_size < 1
            or revision_prompt_window_size > window_size
        ):
            raise ValueError(
                "experience revision_prompt_window_size must be between 1 and window_size"
            )
        self.revision_prompt_window_size = revision_prompt_window_size
        self.update_period = max(1, update_period)
        if semantics_mode not in {"quality_filtered_v2", "paper_faithful_v1"}:
            raise ValueError(f"unknown experience semantics_mode: {semantics_mode}")
        resolved_revision_contract = revision_contract or (
            "full_document_v1" if semantics_mode == "paper_faithful_v1" else "structured_patch_v2"
        )
        if resolved_revision_contract not in {
            "full_document_v1",
            "structured_patch_v2",
        }:
            raise ValueError(f"unknown experience revision_contract: {revision_contract}")
        if document_contract not in {
            EXPERIENCE_LEGACY_DOCUMENT_CONTRACT_VERSION,
            EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION,
            EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION,
        }:
            raise ValueError(f"unknown experience document_contract: {document_contract}")
        if failure_policy != "fail_closed_keep":
            raise ValueError(f"unknown experience failure_policy: {failure_policy}")
        if semantic_review_mode not in {"disabled", "independent_model_v1"}:
            raise ValueError(f"unknown experience semantic_review_mode: {semantic_review_mode}")
        if semantic_review_mode == "independent_model_v1" and (
            semantic_reviewer_client is None or not semantic_reviewer_model.strip()
        ):
            raise ValueError(
                "independent_model_v1 requires semantic_reviewer_client and reviewer model"
            )
        if semantic_review_mode == "independent_model_v1" and (
            revision_strategy != "llm" or resolved_revision_contract != "full_document_v1"
        ):
            raise ValueError(
                "independent_model_v1 supports only the llm + full_document_v1 update path"
            )
        if semantic_review_mode == "independent_model_v1":
            proposer_family = canonical_model_family(model)
            reviewer_family = canonical_model_family(semantic_reviewer_model)
            if proposer_family == reviewer_family:
                raise ValueError(
                    "independent_model_v1 requires proposer and reviewer from different "
                    "model families"
                )
        else:
            proposer_family = ""
            reviewer_family = ""
        self.semantics_mode = semantics_mode
        self.revision_contract = resolved_revision_contract
        self.document_contract = document_contract
        self.failure_policy = failure_policy
        self.semantic_review_mode = semantic_review_mode
        self.semantic_reviewer_client = semantic_reviewer_client
        self.semantic_reviewer_model = semantic_reviewer_model
        self.semantic_proposer_model_family = proposer_family
        self.semantic_reviewer_model_family = reviewer_family
        if revision_strategy not in {"deterministic", "llm"}:
            raise ValueError(f"unknown experience revision strategy: {revision_strategy}")
        if revision_strategy == "llm" and llm_client is None:
            raise ValueError("LLM experience revision requires llm_client")
        self.revision_strategy = revision_strategy
        self.llm_client = llm_client
        self.model = model
        self.trajectories: list[HandTrajectory] = []
        self.skipped_trajectory_hand_ids: list[str] = []
        self.revision_log: list[dict[str, object]] = []
        self.frozen = False
        self.persistence_directory: Path | None = None
        self.persistence_reference_root: Path | None = None
        self.experience_path: Path | None = None
        self.trajectory_store: JsonlStore | None = None
        self.experience_log_store: JsonlStore | None = None
        self._persisted_trajectory_hand_ids: set[str] = set()
        self.context_build_count = 0
        self.last_injected_experience_version: int | None = None
        self.history: list[ExperienceDocument] = [
            ExperienceDocument(
                version=1,
                body=INITIAL_EXPERIENCE,
                source_hand_ids=[],
                updated_at=utc_now_iso(),
                scope=scope,
                metadata={
                    "reason": "initial",
                    "structure": "five_section_experience_doc",
                    "document_contract_version": self.document_contract,
                    "semantics_mode": self.semantics_mode,
                    "revision_contract": self.revision_contract,
                    "failure_policy": self.failure_policy,
                    "semantic_review_mode": self.semantic_review_mode,
                    "semantic_reviewer_model": self.semantic_reviewer_model,
                },
            )
        ]
        self._baseline_document = deepcopy(self.history[0])

    @property
    def current(self) -> ExperienceDocument:

        return self.history[-1]

    def build_context(self, observation: AgentObservation) -> MemoryContext:
        # Paper: Eq. (6) injects the current experience document at decision time.

        self._assert_canonical_document_current()
        self.context_build_count += 1
        self.last_injected_experience_version = self.current.version
        return MemoryContext(
            experience=self.current,
            metadata={
                "mechanism": self.name,
                "scope": self.scope,
                "experience_version": self.current.version,
                "document_contract": self.document_contract,
            },
        )

    def set_frozen(self, frozen: bool) -> None:
        """Enable or disable the read-only generalization lifecycle."""

        self.frozen = bool(frozen)

    def bind_persistence(
        self,
        directory: str | Path,
        *,
        reference_root: str | Path,
    ) -> None:
        """Materialize this Agent's canonical experience and append-only work logs."""

        resolved_directory = Path(directory).resolve()
        resolved_root = Path(reference_root).resolve()
        try:
            resolved_directory.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("experience persistence directory must stay inside run root") from exc
        resolved_directory.mkdir(parents=True, exist_ok=True)
        self.persistence_directory = resolved_directory
        self.persistence_reference_root = resolved_root
        self.experience_path = resolved_directory / "experience.md"
        self.trajectory_store = JsonlStore(resolved_directory / "trajectory_log.jsonl")
        self.experience_log_store = JsonlStore(resolved_directory / "experience_log.jsonl")
        self.trajectory_store.path.touch(exist_ok=True)
        self.experience_log_store.path.touch(exist_ok=True)
        self._persisted_trajectory_hand_ids = {
            str(row.get("hand_index", ""))
            for row in self.trajectory_store.read_all()
            if str(row.get("hand_index", ""))
        }
        self._atomic_write_experience(self.current.body)

    def experience_reference(self) -> dict[str, object]:
        """Return a body-free reference proving which canonical document was injected."""

        self._assert_canonical_document_current()
        relative_path: str | None = None
        if self.experience_path is not None and self.persistence_reference_root is not None:
            relative_path = self.experience_path.relative_to(
                self.persistence_reference_root
            ).as_posix()
        body = self.current.body
        return {
            "version": self.current.version,
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "chars": len(body),
            "path": relative_path,
        }

    def record_trajectory(self, trajectory: HandTrajectory) -> None:
        """Append one full visible hand recap without embedding the experience document."""

        if self.frozen:
            return
        if trajectory.hand_id in self._persisted_trajectory_hand_ids:
            return
        quality = trajectory_quality(trajectory)
        revision_recap = self._compact_revision_recap(trajectory)
        revision_recap_json = json.dumps(
            revision_recap,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        record = {
            "schema_version": EXPERIENCE_TRAJECTORY_SCHEMA_VERSION,
            "hand_index": trajectory.hand_id,
            "table_id": trajectory.table_id,
            "recap": trajectory.summary,
            "outcome": (
                "win"
                if trajectory.final_reward > 0
                else "loss"
                if trajectory.final_reward < 0
                else "draw"
            ),
            "won_amount": trajectory.won_amount,
            "won_amount_status": (
                "observed_from_hand_result_winners"
                if trajectory.won_amount is not None
                else "unavailable_legacy_trajectory"
            ),
            "net_reward": int(trajectory.final_reward),
            "memory_eligible": bool(quality["memory_eligible"]),
            "trajectory": self._trajectory_payload(trajectory),
            "revision_recap_version": EXPERIENCE_REVISION_RECAP_VERSION,
            "revision_recap": revision_recap,
            "revision_recap_sha256": hashlib.sha256(
                revision_recap_json.encode("utf-8")
            ).hexdigest(),
            "revision_recap_chars": len(revision_recap_json),
            "created_at": utc_now_iso(),
        }
        if self.trajectory_store is not None:
            self.trajectory_store.append(record)
        self._persisted_trajectory_hand_ids.add(trajectory.hand_id)

    def on_decision_committed(self, event: DecisionEvent) -> None:
        pass

    def on_hand_finished(
        self,
        trajectory: HandTrajectory,
        *,
        evidence_records: list[object] | None = None,
        evidence_groups: dict[str, list[object]] | None = None,
    ) -> None:
        # Paper: Eq. (7) updates experience from completed-hand evidence.

        if self.frozen:
            return
        self.record_trajectory(trajectory)
        quality = trajectory_quality(trajectory)
        if not trajectory.decision_events:
            self.skipped_trajectory_hand_ids.append(trajectory.hand_id)
            return
        if self.semantics_mode != "paper_faithful_v1" and not quality["memory_eligible"]:
            self.skipped_trajectory_hand_ids.append(trajectory.hand_id)
            return
        self.trajectories.append(trajectory)
        if len(self.trajectories) % self.update_period != 0:
            return
        recent = self.trajectories[-self.window_size :]
        revision_recent = (
            recent[-self.revision_prompt_window_size :]
            if self.revision_prompt_window_size is not None
            else recent
        )
        revision = self._revise_from_window(
            revision_recent,
            evidence_records=evidence_records,
            evidence_groups=evidence_groups,
        )
        revision["revision_prompt_compression"] = {
            "policy": (
                "recent_complete_hand_suffix_v1"
                if len(revision_recent) < len(recent)
                else "not_needed"
            ),
            "configured_window_hand_count": len(recent),
            "retained_complete_hand_count": len(revision_recent),
            "dropped_oldest_complete_hand_count": len(recent) - len(revision_recent),
            "retained_hand_ids": [item.hand_id for item in revision_recent],
        }
        self.commit_revision(revision_recent, revision)

    def commit_revision(
        self,
        recent: list[HandTrajectory],
        revision: dict[str, object],
        *,
        append_log: bool = True,
    ) -> int | None:
        """Audit and install one validated complete-document revision.

        Both per-hand Expr/Sync updates and Async sweeps use this boundary so a
        revision cannot bypass the same keep, failure, status, and document checks.
        Persistence adapters may observe the returned version without duplicating
        the in-memory installation policy.
        """

        if bool(revision.get("keep", False)) or bool(revision.get("fallback_used", False)):
            validation_status = str(revision.get("validation_status", ""))
            revision.setdefault(
                "application_status",
                (
                    "rejected_preserved_current"
                    if validation_status == "rejected"
                    else (
                        "fallback_preserved_current"
                        if bool(revision.get("fallback_used", False))
                        else "keep_preserved_current"
                    )
                ),
            )
            if append_log:
                self.revision_log.append(_compact_revision_record(revision))
            return None
        body = _normalize_markdown(str(revision.get("new_md", "")))
        main_full_document = (
            self.semantics_mode == "paper_faithful_v1"
            and self.document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION
        )
        validation_findings = (
            []
            if main_full_document and body.strip()
            else (
                validate_experience_document_findings(
                    body,
                    max_chars=self.max_chars,
                    document_contract=self.document_contract,
                )
                if body.strip()
                else [_validation_finding("empty_document", "empty experience document")]
            )
        )
        validation_errors = [_format_validation_finding(item) for item in validation_findings]
        advisory_findings = (
            []
            if main_full_document
            else experience_document_advisory_findings(
                body,
                document_contract=self.document_contract,
            )
        )
        revision.setdefault(
            "advisory_findings",
            _compact_validation_findings(advisory_findings, forbidden_ids=set()),
        )
        revision.setdefault("advisory_finding_count", len(advisory_findings))
        if revision.get("validation_status") != "accepted" or validation_errors:
            revision["keep"] = True
            revision["validation_status"] = "rejected"
            revision["failure"] = revision.get("failure") or (
                "invalid_experience_document: " + "; ".join(validation_errors)
                if validation_errors
                else "invalid_revision_status"
            )
            revision["application_status"] = "invalid_revision_preserved_current"
            revision["validation_findings"] = _compact_validation_findings(
                validation_findings, forbidden_ids=set()
            )
            if body.strip():
                revision["proposal_summary"] = _proposal_summary(self.current.body, body)
            if append_log:
                self.revision_log.append(_compact_revision_record(revision))
            return None
        if body == self.current.body:
            revision["keep"] = True
            revision["validation_status"] = "unchanged"
            revision["application_status"] = "unchanged_preserved_current"
            if append_log:
                self.revision_log.append(_compact_revision_record(revision))
            return None
        old_document = self.current
        new_document = ExperienceDocument(
            version=old_document.version + 1,
            body=body,
            source_hand_ids=[item.hand_id for item in recent],
            updated_at=utc_now_iso(),
            scope=self.scope,
            metadata={
                "window_size": len(recent),
                "update_index": len(self.history),
                "strategy": revision["revision_strategy"],
                "schema_version": revision.get("schema_version"),
                "prompt_version": revision.get("prompt_version"),
                "prompt_sha256": revision.get("prompt_sha256"),
                "fallback_used": revision.get("fallback_used", False),
                "failure": revision.get("failure"),
                "validation_status": revision.get("validation_status"),
                "document_contract_version": self.document_contract,
                "semantics_mode": self.semantics_mode,
                "revision_contract": self.revision_contract,
                "failure_policy": self.failure_policy,
                "semantic_review_mode": revision.get(
                    "semantic_review_mode", self.semantic_review_mode
                ),
                "semantic_review_status": revision.get("semantic_review_status"),
                "semantic_review_verdict": revision.get("semantic_review_verdict"),
                "semantic_review_candidate_sha256": revision.get(
                    "semantic_review_candidate_sha256"
                ),
                "semantic_review_evidence_bundle_sha256": revision.get(
                    "semantic_review_evidence_bundle_sha256"
                ),
                "semantic_review_reference_card_sha256": revision.get(
                    "semantic_review_reference_card_sha256"
                ),
                "semantic_review_proposer_family": revision.get("semantic_review_proposer_family"),
                "semantic_review_reviewer_family": revision.get("semantic_review_reviewer_family"),
                "semantic_review_applied_repair_count": revision.get(
                    "semantic_review_applied_repair_count", 0
                ),
                "calibration_note": revision.get("calibration_note", ""),
                "self_check": revision.get("self_check", ""),
                "supporting_fact_ids": revision.get("supporting_fact_ids", []),
                "contradicting_fact_ids": revision.get("contradicting_fact_ids", []),
                "noise_fact_ids": revision.get("noise_fact_ids", []),
                "irrelevant_fact_ids": revision.get("irrelevant_fact_ids", []),
                "ambiguous_fact_ids": revision.get("ambiguous_fact_ids", []),
                "evidence_fact_ids": revision.get("sweep_evidence_fact_ids", []),
                "evidence_classification_status": revision.get(
                    "evidence_classification_status", "not_requested"
                ),
                "provisional_classification_source": revision.get(
                    "provisional_classification_source"
                ),
                "supporting_trajectory_ids": revision.get("supporting_trajectory_ids", []),
                "rule_catalog": revision.get("rule_catalog", []),
                "applied_patches": revision.get("applied_patches", []),
            },
        )
        self._atomic_write_experience(body)
        self.history.append(new_document)
        try:
            self._append_experience_log(old_document, new_document, revision)
        except Exception:
            self.history.pop()
            self._atomic_write_experience(old_document.body)
            raise
        revision["experience_revision_created"] = True
        revision["application_status"] = "applied"
        revision["installed_version"] = new_document.version
        if append_log:
            self.revision_log.append(_compact_revision_record(revision))
        return new_document.version

    def _append_experience_log(
        self,
        old_document: ExperienceDocument,
        new_document: ExperienceDocument,
        revision: dict[str, object],
    ) -> None:
        """Append the only JSON artifact that intentionally retains full old/new versions."""

        if self.experience_log_store is None:
            return
        self.experience_log_store.append(
            {
                "schema_version": "experience_log_v1",
                "rev": new_document.version - 1,
                "hand_index": (
                    new_document.source_hand_ids[-1] if new_document.source_hand_ids else ""
                ),
                "source_hand_ids": list(new_document.source_hand_ids),
                "old_md": old_document.body,
                "new_md": new_document.body,
                "old_md_sha256": hashlib.sha256(old_document.body.encode("utf-8")).hexdigest(),
                "new_md_sha256": hashlib.sha256(new_document.body.encode("utf-8")).hexdigest(),
                "calibration_note": str(revision.get("calibration_note", "")),
                "self_check": str(revision.get("self_check", "")),
                "supporting_fact_ids": list(revision.get("supporting_fact_ids", []) or []),
                "contradicting_fact_ids": list(revision.get("contradicting_fact_ids", []) or []),
                "noise_fact_ids": list(revision.get("noise_fact_ids", []) or []),
                "revision_strategy": revision.get("revision_strategy"),
                "schema_version_response": revision.get("schema_version"),
                "prompt_version": revision.get("prompt_version"),
                "prompt_sha256": revision.get("prompt_sha256"),
                "document_contract": self.document_contract,
                "semantic_review_mode": revision.get(
                    "semantic_review_mode", self.semantic_review_mode
                ),
                "semantic_review_status": revision.get("semantic_review_status"),
                "semantic_review_verdict": revision.get("semantic_review_verdict"),
                "semantic_review_prompt_version": revision.get("semantic_review_prompt_version"),
                "semantic_review_prompt_sha256": revision.get("semantic_review_prompt_sha256"),
                "semantic_review_candidate_sha256": revision.get(
                    "semantic_review_candidate_sha256"
                ),
                "semantic_review_evidence_bundle_sha256": revision.get(
                    "semantic_review_evidence_bundle_sha256"
                ),
                "semantic_review_reference_card_sha256": revision.get(
                    "semantic_review_reference_card_sha256"
                ),
                "semantic_review_issue_codes": list(
                    revision.get("semantic_review_issue_codes", []) or []
                ),
                "semantic_review_applied_repair_count": revision.get(
                    "semantic_review_applied_repair_count", 0
                ),
                "section_rule_counts": _proposal_summary(old_document.body, new_document.body)[
                    "section_rule_counts"
                ],
                "created_at": new_document.updated_at,
            }
        )

    def _atomic_write_experience(self, body: str) -> None:
        """Replace experience.md atomically inside its own directory."""

        if self.experience_path is None:
            return
        self.experience_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".exp.",
            suffix=".tmp",
            dir=str(self.experience_path.parent),
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.experience_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _assert_canonical_document_current(self) -> None:
        """Fail closed if the file used for decisions diverges from the versioned state."""

        if self.experience_path is None:
            return
        if not self.experience_path.exists():
            raise RuntimeError("canonical experience.md is missing")
        persisted = self.experience_path.read_text(encoding="utf-8")
        if persisted != self.current.body:
            raise RuntimeError("canonical experience.md diverged from in-memory current version")

    def snapshot(self) -> MemorySnapshot:

        return MemorySnapshot(
            mechanism=self.name,
            agent_id=self.agent_id,
            scope=self.scope,
            payload={
                "window_size": self.window_size,
                "max_chars": self.max_chars,
                "prompt_token_budget": self.prompt_token_budget,
                "revision_prompt_window_size": self.revision_prompt_window_size,
                "update_period": self.update_period,
                "revision_strategy": self.revision_strategy,
                "revision_model": self.model,
                "semantics_mode": self.semantics_mode,
                "revision_contract": self.revision_contract,
                "document_contract": self.document_contract,
                "failure_policy": self.failure_policy,
                "semantic_review_mode": self.semantic_review_mode,
                "semantic_reviewer_model": self.semantic_reviewer_model,
                "semantic_proposer_model_family": (self.semantic_proposer_model_family),
                "semantic_reviewer_model_family": (self.semantic_reviewer_model_family),
                "revision_recap_version": EXPERIENCE_REVISION_RECAP_VERSION,
                "trajectory_schema_version": EXPERIENCE_TRAJECTORY_SCHEMA_VERSION,
                "history": deepcopy([doc.to_dict() for doc in self.history]),
                "revision_log": deepcopy(self.revision_log),
                "skipped_trajectory_hand_ids": list(self.skipped_trajectory_hand_ids),
                "context_build_count": self.context_build_count,
                "last_injected_experience_version": (self.last_injected_experience_version),
            },
        )

    def restore(self, snapshot: MemorySnapshot) -> None:

        if snapshot.mechanism != self.name:
            raise ValueError(
                f"Experience snapshot mechanism mismatch: {snapshot.mechanism} != {self.name}"
            )
        if snapshot.agent_id != self.agent_id:
            raise ValueError(
                f"Experience snapshot agent_id mismatch: {snapshot.agent_id} != {self.agent_id}"
            )
        if snapshot.scope != self.scope:
            raise ValueError(
                f"Experience snapshot scope mismatch: {snapshot.scope} != {self.scope}"
            )
        if not isinstance(snapshot.payload, dict):
            raise ValueError("Experience snapshot payload must be an object")
        payload = snapshot.payload

        snapshot_semantics_mode = payload.get("semantics_mode", self.semantics_mode)
        if not isinstance(snapshot_semantics_mode, str):
            raise ValueError("Experience snapshot semantics_mode must be a string")
        if snapshot_semantics_mode != self.semantics_mode:
            raise ValueError(
                "Experience snapshot semantics_mode mismatch: "
                f"{snapshot_semantics_mode} != {self.semantics_mode}"
            )
        snapshot_contract = payload.get("revision_contract", self.revision_contract)
        if not isinstance(snapshot_contract, str):
            raise ValueError("Experience snapshot revision_contract must be a string")
        if snapshot_contract != self.revision_contract:
            raise ValueError(
                "Experience snapshot revision_contract mismatch: "
                f"{snapshot_contract} != {self.revision_contract}"
            )
        snapshot_document_contract = payload.get("document_contract", self.document_contract)
        if not isinstance(snapshot_document_contract, str):
            raise ValueError("Experience snapshot document_contract must be a string")
        if snapshot_document_contract != self.document_contract:
            raise ValueError(
                "Experience snapshot document_contract mismatch: "
                f"{snapshot_document_contract} != {self.document_contract}"
            )
        snapshot_failure_policy = payload.get("failure_policy", self.failure_policy)
        if not isinstance(snapshot_failure_policy, str):
            raise ValueError("Experience snapshot failure_policy must be a string")
        if snapshot_failure_policy != self.failure_policy:
            raise ValueError(
                "Experience snapshot failure_policy mismatch: "
                f"{snapshot_failure_policy} != {self.failure_policy}"
            )
        snapshot_semantic_review_mode = payload.get(
            "semantic_review_mode", self.semantic_review_mode
        )
        snapshot_semantic_reviewer_model = payload.get(
            "semantic_reviewer_model", self.semantic_reviewer_model
        )
        snapshot_semantic_proposer_family = payload.get(
            "semantic_proposer_model_family", self.semantic_proposer_model_family
        )
        snapshot_semantic_reviewer_family = payload.get(
            "semantic_reviewer_model_family", self.semantic_reviewer_model_family
        )
        for field, value in (
            ("semantic_review_mode", snapshot_semantic_review_mode),
            ("semantic_reviewer_model", snapshot_semantic_reviewer_model),
            ("semantic_proposer_model_family", snapshot_semantic_proposer_family),
            ("semantic_reviewer_model_family", snapshot_semantic_reviewer_family),
        ):
            if not isinstance(value, str):
                raise ValueError(f"Experience snapshot {field} must be a string")
        raw_snapshot_max_chars = payload.get("max_chars", self.max_chars)
        if type(raw_snapshot_max_chars) is not int or raw_snapshot_max_chars != self.max_chars:
            raise ValueError(
                "Experience snapshot max_chars mismatch: "
                f"{raw_snapshot_max_chars!r} != {self.max_chars}"
            )
        snapshot_prompt_token_budget = _snapshot_positive_integer(
            payload.get("prompt_token_budget", self.prompt_token_budget),
            "Experience snapshot prompt_token_budget",
        )
        raw_snapshot_revision_prompt_window_size = payload.get(
            "revision_prompt_window_size", self.revision_prompt_window_size
        )
        snapshot_revision_prompt_window_size = (
            None
            if raw_snapshot_revision_prompt_window_size is None
            else _snapshot_positive_integer(
                raw_snapshot_revision_prompt_window_size,
                "Experience snapshot revision_prompt_window_size",
            )
        )
        snapshot_window_size = _snapshot_positive_integer(
            payload.get("window_size", self.window_size),
            "Experience snapshot window_size",
        )
        snapshot_update_period = _snapshot_positive_integer(
            payload.get("update_period", self.update_period),
            "Experience snapshot update_period",
        )
        snapshot_revision_strategy = payload.get("revision_strategy", self.revision_strategy)
        snapshot_revision_model = payload.get("revision_model", self.model)
        snapshot_revision_recap_version = payload.get(
            "revision_recap_version", EXPERIENCE_REVISION_RECAP_VERSION
        )
        snapshot_trajectory_schema_version = payload.get(
            "trajectory_schema_version", EXPERIENCE_TRAJECTORY_SCHEMA_VERSION
        )
        if not isinstance(snapshot_revision_strategy, str):
            raise ValueError("Experience snapshot revision_strategy must be a string")
        if not isinstance(snapshot_revision_model, str):
            raise ValueError("Experience snapshot revision_model must be a string")
        if not isinstance(snapshot_revision_recap_version, str):
            raise ValueError("Experience snapshot revision_recap_version must be a string")
        if not isinstance(snapshot_trajectory_schema_version, str):
            raise ValueError("Experience snapshot trajectory_schema_version must be a string")
        semantic_mismatches = [
            field
            for field, snapshot_value, live_value in (
                ("window_size", snapshot_window_size, self.window_size),
                ("update_period", snapshot_update_period, self.update_period),
                (
                    "revision_strategy",
                    snapshot_revision_strategy,
                    self.revision_strategy,
                ),
                ("revision_model", snapshot_revision_model, self.model),
                (
                    "prompt_token_budget",
                    snapshot_prompt_token_budget,
                    self.prompt_token_budget,
                ),
                (
                    "revision_prompt_window_size",
                    snapshot_revision_prompt_window_size,
                    self.revision_prompt_window_size,
                ),
                (
                    "revision_recap_version",
                    snapshot_revision_recap_version,
                    EXPERIENCE_REVISION_RECAP_VERSION,
                ),
                (
                    "trajectory_schema_version",
                    snapshot_trajectory_schema_version,
                    EXPERIENCE_TRAJECTORY_SCHEMA_VERSION,
                ),
                (
                    "semantic_review_mode",
                    snapshot_semantic_review_mode,
                    self.semantic_review_mode,
                ),
                (
                    "semantic_reviewer_model",
                    snapshot_semantic_reviewer_model,
                    self.semantic_reviewer_model,
                ),
                (
                    "semantic_proposer_model_family",
                    snapshot_semantic_proposer_family,
                    self.semantic_proposer_model_family,
                ),
                (
                    "semantic_reviewer_model_family",
                    snapshot_semantic_reviewer_family,
                    self.semantic_reviewer_model_family,
                ),
            )
            if snapshot_value != live_value
        ]
        if semantic_mismatches:
            raise ValueError(
                "Experience snapshot semantic config mismatch: " + ", ".join(semantic_mismatches)
            )

        raw_history = payload.get("history", [])
        restored_history = _validated_experience_history(
            raw_history,
            agent_id=self.agent_id,
            scope=self.scope,
        )
        candidate_history = restored_history or [deepcopy(self._baseline_document)]
        restored_current = candidate_history[-1]
        histories_to_validate = candidate_history
        if self.document_contract in {
            EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION,
            EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION,
        }:
            all_findings: list[dict[str, object]] = []
            for document in histories_to_validate:
                all_findings.extend(
                    validate_experience_document_findings(
                        document.body,
                        max_chars=self.max_chars,
                        document_contract=self.document_contract,
                    )
                )
            if all_findings:
                codes = ",".join(
                    sorted({str(item.get("code", "invalid_document")) for item in all_findings})
                )
                validation_label = (
                    "transferable"
                    if self.document_contract == EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION
                    else "paper"
                )
                raise ValueError(
                    "Experience snapshot current document failed "
                    f"{validation_label} validation: {codes}"
                )
        raw_revision_log = payload.get("revision_log", [])
        if not isinstance(raw_revision_log, list) or not all(
            isinstance(item, dict) for item in raw_revision_log
        ):
            raise ValueError("Experience snapshot revision_log must be an object list")
        restored_revision_log = [
            _compact_revision_record(deepcopy(item)) for item in raw_revision_log
        ]
        raw_skipped = payload.get("skipped_trajectory_hand_ids", [])
        if not isinstance(raw_skipped, list) or not all(
            isinstance(item, str) for item in raw_skipped
        ):
            raise ValueError(
                "Experience snapshot skipped_trajectory_hand_ids must be a string list"
            )
        restored_context_build_count = _snapshot_nonnegative_integer(
            payload.get("context_build_count", 0),
            "Experience snapshot context_build_count",
        )
        raw_injected_version = payload.get("last_injected_experience_version")
        restored_injected_version = (
            None
            if raw_injected_version is None
            else _snapshot_positive_integer(
                raw_injected_version,
                "Experience snapshot last_injected_experience_version",
            )
        )
        if (
            restored_injected_version is not None
            and restored_injected_version > restored_current.version
        ):
            raise ValueError("Experience snapshot last injected version exceeds current version")

        # Persist the candidate document before committing in-memory state.  If
        # the atomic file replacement fails, the live object is unchanged.
        self._atomic_write_experience(restored_current.body)
        self.history = candidate_history
        self.revision_log = restored_revision_log
        self.skipped_trajectory_hand_ids = list(raw_skipped)
        self.context_build_count = restored_context_build_count
        self.last_injected_experience_version = restored_injected_version
        self.trajectories = []

    def metrics(self) -> dict[str, object]:

        main_full_document = (
            self.semantics_mode == "paper_faithful_v1"
            and self.document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION
        )
        validation_findings = (
            []
            if main_full_document
            else validate_experience_document_findings(
                self.current.body,
                max_chars=self.max_chars,
                document_contract=self.document_contract,
            )
        )
        validation_errors = [_format_validation_finding(item) for item in validation_findings]
        advisory_findings = (
            []
            if main_full_document
            else experience_document_advisory_findings(
                self.current.body,
                document_contract=self.document_contract,
            )
        )
        quality_summary = _experience_quality_summary(self.current.body)
        rejected_revisions = [
            item for item in self.revision_log if is_experience_revision_validation_rejection(item)
        ]
        initial_rejections = [
            item for item in self.revision_log if experience_revision_has_initial_rejection(item)
        ]
        operational_failure_count = sum(
            is_experience_revision_operational_failure(item) for item in self.revision_log
        )
        rejection_diagnostic_count = sum(
            experience_revision_rejection_diagnostic_complete(item) for item in rejected_revisions
        )
        initial_rejection_diagnostic_count = sum(
            experience_revision_initial_rejection_diagnostic_complete(item)
            for item in initial_rejections
        )
        return {
            "mechanism": self.name,
            "fact_count": 0,
            "experience_updates": max(0, len(self.history) - 1),
            "experience_version": self.current.version,
            "experience_chars": len(self.current.body),
            "revision_count": len(self.revision_log),
            "revision_attempt_count": len(self.revision_log),
            "revision_applied_count": max(0, len(self.history) - 1),
            "revision_strategy": self.revision_strategy,
            "revision_model": self.model,
            "semantics_mode": self.semantics_mode,
            "revision_contract": self.revision_contract,
            "document_contract": self.document_contract,
            "failure_policy": self.failure_policy,
            "semantic_review_mode": self.semantic_review_mode,
            "semantic_reviewer_model": self.semantic_reviewer_model,
            "semantic_proposer_model_family": self.semantic_proposer_model_family,
            "semantic_reviewer_model_family": self.semantic_reviewer_model_family,
            "prompt_token_budget": self.prompt_token_budget,
            "revision_prompt_window_size": self.revision_prompt_window_size,
            "revision_fallback_count": sum(
                bool(item.get("fallback_used")) for item in self.revision_log
            ),
            "revision_keep_count": sum(bool(item.get("keep")) for item in self.revision_log),
            "revision_validation_rejection_count": sum(
                is_experience_revision_validation_rejection(item) for item in self.revision_log
            ),
            "revision_rejected_count": sum(
                is_experience_revision_validation_rejection(item) for item in self.revision_log
            ),
            "revision_rejection_diagnostic_count": rejection_diagnostic_count,
            "revision_rejection_missing_diagnostic_count": (
                len(rejected_revisions) - rejection_diagnostic_count
            ),
            "revision_repair_initial_rejection_count": len(initial_rejections),
            "revision_repair_initial_diagnostic_count": (initial_rejection_diagnostic_count),
            "revision_repair_initial_missing_diagnostic_count": (
                len(initial_rejections) - initial_rejection_diagnostic_count
            ),
            "revision_contract_violation_count": sum(
                bool(item.get("contract_violation")) or bool(item.get("keep_contract_violation"))
                for item in self.revision_log
            ),
            # Validation rejection is a successful fail-closed guard outcome: the
            # rejected proposal reason remains in ``failure`` for diagnostics, but
            # it is not an operational/provider failure and did not replace the
            # canonical document.
            "revision_failure_count": operational_failure_count,
            "revision_failed_count": operational_failure_count,
            "revision_repair_count": sum(
                bool(item.get("repair_used")) for item in self.revision_log
            ),
            "semantic_review_count": sum(
                bool(item.get("semantic_review_used")) for item in self.revision_log
            ),
            "semantic_review_accept_count": sum(
                item.get("semantic_review_status") == "accepted" for item in self.revision_log
            ),
            "semantic_review_repair_count": sum(
                item.get("semantic_review_status") == "repaired" for item in self.revision_log
            ),
            "semantic_review_reject_count": sum(
                str(item.get("semantic_review_status", "")).startswith("rejected")
                for item in self.revision_log
            ),
            "semantic_review_failure_count": sum(
                item.get("semantic_review_status") == "reviewer_failure"
                for item in self.revision_log
            ),
            "experience_document_valid": not validation_errors,
            "experience_document_validation_errors": validation_errors,
            "experience_document_validation_findings": _compact_validation_findings(
                validation_findings, forbidden_ids=set()
            ),
            "experience_document_advisory_findings": _compact_validation_findings(
                advisory_findings, forbidden_ids=set()
            ),
            "experience_document_advisory_count": len(advisory_findings),
            "experience_rule_counts": quality_summary["section_rule_counts"],
            "experience_rule_count": quality_summary["total_rule_count"],
            "experience_near_duplicate_advisory_count": quality_summary[
                "near_duplicate_advisory_count"
            ],
            "experience_context_build_count": self.context_build_count,
            "last_injected_experience_version": self.last_injected_experience_version,
            "experience_injection_current": (
                self.last_injected_experience_version == self.current.version
                if self.context_build_count
                else False
            ),
            "last_revision": self.revision_log[-1] if self.revision_log else {},
            "skipped_fallback_trajectories": len(self.skipped_trajectory_hand_ids),
        }

    def _revise_from_window(
        self,
        recent: list[HandTrajectory],
        evidence_records: list[object] | None = None,
        evidence_groups: dict[str, list[object]] | None = None,
    ) -> dict[str, object]:
        """Run the selected revision strategy under its explicit failure policy."""

        if getattr(self.llm_client, "terminal_policy_version", None) == PATH1_TERMINAL_POLICY:
            require_path1_terminal_policy(self.llm_client)
            if (self.revision_strategy != "llm" or self.revision_contract != "structured_patch_v2"
                    or self.semantics_mode != "quality_filtered_v2"
                    or self.semantic_review_mode != "disabled"):
                raise Path1TerminalError("configuration", "unsupported_path1_memory_contract")
            with path1_logical_request(self.llm_client, "revision") as budget:
                try:
                    revision = self._revise_structured_patch_from_window(
                        recent, evidence_records=evidence_records,
                    )
                    revision["terminal_policy_receipt"] = budget.receipt()
                    if revision.get("validation_status") in {
                        "rejected", "response_contract_failed",
                    }:
                        revision["terminal_outcome"] = "response_contract_preserved_current"
                        revision["fallback_used"] = False
                    else:
                        revision["terminal_outcome"] = "completed_revision"
                    return revision
                except Exception as exc:
                    terminal = terminal_from_exception(exc, stage="memory_revision")
                    terminal.diagnostics["logical_request"] = budget.receipt()
                    if terminal is exc:
                        raise
                    raise terminal from exc

        if self.revision_strategy == "deterministic":
            return self._deterministic_revision(recent)
        if self.revision_contract == "full_document_v1":
            return self._revise_full_document_from_window(
                recent,
                evidence_records=evidence_records,
                evidence_groups=evidence_groups,
            )
        return self._revise_structured_patch_from_window(
            recent,
            evidence_records=evidence_records,
        )

    def _revise_structured_patch_from_window(
        self,
        recent: list[HandTrajectory],
        evidence_records: list[object] | None = None,
    ) -> dict[str, object]:
        """Run the retained non-paper typed-patch compatibility contract."""

        assert self.llm_client is not None
        trajectory_ids = [item.hand_id for item in recent]
        known_agent_ids = {
            player.agent_id
            for trajectory in recent
            for event in trajectory.decision_events
            for player in event.observation.players
        }
        known_agent_ids.add(self.agent_id)
        fact_payload = [
            {
                "record_id": str(getattr(item, "record_id", "")),
                "state_summary": _redact_agent_ids(
                    str(getattr(item, "state_summary", "")), known_agent_ids, self.agent_id
                ),
                "action_summary": _redact_agent_ids(
                    str(getattr(item, "action_summary", "")), known_agent_ids, self.agent_id
                ),
                "final_reward": int(getattr(item, "final_reward", 0)),
                "stability": float(getattr(item, "stability", 1.0)),
            }
            for item in (evidence_records or [])
        ]
        fact_ids = [str(item["record_id"]) for item in fact_payload if item["record_id"]]
        evidence_ids = [*trajectory_ids, *fact_ids]
        trajectory_payload = [self._compact_revision_recap(item) for item in recent]
        trajectory_payload_json = json.dumps(
            trajectory_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        revision_recap_metadata = {
            "revision_recap_version": EXPERIENCE_REVISION_RECAP_VERSION,
            "revision_recap_sha256": hashlib.sha256(
                trajectory_payload_json.encode("utf-8")
            ).hexdigest(),
            "revision_recap_chars": len(trajectory_payload_json),
        }
        current_rule_catalog = _current_rule_catalog(self.current)
        format_example = {
            "operation": "patch",
            "patches": [
                {
                    "section": EXPERIENCE_SECTION_LABELS[0],
                    "op": "add",
                    "rule_id": "",
                    "when": ["position is favorable and the entry price is low"],
                    "guidance": "choose a call or the minimum legal raise",
                    "invalidators": [
                        "an earlier player re-raises or the effective stack is shallow"
                    ],
                    "basis": "observed_evidence",
                    "status": "tentative",
                    "supporting_trajectory_ids": trajectory_ids[-1:],
                    "supporting_fact_ids": [],
                }
            ],
            "fact_assessments": [
                {"fact_id": fact_id, "label": "irrelevant"} for fact_id in fact_ids
            ],
            "calibration_note": "Format example only; reassess the supplied evidence.",
            "self_check": "One trajectory supports only a tentative rule with invalidators.",
        }
        system_prompt = EXPERIENCE_PATCH_REVISION_SYSTEM_TEMPLATE.format(max_chars=self.max_chars)
        user_prompt = (
            f"{EXPERIENCE_PATCH_UPDATE_PROMPT}\n\n"
            "JSON field example (copy structure only, not strategy content or "
            "classifications):\n"
            f"{json.dumps(format_example, ensure_ascii=False)}\n\n"
            f"Current document:\n{self.current.body}\n\n"
            "Current rule catalog (revise/delete may reference only these rule IDs):\n"
            f"{json.dumps(current_rule_catalog, ensure_ascii=False)}\n\n"
            f"Trajectory evidence:\n{trajectory_payload_json}\n\n"
            f"Retrieved fact evidence:\n{json.dumps(fact_payload, ensure_ascii=False)}"
        )
        prompt_hash = hashlib.sha256(
            (system_prompt + "\n" + user_prompt).encode("utf-8")
        ).hexdigest()
        observation = next(
            (
                event.observation
                for trajectory in reversed(recent)
                for event in reversed(trajectory.decision_events)
            ),
            None,
        )
        if observation is None:
            if active_path1_budget() is not None:
                raise Path1TerminalError("state", "no_decision_observation_for_revision")
            failed_revision = self._fail_closed_revision(
                recent,
                failure="no_decision_observation_for_llm_revision",
                prompt_hash=prompt_hash,
                forbidden_ids=known_agent_ids | set(evidence_ids),
            )
            failed_revision.update(revision_recap_metadata)
            return failed_revision
        request = LLMRequest(
            observation=observation,
            memory_context=MemoryContext(
                experience=self.current,
                metadata={
                    "mechanism": self.name,
                    "scope": self.scope,
                    "experience_version": self.current.version,
                    "document_contract": self.document_contract,
                    "context_purpose": "experience_revision",
                },
            ),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_config={"model": self.model},
            metadata={
                "response_schema": EXPERIENCE_PATCH_REVISION_SCHEMA_VERSION,
                "experience_semantics_mode": self.semantics_mode,
                "experience_revision_contract": self.revision_contract,
                "experience_document_contract": self.document_contract,
                "experience_failure_policy": self.failure_policy,
                "experience_revision_recap_version": EXPERIENCE_REVISION_RECAP_VERSION,
                "experience_prompt_token_budget": self.prompt_token_budget,
                "experience_prompt_budget_policy": (
                    "advisory_after_completed_response"
                    if self.document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION
                    else "hard_fail"
                ),
                "evidence_ids": evidence_ids,
                "trajectory_ids": trajectory_ids,
                "fact_ids": fact_ids,
                "experience_max_chars": self.max_chars,
                "experience_aux_max_chars": 240,
            },
        )
        try:
            payload = self.llm_client.generate_structured(request, dict)
            if not isinstance(payload, dict):
                raise ValueError("LLM revision is not a JSON object")
            revision = self._revision_from_structured_patch(
                recent,
                payload=payload,
                prompt_hash=prompt_hash,
                trajectory_ids=set(trajectory_ids),
                fact_ids=set(fact_ids),
                forbidden_ids=known_agent_ids | set(evidence_ids),
            )
            revision.update(revision_recap_metadata)
            if revision.get("validation_status") != "rejected":
                return revision
            budget = active_path1_budget()
            if budget is not None and not budget.repair_available:
                revision["contract_repair_exhausted"] = True
                return revision

            initial_failure = str(revision.get("failure", "unknown validation failure"))
            initial_proposed_md = str(revision.get("proposed_new_md", ""))
            initial_fingerprint = _text_fingerprint(initial_proposed_md)
            initial_proposal_summary = (
                _proposal_summary(self.current.body, initial_proposed_md)
                if initial_proposed_md
                else {}
            )
            initial_findings = list(revision.get("validation_findings", []) or [])
            initial_provider_diagnostics = (
                dict(revision.get("provider_diagnostics", {}))
                if isinstance(revision.get("provider_diagnostics"), dict)
                else {}
            )
            initial_payload_diagnostic = {
                f"initial_{key}": revision[key]
                for key in (
                    "rejected_payload_sha256",
                    "rejected_payload_chars",
                    "rejected_payload_summary",
                )
                if key in revision
            }
            repair_system_prompt = (
                "You repair the experience-patch contract. The previous response was valid "
                "JSON but failed local structural or semantic validation. Output one "
                "complete JSON object without explanation or reasoning, and do not weaken "
                "any constraint."
            )
            repair_user_prompt = (
                f"Local rejection: {initial_failure}\n"
                "Previous JSON:\n"
                f"{json.dumps(payload, ensure_ascii=False)}\n\n"
                "Regenerate strictly from the original request. If no valid actionable "
                "change exists, return operation=keep and patches=[].\n\n"
                "Original request:\n"
                f"{user_prompt}"
            )
            repair_prompt_hash = hashlib.sha256(
                (repair_system_prompt + "\n" + repair_user_prompt).encode("utf-8")
            ).hexdigest()
            repair_request = LLMRequest(
                observation=observation,
                memory_context=request.memory_context,
                system_prompt=repair_system_prompt,
                user_prompt=repair_user_prompt,
                provider_config=request.provider_config,
                metadata={
                    **request.metadata,
                    "experience_semantic_repair": True,
                    "initial_validation_failure": initial_failure,
                },
            )
            try:
                repaired_payload = self.llm_client.generate_structured(repair_request, dict)
                if not isinstance(repaired_payload, dict):
                    raise ValueError("semantic repair is not a JSON object")
                repaired_payload["_provider_repair_used"] = True
                repaired_revision = self._revision_from_structured_patch(
                    recent,
                    payload=repaired_payload,
                    prompt_hash=prompt_hash,
                    trajectory_ids=set(trajectory_ids),
                    fact_ids=set(fact_ids),
                    forbidden_ids=known_agent_ids | set(evidence_ids),
                )
            except Exception as repair_exc:  # noqa: BLE001
                if active_path1_budget() is not None and not isinstance(
                    repair_exc, Path1ResponseContractError
                ):
                    terminal = terminal_from_exception(repair_exc, stage="memory_contract_repair")
                    if terminal is repair_exc:
                        raise
                    raise terminal from repair_exc
                failed_revision = self._fail_closed_revision(
                    recent,
                    failure=(
                        "semantic_repair_failed: "
                        f"initial={initial_failure}; "
                        f"repair={type(repair_exc).__name__}: {repair_exc}"
                    ),
                    prompt_hash=prompt_hash,
                    payload=_provider_diagnostic_payload_from_exception(repair_exc),
                    forbidden_ids=known_agent_ids | set(evidence_ids),
                )
                failed_revision.update(
                    {
                        "semantic_repair_used": True,
                        "semantic_repair_prompt_sha256": repair_prompt_hash,
                        "initial_validation_failure": initial_failure,
                        "initial_proposed_new_md_sha256": initial_fingerprint["sha256"],
                        "initial_proposed_new_md_chars": initial_fingerprint["chars"],
                        "initial_proposal_summary": initial_proposal_summary,
                        "initial_validation_findings": initial_findings,
                        "initial_provider_diagnostics": initial_provider_diagnostics,
                        **revision_recap_metadata,
                        **initial_payload_diagnostic,
                    }
                )
                if active_path1_budget() is not None:
                    failed_revision.update(fallback_used=False,
                                           validation_status="response_contract_failed")
                return failed_revision
            repaired_revision["semantic_repair_used"] = True
            repaired_revision["semantic_repair_prompt_sha256"] = repair_prompt_hash
            repaired_revision["initial_validation_failure"] = initial_failure
            repaired_revision["initial_proposed_new_md_sha256"] = initial_fingerprint["sha256"]
            repaired_revision["initial_proposed_new_md_chars"] = initial_fingerprint["chars"]
            repaired_revision["initial_proposal_summary"] = initial_proposal_summary
            repaired_revision["initial_validation_findings"] = initial_findings
            repaired_revision["initial_provider_diagnostics"] = initial_provider_diagnostics
            repaired_revision.update(revision_recap_metadata)
            repaired_revision.update(initial_payload_diagnostic)
            if repaired_revision.get("validation_status") == "rejected":
                repaired_failure = str(repaired_revision.get("failure", "unknown"))
                repaired_revision["failure"] = (
                    "semantic_repair_rejected: "
                    f"initial={initial_failure}; repaired={repaired_failure}"
                )
            return repaired_revision
        except Exception as exc:  # noqa: BLE001
            if active_path1_budget() is not None and not isinstance(
                exc, Path1ResponseContractError,
            ):
                terminal = terminal_from_exception(exc, stage="memory_revision")
                if terminal is exc:
                    raise
                raise terminal from exc
            failed_revision = self._fail_closed_revision(
                recent,
                failure=f"{type(exc).__name__}: {exc}",
                prompt_hash=prompt_hash,
                payload=_provider_diagnostic_payload_from_exception(exc),
                forbidden_ids=known_agent_ids | set(evidence_ids),
            )
            failed_revision.update(revision_recap_metadata)
            if active_path1_budget() is not None:
                failed_revision.update(fallback_used=False,
                                       validation_status="response_contract_failed")
            return failed_revision

    def _revise_full_document_from_window(
        self,
        recent: list[HandTrajectory],
        evidence_records: list[object] | None = None,
        evidence_groups: dict[str, list[object]] | None = None,
    ) -> dict[str, object]:
        """Ask the LLM for the complete revised document, matching the paper code."""

        assert self.llm_client is not None
        trajectory_ids = [item.hand_id for item in recent]
        known_agent_ids = {
            player.agent_id
            for trajectory in recent
            for event in trajectory.decision_events
            for player in event.observation.players
        }
        known_agent_ids.add(self.agent_id)
        fact_payload = [
            {
                "record_id": str(getattr(item, "record_id", "")),
                "state_summary": _redact_agent_ids(
                    str(getattr(item, "state_summary", "")),
                    known_agent_ids,
                    self.agent_id,
                ),
                "action_summary": _redact_agent_ids(
                    str(getattr(item, "action_summary", "")),
                    known_agent_ids,
                    self.agent_id,
                ),
                "final_reward": int(getattr(item, "final_reward", 0)),
                "stability": float(getattr(item, "stability", 1.0)),
            }
            for item in (evidence_records or [])
        ]
        fact_ids = [str(item["record_id"]) for item in fact_payload if item["record_id"]]
        fact_payload_by_id = {
            str(item["record_id"]): item for item in fact_payload if item["record_id"]
        }
        grouped_fact_payload: dict[str, list[dict[str, object]]] = {}
        grouped_fact_ids: dict[str, list[str]] = {}
        for group_name, records in (evidence_groups or {}).items():
            group_payload: list[dict[str, object]] = []
            group_ids: list[str] = []
            for record in records:
                record_id = str(getattr(record, "record_id", ""))
                payload_item = fact_payload_by_id.get(record_id)
                if payload_item is None or record_id in group_ids:
                    continue
                group_payload.append(payload_item)
                group_ids.append(record_id)
            grouped_fact_payload[str(group_name)] = group_payload
            grouped_fact_ids[str(group_name)] = group_ids
        trajectory_payload = [self._compact_revision_recap(item) for item in recent]
        trajectory_payload_json = json.dumps(
            trajectory_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        paper_contract = self.document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION
        behavior_digest = (
            paper_revision_behavior_digest(trajectory_payload) if paper_contract else {}
        )
        behavior_digest_json = json.dumps(
            behavior_digest,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        trajectory_prompt_text = (
            render_main_style_revision_window(trajectory_payload)
            if paper_contract
            else trajectory_payload_json
        )
        revision_recap_metadata = {
            "revision_recap_version": EXPERIENCE_REVISION_RECAP_VERSION,
            "revision_recap_sha256": hashlib.sha256(
                trajectory_payload_json.encode("utf-8")
            ).hexdigest(),
            "revision_recap_chars": len(trajectory_payload_json),
            "behavior_digest_version": (
                EXPERIENCE_BEHAVIOR_DIGEST_VERSION if paper_contract else ""
            ),
            "behavior_digest_sha256": hashlib.sha256(
                behavior_digest_json.encode("utf-8")
            ).hexdigest(),
            "behavior_digest_chars": len(behavior_digest_json),
        }
        require_fact_classification = evidence_groups is not None
        response_fields = [
            "keep: boolean",
            "new_md: string",
            "calibration_note: string",
            "self_check: string",
            "supporting_fact_ids: string[]",
        ]
        if require_fact_classification:
            response_fields.extend(
                [
                    "contradicting_fact_ids: string[]",
                    "noise_fact_ids: string[]",
                ]
            )
        if paper_contract:
            system_template = EXPERIENCE_PAPER_REVISION_SYSTEM_TEMPLATE
        elif self.document_contract == EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION:
            system_template = EXPERIENCE_TRANSFERABLE_REVISION_SYSTEM_TEMPLATE
        else:
            system_template = EXPERIENCE_REVISION_SYSTEM_TEMPLATE
        system_prompt = system_template.format(max_chars=self.max_chars)
        fact_prompt_label = (
            "Relevant facts grouped by Async retrieval path (an empty group means no match):"
            if evidence_groups is not None
            else "Relevant facts for this request (an empty array means no fact section):"
        )
        fact_prompt_payload: object = (
            grouped_fact_payload if evidence_groups is not None else fact_payload
        )
        calibration_target_block = (
            "The only current hand for calibration is the final recap above. Compare only its "
            "recorded intent, committed action, and net reward. Do not cite earlier hand numbers "
            "or use a single result to prove or refute a strategy."
            if paper_contract
            else ""
        )
        if (
            require_fact_classification
            and self.document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION
        ):
            # The paper Async contract is self-contained: unlike the legacy Async
            # addendum, it already defines the complete seven-field response.  Do
            # not prepend the Direct/Sync five-field contract.
            update_instruction = EXPERIENCE_PAPER_ASYNC_FACT_CLASSIFICATION_PROMPT
        else:
            update_instruction = (
                EXPERIENCE_PAPER_UPDATE_PROMPT
                if self.document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION
                else EXPERIENCE_UPDATE_PROMPT
            )
            if paper_contract:
                update_instruction = (
                    f"{update_instruction}\n\n"
                    f"{EXPERIENCE_PAPER_OBJECTIVE_COMPATIBILITY_ADDENDUM}\n\n"
                    f"{EXPERIENCE_PAPER_STRATEGY_REVIEW_CARD}"
                )
            if require_fact_classification:
                if self.document_contract == EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION:
                    classification_prompt = EXPERIENCE_ASYNC_TRANSFERABLE_FACT_CLASSIFICATION_PROMPT
                else:
                    classification_prompt = EXPERIENCE_ASYNC_FACT_CLASSIFICATION_PROMPT
                update_instruction = f"{update_instruction}\n\n{classification_prompt}"
        quality_block = ""
        if self.document_contract == EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION:
            update_instruction = f"{update_instruction}\n\n{EXPERIENCE_TRANSFERABLE_QUALITY_PROMPT}"
            quality_summary_json = json.dumps(
                _experience_quality_summary(self.current.body),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            quality_block = (
                "\n\nDeterministic quality summary of the current document "
                "(for merging and correction only):\n"
                f"{quality_summary_json}"
            )
        fact_prompt_json = json.dumps(
            fact_prompt_payload, ensure_ascii=False, separators=(",", ":")
        )
        behavior_digest_block = (
            "Observed behavior-coverage summary (for detecting copying only; behavior does not "
            f"prove optimality):\n{behavior_digest_json}"
        )
        if paper_contract and require_fact_classification:
            trajectory_block = (
                f"(1) Hands since the previous sweep ({len(trajectory_payload)} recaps):\n"
                f"{trajectory_prompt_text}"
            )
            fact_block = f"(2) Facts retrieved from those recaps:\n{fact_prompt_json}"
            current_block = f"(3) Current experience:\n{self.current.body}"
        elif paper_contract:
            current_block = f"Current experience:\n{self.current.body}"
            trajectory_block = (
                f"Recent recaps ({len(trajectory_payload)} hands, including the current hand):\n"
                f"{trajectory_prompt_text}"
            )
            fact_block = f"Relevant historical facts:\n{fact_prompt_json}"
        else:
            current_block = (
                f"Current experience (version {self.current.version}):\n"
                f"{self.current.body}{quality_block}"
            )
            trajectory_block = (
                f"Recent recaps ({len(trajectory_payload)} hands including the current hand; "
                "the complete structured audit is stored in trajectory_log.jsonl):\n"
                f"{trajectory_prompt_text}"
            )
            fact_block = f"{fact_prompt_label}\n{fact_prompt_json}"
        if require_fact_classification:
            # Match the paper Async sweep: recent trajectories -> recalled
            # facts -> current experience -> revision instruction.
            material_blocks = [trajectory_block, fact_block, current_block]
        else:
            material_blocks = [current_block, trajectory_block]
            if fact_payload:
                material_blocks.append(fact_block)
        if not paper_contract:
            material_blocks.extend([behavior_digest_block, calibration_target_block])
            prior_review_feedback = _prior_semantic_review_feedback_block(self.revision_log)
            if prior_review_feedback:
                material_blocks.append(prior_review_feedback)
        user_prompt = "\n\n".join(material_blocks) + f"\n\n{update_instruction}"
        if not paper_contract:
            user_prompt += (
                "\n\nJSON field contract (types only, not a keep/revise example):\n- "
                + "\n- ".join(response_fields)
            )
        prompt_hash = hashlib.sha256(
            (system_prompt + "\n" + user_prompt).encode("utf-8")
        ).hexdigest()
        observation = next(
            (
                event.observation
                for trajectory in reversed(recent)
                for event in reversed(trajectory.decision_events)
            ),
            None,
        )
        if observation is None:
            failed_revision = self._fail_closed_revision(
                recent,
                failure="no_decision_observation_for_llm_revision",
                prompt_hash=prompt_hash,
            )
            failed_revision.update(revision_recap_metadata)
            return failed_revision
        request = LLMRequest(
            observation=observation,
            memory_context=MemoryContext(
                experience=self.current,
                metadata={
                    "mechanism": self.name,
                    "scope": self.scope,
                    "experience_version": self.current.version,
                    "document_contract": self.document_contract,
                    "context_purpose": "experience_revision",
                },
            ),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_config={"model": self.model},
            metadata={
                "response_schema": EXPERIENCE_FULL_DOCUMENT_REVISION_SCHEMA_VERSION,
                "experience_semantics_mode": self.semantics_mode,
                "experience_revision_contract": self.revision_contract,
                "experience_document_contract": self.document_contract,
                "experience_failure_policy": self.failure_policy,
                "experience_revision_recap_version": EXPERIENCE_REVISION_RECAP_VERSION,
                "experience_revision_recap_render_version": (
                    EXPERIENCE_PAPER_RECAP_RENDER_VERSION if paper_contract else "json_v1"
                ),
                "experience_prompt_token_budget": self.prompt_token_budget,
                "experience_prompt_budget_policy": (
                    "advisory_after_completed_response"
                    if self.document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION
                    else "hard_fail"
                ),
                "evidence_ids": [*trajectory_ids, *fact_ids],
                "trajectory_ids": trajectory_ids,
                "fact_ids": fact_ids,
                "fact_evidence_groups": grouped_fact_ids,
                "experience_fact_classification_fields": require_fact_classification,
                "experience_max_chars": self.max_chars,
                "experience_aux_max_chars": (
                    self.max_chars
                    if self.document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION
                    else 240
                ),
            },
        )
        forbidden_ids = known_agent_ids | set(trajectory_ids) | set(fact_ids)
        try:
            payload = self.llm_client.generate_structured(request, dict)
            if not isinstance(payload, dict):
                raise ValueError("LLM revision is not a JSON object")
            revision = self._revision_from_full_document(
                recent,
                payload=payload,
                prompt_hash=prompt_hash,
                fact_ids=set(fact_ids),
                forbidden_ids=forbidden_ids,
                require_fact_classification=require_fact_classification,
            )
            revision.update(revision_recap_metadata)
            # Objective document errors are first returned to the proposer as a
            # bounded correction task.  The independent reviewer is a semantic
            # gate, not a substitute renderer for malformed Markdown.  Sending
            # deterministic findings straight to the reviewer made every host
            # finding mandatory as an issue, so even a correctable candidate
            # could never be accepted and the existing repair path below was
            # unreachable whenever review was enabled.
            if (
                self.semantic_review_mode == "independent_model_v1"
                and revision.get("validation_status") != "rejected"
            ):
                return self._review_full_document_revision(
                    recent,
                    proposal_payload=payload,
                    proposal_revision=revision,
                    proposer_prompt_hash=prompt_hash,
                    observation=observation,
                    memory_context=request.memory_context,
                    trajectory_payload=trajectory_payload,
                    behavior_digest=behavior_digest,
                    fact_payload=fact_payload,
                    fact_groups=grouped_fact_ids,
                    fact_ids=set(fact_ids),
                    forbidden_ids=forbidden_ids,
                    require_fact_classification=require_fact_classification,
                )
            if revision.get("validation_status") != "rejected":
                return revision

            initial_failure = str(revision.get("failure", "unknown validation failure"))
            initial_proposed_md = str(revision.get("proposed_new_md", ""))
            initial_fingerprint = _text_fingerprint(initial_proposed_md)
            initial_proposal_summary = (
                _proposal_summary(self.current.body, initial_proposed_md)
                if initial_proposed_md
                else {}
            )
            initial_findings = list(revision.get("validation_findings", []) or [])
            initial_provider_diagnostics = (
                dict(revision.get("provider_diagnostics", {}))
                if isinstance(revision.get("provider_diagnostics"), dict)
                else {}
            )
            repair_fields = [
                "keep",
                "new_md",
                "calibration_note",
                "self_check",
                "supporting_fact_ids",
            ]
            if require_fact_classification:
                repair_fields.extend(["contradicting_fact_ids", "noise_fact_ids"])
            repair_payload = {
                key: value for key, value in payload.items() if not str(key).startswith("_provider")
            }
            repair_quality_instruction = (
                f"\n\n{EXPERIENCE_TRANSFERABLE_QUALITY_PROMPT}"
                if self.document_contract == EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION
                else ""
            )
            repair_recap_payload = (
                trajectory_payload[-5:] if _semantic_repair_needs_recap(initial_findings) else []
            )
            repair_evidence_blocks: list[str] = []
            if repair_recap_payload:
                repair_evidence_blocks.append(
                    "Recent compact recaps for correcting position, hand-class, cost, "
                    "or action-chain "
                    "errors only; do not add a rule absent from the previous candidate:\n"
                    + json.dumps(
                        repair_recap_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            if fact_payload:
                repair_evidence_blocks.append(fact_block)
            if paper_contract:
                repair_evidence_blocks.append(behavior_digest_block)
            repair_evidence_suffix = (
                "\n\n" + "\n\n".join(repair_evidence_blocks) if repair_evidence_blocks else ""
            )
            repair_evidence_metadata = {
                "semantic_repair_recap_hand_count": len(repair_recap_payload),
                "semantic_repair_fact_count": len(fact_payload),
            }
            repair_system_prompt = (
                "You repair the complete experience-document JSON contract. The previous response "
                "was valid JSON but failed local structural or semantic validation. Output one "
                "complete JSON object, not a patch, with no explanation or reasoning trace. "
                "Delete, "
                "merge, or rewrite only rules in the previous candidate; add no new strategy claim."
            )
            repair_user_prompt = (
                f"Local rejection: {initial_failure}\n"
                "Structured rejection findings (excerpts sanitized):\n"
                f"{json.dumps(initial_findings, ensure_ascii=False)}\n\n"
                "Previous JSON:\n"
                f"{json.dumps(repair_payload, ensure_ascii=False)}\n\n"
                f"Output exactly these fields: {json.dumps(repair_fields, ensure_ascii=False)}. "
                'If no update is justified, return keep=true and new_md="". Otherwise return '
                "keep=false and the complete five-section new_md. Every non-empty bullet needs an "
                "observable condition and explicit poker action. Merge duplicate findings, remove "
                "dealt-card examples, and correct terminology, position, hand class, and action "
                "chains. Do not preserve a rejected bullet unchanged. Retain unaffected valid "
                "rules. "
                "Do not replay raw trajectories or derive a new rule from one result. "
                f"new_md must not exceed {self.max_chars} characters."
                f"{repair_quality_instruction}{repair_evidence_suffix}"
            )
            if self.document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION:
                repair_system_prompt = (
                    "You repair facts, structure, and consolidation in the complete paper-style "
                    "experience document. Correct only reported structure errors, identity leaks, "
                    "explicit poker contradictions, or prohibited mature-document growth. When the "
                    "growth gate fires, merge or remove redundant rules and add no strategy. "
                    "Preserve "
                    "the rest of the candidate and all five sections. Output one JSON object only."
                )
                repair_user_prompt = (
                    f"Local hard rejection: {initial_failure}\n"
                    "Structured hard-rejection findings (excerpts sanitized):\n"
                    f"{json.dumps(initial_findings, ensure_ascii=False)}\n\n"
                    "Previous JSON:\n"
                    f"{json.dumps(repair_payload, ensure_ascii=False)}\n\n"
                    "Output exactly these fields: "
                    f"{json.dumps(repair_fields, ensure_ascii=False)}. "
                    'If no update is justified, return keep=true and new_md=""; otherwise return '
                    "keep=false and the complete fixed five-section new_md. Correct each finding "
                    "without rewriting unaffected rules. Include no player, hand, fact ID, seat, "
                    "or "
                    "hidden information. For paper_merge_before_add_violation, neither total rules "
                    "nor rules in the triggering section may grow. The repaired document is "
                    "limited "
                    "to 64 rules, 16 per section, and two representative starting hands per rule; "
                    "otherwise consolidate to a range or hand class."
                    f"\n{EXPERIENCE_PAPER_STRATEGY_REVIEW_CARD}"
                    f"new_md must not exceed {self.max_chars} characters."
                    f"{repair_evidence_suffix}"
                )
            repair_prompt_hash = hashlib.sha256(
                (repair_system_prompt + "\n" + repair_user_prompt).encode("utf-8")
            ).hexdigest()
            repair_request = LLMRequest(
                observation=observation,
                memory_context=request.memory_context,
                system_prompt=repair_system_prompt,
                user_prompt=repair_user_prompt,
                provider_config=request.provider_config,
                metadata={
                    **request.metadata,
                    "experience_semantic_repair": True,
                    "initial_validation_failure": initial_failure,
                },
            )
            try:
                repaired_payload = self.llm_client.generate_structured(repair_request, dict)
                if not isinstance(repaired_payload, dict):
                    raise ValueError("semantic repair is not a JSON object")
                repaired_payload["_provider_repair_used"] = True
                repaired_revision = self._revision_from_full_document(
                    recent,
                    payload=repaired_payload,
                    prompt_hash=prompt_hash,
                    fact_ids=set(fact_ids),
                    forbidden_ids=forbidden_ids,
                    require_fact_classification=require_fact_classification,
                )
            except Exception as repair_exc:  # noqa: BLE001
                failed_revision = self._fail_closed_revision(
                    recent,
                    failure=(
                        "semantic_repair_failed: "
                        f"initial={initial_failure}; "
                        f"repair={type(repair_exc).__name__}: {repair_exc}"
                    ),
                    prompt_hash=prompt_hash,
                    payload=_provider_diagnostic_payload_from_exception(repair_exc),
                )
                failed_revision.update(
                    {
                        "semantic_repair_used": True,
                        "semantic_repair_prompt_sha256": repair_prompt_hash,
                        "initial_validation_failure": initial_failure,
                        "initial_proposed_new_md_sha256": initial_fingerprint["sha256"],
                        "initial_proposed_new_md_chars": initial_fingerprint["chars"],
                        "initial_proposal_summary": initial_proposal_summary,
                        "initial_validation_findings": initial_findings,
                        "initial_provider_diagnostics": initial_provider_diagnostics,
                        **revision_recap_metadata,
                        **repair_evidence_metadata,
                    }
                )
                return failed_revision
            repaired_revision["semantic_repair_used"] = True
            repaired_revision["semantic_repair_prompt_sha256"] = repair_prompt_hash
            repaired_revision["initial_validation_failure"] = initial_failure
            repaired_revision["initial_proposed_new_md_sha256"] = initial_fingerprint["sha256"]
            repaired_revision["initial_proposed_new_md_chars"] = initial_fingerprint["chars"]
            repaired_revision["initial_proposal_summary"] = initial_proposal_summary
            repaired_revision["initial_validation_findings"] = initial_findings
            repaired_revision["initial_provider_diagnostics"] = initial_provider_diagnostics
            repaired_revision.update(revision_recap_metadata)
            repaired_revision.update(repair_evidence_metadata)
            if repaired_revision.get("validation_status") == "rejected":
                repaired_failure = str(repaired_revision.get("failure", "unknown"))
                repaired_revision["failure"] = (
                    "semantic_repair_rejected: "
                    f"initial={initial_failure}; repaired={repaired_failure}"
                )
                return repaired_revision
            if self.semantic_review_mode == "independent_model_v1":
                reviewed_revision = self._review_full_document_revision(
                    recent,
                    proposal_payload=repaired_payload,
                    proposal_revision=repaired_revision,
                    proposer_prompt_hash=prompt_hash,
                    observation=observation,
                    memory_context=request.memory_context,
                    trajectory_payload=trajectory_payload,
                    behavior_digest=behavior_digest,
                    fact_payload=fact_payload,
                    fact_groups=grouped_fact_ids,
                    fact_ids=set(fact_ids),
                    forbidden_ids=forbidden_ids,
                    require_fact_classification=require_fact_classification,
                )
                reviewed_revision.update(
                    {
                        "semantic_repair_used": True,
                        "semantic_repair_prompt_sha256": repair_prompt_hash,
                        "initial_validation_failure": initial_failure,
                        "initial_proposed_new_md_sha256": initial_fingerprint["sha256"],
                        "initial_proposed_new_md_chars": initial_fingerprint["chars"],
                        "initial_proposal_summary": initial_proposal_summary,
                        "initial_validation_findings": initial_findings,
                        "initial_provider_diagnostics": initial_provider_diagnostics,
                        **revision_recap_metadata,
                        **repair_evidence_metadata,
                    }
                )
                return reviewed_revision
            return repaired_revision
        except Exception as exc:  # noqa: BLE001
            failed_revision = self._fail_closed_revision(
                recent,
                failure=f"{type(exc).__name__}: {exc}",
                prompt_hash=prompt_hash,
                payload=_provider_diagnostic_payload_from_exception(exc),
            )
            failed_revision.update(revision_recap_metadata)
            return failed_revision

    def _review_full_document_revision(
        self,
        recent: list[HandTrajectory],
        *,
        proposal_payload: dict[str, object],
        proposal_revision: dict[str, object],
        proposer_prompt_hash: str,
        observation: AgentObservation,
        memory_context: MemoryContext,
        trajectory_payload: list[dict[str, object]],
        behavior_digest: dict[str, object],
        fact_payload: list[dict[str, object]],
        fact_groups: dict[str, list[str]],
        fact_ids: set[str],
        forbidden_ids: set[str],
        require_fact_classification: bool,
    ) -> dict[str, object]:
        """Independently review one immutable proposal and replay local gates.

        The proposer still follows the original paper's full-document update.
        This boundary never asks either model to rewrite a rejected proposal:
        the independent model may accept, reject, or request at most two
        allowlisted mechanical edits, which are applied and revalidated by code.
        """

        if (
            self.semantic_review_mode != "independent_model_v1"
            or self.semantic_reviewer_client is None
        ):
            return proposal_revision
        candidate_md = str(proposal_revision.get("proposed_new_md", ""))
        proposal_status = str(proposal_revision.get("validation_status", ""))
        proposal_failure = str(proposal_revision.get("failure") or "")
        reviewable_status = proposal_status == "accepted" or (
            proposal_status == "rejected"
            and proposal_failure.startswith("invalid_full_experience_document:")
        )
        if not candidate_md.strip() or not reviewable_status:
            proposal_revision["semantic_review_used"] = False
            proposal_revision["semantic_review_status"] = "not_reviewable"
            return proposal_revision

        host_findings = _semantic_review_host_findings(
            proposal_revision=proposal_revision,
            candidate_md=candidate_md,
            behavior_digest=behavior_digest,
            forbidden_ids=forbidden_ids,
        )
        mandatory_finding_ids = [str(item["finding_id"]) for item in host_findings]
        evidence_bundle = build_semantic_review_evidence_view(
            trajectory_recaps=trajectory_payload,
            behavior_digest=behavior_digest,
            fact_evidence=fact_payload,
            fact_groups=fact_groups,
            deterministic_findings=host_findings,
            calibration_note=str(proposal_revision.get("calibration_note", "")),
            current_experience={
                "version": self.current.version,
                "sha256": hashlib.sha256(self.current.body.encode("utf-8")).hexdigest(),
                "chars": len(self.current.body),
            },
        )
        evidence_window = dict(evidence_bundle["trajectory_window"])
        evidence_hand_count_included = int(evidence_window["included_hand_count"])
        evidence_hand_count_total = int(evidence_window["total_hand_count"])
        bounded_trajectory_payload = list(evidence_bundle["trajectory_recaps"])
        allowed_evidence_refs = _semantic_review_allowed_evidence_refs(
            trajectory_payload=bounded_trajectory_payload,
            fact_payload=fact_payload,
            candidate_md=candidate_md,
            host_findings=host_findings,
        )
        try:
            review_context = bind_semantic_review_context(
                candidate_md=candidate_md,
                calibration_note=str(proposal_revision.get("calibration_note", "")),
                evidence_bundle=evidence_bundle,
                proposer_model_family=self.semantic_proposer_model_family,
                reviewer_model_family=self.semantic_reviewer_model_family,
                mandatory_finding_ids=mandatory_finding_ids,
                allowed_evidence_refs=allowed_evidence_refs,
            )
        except (SemanticReviewContractError, TypeError, ValueError) as exc:
            return _semantic_review_operational_failure(
                proposal_revision,
                current_body=self.current.body,
                failure=f"semantic_review_binding_failed: {type(exc).__name__}",
                reviewer_model=self.semantic_reviewer_model,
            )

        try:
            request_material = build_semantic_review_request_material(
                context=review_context,
                evidence_bundle=evidence_bundle,
            )
        except (SemanticReviewContractError, TypeError, ValueError) as exc:
            return _semantic_review_operational_failure(
                proposal_revision,
                current_body=self.current.body,
                failure=f"semantic_review_request_build_failed: {type(exc).__name__}",
                reviewer_model=self.semantic_reviewer_model,
            )
        reviewer_system_prompt = request_material.system_prompt
        reviewer_user_prompt = request_material.user_prompt
        reviewer_prompt_hash = hashlib.sha256(
            (reviewer_system_prompt + "\n" + reviewer_user_prompt).encode("utf-8")
        ).hexdigest()
        review_metadata = dict(request_material.metadata)
        review_metadata.update(
            {
                "experience_semantic_review": True,
                "experience_semantic_review_prompt_version": (
                    EXPERIENCE_SEMANTIC_REVIEW_PROMPT_VERSION
                ),
            }
        )
        review_request = LLMRequest(
            observation=observation,
            memory_context=memory_context,
            system_prompt=reviewer_system_prompt,
            user_prompt=reviewer_user_prompt,
            provider_config={"model": self.semantic_reviewer_model},
            metadata=review_metadata,
        )
        reviewer_diagnostics: dict[str, object] = {}
        try:
            raw_review_payload = self.semantic_reviewer_client.generate_structured(
                review_request,
                dict,
            )
            if not isinstance(raw_review_payload, dict):
                raise SemanticReviewContractError(
                    "semantic reviewer response must be a JSON object"
                )
            if isinstance(raw_review_payload.get("_provider_diagnostics"), dict):
                reviewer_diagnostics = dict(raw_review_payload["_provider_diagnostics"])
            review_payload = {
                key: value
                for key, value in raw_review_payload.items()
                if not str(key).startswith("_provider")
            }
            review = validate_semantic_review_payload(
                review_payload,
                context=review_context,
            )
            replay = replay_semantic_review(
                review,
                context=review_context,
                deterministic_validator=lambda document: _semantic_review_replay_findings(
                    current_body=self.current.body,
                    proposed_body=document,
                    max_chars=self.max_chars,
                    document_contract=self.document_contract,
                    forbidden_ids=forbidden_ids,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            failed = _semantic_review_operational_failure(
                proposal_revision,
                current_body=self.current.body,
                failure=(f"semantic_review_failed: {type(exc).__name__}"),
                reviewer_model=self.semantic_reviewer_model,
            )
            failed.update(
                {
                    "semantic_review_failure_code": (semantic_review_contract_failure_code(exc)),
                    "semantic_review_prompt_version": (EXPERIENCE_SEMANTIC_REVIEW_PROMPT_VERSION),
                    "semantic_review_prompt_sha256": reviewer_prompt_hash,
                    "semantic_review_candidate_sha256": (review_context.candidate_sha256),
                    "semantic_review_evidence_bundle_sha256": (
                        review_context.evidence_bundle_sha256
                    ),
                    "semantic_review_evidence_hand_count_included": (evidence_hand_count_included),
                    "semantic_review_evidence_hand_count_total": (evidence_hand_count_total),
                    "semantic_review_reference_card_sha256": (review_context.reference_card_sha256),
                    "semantic_review_mandatory_finding_ids": (mandatory_finding_ids),
                    "semantic_review_host_findings": host_findings,
                    "semantic_review_provider_diagnostics": reviewer_diagnostics,
                }
            )
            return failed

        issues = [_compact_semantic_issue(issue) for issue in review.issues]
        semantic_metadata = {
            "semantic_review_used": True,
            "semantic_review_mode": self.semantic_review_mode,
            "semantic_review_status": replay.status,
            "semantic_review_verdict": review.verdict,
            "semantic_review_prompt_version": (EXPERIENCE_SEMANTIC_REVIEW_PROMPT_VERSION),
            "semantic_review_prompt_sha256": reviewer_prompt_hash,
            "semantic_review_candidate_sha256": review_context.candidate_sha256,
            "semantic_review_evidence_bundle_sha256": (review_context.evidence_bundle_sha256),
            "semantic_review_evidence_hand_count_included": (evidence_hand_count_included),
            "semantic_review_evidence_hand_count_total": (evidence_hand_count_total),
            "semantic_review_reference_card_sha256": (review_context.reference_card_sha256),
            "semantic_review_proposer_model": self.model,
            "semantic_review_reviewer_model": self.semantic_reviewer_model,
            "semantic_review_proposer_family": review_context.proposer_model_family,
            "semantic_review_reviewer_family": review_context.reviewer_model_family,
            "semantic_review_independence_passed": review.independence_passed,
            "semantic_review_mandatory_finding_ids": mandatory_finding_ids,
            "semantic_review_acknowledged_finding_ids": list(review.acknowledged_finding_ids),
            "semantic_review_host_findings": host_findings,
            "semantic_review_issue_codes": [issue.code for issue in review.issues],
            "semantic_review_issues": issues,
            "semantic_review_applied_repair_count": replay.applied_repair_count,
            "semantic_review_replay_findings": _compact_validation_findings(
                list(replay.validation_findings),
                forbidden_ids=forbidden_ids,
            ),
            "semantic_review_provider_diagnostics": reviewer_diagnostics,
        }
        if not replay.installable:
            rejected = dict(proposal_revision)
            rejected.update(semantic_metadata)
            rejected.update(
                {
                    "keep": True,
                    "new_md": self.current.body,
                    "validation_status": "rejected",
                    "fallback_used": False,
                    "failure": (
                        "semantic_review_rejected: "
                        + (",".join(item["code"] for item in issues) if issues else replay.status)
                    ),
                    "revision_strategy": ("llm_full_document_independent_review_rejected"),
                }
            )
            return rejected

        reviewed_payload = {
            key: value
            for key, value in proposal_payload.items()
            if not str(key).startswith("_provider")
        }
        reviewed_payload.update(
            {
                "keep": False,
                "new_md": replay.document_md,
                "calibration_note": replay.calibration_note,
            }
        )
        if isinstance(proposal_payload.get("_provider_diagnostics"), dict):
            reviewed_payload["_provider_diagnostics"] = dict(
                proposal_payload["_provider_diagnostics"]
            )
        reviewed_revision = self._revision_from_full_document(
            recent,
            payload=reviewed_payload,
            prompt_hash=proposer_prompt_hash,
            fact_ids=fact_ids,
            forbidden_ids=forbidden_ids,
            require_fact_classification=require_fact_classification,
        )
        reviewed_revision.update(semantic_metadata)
        reviewed_revision["revision_strategy"] = "llm_full_document_independent_review"
        if proposal_status == "rejected":
            reviewed_revision.update(
                {
                    "initial_validation_failure": proposal_failure,
                    "initial_validation_findings": list(
                        proposal_revision.get("validation_findings", []) or []
                    ),
                    "initial_proposed_new_md_sha256": review_context.candidate_sha256,
                    "initial_proposed_new_md_chars": len(candidate_md),
                    "initial_proposal_summary": proposal_revision.get("proposal_summary", {}),
                }
            )
        if reviewed_revision.get("validation_status") != "accepted":
            reviewed_revision["keep"] = True
            reviewed_revision["new_md"] = self.current.body
            reviewed_revision["validation_status"] = "rejected"
            reviewed_revision["failure"] = "semantic_review_replay_rejected: " + str(
                reviewed_revision.get("failure") or "deterministic validation"
            )
        return reviewed_revision

    def _revision_from_full_document(
        self,
        recent: list[HandTrajectory],
        *,
        payload: dict[str, object],
        prompt_hash: str,
        fact_ids: set[str],
        forbidden_ids: set[str],
        require_fact_classification: bool = False,
    ) -> dict[str, object]:
        """Validate the complete-document JSON before allowing whole-body replacement."""

        paper_contract = self.document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION
        if paper_contract:
            return self._revision_from_main_full_document(
                recent,
                payload=payload,
                prompt_hash=prompt_hash,
                require_fact_classification=require_fact_classification,
            )
        prompt_token_counts = _provider_prompt_token_counts(payload)
        over_budget = next(
            (value for value in prompt_token_counts if value > self.prompt_token_budget),
            None,
        )
        if over_budget is not None and not paper_contract:
            return self._fail_closed_revision(
                recent,
                failure=(
                    "experience_prompt_token_budget_exceeded: "
                    f"{over_budget} > {self.prompt_token_budget}"
                ),
                prompt_hash=prompt_hash,
                payload=payload,
            )
        required_fields = {
            "keep",
            "new_md",
            "calibration_note",
            "self_check",
            "supporting_fact_ids",
        }
        if require_fact_classification:
            required_fields.update({"contradicting_fact_ids", "noise_fact_ids"})
        defaulted_response_fields: list[str] = []
        if paper_contract:
            # The original paper code treats absent optional metadata as empty
            # and a missing keep flag as keep=True.  Preserve that proven
            # no-op behavior while retaining stricter parsing for the optional
            # transferable-rulebook contract.
            paper_defaults: dict[str, object] = {
                "keep": True,
                "new_md": "",
                "calibration_note": "",
                "self_check": "",
                "supporting_fact_ids": [],
            }
            if require_fact_classification:
                paper_defaults.update({"contradicting_fact_ids": [], "noise_fact_ids": []})
            defaulted_response_fields = sorted(
                field for field in paper_defaults if field not in payload
            )
            payload = {**paper_defaults, **payload}
        raw_candidate = payload.get("new_md")
        candidate_for_diagnostics = (
            _normalize_markdown(raw_candidate)
            if isinstance(raw_candidate, str) and raw_candidate.strip()
            else ""
        )
        provider_fields = {"_provider_diagnostics", "_provider_repair_used"}
        response_fields = set(payload) - provider_fields
        ignored_response_fields = sorted(response_fields - required_fields)
        if not paper_contract and response_fields != required_fields:
            return self._fail_closed_revision(
                recent,
                failure=(
                    "schema_contract_violation: full_document_v1 must contain "
                    f"exactly the {len(required_fields)} revision fields"
                ),
                prompt_hash=prompt_hash,
                proposed_md=candidate_for_diagnostics,
                payload=payload,
                allowed_ids=fact_ids,
                fallback_used=False,
                validation_status="rejected",
                forbidden_ids=forbidden_ids,
            )
        if type(payload.get("keep")) is not bool:
            return self._fail_closed_revision(
                recent,
                failure="schema_contract_violation: keep must be a boolean",
                prompt_hash=prompt_hash,
                proposed_md=candidate_for_diagnostics,
                payload=payload,
                allowed_ids=fact_ids,
                fallback_used=False,
                validation_status="rejected",
                forbidden_ids=forbidden_ids,
            )
        if not isinstance(payload.get("new_md"), str):
            return self._fail_closed_revision(
                recent,
                failure="schema_contract_violation: new_md must be a string",
                prompt_hash=prompt_hash,
                proposed_md=candidate_for_diagnostics,
                payload=payload,
                allowed_ids=fact_ids,
                fallback_used=False,
                validation_status="rejected",
                forbidden_ids=forbidden_ids,
            )
        try:
            auxiliary_fields = _validated_revision_auxiliary_fields(
                payload,
                forbidden_ids=forbidden_ids,
                document_contract=self.document_contract,
            )
        except ValueError as exc:
            return self._fail_closed_revision(
                recent,
                failure=str(exc),
                prompt_hash=prompt_hash,
                proposed_md=candidate_for_diagnostics,
                payload=payload,
                allowed_ids=fact_ids,
                fallback_used=False,
                validation_status="rejected",
                forbidden_ids=forbidden_ids,
            )
        sanitized_auxiliary_fields = [
            field
            for field in ("calibration_note", "self_check")
            if auxiliary_fields[field] != payload.get(field)
        ]
        ignored_unknown_fact_id_count = 0
        reclassified_trajectory_ids: list[str] = []
        try:
            classification_fields = ["supporting_fact_ids"]
            if require_fact_classification:
                classification_fields.extend(["contradicting_fact_ids", "noise_fact_ids"])
            if paper_contract and not require_fact_classification:
                raw_supporting = payload.get("supporting_fact_ids")
                if not isinstance(raw_supporting, list) or any(
                    not isinstance(value, str) for value in raw_supporting
                ):
                    raise ValueError("supporting_fact_ids must be an array of string fact IDs")
                distinct_values = list(dict.fromkeys(str(value) for value in raw_supporting))
                trajectory_ids = {item.hand_id for item in recent}
                classified = {
                    "supporting_fact_ids": [value for value in distinct_values if value in fact_ids]
                }
                reclassified_trajectory_ids = [
                    value for value in distinct_values if value in trajectory_ids
                ]
                ignored_unknown_fact_id_count = sum(
                    value not in fact_ids and value not in trajectory_ids
                    for value in distinct_values
                )
            else:
                classified = {
                    field: _validated_full_document_fact_ids(payload.get(field), fact_ids, field)
                    for field in classification_fields
                }
            category_sets = [set(values) for values in classified.values()]
            if any(
                left & right
                for index, left in enumerate(category_sets)
                for right in category_sets[index + 1 :]
            ):
                raise ValueError("fact classifications must be pairwise disjoint")
        except ValueError as exc:
            return self._fail_closed_revision(
                recent,
                failure=f"invalid_fact_assessments: {exc}",
                prompt_hash=prompt_hash,
                proposed_md=candidate_for_diagnostics,
                payload=payload,
                allowed_ids=fact_ids,
                fallback_used=False,
                validation_status="rejected",
                forbidden_ids=forbidden_ids,
            )
        if (
            require_fact_classification
            and self.document_contract == EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION
            and len(classified["supporting_fact_ids"]) == 1
        ):
            return self._fail_closed_revision(
                recent,
                failure=(
                    "invalid_fact_assessments: transferable Async supporting evidence "
                    "requires at least two facts"
                ),
                prompt_hash=prompt_hash,
                proposed_md=candidate_for_diagnostics,
                payload=payload,
                allowed_ids=fact_ids,
                fallback_used=False,
                validation_status="rejected",
                forbidden_ids=forbidden_ids,
            )
        calibration_findings = (
            _paper_calibration_findings(
                auxiliary_fields["calibration_note"],
                current_reward=(recent[-1].final_reward if recent else None),
            )
            if paper_contract and not bool(payload["keep"])
            else []
        )
        calibration_note_neutralized = bool(calibration_findings)
        if calibration_note_neutralized:
            # calibration_note is audit metadata, not the persistent strategy
            # document.  A dubious auxiliary claim must not discard an
            # otherwise sound full-document revision; replace it with a fixed
            # evidence-safe note and retain the complete diagnostic findings.
            auxiliary_fields["calibration_note"] = EXPERIENCE_NEUTRALIZED_CALIBRATION_NOTE
            if "calibration_note" not in sanitized_auxiliary_fields:
                sanitized_auxiliary_fields.append("calibration_note")
        common = {
            "rev": len(self.revision_log) + 1,
            "hand_index": recent[-1].hand_id if recent else "",
            "window_hand_count": len(recent),
            "old_md": self.current.body,
            "calibration_note": auxiliary_fields["calibration_note"],
            "calibration_note_neutralized": calibration_note_neutralized,
            "calibration_validation_findings": _compact_validation_findings(
                calibration_findings, forbidden_ids=forbidden_ids
            ),
            "self_check": auxiliary_fields["self_check"],
            "supporting_fact_ids": classified["supporting_fact_ids"],
            "contradicting_fact_ids": classified.get("contradicting_fact_ids", []),
            "noise_fact_ids": classified.get("noise_fact_ids", []),
            "irrelevant_fact_ids": [],
            "fact_assessments": [
                {"fact_id": fact_id, "label": label}
                for field, label in (
                    ("supporting_fact_ids", "supporting"),
                    ("contradicting_fact_ids", "contradicting"),
                    ("noise_fact_ids", "noise"),
                )
                for fact_id in classified.get(field, [])
            ],
            "supporting_trajectory_ids": reclassified_trajectory_ids,
            "revision_strategy": "llm_full_document_revision",
            "schema_version": EXPERIENCE_FULL_DOCUMENT_REVISION_SCHEMA_VERSION,
            "prompt_version": self._active_prompt_version(),
            "revision_recap_version": EXPERIENCE_REVISION_RECAP_VERSION,
            "prompt_sha256": prompt_hash,
            "model": self.model,
            "document_contract": self.document_contract,
            "fallback_used": False,
            "repair_used": bool(payload.get("_provider_repair_used", False)),
            "sanitized_auxiliary_fields": sanitized_auxiliary_fields,
            "ignored_unknown_fact_id_count": ignored_unknown_fact_id_count,
            "reclassified_trajectory_id_count": len(reclassified_trajectory_ids),
            "prompt_token_budget_advisory": (
                {
                    "observed_prompt_tokens": over_budget,
                    "configured_budget": self.prompt_token_budget,
                }
                if over_budget is not None
                else {}
            ),
            "failure": None,
            "contract_violation": False,
            "provider_diagnostics": (
                dict(payload.get("_provider_diagnostics", {}))
                if isinstance(payload.get("_provider_diagnostics"), dict)
                else {}
            ),
            "defaulted_response_fields": defaulted_response_fields,
            "ignored_response_fields": ignored_response_fields,
            "applied_patches": [],
        }
        raw_new_md = str(payload["new_md"])
        proposed_md = _normalize_markdown(raw_new_md) if raw_new_md.strip() else ""
        if bool(payload["keep"]):
            if proposed_md and proposed_md != self.current.body:
                if paper_contract:
                    # Main treats keep as authoritative and never turns a
                    # contradictory new_md into an update.  Retain only a
                    # body-free diagnostic fingerprint in revision_log.
                    return {
                        **common,
                        "keep": True,
                        "new_md": self.current.body,
                        "proposed_new_md": proposed_md,
                        "ignored_keep_new_md": True,
                        "keep_contract_violation": False,
                        "validation_status": "kept_by_model",
                        "rule_catalog": _current_rule_catalog(self.current),
                    }
                revision = self._fail_closed_revision(
                    recent,
                    failure="keep_true_requires_empty_or_current_new_md",
                    prompt_hash=prompt_hash,
                    proposed_md=proposed_md,
                    payload=payload,
                    allowed_ids=fact_ids,
                    fallback_used=False,
                    validation_status="rejected",
                    forbidden_ids=forbidden_ids,
                )
                revision["keep_contract_violation"] = True
                revision["contract_violation"] = True
                return revision
            return {
                **common,
                "keep": True,
                "new_md": self.current.body,
                "proposed_new_md": proposed_md,
                "ignored_keep_new_md": False,
                "keep_contract_violation": False,
                "validation_status": "kept_by_model",
                "rule_catalog": _current_rule_catalog(self.current),
            }
        if not proposed_md:
            return {
                **common,
                "keep": True,
                "new_md": self.current.body,
                "proposed_new_md": "",
                "keep_contract_violation": False,
                "validation_status": "empty_noop",
                "rule_catalog": _current_rule_catalog(self.current),
            }
        if proposed_md == self.current.body:
            return {
                **common,
                "keep": True,
                "new_md": self.current.body,
                "proposed_new_md": proposed_md,
                "keep_contract_violation": False,
                "validation_status": "unchanged",
                "rule_catalog": _current_rule_catalog(self.current),
            }
        advisory_findings = experience_document_advisory_findings(
            proposed_md,
            document_contract=self.document_contract,
        )
        validation_findings = [
            *validate_experience_document_findings(
                proposed_md,
                max_chars=self.max_chars,
                document_contract=self.document_contract,
            ),
        ]
        if paper_contract:
            validation_findings.extend(
                _paper_merge_before_add_findings(self.current.body, proposed_md)
            )
        if _contains_forbidden_identifier(proposed_md, forbidden_ids) and not any(
            str(item.get("code", "")) == "concrete_identifier" for item in validation_findings
        ):
            validation_findings.append(
                {
                    "code": "concrete_identifier",
                    "message": "document contains a concrete identifier",
                    "section": "",
                    "bullet_index": None,
                    "excerpt": "",
                }
            )
        if validation_findings:
            validation_errors = [_format_validation_finding(item) for item in validation_findings]
            return self._fail_closed_revision(
                recent,
                failure="invalid_full_experience_document: " + "; ".join(validation_errors),
                prompt_hash=prompt_hash,
                proposed_md=proposed_md,
                payload=payload,
                allowed_ids=fact_ids,
                fallback_used=False,
                validation_status="rejected",
                validation_findings=validation_findings,
                forbidden_ids=forbidden_ids,
            )
        return {
            **common,
            "keep": False,
            "new_md": proposed_md,
            "proposed_new_md": proposed_md,
            "keep_contract_violation": False,
            "validation_status": "accepted",
            "advisory_findings": _compact_validation_findings(
                advisory_findings,
                forbidden_ids=forbidden_ids,
            ),
            "advisory_finding_count": len(advisory_findings),
            "rule_catalog": _current_rule_catalog(
                ExperienceDocument(
                    version=self.current.version + 1,
                    body=proposed_md,
                    source_hand_ids=[],
                    updated_at=utc_now_iso(),
                    scope=self.scope,
                )
            ),
        }

    def _revision_from_main_full_document(
        self,
        recent: list[HandTrajectory],
        *,
        payload: dict[str, object],
        prompt_hash: str,
        require_fact_classification: bool,
    ) -> dict[str, object]:
        """Reproduce main's safe-json, keep, and whole-document replacement semantics."""

        def _ids(name: str) -> list[object]:
            value = payload.get(name) or []
            return list(value) if isinstance(value, (list, tuple)) else []

        raw_new_md = payload.get("new_md")
        proposed_md = raw_new_md.strip() if isinstance(raw_new_md, str) else ""
        keep = bool(payload.get("keep", True))
        if keep or not proposed_md or proposed_md == self.current.body.strip():
            keep = True
            new_md = self.current.body
            validation_status = "kept_by_model" if payload.get("keep", True) else "empty_noop"
        else:
            new_md = proposed_md
            validation_status = "accepted"
        supporting = _ids("supporting_fact_ids")
        contradicting = _ids("contradicting_fact_ids") if require_fact_classification else []
        noise = _ids("noise_fact_ids") if require_fact_classification else []
        return {
            "rev": len(self.revision_log) + 1,
            "hand_index": recent[-1].hand_id if recent else "",
            "window_hand_count": len(recent),
            "old_md": self.current.body,
            "new_md": new_md,
            "proposed_new_md": proposed_md,
            "keep": keep,
            "calibration_note": str(payload.get("calibration_note", "")),
            "self_check": str(payload.get("self_check", "")),
            "supporting_fact_ids": supporting,
            "contradicting_fact_ids": contradicting,
            "noise_fact_ids": noise,
            "irrelevant_fact_ids": [],
            "fact_assessments": [],
            "supporting_trajectory_ids": [],
            "revision_strategy": "llm_full_document_revision",
            "schema_version": EXPERIENCE_FULL_DOCUMENT_REVISION_SCHEMA_VERSION,
            "prompt_version": self._active_prompt_version(),
            "revision_recap_version": EXPERIENCE_REVISION_RECAP_VERSION,
            "prompt_sha256": prompt_hash,
            "model": self.model,
            "document_contract": self.document_contract,
            "fallback_used": False,
            "repair_used": False,
            "validation_status": validation_status,
            "failure": None,
            "contract_violation": False,
            "provider_diagnostics": (
                dict(payload.get("_provider_diagnostics", {}))
                if isinstance(payload.get("_provider_diagnostics"), dict)
                else {}
            ),
            "applied_patches": [],
        }

    def _revision_from_structured_patch(
        self,
        recent: list[HandTrajectory],
        *,
        payload: dict[str, object],
        prompt_hash: str,
        trajectory_ids: set[str],
        fact_ids: set[str],
        forbidden_ids: set[str],
    ) -> dict[str, object]:
        """Validate typed patches, apply them to stable rules, then render Markdown."""

        prompt_token_counts = _provider_prompt_token_counts(payload)
        over_budget = next(
            (value for value in prompt_token_counts if value > self.prompt_token_budget),
            None,
        )
        if over_budget is not None:
            return self._fail_closed_revision(
                recent,
                failure=(
                    "experience_prompt_token_budget_exceeded: "
                    f"{over_budget} > {self.prompt_token_budget}"
                ),
                prompt_hash=prompt_hash,
                payload=payload,
                forbidden_ids=forbidden_ids,
            )
        operation = payload.get("operation")
        if operation not in {"keep", "patch"}:
            unexpected_new_md = str(payload.get("new_md", ""))
            revision = self._fail_closed_revision(
                recent,
                failure="schema_contract_violation: expected operation=keep|patch",
                prompt_hash=prompt_hash,
                payload=payload,
                allowed_ids=fact_ids,
                fallback_used=False,
                validation_status="rejected",
                forbidden_ids=forbidden_ids,
            )
            revision["keep_contract_violation"] = bool(
                payload.get("keep") is True
                and unexpected_new_md.strip()
                and _normalize_markdown(unexpected_new_md) != self.current.body
            )
            return revision
        patches = payload.get("patches")
        if not isinstance(patches, list):
            return self._fail_closed_revision(
                recent,
                failure="schema_contract_violation: patches must be an array",
                prompt_hash=prompt_hash,
                payload=payload,
                allowed_ids=fact_ids,
                fallback_used=False,
                validation_status="rejected",
                forbidden_ids=forbidden_ids,
            )
        try:
            auxiliary_fields = _validated_revision_auxiliary_fields(
                payload,
                forbidden_ids=forbidden_ids,
                document_contract=self.document_contract,
            )
        except ValueError as exc:
            return self._fail_closed_revision(
                recent,
                failure=str(exc),
                prompt_hash=prompt_hash,
                payload=payload,
                allowed_ids=fact_ids,
                fallback_used=False,
                validation_status="rejected",
                forbidden_ids=forbidden_ids,
            )
        try:
            assessments = _parse_fact_assessments(payload, fact_ids)
        except ValueError as exc:
            return self._fail_closed_revision(
                recent,
                failure=f"invalid_fact_assessments: {exc}",
                prompt_hash=prompt_hash,
                payload=payload,
                allowed_ids=fact_ids,
                fallback_used=False,
                validation_status="rejected",
                forbidden_ids=forbidden_ids,
            )
        classified = {
            label: [item["fact_id"] for item in assessments if item["label"] == label]
            for label in ("supporting", "contradicting", "noise", "irrelevant")
        }
        common = {
            "rev": len(self.revision_log) + 1,
            "hand_index": recent[-1].hand_id if recent else "",
            "window_hand_count": len(recent),
            "old_md": self.current.body,
            "calibration_note": auxiliary_fields["calibration_note"],
            "self_check": auxiliary_fields["self_check"],
            "supporting_fact_ids": classified["supporting"],
            "contradicting_fact_ids": classified["contradicting"],
            "noise_fact_ids": classified["noise"],
            "irrelevant_fact_ids": classified["irrelevant"],
            "fact_assessments": assessments,
            "revision_strategy": "llm_structured_patch_revision",
            "schema_version": EXPERIENCE_PATCH_REVISION_SCHEMA_VERSION,
            "prompt_version": EXPERIENCE_PATCH_PROMPT_VERSION,
            "prompt_sha256": prompt_hash,
            "model": self.model,
            "document_contract": self.document_contract,
            "fallback_used": False,
            "repair_used": bool(payload.get("_provider_repair_used", False)),
            "failure": None,
            "contract_violation": False,
            "provider_diagnostics": (
                dict(payload.get("_provider_diagnostics", {}))
                if isinstance(payload.get("_provider_diagnostics"), dict)
                else {}
            ),
        }
        if operation == "keep":
            if patches:
                revision = self._fail_closed_revision(
                    recent,
                    failure="keep_operation_requires_empty_patches",
                    prompt_hash=prompt_hash,
                    payload=payload,
                    allowed_ids=fact_ids,
                    fallback_used=False,
                    validation_status="rejected",
                    forbidden_ids=forbidden_ids,
                )
                revision["keep_contract_violation"] = True
                revision["contract_violation"] = True
                return revision
            return {
                **common,
                "keep": True,
                "new_md": self.current.body,
                "proposed_new_md": "",
                "keep_contract_violation": False,
                "validation_status": "kept_by_model",
                "supporting_trajectory_ids": [],
                "applied_patches": [],
                "rule_catalog": _current_rule_catalog(self.current),
            }
        if not patches:
            return self._fail_closed_revision(
                recent,
                failure="patch_operation_requires_nonempty_patches",
                prompt_hash=prompt_hash,
                payload=payload,
                allowed_ids=fact_ids,
                fallback_used=False,
                validation_status="rejected",
                forbidden_ids=forbidden_ids,
            )
        try:
            rules, applied_patches, supporting_trajectory_ids = _apply_rule_patches(
                _current_rule_catalog(self.current),
                patches,
                trajectory_ids=trajectory_ids,
                fact_ids=fact_ids,
                fact_assessments=assessments,
                forbidden_ids=forbidden_ids,
            )
        except ValueError as exc:
            return self._fail_closed_revision(
                recent,
                failure=f"invalid_experience_patch: {exc}",
                prompt_hash=prompt_hash,
                payload=payload,
                allowed_ids=fact_ids,
                fallback_used=False,
                validation_status="rejected",
                forbidden_ids=forbidden_ids,
            )
        new_md = _render_rule_catalog(rules)
        validation_findings = validate_experience_document_findings(
            new_md,
            max_chars=self.max_chars,
            document_contract=self.document_contract,
        )
        if validation_findings:
            validation_errors = [_format_validation_finding(item) for item in validation_findings]
            return self._fail_closed_revision(
                recent,
                failure="invalid_experience_patch: " + "; ".join(validation_errors),
                prompt_hash=prompt_hash,
                proposed_md=new_md,
                payload=payload,
                allowed_ids=fact_ids,
                fallback_used=False,
                validation_status="rejected",
                validation_findings=validation_findings,
                forbidden_ids=forbidden_ids,
            )
        keep = new_md == self.current.body
        return {
            **common,
            "keep": keep,
            "new_md": new_md,
            "proposed_new_md": "",
            "keep_contract_violation": False,
            "contract_violation": False,
            "validation_status": "unchanged" if keep else "accepted",
            "supporting_trajectory_ids": supporting_trajectory_ids,
            "applied_patches": applied_patches,
            "rule_catalog": rules,
        }

    def _active_revision_schema_version(self) -> str:
        """Return the response schema selected by the configured revision contract."""

        if self.revision_contract == "full_document_v1":
            return EXPERIENCE_FULL_DOCUMENT_REVISION_SCHEMA_VERSION
        return EXPERIENCE_PATCH_REVISION_SCHEMA_VERSION

    def _active_prompt_version(self) -> str:
        """Return the prompt version selected by the configured revision contract."""

        if self.revision_contract == "full_document_v1":
            if self.document_contract == EXPERIENCE_LEGACY_DOCUMENT_CONTRACT_VERSION:
                return EXPERIENCE_LEGACY_FULL_DOCUMENT_PROMPT_VERSION
            if self.document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION:
                return EXPERIENCE_PAPER_FULL_DOCUMENT_PROMPT_VERSION
            return EXPERIENCE_FULL_DOCUMENT_PROMPT_VERSION
        return EXPERIENCE_PATCH_PROMPT_VERSION

    def _fail_closed_revision(
        self,
        recent: list[HandTrajectory],
        *,
        failure: str,
        prompt_hash: str,
        proposed_md: str = "",
        payload: dict[str, object] | None = None,
        allowed_ids: set[str] | None = None,
        fallback_used: bool = True,
        validation_status: str = "failed_closed",
        validation_findings: list[dict[str, object]] | None = None,
        forbidden_ids: set[str] | None = None,
    ) -> dict[str, object]:
        """Return an auditable no-op revision without replacing learned experience."""

        source = payload or {}
        valid_ids = allowed_ids or set()
        diagnostic_forbidden_ids = forbidden_ids or set()
        raw_calibration_note = str(source.get("calibration_note", ""))
        raw_self_check = str(source.get("self_check", ""))
        calibration_fingerprint = _text_fingerprint(raw_calibration_note)
        self_check_fingerprint = _text_fingerprint(raw_self_check)
        contract_violation = any(
            marker in failure
            for marker in (
                "schema_contract_violation",
                "keep_operation_requires_empty_patches",
                "patch_operation_requires_nonempty_patches",
                "keep_true_requires_empty_or_current_new_md",
            )
        )
        result = {
            "rev": len(self.revision_log) + 1,
            "hand_index": recent[-1].hand_id if recent else "",
            "window_hand_count": len(recent),
            "keep": True,
            "old_md": self.current.body,
            "new_md": self.current.body,
            "proposed_new_md": proposed_md,
            "keep_contract_violation": False,
            "contract_violation": contract_violation,
            "validation_status": validation_status,
            "calibration_note": _sanitize_diagnostic_excerpt(
                raw_calibration_note, diagnostic_forbidden_ids, max_chars=120
            ),
            "calibration_note_sha256": calibration_fingerprint["sha256"],
            "calibration_note_chars": calibration_fingerprint["chars"],
            "self_check": _sanitize_diagnostic_excerpt(
                raw_self_check, diagnostic_forbidden_ids, max_chars=120
            ),
            "self_check_sha256": self_check_fingerprint["sha256"],
            "self_check_chars": self_check_fingerprint["chars"],
            "supporting_fact_ids": _filter_evidence_ids(source, "supporting_fact_ids", valid_ids),
            "contradicting_fact_ids": _filter_evidence_ids(
                source, "contradicting_fact_ids", valid_ids
            ),
            "noise_fact_ids": _filter_evidence_ids(source, "noise_fact_ids", valid_ids),
            "irrelevant_fact_ids": _filter_evidence_ids(source, "irrelevant_fact_ids", valid_ids),
            "supporting_trajectory_ids": [],
            "fact_assessments": [],
            "applied_patches": [],
            "rule_catalog": _current_rule_catalog(self.current),
            "revision_strategy": "llm_fail_closed",
            "schema_version": self._active_revision_schema_version(),
            "prompt_version": self._active_prompt_version(),
            "revision_recap_version": EXPERIENCE_REVISION_RECAP_VERSION,
            "prompt_sha256": prompt_hash,
            "model": self.model,
            "fallback_used": fallback_used,
            "repair_used": bool(source.get("_provider_repair_used", False)),
            "provider_diagnostics": (
                dict(source.get("_provider_diagnostics", {}))
                if isinstance(source.get("_provider_diagnostics"), dict)
                else {}
            ),
            "failure": failure,
            "requested_strategy": "llm",
            "document_contract": self.document_contract,
        }
        if proposed_md:
            result["proposal_summary"] = _proposal_summary(self.current.body, proposed_md)
        elif payload is not None and self.revision_contract == "structured_patch_v2":
            result.update(
                _structured_patch_payload_diagnostic(
                    payload,
                    forbidden_ids=diagnostic_forbidden_ids,
                )
            )
        if validation_findings:
            result["validation_findings"] = _compact_validation_findings(
                validation_findings,
                forbidden_ids=diagnostic_forbidden_ids,
            )
        return result

    def _trajectory_payload(self, trajectory: HandTrajectory) -> dict[str, object]:
        """Render all decision-relevant visible evidence without opponent hidden cards."""

        decisions: list[dict[str, object]] = []
        for event in trajectory.decision_events:
            observation = event.observation
            fallback_used = bool(event.llm_metadata.get("fallback_used"))
            active_players = [player for player in observation.players if not player.busted]
            labels = {
                player.agent_id: (
                    "self" if player.agent_id == self.agent_id else f"opponent_seat_{player.seat}"
                )
                for player in active_players
            }
            self_state = next(
                (player for player in active_players if player.agent_id == self.agent_id),
                None,
            )
            public_actions = [
                _sanitize_public_action(action, labels)
                for action in observation.action_history[-16:]
            ]
            decision_facts = build_decision_facts(observation)
            for field in (
                "effective_stack_by_opponent",
                "effective_stack_after_call_by_opponent",
                "effective_stack_by_opponent_chips",
                "effective_stack_after_call_by_opponent_chips",
                "effective_stack_by_opponent_big_blinds",
                "effective_stack_after_call_by_opponent_big_blinds",
            ):
                per_opponent = decision_facts.get(field)
                if isinstance(per_opponent, dict):
                    decision_facts[field] = {
                        labels.get(str(agent_id), "opponent_unknown"): amount
                        for agent_id, amount in per_opponent.items()
                    }
            big_blind_amount = decision_facts.get("big_blind_amount")
            big_blind_amount = (
                big_blind_amount if type(big_blind_amount) is int and big_blind_amount > 0 else None
            )
            small_blind_amount = decision_facts.get("small_blind_amount")
            small_blind_amount = (
                small_blind_amount
                if type(small_blind_amount) is int and small_blind_amount > 0
                else None
            )
            stack_chips = self_state.stack if self_state is not None else None
            decision_units: dict[str, object] = {
                "chip_unit": "chips",
                "small_blind_amount": small_blind_amount,
                "big_blind_amount": big_blind_amount,
                "pot_chips": observation.pot,
                "current_bet_chips": observation.current_bet,
                "to_call_chips": observation.to_call,
                "stack_chips": stack_chips,
            }
            if big_blind_amount is not None:
                decision_units.update(
                    {
                        "pot_big_blinds": chips_to_big_blinds(observation.pot, big_blind_amount),
                        "current_bet_big_blinds": chips_to_big_blinds(
                            observation.current_bet, big_blind_amount
                        ),
                        "to_call_big_blinds": chips_to_big_blinds(
                            observation.to_call, big_blind_amount
                        ),
                        "stack_big_blinds": (
                            chips_to_big_blinds(stack_chips, big_blind_amount)
                            if type(stack_chips) is int
                            else None
                        ),
                    }
                )
            public_players: list[dict[str, object]] = []
            for player in active_players:
                public_player: dict[str, object] = {
                    "label": labels[player.agent_id],
                    "stack": player.stack,
                    "street_bet": player.current_bet,
                    "total_committed": player.total_committed,
                    "stack_chips": player.stack,
                    "street_bet_chips": player.current_bet,
                    "total_committed_chips": player.total_committed,
                    "folded": player.folded,
                    "all_in": player.all_in,
                }
                if big_blind_amount is not None:
                    public_player.update(
                        {
                            "stack_big_blinds": chips_to_big_blinds(player.stack, big_blind_amount),
                            "street_bet_big_blinds": chips_to_big_blinds(
                                player.current_bet, big_blind_amount
                            ),
                            "total_committed_big_blinds": chips_to_big_blinds(
                                player.total_committed, big_blind_amount
                            ),
                        }
                    )
                public_players.append(public_player)
            committed_action = event.committed_action.to_dict()
            committed_amount = committed_action.get("amount")
            if type(committed_amount) is int:
                committed_action["amount_chips"] = committed_amount
                if big_blind_amount is not None:
                    committed_action["amount_big_blinds"] = chips_to_big_blinds(
                        committed_amount, big_blind_amount
                    )
            raw_action: dict[str, object] = {
                "action_type": event.decision.action_type,
                "amount": event.decision.amount,
            }
            if type(event.decision.amount) is int:
                raw_action["amount_chips"] = event.decision.amount
                if big_blind_amount is not None:
                    raw_action["amount_big_blinds"] = chips_to_big_blinds(
                        event.decision.amount, big_blind_amount
                    )
            decisions.append(
                {
                    "phase": observation.phase,
                    "position": _position_label(observation),
                    "hole_cards": list(observation.hole_cards),
                    "board": list(observation.community_cards),
                    "pot": observation.pot,
                    "current_bet": observation.current_bet,
                    "to_call": observation.to_call,
                    "stack": stack_chips,
                    **decision_units,
                    "public_players": public_players,
                    "recent_public_actions": public_actions,
                    "legal_actions": [
                        action.to_dict() for action in observation.legal_actions.actions
                    ],
                    "committed_action": committed_action,
                    "raw_action": raw_action,
                    "action_guard": {
                        "repaired": bool(event.llm_metadata.get("guard_repaired")),
                        "fallback_used": fallback_used,
                    },
                    "declared_intent_not_ground_truth": _redact_agent_ids(
                        "" if fallback_used else event.decision.intent[:40],
                        set(labels),
                        self.agent_id,
                    ),
                    "declared_intent_source": "raw_decision.intent",
                    "declared_intent_status": (
                        "invalidated_by_action_guard_fallback"
                        if fallback_used
                        else "model_self_report_not_ground_truth"
                    ),
                    "reason_summary_not_ground_truth": _redact_agent_ids(
                        event.decision.reason_summary[:200],
                        set(labels),
                        self.agent_id,
                    ),
                    "deterministic_decision_facts": decision_facts,
                }
            )
        public_labels: dict[str, str] = {self.agent_id: "self"}
        if trajectory.decision_events:
            for player in trajectory.decision_events[-1].observation.players:
                if player.busted:
                    continue
                public_labels[player.agent_id] = (
                    "self" if player.agent_id == self.agent_id else f"opponent_seat_{player.seat}"
                )
        opponent_index = 0
        final_labels: dict[str, str] = {}
        for agent_id in sorted(trajectory.showdown_visible_cards):
            if agent_id == self.agent_id:
                final_labels[agent_id] = "self"
            else:
                opponent_index += 1
                final_labels[agent_id] = public_labels.get(
                    agent_id, f"shown_opponent_{opponent_index}"
                )
        final_observation = (
            trajectory.decision_events[-1].observation if trajectory.decision_events else None
        )
        final_small_blind_amount = (
            final_observation.small_blind_amount
            if final_observation is not None
            and type(final_observation.small_blind_amount) is int
            and final_observation.small_blind_amount > 0
            else None
        )
        final_big_blind_amount = (
            final_observation.big_blind_amount
            if final_observation is not None
            and type(final_observation.big_blind_amount) is int
            and final_observation.big_blind_amount > 0
            else None
        )
        outcome: dict[str, object] = {
            "chip_unit": "chips",
            "small_blind_amount": final_small_blind_amount,
            "big_blind_amount": final_big_blind_amount,
            "won_amount": trajectory.won_amount,
            "won_amount_chips": trajectory.won_amount,
            "invested_chips": (
                trajectory.won_amount - trajectory.final_reward
                if type(trajectory.won_amount) is int
                else None
            ),
            "won_amount_status": (
                "observed_from_hand_result_winners"
                if trajectory.won_amount is not None
                else "unavailable_legacy_trajectory"
            ),
            "final_reward": trajectory.final_reward,
            "final_reward_chips": trajectory.final_reward,
            "final_stack": trajectory.final_stack,
            "final_stack_chips": trajectory.final_stack,
            "final_public_board": list(trajectory.final_community_cards),
            "showdown_visible_cards": {
                final_labels[agent_id]: list(cards)
                for agent_id, cards in trajectory.showdown_visible_cards.items()
            },
        }
        if final_big_blind_amount is not None:
            outcome.update(
                {
                    "won_amount_big_blinds": (
                        chips_to_big_blinds(trajectory.won_amount, final_big_blind_amount)
                        if type(trajectory.won_amount) is int
                        else None
                    ),
                    "invested_big_blinds": (
                        chips_to_big_blinds(
                            trajectory.won_amount - trajectory.final_reward,
                            final_big_blind_amount,
                        )
                        if type(trajectory.won_amount) is int
                        else None
                    ),
                    "final_reward_big_blinds": chips_to_big_blinds(
                        trajectory.final_reward, final_big_blind_amount
                    ),
                    "final_stack_big_blinds": chips_to_big_blinds(
                        trajectory.final_stack, final_big_blind_amount
                    ),
                }
            )
        return {
            "hand_id": trajectory.hand_id,
            "chip_unit": "chips",
            "small_blind_amount": final_small_blind_amount,
            "big_blind_amount": final_big_blind_amount,
            "decisions": decisions,
            "public_action_sequence": [
                _sanitize_public_action(action, public_labels)
                for action in trajectory.public_actions
            ],
            "outcome": outcome,
            "evidence_boundary": (
                "Only self hole cards, public actions, public player state, final public "
                "board, final reward, and showdown-visible cards are included."
            ),
        }

    def _compact_revision_recap(self, trajectory: HandTrajectory) -> dict[str, object]:
        """Render the paper-style intent/action/outcome recap used only by revision LLMs.

        The complete visible audit payload remains in ``trajectory_log.jsonl``.  This
        representation removes repeated player/action/legal-state blocks while retaining
        the objective facts required to reject an incorrect self-reported intent.
        """

        full = self._trajectory_payload(trajectory)
        return compact_revision_recap_payload(full, trajectory_id=trajectory.hand_id)

    def _deterministic_revision(self, recent: list[HandTrajectory]) -> dict[str, object]:
        """Create an auditable fixture patch without inferring optimality from reward."""

        existing_rules = _current_rule_catalog(self.current)
        occupied_sections = {str(rule.get("section")) for rule in existing_rules}
        phase_sections = {
            EXPERIENCE_SECTION_LABELS[0]: {"preflop"},
            EXPERIENCE_SECTION_LABELS[1]: {"flop"},
            EXPERIENCE_SECTION_LABELS[2]: {"turn", "river"},
        }
        candidates = [
            (trajectory, event) for trajectory in recent for event in trajectory.decision_events
        ]
        action_labels = {
            "fold": "fold",
            "check": "check",
            "call": "call",
            "raise": "raise",
        }
        phase_labels = {
            "preflop": "preflop",
            "flop": "flop",
            "turn": "turn",
            "river": "river",
        }
        patches: list[dict[str, object]] = []

        def append_patch(
            section: str,
            trajectory: HandTrajectory,
            event: DecisionEvent,
            *,
            include_position: bool = False,
        ) -> None:
            observation = event.observation
            facts = build_decision_facts(observation)
            action = action_labels.get(
                event.committed_action.action_type,
                event.committed_action.action_type,
            )
            phase = phase_labels.get(observation.phase, observation.phase)
            risk = str(facts.get("call", {}).get("risk_label", "unknown"))
            when = [
                f"street is {phase} and made-hand class is "
                f"{facts.get('made_hand_class', 'unknown')}",
                f"call-cost risk label is {risk}",
            ]
            position = _position_label(observation)
            if include_position and position != "unknown":
                when.append(f"relative position is {position}")
            patches.append(
                {
                    "section": section,
                    "op": "add",
                    "rule_id": "",
                    "when": when,
                    "guidance": (
                        f"treat {action} as a tentative candidate; if raising, use only "
                        "the legal range"
                    ),
                    "invalidators": [
                        "made-hand or draw structure changes",
                        "call cost, effective stack, or prior action line changes materially",
                    ],
                    "basis": "observed_evidence",
                    "status": "tentative",
                    "supporting_trajectory_ids": [trajectory.hand_id],
                    "supporting_fact_ids": [],
                }
            )

        for section, phases in phase_sections.items():
            if section in occupied_sections:
                continue
            matching = [item for item in candidates if item[1].observation.phase in phases]
            if matching:
                append_patch(section, *matching[-1])
        sizing_section = EXPERIENCE_SECTION_LABELS[4]
        if sizing_section not in occupied_sections and candidates:
            append_patch(sizing_section, *candidates[-1], include_position=True)

        payload = {
            "operation": "patch" if patches else "keep",
            "patches": patches,
            "fact_assessments": [],
            "calibration_note": (
                "The deterministic fixture stores the observed action only as a tentative "
                "candidate."
            ),
            "self_check": (
                "No return statistic, identity, or evidence ID was added; archetype rules "
                "require type evidence."
            ),
        }
        revision = self._revision_from_structured_patch(
            recent,
            payload=payload,
            prompt_hash="",
            trajectory_ids={item.hand_id for item in recent},
            fact_ids=set(),
            forbidden_ids={self.agent_id, *(item.hand_id for item in recent)},
        )
        revision.update(
            {
                "revision_strategy": "deterministic_structured_patch_fixture",
                "prompt_version": None,
                "prompt_sha256": None,
                "model": "",
            }
        )
        return revision


def _current_rule_catalog(document: ExperienceDocument) -> list[dict[str, object]]:
    """Load structured rules, or assign stable IDs to legacy Markdown bullets."""

    stored = document.metadata.get("rule_catalog", [])
    if isinstance(stored, list) and stored and all(isinstance(item, dict) for item in stored):
        return json.loads(json.dumps(stored, ensure_ascii=False))
    rules: list[dict[str, object]] = []
    section = ""
    for raw_line in document.body.splitlines():
        line = raw_line.strip()
        if line in EXPERIENCE_REQUIRED_HEADINGS:
            section = line.removeprefix("## ")
            continue
        if section and line.startswith("- ") and line[2:].strip():
            rendered_text = line[2:].strip()
            digest = hashlib.sha256(f"{section}\n{rendered_text}".encode()).hexdigest()[:12]
            rules.append(
                {
                    "rule_id": f"legacy-{digest}",
                    "section": section,
                    "rendered_text": rendered_text,
                    "basis": "legacy_document",
                    "status": "legacy_untyped",
                }
            )
    return rules


def _parse_fact_assessments(
    payload: dict[str, object], allowed_fact_ids: set[str]
) -> list[dict[str, str]]:
    """Require exactly one typed assessment for every supplied fact record."""

    raw_assessments = payload.get("fact_assessments")
    if not isinstance(raw_assessments, list):
        raise ValueError("fact_assessments must be an array")
    assessments: list[dict[str, str]] = []
    seen: set[str] = set()
    allowed_labels = {"supporting", "contradicting", "noise", "irrelevant"}
    for raw in raw_assessments:
        if not isinstance(raw, dict) or set(raw) != {"fact_id", "label"}:
            raise ValueError("each assessment must contain only fact_id and label")
        fact_id = str(raw.get("fact_id", ""))
        label = str(raw.get("label", ""))
        if fact_id not in allowed_fact_ids:
            raise ValueError(f"unknown fact_id {fact_id!r}")
        if fact_id in seen:
            raise ValueError(f"duplicate fact_id {fact_id!r}")
        if label not in allowed_labels:
            raise ValueError(f"unknown fact assessment label {label!r}")
        seen.add(fact_id)
        assessments.append({"fact_id": fact_id, "label": label})
    if seen != allowed_fact_ids:
        missing = sorted(allowed_fact_ids - seen)
        raise ValueError(f"missing fact assessments: {missing}")
    return assessments


def _apply_rule_patches(
    current_rules: list[dict[str, object]],
    raw_patches: list[object],
    *,
    trajectory_ids: set[str],
    fact_ids: set[str],
    fact_assessments: list[dict[str, str]],
    forbidden_ids: set[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    """Apply typed operations atomically; any invalid operation rejects the whole revision."""

    rules = json.loads(json.dumps(current_rules, ensure_ascii=False))
    applied: list[dict[str, object]] = []
    supporting_trajectories: set[str] = set()
    assessment_by_id = {item["fact_id"]: item["label"] for item in fact_assessments}
    required_fields = {
        "section",
        "op",
        "rule_id",
        "when",
        "guidance",
        "invalidators",
        "basis",
        "status",
        "supporting_trajectory_ids",
        "supporting_fact_ids",
    }
    for raw_patch in raw_patches:
        if not isinstance(raw_patch, dict) or set(raw_patch) != required_fields:
            raise ValueError("each patch must contain exactly the structured patch fields")
        section = str(raw_patch["section"])
        operation = str(raw_patch["op"])
        rule_id = str(raw_patch["rule_id"])
        if section not in EXPERIENCE_SECTION_LABELS:
            raise ValueError(f"unknown section {section!r}")
        if operation not in {"add", "revise", "delete"}:
            raise ValueError(f"unknown patch op {operation!r}")
        matching_indexes = [
            index for index, rule in enumerate(rules) if str(rule.get("rule_id")) == rule_id
        ]
        if operation == "delete":
            if len(matching_indexes) != 1:
                raise ValueError(f"delete references unknown rule_id {rule_id!r}")
            if str(rules[matching_indexes[0]].get("section")) != section:
                raise ValueError("delete section does not match existing rule")
            if any(
                raw_patch[field]
                for field in (
                    "when",
                    "guidance",
                    "invalidators",
                    "supporting_trajectory_ids",
                    "supporting_fact_ids",
                )
            ):
                raise ValueError("delete patch must not smuggle replacement content")
            rules.pop(matching_indexes[0])
            applied.append({"section": section, "op": operation, "rule_id": rule_id})
            continue
        if operation == "add" and rule_id:
            raise ValueError("add patch rule_id must be empty")
        if operation == "revise":
            if len(matching_indexes) != 1:
                raise ValueError(f"revise references unknown rule_id {rule_id!r}")
            if str(rules[matching_indexes[0]].get("section")) != section:
                raise ValueError("revise section does not match existing rule")
        when = _validated_string_list(raw_patch["when"], "when", minimum=1)
        invalidators = _validated_string_list(raw_patch["invalidators"], "invalidators", minimum=1)
        guidance = str(raw_patch["guidance"]).strip()
        if not guidance or len(guidance) > 240 or not _contains_action(guidance):
            raise ValueError("guidance must contain a concise poker action")
        basis = str(raw_patch["basis"])
        status = str(raw_patch["status"])
        if basis not in {"observed_evidence", "model_prior"}:
            raise ValueError(f"unknown rule basis {basis!r}")
        if status not in {"tentative", "supported"}:
            raise ValueError(f"unknown rule status {status!r}")
        supporting_trajectory_ids = _validated_id_list(
            raw_patch["supporting_trajectory_ids"], trajectory_ids, "trajectory"
        )
        supporting_fact_ids = _validated_id_list(raw_patch["supporting_fact_ids"], fact_ids, "fact")
        if any(assessment_by_id.get(fact_id) != "supporting" for fact_id in supporting_fact_ids):
            raise ValueError("patch supporting facts must be assessed as supporting")
        if basis == "model_prior":
            if status != "tentative" or supporting_trajectory_ids or supporting_fact_ids:
                raise ValueError("model_prior rules must be tentative and evidence-free")
        else:
            evidence_count = len(set(supporting_trajectory_ids) | set(supporting_fact_ids))
            if evidence_count < 1:
                raise ValueError("observed_evidence rule lacks supporting evidence")
            if status == "supported" and evidence_count < 2:
                raise ValueError("supported rule needs at least two evidence IDs")
        text_fields = [guidance, *when, *invalidators]
        if any(_contains_forbidden_identifier(text, forbidden_ids) for text in text_fields):
            raise ValueError("rule text contains a player, hand, fact, or seat identifier")
        canonical = json.dumps(
            {
                "section": section,
                "when": when,
                "guidance": guidance,
                "invalidators": invalidators,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        generated_id = f"rule-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:12]}"
        new_rule = {
            "rule_id": generated_id if operation == "add" else rule_id,
            "section": section,
            "when": when,
            "guidance": guidance,
            "invalidators": invalidators,
            "basis": basis,
            "status": status,
            "supporting_trajectory_ids": supporting_trajectory_ids,
            "supporting_fact_ids": supporting_fact_ids,
        }
        if operation == "add":
            if any(
                str(rule.get("rule_id")) == generated_id
                or (
                    str(rule.get("section")) == section
                    and str(rule.get("guidance", "")) == guidance
                )
                for rule in rules
            ):
                raise ValueError("add patch duplicates an existing rule")
            rules.append(new_rule)
        else:
            rules[matching_indexes[0]] = new_rule
        supporting_trajectories.update(supporting_trajectory_ids)
        applied.append({"op": operation, **new_rule})
    for section in EXPERIENCE_SECTION_LABELS:
        if sum(str(rule.get("section")) == section for rule in rules) > 4:
            raise ValueError(f"section {section} exceeds four concise rules")
    if len(rules) > 16:
        raise ValueError("document exceeds sixteen concise rules")
    return rules, applied, sorted(supporting_trajectories)


def _snapshot_positive_integer(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _snapshot_nonnegative_integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _validated_experience_history(
    raw_history: object,
    *,
    agent_id: str,
    scope: MemoryScope,
) -> list[ExperienceDocument]:
    if not isinstance(raw_history, list) or not all(isinstance(item, dict) for item in raw_history):
        raise ValueError("Experience snapshot history must be an object list")
    restored: list[ExperienceDocument] = []
    for raw in raw_history:
        try:
            document = ExperienceDocument(**deepcopy(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError("Experience snapshot history schema is invalid") from exc
        if type(document.version) is not int or document.version < 1:
            raise ValueError("Experience snapshot document version must be positive")
        if not isinstance(document.body, str):
            raise ValueError("Experience snapshot document body must be a string")
        if not isinstance(document.source_hand_ids, list) or not all(
            isinstance(item, str) for item in document.source_hand_ids
        ):
            raise ValueError("Experience snapshot document source_hand_ids must be a string list")
        if not isinstance(document.updated_at, str):
            raise ValueError("Experience snapshot document updated_at must be a string")
        if document.scope != scope:
            raise ValueError("Experience snapshot document scope mismatch")
        if not isinstance(document.metadata, dict):
            raise ValueError("Experience snapshot document metadata must be an object")
        metadata_agent = document.metadata.get("agent_id")
        if metadata_agent is not None and metadata_agent != agent_id:
            raise ValueError("Experience snapshot document agent_id mismatch")
        restored.append(document)
    versions = [document.version for document in restored]
    if versions and versions != list(range(1, len(versions) + 1)):
        raise ValueError("Experience snapshot history versions must be contiguous from 1")
    if restored and restored[0].body != INITIAL_EXPERIENCE:
        raise ValueError("Experience snapshot baseline document was altered")
    return restored


def _validated_string_list(raw: object, field: str, *, minimum: int) -> list[str]:
    if not isinstance(raw, list) or not minimum <= len(raw) <= 4:
        raise ValueError(f"{field} must contain {minimum} to 4 items")
    values = [str(value).strip() for value in raw]
    if any(not value or len(value) > 120 for value in values):
        raise ValueError(f"{field} contains an empty or overlong item")
    return values


def _validated_id_list(raw: object, allowed: set[str], label: str) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError(f"supporting_{label}_ids must be an array")
    values = [str(value) for value in raw]
    if len(values) != len(set(values)) or any(value not in allowed for value in values):
        raise ValueError(f"supporting_{label}_ids contains duplicate or unknown IDs")
    return values


def _validated_full_document_fact_ids(
    raw: object,
    allowed: set[str],
    field: str,
) -> list[str]:
    """Validate one v1 fact classification without accepting trajectory IDs."""

    if not isinstance(raw, list):
        raise ValueError(f"{field} must be an array")
    if any(not isinstance(value, str) for value in raw):
        raise ValueError(f"{field} must contain only string fact IDs")
    values = [str(value) for value in raw]
    if len(values) != len(set(values)):
        raise ValueError(f"{field} contains duplicate IDs")
    if any(value not in allowed for value in values):
        raise ValueError(f"{field} contains an unknown or non-fact ID")
    return values


def _contains_forbidden_identifier(text: str, forbidden_ids: set[str]) -> bool:
    if any(identifier and identifier in text for identifier in forbidden_ids):
        return True
    return bool(
        re.search(
            r"(?i)(?:(?:agent|player|opponent|hand|fact|seat)[_\s:#-]*\d+"
            r"(?!\s*[- ]?bet(?:tor)?(?=$|[^A-Za-z])))",
            text,
        )
    )


def _validated_revision_auxiliary_fields(
    payload: dict[str, object],
    *,
    forbidden_ids: set[str],
    document_contract: str,
) -> dict[str, str]:
    """Validate calibration metadata before it can enter a revision artifact."""

    values: dict[str, str] = {}
    for field in ("calibration_note", "self_check"):
        raw = payload.get(field)
        if not isinstance(raw, str):
            raise ValueError(f"schema_contract_violation: {field} must be a string")
        if _contains_secret_material(raw):
            raise ValueError(f"auxiliary_text_contains_secret: {field}")
        if document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION:
            # Main stores these fields as audit metadata, not as experience.md.
            # Redact incidental hand/player/fact identifiers instead of spending a
            # second model call trying to rewrite an otherwise valid document.
            values[field] = _sanitize_diagnostic_text(raw, forbidden_ids)
            continue
        if len(raw) > 240:
            raise ValueError("schema_contract_violation: calibration fields exceed 240 chars")
        if _contains_forbidden_identifier(raw, forbidden_ids):
            raise ValueError(f"auxiliary_text_contains_identifier: {field}")
        if (
            document_contract == EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION
            and _contains_concrete_card_literal(raw)
        ):
            raise ValueError(f"auxiliary_text_contains_concrete_card: {field}")
        values[field] = raw
    return values


def _contains_action(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "check",
            "call",
            "raise",
            "fold",
            "bet",
            "jam",
            "all-in",
        )
    )


def _render_rule_catalog(rules: list[dict[str, object]]) -> str:
    """Render typed rules into the fixed paper-style five-section Markdown shell."""

    lines = [EXPERIENCE_TITLE]
    for heading, section in zip(
        EXPERIENCE_REQUIRED_HEADINGS, EXPERIENCE_SECTION_LABELS, strict=True
    ):
        lines.extend(["", heading])
        section_rules = [rule for rule in rules if rule.get("section") == section]
        if not section_rules:
            lines.append(EXPERIENCE_EMPTY_SECTION)
            continue
        for rule in section_rules:
            rendered_text = str(rule.get("rendered_text", "")).strip()
            if rendered_text:
                lines.append(f"- {rendered_text}")
                continue
            when = " and ".join(_clean_condition(value) for value in rule.get("when", []))
            guidance = str(rule.get("guidance", "")).rstrip(".; ")
            invalidators = " or ".join(
                _clean_condition(value) for value in rule.get("invalidators", [])
            )
            prefix = "[Tentative] " if rule.get("status") == "tentative" else ""
            lines.append(
                f"- {prefix}When {when}, hero should {guidance}; if {invalidators}, reassess."
            )
    return "\n".join(lines).rstrip() + "\n"


def _clean_condition(value: object) -> str:
    text = str(value).strip().rstrip(".; ")
    if text.lower().startswith("when "):
        text = text[5:]
    return text.strip()


def _normalize_markdown(body: str) -> str:
    """Normalize line endings while retaining the complete model document."""

    return body.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


def _prior_semantic_review_feedback_block(
    revision_log: list[dict[str, object]],
) -> str:
    """Expose only the prior finite rejection diagnosis to the next proposal."""

    if not revision_log:
        return ""
    latest = revision_log[-1]
    if not bool(latest.get("semantic_review_used")):
        return ""
    status = str(latest.get("semantic_review_status", ""))
    if not status.startswith("rejected"):
        return ""
    raw_codes = latest.get("semantic_review_issue_codes", [])
    codes = [str(value) for value in raw_codes] if isinstance(raw_codes, list) else []
    raw_issues = latest.get("semantic_review_issues", [])
    diagnoses: list[str] = []
    if isinstance(raw_issues, list):
        for item in raw_issues[:6]:
            if not isinstance(item, dict):
                continue
            diagnosis = str(item.get("diagnosis", "")).strip()
            if diagnosis:
                diagnoses.append(diagnosis[:180])
    return (
        "Previous independent semantic-review rejection (for correcting the next normal "
        "proposal only; not new trajectory evidence):\n"
        f"- issue_codes: {json.dumps(codes, ensure_ascii=False)}\n"
        f"- diagnoses: {json.dumps(diagnoses, ensure_ascii=False)}\n"
        "Resolve these issues without adding unsupported strategy or restating behavior frequency."
    )


def _semantic_review_host_findings(
    *,
    proposal_revision: dict[str, object],
    candidate_md: str,
    behavior_digest: dict[str, object],
    forbidden_ids: set[str],
) -> list[dict[str, object]]:
    """Create stable host finding IDs that the reviewer must acknowledge."""

    raw_findings = proposal_revision.get("validation_findings", [])
    findings = list(raw_findings) if isinstance(raw_findings, list) else []
    findings.extend(_semantic_behavior_copy_findings(candidate_md, behavior_digest))
    result: list[dict[str, object]] = []
    for index, raw in enumerate(findings[:96], start=1):
        item = raw if isinstance(raw, dict) else {"message": str(raw)}
        code = re.sub(
            r"[^a-zA-Z0-9_.-]+",
            "_",
            str(item.get("code", "deterministic_finding")),
        )[:72]
        finding_id = f"F{index:03d}:{code}"
        result.append(
            {
                "finding_id": finding_id,
                "code": code,
                "section": str(item.get("section", ""))[:32],
                "bullet_index": (
                    item.get("bullet_index") if type(item.get("bullet_index")) is int else None
                ),
                "message": _sanitize_diagnostic_excerpt(
                    str(item.get("message", "")),
                    forbidden_ids,
                    max_chars=240,
                ),
                "excerpt": _sanitize_diagnostic_excerpt(
                    str(item.get("excerpt", "")),
                    forbidden_ids,
                    max_chars=180,
                ),
                "semantic_issue_hint": str(item.get("semantic_issue_hint", ""))[:64],
                "candidate_omits_active_branches": (
                    item.get("candidate_omits_active_branches") is True
                ),
            }
        )
    return result


def _semantic_behavior_copy_findings(
    candidate_md: str,
    behavior_digest: dict[str, object],
) -> list[dict[str, object]]:
    """Flag only document-level copying that is supported by measured coverage."""

    warnings = behavior_digest.get("coverage_warnings", [])
    warning_text = " ".join(str(value) for value in warnings) if isinstance(warnings, list) else ""
    bullets = extract_experience_bullets(candidate_md)
    check_guidance_re = re.compile(
        r"(?:should|prefer|choose|usually|default|can).{0,24}check|"
        r"check.{0,24}(?:pot control|realize equity|induce|avoid|retain)",
        re.IGNORECASE,
    )
    proactive_guidance_re = re.compile(
        r"(?:value bet|proactive bet|continuation bet|semi-bluff|pure bluff|equity denial|"
        r"(?:should|prefer|choose|can).{0,20}(?:bet|raise))",
        re.IGNORECASE,
    )
    check_guidance_count = sum(bool(check_guidance_re.search(bullet.text)) for bullet in bullets)
    proactive_guidance_count = sum(
        bool(proactive_guidance_re.search(bullet.text)) for bullet in bullets
    )
    result: list[dict[str, object]] = []
    if (
        "check_copy_bias" in warning_text
        and check_guidance_count >= 3
        and check_guidance_count > proactive_guidance_count
    ):
        result.append(
            {
                "code": "behavior_check_copy_bias",
                "message": (
                    "candidate turns high observed check frequency into a document-wide "
                    "checking preference without counterfactual EV evidence"
                ),
                "section": "",
                "bullet_index": None,
                "excerpt": "",
                "semantic_issue_hint": "CHECK_COPY_BIAS",
                "candidate_omits_active_branches": True,
            }
        )
    if (
        "low_proactive_action_coverage" in warning_text
        and len(bullets) >= 5
        and proactive_guidance_count == 0
    ):
        result.append(
            {
                "code": "behavior_no_proactive_strategy_branch",
                "message": (
                    "candidate omits every value-bet, bluff, or equity-denial branch while "
                    "the evidence window has low proactive-action coverage"
                ),
                "section": "",
                "bullet_index": None,
                "excerpt": "",
                "semantic_issue_hint": "CHECK_COPY_BIAS",
                "candidate_omits_active_branches": True,
            }
        )
    return result


def _semantic_review_allowed_evidence_refs(
    *,
    trajectory_payload: list[dict[str, object]],
    fact_payload: list[dict[str, object]],
    candidate_md: str,
    host_findings: list[dict[str, object]],
) -> list[str]:
    refs = {
        "aggregate:action-counts",
        "aggregate:behavior-coverage",
        "calibration:current-hand",
    }
    for recap in trajectory_payload:
        trajectory_id = str(recap.get("trajectory_id", ""))
        if not trajectory_id:
            continue
        refs.add(f"trajectory:{trajectory_id}")
        decisions = recap.get("hero_decisions", [])
        if isinstance(decisions, list):
            refs.update(
                f"trajectory:{trajectory_id}#decision-{index}"
                for index in range(1, len(decisions) + 1)
            )
    for fact in fact_payload:
        record_id = str(fact.get("record_id", ""))
        if record_id:
            refs.add(f"fact:{record_id}")
    refs.update(
        f"candidate:{bullet.section}:{bullet.index}"
        for bullet in extract_experience_bullets(candidate_md)
    )
    refs.update(
        f"deterministic:{item['finding_id']}" for item in host_findings if item.get("finding_id")
    )
    return sorted(refs)


def _semantic_review_replay_findings(
    *,
    current_body: str,
    proposed_body: str,
    max_chars: int,
    document_contract: str,
    forbidden_ids: set[str],
) -> list[dict[str, object]]:
    findings = validate_experience_document_findings(
        proposed_body,
        max_chars=max_chars,
        document_contract=document_contract,
    )
    if document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION:
        findings.extend(_paper_merge_before_add_findings(current_body, proposed_body))
    if _contains_forbidden_identifier(proposed_body, forbidden_ids) and not any(
        str(item.get("code", "")) == "concrete_identifier" for item in findings
    ):
        findings.append(
            _validation_finding(
                "concrete_identifier",
                "document contains a concrete identifier",
            )
        )
    return findings


def _semantic_review_operational_failure(
    proposal_revision: dict[str, object],
    *,
    current_body: str,
    failure: str,
    reviewer_model: str,
) -> dict[str, object]:
    failed = dict(proposal_revision)
    failed.update(
        {
            "keep": True,
            "new_md": current_body,
            "validation_status": "failed_closed",
            "fallback_used": True,
            "failure": failure,
            "revision_strategy": "llm_full_document_independent_review_failed",
            "semantic_review_used": True,
            "semantic_review_mode": "independent_model_v1",
            "semantic_review_status": "reviewer_failure",
            "semantic_review_reviewer_model": reviewer_model,
        }
    )
    return failed


def _compact_semantic_issue(issue: object) -> dict[str, object]:
    return {
        "issue_id": str(getattr(issue, "issue_id", ""))[:8],
        "code": str(getattr(issue, "code", ""))[:64],
        "scope": str(getattr(issue, "scope", ""))[:32],
        "target_bullet_id": getattr(issue, "target_bullet_id", None),
        "section": getattr(issue, "section", None),
        "bullet_index": getattr(issue, "bullet_index", None),
        "claim_excerpt": str(getattr(issue, "claim_excerpt", ""))[:180],
        "evidence_refs": list(getattr(issue, "evidence_refs", ()))[:12],
        "reference_card_ids": list(getattr(issue, "reference_card_ids", ()))[:6],
        "source_finding_ids": list(getattr(issue, "source_finding_ids", ()))[:16],
        "diagnosis": str(getattr(issue, "diagnosis", ""))[:240],
    }


_COMPACT_REVISION_FIELDS = {
    "terminal_policy_receipt",
    "terminal_outcome",
    "contract_repair_exhausted",
    "rev",
    "hand_index",
    "window_hand_count",
    "keep",
    "keep_contract_violation",
    "ignored_keep_new_md",
    "contract_violation",
    "validation_status",
    "application_status",
    "installed_version",
    "experience_revision_created",
    "calibration_note",
    "calibration_note_sha256",
    "calibration_note_chars",
    "calibration_note_neutralized",
    "calibration_validation_findings",
    "self_check",
    "self_check_sha256",
    "self_check_chars",
    "sanitized_auxiliary_fields",
    "ignored_unknown_fact_id_count",
    "reclassified_trajectory_id_count",
    "prompt_token_budget_advisory",
    "supporting_fact_ids",
    "contradicting_fact_ids",
    "noise_fact_ids",
    "irrelevant_fact_ids",
    "ambiguous_fact_ids",
    "fact_assessments",
    "supporting_trajectory_ids",
    "sweep_evidence_fact_ids",
    "sweep_supporting_fact_ids",
    "sweep_contradicting_fact_ids",
    "sweep_noise_fact_ids",
    "sweep_irrelevant_fact_ids",
    "sweep_evidence_groups",
    "evidence_classification_status",
    "provisional_classification_source",
    "revision_strategy",
    "requested_strategy",
    "schema_version",
    "prompt_version",
    "revision_recap_version",
    "revision_recap_sha256",
    "revision_recap_chars",
    "revision_prompt_compression",
    "behavior_digest_version",
    "behavior_digest_sha256",
    "behavior_digest_chars",
    "prompt_sha256",
    "model",
    "document_contract",
    "fallback_used",
    "repair_used",
    "semantic_repair_used",
    "semantic_repair_prompt_sha256",
    "semantic_repair_recap_hand_count",
    "semantic_repair_fact_count",
    "semantic_review_used",
    "semantic_review_mode",
    "semantic_review_status",
    "semantic_review_verdict",
    "semantic_review_prompt_version",
    "semantic_review_prompt_sha256",
    "semantic_review_candidate_sha256",
    "semantic_review_evidence_bundle_sha256",
    "semantic_review_evidence_hand_count_included",
    "semantic_review_evidence_hand_count_total",
    "semantic_review_reference_card_sha256",
    "semantic_review_proposer_model",
    "semantic_review_reviewer_model",
    "semantic_review_proposer_family",
    "semantic_review_reviewer_family",
    "semantic_review_independence_passed",
    "semantic_review_mandatory_finding_ids",
    "semantic_review_acknowledged_finding_ids",
    "semantic_review_host_findings",
    "semantic_review_issue_codes",
    "semantic_review_issues",
    "semantic_review_applied_repair_count",
    "semantic_review_replay_findings",
    "semantic_review_provider_diagnostics",
    "semantic_review_failure_code",
    "initial_validation_failure",
    "initial_validation_failure_code",
    "initial_validation_failure_sha256",
    "initial_validation_failure_chars",
    "failure",
    "failure_code",
    "failure_sha256",
    "failure_chars",
    "rejected_payload_sha256",
    "rejected_payload_chars",
    "rejected_payload_summary",
    "initial_rejected_payload_sha256",
    "initial_rejected_payload_chars",
    "initial_rejected_payload_summary",
    "provider_diagnostics",
    "initial_provider_diagnostics",
    "defaulted_response_fields",
    "ignored_response_fields",
    "applied_patches",
    "proposal_summary",
    "validation_findings",
    "advisory_findings",
    "advisory_finding_count",
    "initial_validation_findings",
    "initial_proposal_summary",
    "initial_proposed_new_md_sha256",
    "initial_proposed_new_md_chars",
    "old_md_sha256",
    "old_md_chars",
    "new_md_sha256",
    "new_md_chars",
    "proposed_new_md_sha256",
    "proposed_new_md_chars",
}


def _compact_rejected_payload_summary(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    raw_operation = value.get("operation")
    raw_patch_count = value.get("patch_count")
    raw_sections = value.get("sections")
    return {
        "operation": (
            str(raw_operation)
            if raw_operation in {"keep", "patch", "<INVALID_OPERATION>"}
            else "<INVALID_OPERATION>"
        ),
        "patch_count": (
            raw_patch_count if type(raw_patch_count) is int and raw_patch_count >= 0 else None
        ),
        "sections": [
            str(section) if section in EXPERIENCE_SECTION_LABELS else "<INVALID_SECTION>"
            for section in list(raw_sections)[:16]
        ]
        if isinstance(raw_sections, list)
        else [],
    }


def _compact_revision_record(revision: dict[str, object]) -> dict[str, object]:
    """Persist an explicit safe diagnostic allowlist, never complete proposals."""

    protected_fields = {
        "calibration_note",
        "failure",
        "initial_validation_failure",
        "initial_rejected_payload_summary",
        "provider_diagnostics",
        "initial_provider_diagnostics",
        "semantic_review_provider_diagnostics",
        "rejected_payload_summary",
        "self_check",
    }
    compact = {
        key: value
        for key, value in revision.items()
        if key in _COMPACT_REVISION_FIELDS and key not in protected_fields
    }
    for key in ("failure", "initial_validation_failure"):
        if key not in revision or revision.get(key) is None:
            continue
        diagnostic = experience_failure_diagnostic(revision.get(key))
        compact[key] = diagnostic["preview"]
        compact.setdefault(f"{key}_code", diagnostic["code"])
        compact.setdefault(f"{key}_sha256", diagnostic["sha256"])
        compact.setdefault(f"{key}_chars", diagnostic["chars"])
    for key in ("calibration_note", "self_check"):
        if key not in revision or revision.get(key) is None:
            continue
        raw = str(revision.get(key))
        fingerprint = _text_fingerprint(raw)
        sanitized = _sanitize_diagnostic_excerpt(raw, set(), max_chars=120)
        compact[key] = sanitized
        provided_sha256 = revision.get(f"{key}_sha256")
        provided_chars = revision.get(f"{key}_chars")
        if (
            sanitized == raw
            and isinstance(provided_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", provided_sha256)
            and type(provided_chars) is int
            and provided_chars >= 0
        ):
            compact[f"{key}_sha256"] = provided_sha256
            compact[f"{key}_chars"] = provided_chars
        else:
            compact[f"{key}_sha256"] = fingerprint["sha256"]
            compact[f"{key}_chars"] = fingerprint["chars"]
    for key in (
        "provider_diagnostics",
        "initial_provider_diagnostics",
        "semantic_review_provider_diagnostics",
    ):
        if isinstance(revision.get(key), dict):
            compact[key] = _compact_provider_diagnostics(revision[key])
    for key in ("rejected_payload_summary", "initial_rejected_payload_summary"):
        if isinstance(revision.get(key), dict):
            compact[key] = _compact_rejected_payload_summary(revision[key])
    for key in ("old_md", "new_md", "proposed_new_md"):
        value = revision.get(key)
        if not isinstance(value, str):
            continue
        if (
            key == "proposed_new_md"
            and not value
            and not (
                is_experience_revision_validation_rejection(revision)
                and revision.get("schema_version")
                == EXPERIENCE_FULL_DOCUMENT_REVISION_SCHEMA_VERSION
            )
        ):
            continue
        compact[f"{key}_sha256"] = hashlib.sha256(value.encode("utf-8")).hexdigest()
        compact[f"{key}_chars"] = len(value)
    return compact


def is_experience_revision_validation_rejection(
    revision: dict[str, object],
) -> bool:
    """Return whether the local quality/semantic gate rejected one proposal."""

    return revision.get("validation_status") == "rejected"


def is_experience_revision_operational_failure(
    revision: dict[str, object],
) -> bool:
    """Separate provider/runtime failures from successful fail-closed rejection.

    Rejected proposals retain a compact ``failure`` reason for diagnosis.  That
    field must not by itself turn a successful guard decision into an execution
    failure when the canonical document was preserved.
    """

    receipt = revision.get("terminal_policy_receipt")
    response_contract_preserved = (
        isinstance(receipt, dict) and receipt.get("policy") == PATH1_TERMINAL_POLICY
        and receipt.get("initial_requests") == 1 and receipt.get("contract_repairs") == 1
        and receipt.get("completed_model_responses") == 2
        and revision.get("terminal_outcome") == "response_contract_preserved_current"
        and revision.get("validation_status") == "response_contract_failed"
        and revision.get("keep") is True and revision.get("fallback_used") is False
    )
    return bool(revision.get("failure")) and not (
        is_experience_revision_validation_rejection(revision) or response_contract_preserved
    )


def experience_revision_has_initial_rejection(
    revision: dict[str, object],
) -> bool:
    """Return whether semantic repair was triggered by an initial rejection."""

    return bool(revision.get("semantic_repair_used")) or any(
        key in revision
        for key in (
            "initial_validation_failure",
            "initial_proposed_new_md_sha256",
            "initial_proposed_new_md_chars",
            "initial_validation_findings",
            "initial_proposal_summary",
            "initial_rejected_payload_sha256",
            "initial_rejected_payload_chars",
            "initial_rejected_payload_summary",
        )
    )


def experience_revision_rejection_diagnostic_complete(
    revision: dict[str, object],
) -> bool:
    """Validate body-free diagnostics for a final fail-closed rejection."""

    if not is_experience_revision_validation_rejection(revision):
        return False
    findings = revision.get("validation_findings", [])
    has_reason = bool(str(revision.get("failure", "")).strip()) or (
        isinstance(findings, list) and bool(findings)
    )
    return has_reason and (
        _revision_document_fingerprint_complete(revision, prefix="proposed_new_md")
        or _revision_payload_fingerprint_complete(revision, prefix="rejected_payload")
    )


def experience_revision_initial_rejection_diagnostic_complete(
    revision: dict[str, object],
) -> bool:
    """Validate body-free diagnostics for the proposal that triggered repair."""

    if not experience_revision_has_initial_rejection(revision):
        return False
    findings = revision.get("initial_validation_findings", [])
    has_reason = bool(str(revision.get("initial_validation_failure", "")).strip()) or (
        isinstance(findings, list) and bool(findings)
    )
    return has_reason and (
        _revision_document_fingerprint_complete(
            revision,
            prefix="initial_proposed_new_md",
            summary_field="initial_proposal_summary",
        )
        or _revision_payload_fingerprint_complete(
            revision,
            prefix="initial_rejected_payload",
            summary_field="initial_rejected_payload_summary",
        )
    )


def _revision_document_fingerprint_complete(
    revision: dict[str, object],
    *,
    prefix: str,
    summary_field: str = "proposal_summary",
) -> bool:
    chars = revision.get(f"{prefix}_chars")
    sha256 = revision.get(f"{prefix}_sha256")
    if (
        type(chars) is not int
        or chars < 0
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
    ):
        return False
    return chars == 0 or isinstance(revision.get(summary_field), dict)


def _revision_payload_fingerprint_complete(
    revision: dict[str, object],
    *,
    prefix: str,
    summary_field: str | None = None,
) -> bool:
    chars = revision.get(f"{prefix}_chars")
    sha256 = revision.get(f"{prefix}_sha256")
    resolved_summary_field = summary_field or f"{prefix}_summary"
    return (
        type(chars) is int
        and chars > 0
        and isinstance(sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", sha256) is not None
        and isinstance(revision.get(resolved_summary_field), dict)
    )


def validate_experience_document(
    body: str,
    *,
    max_chars: int,
    document_contract: str = EXPERIENCE_LEGACY_DOCUMENT_CONTRACT_VERSION,
) -> list[str]:
    """Validate a versioned five-section document and return stable messages."""

    return [
        _format_validation_finding(item)
        for item in validate_experience_document_findings(
            body,
            max_chars=max_chars,
            document_contract=document_contract,
        )
    ]


def validate_experience_document_findings(
    body: str,
    *,
    max_chars: int,
    document_contract: str = EXPERIENCE_LEGACY_DOCUMENT_CONTRACT_VERSION,
) -> list[dict[str, object]]:
    """Return structured findings without mutating or auto-rewriting LLM prose."""

    if document_contract not in {
        EXPERIENCE_LEGACY_DOCUMENT_CONTRACT_VERSION,
        EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION,
        EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION,
    }:
        raise ValueError(f"unknown experience document contract: {document_contract}")
    findings: list[dict[str, object]] = []
    normalized = _normalize_markdown(body)
    if len(normalized) > max_chars:
        findings.append(
            _validation_finding(
                "document_too_long",
                f"length {len(normalized)} exceeds max_chars {max_chars}",
            )
        )
    lines = normalized.splitlines()
    if not lines or lines[0].strip() != EXPERIENCE_TITLE:
        findings.append(
            _validation_finding("missing_title", f"missing exact title {EXPERIENCE_TITLE!r}")
        )
    secondary = [line.strip() for line in lines if line.startswith("## ")]
    if secondary != list(EXPERIENCE_REQUIRED_HEADINGS):
        findings.append(
            _validation_finding(
                "heading_contract",
                "five required headings are missing, duplicated, extra, or out of order",
            )
        )
        return findings
    heading_indexes = [lines.index(heading) for heading in EXPERIENCE_REQUIRED_HEADINGS]
    preamble = [line.strip() for line in lines[1 : heading_indexes[0]] if line.strip()]
    if preamble:
        findings.append(
            _validation_finding(
                "document_preamble_contract",
                "content between the title and first required section is not allowed",
                excerpt=" ".join(preamble),
            )
        )
    section_bullets: dict[str, list[str]] = {}
    for index, heading in enumerate(EXPERIENCE_REQUIRED_HEADINGS):
        start = heading_indexes[index] + 1
        end = heading_indexes[index + 1] if index + 1 < len(heading_indexes) else len(lines)
        content = [line.strip() for line in lines[start:end] if line.strip()]
        if len(content) == 1 and content[0] in EXPERIENCE_ACCEPTED_EMPTY_SECTIONS:
            section_bullets[heading] = []
            continue
        if document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION:
            # Main accepts ordinary Markdown prose generated as a wrapped list
            # item.  Treat non-heading lines after a bullet as continuations
            # for validation without rewriting the model's canonical document.
            paper_bullets: list[str] = []
            invalid_content = False
            for line in content:
                if line.startswith("- ") and line[2:].strip():
                    paper_bullets.append(line[2:].strip())
                elif numbered := _PAPER_NUMBERED_BULLET_RE.match(line):
                    # Keep the format failure, but parse the rule provisionally
                    # so one repair request receives all objective defects at
                    # once instead of discovering them one layer at a time.
                    invalid_content = True
                    paper_bullets.append(numbered.group(1).strip())
                elif paper_bullets and not line.startswith(("-", "#", "```")):
                    paper_bullets[-1] = f"{paper_bullets[-1]} {line}"
                else:
                    invalid_content = True
            if invalid_content or not paper_bullets:
                findings.append(
                    _validation_finding(
                        "section_content_contract",
                        f"section {heading} must contain Markdown bullets "
                        "or the exact empty marker",
                        section=heading,
                    )
                )
            section_bullets[heading] = paper_bullets
            continue
        bullets = [line for line in content if line.startswith("- ") and line[2:].strip()]
        if len(bullets) != len(content):
            findings.append(
                _validation_finding(
                    "section_content_contract",
                    f"section {heading} must contain only bullets or the exact empty marker",
                    section=heading,
                )
            )
            section_bullets[heading] = [line[2:].strip() for line in bullets]
            continue
        section_bullets[heading] = [line[2:].strip() for line in bullets]
        for bullet_index, bullet in enumerate(section_bullets[heading], start=1):
            if document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION:
                continue
            actionable = (
                _is_transferable_actionable_rule(bullet)
                if document_contract == EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION
                else _is_actionable_rule(bullet)
            )
            if not actionable:
                findings.append(
                    _validation_finding(
                        "non_actionable_rule",
                        f"section {heading} contains a non-actionable bullet",
                        section=heading,
                        bullet_index=bullet_index,
                        excerpt=bullet,
                    )
                )
    if document_contract == EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION:
        findings.extend(_transferable_rulebook_findings(section_bullets))
        findings.extend(_poker_semantic_findings(normalized, section_bullets))
    elif document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION:
        findings.extend(
            _paper_document_blocking_findings(
                _paper_semantic_diagnostic_body(normalized), section_bullets
            )
        )
    return _deduplicate_validation_findings(findings)


def _paper_semantic_diagnostic_body(body: str) -> str:
    """Expose malformed numbered items to semantic checks without accepting them."""

    return re.sub(r"(?m)^(\s*)\d{1,2}[.)]\s+", r"\1- ", body)


def _is_actionable_rule(rule: str) -> bool:
    """Require a visible condition and a concrete hero action."""

    if _contains_audit_narrative(rule):
        return False
    has_condition = bool(
        re.search(
            r"\b(?:when|if|unless|facing|against|with|on the|at an?\b|SPR|pot odds|"
            r"effective stack|position)\b",
            rule,
            re.IGNORECASE,
        )
    )
    has_action = bool(
        re.search(
            r"\b(?:hero|we)\b.{0,64}\b(?:should|can|prefer|choose|fold|check|call|bet|raise|jam)\b",
            rule,
            re.IGNORECASE,
        )
    )
    return has_condition and has_action


def _is_transferable_actionable_rule(rule: str) -> bool:
    """Apply the stricter publication actionability gate."""

    if not _is_actionable_rule(rule):
        return False
    return not re.search(
        r"\b(?:consider|observe|record|review|maybe|sometimes)\b.{0,20}$",
        rule,
        re.IGNORECASE,
    )


def _transferable_rulebook_findings(
    section_bullets: dict[str, list[str]],
    *,
    max_rules_total: int = EXPERIENCE_MAX_RULES_TOTAL,
    max_rules_per_section: int = EXPERIENCE_MAX_RULES_PER_SECTION,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    total_rules = sum(len(items) for items in section_bullets.values())
    if total_rules > max_rules_total:
        findings.append(
            _validation_finding(
                "document_rule_capacity",
                f"document contains {total_rules} rules; maximum is {max_rules_total}",
            )
        )
    seen_exact: dict[str, tuple[str, int]] = {}
    seen_rules: list[tuple[str, int, str]] = []
    for section, bullets in section_bullets.items():
        if len(bullets) > max_rules_per_section:
            findings.append(
                _validation_finding(
                    "section_rule_capacity",
                    f"section {section} contains {len(bullets)} rules; maximum is "
                    f"{max_rules_per_section}",
                    section=section,
                )
            )
        for bullet_index, bullet in enumerate(bullets, start=1):
            starting_hand_examples = list(
                dict.fromkeys(
                    match.group(0).upper() for match in _STARTING_HAND_EXAMPLE_RE.finditer(bullet)
                )
            )
            if len(starting_hand_examples) > 2:
                findings.append(
                    _validation_finding(
                        "starting_hand_example_inventory",
                        "a rule may contain at most two representative starting-hand "
                        "examples; use a range or category instead of an inventory",
                        section=section,
                        bullet_index=bullet_index,
                        excerpt=bullet,
                    )
                )
            if _contains_audit_narrative(bullet):
                findings.append(
                    _validation_finding(
                        "audit_narrative_rule",
                        "long-term experience cannot be a recent-result or sample narrative",
                        section=section,
                        bullet_index=bullet_index,
                        excerpt=bullet,
                    )
                )
            if _contains_concrete_card_literal(bullet):
                findings.append(
                    _validation_finding(
                        "concrete_deal_example",
                        "rule contains a suit-specific dealt card; use a transferable range",
                        section=section,
                        bullet_index=bullet_index,
                        excerpt=bullet,
                    )
                )
            if _contains_forbidden_identifier(bullet, set()):
                findings.append(
                    _validation_finding(
                        "concrete_identifier",
                        "rule contains a concrete hand, seat, player, or opponent identifier",
                        section=section,
                        bullet_index=bullet_index,
                        excerpt=bullet,
                    )
                )
            if section == EXPERIENCE_REQUIRED_HEADINGS[3] and not _is_opponent_archetype_rule(
                bullet
            ):
                findings.append(
                    _validation_finding(
                        "opponent_response_not_typed",
                        "opponent-response rules must name a reusable type or action line",
                        section=section,
                        bullet_index=bullet_index,
                        excerpt=bullet,
                    )
                )
            if (
                section == EXPERIENCE_REQUIRED_HEADINGS[4]
                and _PREFLOP_RANGE_TERM_RE.search(bullet)
                and _PREFLOP_STRATEGY_RE.search(bullet)
            ):
                findings.append(
                    _validation_finding(
                        "misplaced_preflop_range_rule",
                        "explicit starting-hand ranges belong in the starting-hand section; do "
                        "not repeat them in the sizing-and-position section",
                        section=section,
                        bullet_index=bullet_index,
                        excerpt=bullet,
                    )
                )
            canonical = _canonical_rule_text(bullet)
            previous = seen_exact.get(canonical)
            if previous is not None:
                findings.append(
                    _validation_finding(
                        "duplicate_rule",
                        "rule duplicates an existing bullet and must be merged",
                        section=section,
                        bullet_index=bullet_index,
                        excerpt=bullet,
                        peer_section=previous[0],
                        peer_index=previous[1],
                    )
                )
            else:
                seen_exact[canonical] = (section, bullet_index)
            for peer_section, peer_index, peer_text in seen_rules:
                if _canonical_rule_text(peer_text) == canonical:
                    continue
                if _rules_are_high_confidence_near_duplicates(
                    peer_text, bullet
                ) or _rules_share_high_confidence_duplicate_fragment(peer_text, bullet):
                    findings.append(
                        _validation_finding(
                            "near_duplicate_rule",
                            "rule contains the same actionable strategy as another "
                            "bullet and must be merged",
                            section=section,
                            bullet_index=bullet_index,
                            excerpt=bullet,
                            peer_section=peer_section,
                            peer_index=peer_index,
                        )
                    )
                    break
            seen_rules.append((section, bullet_index, bullet))
    return findings


def _poker_semantic_findings(
    body: str,
    section_bullets: dict[str, list[str]],
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for issue in validate_poker_experience_document(body):
        section = f"## {issue.section}" if issue.section else ""
        bullets = section_bullets.get(section, [])
        bullet_index = next(
            (index for index, text in enumerate(bullets, start=1) if text == issue.bullet),
            None,
        )
        findings.append(
            _validation_finding(
                issue.code,
                issue.message,
                section=section,
                bullet_index=bullet_index,
                excerpt=issue.bullet,
            )
        )
    return findings


_PAPER_BLOCKING_TRANSFERABLE_CODES = {
    "audit_narrative_rule",
    "concrete_deal_example",
    "concrete_identifier",
    "document_rule_capacity",
    "duplicate_rule",
    "misplaced_preflop_range_rule",
    "near_duplicate_rule",
    "opponent_response_not_typed",
    "section_rule_capacity",
    "starting_hand_example_inventory",
}

_PAPER_BLOCKING_POKER_CODES = {
    "board_flush_claim",
    "board_straight_claim",
    "call_protection",
    "check_facing_bet",
    "concrete_suited_card",
    "connectivity_label",
    "flop_section",
    "gutshot_outs",
    "hand_notation_order",
    "hidden_hand_inference",
    "hu_check_back_order",
    "hu_isolation",
    "hu_preflop_order",
    "invalid_bet_purpose",
    "late_street_section",
    "missing_action",
    "missing_hero_guidance",
    "open_ended_outs",
    "pair_suffix",
    "paired_board_claim",
    "position_street_scope",
    "preflop_section",
    "preflop_spr",
    "raise_without_bet",
    "reraise_chain",
    "river_future_card",
    "single_action_archetype",
    "stack_spr_conflation",
    "unconditional_rule",
    "unknown_section",
    "value_premise_missing",
}


def _paper_document_blocking_findings(
    body: str,
    section_bullets: dict[str, list[str]],
) -> list[dict[str, object]]:
    """Compile objective paper-document invariants into the production write gate.

    Main remains the semantic mother implementation: the model still revises and
    atomically replaces one complete five-section document.  This function only
    rejects machine-checkable corruption before it can become the next prompt's
    inherited state.  Coverage and strategy strength remain external audit signals.
    """

    transferable = [
        item
        for item in _transferable_rulebook_findings(
            section_bullets,
            max_rules_total=EXPERIENCE_PAPER_CONSOLIDATION_MAX_RULES_TOTAL,
            max_rules_per_section=(EXPERIENCE_PAPER_CONSOLIDATION_MAX_RULES_PER_SECTION),
        )
        if str(item.get("code", "")) in _PAPER_BLOCKING_TRANSFERABLE_CODES
    ]
    poker = [
        item
        for item in _poker_semantic_findings(body, section_bullets)
        if str(item.get("code", "")) in _PAPER_BLOCKING_POKER_CODES
    ]
    actionable: list[dict[str, object]] = []
    for section, bullets in section_bullets.items():
        for bullet_index, bullet in enumerate(bullets, start=1):
            if _contains_audit_narrative(bullet) or _is_transferable_actionable_rule(bullet):
                continue
            actionable.append(
                _validation_finding(
                    "non_actionable_rule",
                    "every persistent experience bullet must prescribe an observable action "
                    "or sizing adjustment",
                    section=section,
                    bullet_index=bullet_index,
                    excerpt=bullet,
                )
            )

    postflop_heading = EXPERIENCE_REQUIRED_HEADINGS[1]
    flop_bullets = section_bullets.get(postflop_heading, [])
    flop_coverage: list[dict[str, object]] = []
    if flop_bullets and not any(
        bool(recommended_action_streets(bullet).intersection({"flop", "unspecified"}))
        or (not recommended_action_streets(bullet) and _is_transferable_actionable_rule(bullet))
        for bullet in flop_bullets
    ):
        flop_coverage.append(
            _validation_finding(
                "flop_section_without_flop_guidance",
                "the flop chapter contains no rule whose recommended hero action occurs "
                "on the flop",
                section=postflop_heading,
            )
        )
    return [*transferable, *poker, *actionable, *flop_coverage]


_CALIBRATION_COMPACT_BOARD_RE = re.compile(
    r"(?<![A-Za-z0-9])[2-9TJQKA]{3,5}(?![A-Za-z0-9])|"
    r"(?<!\d)[2-9TJQKA](?:\s*[-–—]\s*[2-9TJQKA]){2,4}(?!\d)",
    re.IGNORECASE,
)
_CALIBRATION_DERIVED_HAND_RE = re.compile(
    r"\b(?:hero|opponent|villain)\b.{0,24}\b(?:has|holds|made)\b.{0,16}"
    r"\b(?:high card|pair|two pair|trips|straight|flush|full house|quads|draw)\b",
    re.IGNORECASE,
)
_CALIBRATION_ANY_HAND_CLASS_RE = re.compile(
    r"\b(?:high card|top pair|bottom pair|middle pair|overpair|pair|two pair|trips|"
    r"straight|flush|full house|quads|straight flush|royal flush|draw)\b",
    re.IGNORECASE,
)
_CALIBRATION_HAND_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:hand\s*)?[hH]?[-_ ]?\d+(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_CALIBRATION_COUNTERFACTUAL_RE = re.compile(
    r"\b(?:missed value|would have|could have won|unrealized value|lost opportunity)\b",
    re.IGNORECASE,
)
_CALIBRATION_SINGLE_OUTCOME_VALIDATION_RE = re.compile(
    r"\b(?:won|lost|win|loss|result|net reward)\b.{0,40}"
    r"\b(?:proves?|shows?|validates?|means?)\b.{0,30}"
    r"\b(?:strategy|action|decision|line)\b.{0,20}"
    r"\b(?:correct|incorrect|optimal|better|valid|failed)\b",
    re.IGNORECASE,
)
_CALIBRATION_OUTCOME_INFERENCE_NEGATION_RE = re.compile(
    r"\b(?:does not|cannot|is not enough to|must not|should not)\b.{0,24}"
    r"\b(?:prove|show|validate|mean|infer)\b",
    re.IGNORECASE,
)
_CALIBRATION_OPPONENT_OUTCOME_CLAIM_RE = re.compile(
    r"\b(?:opponent|villain)\b.{0,24}\b(?:won|lost|win|loss)\b",
    re.IGNORECASE,
)


def _paper_calibration_findings(
    note: str,
    *,
    current_reward: int | None,
) -> list[dict[str, object]]:
    """Keep calibration tied to recorded intent/action/outcome only.

    The revision model may compare intent with the observed committed action and
    reward.  It may not recompute hand classes from prose, reveal card examples,
    or claim the payoff of an action that was never taken.
    """

    if not note:
        return []
    findings: list[dict[str, object]] = []
    if (
        "<specific-card>" in note
        or _STARTING_HAND_EXAMPLE_RE.search(note)
        or _CALIBRATION_COMPACT_BOARD_RE.search(note)
        or _CALIBRATION_DERIVED_HAND_RE.search(note)
        or _CALIBRATION_ANY_HAND_CLASS_RE.search(note)
    ):
        findings.append(
            _validation_finding(
                "calibration_unverified_hand_claim",
                "calibration must not restate cards or derive hand categories; compare "
                "recorded intent, committed action, net reward and visible showdown only",
                excerpt=note,
            )
        )
    if _CALIBRATION_HAND_REFERENCE_RE.search(note):
        findings.append(
            _validation_finding(
                "calibration_hand_reference_forbidden",
                "calibration is scoped to the current recap and must not name a hand number; "
                "this prevents stale-window evidence from being described as the current hand",
                excerpt=note,
            )
        )
    if _CALIBRATION_COUNTERFACTUAL_RE.search(note):
        findings.append(
            _validation_finding(
                "calibration_unobserved_counterfactual",
                "the trajectory does not reveal the payoff of an action that was not taken",
                excerpt=note,
            )
        )
    outcome_validation_clauses = [
        clause
        for clause in re.split(r"[.;\n]", note)
        if _CALIBRATION_SINGLE_OUTCOME_VALIDATION_RE.search(clause)
        and not _CALIBRATION_OUTCOME_INFERENCE_NEGATION_RE.search(clause)
    ]
    if outcome_validation_clauses:
        findings.append(
            _validation_finding(
                "calibration_single_outcome_strategy_validation",
                "one realized hand outcome cannot establish that the chosen strategy was "
                "correct or incorrect without a counterfactual or broader stable evidence",
                excerpt=outcome_validation_clauses[0],
            )
        )
    if current_reward is not None and re.search(r"\bthis hand\b", note, re.IGNORECASE):
        hero_claim_text = _CALIBRATION_OPPONENT_OUTCOME_CLAIM_RE.sub("", note)
        claims_win = bool(
            re.search(
                r"\bwin\b|\bwon\b",
                hero_claim_text,
                re.IGNORECASE,
            )
        )
        claims_loss = bool(
            re.search(
                r"\bloss\b|\blost\b",
                hero_claim_text,
                re.IGNORECASE,
            )
        )
        if (claims_win and current_reward <= 0) or (claims_loss and current_reward >= 0):
            findings.append(
                _validation_finding(
                    "calibration_outcome_polarity_conflict",
                    "the claimed current-hand win/loss polarity conflicts with net_reward_chips",
                    excerpt=note,
                )
            )
    return findings


def validate_experience_calibration_findings(
    note: str,
    *,
    current_reward: int | None = None,
) -> list[dict[str, object]]:
    """Expose the paper calibration gate for independent artifact replay."""

    return _paper_calibration_findings(note, current_reward=current_reward)


def experience_document_advisory_findings(
    body: str,
    *,
    document_contract: str,
) -> list[dict[str, object]]:
    """Return non-blocking quality diagnostics for the paper-aligned contract."""

    if document_contract != EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION:
        return []
    sections = _extract_section_bullets(body)
    findings: list[dict[str, object]] = []
    for section, bullets in sections.items():
        for bullet_index, bullet in enumerate(bullets, start=1):
            if not _is_transferable_actionable_rule(bullet):
                findings.append(
                    _validation_finding(
                        "non_actionable_rule",
                        f"section {section} contains a non-actionable bullet",
                        section=section,
                        bullet_index=bullet_index,
                        excerpt=bullet,
                    )
                )
    findings.extend(
        item
        for item in _transferable_rulebook_findings(
            sections,
            max_rules_total=EXPERIENCE_PAPER_CONSOLIDATION_MAX_RULES_TOTAL,
            max_rules_per_section=(EXPERIENCE_PAPER_CONSOLIDATION_MAX_RULES_PER_SECTION),
        )
        if str(item.get("code", "")) not in _PAPER_BLOCKING_TRANSFERABLE_CODES
    )
    findings.extend(
        item
        for item in _poker_semantic_findings(body, sections)
        if str(item.get("code", "")) not in _PAPER_BLOCKING_POKER_CODES
    )
    return [
        {**item, "severity": "advisory_quality"}
        for item in _deduplicate_validation_findings(findings)
    ]


def _validation_finding(
    code: str,
    message: str,
    *,
    section: str = "",
    bullet_index: int | None = None,
    excerpt: str = "",
    peer_section: str = "",
    peer_index: int | None = None,
) -> dict[str, object]:
    return {
        "code": code,
        "message": message,
        "section": section,
        "bullet_index": bullet_index,
        "excerpt": excerpt,
        "peer_section": peer_section,
        "peer_index": peer_index,
    }


def _format_validation_finding(finding: dict[str, object]) -> str:
    return str(finding.get("message", finding.get("code", "invalid document")))


def _deduplicate_validation_findings(
    findings: list[dict[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for finding in findings:
        key = (
            finding.get("code"),
            finding.get("section"),
            finding.get("bullet_index"),
            finding.get("peer_section"),
            finding.get("peer_index"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(finding)
    return result


_CONCRETE_CARD_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:(?:10|[2-9TJQKA])[cCdDhHsS]){1,5}"
    r"(?![A-Za-z0-9])|"
    r"[♠♥♦♣](?:10|[2-9TJQKA])|(?:10|[2-9TJQKA])[♠♥♦♣]"
)
_STARTING_HAND_EXAMPLE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[2-9TJQKA]{2}(?:[sSoO])?\+?)(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])[2-9TJQKA]\s*[-—–]\s*[2-9TJQKA]\s*"
    r"[sSoO](?![A-Za-z0-9])",
    re.IGNORECASE,
)
_PREFLOP_RANGE_TERM_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[2-9TJQKA]{2}(?:[sSoO])?\+?|[2-9]{2}\+?)(?![A-Za-z0-9])|"
    r"pocket pairs?|medium pairs?|small pairs?|suited broadway|suited connectors?|"
    r"weak kicker|trash hand",
    re.IGNORECASE,
)
_PREFLOP_STRATEGY_RE = re.compile(
    r"preflop|starting hand|limp|open[- ]?raise|(?:2|3|4|5)[- ]?bet|"
    r"calling range|entry range|check to see the flop|isolation raise",
    re.IGNORECASE,
)
_DIAGNOSTIC_IDENTIFIER_RE = re.compile(
    r"(?i)(?:(?:shown[_-])?"
    r"(?:agent|player|opponent|hand|fact|seat|rule|trajectory)"
    r"(?:[_:#-]+[A-Za-z0-9_-]+|\s+ID\s*[:=#-]?\s*[A-Za-z0-9_-]+|\s+\d+))"
)
_SECRET_LIKE_RE = re.compile(r"(?i)\b(?:sk-[A-Za-z0-9_-]{8,}|bearer\s+\S+)")
_NAMED_SECRET_RE = re.compile(
    r"(?ix)(?<![A-Za-z0-9_])"
    r"[\"']?(?:[A-Za-z0-9][A-Za-z0-9_.-]*[_-])?"
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password|"
    r"authorization)[\"']?\s*[:=]\s*"
    r"(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s,;}\]]+)"
)
_SECRET_FIELD_NAME_RE = re.compile(
    r"(?i)(?:^|[_-])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"secret|password|authorization)$"
)


def _contains_concrete_card_literal(text: str) -> bool:
    return bool(_CONCRETE_CARD_RE.search(text))


def _contains_secret_material(text: str) -> bool:
    return bool(_NAMED_SECRET_RE.search(text) or _SECRET_LIKE_RE.search(text))


def _contains_audit_narrative(text: str) -> bool:
    return bool(
        re.search(
            r"recent return|average return|net return|sample win rate|sample size|"
            r"this hand|action frequency|raise_rate|fold_rate|supporting_fact|"
            r"contradicting_fact|noise_fact|hand_|-fact-|last\s*\d+\s*hands",
            text,
            re.IGNORECASE,
        )
    )


def _is_opponent_archetype_rule(text: str) -> bool:
    return bool(
        re.search(
            r"(?:calling station|station|nit|passive|aggressive|TAG|LAG|"
            r"loose[- ]?passive|tight[- ]?aggressive|check[- ]?raise|"
            r"3\s*bet[- ]?fold|repeated c[- ]?bet|action line|range tendency)",
            text,
            re.IGNORECASE,
        )
    )


def _canonical_rule_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s.;,:()\[\]]+", "", normalized)


def _rules_are_high_confidence_near_duplicates(left: str, right: str) -> bool:
    """Detect only contained or almost-identical rules with the same action/context."""

    left_canonical = _canonical_rule_text(left)
    right_canonical = _canonical_rule_text(right)
    shorter, longer = sorted((left_canonical, right_canonical), key=len)
    if len(shorter) < 18:
        return False
    left_actions = _rule_action_signature(left)
    right_actions = _rule_action_signature(right)
    if not left_actions or left_actions != right_actions:
        return False
    left_context = _rule_context_anchors(left)
    right_context = _rule_context_anchors(right)
    if left_context and right_context and left_context != right_context:
        return False
    if shorter in longer and len(shorter) / max(1, len(longer)) >= 0.65:
        return True
    return SequenceMatcher(None, left_canonical, right_canonical).ratio() >= 0.90


def _rules_share_high_confidence_duplicate_fragment(left: str, right: str) -> bool:
    """Compare actionable sentence-level rules even when later exceptions differ.

    A long bullet may contain a core rule plus an exception.  Comparing the
    whole bullet let a different exception hide a duplicated core rule.  The
    fragment gate remains conservative: both fragments must be substantial,
    carry the same action signature and street/context anchors, and be nearly
    identical English prose.
    """

    left_fragments = _actionable_rule_fragments(left)
    right_fragments = _actionable_rule_fragments(right)
    for left_fragment in left_fragments:
        for right_fragment in right_fragments:
            left_actions = _rule_action_signature(left_fragment)
            right_actions = _rule_action_signature(right_fragment)
            if not left_actions or left_actions != right_actions:
                continue
            left_context = _rule_context_anchors(left_fragment)
            right_context = _rule_context_anchors(right_fragment)
            if left_context and right_context and left_context != right_context:
                continue
            left_canonical = _canonical_rule_text(left_fragment)
            right_canonical = _canonical_rule_text(right_fragment)
            shorter, longer = sorted((left_canonical, right_canonical), key=len)
            if len(shorter) < 18:
                continue
            if shorter in longer and len(shorter) / max(1, len(longer)) >= 0.72:
                return True
            if SequenceMatcher(None, left_canonical, right_canonical).ratio() >= 0.84:
                return True
    return False


def _actionable_rule_fragments(text: str) -> tuple[str, ...]:
    fragments: list[str] = []
    for raw_fragment in re.split(r"[.;]", text):
        candidates = [raw_fragment]
        contrast = re.search(r",(?:rather than|but|if|otherwise|conversely)", raw_fragment, re.I)
        if contrast is not None:
            candidates.append(raw_fragment[: contrast.start()])
        for candidate in candidates:
            fragment = candidate.strip(" ,")
            if (
                len(_canonical_rule_text(fragment)) >= 18
                and _rule_action_signature(fragment)
                and fragment not in fragments
            ):
                fragments.append(fragment)
    return tuple(fragments)


def _rule_action_signature(text: str) -> tuple[str, ...]:
    return tuple(
        marker
        for marker in (
            "fold",
            "check",
            "call",
            "raise",
            "bet",
            "jam",
            "defend",
            "pressure",
        )
        if marker in text
    )


def _rule_context_anchors(text: str) -> tuple[str, ...]:
    return tuple(
        marker
        for marker in (
            "starting hand",
            "preflop",
            "flop",
            "turn",
            "river",
            "BTN",
            "SB",
            "BB",
            "CO",
            "HJ",
            "UTG",
            "multiway pot",
            "heads-up pot",
            "calling station",
            "tight-aggressive",
            "loose-aggressive",
            "loose-passive",
        )
        if marker.lower() in text.lower()
    )


def _extract_section_bullets(body: str) -> dict[str, list[str]]:
    result = {heading: [] for heading in EXPERIENCE_REQUIRED_HEADINGS}
    current = ""
    for line in _normalize_markdown(body).splitlines():
        stripped = line.strip()
        if stripped in result:
            current = stripped
        elif current and stripped.startswith("- ") and stripped[2:].strip():
            result[current].append(stripped[2:].strip())
        elif (
            current
            and result[current]
            and stripped
            and stripped not in EXPERIENCE_ACCEPTED_EMPTY_SECTIONS
            and not stripped.startswith(("-", "#", "```"))
        ):
            result[current][-1] = f"{result[current][-1]} {stripped}"
    return result


def _experience_quality_summary(body: str) -> dict[str, object]:
    sections = _extract_section_bullets(body)
    findings = validate_experience_document_findings(
        body,
        max_chars=max(1, len(_normalize_markdown(body))),
        document_contract=EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION,
    )
    near_duplicate_count = 0
    for bullets in sections.values():
        for left_index, left in enumerate(bullets):
            left_canonical = _canonical_rule_text(left)
            for right in bullets[left_index + 1 :]:
                ratio = SequenceMatcher(None, left_canonical, _canonical_rule_text(right)).ratio()
                if ratio >= 0.72:
                    near_duplicate_count += 1
    return {
        "document_contract": EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION,
        "section_rule_counts": {
            heading.removeprefix("## "): len(bullets) for heading, bullets in sections.items()
        },
        "total_rule_count": sum(len(items) for items in sections.values()),
        "hard_limits": {
            "per_section": EXPERIENCE_MAX_RULES_PER_SECTION,
            "total": EXPERIENCE_MAX_RULES_TOTAL,
        },
        "finding_codes": dict(Counter(str(item["code"]) for item in findings)),
        "near_duplicate_advisory_count": near_duplicate_count,
    }


def _paper_experience_quality_summary(body: str) -> dict[str, object]:
    """Give the main-style reviser a small, non-gating consolidation hint."""

    sections = _extract_section_bullets(body)
    all_rules = [rule for rules in sections.values() for rule in rules]
    near_duplicate_count = sum(
        _rules_are_high_confidence_near_duplicates(left, right)
        for left_index, left in enumerate(all_rules)
        for right in all_rules[left_index + 1 :]
    )
    total_rule_count = len(all_rules)
    return {
        "section_rule_counts": {
            heading.removeprefix("## "): len(bullets) for heading, bullets in sections.items()
        },
        "total_rule_count": total_rule_count,
        "merge_before_add": True,
        "growth_limits": {
            "per_section": EXPERIENCE_PAPER_CONSOLIDATION_MAX_RULES_PER_SECTION,
            "total": EXPERIENCE_PAPER_CONSOLIDATION_MAX_RULES_TOTAL,
        },
        "high_confidence_near_duplicate_pairs": near_duplicate_count,
    }


def _proposal_summary(current_body: str, proposed_body: str) -> dict[str, object]:
    current_sections = _extract_section_bullets(current_body)
    proposed_sections = _extract_section_bullets(proposed_body)
    current_section_counts = {
        heading.removeprefix("## "): len(rules) for heading, rules in current_sections.items()
    }
    proposed_section_counts = {
        heading.removeprefix("## "): len(rules) for heading, rules in proposed_sections.items()
    }
    section_rule_deltas = {
        label: proposed_section_counts[label] - current_section_counts[label]
        for label in current_section_counts
    }
    current_total = sum(current_section_counts.values())
    proposed_total = sum(proposed_section_counts.values())
    threshold_hits: list[dict[str, object]] = []
    if (
        proposed_total > EXPERIENCE_PAPER_CONSOLIDATION_MAX_RULES_TOTAL
        and proposed_total > current_total
    ):
        threshold_hits.append(
            {
                "scope": "document",
                "threshold": EXPERIENCE_PAPER_CONSOLIDATION_MAX_RULES_TOTAL,
                "current": current_total,
                "proposed": proposed_total,
            }
        )
    for label in current_section_counts:
        current_count = current_section_counts[label]
        proposed_count = proposed_section_counts[label]
        if (
            proposed_count > EXPERIENCE_PAPER_CONSOLIDATION_MAX_RULES_PER_SECTION
            and proposed_count > current_count
        ):
            threshold_hits.append(
                {
                    "scope": "section",
                    "section": label,
                    "threshold": EXPERIENCE_PAPER_CONSOLIDATION_MAX_RULES_PER_SECTION,
                    "current": current_count,
                    "proposed": proposed_count,
                }
            )
    current_rules = Counter(
        _canonical_rule_text(rule) for rules in current_sections.values() for rule in rules
    )
    proposed_rules = Counter(
        _canonical_rule_text(rule) for rules in proposed_sections.values() for rule in rules
    )
    return {
        # Backward-compatible aliases describe the proposed document.
        "section_rule_counts": proposed_section_counts,
        "total_rule_count": proposed_total,
        "current_section_rule_counts": current_section_counts,
        "proposed_section_rule_counts": proposed_section_counts,
        "section_rule_deltas": section_rule_deltas,
        "current_total_rule_count": current_total,
        "proposed_total_rule_count": proposed_total,
        "net_rule_delta": proposed_total - current_total,
        "added_rule_count": sum((proposed_rules - current_rules).values()),
        "removed_rule_count": sum((current_rules - proposed_rules).values()),
        "consolidation_thresholds": {
            "per_section": EXPERIENCE_PAPER_CONSOLIDATION_MAX_RULES_PER_SECTION,
            "total": EXPERIENCE_PAPER_CONSOLIDATION_MAX_RULES_TOTAL,
        },
        "consolidation_threshold_triggered": bool(threshold_hits),
        "consolidation_threshold_hits": threshold_hits,
    }


def _paper_merge_before_add_findings(
    current_body: str,
    proposed_body: str,
) -> list[dict[str, object]]:
    """Reject only net growth beyond mature-paper consolidation thresholds."""

    summary = _proposal_summary(current_body, proposed_body)
    if not summary["consolidation_threshold_triggered"]:
        return []
    hits = summary["consolidation_threshold_hits"]
    scopes = ", ".join(
        str(item.get("section", item.get("scope", "document")))
        for item in hits
        if isinstance(item, dict)
    )
    return [
        _validation_finding(
            "paper_merge_before_add_violation",
            "mature paper experience must merge before further rule growth; "
            f"triggered scopes: {scopes or 'document'}",
        )
    ]


def summarize_experience_proposal(current_body: str, proposed_body: str) -> dict[str, object]:
    """Expose the body-free proposal delta summary to composite memory mechanisms."""

    return _proposal_summary(current_body, proposed_body)


def _text_fingerprint(text: str) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
    }


def _sanitize_diagnostic_text(text: str, forbidden_ids: set[str]) -> str:
    sanitized = text
    for identifier in sorted(forbidden_ids, key=len, reverse=True):
        if identifier:
            sanitized = sanitized.replace(identifier, "<ID>")
    sanitized = _DIAGNOSTIC_IDENTIFIER_RE.sub("<ID>", sanitized)
    sanitized = _NAMED_SECRET_RE.sub("<SECRET>", sanitized)
    sanitized = _SECRET_LIKE_RE.sub("<SECRET>", sanitized)
    sanitized = _CONCRETE_CARD_RE.sub("<specific-card>", sanitized)
    return re.sub(r"\s+", " ", sanitized).strip()


def _sanitize_diagnostic_excerpt(
    text: str,
    forbidden_ids: set[str],
    *,
    max_chars: int = 160,
) -> str:
    return _sanitize_diagnostic_text(text, forbidden_ids)[:max_chars]


def _sanitized_payload_value(
    value: object,
    *,
    forbidden_ids: set[str],
) -> object:
    """Build a canonical JSON-safe value without retaining IDs or credentials."""

    if isinstance(value, dict):
        items: list[list[object]] = []
        for raw_key, raw_value in sorted(value.items(), key=lambda item: str(item[0])):
            key = str(raw_key)
            if _SECRET_FIELD_NAME_RE.search(key):
                items.append(["<SECRET_FIELD>", "<SECRET>"])
                continue
            items.append(
                [
                    _sanitize_diagnostic_text(key, forbidden_ids),
                    _sanitized_payload_value(
                        raw_value,
                        forbidden_ids=forbidden_ids,
                    ),
                ]
            )
        return {"mapping": items}
    if isinstance(value, list):
        return [_sanitized_payload_value(item, forbidden_ids=forbidden_ids) for item in value]
    if isinstance(value, str):
        return _sanitize_diagnostic_text(value, forbidden_ids)
    if value is None or type(value) in {bool, int, float}:
        return value
    return _sanitize_diagnostic_text(str(value), forbidden_ids)


def _structured_patch_payload_diagnostic(
    payload: dict[str, object],
    *,
    forbidden_ids: set[str],
) -> dict[str, object]:
    """Fingerprint a rejected JSON patch without persisting its prose or IDs."""

    canonical = json.dumps(
        _sanitized_payload_value(payload, forbidden_ids=forbidden_ids),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = _text_fingerprint(canonical)
    raw_operation = payload.get("operation")
    operation = str(raw_operation) if raw_operation in {"keep", "patch"} else "<INVALID_OPERATION>"
    raw_patches = payload.get("patches")
    patch_count = len(raw_patches) if isinstance(raw_patches, list) else None
    sections: list[str] = []
    if isinstance(raw_patches, list):
        for raw_patch in raw_patches:
            if not isinstance(raw_patch, dict):
                continue
            raw_section = raw_patch.get("section")
            section = (
                str(raw_section)
                if raw_section in EXPERIENCE_SECTION_LABELS
                else "<INVALID_SECTION>"
            )
            if section not in sections:
                sections.append(section)
    return {
        "rejected_payload_sha256": fingerprint["sha256"],
        "rejected_payload_chars": fingerprint["chars"],
        "rejected_payload_summary": {
            "operation": operation,
            "patch_count": patch_count,
            "sections": sections,
        },
    }


def experience_failure_diagnostic(value: object) -> dict[str, object]:
    """Return a compact, redacted failure descriptor safe for persistent artifacts."""

    raw = str(value)
    fingerprint = _text_fingerprint(raw)
    prefix = raw.split(":", 1)[0].split(";", 1)[0].strip()
    code = re.sub(r"[^\w.-]+", "_", prefix, flags=re.UNICODE).strip("_")[:80]
    return {
        "code": code or "experience_revision_failure",
        "preview": _sanitize_diagnostic_excerpt(raw, set(), max_chars=240),
        "sha256": fingerprint["sha256"],
        "chars": fingerprint["chars"],
    }


def _compact_provider_diagnostics(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, object] = {}
    if value.get("schema_version") == "openai_compatible_http_calls_v1":
        result["schema_version"] = "openai_compatible_http_calls_v1"
    if value.get("token_accounting") == "per_http_call_v1":
        result["token_accounting"] = "per_http_call_v1"
    identity_schema = value.get("response_model_identity_schema_version")
    if identity_schema == "openai_compatible_response_model_identity_v1":
        result["response_model_identity_schema_version"] = identity_schema
    identity_source = value.get("response_model_identity_source")
    if identity_source == "chat_completions_top_level_model_self_report":
        result["response_model_identity_source"] = identity_source
    if "response_model_identity_cryptographic_attestation" in value:
        result["response_model_identity_cryptographic_attestation"] = bool(
            value["response_model_identity_cryptographic_attestation"]
        )
    identity_policy = value.get("response_model_identity_policy")
    if identity_policy in {
        "off",
        "strict_exact",
        "if_present_allowlisted_alias",
    }:
        result["response_model_identity_policy"] = identity_policy
    for key in ("requested_model", "response_model"):
        if key not in value:
            continue
        raw_model = value.get(key)
        if raw_model is None:
            result[key] = None
        elif _safe_provider_model_identity(raw_model):
            result[key] = raw_model
    report_status = value.get("response_model_report_status")
    if report_status in {
        "unreported",
        "reported_valid",
        "invalid_reported_value",
        "no_completed_response",
    }:
        result["response_model_report_status"] = report_status
    identity_statuses = {
        "not_available",
        "not_checked",
        "unreported",
        "reported_unverified",
        "reported_invalid_unverified",
        "verified_exact",
        "verified_allowlisted_alias",
        "rejected_unreported",
        "rejected_invalid_reported_value",
        "rejected_missing_requested_model",
        "rejected_exact_mismatch",
        "rejected_cross_family",
        "rejected_not_allowlisted",
    }
    identity_status = value.get("response_model_identity_status")
    if identity_status in identity_statuses:
        result["response_model_identity_status"] = identity_status
    for key in (
        "response_model_identity_verified",
        "response_model_exact_match",
    ):
        if key in value:
            result[key] = bool(value[key])
    raw_status_counts = value.get("response_model_identity_status_counts")
    if isinstance(raw_status_counts, dict):
        status_counts = {
            str(key): raw
            for key, raw in raw_status_counts.items()
            if key in identity_statuses and type(raw) is int and raw >= 0
        }
        result["response_model_identity_status_counts"] = status_counts
    raw_observed_models = value.get("observed_response_models")
    if isinstance(raw_observed_models, list):
        result["observed_response_models"] = [
            raw_model
            for raw_model in raw_observed_models[:16]
            if _safe_provider_model_identity(raw_model)
        ]
    if "finish_reason" in value:
        raw_finish_reason = value.get("finish_reason")
        if raw_finish_reason is None:
            result["finish_reason"] = None
        else:
            finish_reason = str(raw_finish_reason)
            result["finish_reason"] = _safe_provider_finish_reason(finish_reason)
    for key in ("reasoning_present",):
        if key in value:
            result[key] = bool(value[key])
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "retry_count",
        "outer_attempt_count",
        "http_call_count",
        "completed_http_call_count",
        "failed_http_call_count",
        "json_repair_call_count",
        "usage_missing_call_count",
        "prompt_tokens_total_observed",
        "prompt_tokens_max_observed",
    ):
        if key not in value:
            continue
        raw = value.get(key)
        if raw is None:
            result[key] = None
        elif type(raw) is int and raw >= 0:
            result[key] = raw
    raw_calls = value.get("calls")
    if isinstance(raw_calls, list):
        calls: list[dict[str, object]] = []
        for raw_call in raw_calls[:32]:
            if not isinstance(raw_call, dict):
                continue
            call: dict[str, object] = {}
            for key in ("call_index", "outer_attempt"):
                raw = raw_call.get(key)
                if type(raw) is int and raw >= 1:
                    call[key] = raw
            stage = raw_call.get("stage")
            if stage in {"primary", "json_repair"}:
                call["stage"] = stage
            status = raw_call.get("status")
            if status in {"completed", "failed"}:
                call["status"] = status
            finish_reason = raw_call.get("finish_reason")
            if finish_reason is None:
                call["finish_reason"] = None
            else:
                call["finish_reason"] = _safe_provider_finish_reason(finish_reason)
            for key in ("reasoning_present", "usage_available"):
                if key in raw_call:
                    call[key] = bool(raw_call[key])
            for key in ("prompt_tokens", "completion_tokens"):
                raw = raw_call.get(key)
                call[key] = raw if type(raw) is int and raw >= 0 else None
            for key in (
                "estimated_prompt_tokens_upper_bound",
                "reserved_output_tokens",
                "resolved_context_limit",
            ):
                raw = raw_call.get(key)
                if type(raw) is int and raw >= 0:
                    call[key] = raw
            json_status = raw_call.get("json_status")
            if json_status in {"not_checked", "not_available", "valid", "invalid"}:
                call["json_status"] = json_status
            budget_status = raw_call.get("budget_status")
            if budget_status in {
                "not_checked",
                "not_available",
                "within_budget",
                "usage_missing",
                "invalid_budget",
                "exceeded",
                "exceeded_advisory",
            }:
                call["budget_status"] = budget_status
            failure_code = raw_call.get("failure_code")
            if failure_code in {
                None,
                "provider_error",
                "unexpected_error",
                "response_model_unreported",
                "response_model_invalid_report",
                "requested_model_invalid",
                "response_model_exact_mismatch",
                "response_model_cross_family",
                "response_model_not_allowlisted",
            }:
                call["failure_code"] = failure_code
            call_identity_schema = raw_call.get("response_model_identity_schema_version")
            if call_identity_schema == "openai_compatible_response_model_identity_v1":
                call["response_model_identity_schema_version"] = call_identity_schema
            call_identity_source = raw_call.get("response_model_identity_source")
            if call_identity_source == "chat_completions_top_level_model_self_report":
                call["response_model_identity_source"] = call_identity_source
            call_identity_policy = raw_call.get("response_model_identity_policy")
            if call_identity_policy in {
                "off",
                "strict_exact",
                "if_present_allowlisted_alias",
            }:
                call["response_model_identity_policy"] = call_identity_policy
            for key in ("requested_model", "response_model"):
                if key not in raw_call:
                    continue
                raw_model = raw_call.get(key)
                if raw_model is None:
                    call[key] = None
                elif _safe_provider_model_identity(raw_model):
                    call[key] = raw_model
            call_report_status = raw_call.get("response_model_report_status")
            if call_report_status in {
                "unreported",
                "reported_valid",
                "invalid_reported_value",
                "no_completed_response",
            }:
                call["response_model_report_status"] = call_report_status
            call_identity_status = raw_call.get("response_model_identity_status")
            if call_identity_status in identity_statuses:
                call["response_model_identity_status"] = call_identity_status
            for key in (
                "response_model_identity_verified",
                "response_model_exact_match",
            ):
                if key in raw_call:
                    call[key] = bool(raw_call[key])
            calls.append(call)
        result["calls"] = calls
    return result


def _safe_provider_model_identity(value: object) -> bool:
    """Accept only bounded model identifiers without common credential markers."""

    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,199}", value
    ):
        return False
    return (
        re.search(
            r"(?:^|[._:/+\-])(?:bearer|api[_-]?key|secret|access[_-]?token|sk-)",
            value,
            flags=re.IGNORECASE,
        )
        is None
    )


def _safe_provider_finish_reason(value: object) -> str:
    """Allow only documented finish reason labels in persisted diagnostics."""

    text = str(value or "unknown")
    return (
        text
        if text
        in {
            "stop",
            "length",
            "content_filter",
            "tool_calls",
            "function_call",
            "unknown",
            "other",
        }
        else "other"
    )


def _compact_validation_findings(
    findings: list[dict[str, object]],
    *,
    forbidden_ids: set[str],
) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    for finding in findings[:32]:
        excerpt = str(finding.get("excerpt", ""))
        fingerprint = _text_fingerprint(excerpt)
        compact.append(
            {
                "code": str(finding.get("code", "invalid_document")),
                "message": str(finding.get("message", "invalid document"))[:240],
                "section": str(finding.get("section", "")),
                "bullet_index": finding.get("bullet_index"),
                "peer_section": str(finding.get("peer_section", "")),
                "peer_index": finding.get("peer_index"),
                "excerpt": _sanitize_diagnostic_excerpt(excerpt, forbidden_ids, max_chars=120),
                "excerpt_sha256": fingerprint["sha256"],
                "excerpt_chars": fingerprint["chars"],
            }
        )
    return compact


def _provider_prompt_token_counts(payload: dict[str, object]) -> list[int]:
    """Read every completed HTTP call count, with legacy top-level fallback."""

    diagnostics = payload.get("_provider_diagnostics", {})
    if not isinstance(diagnostics, dict):
        return []
    calls = diagnostics.get("calls")
    if isinstance(calls, list):
        return [
            int(call["prompt_tokens"])
            for call in calls
            if isinstance(call, dict)
            and call.get("status") == "completed"
            and type(call.get("prompt_tokens")) is int
            and int(call["prompt_tokens"]) >= 0
        ]
    value = diagnostics.get("prompt_tokens")
    return [value] if type(value) is int and value >= 0 else []


def _provider_diagnostic_payload_from_exception(
    error: Exception,
) -> dict[str, object] | None:
    """Extract only ProviderError's structured diagnostics for fail-closed persistence."""

    diagnostics = getattr(error, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        return None
    return {"_provider_diagnostics": dict(diagnostics)}


def _redact_agent_ids(text: str, known_ids: set[str], self_id: str) -> str:
    """Replace stable player identifiers before they enter the revision prompt."""

    redacted = text
    for agent_id in sorted(known_ids, key=len, reverse=True):
        if not agent_id:
            continue
        replacement = "self" if agent_id == self_id else "opponent"
        redacted = redacted.replace(agent_id, replacement)
    return redacted


def _sanitize_public_action(action: dict[str, object], labels: dict[str, str]) -> dict[str, object]:
    """Keep only public action fields and replace stable player identifiers."""

    raw_agent_id = str(action.get("agent_id", action.get("player_id", "")))
    return {
        "phase": action.get("phase"),
        "actor": labels.get(raw_agent_id, "opponent"),
        "action_type": action.get("action_type", action.get("action")),
        "amount": action.get("amount"),
        "committed": action.get("committed"),
        "to_call": action.get("to_call"),
        "pot_before": action.get("pot_before"),
        "pot_after": action.get("pot_after"),
        "effective_raise": bool(action.get("effective_raise", False)),
        "last_raise_size": action.get("last_raise_size"),
    }


def _position_label(observation: AgentObservation) -> str:
    """Use the canonical public position truth shared with factual retrieval."""

    return public_position_label(observation)


def _compact_decision_facts(facts: object, *, phase: str) -> dict[str, object]:
    """Keep only deterministic facts needed for transferable rule calibration."""

    if not isinstance(facts, dict):
        return {}
    draw = facts.get("draw", {})
    draw = draw if isinstance(draw, dict) else {}
    hero_draw = draw.get("hero_draw", {})
    hero_draw = hero_draw if isinstance(hero_draw, dict) else {}
    call = facts.get("call", {})
    call = call if isinstance(call, dict) else {}
    result: dict[str, object] = {"made_hand_class": facts.get("made_hand_class")}
    unit_is_explicit = facts.get("chip_unit") == "chips"
    if unit_is_explicit:
        result["effective_stack_chips"] = facts.get("effective_stack_chips")
        for key in ("effective_stack_big_blinds",):
            if key in facts:
                result[key] = facts.get(key)
    elif "effective_stack" in facts:
        result["effective_stack"] = facts.get("effective_stack")
    draw_outs = hero_draw.get("outs")
    if (
        bool(hero_draw.get("flush_draw"))
        or bool(hero_draw.get("straight_draw"))
        or (type(draw_outs) is int and draw_outs > 0)
    ):
        result["hero_draw"] = {
            "flush_draw": bool(hero_draw.get("flush_draw")),
            "straight_draw": bool(hero_draw.get("straight_draw")),
            "nominal_outs_not_clean_equity": draw_outs,
            "cards_to_come": draw.get("cards_to_come"),
        }
    call_cost = call.get("call_cost")
    if type(call_cost) is int and call_cost > 0:
        compact_call: dict[str, object] = {
            "required_equity": call.get("required_equity"),
            "stack_fraction": call.get("stack_fraction"),
            "is_all_in": bool(call.get("is_all_in")),
        }
        if unit_is_explicit:
            compact_call.update(
                {
                    "cost_chips": call.get("call_cost_chips"),
                    "contestable_pot_before_call_chips": call.get(
                        "contestable_pot_before_call_chips"
                    ),
                }
            )
            for key in (
                "call_cost_big_blinds",
                "contestable_pot_before_call_big_blinds",
            ):
                if key in call:
                    compact_call[key] = call.get(key)
        result["call"] = compact_call
    if phase in {"flop", "turn", "river"}:
        result["spr"] = facts.get("spr")
        if type(call_cost) is int and call_cost > 0:
            result["spr_after_call"] = facts.get("spr_after_call")
    multiway_players = facts.get("multiway_players")
    if type(multiway_players) is int and multiway_players > 2:
        result["multiway_players"] = multiway_players
    return result


def _compact_public_actions(actions: object) -> list[dict[str, object]]:
    """Represent the public action sequence once, without repeated pot bookkeeping."""

    if not isinstance(actions, list):
        return []
    result: list[dict[str, object]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        compact = {
            "phase": action.get("phase"),
            "actor": action.get("actor"),
            "action_type": action.get("action_type"),
        }
        if action.get("amount") is not None:
            compact["amount_chips"] = action.get("amount")
        committed = action.get("committed")
        if type(committed) is int and committed > 0:
            compact["committed_chips"] = committed
        result.append(compact)
    return result


def compact_revision_recap_payload(
    trajectory_payload: dict[str, object],
    *,
    trajectory_id: str,
) -> dict[str, object]:
    """Deterministically project a persisted visible trajectory into revision evidence.

    The artifact auditor imports this pure boundary and recomputes the recap from the
    full body-free trajectory.  A producer and its audit therefore cannot silently
    drift to two merely similar field selections.
    """

    raw_decisions = trajectory_payload.get("decisions", [])
    if not isinstance(raw_decisions, list):
        raise ValueError("trajectory payload decisions must be a list")
    if raw_decisions:
        first = raw_decisions[0]
        if not isinstance(first, dict):
            raise ValueError("trajectory payload decision must be an object")
        first_position = first.get("position")
        first_hole_cards = first.get("hole_cards", [])
        for item in raw_decisions[1:]:
            if not isinstance(item, dict):
                raise ValueError("trajectory payload decision must be an object")
            if item.get("position") != first_position:
                raise ValueError("trajectory decision positions are inconsistent")
            if item.get("hole_cards", []) != first_hole_cards:
                raise ValueError("trajectory decision hole cards are inconsistent")
    decisions: list[dict[str, object]] = []
    for item in raw_decisions:
        if not isinstance(item, dict):
            raise ValueError("trajectory payload decision must be an object")
        facts = item.get("deterministic_decision_facts", {})
        committed = item.get("committed_action", {})
        compact_committed_action: dict[str, object] = {
            "action_type": committed.get("action_type") if isinstance(committed, dict) else None
        }
        if isinstance(committed, dict) and committed.get("amount_chips") is not None:
            compact_committed_action["amount_chips"] = committed.get("amount_chips")
        if isinstance(committed, dict) and committed.get("amount_big_blinds") is not None:
            compact_committed_action["amount_big_blinds"] = committed.get("amount_big_blinds")
        compact_decision: dict[str, object] = {
            "phase": item.get("phase"),
            "board": item.get("board", []),
            "committed_action": compact_committed_action,
            "declared_intent_not_ground_truth": item.get("declared_intent_not_ground_truth", ""),
            "deterministic_decision_facts": _compact_decision_facts(
                facts,
                phase=str(item.get("phase", "")),
            ),
        }
        if item.get("chip_unit") == "chips":
            compact_decision.update(
                {
                    "pot_chips": item.get("pot_chips"),
                    "current_bet_chips": item.get("current_bet_chips"),
                    "to_call_chips": item.get("to_call_chips"),
                    "stack_chips": item.get("stack_chips"),
                }
            )
            for key in (
                "pot_big_blinds",
                "current_bet_big_blinds",
                "to_call_big_blinds",
                "stack_big_blinds",
            ):
                if key in item:
                    compact_decision[key] = item.get(key)
        guard = item.get("action_guard", {})
        if isinstance(guard, dict) and (
            bool(guard.get("repaired")) or bool(guard.get("fallback_used"))
        ):
            compact_decision["action_guard"] = guard
            compact_decision["raw_action_before_guard"] = item.get("raw_action", {})
        decisions.append(compact_decision)

    outcome = trajectory_payload.get("outcome", {})
    outcome = outcome if isinstance(outcome, dict) else {}
    compact_outcome: dict[str, object] = {
        "net_reward_chips": outcome.get("final_reward"),
        "final_stack_chips": outcome.get("final_stack_chips", outcome.get("final_stack")),
    }
    if outcome.get("chip_unit") == "chips":
        for key in (
            "invested_chips",
            "invested_big_blinds",
            "won_amount_chips",
            "won_amount_big_blinds",
            "final_reward_big_blinds",
            "final_stack_big_blinds",
        ):
            if key in outcome:
                compact_outcome[key] = outcome.get(key)
    elif outcome.get("won_amount") is not None:
        compact_outcome["won_amount"] = outcome.get("won_amount")
    for key in ("final_public_board", "showdown_visible_cards"):
        value = outcome.get(key)
        if value not in (None, [], {}):
            compact_outcome[key] = value
    first_decision = raw_decisions[0] if raw_decisions else {}
    compact_payload: dict[str, object] = {
        "trajectory_id": trajectory_id,
        "hero_position": (
            first_decision.get("position") if isinstance(first_decision, dict) else None
        ),
        "hero_hole_cards": (
            first_decision.get("hole_cards", []) if isinstance(first_decision, dict) else []
        ),
        "hero_decisions": decisions,
        "public_action_sequence": _compact_public_actions(
            trajectory_payload.get("public_action_sequence", [])
        ),
        "outcome": compact_outcome,
    }
    if trajectory_payload.get("chip_unit") == "chips":
        compact_payload.update(
            {
                "small_blind_amount_chips": trajectory_payload.get("small_blind_amount"),
                "big_blind_amount_chips": trajectory_payload.get("big_blind_amount"),
            }
        )
    return compact_payload


def render_main_style_revision_window(
    recap_payloads: list[dict[str, object]],
) -> str:
    """Render compact evidence in the human-readable layout used by main.

    The compact dictionaries remain the canonical audit payload.  This renderer
    changes only what the revising LLM reads: complete public actions, the
    committed action and contemporaneous intent, final chips, and visible
    showdown information are presented as concise human-readable hand recaps instead
    of a dense JSON array.
    """

    return "\n\n".join(_render_main_style_revision_recap(payload) for payload in recap_payloads)


def paper_revision_behavior_digest(
    recap_payloads: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize observed action coverage without treating behavior as a label."""

    actions: Counter[str] = Counter()
    phase_actions: dict[str, Counter[str]] = {}
    positions: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    decision_count = 0
    for recap in recap_payloads:
        position = str(recap.get("hero_position") or "unknown")
        positions[position] += 1
        outcome = recap.get("outcome", {})
        reward = outcome.get("net_reward_chips") if isinstance(outcome, dict) else None
        outcome_counts[
            "win"
            if isinstance(reward, (int, float)) and reward > 0
            else "loss"
            if isinstance(reward, (int, float)) and reward < 0
            else "draw"
        ] += 1
        decisions = recap.get("hero_decisions", [])
        if not isinstance(decisions, list):
            continue
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            committed = decision.get("committed_action", {})
            if not isinstance(committed, dict):
                continue
            raw_action = str(committed.get("action_type", "unknown"))
            phase = str(decision.get("phase", "unknown"))
            to_call = decision.get("to_call_chips")
            semantic_action = raw_action
            if raw_action == "raise" and phase in {"flop", "turn", "river"}:
                semantic_action = "bet" if to_call == 0 else "raise"
            elif (
                raw_action == "call"
                and phase == "preflop"
                and position == "BTN/SB"
                and isinstance(to_call, (int, float))
                and to_call > 0
            ):
                semantic_action = "complete_small_blind"
            actions[semantic_action] += 1
            phase_actions.setdefault(phase, Counter())[semantic_action] += 1
            decision_count += 1
    check_count = actions.get("check", 0)
    check_share = round(check_count / decision_count, 4) if decision_count else None
    warnings: list[str] = []
    if isinstance(check_share, float) and check_share >= 0.65:
        warnings.append(
            "check_copy_bias: high observed check share is behavior coverage, not evidence "
            "that checking is the best default"
        )
    proactive_count = actions.get("bet", 0) + actions.get("raise", 0)
    if decision_count >= 10 and proactive_count / decision_count < 0.1:
        warnings.append(
            "low_proactive_action_coverage: the window cannot justify a general policy that "
            "removes value betting, bluffing, or equity denial"
        )
    return {
        "schema_version": EXPERIENCE_BEHAVIOR_DIGEST_VERSION,
        "hand_count": len(recap_payloads),
        "decision_count": decision_count,
        "committed_action_counts": dict(sorted(actions.items())),
        "phase_action_counts": {
            phase: dict(sorted(counts.items())) for phase, counts in sorted(phase_actions.items())
        },
        "hero_position_hand_counts": dict(sorted(positions.items())),
        "realized_outcome_counts": dict(sorted(outcome_counts.items())),
        "observed_check_share": check_share,
        "coverage_warnings": warnings,
        "evidence_semantics": (
            "behavior_not_strategy: counts report what the agent did; they do not reveal the "
            "counterfactual EV of actions it did not choose"
        ),
    }


def _render_main_style_revision_recap(payload: dict[str, object]) -> str:
    trajectory_id = str(payload.get("trajectory_id", ""))
    outcome = payload.get("outcome", {})
    outcome = outcome if isinstance(outcome, dict) else {}
    net_reward = outcome.get("net_reward_chips")
    outcome_label = (
        "win"
        if isinstance(net_reward, (int, float)) and net_reward > 0
        else "loss"
        if isinstance(net_reward, (int, float)) and net_reward < 0
        else "draw"
    )
    won_amount = outcome.get("won_amount_chips", outcome.get("won_amount", "unknown"))
    hero_position = str(payload.get("hero_position") or "unknown")
    position_note = (
        " (HU: acts first preflop and second postflop)"
        if hero_position == "BTN/SB"
        else " (HU: acts after BTN/SB preflop and first postflop)"
        if hero_position == "BB"
        else ""
    )
    lines = [
        f"--- hand {trajectory_id} (outcome={outcome_label}, won={won_amount}) ---",
        f"=== hand {trajectory_id} ===",
        f"Position: {hero_position}{position_note}",
        "Hero hole cards: " + json.dumps(payload.get("hero_hole_cards", []), ensure_ascii=False),
    ]
    final_board = outcome.get("final_public_board")
    if final_board is None:
        decisions = payload.get("hero_decisions", [])
        decision_rows = decisions if isinstance(decisions, list) else []
        last_decision = decision_rows[-1] if decision_rows else {}
        final_board = last_decision.get("board", []) if isinstance(last_decision, dict) else []
    lines.extend(
        [
            "Public board: " + json.dumps(final_board, ensure_ascii=False),
            "",
            "Complete action sequence:",
        ]
    )
    public_actions = payload.get("public_action_sequence", [])
    action_rows = public_actions if isinstance(public_actions, list) else []
    if not action_rows:
        lines.append("  (none)")
    rendered_hero_actions: list[tuple[str, str]] = []
    for action_index, action in enumerate(action_rows):
        if not isinstance(action, dict):
            continue
        amount = action.get("amount_chips")
        amount_text = f" {amount} chips" if amount is not None else ""
        committed = action.get("committed_chips")
        committed_text = f" (committed {committed} chips)" if committed is not None else ""
        raw_action_type = str(action.get("action_type", ""))
        semantic_action_type = _main_style_semantic_action_type(action_rows, action_index)
        action_text = semantic_action_type
        if semantic_action_type != raw_action_type:
            action_text += f" (engine_action={raw_action_type})"
        if str(action.get("actor", "")) == "self":
            rendered_hero_actions.append((str(action.get("phase", "")), action_text))
        lines.append(
            f"  [{action.get('phase')}] {action.get('actor')} "
            f"{action_text}{amount_text}{committed_text}"
        )
    lines.extend(["", "Decision-time self-reports:"])
    decisions = payload.get("hero_decisions", [])
    decision_rows = decisions if isinstance(decisions, list) else []
    hero_action_cursor = 0
    for decision in decision_rows:
        if not isinstance(decision, dict):
            continue
        committed_action = decision.get("committed_action", {})
        committed_action = committed_action if isinstance(committed_action, dict) else {}
        amount = committed_action.get("amount_chips")
        amount_text = f" {amount} chips" if amount is not None else ""
        raw_decision_action = str(committed_action.get("action_type", ""))
        decision_action_text = raw_decision_action
        if hero_action_cursor < len(rendered_hero_actions):
            action_phase, rendered_action = rendered_hero_actions[hero_action_cursor]
            if action_phase == str(decision.get("phase", "")):
                decision_action_text = rendered_action
                hero_action_cursor += 1
        lines.append(
            f"  [{decision.get('phase')}] intent="
            f"{decision.get('declared_intent_not_ground_truth', '')!r}  "
            f"action={decision_action_text}{amount_text}"
        )
        visible_state = {
            key: decision.get(key)
            for key in (
                "board",
                "pot_chips",
                "to_call_chips",
                "stack_chips",
                "pot_big_blinds",
                "to_call_big_blinds",
                "stack_big_blinds",
            )
            if decision.get(key) is not None
        }
        deterministic_facts = decision.get("deterministic_decision_facts", {})
        if isinstance(deterministic_facts, dict):
            visible_state["deterministic_decision_facts"] = deterministic_facts
        lines.append(
            "    Objective state: "
            + json.dumps(visible_state, ensure_ascii=False, separators=(",", ":"))
        )
    lines.extend(
        [
            "",
            (
                f"Outcome: stack={outcome.get('final_stack_chips')} chips, "
                f"invested={outcome.get('invested_chips')} chips, "
                f"returned from pot={won_amount} chips, net reward={net_reward} chips"
            ),
            "Visible opponent showdown cards: "
            + json.dumps(outcome.get("showdown_visible_cards", {}), ensure_ascii=False),
        ]
    )
    return "\n".join(lines)


def _main_style_semantic_action_type(action_rows: list[object], action_index: int) -> str:
    """Render engine call/raise tokens as their observable poker action."""

    action = action_rows[action_index]
    if not isinstance(action, dict):
        return "unknown"
    phase = str(action.get("phase", ""))
    action_type = str(action.get("action_type", ""))
    prior_same_street = [
        row
        for row in action_rows[:action_index]
        if isinstance(row, dict) and str(row.get("phase", "")) == phase
    ]
    if phase == "preflop" and action_type == "call" and not prior_same_street:
        return "complete_small_blind"
    if phase == "preflop" and action_type == "raise" and not prior_same_street:
        return "open_raise"
    if phase in {"flop", "turn", "river"} and action_type == "raise":
        prior_wager = any(
            str(row.get("action_type", "")) == "raise"
            for row in prior_same_street
            if isinstance(row, dict)
        )
        return "raise" if prior_wager else "bet"
    return action_type


def _semantic_repair_needs_recap(findings: list[object]) -> bool:
    """Replay only bounded recaps when a rejection depends on observed hand state."""

    evidence_code_fragments = (
        "action_sequence",
        "calibration_",
        "draw",
        "free_card",
        "hand_",
        "hu_",
        "made_hand",
        "outs",
        "position",
        "pot_odds",
        "spr",
        "stack_depth",
    )
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        code = str(finding.get("code", "")).lower()
        if any(fragment in code for fragment in evidence_code_fragments):
            return True
    return False


def _filter_evidence_ids(payload: dict[str, object], key: str, allowed_ids: set[str]) -> list[str]:
    values = payload.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value) in allowed_ids]
