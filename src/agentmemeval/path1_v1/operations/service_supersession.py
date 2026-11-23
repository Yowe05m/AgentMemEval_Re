"""One reviewed v2-to-v3 E0 resource-policy transition after a sealed rejection.

The original intent and attempt stay unchanged. This is not a scientific retry,
an arbitrary version reset, or permission to execute any JSON receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentmemeval.path1_v1.operations import e0_reconcile as er
from agentmemeval.path1_v1.operations.contracts import (
    canonical_intent_id,
    content_digest,
    validate_campaign_contract,
)
from agentmemeval.path1_v1.operations.service_profile import validate_service_profile
from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled

_ISSUER = object()
_KIND = "path1_service_resource_policy_supersession_v1"


def _require(value, message):
    if not value:
        raise SubmissionDisabled("service supersession: " + message)


def validate_transition(prior: dict, current: dict, old_config: dict, new_config: dict) -> None:
    """Structural policy only; this function does not issue a capability."""
    for contract in (prior, current):
        validate_campaign_contract(contract)
        _require(
            contract["phase"] == "E0"
            and contract["job"]["kind"] == "service_probe"
            and contract["job"]["attempt"] == 1
            and contract["job"]["retry_evidence"] is None
            and contract["not_for_paper"] is True,
            "only a first technical service unit may change its resource policy",
        )
    for key in (
        "campaign_id", "source_contract", "protocol_sha256", "seed_namespace", "cohort", "job"
    ):
        _require(prior[key] == current[key], "scientific/unit identity changed: " + key)
    for key in ("core_files", "dependency_lock", "model"):
        _require(
            prior["runtime_identity"][key] == current["runtime_identity"][key],
            "core/model/dependency identity changed: " + key,
        )
    validate_service_profile(old_config)
    validate_service_profile(new_config)
    _require(old_config["resource_profile_version"] == 2, "predecessor is not profile v2")
    expected = {
        **old_config,
        "resource_profile_version": 3,
        "memory_mib": None,
        "memory_policy": "site_default",
    }
    _require(new_config == expected, "change exceeds the exact v2-to-v3 memory-policy correction")
    _require(canonical_intent_id(prior) != canonical_intent_id(current), "new intent required")


@dataclass(frozen=True)
class ServiceSupersession:
    store_campaign_id: str
    store_directory: str
    campaign_id: str
    new_intent_id: str
    new_contract_sha256: str
    new_probe_request_sha256: str
    new_runtime_identity_sha256: str
    prior_intent_id: str
    prior_terminal_receipt_sha256: str
    prior_evidence_file_sha256: str
    prior_config_sha256: str
    new_config_sha256: str
    _issuer: Any = field(default=None, repr=False, compare=False)

    def payload(self) -> dict:
        _require(self._issuer is _ISSUER, "not issued from an authoritative rejection seal")
        return {
            "kind": _KIND,
            **{key: getattr(self, key) for key in self.__dataclass_fields__ if key != "_issuer"},
            "from_profile": 2,
            "to_profile": 3,
            "original_attempt_unchanged": True,
            "retry_eligible": False,
            "scientific_admission": False,
        }


def resolve_service_supersession(store, contract: dict, probe: dict, previous: dict):
    """Derive one transition from the original store and both exact Git identities."""
    from agentmemeval.path1_v1.operations.intent_executor import NAMESPACE, DurableIntentExecutor
    from agentmemeval.path1_v1.operations.submission_rejection import (
        RejectionTerminalResolution,
        resolve_rejection,
    )

    current_intent = canonical_intent_id(validate_campaign_contract(contract))
    _require(
        probe["intent_id"] == current_intent
        and probe["request_sha256"] == content_digest(
            {k: v for k, v in probe.items() if k != "request_sha256"}),
        "new probe binding changed",
    )
    snapshot = store.read_registry()
    records = snapshot.registry[NAMESPACE]["intents"]
    _require(records.get(previous["intent_id"]) == previous, "predecessor is not authoritative")
    _require(
        previous["status"] == "submit_unknown"
        and previous["job_id"] is None
        and previous["submitter_finished"] is True
        and "operational_supersession" not in previous,
        "predecessor is not the original never-accepted submission",
    )
    resolution = resolve_rejection(store, previous["intent_id"])
    _require(
        type(resolution) is RejectionTerminalResolution
        and resolution.classification == "site_policy_rejected_before_job"
        and resolution.retry_eligible is False,
        "exact typed site-policy rejection seal required",
    )
    resolution.validate(previous, store=store)
    prior = DurableIntentExecutor._stored_probe_contract(previous)
    prior_probe = er._json(er._path(previous["probe_request_path"]))
    _require(
        prior_probe["repository_root"] == probe["repository_root"]
        and prior_probe["model_root"] == probe["model_root"]
        and prior_probe["model_manifest_path"] == probe["model_manifest_path"]
        and prior_probe["launch"] == probe["launch"],
        "repository/model/interpreter/launch changed",
    )
    _require(
        er._path(prior_probe["output_root"]) != er._path(probe["output_root"]),
        "new technical unit requires an independent output root",
    )
    repository = er._path(probe["repository_root"])
    old_config = er._historical_identity(repository, prior["runtime_identity"], allow_model=True)
    new_config = er._historical_identity(repository, contract["runtime_identity"], allow_model=True)
    validate_transition(prior, contract, old_config, new_config)
    # Count actual source-bound service units, including claims of unknown status.
    siblings = []
    for intent, record in records.items():
        if intent == current_intent or record.get("probe_request_path") is None:
            continue
        other = DurableIntentExecutor._stored_probe_contract(record)
        if (other["campaign_id"] == contract["campaign_id"]
                and other["job"]["kind"] == "service_probe"):
            siblings.append(intent)
    _require(siblings == [previous["intent_id"]], "another service unit consumes this transition")
    return ServiceSupersession(
        store_campaign_id=store.campaign_id,
        store_directory=str(store.directory),
        campaign_id=contract["campaign_id"],
        new_intent_id=current_intent,
        new_contract_sha256=probe["contract_sha256"],
        new_probe_request_sha256=probe["request_sha256"],
        new_runtime_identity_sha256=content_digest(contract["runtime_identity"]),
        prior_intent_id=previous["intent_id"],
        prior_terminal_receipt_sha256=resolution.prior_terminal_receipt_sha256,
        prior_evidence_file_sha256=resolution.evidence_file_sha256,
        prior_config_sha256=prior["runtime_identity"]["config"]["sha256"],
        new_config_sha256=contract["runtime_identity"]["config"]["sha256"],
        _issuer=_ISSUER,
    )


def validate_stored_supersession(record: dict, store) -> None:
    """A saved JSON payload is provenance only and must be independently rebuilt."""
    from agentmemeval.path1_v1.operations.intent_executor import NAMESPACE, DurableIntentExecutor

    saved = record["operational_supersession"]
    _require(type(saved) is dict and saved.get("kind") == _KIND, "invalid persisted transition")
    records = store.read_registry().registry[NAMESPACE]["intents"]
    previous = records.get(saved.get("prior_intent_id"))
    _require(type(previous) is dict, "persisted predecessor missing")
    contract = DurableIntentExecutor._stored_probe_contract(record)
    probe = er._json(er._path(record["probe_request_path"]))
    expected = resolve_service_supersession(store, contract, probe, previous)
    _require(saved == expected.payload(), "persisted transition differs from original evidence")
