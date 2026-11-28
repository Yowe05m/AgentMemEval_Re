"""Complete-unit E0 retry after a sealed pre-HTTP CUDA driver failure.

This derives eligibility from original files and the authoritative failed claim;
it neither retries an intent nor grants scientific/runtime admission.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentmemeval.path1_v1.operations import e0_reconcile as er
from agentmemeval.path1_v1.operations import service_terminal as st
from agentmemeval.path1_v1.operations.contracts import (
    canonical_intent_id,
    content_digest,
    validate_campaign_contract,
)
from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled

_ISSUER = object()
ENVIRONMENT_FIELDS = frozenset(
    {
        "dependency_lock",
        "native_runtime",
        "python_executable",
        "python_sha256",
        "python_prefix",
        "service_python",
        "service_python_sha256",
        "service_python_prefix",
    }
)


def _require(value, message):
    if not value:
        raise SubmissionDisabled("service retry: " + message)


@dataclass(frozen=True)
class ServiceRetryEligibility:
    store_campaign_id: str
    store_directory: str
    prior_intent_id: str
    prior_contract_sha256: str
    prior_terminal_receipt_sha256: str
    prior_evidence_file_sha256: str
    driver_log_sha256: str
    driver_api_version: int
    _issuer: Any = field(default=None, repr=False, compare=False)

    def payload(self):
        _require(self._issuer is _ISSUER, "untrusted eligibility object")
        return {
            "kind": "path1_e0_service_driver_retry_eligibility_v1",
            **{k: getattr(self, k) for k in self.__dataclass_fields__ if k != "_issuer"},
            "scope": "complete_unit",
            "seed_changed": False,
            "runtime_admission": False,
            "scientific_admission": False,
        }

    def retry_evidence(self):
        return {
            "prior_intent_id": self.prior_intent_id,
            "prior_terminal_receipt_sha256": self.prior_terminal_receipt_sha256,
            "eligibility_receipt_sha256": content_digest(self.payload()),
            "scope": "complete_unit",
            "seed_changed": False,
        }


def resolve_service_retry(store, prior_intent_id: str):
    """Revalidate the published service seal and exact original failure bytes."""
    from agentmemeval.path1_v1.operations.intent_executor import NAMESPACE, DurableIntentExecutor

    snapshot = store.read_registry()
    previous = snapshot.registry[NAMESPACE]["intents"].get(prior_intent_id)
    _require(
        type(previous) is dict
        and previous["status"] == "accepted"
        and previous["submitter_finished"] is True
        and not previous.get("cleanup_persistence_failed", False),
        "only an accepted, finished original submission may qualify",
    )
    resolution = er.resolve_e0_terminal_intent(store, prior_intent_id)
    _require(
        type(resolution) is st.ServiceTerminalResolution
        and resolution.claim_status == "failed"
        and resolution.scheduler_state == "FAILED"
        and resolution.service_observation_complete is False,
        "sealed failed service allocation required",
    )
    resolution.validate(previous, store=store)
    prior = DurableIntentExecutor._stored_probe_contract(previous)
    _require(
        prior["phase"] == "E0"
        and prior["job"]["kind"] == "service_probe"
        and prior["job"]["attempt"] < 3,
        "external retry limit reached or wrong unit",
    )
    audit = er._json(st._audit_path(store, prior_intent_id))
    _require(
        er._sha(st._audit_path(store, prior_intent_id)) == resolution.evidence_file_sha256,
        "terminal audit changed",
    )
    facts = audit["facts"]
    _require(
        facts["cleanup_returncode"] == 1
        and "host_service_missing" in facts["observation_issues"]
        and "probe_receipt_missing" in facts["observation_issues"],
        "failure must precede HTTP observations and have verified cleanup",
    )
    log = er._path(previous["spec"]["work_dir"]) / "e0_probe/service_stderr.log"
    raw = er._path(log).read_bytes()
    descriptor = er._path(Path(audit["descriptor_path"]))
    relative = "run/" + log.relative_to(descriptor.parent).as_posix()
    matches_inventory = [r for r in audit["first_inventory"] if r["path"] == relative]
    _require(
        len(matches_inventory) == 1
        and matches_inventory[0]["size"] == len(raw)
        and matches_inventory[0]["sha256"] == hashlib.sha256(raw).hexdigest(),
        "driver log is not the sealed original bytes",
    )
    text = raw.decode("utf-8", errors="strict")
    matches = re.findall(
        r"The NVIDIA driver on your system is too old \(found version (\d+)\)", text
    )
    _require(
        len(matches) == 1 and "torch._C._cuda_init()" in text and "gpu_worker.py" in text,
        "exact pre-initialization CUDA driver failure missing",
    )
    _require(store.read_registry() == snapshot, "store changed during eligibility verification")
    return ServiceRetryEligibility(
        store.campaign_id,
        str(store.directory),
        prior_intent_id,
        content_digest(prior),
        resolution.terminal_receipt_sha256,
        resolution.evidence_file_sha256,
        hashlib.sha256(raw).hexdigest(),
        int(matches[0]),
        _issuer=_ISSUER,
    )


def validate_transition(prior, current, old_config, new_config):
    """Structural comparison only; never grants eligibility by itself."""
    for value in (prior, current):
        validate_campaign_contract(value)
        _require(
            value["phase"] == "E0"
            and value["job"]["kind"] == "service_probe"
            and value["not_for_paper"] is True,
            "only E0 service units may retry",
        )
    _require(
        current["job"]["attempt"] == prior["job"]["attempt"] + 1 <= 3,
        "consecutive attempt, at most two external retries, required",
    )
    for key in ("campaign_id", "source_contract", "protocol_sha256", "seed_namespace", "cohort"):
        _require(prior[key] == current[key], "unit identity changed: " + key)
    for key in ("core_files", "model"):
        _require(
            prior["runtime_identity"][key] == current["runtime_identity"][key],
            "core/model identity changed: " + key,
        )
    for key in prior["job"]:
        if key not in {"attempt", "retry_evidence", "output_relpath"}:
            _require(prior["job"][key] == current["job"][key], "job identity changed: " + key)
    _require(
        current["job"]["output_relpath"]
        == current["campaign_id"] + f"/E0/service/attempt_{current['job']['attempt']:02d}",
        "fresh attempt output required",
    )
    _require(
        {k: v for k, v in old_config.items() if k not in ENVIRONMENT_FIELDS}
        == {k: v for k, v in new_config.items() if k not in ENVIRONMENT_FIELDS},
        "change exceeds declared interpreter/dependency environment",
    )
    _require(
        old_config["resource_profile_version"] == new_config["resource_profile_version"] == 3,
        "retry requires unchanged site-default text-only profile",
    )


def validate_service_retry(store, contract, probe):
    """Recompute eligibility before intent reservation and again before helper go."""
    from agentmemeval.path1_v1.operations.intent_executor import NAMESPACE, DurableIntentExecutor

    retry = contract["job"]["retry_evidence"]
    _require(type(retry) is dict, "retry binding required")
    _require(
        probe["intent_id"] == canonical_intent_id(contract)
        and probe["request_sha256"]
        == content_digest({k: v for k, v in probe.items() if k != "request_sha256"}),
        "new probe binding changed",
    )
    eligible = resolve_service_retry(store, retry["prior_intent_id"])
    _require(retry == eligible.retry_evidence(), "retry evidence differs from original seal")
    records = store.read_registry().registry[NAMESPACE]["intents"]
    previous = records[eligible.prior_intent_id]
    prior = DurableIntentExecutor._stored_probe_contract(previous)
    old_probe = er._json(er._path(previous["probe_request_path"]))
    for key in ("repository_root", "model_root", "model_manifest_path"):
        _require(old_probe[key] == probe[key], "repository/model path changed: " + key)
    _require(old_probe["output_root"] != probe["output_root"], "independent run root required")
    repository = er._path(probe["repository_root"])
    old_config = er._historical_identity(repository, prior["runtime_identity"], allow_model=True)
    new_config = er._historical_identity(repository, contract["runtime_identity"], allow_model=True)
    validate_transition(prior, contract, old_config, new_config)
    _require(content_digest(prior) == eligible.prior_contract_sha256, "prior contract changed")
    if prior["job"]["attempt"] > 1:
        validate_service_retry(store, prior, old_probe)
    current_intent = canonical_intent_id(contract)
    for intent, record in records.items():
        if intent == current_intent or record.get("probe_request_path") is None:
            continue
        other = DurableIntentExecutor._stored_probe_contract(record)
        if (
            other["campaign_id"] != contract["campaign_id"]
            or other["job"]["kind"] != "service_probe"
        ):
            continue
        other_retry = other["job"]["retry_evidence"]
        _require(
            other["job"]["attempt"] != contract["job"]["attempt"]
            and (other_retry is None or other_retry["prior_intent_id"] != eligible.prior_intent_id),
            "attempt/prior seal already consumed by another child",
        )
    return eligible
