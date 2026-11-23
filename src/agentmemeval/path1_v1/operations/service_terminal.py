"""Historical service E0 evidence plus fresh scheduler technical closure.

This is not a process-control capability, retry authorization, backend attestation
or scientific admission. Missing cleanup keeps the original claim occupied.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentmemeval.path1_v1.operations import e0_reconcile as er
from agentmemeval.path1_v1.operations import resource_terminal as rt
from agentmemeval.path1_v1.operations import terminal_common as tc
from agentmemeval.path1_v1.operations.contracts import content_digest
from agentmemeval.path1_v1.operations.e0_service_cli import read_prepared
from agentmemeval.path1_v1.operations.runtime_probe import verify_http_observation
from agentmemeval.path1_v1.operations.service_profile import planned_service_argv
from agentmemeval.path1_v1.operations.state import CampaignStore, RunClaim

KIND = "path1_e0_service_terminal_v1"
_MODULE_PATH = "src/agentmemeval/path1_v1/operations/service_terminal.py"
_ISSUER = object()
_CASES = ("action", "patch", "repair", "patch_repair", "near_context", "over_context")


def _assert(value: bool, message: str) -> None:
    er._require(value, "service terminal: " + message)


def _audit_path(store: CampaignStore, intent: str) -> Path:
    er.digest(intent)
    return store.directory / "service_terminals" / intent / "evidence.json"


def _verifier_identity(repository: Path, config: dict) -> dict:
    identity = er._verifier_identity(repository, config)
    for module, relative in ((Path(__file__), _MODULE_PATH), (Path(rt.__file__), rt._MODULE_PATH)):
        _assert(module.resolve() == repository / relative, "verifier loaded from another checkout")
        _assert(
            any(
                row["path"] == relative and row["sha256"] == er._sha(module)
                for row in identity["operations_files"]
            ),
            "verifier is not published",
        )
    tc.verify_implementation_source(repository, identity)
    return identity


def _bound_inputs(
    store, descriptor_path, expected_sha, record, claim, *, require_empty_output=False
):
    _assert(require_empty_output is False, "service cannot use never-started resource semantics")
    descriptor_path = er._path(descriptor_path)
    _assert(descriptor_path.name == "submission.json", "descriptor filename differs")
    run = descriptor_path.parent
    descriptor, config, request = read_prepared(run, expected_sha)
    contract, probe = er._json(run / "contract.json"), er._json(run / "probe_request.json")
    repository = er._path(descriptor["repository_root"])
    _assert(
        er._historical_identity(repository, contract["runtime_identity"], allow_model=True)
        == config,
        "frozen service configuration differs from historical Git",
    )
    bound = er._accepted_inputs(
        store,
        descriptor,
        request,
        run,
        probe,
        contract,
        config,
        repository,
        record,
        claim,
        require_empty_output=False,
    )
    manifest = er._json(bound["invocation"] / "invocation.json")
    pins = {row["path"]: row["sha256"] for row in manifest["input_files"]}
    _assert(
        pins.get(probe["model_manifest_path"])
        == contract["runtime_identity"]["model"]["manifest_sha256"],
        "submitted model manifest pin differs",
    )
    return bound


def _service_facts(bound: dict, record: dict) -> dict:
    _assert(
        type(record.get("effective_user")) is dict
        and type(record["effective_user"].get("uid")) is int,
        "actual submitting UID is missing",
    )
    root = er._path(bound["request"].spec.work_dir) / "e0_probe"
    launch, cleanup = er._json(root / "launch.json"), er._json(root / "service_cleanup.json")
    contract, probe = bound["contract"], bound["probe"]
    binding = launch["binding"]
    expected = {
        "campaign_id": contract["campaign_id"],
        "intent_id": record["intent_id"],
        "job_id": record["job_id"],
        "source_sha256": content_digest(contract["runtime_identity"]),
        "role": "model_service",
        "service_instance_id": binding["service_instance_id"],
    }
    _assert(
        type(binding["service_instance_id"]) is str
        and re.fullmatch(r"[0-9a-f]{32}", binding["service_instance_id"]) is not None
        and binding == cleanup["binding"] == expected,
        "service launch/cleanup binding differs",
    )
    argv = planned_service_argv(bound["config"], contract["source_contract"])
    python = probe["launch"]["service_python_identity"]
    _assert(
        launch["argv"] == argv
        and launch["python_resolved"] == python["resolved_executable"]
        and launch["python_sha256"] == python["executable_sha256"],
        "actual service launch differs from prepared argv/interpreter",
    )
    er._keys(
        cleanup,
        {
            "binding",
            "identity",
            "returncode",
            "term_sent",
            "kill_sent",
            "leader_reaped",
            "group_cleanup_verified",
            "remaining_non_zombie_pids",
            "errors",
        },
        "service cleanup",
    )
    _assert(
        cleanup["leader_reaped"] is True
        and cleanup["group_cleanup_verified"] is True
        and cleanup["remaining_non_zombie_pids"] == []
        and cleanup["errors"] == []
        and type(cleanup["returncode"]) is int
        and type(cleanup["term_sent"]) is bool
        and type(cleanup["kill_sent"]) is bool,
        "service owned group cleanup is unverified",
    )
    child = cleanup["identity"]
    er._keys(
        child,
        {
            "host",
            "boot_id",
            "effective_uid",
            "pid",
            "start_marker",
            "session_id",
            "pgid",
            "cgroup",
            "slurm_job_id",
            "step_id",
        },
        "service process identity",
    )
    _assert(
        all(type(child[k]) is int and child[k] > 0 for k in ("pid", "pgid", "session_id"))
        and child["pid"] == child["pgid"] == child["session_id"]
        and type(child["start_marker"]) is str
        and child["start_marker"].isdigit()
        and int(child["start_marker"]) > 0
        and all(type(child[k]) is str and bool(child[k]) for k in ("host", "boot_id"))
        and type(child["effective_uid"]) is int
        and child["effective_uid"] == record["effective_user"]["uid"]
        and child["slurm_job_id"] == record["job_id"],
        "service kernel identity differs",
    )
    cgroup = child["cgroup"]
    _assert(
        type(cgroup) is list
        and cgroup
        and all(type(x) is str for x in cgroup)
        and any("job_" + record["job_id"] in line.split("/") for line in cgroup),
        "service cgroup does not bind the allocation",
    )
    steps = {"batch"}
    if child["step_id"] is not None:
        _assert(
            type(child["step_id"]) is str and re.fullmatch(r"[A-Za-z0-9_-]+", child["step_id"]),
            "invalid service step",
        )
        steps.add(child["step_id"])
    for line in cgroup:
        steps.update(
            part.removeprefix("step_") for part in line.split("/") if part.startswith("step_")
        )
    for name in ("service_stdout.log", "service_stderr.log"):
        _assert(er._path(root / name).is_file(), "service command log missing")
    facts = {
        "output_directory": str(root),
        "hostname": child["host"],
        "boot_id": child["boot_id"],
        "service_pid": child["pid"],
        "service_instance_id": binding["service_instance_id"],
        "cleanup_returncode": cleanup["returncode"],
        "cleanup_term_sent": cleanup["term_sent"],
        "cleanup_kill_sent": cleanup["kill_sent"],
        "required_steps": sorted(steps),
        "launch_sha256": er._sha(root / "launch.json"),
        "cleanup_sha256": er._sha(root / "service_cleanup.json"),
    }
    facts.update(_observations(root, bound, child, binding, argv))
    return facts


def _observations(root, bound, child, binding, argv):
    """Validate original producer output; incompleteness is a technical failed result."""
    issues = []
    output = root / "observations"
    probe, contract = bound["probe"], bound["contract"]
    host_path = output / "host_service.json"
    if host_path.exists():
        host_service = er._json(host_path)
        service, host = host_service["service"], host_service["host"]
        _assert(
            service["binding"] == binding
            and service["identity"] == child
            and service["cmdline"] == argv,
            "observed live service differs from cleanup",
        )
        _assert(
            host["hostname"] == child["host"]
            and host["boot_id"] == child["boot_id"]
            and type(host["euid"]) is int
            and host["euid"] == child["effective_uid"]
            and host["job_id"] == child["slurm_job_id"]
            and host["cgroup"].splitlines() == child["cgroup"]
            and type(host["pid"]) is int
            and host["pid"] > 0
            and host["pid"] != child["pid"],
            "observed launcher/service host differs",
        )
    else:
        issues.append("host_service_missing")
    failure_path = root / "service_failure.json"
    if failure_path.exists():
        failure = er._json(failure_path)
        _assert(
            failure["request_sha256"] == probe["request_sha256"]
            and failure["runtime_admission"] is False
            and type(failure["failure_type"]) is str
            and failure["failure_type"],
            "service failure belongs to another request",
        )
        issues.append("service_failure:" + failure["failure_type"])
    receipt_path = output / "probe_receipt.json"
    if not receipt_path.exists():
        issues.append("probe_receipt_missing")
    else:
        receipt = er._json(receipt_path)
        _assert(
            type(receipt["schema_version"]) is int
            and receipt["schema_version"] == 1
            and receipt["kind"] == "path1_live_e0_observation"
            and receipt["intent_id"] == probe["intent_id"]
            and receipt["probe_request_sha256"] == probe["request_sha256"]
            and receipt["campaign_id"] == contract["campaign_id"]
            and receipt["runtime_admission"] is False,
            "probe receipt binding/admission differs",
        )
        _assert(
            receipt["status"] in {"incomplete", "observations_complete_backend_execution_pending"},
            "unknown service observation status",
        )
        if receipt["status"] == "incomplete":
            issues.append("probe_incomplete")
        else:
            _complete_observations(output, bound, child, binding, argv, receipt)
    return {
        "service_observation_complete": not issues,
        "observation_issues": issues,
        "allocation_gpu_mapping_verified": False,
        "backend_execution_verified": False,
        "preinventory_unknown_artifact_completeness_proven": False,
    }


def _complete_observations(output, bound, child, binding, argv, receipt):
    rows = receipt["files"]
    _assert(type(rows) is list and rows, "complete observation inventory missing")
    actual = [
        {"path": p.name, "size": p.stat().st_size, "sha256": er._sha(er._path(p))}
        for p in sorted(output.iterdir())
        if p.name != "probe_receipt.json" and p.is_file()
    ]
    _assert(rows == actual, "producer observation inventory changed")
    required = {
        "host_service.json",
        "service_arguments.json",
        "gpu.json",
        "dependencies.json",
        "model_files.json",
        "tokenizer.json",
        "endpoint.json",
    }
    required.update(
        name + suffix for name in _CASES for suffix in ("_request.json", "_response.json")
    )
    _assert(required <= {r["path"] for r in rows}, "complete service observations missing files")
    endpoint = er._json(output / "endpoint.json")
    proc = endpoint["process"]
    _assert(
        proc["binding"] == binding
        and proc["identity"] == child
        and proc["cmdline"] == argv
        and endpoint["base_url"] == f"http://127.0.0.1:{bound['config']['service_port']}/v1"
        and type(endpoint["listener_inodes"]) is list
        and endpoint["listener_inodes"]
        and all(type(x) is int and x > 0 for x in endpoint["listener_inodes"])
        and type(endpoint["socket_holders"]) is list
        and endpoint["socket_holders"]
        and type(endpoint["network_namespace"]) is str
        and endpoint["network_namespace"],
        "endpoint belongs to another service",
    )
    holders = endpoint["socket_holders"]
    _assert(
        all(
            type(row) is list
            and len(row) == 3
            and type(row[0]) is int
            and row[0] > 0
            and type(row[1]) is str
            and row[1].isdigit()
            and int(row[1]) > 0
            and type(row[2]) is int
            and row[2] in endpoint["listener_inodes"]
            for row in holders
        )
        and {row[2] for row in holders} == set(endpoint["listener_inodes"]),
        "endpoint listener ownership evidence incomplete",
    )
    expected_model = bound["contract"]["runtime_identity"]["model"]
    _assert(
        er._json(output / "model_files.json")["identity"] == expected_model,
        "observed model identity differs",
    )
    tokenizer = er._json(output / "tokenizer.json")
    import hashlib

    _assert(
        hashlib.sha256(tokenizer["chat_template"].encode()).hexdigest()
        == tokenizer["chat_template_sha256"]
        == expected_model["manifest"]["chat_template_sha256"],
        "observed tokenizer template differs",
    )
    observations = receipt["observations"]
    _assert(
        type(observations) is list and [row["name"] for row in observations] == list(_CASES),
        "six service observations are not complete and ordered",
    )
    for index, row in enumerate(observations):
        name = row["name"]
        payload, response = (
            er._json(output / (name + "_request.json")),
            er._json(output / (name + "_response.json")),
        )
        budget = (256, 8192, 256, 4096, 256, 256)[index]
        _assert(
            payload["model"]
            == bound["contract"]["source_contract"]["provider"]["served_model_name"]
            and type(payload["seed"]) is int
            and payload["seed"] == 2**62 + 1729 + index
            and type(payload["max_tokens"]) is int
            and payload["max_tokens"] == budget
            and payload["temperature"] == (1.0 if name in {"patch", "patch_repair"} else 0.0)
            and payload["chat_template_kwargs"] == {"enable_thinking": False}
            and type(row["local_token_count"]) is int
            and row["local_token_count"] > 0,
            "service HTTP request contract differs",
        )
        er.digest(row["token_ids_sha256"])
        computed = verify_http_observation(
            payload, response, row["local_token_count"], over_limit=name == "over_context"
        )
        computed.update(
            name=name,
            local_token_count=row["local_token_count"],
            token_ids_sha256=row["token_ids_sha256"],
            output_budget=budget,
            seed_scope="E0_technical_fixture_only",
        )
        _assert(computed == row, "saved service HTTP interpretation differs")
        if name == "patch":
            _assert(
                row["content"]["operation"] == "patch" and row["content"]["patches"],
                "patch branch unobserved",
            )


def _claim_status(facts, scheduler):
    # A normally supervised server receives TERM after collection; -15 is not a failed probe.
    normal_exit = facts["cleanup_returncode"] == 0 or (
        facts["cleanup_returncode"] == -15
        and facts["cleanup_term_sent"]
        and not facts["cleanup_kill_sent"]
    )
    success = (
        facts["service_observation_complete"]
        and normal_exit
        and scheduler["state"] == "COMPLETED"
        and scheduler["exit_code"] == "0:0"
        and all(
            row["state"] == "COMPLETED" and row["exit_code"] == "0:0" for row in scheduler["steps"]
        )
    )
    return "completed" if success else "failed"


@dataclass(frozen=True)
class ServiceTerminalResolution:
    intent_id: str
    job_id: str
    campaign_id: str
    submission_request_sha256: str
    descriptor_sha256: str
    evidence_file_sha256: str
    terminal_receipt_sha256: str
    original_code_sha: str
    verifier_code_sha: str
    scheduler_state: str
    claim_status: str
    service_observation_complete: bool
    classification: str = "service_terminal_observed"
    phase: str = "E0"
    retry_eligible: bool = False
    runtime_admission: bool = False
    scientific_admission: bool = False
    primary_analysis_admitted: bool = False
    scientific_samples: int = 0
    _issuer: Any = field(default=None, repr=False, compare=False)

    def validate(self, record: dict, *, store: CampaignStore):
        _assert(self._issuer is _ISSUER, "resolution was not issued by verifier")
        current = er.resolve_e0_terminal_intent(store, self.intent_id)
        _assert(
            type(current) is ServiceTerminalResolution
            and current == self
            and record["request_sha256"] == self.submission_request_sha256,
            "resolution changed",
        )
        return current


def _policy():
    return tc.TerminalPolicy(
        kind=KIND,
        evidence_kind="path1_e0_service_terminal_evidence_v1",
        readonly_kind="path1_e0_service_readonly_check_v1",
        classification="service_terminal_observed",
        observation_key="service_observation_complete",
        module_path=_MODULE_PATH,
        issuer=_ISSUER,
        resolution_type=ServiceTerminalResolution,
        audit_path=_audit_path,
        bound_inputs=_bound_inputs,
        verifier_identity=_verifier_identity,
        facts=_service_facts,
        scheduler=rt.validate_resource_scheduler,
        archive=rt._archive_sources,
        verify_archives=rt._verify_archives,
        claim_status=_claim_status,
    )


def validate_service_terminal(record: dict, claim: RunClaim, *, store: CampaignStore):
    return tc.validate_terminal(_policy(), record, claim, store=store)


def check_service_terminal(
    store: CampaignStore,
    *,
    descriptor_path: Path,
    descriptor_sha256: str,
    reader=None,
    clock=lambda: datetime.now(UTC),
) -> dict:
    return tc.check_terminal(
        _policy(),
        store,
        descriptor_path=descriptor_path,
        descriptor_sha256=descriptor_sha256,
        reader=reader,
        clock=clock,
    )


def reconcile_service_terminal(
    store: CampaignStore,
    *,
    descriptor_path: Path,
    descriptor_sha256: str,
    reader=None,
    clock=lambda: datetime.now(UTC),
    monotonic=time.monotonic,
    sleep=time.sleep,
) -> ServiceTerminalResolution:
    return tc.reconcile_terminal(
        _policy(),
        store,
        descriptor_path=descriptor_path,
        descriptor_sha256=descriptor_sha256,
        reader=reader,
        clock=clock,
        monotonic=monotonic,
        sleep=sleep,
    )
