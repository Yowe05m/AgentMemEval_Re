"""Contract-bound Path1 worker with opt-in, non-admitting live E0 probes.

The executor owns allocation and service lifecycle. This worker owns one reserved
payload directory and never retries an incomplete scientific unit. CLI admission
is deliberately closed until a real in-job runtime verifier is installed in code.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from agentmemeval.path1_v1.experiments import path1
from agentmemeval.path1_v1.llm.providers.openai_compatible import (
    OpenAICompatibleClient,
    ProviderExchangeRecorder,
)
from agentmemeval.path1_v1.llm.request_identity import (
    SAMPLER_ALGORITHM,
    EpisodeRequestContext,
    RequestIdentityBinding,
    SamplerSeedLedger,
)
from agentmemeval.path1_v1.llm.terminal_policy import PATH1_TERMINAL_POLICY, Path1TerminalError
from agentmemeval.path1_v1.operations.contracts import (
    SOURCE_CONTRACT_SHA256,
    ContractError,
    _object,
    _read_json,
    _require,
    _sha,
    _slug,
    build_runtime_identity,
    canonical_intent_id,
    content_digest,
    validate_campaign_contract,
)
from agentmemeval.path1_v1.operations.model_verification import (
    VerifiedModelFiles,
    reuse_model_files,
    verify_model_files,
)
from agentmemeval.path1_v1.operations.state import CampaignStore
from agentmemeval.path1_v1.storage.snapshots import load_snapshot


class RuntimeAdmissionRequired(RuntimeError):
    pass


def _absolute(value: Any, *, exists: bool = True) -> Path:
    _require(
        type(value) is str and bool(value) and Path(value).is_absolute(),
        "worker paths must be absolute",
    )
    path = Path(value)
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "worker symlink path rejected")
    return path.resolve(strict=exists)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_json(path_value: str, sha: str) -> dict:
    _sha(sha, "bound file SHA256")
    path = _absolute(path_value)
    _require(_hash_file(path) == sha, "worker input byte hash mismatch")
    value = _read_json(path)
    _require(_hash_file(path) == sha, "worker input changed while reading")
    _require(type(value) is dict, "worker input must be an object")
    return value


def design_from_contract(contract: Mapping[str, Any], *, root_seed: int) -> path1.Path1Design:
    source = contract["source_contract"]
    technical = contract["phase"] == "E1"
    return path1.Path1Design(
        campaign_id=contract["campaign_id"],
        artifact_id=contract["job"]["artifact_id"],
        phase=contract["phase"],
        root_seed=root_seed,
        seed_namespace=contract["seed_namespace"],
        formation_episodes=source["technical"]["formation_episodes"] if technical else 6,
        formation_episode_hands=source["technical"]["formation_episode_hands"] if technical else 50,
        evaluation_episode_hands=source["technical"]["evaluation_episode_hands"]
        if technical
        else 50,
    )


def _planned_streams(design: path1.Path1Design, pair_count: int) -> dict[str, int]:
    ledger = path1.SeedLedger(design)
    actors = [path1.HERO_ID, *[f"opponent_{i:02d}" for i in range(1, 8)]]
    for stage, count, hands, arms in (
        ("formation", design.formation_episodes, design.formation_episode_hands, ("formation",)),
        ("evaluation", pair_count, design.evaluation_episode_hands, path1.ARMS),
    ):
        for episode in range(count):
            for hand in range(hands):
                ledger.get(stage, episode, "deal", hand)
            for arm in arms:
                for actor in actors:
                    ledger.get(stage, episode, arm, "client", actor)
    return ledger.entries


def validate_seed_inventory(inventory: Any, contract: Mapping[str, Any]) -> dict:
    """Check the whole cohort mapping and all precomputable deal/client streams."""
    _object(
        inventory,
        {
            "schema_version",
            "kind",
            "campaign_id",
            "phase",
            "seed_namespace",
            "root_seed",
            "environment_algorithm",
            "sampler_algorithm",
            "forbidden_environment_seeds",
            "forbidden_sampler_seeds",
            "artifacts",
        },
        "seed inventory",
    )
    _require(
        inventory["schema_version"] == 1
        and type(inventory["schema_version"]) is int
        and inventory["kind"] == "path1_worker_seed_inventory",
        "seed inventory schema",
    )
    for key in ("campaign_id", "phase", "seed_namespace"):
        _require(inventory[key] == contract[key], "seed inventory namespace mismatch")
    _require(
        inventory["environment_algorithm"] == "path1-sha256-63bit-v2"
        and inventory["sampler_algorithm"] == SAMPLER_ALGORITHM,
        "seed algorithm mismatch",
    )
    _require(
        type(inventory["root_seed"]) is int and inventory["root_seed"] >= 0,
        "root seed must be nonnegative integer",
    )
    for key, limit in (
        ("forbidden_environment_seeds", 2**63),
        ("forbidden_sampler_seeds", 2**63),
    ):
        values = inventory[key]
        _require(
            type(values) is list
            and all(type(v) is int and 0 <= v < limit for v in values)
            and values == sorted(set(values)),
            "forbidden seed inventory invalid",
        )
    artifacts = inventory["artifacts"]
    _require(
        type(artifacts) is list and len(artifacts) == contract["cohort"]["formation_count"],
        "seed inventory must cover the complete formation cohort",
    )
    artifact_ids, all_seeds = [], set(inventory["forbidden_environment_seeds"])
    b = contract["cohort"]["episodes_per_artifact_per_arm"]
    for artifact in artifacts:
        _object(artifact, {"artifact_id", "pairs", "streams"}, "seed artifact")
        _slug(artifact["artifact_id"], "seed artifact ID")
        artifact_ids.append(artifact["artifact_id"])
        pairs = artifact["pairs"]
        _require(type(pairs) is list and len(pairs) == b, "incomplete pair inventory")
        pair_ids = []
        for index, pair in enumerate(pairs):
            _object(pair, {"pair_id", "episode_index", "hero_seat"}, "pair mapping")
            _slug(pair["pair_id"], "pair ID")
            pair_ids.append(pair["pair_id"])
            _require(
                type(pair["episode_index"]) is int
                and pair["episode_index"] == index
                and type(pair["hero_seat"]) is int
                and pair["hero_seat"] == index % 8,
                "pair mapping must retain global episode indices and seat strata",
            )
        _require(len({x.casefold() for x in pair_ids}) == b, "duplicate pair IDs")
        item_contract = copy.deepcopy(dict(contract))
        item_contract["job"]["artifact_id"] = artifact["artifact_id"]
        design = design_from_contract(item_contract, root_seed=inventory["root_seed"])
        expected = _planned_streams(design, b)
        _require(
            type(artifact["streams"]) is dict
            and all(type(value) is int for value in artifact["streams"].values())
            and artifact["streams"] == expected,
            "missing, extra, or changed actual deal/client seeds",
        )
        values = list(expected.values())
        _require(not all_seeds.intersection(values), "cross-artifact/forbidden seed collision")
        all_seeds.update(values)
    _require(
        artifact_ids == sorted(artifact_ids)
        and len({x.casefold() for x in artifact_ids}) == len(artifact_ids),
        "artifact inventory must be unique and sorted",
    )
    _require(
        contract["job"]["artifact_id"] in artifact_ids, "job artifact is absent from inventory"
    )
    selected = next(a for a in artifacts if a["artifact_id"] == contract["job"]["artifact_id"])
    ordered_ids = [
        p["pair_id"] for p in selected["pairs"] if p["pair_id"] in contract["job"]["pair_ids"]
    ]
    _require(
        ordered_ids == contract["job"]["pair_ids"],
        "job pair list must be an ordered subset of the global mapping",
    )
    _require(
        content_digest(inventory) == contract["cohort"]["seed_inventory_sha256"],
        "cohort seed inventory digest mismatch",
    )
    return copy.deepcopy(inventory)


@dataclass(frozen=True)
class ValidatedWorkerRequest:
    request: dict
    contract: dict
    seed_inventory: dict
    claim: dict
    intent_id: str
    request_sha256: str
    output_directory: Path
    model_verification: VerifiedModelFiles | None = field(default=None, compare=False, repr=False)


def validate_worker_request(
    request_path: Path | str, *, model_verification: VerifiedModelFiles | None = None,
) -> ValidatedWorkerRequest:
    """Read-only content/claim validation. Success grants no runtime admission."""
    request = _read_json(_absolute(str(request_path)))
    _object(
        request,
        {
            "schema_version",
            "kind",
            "intent_id",
            "contract_path",
            "contract_sha256",
            "repository_root",
            "model_manifest_path",
            "model_root",
            "seed_inventory_path",
            "seed_inventory_sha256",
            "runtime_gate_path",
            "runtime_gate_sha256",
            "output_root",
            "claim_path",
            "claim_sha256",
            "artifact_root",
            "request_sha256",
        },
        "worker request",
    )
    _require(
        type(request["schema_version"]) is int
        and request["schema_version"] == 1
        and request["kind"] == "path1_worker_request",
        "worker request schema",
    )
    _require(
        content_digest({k: v for k, v in request.items() if k != "request_sha256"})
        == request["request_sha256"],
        "worker request digest mismatch",
    )
    repository = _absolute(request["repository_root"])
    model_manifest, model_root = (
        _absolute(request["model_manifest_path"]),
        _absolute(request["model_root"]),
    )
    contract = _bound_json(request["contract_path"], request["contract_sha256"])
    _require(contract.get("phase") in {"E1", "E2", "E3"}, "worker supports E1/E2/E3, not E0 probes")
    _require(model_verification is None or contract["phase"] == "E1",
             "model verification reuse is currently E1 only")
    artifact_root = (
        None if request["artifact_root"] is None else _absolute(request["artifact_root"])
    )
    contract = validate_campaign_contract(
        contract,
        repository_root=repository,
        model_manifest_path=model_manifest,
        model_root=model_root,
        artifact_root=artifact_root,
    )
    identity = contract["runtime_identity"]
    model_arguments = {}
    if contract["phase"] == "E1":
        # An in-process content capability is not runtime admission. New callers
        # always hash the files; only the exact request can reuse this 120s lease.
        if model_verification is None:
            model_verification = verify_model_files(
                model_manifest, model_root, expected_identity=identity["model"],
                binding_sha256=request["request_sha256"], max_age_seconds=120.0,
            )
        else:
            reuse_model_files(
                model_verification, model_manifest, model_root,
                binding_sha256=request["request_sha256"], expected_identity=identity["model"],
            )
        model_arguments = {
            "model_verification": model_verification,
            "model_binding_sha256": request["request_sha256"],
        }
    # E1 also needs fresh source/model binding; being not-for-paper is not an
    # exemption from identifying the code and files that the worker will use.
    observed = build_runtime_identity(
        repository,
        core_paths=[x["path"] for x in identity["core_files"]],
        operations_paths=[x["path"] for x in identity["operations_files"]],
        config_path=identity["config"]["path"],
        dependency_lock_path=identity["dependency_lock"]["path"],
        model_manifest_path=model_manifest,
        model_root=model_root,
        **model_arguments,
    )
    _require(observed == identity, "worker source/model identity changed")
    intent = canonical_intent_id(contract)
    _require(intent == request["intent_id"], "worker intent mismatch")
    seeds = validate_seed_inventory(
        _bound_json(request["seed_inventory_path"], request["seed_inventory_sha256"]),
        contract,
    )
    if request["runtime_gate_path"] is None:
        _require(request["runtime_gate_sha256"] is None, "runtime gate path/hash mismatch")
    else:
        _bound_json(request["runtime_gate_path"], request["runtime_gate_sha256"])
    output_root = _absolute(request["output_root"])
    output = output_root.joinpath(*contract["job"]["output_relpath"].split("/"), "payload")
    _absolute(str(output), exists=False)
    _require(not output.exists(), "worker payload directory already exists")
    _require(
        not any(output == root or output.is_relative_to(root) for root in (repository, model_root)),
        "worker output must be outside source and model directories",
    )
    claim = _bound_json(request["claim_path"], request["claim_sha256"])
    _object(
        claim,
        {
            "schema_version",
            "kind",
            "campaign_id",
            "intent_id",
            "control_directory",
            "run_id",
            "run_directory",
            "token",
        },
        "worker claim",
    )
    _require(
        type(claim["schema_version"]) is int
        and claim["schema_version"] == 1
        and claim["kind"] == "path1_worker_claim",
        "worker claim schema",
    )
    _require(
        claim["campaign_id"] == contract["campaign_id"]
        and claim["intent_id"] == intent
        and _absolute(claim["run_directory"], exists=False) == output,
        "worker claim identity/output mismatch",
    )
    store = CampaignStore(_absolute(claim["control_directory"]), contract["campaign_id"])
    matches = [c for c in store.read_registry().claims if c.run_id == claim["run_id"]]
    _require(
        len(matches) == 1
        and matches[0].status == "reserved"
        and matches[0].intent_id == intent
        and matches[0].token == claim["token"]
        and Path(matches[0].run_directory) == output,
        "worker claim is not an actual reservation",
    )
    return ValidatedWorkerRequest(
        request, contract, seeds, claim, intent, request["request_sha256"], output,
        model_verification,
    )


@dataclass(frozen=True)
class RuntimeProbeFile:
    purpose: str
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class RuntimeAdmission:
    intent_id: str
    request_sha256: str
    core_manifest_sha256: str
    operations_manifest_sha256: str
    config_sha256: str
    model_manifest_sha256: str
    slurm_job_id: str
    service_instance_id: str
    issued_at_utc: datetime
    deadline_utc: datetime
    probes: tuple[RuntimeProbeFile, ...]


class RuntimeVerifier(Protocol):
    def verify_in_job(self, request: ValidatedWorkerRequest) -> RuntimeAdmission:
        """Trusted implementation must verify actual node/service/loaded code and
        raw seed-range/execution, tokenizer capacity, schema and model probes.
        Parsing self-reported JSON booleans is not an implementation. None ships
        here: fake implementations are permitted only in labelled offline tests.
        """
        ...


def _check_runtime(
    admission: RuntimeAdmission, request: ValidatedWorkerRequest, now: datetime
) -> None:
    if type(admission) is not RuntimeAdmission:
        raise RuntimeAdmissionRequired("trusted verifier did not return a typed runtime receipt")
    identity = request.contract["runtime_identity"]
    expected = (
        request.intent_id,
        request.request_sha256,
        identity["core_manifest_sha256"],
        identity["operations_manifest_sha256"],
        identity["config"]["sha256"],
        identity["model"]["manifest_sha256"],
    )
    actual = (
        admission.intent_id,
        admission.request_sha256,
        admission.core_manifest_sha256,
        admission.operations_manifest_sha256,
        admission.config_sha256,
        admission.model_manifest_sha256,
    )
    if expected != actual:
        raise RuntimeAdmissionRequired("runtime receipt is bound to different inputs")
    if (
        not admission.slurm_job_id.isdigit()
        or not admission.service_instance_id
        or os.environ.get("SLURM_JOB_ID") != admission.slurm_job_id
        or now.tzinfo is None
        or admission.issued_at_utc.tzinfo is None
        or admission.deadline_utc.tzinfo is None
        or not admission.issued_at_utc <= now < admission.deadline_utc
    ):
        raise RuntimeAdmissionRequired("runtime job/service/deadline evidence is invalid")
    purposes = {
        "sampler_seed_execution_63bit",
        "tokenizer_total_context",
        "structured_schema",
        "loaded_model_and_source_identity",
    }
    if type(admission.probes) is not tuple or {p.purpose for p in admission.probes} != purposes:
        raise RuntimeAdmissionRequired("runtime probe coverage is incomplete")
    if len(admission.probes) != len(purposes):
        raise RuntimeAdmissionRequired("runtime probe purposes must be unique")
    for probe in admission.probes:
        if type(probe) is not RuntimeProbeFile:
            raise RuntimeAdmissionRequired("runtime probe is not typed evidence")
        path = _absolute(probe.path)
        _sha(probe.sha256, "probe SHA")
        if (
            type(probe.size) is not int
            or path.stat().st_size != probe.size
            or (_hash_file(path) != probe.sha256)
        ):
            raise RuntimeAdmissionRequired("runtime probe bytes changed")


class StrictPath1ClientFactory:
    """Contract-to-HTTP settings. Constructing it does not confer live admission."""

    def __init__(
        self,
        source_contract: Mapping[str, Any],
        *,
        forbidden_sampler_seeds: set[int] | None = None,
        before_request: Callable[[], None] | None = None,
    ) -> None:
        self.source = copy.deepcopy(dict(source_contract))
        _require(
            content_digest(self.source) == SOURCE_CONTRACT_SHA256,
            "strict factory requires the unchanged source contract",
        )
        self.ledger = SamplerSeedLedger(forbidden_seeds=forbidden_sampler_seeds)
        self.before_request = before_request
        self.recorders: dict[Path, ProviderExchangeRecorder] = {}
        self.bindings: list[RequestIdentityBinding] = []
        self.contexts: set[EpisodeRequestContext] = set()

    def create_episode_client(
        self, context: EpisodeRequestContext, audit_path: Path
    ) -> OpenAICompatibleClient:
        if context in self.contexts:
            raise Path1TerminalError("state", "duplicate_episode_actor_client")
        self.contexts.add(context)
        source = self.source["provider"]
        keys = (
            "structured_output_mode",
            "json_schema_unsupported_keywords",
            "temperature",
            "experience_temperature",
            "timeout_seconds",
            "max_output_tokens",
            "repair_max_output_tokens",
            "experience_max_output_tokens",
            "experience_repair_max_output_tokens",
            "context_limit",
        )
        config = {key: copy.deepcopy(source[key]) for key in keys}
        config.update(
            provider="openai_compatible",
            model=source["served_model_name"],
            terminal_policy=PATH1_TERMINAL_POLICY,
            max_retries=0,
            base_url_env="PATH1_MODEL_BASE_URL",
            api_key_env="PATH1_MODEL_API_KEY",
            api_key_required=False,
            response_model_identity_policy="strict_exact",
            enable_thinking=self.source["service"]["enable_thinking"],
        )
        client = OpenAICompatibleClient(config)
        path = Path(audit_path).resolve()
        if path not in self.recorders:
            self.recorders[path] = ProviderExchangeRecorder(path)
        client.bind_exchange_recorder(self.recorders[path])
        binding = RequestIdentityBinding(context, self.ledger, before_request=self.before_request)
        client.bind_request_identity(binding)
        self.bindings.append(binding)
        return client


def _execute(request: ValidatedWorkerRequest, factory: StrictPath1ClientFactory) -> dict:
    contract, inventory, output = request.contract, request.seed_inventory, request.output_directory
    design = design_from_contract(contract, root_seed=inventory["root_seed"])
    source = contract["source_contract"]
    arguments = {
        "design": design,
        "hero_config": source["controller"]["hero"],
        "model": source["provider"]["served_model_name"],
        "client_factory": factory,
        "forbidden_seeds": set(inventory["forbidden_environment_seeds"]),
    }
    job = contract["job"]
    completed = []
    if job["kind"] in {"formation", "technical_closed_loop"}:
        snapshot, receipt = path1.run_formation(output_dir=output / "formation", **arguments)
        completed.append({"unit": "formation", "receipt": receipt})
        if job["kind"] == "formation":
            identity = contract["runtime_identity"]
            snapshot_path = output / "formation" / "artifact_snapshot.json"
            immutable = {
                "schema_version": 1,
                "kind": "path1_immutable_formation_receipt",
                "campaign_id": contract["campaign_id"],
                "phase": contract["phase"],
                "artifact_id": job["artifact_id"],
                "formation_intent_id": request.intent_id,
                "seed_namespace": contract["seed_namespace"],
                "protocol_sha256": contract["protocol_sha256"],
                "core_manifest_sha256": identity["core_manifest_sha256"],
                "operations_manifest_sha256": identity["operations_manifest_sha256"],
                "model_manifest_sha256": identity["model"]["manifest_sha256"],
                "formation_config_sha256": identity["config"]["sha256"],
                "snapshot": {
                    "path": "formation/artifact_snapshot.json",
                    "size": snapshot_path.stat().st_size,
                    "sha256": _hash_file(snapshot_path),
                },
                "completed_hands": receipt["settled_hands"],
                "completed_episodes": receipt["completed_episodes"],
                "selected_after_hand": receipt["settled_hands"],
                "selection_policy": source["formation"]["primary_snapshot"],
                "status": "complete",
            }
            immutable["receipt_sha256"] = content_digest(immutable)
            from agentmemeval.path1_v1.operations.contracts import validate_artifact_receipt

            validate_artifact_receipt(immutable, contract)
            path1._new_json(output / "immutable_formation_receipt.json", immutable)
    else:
        record = job["formation_receipt"]["snapshot"]
        snapshot = load_snapshot(Path(request.request["artifact_root"]) / record["path"])
    artifact = next(a for a in inventory["artifacts"] if a["artifact_id"] == job["artifact_id"])
    # Enumerating this job's subset would duplicate streams across different jobs.
    for pair in artifact["pairs"]:
        if pair["pair_id"] not in job["pair_ids"]:
            continue
        receipt = path1.run_evaluation_pair(
            output_dir=output / pair["pair_id"],
            snapshot=snapshot,
            episode_index=pair["episode_index"],
            **arguments,
        )
        completed.append(
            {"unit": pair["pair_id"], "episode_index": pair["episode_index"], "receipt": receipt}
        )
    return {"completed_units": completed}


def _produced_artifacts(output: Path) -> list[dict]:
    """Seal completed producer bytes before publishing the terminal receipt."""
    paths = sorted(output.rglob("*"))
    records = []
    for path in paths:
        if path.is_symlink():
            raise RuntimeAdmissionRequired("payload symlink cannot be sealed")
        if not path.is_file():
            continue
        relative = path.relative_to(output).as_posix()
        if relative == "worker_terminal.json":
            raise RuntimeAdmissionRequired("worker terminal already exists before sealing")
        before = path.stat()
        digest = _hash_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino
        ):
            raise RuntimeAdmissionRequired("payload changed during producer sealing")
        records.append({"path": relative, "size": after.st_size, "sha256": digest})
    if paths != sorted(output.rglob("*")):
        raise RuntimeAdmissionRequired("payload inventory changed during producer sealing")
    return records


def run_worker(
    request_path: Path | str,
    *,
    runtime_verifier: RuntimeVerifier | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict:
    request = validate_worker_request(request_path)
    if runtime_verifier is None:
        raise RuntimeAdmissionRequired(
            "no trusted in-job runtime verifier is installed; no HTTP allowed"
        )
    admission = runtime_verifier.verify_in_job(request)
    if validate_worker_request(request_path) != request:
        raise RuntimeAdmissionRequired("worker inputs changed during runtime verification")
    _check_runtime(admission, request, clock())
    claim = request.claim
    store = CampaignStore(claim["control_directory"], claim["campaign_id"])
    with store.lock():
        current = store.read_registry()
        store.start_claim(
            expected_generation=current.generation, run_id=claim["run_id"], token=claim["token"]
        )
    output = request.output_directory

    def live_request_gate() -> None:
        try:
            _check_runtime(admission, request, clock())
            if hasattr(runtime_verifier, "verify_live"):
                runtime_verifier.verify_live(request, admission)
        except (RuntimeError, ContractError, OSError) as exc:
            raise Path1TerminalError("state", "runtime_evidence_revalidation_failed") from exc

    factory = StrictPath1ClientFactory(
        request.contract["source_contract"],
        forbidden_sampler_seeds=set(request.seed_inventory["forbidden_sampler_seeds"]),
        before_request=live_request_gate,
    )
    receipt = {
        "schema_version": 1,
        "kind": "path1_worker_terminal",
        "intent_id": request.intent_id,
        "worker_request_sha256": request.request_sha256,
        "primary_analysis_admitted": False,
        "retry_eligibility": "not_established",
        "service_release_verified": False,
        "campaign_id": request.contract["campaign_id"],
        "phase": request.contract["phase"],
        "run_id": claim["run_id"],
        "slurm_job_id": admission.slurm_job_id,
        "service_instance_id": admission.service_instance_id,
        "core_manifest_sha256": admission.core_manifest_sha256,
        "operations_manifest_sha256": admission.operations_manifest_sha256,
        "config_sha256": admission.config_sha256,
        "model_manifest_sha256": admission.model_manifest_sha256,
        "protocol_sha256": request.contract["protocol_sha256"],
    }
    owns_output = False
    try:
        output.mkdir(parents=True, exist_ok=False)
        owns_output = True
        path1._new_json(
            output / "worker_started.json",
            {
                "intent_id": request.intent_id,
                "request_sha256": request.request_sha256,
                "slurm_job_id": admission.slurm_job_id,
                "service_instance_id": admission.service_instance_id,
                "sampler_algorithm": SAMPLER_ALGORITHM,
            },
        )
        path1._new_json(
            output / "runtime_admission.json",
            {
                **asdict(admission),
                "issued_at_utc": admission.issued_at_utc.isoformat(),
                "deadline_utc": admission.deadline_utc.isoformat(),
            },
        )
        result = _execute(request, factory)
        receipt.update(status="completed", **result)
        path1._new_json(
            output / "sampler_seed_manifest.json",
            {
                "algorithm": SAMPLER_ALGORITHM,
                "streams": factory.ledger.entries,
                "backend_execution_attested_only_by_runtime_verifier": True,
            },
        )
        receipt["produced_artifacts"] = _produced_artifacts(output)
        path1._new_json(output / "worker_terminal.json", receipt)
    except BaseException as exc:
        receipt.update(status="incomplete", failure_type=type(exc).__name__)
        if isinstance(exc, Path1TerminalError):
            receipt["failure_category"] = exc.diagnostics["failure_category"]
            receipt["failure_code"] = exc.diagnostics["failure_code"]
        if owns_output:
            path1._new_json(output / "worker_terminal.json", receipt)
        with store.lock():
            current = store.read_registry()
            store.finish_claim(
                expected_generation=current.generation,
                run_id=claim["run_id"],
                token=claim["token"],
                status="failed",
                receipt=receipt,
            )
        raise
    with store.lock():
        current = store.read_registry()
        store.finish_claim(
            expected_generation=current.generation,
            run_id=claim["run_id"],
            token=claim["token"],
            status="completed",
            receipt=receipt,
        )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--runtime-probe-output", type=Path)
    parser.add_argument("--service-python", type=Path)
    parser.add_argument("--service-port", type=int)
    args = parser.parse_args(argv)
    try:
        if args.runtime_probe_output is not None:
            from agentmemeval.path1_v1.operations.runtime_probe import run_service_probe

            if args.service_python is None or args.service_port is None:
                raise RuntimeAdmissionRequired("live probe needs explicit service Python and port")
            run_service_probe(args.request, args.runtime_probe_output,
                              args.service_python, args.service_port)
            # Exit success means only the explicitly requested E0 collection
            # completed. Its report still has runtime_admission=False.
            return 0
        if args.service_python is not None or args.service_port is not None:
            raise RuntimeAdmissionRequired("service options require an explicit E0 probe")
        run_worker(args.request)
    except (ContractError, RuntimeAdmissionRequired):
        # No serialized "verified" flag or CLI switch can install a trusted verifier.
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
