"""Prepare E1, then explicitly submit or look up; neither grants runtime admission."""
from __future__ import annotations

import argparse
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from agentmemeval.path1_v1.operations.contracts import (
    SOURCE_CONTRACT_SHA256,
    _object,
    _require,
    _sha,
    _slug,
    build_runtime_identity,
    canonical_intent_id,
    canonical_json,
    content_digest,
    validate_campaign_contract,
)
from agentmemeval.path1_v1.operations.interpreters import verify_python_interpreter
from agentmemeval.path1_v1.operations.state import CampaignStore
from agentmemeval.path1_v1.operations.submission_process import (
    file_sha256,
    read_json,
    write_exclusive_json,
)
from agentmemeval.path1_v1.operations.worker import (
    SAMPLER_ALGORITHM,
    _absolute,
    _bound_json,
    _planned_streams,
    design_from_contract,
    validate_seed_inventory,
)

BACKEND_MODULES = frozenset({
    "vllm.v1.engine.async_llm", "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.sample.ops.topk_topp_sampler", "vllm.v1.sample.sampler",
    "vllm.v1.worker.gpu_input_batch",
})
CASES = ("action", "patch", "repair", "patch_repair", "near_context", "over_context")
PROFILE_PATH = "src/agentmemeval/path1_v1/assets/path1_e1_execution_profile.json"


def _same(left, right):
    return canonical_json(left) == canonical_json(right)


def _checked_seeds(values):
    _require(type(values) is list and all(type(seed) is int and 0 <= seed < 2**63
                                         for seed in values)
             and values == sorted(set(values)), "forbidden seeds must be sorted unique int63")
    return list(values)


def _e0_observations(prior: dict, contract: dict, service_identity: dict) -> dict:
    """Re-read historical producer files. This grants no live owner/backend rights."""
    from agentmemeval.path1_v1.operations import runtime_probe as probe_runtime
    from agentmemeval.path1_v1.operations.e0_service_cli import read_prepared

    _object(prior, {"run_root", "descriptor_sha256", "receipt_path", "receipt_sha256"},
            "prior E0 observation")
    root = _absolute(prior["run_root"])
    _sha(prior["descriptor_sha256"], "prior descriptor SHA")
    descriptor, config, _submission = read_prepared(root, prior["descriptor_sha256"])
    probe = read_json(root / "probe_request.json")
    old = _bound_json(probe["contract_path"], probe["contract_sha256"])
    _require(_same(old["source_contract"], contract["source_contract"])
             and old["campaign_id"] == contract["campaign_id"]
             and _same(old["runtime_identity"]["model"], contract["runtime_identity"]["model"])
             and old["runtime_identity"]["dependency_lock"]["sha256"]
             == contract["runtime_identity"]["dependency_lock"]["sha256"],
             "prior E0 model/provider/dependency identity differs")
    current_core = {row["path"]: row for row in contract["runtime_identity"]["core_files"]}
    _require(all(_same(current_core.get(row["path"]), row)
                 for row in old["runtime_identity"]["core_files"]),
             "scientific core differs from the observed E0 source")
    _require(config["service_python"] == service_identity["entry_path"]
             and config["service_python_sha256"] == service_identity["executable_sha256"]
             and config["service_python_prefix"] == service_identity["prefix"]
             and config.get("language_model_only") is True,
             "prior E0 service environment/text-only identity differs")
    receipt_path = _absolute(prior["receipt_path"])
    expected = (Path(probe["output_root"]) / old["job"]["output_relpath"]
                / "e0_probe" / "observations" / "probe_receipt.json")
    _require(receipt_path == expected, "prior E0 receipt path is not producer-bound")
    receipt = _bound_json(str(receipt_path), prior["receipt_sha256"])
    _require(receipt.get("kind") == "path1_live_e0_observation"
             and type(receipt.get("schema_version")) is int and receipt["schema_version"] == 1
             and receipt.get("status") == "observations_complete_backend_execution_pending"
             and receipt.get("runtime_admission") is False
             and receipt.get("intent_id") == descriptor["intent_id"]
             and receipt.get("probe_request_sha256") == probe["request_sha256"]
             and receipt.get("campaign_id") == old["campaign_id"],
             "prior E0 did not complete the observation protocol")
    records = receipt.get("files")
    _require(type(records) is list, "prior E0 file inventory missing")
    names = []
    for record in records:
        _object(record, {"path", "size", "sha256"}, "prior E0 evidence file")
        name = record["path"]
        _require(type(name) is str and Path(name).name == name
                 and name not in {".", "..", "probe_receipt.json"}
                 and "/" not in name and "\\" not in name,
                 "unsafe prior E0 evidence file")
        path = _absolute(str(receipt_path.parent / name))
        _require(type(record["size"]) is int and record["size"] >= 0
                 and path.is_file() and path.stat().st_size == record["size"]
                 and file_sha256(path) == record["sha256"], "prior E0 evidence bytes differ")
        names.append(name)
    required = {"host_service.json", "service_arguments.json", "gpu.json", "dependencies.json",
                "model_files.json", "tokenizer.json", "endpoint.json"}
    required.update(name + suffix for name in CASES
                    for suffix in ("_request.json", "_response.json"))
    _require(len(set(names)) == len(names) and required <= set(names),
             "prior E0 evidence inventory is incomplete or duplicated")
    model = read_json(receipt_path.parent / "model_files.json")
    _require(_same(model.get("identity"), contract["runtime_identity"]["model"]),
             "prior E0 observed model identity differs")
    observations = receipt.get("observations")
    _require(type(observations) is list and [x.get("name") for x in observations] == list(CASES),
             "prior E0 six-case observations incomplete")
    action_schema = probe_runtime._response_format("json_schema")
    patch_schema = probe_runtime._experience_patch_response_format(SimpleNamespace(metadata={}))
    patch_schema = probe_runtime._remove_json_schema_keywords(patch_schema, {"uniqueItems"})
    for index, observation in enumerate(observations):
        name = CASES[index]
        payload = read_json(receipt_path.parent / (name + "_request.json"))
        response = read_json(receipt_path.parent / (name + "_response.json"))
        _require(type(observation.get("local_token_count")) is int
                 and observation["local_token_count"] >= 0
                 and payload.get("seed") == 2**62 + 1729 + index
                 and payload.get("model")
                 == contract["source_contract"]["provider"]["served_model_name"],
                 "prior E0 request identity differs")
        patch = name in {"patch", "patch_repair"}
        budget = 8192 if name == "patch" else 4096 if patch else 256
        _require(_same(payload.get("response_format"), patch_schema if patch else action_schema)
                 and type(payload.get("max_tokens")) is int and payload["max_tokens"] == budget
                 and _same(payload.get("temperature"), 1.0 if patch else 0.0)
                 and _same(payload.get("chat_template_kwargs"), {"enable_thinking": False}),
                 "prior E0 HTTP protocol differs")
        if name in {"near_context", "over_context"}:
            target = 65536 - 256 if name == "near_context" else 65536 + 32
            _require(0 <= target - observation["local_token_count"] <= 16,
                     "prior E0 context boundary was not exercised")
        observed = probe_runtime.verify_http_observation(
            payload, response, observation["local_token_count"], over_limit=name == "over_context")
        if name == "patch":
            _require(observed["content"]["operation"] == "patch"
                     and bool(observed["content"]["patches"]),
                     "prior E0 patch branch was not exercised")
        _require(all(_same(observation.get(key), value) for key, value in observed.items()),
                 "prior E0 observation differs from raw HTTP evidence")
    _require(file_sha256(receipt_path) == prior["receipt_sha256"],
             "prior E0 receipt changed during validation")
    return config


def read_runtime_plan(path: Path, *, contract: dict, request: dict | None = None) -> dict:
    """Validate launch inputs and the E0 prerequisite, never return RuntimeAdmission."""
    path = _absolute(str(path))
    initial_sha = file_sha256(path)
    value = _bound_json(str(path), initial_sha)
    _object(value, {"schema_version", "kind", "service_port", "service_python_identity",
                    "source_pins", "startup_seconds", "cleanup_seconds", "backend_probe_seeds",
                    "prior_e0", "language_model_only"}
            | ({"native_runtime"} if "native_runtime" in value else set())
            | ({"cuda_toolkit"} if "cuda_toolkit" in value else set()), "E1 runtime plan")
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1
             and value["kind"] == "path1_e1_runtime_plan"
             and contract["phase"] == "E1", "E1 runtime plan schema/phase differs")
    _require(type(value["service_port"]) is int and 1024 <= value["service_port"] <= 65535
             and type(value["startup_seconds"]) is int and value["startup_seconds"] == 1200
             and type(value["cleanup_seconds"]) is int and value["cleanup_seconds"] == 600
             and value["language_model_only"] is True, "E1 runtime launch policy differs")
    proof = value["service_python_identity"]
    _require(type(proof) is dict, "service interpreter proof missing")
    current = verify_python_interpreter(Path(proof["entry_path"]), proof["executable_sha256"],
                                        expected_prefix=proof["prefix"])
    _require(_same(proof, current), "service interpreter changed")
    pins = value["source_pins"]
    _require(type(pins) is list and len(pins) == len(BACKEND_MODULES), "backend pin count differs")
    modules = []
    for pin in pins:
        _object(pin, {"module_name", "path", "sha256"}, "backend source pin")
        _sha(pin["sha256"], "backend pin SHA")
        _require(file_sha256(_absolute(pin["path"])) == pin["sha256"], "backend source changed")
        modules.append(pin["module_name"])
    _require(set(modules) == BACKEND_MODULES, "backend module inventory differs")
    seeds = value["backend_probe_seeds"]
    _require(type(seeds) is list and len(seeds) == 2
             and all(type(seed) is int and 2**32 <= seed < 2**63 for seed in seeds)
             and len(set(seeds)) == 2 and seeds[0] % 2**32 == seeds[1] % 2**32,
             "two distinct high 63-bit probes with equal low 32 bits required")
    site = _e0_observations(value["prior_e0"], contract, proof)
    _require(("native_runtime" in value) == ("native_runtime" in site)
             and _same(value.get("native_runtime"), site.get("native_runtime")),
             "E1 native runtime must match the observed E0 environment")
    if "native_runtime" in value:
        from agentmemeval.path1_v1.operations.native_runtime import verify_file
        verify_file(value["native_runtime"])
    _require(("cuda_toolkit" in value) == ("cuda_toolkit" in site)
             and _same(value.get("cuda_toolkit"), site.get("cuda_toolkit")),
             "E1 CUDA toolkit must match the observed E0 environment")
    if "cuda_toolkit" in value:
        from agentmemeval.path1_v1.operations.cuda_toolkit import verify_file
        verify_file(value["cuda_toolkit"])
    if request is not None:
        _require(request["runtime_gate_path"] == str(path)
                 and request["runtime_gate_sha256"] == initial_sha,
                 "runtime plan is not bound to worker request")
        claim = _bound_json(request["claim_path"], request["claim_sha256"])
        store = CampaignStore(claim["control_directory"], contract["campaign_id"])
        rows = [row for row in store.read_registry().claims if row.run_id == claim["run_id"]]
        _require(len(rows) == 1 and rows[0].metadata.get("phase") == "E1"
                 and rows[0].metadata.get("runtime_plan_sha256") == initial_sha,
                 "runtime plan differs from reserved E1 claim")
    _require(file_sha256(path) == initial_sha, "runtime plan changed during validation")
    return value


def prepare_campaign(*, repository_root: Path, dependency_lock_relative: str,
                     model_manifest_path: Path, model_root: Path, runtime_plan_path: Path,
                     run_root: Path, control_directory: Path, expected_generation: int,
                     campaign_id: str, root_seed: int, seed_namespace: str,
                     python_executable: Path, python_sha256: str, python_prefix: str,
                     cluster: str, account: str, user: str, partition: str,
                     forbidden_environment_seeds: list[int],
                     forbidden_sampler_seeds: list[int]) -> Path:
    """Build/claim the 28-hand unit under the existing site control root.

    A post-reservation failure deliberately leaves a reserved claim for explicit
    recovery. Changing a directory or calling again is not an automatic retry.
    """
    from agentmemeval.path1_v1.operations.e1_prepare import RESOURCE_PROFILE, prepare

    _slug(campaign_id, "campaign ID")
    _slug(seed_namespace, "E1 seed namespace")
    _require(type(root_seed) is int and root_seed >= 0, "root_seed must be nonnegative integer")
    _require(type(expected_generation) is int and expected_generation >= 0, "invalid generation")
    forbidden_environment_seeds = _checked_seeds(forbidden_environment_seeds)
    forbidden_sampler_seeds = _checked_seeds(forbidden_sampler_seeds)
    repository = _absolute(str(repository_root))
    model = _absolute(str(model_root))
    manifest = _absolute(str(model_manifest_path))
    plan_path = _absolute(str(runtime_plan_path))
    run_root = _absolute(str(run_root), exists=False)
    _require(not run_root.exists() and run_root.parent.is_dir(), "E1 run root must be new")
    control = _absolute(str(control_directory))
    _require(control.is_dir(), "an existing site control root is required")
    _require(all(not run_root.is_relative_to(p) and not p.is_relative_to(run_root)
                 for p in (repository, model, control)), "E1 output overlaps protected roots")
    _require(_same(read_json(repository / PROFILE_PATH), RESOURCE_PROFILE),
             "E1 fixed execution profile differs")
    tracked = subprocess.run(["git", "-C", str(repository), "ls-files", "-z"],
                             capture_output=True, check=True).stdout.decode("utf-8").split("\0")
    dependencies = {p for p in tracked if p.startswith(("src/", "prompts/", "tools/"))
                    or p in {"pyproject.toml", ".gitattributes"}}
    ops = {p for p in dependencies if p.startswith("src/agentmemeval/path1_v1/operations/")}
    identity = build_runtime_identity(repository, core_paths=sorted(dependencies - ops),
        operations_paths=sorted(ops), config_path=PROFILE_PATH,
        dependency_lock_path=dependency_lock_relative,
        model_manifest_path=manifest, model_root=model)
    _require(not identity["dirty"] and identity["content_matches_head"]
             and not identity["untracked_declared_paths"]
             and not identity["uncovered_execution_paths"]
             and not identity["untracked_execution_paths"], "E1 source is not frozen/complete")
    contract = {"schema_version": 1, "kind": "path1_campaign_contract", "campaign_id": campaign_id,
        "phase": "E1", "source_contract": read_json(repository /
            "src/agentmemeval/path1_v1/assets/path1_source_contract.json"),
        "protocol_sha256": SOURCE_CONTRACT_SHA256, "runtime_identity": identity,
        "seed_namespace": seed_namespace,
        "cohort": {"formation_count": 1, "episodes_per_artifact_per_arm": 2,
                   "seed_inventory_sha256": "a" * 64, "freeze_receipt_sha256": None},
        "job": {"kind": "technical_closed_loop", "artifact_id": "technical_0001",
                "pair_ids": ["pair_0001", "pair_0002"], "arms": ["retained", "removed"],
                "domain": "source", "attempt": 1,
                "output_relpath": campaign_id + "/E1/technical_0001/attempt_01",
                "formation_receipt": None, "retry_evidence": None},
        "not_for_paper": True, "execution_admission": "not_granted_by_contract_validation"}
    plan = read_runtime_plan(plan_path, contract=contract)
    from agentmemeval.path1_v1.operations.e0_service_cli import read_prepared as read_e0

    _old_descriptor, site, _old_submission = read_e0(
        Path(plan["prior_e0"]["run_root"]), plan["prior_e0"]["descriptor_sha256"])
    _require(control == _absolute(site["control_root"])
             and (cluster, account, user, partition)
             == tuple(site[key] for key in ("cluster", "account", "user", "partition")),
             "E1 must retain the observed site and campaign control scope")
    _require((str(python_executable), python_sha256, python_prefix) == tuple(site[key]
             for key in ("python_executable", "python_sha256", "python_prefix")),
             "E1 launcher interpreter must match the observed environment")
    verify_python_interpreter(python_executable, python_sha256, expected_prefix=python_prefix)
    forbidden = list(forbidden_sampler_seeds)
    forbidden += (plan["backend_probe_seeds"] + [2**62 + 1729 + i for i in range(6)]
                  + [2**62 + 1000 + i for i in range(6)])
    inventory = {"schema_version": 1, "kind": "path1_worker_seed_inventory",
        "campaign_id": campaign_id, "phase": "E1", "seed_namespace": seed_namespace,
        "root_seed": root_seed, "environment_algorithm": "path1-sha256-63bit-v2",
        "sampler_algorithm": SAMPLER_ALGORITHM,
        "forbidden_environment_seeds": forbidden_environment_seeds,
        "forbidden_sampler_seeds": sorted(set(forbidden)),
        "artifacts": [{"artifact_id": "technical_0001",
            "pairs": [{"pair_id": "pair_0001", "episode_index": 0, "hero_seat": 0},
                      {"pair_id": "pair_0002", "episode_index": 1, "hero_seat": 1}],
            "streams": _planned_streams(design_from_contract(contract, root_seed=root_seed), 2)}]}
    contract["cohort"]["seed_inventory_sha256"] = content_digest(inventory)
    validate_seed_inventory(inventory, contract)
    validate_campaign_contract(contract)
    intent = canonical_intent_id(contract)
    worker_control = control / "e1_worker" / campaign_id
    _absolute(str(worker_control), exists=False)
    store = CampaignStore(worker_control, campaign_id)
    _require(store.database_path.exists() or expected_generation == 0,
             "new E1 worker store requires generation zero")
    with store.lock():
        if not store.database_path.exists():
            store.initialize(registry={"kind": "path1_e1_worker_registry"}, state={})
        snapshot = store.read_registry()
        _require(snapshot.generation == expected_generation, "campaign generation changed")
        _require(all(c.status in {"completed", "failed"} for c in snapshot.claims),
                 "unresolved campaign occupancy prevents E1 preparation")
        _require(not any(c.metadata.get("phase") == "E1" for c in snapshot.claims),
                 "E1 unit already reserved; explicit retry policy is not implemented")
        run_root.mkdir(exist_ok=False)
        output = run_root / "output"
        output.mkdir()
        payload = output / contract["job"]["output_relpath"] / "payload"
        claim = store.claim_run(expected_generation=expected_generation,
            run_id="e1_technical_0001", intent_id=intent, run_directory=payload,
            metadata={"phase": "E1", "kind": "technical_closed_loop", "not_for_paper": True,
                      "runtime_plan_sha256": file_sha256(plan_path), "planned_hands": 28})
    claim_value = {"schema_version": 1, "kind": "path1_worker_claim", "campaign_id": campaign_id,
        "intent_id": intent, "control_directory": str(worker_control), "run_id": claim.run_id,
        "run_directory": str(payload), "token": claim.token}
    for name, value in (("contract.json", contract), ("seed_inventory.json", inventory),
                        ("claim.json", claim_value)):
        write_exclusive_json(run_root / name, value)
    request = {"schema_version": 1, "kind": "path1_worker_request", "intent_id": intent,
        "contract_path": str(run_root / "contract.json"),
        "contract_sha256": file_sha256(run_root / "contract.json"),
        "repository_root": str(repository), "model_manifest_path": str(manifest),
        "model_root": str(model), "seed_inventory_path": str(run_root / "seed_inventory.json"),
        "seed_inventory_sha256": file_sha256(run_root / "seed_inventory.json"),
        "runtime_gate_path": str(plan_path), "runtime_gate_sha256": file_sha256(plan_path),
        "output_root": str(output), "claim_path": str(run_root / "claim.json"),
        "claim_sha256": file_sha256(run_root / "claim.json"), "artifact_root": None}
    request["request_sha256"] = content_digest(request)
    write_exclusive_json(run_root / "worker_request.json", request)
    prepared = prepare(worker_request_path=run_root / "worker_request.json",
        prepared_directory=run_root / "prepared", python_executable=python_executable,
        python_sha256=python_sha256, python_prefix=python_prefix, cluster=cluster,
        account=account, user=user, partition=partition,
        work_dir=str(payload.parent), script_path=str(run_root / "prepared/e1.sh"),
        runtime_plan_path=plan_path)
    write_exclusive_json(run_root / "campaign_prepared.json", {
        "schema_version": 1, "kind": "path1_e1_campaign_prepared", "intent_id": intent,
        "prepared_path": str(prepared), "prepared_sha256": file_sha256(prepared),
        "worker_request_sha256": request["request_sha256"],
        "claim_generation": claim.created_generation,
        "planned_hands": {"formation": 12, "evaluation": 16, "total": 28},
        "runtime_admission": False, "scientific_admission": False, "submission_performed": False})
    return prepared


def execute_prepared(prepared_path: Path, expected_descriptor_sha256: str, *, lookup: bool) -> dict:
    """Explicit scheduler action through the durable executor; never retry UNKNOWN.

    Lookup consumes frozen submission bytes without pretending the worker is
    still reserved or its payload absent. Submit performs full fresh validation.
    """
    from agentmemeval.path1_v1.operations.e0_service_cli import read_prepared as read_e0
    from agentmemeval.path1_v1.operations.e1_preflight import E1PreSubmitVerifier
    from agentmemeval.path1_v1.operations.e1_prepare import (
        RESOURCE_PROFILE,
        E1LaunchFiles,
        read_prepared,
    )
    from agentmemeval.path1_v1.operations.intent_executor import (
        DurableIntentExecutor,
        VerifiedCommand,
    )
    from agentmemeval.path1_v1.operations.slurm import SubmissionRequest, SubmitSpec
    from agentmemeval.path1_v1.operations.user_capacity import (
        PinnedUserCommand,
        UserAssociationReader,
    )

    _require(type(lookup) is bool, "lookup must be boolean")
    path = _absolute(str(prepared_path))
    descriptor = _bound_json(str(path), expected_descriptor_sha256)
    _object(descriptor, {"schema_version", "kind", "worker_request_path", "worker_file_sha256",
        "worker_request_sha256", "python_identity", "resource_profile", "submission",
        "submission_request_sha256", "script_file", "descriptor_sha256", "runtime_plan_path",
        "runtime_plan_sha256"}, "E1 prepared descriptor")
    _require(type(descriptor.get("schema_version")) is int and descriptor["schema_version"] == 2
             and descriptor.get("kind") == "path1_e1_prepared"
             and _same(descriptor.get("resource_profile"), RESOURCE_PROFILE)
             and descriptor.get("descriptor_sha256") == content_digest({
                 key: value for key, value in descriptor.items() if key != "descriptor_sha256"}),
             "explicit E1 action requires a frozen v2 descriptor")
    raw = descriptor["submission"]
    request = SubmissionRequest(raw["campaign_id"], raw["intent_id"],
                                SubmitSpec.from_payload(raw["spec"]))
    _require(request.request_sha256 == descriptor["submission_request_sha256"],
             "E1 submission hash differs")
    worker_request = _bound_json(descriptor["worker_request_path"],
                                 descriptor["worker_file_sha256"])
    _require(worker_request["request_sha256"] == descriptor["worker_request_sha256"]
             == content_digest({k: v for k, v in worker_request.items() if k != "request_sha256"}),
             "E1 worker request differs")
    plan = _bound_json(descriptor["runtime_plan_path"], descriptor["runtime_plan_sha256"])
    _require(worker_request["runtime_gate_path"] == descriptor["runtime_plan_path"]
             and worker_request["runtime_gate_sha256"] == descriptor["runtime_plan_sha256"],
             "E1 worker/plan binding differs")
    contract = _bound_json(worker_request["contract_path"], worker_request["contract_sha256"])
    _require(contract["phase"] == "E1" and canonical_intent_id(contract) == request.intent_id
             == worker_request["intent_id"] and contract["campaign_id"] == request.campaign_id,
             "E1 contract/submission identity differs")
    claim = _bound_json(worker_request["claim_path"], worker_request["claim_sha256"])
    _old, config, _submission = read_e0(Path(plan["prior_e0"]["run_root"]),
                                       plan["prior_e0"]["descriptor_sha256"])
    expected_worker_control = Path(config["control_root"]) / "e1_worker" / request.campaign_id
    _require(claim["control_directory"] == str(expected_worker_control)
             and (request.spec.cluster, request.spec.account, request.spec.user,
                  request.spec.partition) == tuple(config[key] for key in (
                      "cluster", "account", "user", "partition")),
             "E1 submission changed campaign/site scope")
    if not lookup:
        current = read_prepared(path)
        _require(_same(current.descriptor, descriptor), "E1 prepared files changed")
    commands = config["commands"]

    def command(name):
        row = commands[name]
        return VerifiedCommand((row["path"],), ((row["path"], row["sha256"]),))

    reader = UserAssociationReader(cluster=request.spec.cluster, account=request.spec.account,
        user=request.spec.user,
        commands={name: PinnedUserCommand(**commands[name])
                  for name in ("sacctmgr", "squeue", "sacct")},
        since=datetime.fromisoformat(config["accounting_since"]))
    preflight = path.parent / "preflight"
    if not lookup:
        _require(not preflight.is_symlink(), "preflight symlink rejected")
        if not preflight.exists():
            preflight.mkdir(exist_ok=False)
    # The lookup branch never calls this producer or requires a live worker.
    executor = DurableIntentExecutor(Path(config["control_root"]),
        cluster=request.spec.cluster, account=request.spec.account, command=command("sbatch"),
        launch_files=lambda _: E1LaunchFiles(path),
        pre_submit_verifier=(E1PreSubmitVerifier(preflight, partition_command=command("scontrol"))
                             if not lookup else None), account_reader=reader)
    result = (executor.lookup(request, timeout_seconds=60) if lookup
              else executor.submit_once(request, timeout_seconds=60))
    return {"descriptor_sha256": expected_descriptor_sha256, "lookup_only": lookup,
            "receipt": asdict(result) if result is not None else None,
            "runtime_admission": False, "scientific_admission": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prep = actions.add_parser("prepare")
    prep.add_argument("--input", type=Path, required=True)
    prep.add_argument("--execute", action="store_true",
                      help="create local E1 reservation and files, never submit")
    for name in ("submit", "lookup"):
        command = actions.add_parser(name)
        command.add_argument("--prepared", type=Path, required=True)
        command.add_argument("--descriptor-sha256", required=True)
    args = parser.parse_args(argv)
    if args.action != "prepare":
        import json

        print(json.dumps(execute_prepared(args.prepared, args.descriptor_sha256,
                                         lookup=args.action == "lookup"), sort_keys=True))
        return 0
    values = read_json(_absolute(str(args.input)))
    if not args.execute:
        print("plan-only: no claim, files, scheduler call, or runtime admission created")
        return 0
    for name in ("repository_root", "model_manifest_path", "model_root", "runtime_plan_path",
                 "run_root", "control_directory", "python_executable"):
        values[name] = Path(values[name])
    print(prepare_campaign(**values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
