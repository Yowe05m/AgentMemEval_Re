"""Real files/Git/SQLite preparation fixtures; synthetic HTTP is never GPU evidence."""
from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentmemeval.path1_v1.operations import (
    e0_service_cli,
    e1_prepare,
    intent_executor,
    runtime_probe,
    worker,
)
from agentmemeval.path1_v1.operations import e1_campaign as campaign
from agentmemeval.path1_v1.operations.contracts import canonical_json, content_digest
from agentmemeval.path1_v1.operations.interpreters import verify_python_interpreter
from agentmemeval.path1_v1.operations.state import CampaignStore
from agentmemeval.path1_v1.operations.submission_process import file_sha256, read_json
from path1_v1.unit.test_e0_service_cli import fixture as service_fixture
from path1_v1.unit.test_operations_contracts import _git
from path1_v1.unit.test_operations_worker import make_worker_fixture, write_json
from path1_v1.unit.test_runtime_probe import response


def write_observations(root, probe, contract):
    """Exercise actual historical HTTP validation using explicitly synthetic bytes."""
    directory = (Path(probe["output_root"]) / contract["job"]["output_relpath"]
                 / "e0_probe/observations")
    directory.mkdir(parents=True)
    for name in ("host_service", "service_arguments", "gpu", "dependencies", "tokenizer",
                 "endpoint"):
        write_json(directory / (name + ".json"), {"offline_fixture": True})
    write_json(directory / "model_files.json", {"identity": contract["runtime_identity"]["model"]})
    patch_schema = runtime_probe._experience_patch_response_format(SimpleNamespace(metadata={}))
    patch_schema = runtime_probe._remove_json_schema_keywords(patch_schema, {"uniqueItems"})
    section = patch_schema["json_schema"]["schema"]["properties"]["patches"]["items"][
        "anyOf"][0]["properties"]["section"]["enum"][0]
    example = {"operation": "patch", "patches": [{"section": section, "op": "add",
        "rule_id": "", "when": ["technical fixture"], "guidance": "Evaluate legal actions.",
        "invalidators": ["fixture ends"], "basis": "model_prior", "status": "tentative",
        "supporting_trajectory_ids": [], "supporting_fact_ids": []}], "fact_assessments": [],
        "calibration_note": "E0 test fixture", "self_check": "not GPU evidence"}
    observations = []
    for index, name in enumerate(campaign.CASES):
        patch = name in {"patch", "patch_repair"}
        message = ("Return one valid technical patch: " + json.dumps(example)
                   if name == "patch" else "fixture")
        payload = {"model": contract["source_contract"]["provider"]["served_model_name"],
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 8192 if name == "patch" else 4096 if patch else 256,
            "temperature": 1.0 if patch else 0.0, "seed": 2**62 + 1729 + index,
            "response_format": patch_schema if patch else runtime_probe._response_format(
                "json_schema"), "chat_template_kwargs": {"enable_thinking": False}}
        count = 65536 - 256 if name == "near_context" else 65536 + 32 if (
            name == "over_context") else 20
        observed = response(payload, count)
        write_json(directory / (name + "_request.json"), payload)
        write_json(directory / (name + "_response.json"), observed)
        result = runtime_probe.verify_http_observation(payload, observed, count,
                                                       over_limit=name == "over_context")
        result.update(name=name, local_token_count=count)
        observations.append(result)
    receipt = {"schema_version": 1, "kind": "path1_live_e0_observation",
        "intent_id": probe["intent_id"], "probe_request_sha256": probe["request_sha256"],
        "campaign_id": contract["campaign_id"], "runtime_admission": False,
        "status": "observations_complete_backend_execution_pending", "observations": observations,
        "files": [{"path": p.name, "size": p.stat().st_size, "sha256": file_sha256(p)}
                  for p in sorted(directory.iterdir())]}
    path = directory / "probe_receipt.json"
    write_json(path, receipt)
    return path


def plan_value(tmp_path, proof, prior):
    pins = []
    for index, name in enumerate(sorted(campaign.BACKEND_MODULES)):
        file = tmp_path / ("backend_" + str(index) + ".py")
        file.write_bytes(b"# synthetic backend source, not imported\n")
        pins.append({"module_name": name, "path": str(file), "sha256": file_sha256(file)})
    return {"schema_version": 1, "kind": "path1_e1_runtime_plan", "service_port": 18232,
        "service_python_identity": proof, "source_pins": pins, "startup_seconds": 1200,
        "cleanup_seconds": 600, "backend_probe_seeds": [2**62 + 2000, 2**61 + 2000],
        "prior_e0": prior, "language_model_only": True}


@pytest.fixture
def runtime_plan_fixture(tmp_path, monkeypatch):
    data = make_worker_fixture(tmp_path)
    prior = tmp_path / "prior"
    prior.mkdir()
    old = copy.deepcopy(data["contract"])
    old["phase"] = "E0"
    write_json(prior / "contract.json", old)
    probe = {"intent_id": "a" * 64, "request_sha256": "b" * 64,
             "contract_path": str(prior / "contract.json"),
             "contract_sha256": file_sha256(prior / "contract.json"),
             "output_root": str(prior / "output")}
    write_json(prior / "probe_request.json", probe)
    receipt = write_observations(prior, probe, old)
    python = Path(sys.executable).absolute()
    proof = verify_python_interpreter(python, file_sha256(python))
    config = {"service_python": proof["entry_path"],
              "service_python_sha256": proof["executable_sha256"],
              "service_python_prefix": proof["prefix"], "language_model_only": True}
    # Only this old-descriptor boundary is a stub; full Linux producer test below
    # uses service.prepare/read_prepared without replacing either function.
    monkeypatch.setattr(e0_service_cli, "read_prepared", lambda *_:
                        ({"intent_id": probe["intent_id"]}, config, None))
    plan = plan_value(tmp_path, proof, {"run_root": str(prior), "descriptor_sha256": "c" * 64,
        "receipt_path": str(receipt), "receipt_sha256": file_sha256(receipt)})
    path = tmp_path / "runtime_plan.json"
    write_json(path, plan)
    return {**data, "plan": plan, "plan_path": path, "receipt": receipt}


def test_prior_observation_is_validated_without_granting_runtime(runtime_plan_fixture):
    data = runtime_plan_fixture
    value = campaign.read_runtime_plan(data["plan_path"], contract=data["contract"])
    assert value == data["plan"] and "runtime_admission" not in value
    assert not data["output"].exists()


@pytest.mark.parametrize("change", ["incomplete", "raw_missing", "raw_tampered", "duplicate",
                                    "wrong_model", "changed_core", "wrong_protocol"])
def test_prior_observations_fail_closed(runtime_plan_fixture, change):
    data = runtime_plan_fixture
    receipt = read_json(data["receipt"])
    contract = copy.deepcopy(data["contract"])
    if change == "incomplete":
        receipt["status"] = "incomplete"
    elif change == "duplicate":
        receipt["files"].append(receipt["files"][0])
    elif change == "wrong_model":
        contract["runtime_identity"]["model"]["manifest_sha256"] = "d" * 64
    elif change == "changed_core":
        contract["runtime_identity"]["core_files"][0]["sha256"] = "d" * 64
    else:
        path = data["receipt"].parent / "action_request.json"
        if change == "raw_missing":
            path.unlink()
        elif change == "raw_tampered":
            path.write_bytes(b"{}")
        else:
            payload = read_json(path)
            payload["temperature"] = 1.0
            write_json(path, payload)
            for row in receipt["files"]:
                if row["path"] == path.name:
                    row.update(size=path.stat().st_size, sha256=file_sha256(path))
    write_json(data["receipt"], receipt)
    data["plan"]["prior_e0"]["receipt_sha256"] = file_sha256(data["receipt"])
    write_json(data["plan_path"], data["plan"])
    with pytest.raises((ValueError, FileNotFoundError)):
        campaign.read_runtime_plan(data["plan_path"], contract=contract)


@pytest.mark.parametrize("field,value", [("service_port", True), ("startup_seconds", 1200.0),
    ("cleanup_seconds", 120), ("backend_probe_seeds", [1, 2]),
    ("backend_probe_seeds", [2**62 + 1, 2**61 + 2]),
    ("backend_probe_seeds", [2**62, 2**62]), ("language_model_only", False),
    ("runtime_admission", True)])
def test_runtime_plan_rejects_unfrozen_or_self_admitted_fields(runtime_plan_fixture, field, value):
    data = runtime_plan_fixture
    data["plan"][field] = value
    write_json(data["plan_path"], data["plan"])
    with pytest.raises(ValueError):
        campaign.read_runtime_plan(data["plan_path"], contract=data["contract"])


def test_plan_only_cli_does_not_mutate_campaign(tmp_path, monkeypatch):
    path = tmp_path / "input.json"
    path.write_bytes(b"{}")
    monkeypatch.setattr(campaign, "prepare_campaign", lambda **_:
                        pytest.fail("plan-only created a reservation"))
    assert campaign.main(["prepare", "--input", str(path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["input.json"]


@pytest.mark.parametrize("values", [[True], [1, True], [1, 1], [2, 1], [-1], [2**63]])
def test_forbidden_seed_input_rejects_coercion_and_duplicates(values):
    with pytest.raises(ValueError, match="sorted unique int63"):
        campaign._checked_seeds(values)


def make_campaign_fixture(tmp_path, monkeypatch, *, prior_attempt=1):
    repo, _config_path, config = service_fixture(tmp_path, monkeypatch)
    # Required execution modules are tracked fixture identities, never a claim
    # that these tiny files implement a running GPU backend.
    operations = repo / "src/agentmemeval/path1_v1/operations"
    for name in ("worker", "e1_prepare", "e1_launcher", "e1_preflight", "e1_campaign",
                 "e1_runtime", "e1_backend_runtime", "interpreters", "intent_executor",
                 "submission_process", "user_capacity"):
        (operations / (name + ".py")).write_bytes(b"# offline identity fixture\n")
    (repo / campaign.PROFILE_PATH).write_bytes(canonical_json(e1_prepare.RESOURCE_PROFILE))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "E1 fixture closure")
    prior = tmp_path / "prior"
    if prior_attempt == 1:
        descriptor = e0_service_cli.prepare(repo, prior, "campaign", "configs/path1_v1/site.json")
    else:
        # The genuine versioned retry producer/reader is exercised. Eligibility
        # is an explicit offline double, independently tested in service_retry.
        from agentmemeval.path1_v1.operations import service_retry

        first = tmp_path / "prior_first"
        initial = e0_service_cli.prepare(repo, first, "campaign", "configs/path1_v1/site.json")
        previous = read_json(first / "contract.json")
        old_probe = read_json(first / "probe_request.json")
        scope = content_digest({"cluster": config["cluster"], "account": config["account"]})
        executor_store = CampaignStore(Path(config["control_root"]) / "ie1" / scope[:16],
                                       "executor_" + scope[:32])
        record = {"intent_id": initial["intent_id"], "campaign_id": "campaign",
                  "probe_request_path": str(first / "probe_request.json"),
                  "probe_request_file_sha256": file_sha256(first / "probe_request.json"),
                  "worker_request_sha256": old_probe["request_sha256"]}
        with executor_store.lock():
            executor_store.initialize(registry={intent_executor.NAMESPACE: {
                "intents": {initial["intent_id"]: record}}})
        eligibility = service_retry.ServiceRetryEligibility(executor_store.campaign_id,
            str(executor_store.directory), initial["intent_id"], content_digest(previous),
            "b" * 64, "c" * 64, "d" * 64, 12040, _issuer=service_retry._ISSUER)
        monkeypatch.setattr(service_retry, "resolve_service_retry", lambda *_: eligibility)
        descriptor = e0_service_cli.prepare(repo, prior, "campaign", "configs/path1_v1/site.json",
            attempt=prior_attempt, prior_intent_id=initial["intent_id"])
    probe, old = read_json(prior / "probe_request.json"), read_json(prior / "contract.json")
    receipt = write_observations(prior, probe, old)
    proof = verify_python_interpreter(Path(config["service_python"]),
        config["service_python_sha256"], expected_prefix=config["service_python_prefix"])
    plan = plan_value(tmp_path, proof, {"run_root": str(prior),
        "descriptor_sha256": descriptor["descriptor_sha256"], "receipt_path": str(receipt),
        "receipt_sha256": file_sha256(receipt)})
    plan_path = tmp_path / "runtime_plan.json"
    write_json(plan_path, plan)
    store = CampaignStore(tmp_path / "control/e1_worker/campaign", "campaign")
    args = {"repository_root": repo, "dependency_lock_relative": config["dependency_lock"],
        "model_manifest_path": tmp_path / "model_manifest.json", "model_root": tmp_path / "model",
        "runtime_plan_path": plan_path, "run_root": tmp_path / "e1",
        "control_directory": tmp_path / "control", "expected_generation": 0,
        "campaign_id": "campaign",
        "root_seed": 20261127, "seed_namespace": "campaign_e1", "python_executable": Path(
            config["python_executable"]), "python_sha256": config["python_sha256"],
        "python_prefix": config["python_prefix"], "cluster": "cluster", "account": "account",
        "user": "owner", "partition": "gpu", "forbidden_environment_seeds": [],
        "forbidden_sampler_seeds": []}
    prepared_path = campaign.prepare_campaign(**args)
    return {"prepared_path": prepared_path, "arguments": args, "plan": plan,
            "plan_path": plan_path, "store": store, "repo": repo, "config": config}


@pytest.mark.skipif(sys.platform != "linux", reason="actual Slurm absolute argv require Linux")
@pytest.mark.skipif("attempt" not in inspect.signature(e0_service_cli.prepare).parameters,
                    reason="requires the separately frozen E0 retry integration")
def test_real_e1_prepare_accepts_actual_attempt_two_descriptor(tmp_path, monkeypatch):
    data = make_campaign_fixture(tmp_path, monkeypatch, prior_attempt=2)
    prior = Path(data["plan"]["prior_e0"]["run_root"])
    assert read_json(prior / "contract.json")["job"]["attempt"] == 2
    prepared = e1_prepare.read_prepared(data["prepared_path"])
    assert prepared.contract["phase"] == "E1"
    assert prepared.contract["job"]["attempt"] == 1
    assert prepared.runtime_plan["prior_e0"]["run_root"] == str(prior)
    assert data["store"].read_registry().claims[0].status == "reserved"
    assert not prepared.output_directory.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="actual Slurm absolute argv require Linux")
def test_real_prepare_from_prior_descriptor_reserves_exact_28_hand_unit(tmp_path, monkeypatch):
    data = make_campaign_fixture(tmp_path, monkeypatch)
    prepared_path, plan, store = data["prepared_path"], data["plan"], data["store"]
    args, plan_path = data["arguments"], data["plan_path"]
    prepared = e1_prepare.read_prepared(prepared_path)
    assert prepared.runtime_plan == plan and prepared.descriptor["schema_version"] == 2
    assert prepared.contract["phase"] == "E1" and prepared.contract["not_for_paper"] is True
    design = worker.design_from_contract(prepared.contract, root_seed=20261127)
    assert design.formation_episodes * design.formation_episode_hands == 12
    assert 2 * 2 * design.evaluation_episode_hands == 16
    assert len(prepared.seed_inventory["artifacts"]) == 1
    assert len(prepared.seed_inventory["artifacts"][0]["pairs"]) == 2
    assert set(plan["backend_probe_seeds"]) <= set(
        prepared.seed_inventory["forbidden_sampler_seeds"])
    assert {2**62 + 1000 + i for i in range(6)} <= set(
        prepared.seed_inventory["forbidden_sampler_seeds"])
    registry = store.read_registry()
    assert registry.generation == 1 and len(registry.claims) == 1
    assert registry.claims[0].status == "reserved" and registry.claims[0].terminal_receipt is None
    assert registry.claims[0].metadata["planned_hands"] == 28
    assert not prepared.output_directory.exists()
    assert prepared.submission.spec.walltime_seconds == 14400
    assert prepared.submission.spec.cpus_per_task == 8
    assert prepared.submission.spec.memory_policy == "site_default"
    receipt = read_json(tmp_path / "e1/campaign_prepared.json")
    assert receipt["submission_performed"] is False and receipt["runtime_admission"] is False
    with pytest.raises(ValueError, match="new"):
        campaign.prepare_campaign(**args)
    assert store.read_registry() == registry
    with pytest.raises(ValueError, match="unresolved"):
        campaign.prepare_campaign(**dict(args, run_root=tmp_path / "attempt_elsewhere",
                                         expected_generation=1))
    assert not (tmp_path / "attempt_elsewhere").exists()
    assert store.read_registry() == registry
    # Routing fixture only: scheduler methods are replaced, no job is submitted.
    calls = []

    class RoutedExecutor:
        def __init__(self, control, **kwargs):
            assert control == Path(data["config"]["control_root"])
            assert kwargs["launch_files"](prepared.submission).prepared_path == prepared_path
            assert kwargs["command"].argv_prefix == (
                data["config"]["commands"]["sbatch"]["path"],)
            self.kwargs = kwargs

        def submit_once(self, request, *, timeout_seconds):
            assert request == prepared.submission and timeout_seconds == 60
            assert self.kwargs["pre_submit_verifier"] is not None
            calls.append("submit")
            return None

        def lookup(self, request, *, timeout_seconds):
            assert request == prepared.submission and timeout_seconds == 60
            assert self.kwargs["pre_submit_verifier"] is None
            calls.append("lookup")
            return None

    monkeypatch.setattr(intent_executor, "DurableIntentExecutor", RoutedExecutor)
    action = campaign.execute_prepared(prepared_path, file_sha256(prepared_path), lookup=False)
    assert action["runtime_admission"] is False and action["receipt"] is None
    prepared.output_directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        e1_prepare.read_prepared(prepared_path)
    action = campaign.execute_prepared(prepared_path, file_sha256(prepared_path), lookup=True)
    assert action["lookup_only"] is True and calls == ["submit", "lookup"]
    assert store.read_registry() == registry
    # Changing the plan AND re-signing the mutable request/descriptor must still
    # lose against the original immutable reservation metadata.
    changed = copy.deepcopy(plan)
    changed["service_port"] += 1
    write_json(plan_path, changed)
    request = copy.deepcopy(prepared.request)
    request["runtime_gate_sha256"] = file_sha256(plan_path)
    request["request_sha256"] = content_digest({k: v for k, v in request.items()
                                               if k != "request_sha256"})
    with pytest.raises(ValueError, match="reserved E1 claim"):
        campaign.read_runtime_plan(plan_path, contract=prepared.contract, request=request)
    assert store.read_registry() == registry
