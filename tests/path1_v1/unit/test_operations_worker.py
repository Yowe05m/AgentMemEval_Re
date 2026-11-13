from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from agentmemeval.path1_v1.operations import worker
from agentmemeval.path1_v1.operations.contracts import (
    canonical_intent_id,
    canonical_json,
    content_digest,
)
from agentmemeval.path1_v1.operations.state import CampaignStore


def write_json(path, value):
    path.write_bytes(canonical_json(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_worker_fixture(tmp_path, phase="E1"):
    """Synthetic source/model files; never runtime or GPU admission evidence."""
    from path1_v1.unit.test_operations_contracts import _contract, _git, evidence

    data = evidence.__wrapped__(tmp_path)
    module_path = data["repo"] / "src/agentmemeval/path1_v1/operations/worker.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# offline worker source identity fixture\n", encoding="utf-8")
    _git(data["repo"], "add", "src/agentmemeval/path1_v1/operations/worker.py")
    _git(data["repo"], "commit", "-qm", "add offline worker identity fixture")
    data["build_kwargs"]["operations_paths"].append("src/agentmemeval/path1_v1/operations/worker.py")
    data["identity"] = worker.build_runtime_identity(data["repo"], **data["build_kwargs"])
    contract = _contract(data, phase)
    count = contract["cohort"]["formation_count"]
    pair_count = contract["cohort"]["episodes_per_artifact_per_arm"]
    root_seed = 20261113
    artifacts = []
    for index in range(count):
        artifact_id = f"artifact_{index + 1:04d}"
        item = copy.deepcopy(contract)
        item["job"]["artifact_id"] = artifact_id
        design = worker.design_from_contract(item, root_seed=root_seed)
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "pairs": [
                    {"pair_id": f"pair_{i + 1:04d}", "episode_index": i, "hero_seat": i % 8}
                    for i in range(pair_count)
                ],
                "streams": worker._planned_streams(design, pair_count),
            }
        )
    inventory = {
        "schema_version": 1,
        "kind": "path1_worker_seed_inventory",
        "campaign_id": contract["campaign_id"],
        "phase": phase,
        "seed_namespace": contract["seed_namespace"],
        "root_seed": root_seed,
        "environment_algorithm": "path1-sha256-63bit-v2",
        "sampler_algorithm": worker.SAMPLER_ALGORITHM,
        "forbidden_environment_seeds": [],
        "forbidden_sampler_seeds": [],
        "artifacts": artifacts,
    }
    contract["cohort"]["seed_inventory_sha256"] = content_digest(inventory)
    intent = canonical_intent_id(contract)
    output_root = tmp_path / "out"
    output_root.mkdir()
    output = output_root / contract["job"]["output_relpath"] / "payload"
    store = CampaignStore(tmp_path / "control", contract["campaign_id"])
    with store.lock():
        initial = store.initialize(registry={}, state={})
        claim = store.claim_run(
            expected_generation=initial.generation,
            run_id="worker_fixture",
            intent_id=intent,
            run_directory=output,
        )
    claim_json = {
        "schema_version": 1,
        "kind": "path1_worker_claim",
        "campaign_id": contract["campaign_id"],
        "intent_id": intent,
        "control_directory": str(store.directory),
        "run_id": claim.run_id,
        "run_directory": str(output),
        "token": claim.token,
    }
    contract_path, seed_path, claim_path = (
        tmp_path / name
        for name in (
            "contract.json",
            "seeds.json",
            "claim.json",
        )
    )
    request = {
        "schema_version": 1,
        "kind": "path1_worker_request",
        "intent_id": intent,
        "contract_path": str(contract_path),
        "contract_sha256": write_json(contract_path, contract),
        "repository_root": str(data["repo"]),
        "model_manifest_path": str(data["manifest"]),
        "model_root": str(data["model"]),
        "seed_inventory_path": str(seed_path),
        "seed_inventory_sha256": write_json(seed_path, inventory),
        "runtime_gate_path": None,
        "runtime_gate_sha256": None,
        "output_root": str(output_root),
        "claim_path": str(claim_path),
        "claim_sha256": write_json(claim_path, claim_json),
        "artifact_root": None,
    }
    request["request_sha256"] = content_digest(request)
    request_path = tmp_path / "worker_request.json"
    write_json(request_path, request)
    return {
        **data,
        "request": request,
        "request_path": request_path,
        "contract": contract,
        "inventory": inventory,
        "store": store,
        "claim": claim_json,
        "output": output,
    }


class OfflineVerifier:
    """Test double proving only the worker boundary, never real service support."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path

    def verify_in_job(self, request):
        identity = request.contract["runtime_identity"]
        probes = []
        for index, purpose in enumerate(
            (
                "sampler_seed_execution_63bit",
                "tokenizer_total_context",
                "structured_schema",
                "loaded_model_and_source_identity",
            )
        ):
            path = self.tmp_path / f"fake_probe_{index}.json"
            path.write_bytes(b'{"offline_fixture":true}')
            probes.append(
                worker.RuntimeProbeFile(
                    purpose,
                    str(path),
                    path.stat().st_size,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        now = datetime.now(UTC)
        return worker.RuntimeAdmission(
            request.intent_id,
            request.request_sha256,
            identity["core_manifest_sha256"],
            identity["operations_manifest_sha256"],
            identity["config"]["sha256"],
            identity["model"]["manifest_sha256"],
            "12345",
            "offline-test-service",
            now - timedelta(seconds=1),
            now + timedelta(minutes=2),
            tuple(probes),
        )


def test_valid_request_is_read_only_and_cli_cannot_self_attest(tmp_path):
    fixture = make_worker_fixture(tmp_path)
    before = fixture["store"].read_registry()
    result = worker.validate_worker_request(fixture["request_path"])
    assert result.intent_id == fixture["request"]["intent_id"]
    assert result.output_directory == fixture["output"]
    with pytest.raises(worker.RuntimeAdmissionRequired, match="no trusted"):
        worker.run_worker(fixture["request_path"])
    assert worker.main(["--request", str(fixture["request_path"])]) == 2
    assert fixture["store"].read_registry() == before
    assert not fixture["output"].exists()


@pytest.mark.parametrize("change", ["seed", "pair_index", "pair_seat", "claim", "source", "extra"])
def test_worker_rejects_tampered_inputs_before_admission(tmp_path, change):
    fixture = make_worker_fixture(tmp_path)
    request = fixture["request"]
    if change in {"seed", "pair_index", "pair_seat"}:
        inventory = fixture["inventory"]
        if change == "seed":
            first = next(iter(inventory["artifacts"][0]["streams"]))
            inventory["artifacts"][0]["streams"][first] += 1
        else:
            inventory["artifacts"][0]["pairs"][1][
                "episode_index" if change == "pair_index" else "hero_seat"
            ] = 0
        request["seed_inventory_sha256"] = write_json(
            fixture["tmp"] / "seeds.json",
            inventory,
        )
    elif change == "claim":
        fixture["claim"]["token"] = "wrong"
        request["claim_sha256"] = write_json(fixture["tmp"] / "claim.json", fixture["claim"])
    elif change == "source":
        (fixture["repo"] / "src/core.py").write_bytes(b"# changed source")
    else:
        request["verified"] = True
    request["request_sha256"] = content_digest(
        {k: v for k, v in request.items() if k != "request_sha256"}
    )
    write_json(fixture["request_path"], request)
    with pytest.raises((ValueError, worker.RuntimeAdmissionRequired)):
        worker.validate_worker_request(fixture["request_path"])
    assert not fixture["output"].exists()


def test_worker_consumes_real_claim_once_and_keeps_job_container(tmp_path, monkeypatch):
    fixture = make_worker_fixture(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    fixture["output"].parent.mkdir(parents=True)
    log = fixture["output"].parent / "slurm-12345.out"
    log.write_text("scheduler log", encoding="utf-8")
    calls = []

    def offline_execute(request, factory):
        assert fixture["store"].read_registry().claims[0].status == "started"
        calls.append((request.intent_id, type(factory).__name__))
        return {"completed_units": [{"unit": "offline_fixture"}]}

    monkeypatch.setattr(worker, "_execute", offline_execute)
    receipt = worker.run_worker(fixture["request_path"], runtime_verifier=OfflineVerifier(tmp_path))
    assert receipt["status"] == "completed" and not receipt["primary_analysis_admitted"]
    assert not receipt["service_release_verified"]
    assert log.read_text() == "scheduler log"
    assert fixture["store"].read_registry().claims[0].status == "completed"
    with pytest.raises(ValueError, match="already exists"):
        worker.run_worker(fixture["request_path"], runtime_verifier=OfflineVerifier(tmp_path))
    assert len(calls) == 1


def test_worker_failed_unit_keeps_terminal_and_consumed_claim(tmp_path, monkeypatch):
    fixture = make_worker_fixture(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    def fail(_request, _factory):
        raise worker.Path1TerminalError("transport", "fixture_timeout")

    monkeypatch.setattr(worker, "_execute", fail)
    with pytest.raises(worker.Path1TerminalError):
        worker.run_worker(fixture["request_path"], runtime_verifier=OfflineVerifier(tmp_path))
    receipt = json.loads((fixture["output"] / "worker_terminal.json").read_text())
    assert receipt["status"] == "incomplete" and receipt["failure_category"] == "transport"
    assert receipt["retry_eligibility"] == "not_established"
    assert fixture["store"].read_registry().claims[0].status == "failed"


@pytest.mark.parametrize("problem", ["dict", "deadline", "job", "probe", "identity"])
def test_invalid_runtime_never_consumes_claim(tmp_path, monkeypatch, problem):
    fixture = make_worker_fixture(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    class BrokenVerifier(OfflineVerifier):
        def verify_in_job(self, request):
            result = super().verify_in_job(request)
            if problem == "dict":
                return {"verified": True}
            if problem == "deadline":
                return replace(result, deadline_utc=datetime.now(UTC) - timedelta(seconds=10))
            if problem == "job":
                return replace(result, slurm_job_id="99999")
            if problem == "probe":
                return replace(result, probes=result.probes[:-1])
            return replace(result, config_sha256="f" * 64)

    with pytest.raises(worker.RuntimeAdmissionRequired):
        worker.run_worker(fixture["request_path"], runtime_verifier=BrokenVerifier(tmp_path))
    assert fixture["store"].read_registry().claims[0].status == "reserved"
    assert not fixture["output"].exists()


def test_technical_design_and_global_pair_indices_are_not_job_local(tmp_path, monkeypatch):
    fixture = make_worker_fixture(tmp_path)
    validated = worker.validate_worker_request(fixture["request_path"])
    d = worker.design_from_contract(
        validated.contract, root_seed=validated.seed_inventory["root_seed"]
    )
    assert (d.formation_episodes, d.formation_episode_hands, d.evaluation_episode_hands) == (
        2,
        6,
        4,
    )
    seen = []
    monkeypatch.setattr(worker.path1, "run_formation", lambda **_: ("fixture_snapshot", {}))
    monkeypatch.setattr(
        worker.path1,
        "run_evaluation_pair",
        lambda **kw: seen.append((kw["episode_index"], kw["snapshot"])) or {},
    )
    # The validated whole inventory remains complete; a selected job subset must
    # use its recorded index rather than enumerate its first item as zero.
    validated.contract["job"]["pair_ids"] = ["pair_0002"]
    worker._execute(
        validated, worker.StrictPath1ClientFactory(validated.contract["source_contract"])
    )
    assert seen == [(1, "fixture_snapshot")]


def test_pilot_inventory_checks_whole_cohort_and_seat_balanced_global_pairs(tmp_path):
    fixture = make_worker_fixture(tmp_path, phase="E2")
    result = worker.validate_worker_request(fixture["request_path"])
    assert len(result.seed_inventory["artifacts"]) == 16
    all_streams = []
    for artifact in result.seed_inventory["artifacts"]:
        seats = [row["hero_seat"] for row in artifact["pairs"]]
        assert [seats.count(seat) for seat in range(8)] == [2] * 8
        all_streams.extend(artifact["streams"].values())
    assert len(all_streams) == len(set(all_streams))
    incomplete = copy.deepcopy(result.seed_inventory)
    incomplete["artifacts"].pop()
    with pytest.raises(ValueError, match="complete formation cohort"):
        worker.validate_seed_inventory(incomplete, result.contract)
    changed = copy.deepcopy(result.seed_inventory)
    last = changed["artifacts"][-1]
    last["streams"][next(iter(last["streams"]))] += 1
    with pytest.raises(ValueError, match="changed actual deal/client seeds"):
        worker.validate_seed_inventory(changed, result.contract)


def test_typed_runtime_probe_byte_hash_is_checked(tmp_path, monkeypatch):
    fixture = make_worker_fixture(tmp_path)
    request = worker.validate_worker_request(fixture["request_path"])
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    admission = OfflineVerifier(tmp_path).verify_in_job(request)
    probe = admission.probes[0]
    from pathlib import Path

    Path(probe.path).write_bytes(b"changed fixture bytes")
    with pytest.raises(worker.RuntimeAdmissionRequired, match="probe bytes changed"):
        worker._check_runtime(admission, request, datetime.now(UTC))
    assert fixture["store"].read_registry().claims[0].status == "reserved"


@pytest.mark.parametrize("non_integer", [1.0, True])
def test_seed_values_require_exact_integer_type_even_when_python_equality_matches(
    tmp_path,
    monkeypatch,
    non_integer,
):
    fixture = make_worker_fixture(tmp_path)
    inventory = fixture["inventory"]
    inventory["artifacts"][0]["streams"] = {"fixture_label": non_integer}
    monkeypatch.setattr(worker, "_planned_streams", lambda *_: {"fixture_label": 1})
    fixture["contract"]["cohort"]["seed_inventory_sha256"] = content_digest(inventory)
    assert non_integer == 1
    with pytest.raises(ValueError, match="changed actual deal/client seeds"):
        worker.validate_seed_inventory(inventory, fixture["contract"])


def test_largest_episode_grid_uses_unique_actual_63bit_streams():
    from agentmemeval.path1_v1.experiments.path1 import Path1Design

    values = set()
    for artifact in range(40):
        design = Path1Design(
            "formal_fixture",
            f"artifact_{artifact:04d}",
            "E3",
            20261113,
            seed_namespace="large_cohort_fixture",
        )
        streams = worker._planned_streams(design, 256)
        assert not values.intersection(streams.values())
        values.update(streams.values())
    assert len(values) == 40 * (300 + 6 * 8 + 256 * (50 + 2 * 8))
    assert all(type(value) is int and 0 <= value < 2**63 for value in values)
    assert max(values) >= 2**32
