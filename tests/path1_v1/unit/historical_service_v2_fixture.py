"""Offline historical v2 producer; never a production preparation bypass.

Frozen prepare function from commit 765d1509ade3631d8ff8a3839d9ca81261caa38f,
src/agentmemeval/path1_v1/operations/e0_service_cli.py.
Source-file SHA256: fb5b119a7413ff313952c0bc9dd5676a1a94dcddfbfe49e310d67365f71d3279
Original function AST SHA256: 930beeb4c659fd205470863021a1aca6c8ba58470141020673acf725db231480
Only its function name and imports differ. Current product helpers still validate
config/identity/model manifest/script/profile. New live prepare remains v3-only.
"""

from __future__ import annotations

from agentmemeval.path1_v1.operations.e0_service_cli import (
    SOURCE_CONTRACT_SHA256,
    UTC,
    Path,
    SimpleNamespace,
    SubmissionRequest,
    SubmitSpec,
    _absolute,
    _config,
    _config_path,
    _require,
    _strict_json,
    _write_raw,
    build_runtime_identity,
    canonical_intent_id,
    content_digest,
    datetime,
    file_sha256,
    hashlib,
    identifier,
    render_probe_script,
    subprocess,
    validate_campaign_contract,
    validate_service_submission,
    verify_environment,
    verify_python_interpreter,
    write_exclusive_json,
)


def prepare_historical_service_v2(
    repository: Path, run_root: Path, campaign_id: str, config_relative: str,
) -> dict:
    repository = _absolute(repository)
    _absolute(run_root, exists=False)
    if run_root.exists():
        raise FileExistsError("service preparation root already exists")
    identifier(campaign_id)
    config_path = _config_path(repository, config_relative)
    raw = config_path.read_bytes()
    config = _config(raw)
    _require(config["resource_profile_version"] == 2 and config["language_model_only"] is True,
             "new service preparation requires text-only profile version 2")
    lock_path = _config_path(repository, config["dependency_lock"])
    lock_raw = lock_path.read_bytes()
    verify_environment(lock_path)
    python = Path(config["python_executable"])
    python_identity = verify_python_interpreter(
        python, config["python_sha256"], expected_prefix=config["python_prefix"]
    )
    service_identity = verify_python_interpreter(
        Path(config["service_python"]),
        config["service_python_sha256"],
        expected_prefix=config["service_python_prefix"],
    )
    model = _absolute(Path(config["model_root"]))
    manifest = _absolute(Path(config["model_manifest_path"]))
    manifest_raw = manifest.read_bytes()
    _require(
        hashlib.sha256(manifest_raw).hexdigest() == config["model_manifest_sha256"],
        "model manifest differs from frozen service config",
    )
    tracked = (
        subprocess.run(
            ["git", "-C", str(repository), "ls-files", "-z"], capture_output=True, check=True
        )
        .stdout.decode()
        .split("\0")
    )
    dependencies = [
        p
        for p in tracked
        if p.startswith(("src/", "prompts/", "tools/")) or p in {"pyproject.toml", ".gitattributes"}
    ]
    operations = [p for p in dependencies if p.startswith("src/agentmemeval/path1_v1/operations/")]
    identity = build_runtime_identity(
        repository,
        core_paths=sorted(set(dependencies) - set(operations)),
        operations_paths=sorted(operations),
        config_path=config_relative,
        dependency_lock_path=config["dependency_lock"],
        model_manifest_path=manifest,
        model_root=model,
    )
    _require(
        not identity["dirty"]
        and identity["content_matches_head"]
        and not identity["uncovered_execution_paths"]
        and not identity["untracked_execution_paths"]
        and not identity["untracked_declared_paths"],
        "service source is not clean/complete",
    )
    originals = {
        "site_config.json": (config_path, raw),
        "dependency_lock.json": (lock_path, lock_raw),
        "model_manifest.json": (manifest, manifest_raw),
    }
    expected = {
        "site_config.json": identity["config"]["sha256"],
        "dependency_lock.json": identity["dependency_lock"]["sha256"],
        "model_manifest.json": identity["model"]["manifest_sha256"],
    }
    for name, (path, data) in originals.items():
        _require(
            path.read_bytes() == data and hashlib.sha256(data).hexdigest() == expected[name],
            "service original input changed during preparation",
        )
    source = _strict_json(
        (repository / ("src/agentmemeval/path1_v1/assets/path1_source_contract.json")).read_bytes()
    )
    contract = {
        "schema_version": 1,
        "kind": "path1_campaign_contract",
        "campaign_id": campaign_id,
        "phase": "E0",
        "source_contract": source,
        "protocol_sha256": SOURCE_CONTRACT_SHA256,
        "runtime_identity": identity,
        "seed_namespace": campaign_id + "_technical",
        "cohort": {
            "formation_count": 0,
            "episodes_per_artifact_per_arm": 0,
            "seed_inventory_sha256": None,
            "freeze_receipt_sha256": None,
        },
        "job": {
            "kind": "service_probe",
            "artifact_id": None,
            "pair_ids": [],
            "arms": [],
            "domain": "source",
            "attempt": 1,
            "output_relpath": campaign_id + "/E0/service/attempt_01",
            "formation_receipt": None,
            "retry_evidence": None,
        },
        "not_for_paper": True,
        "execution_admission": "not_granted_by_contract_validation",
    }
    validate_campaign_contract(contract)
    run_root.mkdir(exist_ok=False)
    for name in ("output", "preflight", "inputs"):
        (run_root / name).mkdir()
    inputs = []
    for name, (path, data) in originals.items():
        _write_raw(run_root / "inputs" / name, data)
        inputs.append(
            {
                "source_path": str(path),
                "path": "inputs/" + name,
                "size": len(data),
                "sha256": expected[name],
            }
        )
    contract_path = run_root / "contract.json"
    write_exclusive_json(contract_path, contract)
    launch = {
        key: config[key]
        for key in (
            "python_executable",
            "python_sha256",
            "service_python",
            "service_python_sha256",
            "service_port",
            "command_seconds",
            "cleanup_seconds",
            "startup_seconds",
        )
    }
    launch.update(python_identity=python_identity, service_python_identity=service_identity)
    launch["language_model_only"] = config["language_model_only"]
    probe = {
        "schema_version": 2,
        "kind": "path1_e0_probe_request",
        "intent_id": canonical_intent_id(contract),
        "contract_path": str(contract_path),
        "contract_sha256": file_sha256(contract_path),
        "repository_root": str(repository),
        "model_root": str(model),
        "model_manifest_path": str(manifest),
        "output_root": str(run_root / "output"),
        "launch": launch,
    }
    probe["request_sha256"] = content_digest(probe)
    request_path = run_root / "probe_request.json"
    write_exclusive_json(request_path, probe)
    script = run_root / "probe.sh"
    _write_raw(
        script,
        render_probe_script(
            python_executable=python, repository_root=repository, probe_request_path=request_path
        ),
    )
    spec = SubmitSpec(
        config["cluster"],
        config["account"],
        config["user"],
        config["partition"],
        str(run_root / "output" / contract["job"]["output_relpath"]),
        str(script),
        file_sha256(script),
        walltime_seconds=7200,
        cpus_per_task=config["cpus_per_task"],
        memory_mib=config["memory_mib"],
    )
    submission = SubmissionRequest(campaign_id, probe["intent_id"], spec)
    profile = validate_service_submission(
        submission, SimpleNamespace(request=probe, contract=contract), config
    )
    descriptor = {
        "schema_version": 1,
        "kind": "path1_prepared_service_submission",
        "created_at": datetime.now(UTC).isoformat(),
        "repository_root": str(repository),
        "config_relative": config_relative,
        "config_sha256": identity["config"]["sha256"],
        "campaign_id": campaign_id,
        "intent_id": probe["intent_id"],
        "spec": spec.payload(),
        "submission_request_sha256": submission.request_sha256,
        "probe_request_sha256": file_sha256(request_path),
        "inputs": inputs,
        "service_profile": profile,
        "implementation_status": "implemented_only",
        "runtime_admission": False,
        "scientific_admission": False,
    }
    path = run_root / "submission.json"
    write_exclusive_json(path, descriptor)
    return {"descriptor_path": str(path), "descriptor_sha256": file_sha256(path), **descriptor}
