from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations.contracts import (
    MODEL,
    MODEL_REVISION,
    SOURCE_CONTRACT_SHA256,
    ContractError,
    build_runtime_identity,
    canonical_intent_id,
    canonical_json,
    content_digest,
    validate_campaign_contract,
)

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src/agentmemeval/path1_v1/assets/path1_source_contract.json"
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Contract Fixture",
                    "-c", "user.email=fixture@example.invalid", *args], check=True,
                   capture_output=True)


def _record(root: Path, relative: str) -> dict[str, object]:
    data = (root / relative).read_bytes()
    return {"path": relative, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _seal_identity(identity: dict) -> None:
    identity["identity_sha256"] = content_digest(
        {key: value for key, value in identity.items() if key != "identity_sha256"}
    )


@pytest.fixture
def evidence(tmp_path: Path) -> dict:
    repo = tmp_path / "repository"
    repo.mkdir()
    (repo / "src").mkdir()
    for name, data in {
        "src/core.py": "# core\n",
        "src/operations.py": "# ops\n",
        "pyproject.toml": "[project]\nname='fixture'\n",
        "resolved.json": '{"setting":1}\n',
        "requirements.lock": "dependency==1.0\n",
        ".gitignore": "src/ignored_dependency.py\n",
    }.items():
        (repo / name).write_text(data, encoding="utf-8", newline="\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "test fixture")
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"test weights; never loaded")
    (model / "tokenizer.json").write_bytes(b'{"test":true}')
    (model / "chat_template.txt").write_bytes(b"fixture chat template")
    manifest = {
        "schema_version": 1, "model": MODEL, "revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "chat_template_sha256": hashlib.sha256(b"fixture chat template").hexdigest(),
        "files": [_record(model, x) for x in sorted(p.name for p in model.iterdir())],
    }
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_bytes(canonical_json(manifest))
    kwargs = {
        "core_paths": ["pyproject.toml", "src/core.py"],
        "operations_paths": ["src/operations.py"],
        "config_path": "resolved.json", "dependency_lock_path": "requirements.lock",
        "model_manifest_path": manifest_path, "model_root": model,
    }
    identity = build_runtime_identity(repo, **kwargs)
    return {"repo": repo, "model": model, "manifest": manifest_path,
            "build_kwargs": kwargs, "identity": identity, "tmp": tmp_path}


def _contract(evidence: dict, phase: str = "E1", kind: str | None = None) -> dict:
    defaults = {"E0": (0, 0, "resource_probe"), "E1": (1, 2, "technical_closed_loop"),
                "E2": (16, 16, "formation"), "E3": (40, 16, "formation")}
    n, b, default_kind = defaults[phase]
    job_kind = kind or default_kind
    paired = job_kind in {"evaluation", "technical_closed_loop"}
    return {
        "schema_version": 1, "kind": "path1_campaign_contract", "campaign_id": "test_campaign",
        "phase": phase, "source_contract": json.loads(SOURCE.read_text(encoding="utf-8")),
        "protocol_sha256": SOURCE_CONTRACT_SHA256, "runtime_identity": copy.deepcopy(
            evidence["identity"]), "seed_namespace": "test_namespace",
        "cohort": {"formation_count": n, "episodes_per_artifact_per_arm": b,
                   "seed_inventory_sha256": None if phase == "E0" else "a" * 64,
                   "freeze_receipt_sha256": "b" * 64 if phase == "E3" else None},
        "job": {"kind": job_kind, "artifact_id": None if phase == "E0" else "artifact_0001",
                "pair_ids": ["pair_0001", "pair_0002"] if paired else [],
                "arms": ["retained", "removed"] if paired else [], "domain": "source",
                "attempt": 1, "output_relpath": "test_campaign/E1/job_1/attempt_01",
                "formation_receipt": None, "retry_evidence": None},
        "not_for_paper": phase != "E3",
        "execution_admission": "not_granted_by_contract_validation",
    }


def _roots(evidence: dict) -> dict:
    return {"repository_root": evidence["repo"], "model_manifest_path": evidence["manifest"],
            "model_root": evidence["model"]}


def _evaluation(evidence: dict) -> tuple[dict, Path]:
    contract = _contract(evidence, "E2", "evaluation")
    artifact_root = evidence["tmp"] / "artifact"
    artifact_root.mkdir()
    (artifact_root / "snapshot.json").write_bytes(b'{"fixture":"snapshot"}')
    identity = contract["runtime_identity"]
    receipt = {
        "schema_version": 1, "kind": "path1_immutable_formation_receipt",
        "campaign_id": contract["campaign_id"], "phase": "E2", "artifact_id": "artifact_0001",
        "formation_intent_id": "c" * 64, "seed_namespace": contract["seed_namespace"],
        "protocol_sha256": SOURCE_CONTRACT_SHA256,
        "core_manifest_sha256": identity["core_manifest_sha256"],
        "operations_manifest_sha256": identity["operations_manifest_sha256"],
        "model_manifest_sha256": identity["model"]["manifest_sha256"],
        "formation_config_sha256": "d" * 64,
        "snapshot": _record(artifact_root, "snapshot.json"),
        "completed_hands": 300, "completed_episodes": 6, "selected_after_hand": 300,
        "selection_policy": contract["source_contract"]["formation"]["primary_snapshot"],
        "status": "complete",
    }
    receipt["receipt_sha256"] = content_digest(receipt)
    contract["job"]["formation_receipt"] = receipt
    return contract, artifact_root


def test_source_matches_frozen_plan_and_is_not_an_executable_campaign() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert content_digest(source) == SOURCE_CONTRACT_SHA256
    assert source["formation"]["hands"] == 6 * 50
    assert source["controller"]["hero"]["experience_prompt_token_budget"] == 65536
    assert source["controller"]["terminal_policy"]["revision_max_contract_repairs"] == 1
    with pytest.raises(ContractError, match="missing fields"):
        validate_campaign_contract(source)


def test_stable_identity_order_independence_and_detached_validation(evidence: dict) -> None:
    contract = _contract(evidence)
    shuffled = dict(reversed(list(contract.items())))
    assert canonical_intent_id(contract) == canonical_intent_id(shuffled)
    result = validate_campaign_contract(contract)
    result["job"]["pair_ids"].append("unrelated")
    assert len(contract["job"]["pair_ids"]) == 2
    assert result["execution_admission"] == "not_granted_by_contract_validation"


def test_config_bytes_change_intent_even_when_json_meaning_is_same(evidence: dict) -> None:
    contract = _contract(evidence)
    before = canonical_intent_id(contract)
    (evidence["repo"] / "resolved.json").write_bytes(b'{ "setting": 1 }\n')
    identity = build_runtime_identity(evidence["repo"], **evidence["build_kwargs"])
    assert identity["config"]["sha256"] != contract["runtime_identity"]["config"]["sha256"]
    contract["runtime_identity"] = identity
    assert canonical_intent_id(contract) != before


@pytest.mark.parametrize("field,value", [("campaign_id", "other_campaign"),
                                         ("seed_namespace", "other_namespace")])
def test_logical_scope_is_part_of_intent(evidence: dict, field: str, value: str) -> None:
    contract = _contract(evidence)
    before = canonical_intent_id(contract)
    contract[field] = value
    assert before != canonical_intent_id(contract)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/drive", "a\\b", "a//b",
                                  "a/./b", "NUL.json", "a/b.", "a/b ", "a:x", "a/\n",
                                  ".git/config", ".env", "a/LPT1.txt"])
def test_unsafe_paths_rejected_before_access(evidence: dict, path: str) -> None:
    contract = _contract(evidence)
    contract["job"]["output_relpath"] = path
    with pytest.raises(ContractError):
        validate_campaign_contract(contract)


@pytest.mark.parametrize("section", [None, "job", "cohort", "runtime_identity"])
def test_unknown_fields_fail_closed(evidence: dict, section: str | None) -> None:
    contract = _contract(evidence)
    target = contract if section is None else contract[section]
    target["surprise"] = "must not be silently dropped"
    with pytest.raises(ContractError, match="unknown fields"):
        validate_campaign_contract(contract)


def test_source_semantics_or_unknown_field_cannot_drift(evidence: dict) -> None:
    contract = _contract(evidence)
    contract["source_contract"]["controller"]["hero"]["window_size"] = 2
    with pytest.raises(ContractError, match="frozen"):
        validate_campaign_contract(contract)
    contract = _contract(evidence)
    contract["source_contract"]["hidden_override"] = True
    with pytest.raises(ContractError, match="frozen"):
        validate_campaign_contract(contract)


@pytest.mark.parametrize("bad", [True, 1.5, float("nan")])
def test_bad_counter_types_not_coerced(evidence: dict, bad: object) -> None:
    contract = _contract(evidence)
    contract["job"]["attempt"] = bad
    with pytest.raises(ContractError):
        validate_campaign_contract(contract)


def test_missing_unknown_sha_and_false_runtime_claim_rejected(evidence: dict) -> None:
    contract = _contract(evidence)
    identity = contract["runtime_identity"]
    identity["core_sha"] = "unknown"
    with pytest.raises(ContractError, match="complete lowercase"):
        validate_campaign_contract(contract)
    contract = _contract(evidence)
    contract["runtime_identity"]["runtime_attestation"] = "verified_gpu"
    with pytest.raises(ContractError, match="runtime attestation"):
        validate_campaign_contract(contract)


def test_e0_explicitly_not_for_paper_and_accepts_missing_model_inventory(evidence: dict) -> None:
    kwargs = dict(evidence["build_kwargs"])
    kwargs.update(model_manifest_path=None, model_root=None)
    contract = _contract(evidence, "E0")
    contract["runtime_identity"] = build_runtime_identity(evidence["repo"], **kwargs)
    assert validate_campaign_contract(contract)["not_for_paper"] is True
    contract["not_for_paper"] = False
    with pytest.raises(ContractError, match="not_for_paper"):
        validate_campaign_contract(contract)


@pytest.mark.parametrize("phase", ["E2", "E3"])
def test_scientific_identity_requires_current_content(evidence: dict, phase: str) -> None:
    contract = _contract(evidence, phase)
    with pytest.raises(ContractError, match="fresh source/model"):
        validate_campaign_contract(contract)
    assert validate_campaign_contract(contract, **_roots(evidence)) == contract
    (evidence["repo"] / "src/core.py").write_text("# changed\n", encoding="utf-8")
    with pytest.raises(ContractError, match="changed since planning"):
        validate_campaign_contract(contract, **_roots(evidence))
    contract["runtime_identity"] = build_runtime_identity(
        evidence["repo"], **evidence["build_kwargs"])
    with pytest.raises(ContractError, match="dirty"):
        validate_campaign_contract(contract, **_roots(evidence))


def test_ignored_untracked_execution_dependency_is_detected(evidence: dict) -> None:
    (evidence["repo"] / "src/ignored_dependency.py").write_text("# ignored\n", encoding="utf-8")
    identity = build_runtime_identity(evidence["repo"], **evidence["build_kwargs"])
    assert identity["dirty"] is False
    assert identity["untracked_execution_paths"] == ["src/ignored_dependency.py"]
    contract = _contract(evidence, "E2")
    contract["runtime_identity"] = identity
    with pytest.raises(ContractError, match="tracked complete"):
        validate_campaign_contract(contract, **_roots(evidence))


def test_declared_inventory_cannot_omit_tracked_execution_dependency(evidence: dict) -> None:
    kwargs = dict(evidence["build_kwargs"], core_paths=["src/core.py"])
    identity = build_runtime_identity(evidence["repo"], **kwargs)
    assert identity["uncovered_execution_paths"] == ["pyproject.toml"]
    contract = _contract(evidence, "E2")
    contract["runtime_identity"] = identity
    with pytest.raises(ContractError, match="tracked complete"):
        validate_campaign_contract(contract, **_roots(evidence))


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_hidden_dirty_content_is_compared_to_head_blob(evidence: dict, flag: str) -> None:
    _git(evidence["repo"], "update-index", flag, "src/core.py")
    (evidence["repo"] / "src/core.py").write_bytes(b"# hidden change\n")
    identity = build_runtime_identity(evidence["repo"], **evidence["build_kwargs"])
    assert identity["dirty"] is False
    assert identity["content_matches_head"] is False
    assert identity["head_content_mismatches"] == ["src/core.py"]
    contract = _contract(evidence, "E2")
    contract["runtime_identity"] = identity
    with pytest.raises(ContractError, match="non-HEAD content"):
        validate_campaign_contract(contract, **_roots(evidence))


def test_git_attribute_normalization_preserves_head_identity(evidence: dict) -> None:
    (evidence["repo"] / ".gitattributes").write_bytes(b"*.py text eol=lf\n")
    _git(evidence["repo"], "add", ".gitattributes")
    _git(evidence["repo"], "commit", "-qm", "pin text attributes")
    (evidence["repo"] / "src/core.py").write_bytes(b"# core\r\n")
    kwargs = dict(evidence["build_kwargs"])
    kwargs["core_paths"] = [".gitattributes", "pyproject.toml", "src/core.py"]
    identity = build_runtime_identity(evidence["repo"], **kwargs)
    assert identity["head_content_mismatches"] == []
    assert identity["content_matches_head"] is True
    assert identity["uncovered_execution_paths"] == []


def test_tampered_manifest_and_resealed_content_identity_rejected(evidence: dict) -> None:
    contract = _contract(evidence, "E2")
    contract["runtime_identity"]["core_files"][0]["sha256"] = "f" * 64
    with pytest.raises(ContractError, match="content manifest mismatch"):
        validate_campaign_contract(contract, **_roots(evidence))
    identity = contract["runtime_identity"]
    identity["core_manifest_sha256"] = content_digest(identity["core_files"])
    _seal_identity(identity)
    with pytest.raises(ContractError, match="changed since planning"):
        validate_campaign_contract(contract, **_roots(evidence))


@pytest.mark.parametrize("change", ["extra", "missing", "changed"])
def test_model_exact_inventory_and_bytes_verified(evidence: dict, change: str) -> None:
    if change == "extra":
        (evidence["model"] / "extra.bin").write_bytes(b"extra")
    elif change == "missing":
        # Rename only within the exclusive disposable fixture; no deletion.
        (evidence["model"] / "weights.bin").rename(evidence["tmp"] / "preserved_weights.bin")
    else:
        (evidence["model"] / "weights.bin").write_bytes(b"changed")
    with pytest.raises(ContractError, match="model file"):
        build_runtime_identity(evidence["repo"], **evidence["build_kwargs"])


def test_duplicate_model_json_keys_rejected(evidence: dict) -> None:
    evidence["manifest"].write_bytes(b'{"model":"first","model":"second"}')
    with pytest.raises(ContractError, match="duplicate JSON key"):
        build_runtime_identity(evidence["repo"], **evidence["build_kwargs"])


def test_evaluation_requires_bound_fixed_terminal_snapshot_and_both_arms(evidence: dict) -> None:
    contract, artifact_root = _evaluation(evidence)
    roots = dict(_roots(evidence), artifact_root=artifact_root)
    assert validate_campaign_contract(contract, **roots) == contract
    split = copy.deepcopy(contract)
    split["job"]["arms"] = ["retained"]
    with pytest.raises(ContractError, match="whole retained/removed pairs"):
        validate_campaign_contract(split, **roots)
    tampered = copy.deepcopy(contract)
    tampered["job"]["formation_receipt"]["selected_after_hand"] = 90
    with pytest.raises(ContractError, match="selected_after_hand"):
        validate_campaign_contract(tampered, **roots)
    (artifact_root / "snapshot.json").write_bytes(b"tampered")
    with pytest.raises(ContractError, match="snapshot content mismatch"):
        validate_campaign_contract(contract, **roots)


def test_receipt_cannot_cross_campaign_or_be_accepted_as_partial(evidence: dict) -> None:
    contract, artifact_root = _evaluation(evidence)
    roots = dict(_roots(evidence), artifact_root=artifact_root)
    receipt = contract["job"]["formation_receipt"]
    receipt["campaign_id"] = "another_campaign"
    with pytest.raises(ContractError, match="campaign_id mismatch"):
        validate_campaign_contract(contract, **roots)
    receipt["campaign_id"] = contract["campaign_id"]
    receipt["status"] = "partial"
    with pytest.raises(ContractError, match="completed fixed-terminal"):
        validate_campaign_contract(contract, **roots)


def test_formal_freeze_and_balanced_design_required(evidence: dict) -> None:
    contract = _contract(evidence, "E3")
    contract["cohort"]["freeze_receipt_sha256"] = None
    with pytest.raises(ContractError, match="freeze_receipt"):
        validate_campaign_contract(contract, **_roots(evidence))
    contract["cohort"]["freeze_receipt_sha256"] = "a" * 64
    contract["cohort"]["formation_count"] = 41
    with pytest.raises(ContractError, match="sample size/seat balance"):
        validate_campaign_contract(contract, **_roots(evidence))


def test_retries_need_separate_eligibility_reference_and_do_not_change_seed(evidence: dict) -> None:
    contract = _contract(evidence)
    original = canonical_intent_id(contract)
    contract["job"]["attempt"] = 2
    with pytest.raises(ContractError, match="retry_evidence"):
        validate_campaign_contract(contract)
    contract["job"]["retry_evidence"] = {
        "eligibility_receipt_sha256": "a" * 64, "prior_intent_id": original,
        "prior_terminal_receipt_sha256": "b" * 64,
        "scope": "complete_unit", "seed_changed": False,
    }
    assert canonical_intent_id(contract) != original
    contract["job"]["retry_evidence"]["seed_changed"] = True
    with pytest.raises(ContractError, match="retain seeds"):
        validate_campaign_contract(contract)


def test_legacy_manifest_is_not_implicitly_admitted() -> None:
    legacy = {"run_id": "legacy", "git_sha": "a" * 40, "run_mode": "smoke"}
    before = copy.deepcopy(legacy)
    with pytest.raises(ContractError, match="campaign_contract"):
        validate_campaign_contract(legacy)
    assert legacy == before
