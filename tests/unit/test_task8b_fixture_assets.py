"""Keep offline TASK8B contracts bound to the original approved frozen bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agentmemeval.experiments.task8b_phase_f_materializer import APPROVED_PREUNLOCK_SHA256

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "task8b_phase_f"
MANIFEST_NAME = "phase_f_pre_unlock_manifest_b382978_six_seed.json"
TEMPLATE_NAMES = (
    "analysis_contract.md",
    "data_lineage.schema.json",
    "data_lineage_template.csv",
    "exclusion_ledger_template.csv",
)


def _manifest() -> dict[str, object]:
    payload = (FIXTURES / MANIFEST_NAME).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == APPROVED_PREUNLOCK_SHA256
    return json.loads(payload)


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
def test_fixed_template_preserves_frozen_size_and_sha256(name: str) -> None:
    manifest = _manifest()
    record = next(row for row in manifest["phase_f_files"] if row["relative_path"] == name)
    payload = (FIXTURES / "phase_f_generated" / name).read_bytes()
    assert len(payload) == record["size"]
    assert hashlib.sha256(payload).hexdigest() == record["sha256"]


def test_fixed_manifest_binds_actual_repository_contract_bytes() -> None:
    manifest = _manifest()
    relative = "configs/formal/task8b_six_seed_analysis_contract.json"
    payload = (REPO_ROOT / relative).read_bytes()
    record = next(
        row for row in manifest["analysis_code_files"] if row["relative_path"] == relative
    )
    assert len(payload) == record["size"]
    assert hashlib.sha256(payload).hexdigest() == record["sha256"]
    assert record["sha256"] == manifest["analysis_contract_sha256"]


def test_fixture_provenance_inventory_is_exact_and_byte_preserving() -> None:
    provenance = json.loads((FIXTURES / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["approved_preunlock_sha256"] == APPROVED_PREUNLOCK_SHA256
    expected = {MANIFEST_NAME, *(f"phase_f_generated/{name}" for name in TEMPLATE_NAMES)}
    assert {row["fixture_path"] for row in provenance["files"]} == expected
    assert len(provenance["files"]) == len(expected)
    for row in provenance["files"]:
        payload = (FIXTURES / row["fixture_path"]).read_bytes()
        assert row["source_project_relative_path"] == (
            "02_项目文档/task-records/TASK8B/" + row["fixture_path"]
        )
        assert row["transformation"] == "none; byte-for-byte copy"
        assert row["size"] == len(payload)
        assert row["sha256"] == hashlib.sha256(payload).hexdigest()
