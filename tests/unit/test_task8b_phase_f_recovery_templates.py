from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
PHASE_F_DIR = WORKSPACE_ROOT / "docs" / "task-records" / "TASK8B" / "phase_f_generated"
CONTRACT_PATH = REPO_ROOT / "configs" / "formal" / "task8b_phase_f_contract.json"
SCHEMA_PATH = PHASE_F_DIR / "data_lineage.schema.json"
LINEAGE_TEMPLATE_PATH = PHASE_F_DIR / "data_lineage_template.csv"
LEDGER_TEMPLATE_PATH = PHASE_F_DIR / "exclusion_ledger_template.csv"
ANALYSIS_CONTRACT_PATH = PHASE_F_DIR / "analysis_contract.md"

RECOVERY_DISPOSITION = "VERIFIER_FALSE_POSITIVE_SAME_ATTEMPT_RECOVERY"
SCIENTIFIC_SHA = "a1d1eb97efb41d52585057ab7c9594dcd19227ae"
RECOVERY_EVIDENCE_FIELDS = {
    "protocol_amendment_sha256",
    "verifier_code_sha",
    "pre_recovery_archive_sha256",
    "pre_recovery_file_manifest_sha256",
    "original_terminal_state_sha256",
    "original_expected_config_sha256",
    "corrected_config_sha256",
    "canonicalization_equivalence_audit_sha256",
    "recovery_certificate_sha256",
    "task1_adoption_attestation_sha256",
}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 1
    assert len(rows[0]) == len(set(rows[0]))
    return rows[0]


def test_phase_f_contract_preserves_frozen_statistics_and_recovery_scope() -> None:
    contract = _read_json(CONTRACT_PATH)
    assert contract["n_planned"] == 12
    assert contract["primary_endpoint"] == "final_test_bb_per_100"
    assert contract["primary_checkpoint"] == 300
    assert contract["primary_mode"] == "Frozen"
    assert contract["statistical_unit"] == "seed"
    assert contract["bootstrap"] == {
        "level": 0.95,
        "method": "seed_cluster_percentile",
        "prng": "PCG64",
        "replicates": 10000,
        "seed": 2026090199,
    }
    assert contract["holm_family"] == ["Expr_vs_Fact", "Async_vs_Fact"]
    assert contract["attempt_selection"] == (
        "first_numerically_ordered_complete_valid_eligible_attempt_multiple_valid_fail_closed"
    )

    recovery = contract["verifier_false_positive_recovery"]
    assert recovery["recovery_disposition"] == RECOVERY_DISPOSITION
    assert recovery["preserved_workers"] == [f"P{index:02d}" for index in range(1, 12)]
    assert recovery["preserved_seeds"] == list(range(2026090101, 2026090112))
    assert recovery["hands_per_worker"] == 1350
    assert recovery["preserved_raw_hands_total"] == 14850
    assert recovery["outer_attempt"] == "attempt_01"
    assert recovery["recovery_is_retry"] is False
    assert recovery["task1_rerun_allowed"] is False
    assert recovery["formal_effects_read"] is False
    assert recovery["effect_metrics_read"] is False
    assert recovery["scientific_execution_code_sha"] == SCIENTIFIC_SHA
    assert recovery["original_terminal_state_sha256_semantics"] == (
        "sha256_of_canonical_original_terminal_row_not_full_state_file"
    )
    assert recovery["raw_artifact_reused_by_source_execution_stage"] == {
        "post_adoption_task2_plus": False,
        "pre_recovery_task1": True,
    }
    assert set(recovery["required_evidence_fields"]) == RECOVERY_EVIDENCE_FIELDS


def test_lineage_schema_requires_auditable_recovery_fields() -> None:
    schema = _read_json(SCHEMA_PATH)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "experiment_code_sha" in schema["required"]
    source = schema["$defs"]["sourceRecord"]
    required = set(source["required"])
    assert RECOVERY_EVIDENCE_FIELDS <= required
    assert {
        "recovery_disposition",
        "scientific_execution_code_sha",
        "source_execution_stage",
        "raw_artifact_reused",
        "task1_rerun_performed",
    } <= required
    dispositions = source["properties"]["recovery_disposition"]["enum"]
    assert dispositions == ["NONE", RECOVERY_DISPOSITION]
    recovery_then = source["allOf"][0]["then"]["properties"]
    assert recovery_then["attempt"]["const"] == "attempt_01"
    assert recovery_then["experiment_code_sha"]["const"] == SCIENTIFIC_SHA
    assert recovery_then["scientific_execution_code_sha"]["const"] == SCIENTIFIC_SHA
    assert recovery_then["task1_rerun_performed"]["const"] is False
    terminal_state = source["properties"]["original_terminal_state_sha256"]
    assert "terminal state row" in terminal_state["description"]
    assert "not the hash of the complete state file" in terminal_state["description"]

    stage_branches = {
        branch["if"]["properties"]["source_execution_stage"]["const"]: branch["then"][
            "properties"
        ]["raw_artifact_reused"]["const"]
        for branch in source["allOf"][1:]
    }
    assert stage_branches == {
        "pre_recovery_task1": True,
        "post_adoption_task2_plus": False,
    }
    for field in RECOVERY_EVIDENCE_FIELDS - {"verifier_code_sha"}:
        assert recovery_then[field] == {"$ref": "#/$defs/sha256"}


def test_csv_templates_are_header_only_parseable_and_recovery_complete() -> None:
    lineage = set(_csv_header(LINEAGE_TEMPLATE_PATH))
    ledger = set(_csv_header(LEDGER_TEMPLATE_PATH))
    shared_recovery_fields = RECOVERY_EVIDENCE_FIELDS | {
        "recovery_disposition",
        "protocol_amendment_id",
        "experiment_code_sha",
        "scientific_execution_code_sha",
        "source_execution_stage",
        "raw_artifact_reused",
        "task1_rerun_performed",
    }
    assert shared_recovery_fields <= lineage
    assert shared_recovery_fields <= ledger
    assert {
        "run_id",
        "seed",
        "condition",
        "checkpoint_hand",
        "heldout_table_id",
        "attempt",
        "experiment_code_sha",
        "config_sha256",
        "prompt_sha256",
        "model_fingerprint",
        "source_file_relative_path",
        "row_selector",
        "exclusion_status",
    } <= lineage
    assert {
        "reason_code",
        "eligible_retry",
        "authoritative_attempt",
        "task1_raw_hands_preserved",
        "effect_metrics_read",
    } <= ledger


def test_analysis_contract_freezes_recovery_without_statistical_drift() -> None:
    text = ANALYSIS_CONTRACT_PATH.read_text(encoding="utf-8")
    for token in (
        "14,850 hands",
        RECOVERY_DISPOSITION,
        SCIENTIFIC_SHA,
        "formal_effects_read=false",
        "effect_metrics_read=false",
        "task1_rerun_performed=false",
        "pre_recovery_task1=true",
        "post_adoption_task2_plus=false",
        "terminal state row",
        "不是整个 state 文件",
        "10,000",
        "PCG64",
        "Holm",
        "n=12",
    ):
        assert token in text
