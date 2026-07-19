"""Build a fail-closed Task4 study report from verified campaign evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ANALYSIS_SCHEMA = "task4_campaign_analysis_bundle_v3"
STUDY_SPEC_SCHEMA = "task4_study_report_spec_v1"
STUDY_BUNDLE_SCHEMA = "task4_study_report_bundle_v1"


def build_task4_study_report(
    study_spec_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build a non-overwriting Chinese report and bind every cited input by hash."""

    spec_path = Path(study_spec_path).resolve()
    spec = _read_json(spec_path)
    if spec.get("schema_version") != STUDY_SPEC_SCHEMA:
        raise ValueError(f"study spec schema must be {STUDY_SPEC_SCHEMA}")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    evidence: list[dict[str, Any]] = []
    blockers: list[str] = []
    analyses: dict[str, dict[str, Any]] = {}
    table_rows: list[dict[str, str]] = []
    for campaign_key in ("campaign_p", "campaign_e"):
        raw = spec.get(f"{campaign_key}_analysis_manifest")
        if not raw:
            blockers.append(f"{campaign_key} analysis manifest is missing")
            continue
        manifest_path = _resolve_input(spec_path, raw)
        analysis = _validate_analysis_manifest(manifest_path)
        analyses[campaign_key] = analysis
        evidence.extend(analysis["evidence"])
        for row in analysis["table_rows"]:
            table_rows.append({"campaign": campaign_key, **row})
        blockers.extend(
            _analysis_scope_blockers(campaign_key, analysis["table_rows"])
        )
        if analysis["paper_inference_eligible"] is not True:
            blockers.append(
                f"{campaign_key} analysis is not formal inference eligible"
            )

    run_map_audits: dict[str, dict[str, Any]] = {}
    for campaign_key in ("campaign_p", "campaign_e"):
        raw = spec.get(f"{campaign_key}_run_map")
        if not raw:
            blockers.append(f"{campaign_key} run map is missing")
            continue
        path = _resolve_input(spec_path, raw)
        expected = int(analyses.get(campaign_key, {}).get("expected_run_count", 0))
        audit = _audit_run_map(path, expected_eligible_count=expected)
        run_map_audits[campaign_key] = audit
        evidence.append(_evidence(campaign_key + "_run_map", path, audit["status"]))
        if audit["status"] != "verified_formal_candidates_cover_analysis":
            blockers.append(
                f"{campaign_key} run map does not cover the formal analysis matrix"
            )

    resources: dict[str, dict[str, Any]] = {}
    for campaign_key in ("campaign_p", "campaign_e"):
        raw = spec.get(f"{campaign_key}_resource_audit")
        if not raw:
            blockers.append(f"{campaign_key} resource audit is missing")
            continue
        path = _resolve_input(spec_path, raw)
        audit = _read_json(path)
        resources[campaign_key] = audit
        status = _resource_status(audit)
        evidence.append(_evidence(campaign_key + "_resource_audit", path, status))
        if status != "verified_zero_fallback":
            blockers.append(f"{campaign_key} resource audit is not zero-fallback")

    runtime_lock: dict[str, Any] = {}
    raw_runtime_lock = spec.get("formal_runtime_lock")
    if not raw_runtime_lock:
        blockers.append("formal runtime lock is missing")
    else:
        path = _resolve_input(spec_path, raw_runtime_lock)
        runtime_lock = _read_json(path)
        status = str(runtime_lock.get("status", ""))
        evidence.append(_evidence("formal_runtime_lock", path, status or "missing"))
        if status != "verified_from_real_service_run_manifest":
            blockers.append("formal runtime lock is not verified")

    seal_audits: dict[str, dict[str, Any]] = {}
    for campaign_key in ("campaign_p", "campaign_e"):
        raw = spec.get(f"{campaign_key}_seal_readiness")
        if not raw:
            blockers.append(f"{campaign_key} seal-readiness audit is missing")
            continue
        path = _resolve_input(spec_path, raw)
        audit = _read_json(path)
        seal_audits[campaign_key] = audit
        status = _seal_status(audit)
        evidence.append(_evidence(campaign_key + "_seal_readiness", path, status))
        if status != "verified_ready_to_seal":
            blockers.append(f"{campaign_key} seal-readiness audit is not verified")

    archive_pairs: dict[str, dict[str, Any]] = {}
    for campaign_key in ("campaign_p", "campaign_e"):
        build_raw = spec.get(f"{campaign_key}_archive_build_receipt")
        extraction_raw = spec.get(f"{campaign_key}_archive_extraction_receipt")
        if not build_raw:
            blockers.append(f"{campaign_key} archive build receipt is missing")
        if not extraction_raw:
            blockers.append(f"{campaign_key} archive extraction receipt is missing")
        if not build_raw or not extraction_raw:
            continue
        build_path = _resolve_input(spec_path, build_raw)
        extraction_path = _resolve_input(spec_path, extraction_raw)
        build_receipt = _read_json(build_path)
        extraction_receipt = _read_json(extraction_path)
        status = _archive_pair_status(build_receipt, extraction_receipt)
        archive_pairs[campaign_key] = {
            "status": status,
            "build": build_receipt,
            "extraction": extraction_receipt,
        }
        evidence.append(
            _evidence(
                campaign_key + "_archive_build_receipt",
                build_path,
                _archive_build_receipt_status(build_receipt),
            )
        )
        evidence.append(
            _evidence(
                campaign_key + "_archive_extraction_receipt",
                extraction_path,
                _archive_extraction_receipt_status(extraction_receipt),
            )
        )
        if status != "verified_archive_build_extract_pair":
            blockers.append(f"{campaign_key} archive receipt pair is not verified")

    protocol_items = spec.get("protocol_evidence", [])
    if not isinstance(protocol_items, list) or not protocol_items:
        blockers.append("protocol evidence is missing")
        protocol_items = []
    for index, item in enumerate(protocol_items, start=1):
        if not isinstance(item, dict) or not item.get("path"):
            blockers.append(f"protocol evidence {index} is malformed")
            continue
        path = _resolve_input(spec_path, item["path"])
        status = str(item.get("status", ""))
        evidence.append(
            _evidence(
                str(item.get("label") or f"protocol_evidence_{index}"),
                path,
                status or "missing",
            )
        )
        if status != "verified":
            blockers.append(f"protocol evidence {index} is not verified")

    gpu_status, gpu_identities = _gpu_homogeneity(resources)
    if gpu_status != "verified_uniform_gpu":
        blockers.append("campaign resource audits do not prove one uniform GPU identity")

    blockers = sorted(set(blockers))
    paper_inference_eligible = not blockers
    classification = (
        "paper_inference_ready"
        if paper_inference_eligible
        else "interim_or_blocked_no_paper_conclusion"
    )

    table_path = output / "study_effects.csv"
    evidence_path = output / "evidence_index.csv"
    status_path = output / "verification_status.csv"
    report_path = output / "task4_paper_report_zh.md"
    fields = tuple(table_rows[0].keys()) if table_rows else ("campaign",)
    _write_csv(table_path, fields, table_rows)
    _write_csv(
        evidence_path,
        ("label", "path", "sha256", "status"),
        evidence,
    )
    status_rows = _status_rows(
        analyses=analyses,
        run_maps=run_map_audits,
        resources=resources,
        seal_audits=seal_audits,
        archive_pairs=archive_pairs,
        gpu_status=gpu_status,
        blockers=blockers,
    )
    _write_csv(
        status_path,
        ("item", "classification", "evidence"),
        status_rows,
    )
    report_path.write_text(
        _render_report(
            spec=spec,
            spec_path=spec_path,
            table_rows=table_rows,
            run_maps=run_map_audits,
            resources=resources,
            runtime_lock=runtime_lock,
            seal_audits=seal_audits,
            archive_pairs=archive_pairs,
            gpu_status=gpu_status,
            gpu_identities=gpu_identities,
            blockers=blockers,
            classification=classification,
        ),
        encoding="utf-8",
    )

    outputs = [table_path, evidence_path, status_path, report_path]
    manifest = {
        "schema_version": STUDY_BUNDLE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_spec": str(spec_path),
        "study_spec_sha256": _sha256(spec_path),
        "classification": classification,
        "paper_inference_eligible": paper_inference_eligible,
        "paper_conclusion_prohibited": not paper_inference_eligible,
        "blockers": blockers,
        "evidence_count": len(evidence),
        "effect_row_count": len(table_rows),
        "outputs": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in outputs
        },
    }
    manifest_path = output / "study_report_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"output_dir": str(output), "manifest": str(manifest_path), **manifest}


def _validate_analysis_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("schema_version") != ANALYSIS_SCHEMA:
        raise ValueError(f"analysis manifest schema must be {ANALYSIS_SCHEMA}: {path}")
    evidence = [_evidence(path.stem, path, "verified_manifest_schema")]
    source = Path(str(manifest.get("source_aggregate", ""))).resolve()
    expected_source_hash = str(manifest.get("source_aggregate_sha256", ""))
    if not source.is_file() or _sha256(source) != expected_source_hash:
        raise ValueError(f"analysis source aggregate hash mismatch: {source}")
    evidence.append(_evidence("source_aggregate", source, "verified"))
    aggregate = _read_json(source)
    if manifest.get("campaign_status") != aggregate.get("status"):
        raise ValueError(f"analysis manifest campaign status mismatch: {path}")
    if (
        manifest.get("paper_inference_eligible") is True
        and aggregate.get("status") != "ready"
    ):
        raise ValueError(f"non-ready aggregate marked paper eligible: {path}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"analysis outputs missing: {path}")
    for name, raw in outputs.items():
        if not isinstance(raw, dict):
            raise ValueError(f"analysis output entry malformed: {name}")
        output_path = Path(str(raw.get("path", ""))).resolve()
        if not output_path.is_file() or _sha256(output_path) != raw.get("sha256"):
            raise ValueError(f"analysis output hash mismatch: {output_path}")
        evidence.append(_evidence(f"analysis_output:{name}", output_path, "verified"))
    table_entry = outputs.get("main_table.csv")
    if not isinstance(table_entry, dict):
        raise ValueError(f"analysis main_table.csv missing: {path}")
    table_path = Path(str(table_entry["path"])).resolve()
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        table_rows = list(csv.DictReader(handle))
    return {
        "paper_inference_eligible": manifest.get("paper_inference_eligible") is True,
        "campaign_status": str(manifest.get("campaign_status", "")),
        "expected_run_count": int(aggregate.get("expected_run_count", 0)),
        "table_rows": table_rows,
        "evidence": evidence,
    }


def _audit_run_map(
    path: Path,
    *,
    expected_eligible_count: int,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [
        row
        for row in rows
        if str(row.get("formal_main_table_eligible", "")).lower() == "true"
    ]
    status = (
        "verified_formal_candidates_cover_analysis"
        if expected_eligible_count > 0 and len(eligible) == expected_eligible_count
        else "blocked_formal_candidate_count_mismatch"
    )
    return {
        "status": status,
        "attempt_count": len(rows),
        "eligible_attempt_count": len(eligible),
        "excluded_attempt_count": len(rows) - len(eligible),
    }


def _analysis_scope_blockers(
    campaign_key: str,
    rows: list[dict[str, str]],
) -> list[str]:
    endpoints = {
        "final_test_bb_per_100",
        "final_test_chip_per_hand",
        "train_bb_per_100",
        "train_chip_per_hand",
        "generalization_gap_bb_per_100",
    }
    contrasts = (
        {
            "expr_vs_fact",
            "fact_expr_sync_vs_fact",
            "fact_expr_async_vs_fact",
        }
        if campaign_key == "campaign_p"
        else {"fact_target", "expr_target", "sync_target", "async_target"}
    )
    observed = {
        (str(row.get("contrast", "")), str(row.get("endpoint", "")))
        for row in rows
    }
    missing = sorted(
        (contrast, endpoint)
        for contrast in contrasts
        for endpoint in endpoints
        if (contrast, endpoint) not in observed
    )
    return [
        f"{campaign_key} analysis is missing required cell {contrast}/{endpoint}"
        for contrast, endpoint in missing
    ]


def _resource_status(audit: dict[str, Any]) -> str:
    if audit.get("schema_version") != "task4_campaign_resource_audit_v1":
        return "blocked_wrong_schema"
    if int(audit.get("completed_leaf_count", 0)) < 1:
        return "blocked_no_completed_leaves"
    if int(audit.get("action_fallback_count", -1)) != 0:
        return "blocked_action_fallback"
    if int(audit.get("experience_revision_fallback_count", -1)) != 0:
        return "blocked_revision_fallback"
    return "verified_zero_fallback"


def _seal_status(audit: dict[str, Any]) -> str:
    if audit.get("schema_version") != "task4_campaign_seal_readiness_v1":
        return "blocked_wrong_schema"
    if audit.get("status") != "ready_to_seal":
        return "blocked_not_ready_to_seal"
    if audit.get("blockers") != []:
        return "blocked_nonempty_blockers"
    for field in ("campaign_manifest_sha256", "state_tsv_sha256"):
        value = str(audit.get(field, ""))
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            return f"blocked_invalid_{field}"
    if int(audit.get("expected_matrix_count", 0)) < 1:
        return "blocked_empty_matrix"
    if audit.get("complete_latest_attempt_count") != audit.get(
        "expected_matrix_count"
    ):
        return "blocked_incomplete_latest_matrix"
    quiet = audit.get("observed_quiet_seconds")
    minimum = audit.get("minimum_quiet_seconds")
    if not isinstance(quiet, (int, float)) or not isinstance(minimum, int):
        return "blocked_invalid_quiet_period"
    if quiet < minimum:
        return "blocked_insufficient_quiet_period"
    return "verified_ready_to_seal"


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _verified_nested(
    payload: Any,
    *,
    schema_version: str,
) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == schema_version
        and payload.get("verified") is True
        and payload.get("status") == "verified"
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _archive_build_receipt_status(receipt: dict[str, Any]) -> str:
    if receipt.get("schema_version") != "task4_snapshot_archive_receipt_v1":
        return "blocked_wrong_build_schema"
    if receipt.get("status") != "verified":
        return "blocked_build_not_verified"
    if not _valid_sha256(receipt.get("archive_sha256")):
        return "blocked_invalid_archive_sha256"
    if not _valid_sha256(receipt.get("manifest_sha256")):
        return "blocked_invalid_manifest_sha256"
    if not _positive_int(receipt.get("archive_size_bytes")):
        return "blocked_invalid_archive_size"
    file_count = receipt.get("file_count")
    if not _positive_int(file_count):
        return "blocked_invalid_file_count"
    total_size = receipt.get("total_uncompressed_size_bytes")
    if (
        not isinstance(total_size, int)
        or isinstance(total_size, bool)
        or total_size < 0
    ):
        return "blocked_invalid_uncompressed_size"
    if not all(
        isinstance(receipt.get(field), str) and bool(receipt.get(field))
        for field in ("root", "archive", "manifest", "checksum")
    ):
        return "blocked_missing_build_paths"

    source = receipt.get("source_verification")
    if not _verified_nested(
        source,
        schema_version="task4_file_manifest_verification_v2",
    ):
        return "blocked_source_verification"
    archive = receipt.get("archive_verification")
    if not _verified_nested(
        archive,
        schema_version="task4_snapshot_archive_member_verification_v1",
    ):
        return "blocked_archive_member_verification"
    checksum = receipt.get("checksum_verification")
    if not _verified_nested(
        checksum,
        schema_version="task4_snapshot_archive_checksum_verification_v1",
    ):
        return "blocked_checksum_verification"
    if source.get("manifest_sha256") != receipt.get("manifest_sha256"):
        return "blocked_source_manifest_hash_mismatch"
    if source.get("expected_file_count") != file_count:
        return "blocked_source_expected_file_count_mismatch"
    if source.get("verified_file_count") != file_count:
        return "blocked_source_verified_file_count_mismatch"
    if archive.get("expected_file_count") != file_count:
        return "blocked_archive_expected_file_count_mismatch"
    if archive.get("verified_file_count") != file_count:
        return "blocked_archive_verified_file_count_mismatch"
    if checksum.get("expected_sha256") != receipt.get("archive_sha256"):
        return "blocked_checksum_expected_hash_mismatch"
    if checksum.get("observed_sha256") != receipt.get("archive_sha256"):
        return "blocked_checksum_observed_hash_mismatch"
    if checksum.get("hash_matches") is not True:
        return "blocked_checksum_hash_mismatch"
    if checksum.get("format_errors") != []:
        return "blocked_checksum_format_errors"
    if checksum.get("archive_is_symlink") is not False:
        return "blocked_archive_symlink"
    if checksum.get("checksum_is_symlink") is not False:
        return "blocked_checksum_symlink"
    return "verified_archive_build_receipt"


def _archive_extraction_receipt_status(receipt: dict[str, Any]) -> str:
    if receipt.get("schema_version") != "task4_snapshot_extraction_receipt_v1":
        return "blocked_wrong_extraction_schema"
    if receipt.get("status") != "verified":
        return "blocked_extraction_not_verified"
    if not _valid_sha256(receipt.get("manifest_sha256")):
        return "blocked_invalid_extraction_manifest_sha256"
    if not all(
        isinstance(receipt.get(field), str) and bool(receipt.get(field))
        for field in (
            "archive",
            "checksum",
            "manifest",
            "output_dir",
            "extracted_root",
            "root_name",
        )
    ):
        return "blocked_missing_extraction_paths"
    checksum = receipt.get("checksum_verification")
    if not _verified_nested(
        checksum,
        schema_version="task4_snapshot_archive_checksum_verification_v1",
    ):
        return "blocked_extraction_checksum_verification"
    archive = receipt.get("archive_verification")
    if not _verified_nested(
        archive,
        schema_version="task4_snapshot_archive_member_verification_v1",
    ):
        return "blocked_extraction_archive_verification"
    extracted = receipt.get("extracted_verification")
    if not _verified_nested(
        extracted,
        schema_version="task4_file_manifest_verification_v2",
    ):
        return "blocked_extracted_manifest_verification"
    if extracted.get("manifest_sha256") != receipt.get("manifest_sha256"):
        return "blocked_extracted_manifest_hash_mismatch"
    if checksum.get("hash_matches") is not True:
        return "blocked_extraction_checksum_hash_mismatch"
    if checksum.get("format_errors") != []:
        return "blocked_extraction_checksum_format_errors"
    counts = (
        archive.get("expected_file_count"),
        archive.get("verified_file_count"),
        extracted.get("expected_file_count"),
        extracted.get("verified_file_count"),
    )
    if not all(_positive_int(value) for value in counts):
        return "blocked_invalid_extraction_file_counts"
    if len(set(counts)) != 1:
        return "blocked_extraction_file_count_mismatch"
    return "verified_archive_extraction_receipt"


def _archive_pair_status(
    build_receipt: dict[str, Any],
    extraction_receipt: dict[str, Any],
) -> str:
    build_status = _archive_build_receipt_status(build_receipt)
    if build_status != "verified_archive_build_receipt":
        return build_status
    extraction_status = _archive_extraction_receipt_status(extraction_receipt)
    if extraction_status != "verified_archive_extraction_receipt":
        return extraction_status
    checksum = extraction_receipt["checksum_verification"]
    archive_sha256 = build_receipt["archive_sha256"]
    if checksum.get("expected_sha256") != archive_sha256:
        return "blocked_pair_expected_archive_hash_mismatch"
    if checksum.get("observed_sha256") != archive_sha256:
        return "blocked_pair_observed_archive_hash_mismatch"
    if extraction_receipt.get("manifest_sha256") != build_receipt.get(
        "manifest_sha256"
    ):
        return "blocked_pair_manifest_hash_mismatch"
    file_counts = (
        build_receipt["file_count"],
        build_receipt["source_verification"]["expected_file_count"],
        build_receipt["archive_verification"]["verified_file_count"],
        extraction_receipt["archive_verification"]["expected_file_count"],
        extraction_receipt["extracted_verification"]["verified_file_count"],
    )
    if len(set(file_counts)) != 1:
        return "blocked_pair_file_count_mismatch"
    if Path(str(build_receipt["root"])).name != extraction_receipt.get("root_name"):
        return "blocked_pair_root_name_mismatch"
    return "verified_archive_build_extract_pair"


def _gpu_homogeneity(
    resources: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    if set(resources) != {"campaign_p", "campaign_e"}:
        return "blocked_missing_campaign_resource_audit", []
    identities = {
        (
            str(item.get("name", "")),
            str(item.get("driver", "")),
            str(item.get("pci_bus_id", "")),
        )
        for audit in resources.values()
        for item in audit.get("gpu_identities", [])
        if isinstance(item, dict)
    }
    serialized = [
        {"name": name, "driver": driver, "pci_bus_id": bus}
        for name, driver, bus in sorted(identities)
    ]
    if len(identities) == 1 and all(all(value for value in row) for row in serialized):
        return "verified_uniform_gpu", serialized
    return "blocked_gpu_identity_heterogeneous_or_incomplete", serialized


def _status_rows(
    *,
    analyses: dict[str, dict[str, Any]],
    run_maps: dict[str, dict[str, Any]],
    resources: dict[str, dict[str, Any]],
    seal_audits: dict[str, dict[str, Any]],
    archive_pairs: dict[str, dict[str, Any]],
    gpu_status: str,
    blockers: list[str],
) -> list[dict[str, str]]:
    rows = []
    for campaign in ("campaign_p", "campaign_e"):
        analysis = analyses.get(campaign)
        rows.append(
            {
                "item": campaign + "_analysis",
                "classification": (
                    "verified"
                    if analysis and analysis["paper_inference_eligible"]
                    else "blocked"
                ),
                "evidence": analysis["campaign_status"] if analysis else "missing",
            }
        )
        run_map = run_maps.get(campaign)
        rows.append(
            {
                "item": campaign + "_run_map",
                "classification": (
                    "verified"
                    if run_map
                    and run_map["status"]
                    == "verified_formal_candidates_cover_analysis"
                    else "blocked"
                ),
                "evidence": run_map["status"] if run_map else "missing",
            }
        )
        resource = resources.get(campaign)
        rows.append(
            {
                "item": campaign + "_resource_audit",
                "classification": (
                    "verified"
                    if resource
                    and _resource_status(resource) == "verified_zero_fallback"
                    else "blocked"
                ),
                "evidence": _resource_status(resource) if resource else "missing",
            }
        )
        seal = seal_audits.get(campaign)
        rows.append(
            {
                "item": campaign + "_seal_readiness",
                "classification": (
                    "verified"
                    if seal and _seal_status(seal) == "verified_ready_to_seal"
                    else "blocked"
                ),
                "evidence": _seal_status(seal) if seal else "missing",
            }
        )
        archive_pair = archive_pairs.get(campaign)
        rows.append(
            {
                "item": campaign + "_local_archive",
                "classification": (
                    "verified"
                    if archive_pair
                    and archive_pair["status"]
                    == "verified_archive_build_extract_pair"
                    else "blocked"
                ),
                "evidence": archive_pair["status"] if archive_pair else "missing",
            }
        )
    rows.extend(
        [
            {
                "item": "uniform_gpu",
                "classification": (
                    "verified" if gpu_status == "verified_uniform_gpu" else "blocked"
                ),
                "evidence": gpu_status,
            },
            {
                "item": "paper_conclusion",
                "classification": "verified" if not blockers else "blocked",
                "evidence": "all gates passed" if not blockers else " | ".join(blockers),
            },
        ]
    )
    return rows


def _render_report(
    *,
    spec: dict[str, Any],
    spec_path: Path,
    table_rows: list[dict[str, str]],
    run_maps: dict[str, dict[str, Any]],
    resources: dict[str, dict[str, Any]],
    runtime_lock: dict[str, Any],
    seal_audits: dict[str, dict[str, Any]],
    archive_pairs: dict[str, dict[str, Any]],
    gpu_status: str,
    gpu_identities: list[dict[str, str]],
    blockers: list[str],
    classification: str,
) -> str:
    title = str(spec.get("title") or "AgentMemEval TASK4 论文级实验总报告")
    lines = [
        f"# {title}",
        "",
        "## 结论资格",
        "",
        f"- 报告分类：`{classification}`",
        f"- 论文推断资格：`{not blockers}`",
        f"- 研究规格：`{spec_path}`",
        f"- 研究规格 SHA-256：`{_sha256(spec_path)}`",
    ]
    if blockers:
        lines.extend(
            [
                "- 当前禁止把结果写成正式论文结论；以下项目尚未通过：",
                *[f"  - `{item}`" for item in blockers],
            ]
        )
    else:
        lines.append("- P/E、运行、归档与环境证据均通过 fail-closed 门禁。")
    lines.extend(
        [
            "",
            "## 方法与协议",
            "",
            "- Campaign P：8 人混合桌，Fact/Expr/FactExprSync/FactExprAsync 各 2；"
            "独立单位为同一 seed 下的一次完整 table/run。",
            "- Campaign E：一个 target 与 7 个 NoMemory 对手；五条件使用同 seed 配对，"
            "主要比较为四种记忆机制分别相对 NoMemory target。",
            "- 主终点为 final heldout/test BB/100；手牌、checkpoint 和同桌 Agent "
            "不作为独立重复。",
            "- Pilot 只用于阈值与功效规划；只有 aggregate `ready` 的 formal "
            "分析才允许进入正式推断。",
            "",
            "## 环境与同质性",
            "",
            f"- Formal runtime lock：`{runtime_lock.get('status', 'missing')}`",
            f"- GPU 同质性：`{gpu_status}`",
        ]
    )
    for identity in gpu_identities:
        lines.append(
            f"- GPU：{identity['name']}；driver={identity['driver']}；"
            f"pci_bus_id={identity['pci_bus_id']}"
        )
    lines.extend(["", "## Campaign P：混合桌结果", ""])
    _append_campaign_rows(lines, table_rows, "campaign_p")
    lines.extend(["", "## Campaign E：训练、泛化与 Generalization Gap", ""])
    _append_campaign_rows(lines, table_rows, "campaign_e")
    lines.extend(
        [
            "",
            "## 统计不确定性",
            "",
            "- 表中 n 是独立 seed 配对数；效应为同 seed 对比。",
            "- 主终点使用预注册多重比较流程；次要训练、chip 与 gap 指标"
            "不复用主终点 p 值。",
            "- 均值效应、bootstrap 95% CI 与 Holm 校正 p 值来自各 Campaign "
            "已哈希绑定的 V3 分析包。",
            "",
            "## 异常、排除与运行健康",
            "",
        ]
    )
    for campaign in ("campaign_p", "campaign_e"):
        run_map = run_maps.get(campaign, {})
        resource = resources.get(campaign, {})
        lines.append(
            f"- {campaign}: attempts={run_map.get('attempt_count', 0)}，"
            f"eligible={run_map.get('eligible_attempt_count', 0)}，"
            f"excluded={run_map.get('excluded_attempt_count', 0)}，"
            f"action fallback={resource.get('action_fallback_count', 'NA')}，"
            "experience revision fallback="
            f"{resource.get('experience_revision_fallback_count', 'NA')}。"
        )
    lines.extend(["", "## 资源成本", ""])
    for campaign in ("campaign_p", "campaign_e"):
        resource = resources.get(campaign, {})
        token = resource.get("token_accounting", {})
        lines.append(
            f"- {campaign}: wall={resource.get('campaign_wall_hours', 'NA')} h，"
            f"action requests={resource.get('action_request_count', 'NA')}，"
            f"token status=`{token.get('status', 'missing')}`，"
            f"estimated total tokens={token.get('estimated_total_tokens', 'NA')}。"
        )
    lines.extend(
        [
            "- 本地 vLLM 无 provider invoice；货币成本若不可获得，保持 unavailable，"
            "不得把 token 代理估计冒充账单。",
            "",
            "## 归档与可复现性",
            "",
            "- P/E 均要求服务器 build receipt 与本地 extraction receipt "
            "按 archive hash、manifest hash、文件数和根目录名成对一致。",
            "- `evidence_index.csv` 记录全部输入路径、状态和 SHA-256。",
            "- `study_effects.csv` 是本报告效应表的数据源；"
            "`verification_status.csv` 保存 verified/blocked 判定。",
            "- 本报告由 study spec 一键重建；输出目录必须不存在，避免覆盖旧报告。",
        ]
    )
    for campaign in ("campaign_p", "campaign_e"):
        seal = seal_audits.get(campaign, {})
        archive_pair = archive_pairs.get(campaign, {})
        build_receipt = archive_pair.get("build", {})
        lines.append(
            f"- {campaign} seal：`{_seal_status(seal) if seal else 'missing'}`；"
            f"matrix={seal.get('complete_latest_attempt_count', 'NA')}/"
            f"{seal.get('expected_matrix_count', 'NA')}；"
            f"quiet={seal.get('observed_quiet_seconds', 'NA')} s；"
            f"files={seal.get('file_count', 'NA')}；"
            f"bytes={seal.get('total_bytes', 'NA')}。"
        )
        lines.append(
            f"- {campaign} archive："
            f"`{archive_pair.get('status', 'missing')}`；"
            f"files={build_receipt.get('file_count', 'NA')}；"
            f"bytes={build_receipt.get('total_uncompressed_size_bytes', 'NA')}；"
            f"archive_sha256={build_receipt.get('archive_sha256', 'NA')}；"
            f"manifest_sha256={build_receipt.get('manifest_sha256', 'NA')}。"
        )
    lines.extend(
        [
            "",
            "## 局限性",
            "",
        ]
    )
    limitations = spec.get("limitations", [])
    if isinstance(limitations, list) and limitations:
        lines.extend(f"- {item}" for item in limitations)
    else:
        lines.append("- 未提供额外局限性说明；该缺失不应被解释为不存在局限。")
    lines.append("")
    return "\n".join(lines)


def _append_campaign_rows(
    lines: list[str],
    rows: list[dict[str, str]],
    campaign: str,
) -> None:
    selected = [row for row in rows if row.get("campaign") == campaign]
    if not selected:
        lines.append("当前没有可引用的分析行。")
        return
    lines.extend(
        [
            "| 对比 | 指标 | n | 均值效应 | bootstrap 95% CI | Holm p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in selected:
        p_value = row.get("holm_adjusted_p_value") or "NA"
        lines.append(
            f"| {row.get('contrast')} | {row.get('endpoint')} | "
            f"{row.get('n_seed_pairs')} | {row.get('mean_effect')} | "
            f"[{row.get('bootstrap_ci95_low')}, "
            f"{row.get('bootstrap_ci95_high')}] | {p_value} |"
        )


def _resolve_input(spec_path: Path, raw: Any) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path.resolve()
    return (spec_path.parent / path).resolve()


def _evidence(label: str, path: Path, status: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "label": label,
        "path": str(path),
        "sha256": _sha256(path),
        "status": status,
    }


def _write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
