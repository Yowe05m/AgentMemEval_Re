from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from agentmemeval.core.errors import ConfigError
from agentmemeval.experiments.task8b_phase_f_adoption import (
    ADOPTED_PAPER_ELIGIBILITY,
    CANONICAL_SOURCE_MAP_SHA256,
    EXPECTED_WORKERS,
    compare_and_adopt_task8b_phase_f,
    main,
)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_canonical_root(root: Path) -> None:
    manifest_path = root / "canonical_file_manifest.tsv"
    seal_path = root / "canonical_root_seal.json"
    for path in (manifest_path, seal_path):
        if path.exists():
            path.unlink()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            lineterminator="\n",
            fieldnames=["relative_path", "size", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)
    seal = {
        "schema_version": "task8b-six-seed-canonical-root-seal-v1",
        "status": "sealed_with_declared_limitation",
        "source_map_sha256": CANONICAL_SOURCE_MAP_SHA256,
        "canonical_file_manifest": {
            "relative_path": manifest_path.name,
            "size": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        },
        "manifest_entry_count": len(rows),
        "manifest_entry_bytes": sum(int(row["size"]) for row in rows),
        "published_tree_file_count": len(rows) + 2,
        "paper_eligibility": "PENDING_DECLARED_LIMITATION_REVIEW",
        "limitations": [
            {
                "code": "test-declared-limitation",
                "scope": "fixture",
                "evidence_status": "DECLARED",
                "impact": "review required",
            }
        ],
        "effect_fields_read": False,
        "formal_effect_fields_read": False,
    }
    seal_path.write_text(
        json.dumps(seal, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _tree_identity(root: Path) -> list[tuple[str, int, str]]:
    return [
        (path.relative_to(root).as_posix(), path.stat().st_size, _sha256(path))
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _make_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    canonical = tmp_path / "canonical"
    provisional = tmp_path / "provisional"
    canonical.mkdir()
    provisional.mkdir()
    workers: list[dict[str, object]] = []
    first_payload = ""
    for worker_id in EXPECTED_WORKERS:
        seed = 2026090100 + int(worker_id[1:])
        attempt = f"snapshots/{worker_id}/{seed}/attempt_01"
        payload_path = f"{attempt}/runs/task/payload.bin"
        _write(canonical / payload_path, f"payload-{worker_id}\n".encode())
        if not first_payload:
            first_payload = payload_path
        _write(canonical / f"manifests/{worker_id}.json", b"{}\n")
        _write(canonical / f"manifests/composition/{worker_id}.json", b"{}\n")
        _write(
            canonical / f"manifests/composition/execution/{worker_id}/receipt.json",
            b"{}\n",
        )
        workers.append(
            {
                "worker_id": worker_id,
                "attempts": [{"attempt": 1, "relative_path": attempt}],
                "shard_lineage": {
                    "composition_receipt_path": f"manifests/composition/{worker_id}.json"
                },
            }
        )
    _write(canonical / "manifests/manifest_index.json", b"{}\n")
    _write(
        canonical / "analysis_input.json",
        (json.dumps({"workers": workers}, sort_keys=True) + "\n").encode(),
    )
    _write(canonical / "exclusion_ledger.csv", b"ledger_entry_id\n")
    _write(canonical / "provenance/archive_receipts/source.json", b"ignored\n")
    _write(canonical / "materialization_note.txt", b"ignored root file\n")

    input_root = provisional / "01_INPUT"
    consumption_files = [canonical / "analysis_input.json", canonical / "exclusion_ledger.csv"]
    consumption_files.extend(
        path
        for root in (canonical / "manifests", canonical / "snapshots")
        for path in root.rglob("*")
        if path.is_file()
    )
    rows: list[dict[str, object]] = []
    for source in sorted(consumption_files):
        relative = source.relative_to(canonical)
        target = input_root / relative
        _write(target, source.read_bytes())
        rows.append(
            {
                "relative_path": f"01_INPUT/{relative.as_posix()}",
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
                "evidence_status": "PROVISIONAL_UNVERIFIED",
                "paper_eligibility": "NOT_YET_PAPER_ELIGIBLE",
            }
        )
    receipt_root = provisional / "03_RECEIPTS"
    receipt_root.mkdir()
    with (receipt_root / "phase_f_consumed_input_sha256.tsv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            lineterminator="\n",
            fieldnames=[
                "relative_path",
                "bytes",
                "sha256",
                "evidence_status",
                "paper_eligibility",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    _write(receipt_root / "PROVISIONAL_STATUS.json", b"{}\n")
    _seal_canonical_root(canonical)
    return canonical, provisional, first_payload


def test_cli_adopts_exact_consumption_set_and_ignores_provenance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canonical, provisional, _first_payload = _make_fixture(tmp_path)
    before = _tree_identity(provisional)
    output = tmp_path / "adoption"

    assert (
        main(
            [
                "--canonical-root",
                str(canonical),
                "--provisional-root",
                str(provisional),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    receipt = json.loads(
        (output / "provisional_adoption_receipt.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "adopted"
    assert result["adopted_without_rerun"] is True
    assert result["formal_adopted"] is True
    assert result["effect_fields_read"] is False
    assert receipt["adoption_status"] == "FORMAL_ADOPTED"
    assert receipt["paper_eligibility"] == ADOPTED_PAPER_ELIGIBILITY
    assert receipt["exact_path_size_sha256_equality"] is True
    assert receipt["mismatch_count"] == 0
    assert receipt["provisional_tree_modified"] is False
    materialization = json.loads(
        (output / "materialization_completion_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert materialization["full_canonical_validation_performed"] is True
    assert materialization["canonical_file_manifest_sha256"] == _sha256(
        canonical / "canonical_file_manifest.tsv"
    )
    assert materialization["canonical_root_seal_sha256"] == _sha256(
        canonical / "canonical_root_seal.json"
    )
    assert materialization["declared_limitations"]
    assert before == _tree_identity(provisional)
    assert {path.name for path in output.iterdir()} == {
        "canonical_rehash_sha256.tsv",
        "comparison.tsv",
        "materialization_completion_receipt.json",
        "provisional_adoption_receipt.json",
        "adoption_package_sha256.tsv",
    }
    package_rows = list(
        csv.DictReader(
            (output / "adoption_package_sha256.tsv").open(encoding="utf-8"),
            delimiter="\t",
        )
    )
    assert {row["relative_path"] for row in package_rows} == {
        "canonical_rehash_sha256.tsv",
        "comparison.tsv",
        "materialization_completion_receipt.json",
        "provisional_adoption_receipt.json",
    }


def test_missing_canonical_consumed_file_isolates_provisional(tmp_path: Path) -> None:
    canonical, provisional, first_payload = _make_fixture(tmp_path)
    (canonical / first_payload).unlink()
    _seal_canonical_root(canonical)

    result = compare_and_adopt_task8b_phase_f(
        canonical,
        provisional,
        tmp_path / "missing-adoption",
    )

    receipt = json.loads(
        (tmp_path / "missing-adoption/provisional_adoption_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "isolated"
    assert result["formal_adopted"] is False
    assert receipt["comparison_counts"]["missing_canonical"] == 1
    assert receipt["adopted_without_rerun"] is False


def test_extra_canonical_file_inside_consumption_scope_isolates(tmp_path: Path) -> None:
    canonical, provisional, first_payload = _make_fixture(tmp_path)
    _write((canonical / first_payload).parent / "extra.bin", b"extra\n")
    _seal_canonical_root(canonical)

    result = compare_and_adopt_task8b_phase_f(
        canonical,
        provisional,
        tmp_path / "extra-adoption",
    )

    receipt = json.loads(
        (tmp_path / "extra-adoption/provisional_adoption_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "isolated"
    assert receipt["comparison_counts"]["extra_canonical"] == 1
    assert receipt["exact_path_size_sha256_equality"] is False


def test_same_size_sha256_mismatch_isolates(tmp_path: Path) -> None:
    canonical, provisional, first_payload = _make_fixture(tmp_path)
    payload = canonical / first_payload
    payload.write_bytes(b"X" * payload.stat().st_size)
    _seal_canonical_root(canonical)

    result = compare_and_adopt_task8b_phase_f(
        canonical,
        provisional,
        tmp_path / "hash-adoption",
    )

    receipt = json.loads(
        (tmp_path / "hash-adoption/provisional_adoption_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "isolated"
    assert receipt["comparison_counts"]["sha256_mismatch"] == 1
    assert receipt["comparison_counts"]["size_mismatch"] == 0


@pytest.mark.parametrize("tamper_kind", ["provenance", "root_extra"])
def test_tampered_non_consumed_full_tree_fails_before_adoption(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    canonical, provisional, _first_payload = _make_fixture(tmp_path)
    if tamper_kind == "provenance":
        (canonical / "provenance/archive_receipts/source.json").write_bytes(b"tampered\n")
    else:
        _write(canonical / "unexpected_root_file.bin", b"extra\n")
    output = tmp_path / f"tampered-{tamper_kind}-adoption"

    with pytest.raises(ConfigError, match="canonical root manifest"):
        compare_and_adopt_task8b_phase_f(canonical, provisional, output)

    assert not output.exists()


def test_existing_output_directory_is_never_modified(tmp_path: Path) -> None:
    canonical, provisional, _first_payload = _make_fixture(tmp_path)
    output = tmp_path / "existing-adoption"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_bytes(b"preserve\n")

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        compare_and_adopt_task8b_phase_f(canonical, provisional, output)

    assert sentinel.read_bytes() == b"preserve\n"
    assert list(output.iterdir()) == [sentinel]
