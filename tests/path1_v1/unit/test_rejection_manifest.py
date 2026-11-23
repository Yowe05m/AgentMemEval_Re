"""Strict rejection-only historical TSV dialect; real results stay outside Git."""
from __future__ import annotations

import csv
import hashlib
import os
import sys
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations import e0_reconcile as er
from agentmemeval.path1_v1.operations import submission_rejection as sr

SHA = hashlib.sha256(b"anonymous fixture").hexdigest()
HEADER = "path\tsize\tsha256\n"
ROW = f"run/contract.json\t17\t{SHA}\n"


def read(tmp_path, text):
    path = tmp_path / "manifest.tsv"
    path.write_text(text, encoding="utf-8", newline="")
    return sr._baseline(path, er._sha(path))


def test_historical_dialect_preserves_invocation_prefix_and_validates_metadata(tmp_path):
    rows = read(tmp_path, HEADER + ROW + f"executor/invocation/result.json\t17\t{SHA}\n"
                + f"diagnostics/before.json\t17\t{SHA}\n")
    assert rows == [
        {"path": "executor/invocation/result.json", "size": 17, "sha256": SHA},
        {"path": "run/contract.json", "size": 17, "sha256": SHA},
    ]


def test_canonical_reader_unchanged_and_resource_reader_rejects_legacy(tmp_path):
    path = tmp_path / "canonical.tsv"
    path.write_text("relative_path\tsize_bytes\tsha256\n" + ROW, encoding="utf-8")
    assert sr._baseline(path, er._sha(path)) == er._baseline(path, er._sha(path))
    path.write_text(HEADER + ROW, encoding="utf-8")
    with pytest.raises(er.E0ReconciliationError, match="unsupported original"):
        er._baseline(path, er._sha(path))


@pytest.mark.parametrize("text", [
    "size\tpath\tsha256\n17\trun/x\t" + SHA + "\n",
    "path\tsize\tsha256\textra\n" + ROW,
    HEADER + ROW.rstrip("\n") + "\textra\n",
    HEADER + "run/x\t17\n",
    HEADER + ROW + ROW,
    HEADER + ROW + f"RUN/CONTRACT.JSON\t17\t{SHA}\n",
    HEADER + f"run/../secret\t17\t{SHA}\n",
    HEADER + f"/run/x\t17\t{SHA}\n",
    HEADER + f"run\\x\t17\t{SHA}\n",
    HEADER + f"run//x\t17\t{SHA}\n",
    HEADER + f"run/./x\t17\t{SHA}\n",
    HEADER + f"run/C:x\t17\t{SHA}\n",
    HEADER + f"run/x\x00\t17\t{SHA}\n",
    HEADER + f"executor/other/result.json\t17\t{SHA}\n",
    HEADER + f"diagnostics/unknown.json\t17\t{SHA}\n",
    HEADER + ROW + f"diagnostics/before.json\t-1\t{SHA}\n",
    HEADER + ROW + "diagnostics/before.json\t17\tbroken\n",
    HEADER + f"run/x\t01\t{SHA}\n",
    HEADER + f"run/x\t-1\t{SHA}\n",
    HEADER + "run/x\t17\tbad\n",
    HEADER,
    HEADER + f"control/campaign.sqlite3\t17\t{SHA}\n",
])
def test_historical_dialect_rejects_malformed_scope_and_content(tmp_path, text):
    with pytest.raises((er.E0ReconciliationError, ValueError)):
        read(tmp_path, text)


def test_original_sha_commitment_is_checked_before_dialect(tmp_path):
    path = tmp_path / "manifest.tsv"
    path.write_text(HEADER + ROW, encoding="utf-8")
    with pytest.raises(er.E0ReconciliationError, match="baseline manifest changed"):
        sr._baseline(path, "1" * 64)


def test_external_original_tsv_and_all_original_bytes_are_unchanged():
    location = os.environ.get("PATH1_REJECTION_ORIGINAL_ARCHIVE")
    if location is None:
        pytest.skip("set PATH1_REJECTION_ORIGINAL_ARCHIVE for explicit read-only historical replay")
    root = Path(location)
    manifest = root / "snapshot.files.tsv"
    original = manifest.read_bytes()
    expected = "611645e36df136dc2a753b5239b73f93ab6ef8bb1689e8f2cb6c54c7241a7568"
    assert hashlib.sha256(original).hexdigest() == expected
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 34
    for row in rows:
        raw = (root / "extracted" / row["path"]).read_bytes()
        assert len(raw) == int(row["size"])
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    mapped = sr._baseline(manifest, expected)
    actual = er._content(er._content_inventory(root / "extracted/run",
                                               root / "extracted/executor/invocation"))
    assert len(mapped) == 29 and mapped == actual
    assert manifest.read_bytes() == original


@pytest.mark.skipif(sys.platform != "linux", reason="canonical service fixture uses Linux paths")
@pytest.mark.parametrize("missing_invocation", [False, True])
def test_legacy_dialect_full_seal_or_missing_file_preserves_claim(
    tmp_path, monkeypatch, missing_invocation
):
    from test_submission_rejection import make_rejection_fixture

    data = make_rejection_fixture(tmp_path, monkeypatch)
    path = data.args["baseline_manifest"]
    raw = path.read_text(encoding="utf-8")
    assert raw.startswith("relative_path\tsize_bytes\tsha256\n")
    converted = HEADER + raw.split("\n", 1)[1]
    if missing_invocation:
        converted = "\n".join(line for line in converted.split("\n")
                              if not line.startswith("executor/invocation/result.json\t"))
    path.write_text(converted, encoding="utf-8", newline="")
    data.args["baseline_manifest_sha256"] = er._sha(path)
    before = data.store.read_registry()
    if missing_invocation:
        with pytest.raises(er.E0ReconciliationError, match="original artifact baseline differs"):
            sr.reconcile_rejection(data.store, **data.args, check_only=True)
        assert data.store.read_registry() == before
        assert not (data.store.directory / "submission_rejections").exists()
    else:
        checked = sr.reconcile_rejection(data.store, **data.args, check_only=True)
        assert checked["claim_released"] is False and data.store.read_registry() == before
        result = sr.reconcile_rejection(data.store, **data.args)
        assert result.job_id is None and result.retry_eligible is False
        assert result.scientific_admission is False
        after = data.store.read_registry()
        assert after.registry == before.registry and after.generation == before.generation + 1
        assert after.claims[0].status == "failed"
        assert sr.resolve_rejection(data.store, data.request.intent_id) == result
