"""Exercise independent object hashes, no-overwrite and cache-link boundaries."""

import hashlib

import pytest

from agentmemeval.path1_v1.operations import model_materialize
from agentmemeval.path1_v1.operations.contracts import MODEL_REVISION
from agentmemeval.path1_v1.operations.model_materialize import (
    ModelMaterializationError,
    _copy_verified,
    _resolve_cached_file,
    materialize_model,
    pinned_model_source,
)


def record(content, *, lfs):
    row = {"rfilename": "weights.bin", "size": len(content),
           "blobId": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()}
    if lfs:
        row["lfs"] = {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
    return row


@pytest.mark.parametrize("lfs", [False, True])
def test_copy_matches_official_object_and_has_independent_bytes(tmp_path, lfs):
    original = b"independently pinned fixture\n\0weights\xff"
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(original)
    result = _copy_verified(source, target, record(original, lfs=lfs))
    assert result["sha256"] == hashlib.sha256(original).hexdigest()
    assert target.read_bytes() == original and not target.is_symlink()
    source.write_bytes(b"later cache mutation")
    assert target.read_bytes() == original
    with pytest.raises(FileExistsError):
        _copy_verified(target, target, record(original, lfs=lfs))


@pytest.mark.parametrize("lfs", [False, True])
def test_same_size_corruption_rejected_and_partial_output_retained(tmp_path, lfs):
    source, target = tmp_path / "source", tmp_path / "partial"
    source.write_bytes(b"WRONG")
    with pytest.raises(ModelMaterializationError, match="pinned Hub object"):
        _copy_verified(source, target, record(b"RIGHT", lfs=lfs))
    assert source.read_bytes() == b"WRONG" and target.read_bytes() == b"WRONG"


def test_size_mismatch_does_not_create_output(tmp_path):
    source, target = tmp_path / "source", tmp_path / "new"
    source.write_bytes(b"short")
    with pytest.raises(ModelMaterializationError, match="source size"):
        _copy_verified(source, target, record(b"longer", lfs=True))
    assert not target.exists()


def test_pinned_catalog_has_all_official_shards_and_runtime_assets():
    manifest = pinned_model_source()
    rows = {r["rfilename"]: r for r in manifest["files"]}
    assert len(rows) == 16
    assert {"config.json", "tokenizer.json", "chat_template.jinja",
            "model.safetensors.index.json"} <= rows.keys()
    shards = [r for n, r in rows.items() if n.endswith(".safetensors")]
    assert len(shards) == 4 and sum(r["size"] for r in shards) == 19306310880
    assert all(r["size"] == r["lfs"]["size"] for r in shards)


def test_materialization_requires_pinned_layout_without_creating_destination(tmp_path):
    source, destination = tmp_path / "wrong", tmp_path / "output"
    source.mkdir()
    with pytest.raises(ModelMaterializationError, match="pinned revision"):
        materialize_model(source, destination)
    assert not destination.exists()


def test_cache_link_must_point_to_exact_expected_blob(tmp_path):
    content = b"fixed object"
    row = record(content, lfs=True)
    snapshot, blobs = tmp_path / "snapshot", tmp_path / "blobs"
    snapshot.mkdir()
    blobs.mkdir()
    blob = blobs / row["lfs"]["sha256"]
    blob.write_bytes(content)
    link = snapshot / row["rfilename"]
    try:
        link.symlink_to(blob)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    assert _resolve_cached_file(snapshot, blobs, row) == blob
    # Identical bytes in an unexpected object must not borrow the known identity.
    wrong = blobs / "wrong"
    wrong.write_bytes(content)
    bad = snapshot / "wrong.bin"
    bad.symlink_to(wrong)
    with pytest.raises(ModelMaterializationError, match="expected Hub blob"):
        _resolve_cached_file(snapshot, blobs, dict(row, rfilename="wrong.bin"))


def test_path_traversal_is_rejected_before_read(tmp_path):
    with pytest.raises(ModelMaterializationError, match="unsafe model filename"):
        _resolve_cached_file(tmp_path, tmp_path, {"rfilename": "../outside", "blobId": "a" * 40})


def test_full_materialization_preserves_cache_and_rejects_reuse(tmp_path, monkeypatch):
    cache = tmp_path / "models--Qwen--Qwen3.5-9B"
    snapshot = cache / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True)
    (cache / "blobs").mkdir()
    content = b"small offline complete fixture"
    row = record(content, lfs=False)
    (snapshot / row["rfilename"]).write_bytes(content)
    # Substitute only the official catalog for this offline filesystem test.
    monkeypatch.setattr(model_materialize, "pinned_model_source", lambda: {"files": [row]})
    destination = tmp_path / "copied"
    result = materialize_model(snapshot, destination)
    assert result["runtime_admission"] is False
    assert result["files"] == [{"path": row["rfilename"], "size": len(content),
                                "sha256": hashlib.sha256(content).hexdigest()}]
    assert (snapshot / row["rfilename"]).read_bytes() == content
    assert (destination / row["rfilename"]).read_bytes() == content
    with pytest.raises(FileExistsError):
        materialize_model(snapshot, destination)
    extra = snapshot / "unreviewed_file"
    extra.write_text("unexpected", encoding="utf-8")
    with pytest.raises(ModelMaterializationError, match="inventory differs"):
        materialize_model(snapshot, tmp_path / "not_created")
    assert not (tmp_path / "not_created").exists()


def test_resolved_shared_blob_directory_cannot_be_destination(tmp_path, monkeypatch):
    cache = tmp_path / "models--Qwen--Qwen3.5-9B"
    snapshot = cache / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True)
    shared = tmp_path / "shared_blobs"
    shared.mkdir()
    try:
        (cache / "blobs").symlink_to(shared, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    monkeypatch.setattr(model_materialize, "pinned_model_source", lambda: {"files": []})
    with pytest.raises(ModelMaterializationError, match="outside the shared model cache"):
        materialize_model(snapshot, shared / "would_modify_cache")
    assert list(shared.iterdir()) == []


def test_earlier_destination_mutation_during_later_copy_is_rejected(tmp_path, monkeypatch):
    cache = tmp_path / "models--Qwen--Qwen3.5-9B"
    snapshot = cache / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True)
    (cache / "blobs").mkdir()
    content = b"RIGHT"
    rows = [dict(record(content, lfs=True), rfilename=name) for name in ("first", "second")]
    for row in rows:
        (snapshot / row["rfilename"]).write_bytes(content)
    monkeypatch.setattr(model_materialize, "pinned_model_source", lambda: {"files": rows})
    original_copy = model_materialize._copy_verified

    def intervening_copy(source, target, expected):
        result = original_copy(source, target, expected)
        if expected["rfilename"] == "second":
            (target.parent / "first").write_bytes(b"WRONG")
        return result

    monkeypatch.setattr(model_materialize, "_copy_verified", intervening_copy)
    destination = tmp_path / "partial"
    with pytest.raises(ModelMaterializationError, match="changed before completion"):
        materialize_model(snapshot, destination)
    assert (destination / "first").read_bytes() == b"WRONG"
    assert all((snapshot / row["rfilename"]).read_bytes() == content for row in rows)
