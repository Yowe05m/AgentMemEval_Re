"""Archive evidence must not resolve differently on Linux and Windows recovery."""

import hashlib
import io
import tarfile

import pytest

from agentmemeval.path1_v1.storage.archive import _filesystem_path, verify_file_manifest
from agentmemeval.path1_v1.storage.snapshot_archive import (
    build_snapshot_archive,
    extract_snapshot_archive,
)

BAD_PATHS = [
    "../outside", "/outside", "C:/outside", "C:outside", "..\\outside",
    "dir\\outside", "data:stream", "a//b", "./file", "a/../b", "a/./b",
    "NUL.txt", "COM1", "dir/LPT9.log", "name.", "name ", "a?b",
]


def manifest(path, rows):
    path.write_text(
        "relative_path\tsize_bytes\tsha256\n"
        + "".join(f"{name}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}\n"
                  for name, data in rows), encoding="utf-8",
    )


@pytest.mark.parametrize("relative", BAD_PATHS)
def test_manifest_rejects_portable_path_escape_and_alias(tmp_path, relative):
    root = tmp_path / "run"
    root.mkdir()
    path = tmp_path / "files.tsv"
    manifest(path, [(relative, b"x")])
    result = verify_file_manifest(root, path)
    assert result["verified"] is False
    assert result["unsafe_or_duplicate_paths"] == [relative]
    assert result["verified_file_count"] == 0


def bundle(tmp_path, names, manifest_rows=None):
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "x:gz") as handle:
        for name in names:
            member = tarfile.TarInfo(name)
            member.size = 1
            handle.addfile(member, io.BytesIO(b"x"))
    checksum = tmp_path / "bundle.sha256"
    checksum.write_text(f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n",
                        encoding="ascii")
    table = tmp_path / "files.tsv"
    manifest(table, manifest_rows or [("file", b"x")])
    return archive, checksum, table


@pytest.mark.parametrize("relative", BAD_PATHS)
def test_archive_rejects_alias_before_creating_output(tmp_path, relative):
    paths = bundle(tmp_path, ["run/" + relative])
    output = tmp_path / "extracted"
    with pytest.raises(ValueError):
        extract_snapshot_archive(*paths, output, tmp_path / "receipt.json")
    assert not output.exists()


def test_archive_rejects_case_colliding_members_before_creating_output(tmp_path):
    paths = bundle(tmp_path, ["run/File", "run/file"], [("File", b"x")])
    with pytest.raises(ValueError, match="member verification"):
        extract_snapshot_archive(*paths, tmp_path / "out", tmp_path / "receipt.json")
    assert not (tmp_path / "out").exists()


def test_manifest_rejects_case_colliding_rows(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "file").write_bytes(b"x")
    table = tmp_path / "files.tsv"
    manifest(table, [("file", b"x"), ("FILE", b"x")])
    result = verify_file_manifest(root, table)
    assert result["verified"] is False
    assert result["unsafe_or_duplicate_paths"] == ["FILE"]


@pytest.mark.parametrize("tail", ["\tunexpected", ""])
def test_archive_rejects_invalid_manifest_columns_or_negative_size(tmp_path, tail):
    paths = bundle(tmp_path, ["run/file"])
    paths[2].write_text(
        "relative_path\tsize_bytes\tsha256\n"
        f"file\t-1\t{hashlib.sha256(b'x').hexdigest()}{tail}\n", encoding="utf-8",
    )
    with pytest.raises(ValueError):
        extract_snapshot_archive(*paths, tmp_path / "out", tmp_path / "receipt.json")
    assert not (tmp_path / "out").exists()


def test_archive_accepts_unicode_portable_relative_path(tmp_path):
    paths = bundle(tmp_path, ["run/子目录/证据.json"], [("子目录/证据.json", b"x")])
    receipt = extract_snapshot_archive(*paths, tmp_path / "out", tmp_path / "receipt.json")
    assert receipt["status"] == "verified"
    assert (tmp_path / "out/run/子目录/证据.json").read_bytes() == b"x"


def test_archive_round_trip_retains_long_path_evidence(tmp_path):
    root = tmp_path / "source"
    relative = "/".join(["a" * 90, "b" * 90, "c" * 60, "snapshot.json"])
    target = _filesystem_path(root / relative)
    target.parent.mkdir(parents=True)
    target.write_bytes(b'{"evidence":true}')
    archive, table, checksum = (tmp_path / name for name in ("b.tar.gz", "f.tsv", "s.sha256"))
    built = build_snapshot_archive(root, archive, table, checksum, tmp_path / "built.json")
    recovered = extract_snapshot_archive(archive, checksum, table,
                                         tmp_path / "out", tmp_path / "recovered.json")
    assert built["status"] == recovered["status"] == "verified"
    assert _filesystem_path(tmp_path / "out/source" / relative).read_bytes() == b'{"evidence":true}'
