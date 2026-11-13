import os
import tarfile

from agentmemeval.path1_v1.storage.snapshot_archive import (
    build_snapshot_archive,
    extract_snapshot_archive,
)


def test_published_hardlinks_archive_as_regular_files_and_recover_independently(tmp_path):
    source = tmp_path / "job"
    source.mkdir()
    published = source / "terminal.json"
    staging = source / "_publication_staging"
    staging.mkdir()
    pending = staging / "terminal.pending"
    pending.write_bytes(b'{"status":"fixture"}\n')
    os.link(pending, published)
    assert pending.samefile(published)
    archive, manifest, checksum, receipt = [
        tmp_path / name for name in ("job.tar.gz", "files.tsv", "archive.sha256", "build.json")
    ]
    result = build_snapshot_archive(source, archive, manifest, checksum, receipt)
    assert result["archive_verification"]["verified"]
    with tarfile.open(archive, "r:gz") as handle:
        files = [member for member in handle.getmembers() if not member.isdir()]
        assert len(files) == 2 and all(member.isfile() for member in files)
        assert not any(member.islnk() or member.issym() for member in files)
    extract_snapshot_archive(
        archive, checksum, manifest, tmp_path / "recovered", tmp_path / "extract.json"
    )
    recovered = tmp_path / "recovered/job"
    assert (recovered / "terminal.json").read_bytes() == pending.read_bytes()
    assert (
        recovered / "_publication_staging/terminal.pending"
    ).read_bytes() == pending.read_bytes()
    assert not (recovered / "terminal.json").samefile(
        recovered / "_publication_staging/terminal.pending"
    )
