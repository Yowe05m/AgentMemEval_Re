from __future__ import annotations

import csv
from pathlib import Path

import pytest

from agentmemeval.experiments import task8b_frozen_secondary_launcher as frozen_launcher
from agentmemeval.experiments import task8b_rolling_fanout as fanout


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fanout._json_bytes(value))


def _row(bundle_root: Path, producer: str) -> dict[str, str]:
    suffix = producer.removeprefix("P")
    receipt = (
        bundle_root / "receipts" / f"{producer}.json"
        if producer == "P12"
        else bundle_root / "receipt_root" / "receipts" / f"{producer}.json"
    )
    return {
        "producer_worker": producer,
        "producer_receipt_path": str(receipt),
        "consumer_worker": f"S{suffix}",
    }


def test_receipt_root_for_p01_uses_legacy_child(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"

    assert fanout.receipt_root_for_row(_row(bundle_root, "P01"), bundle_root) == (
        bundle_root / "receipt_root"
    ).resolve()


def test_receipt_root_for_p12_uses_bundle_root_exception(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"

    assert fanout.receipt_root_for_row(_row(bundle_root, "P12"), bundle_root) == (
        bundle_root.resolve()
    )


def test_receipt_root_for_p12_rejects_legacy_child(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    row = _row(bundle_root, "P12")
    row["producer_receipt_path"] = str(
        bundle_root / "receipt_root" / "receipts" / "P12.json"
    )

    with pytest.raises(fanout.RollingFanoutError, match="P12 receipt-root binding mismatch"):
        fanout.receipt_root_for_row(row, bundle_root)


def test_build_launch_argv_is_fresh_argument_array(tmp_path: Path) -> None:
    python = tmp_path / "bin" / "python"
    launcher = Path(frozen_launcher.__file__).resolve()
    checkout = tmp_path / "checkout"
    manifest = tmp_path / "bundle" / "manifests" / "S12.json"
    receipt_root = tmp_path / "bundle"

    argv = fanout.build_launch_argv(
        python, launcher, checkout, manifest, receipt_root
    )

    assert argv == [
        str(python.resolve()),
        str(launcher.resolve()),
        "--checkout",
        str(checkout.resolve()),
        "--manifest",
        str(manifest.resolve()),
        "--receipt-root",
        str(receipt_root.resolve()),
    ]
    assert "--resume-existing" not in argv
    assert "formal-worker" not in argv
    assert all(not any(operator in arg for operator in ("&&", ";", "|")) for arg in argv)


@pytest.mark.parametrize("existing", ["output", "cache", "attempt"])
def test_fresh_consumer_paths_reject_existing_artifacts(
    tmp_path: Path, existing: str
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    manifest = {
        "instance_identity": {
            "output_path": "outputs/formal/task8b/S12/2026090112",
            "cache_namespace": "task8b/S12/2026090112",
        }
    }
    output = checkout / "outputs" / "formal" / "task8b" / "S12" / "2026090112"
    cache = checkout / "task8b" / "S12" / "2026090112"
    if existing == "output":
        output.mkdir(parents=True)
    elif existing == "cache":
        cache.mkdir(parents=True)
    else:
        output.with_name("2026090112__attempt_02").mkdir(parents=True)

    with pytest.raises(fanout.RollingFanoutError, match="fresh consumer"):
        fanout._assert_fresh_consumer_paths(checkout, manifest)


def test_load_manifest_rejects_sha_mismatch_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "S12.json"
    _write_json(manifest_path, {"worker_id": "S12", "role": "secondary"})
    validation_called = False

    def _unexpected_validation(value: object) -> object:
        nonlocal validation_called
        validation_called = True
        return value

    monkeypatch.setattr(fanout, "validate_worker_manifest", _unexpected_validation)

    with pytest.raises(fanout.RollingFanoutError, match="S12 manifest SHA-256 mismatch"):
        fanout._load_manifest(
            manifest_path,
            worker_id="S12",
            role="secondary",
            expected_sha="0" * 64,
        )
    assert validation_called is False


def _release(tmp_path: Path) -> tuple[dict[str, object], Path, Path, Path]:
    checkout = tmp_path / "checkout"
    python = tmp_path / "bin" / "python"
    launcher = Path(frozen_launcher.__file__).resolve()
    manifest = tmp_path / "bundle" / "manifests" / "S12.json"
    receipt_root = tmp_path / "bundle"
    control_root = receipt_root / "control" / "task8b" / "rolling_fanout"
    ledger_root = control_root / "H12" / "S12"
    consumer_output = checkout / "outputs/formal/task8b/S12/2026090112"
    consumer_cache = checkout / "task8b/S12/2026090112"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"frozen-python")
    _write_json(manifest, {"worker_id": "S12", "role": "secondary"})
    draft: dict[str, object] = {
        "schema_version": fanout.RELEASE_SCHEMA,
        "active": False,
        "physical_slot": "H12",
        "producer_worker_id": "P12",
        "consumer_worker_id": "S12",
        "seed": 2026090112,
        "release_id": "release-12",
        "scientific_checkout": str(checkout.resolve()),
        "consumer_manifest_path": str(manifest.resolve()),
        "consumer_manifest_sha256": fanout._sha256_file(manifest),
        "receipt_root": str(receipt_root.resolve()),
        "python_path": str(python.resolve()),
        "python_sha256": fanout._sha256_file(python),
        "launcher_path": str(launcher.resolve()),
        "launcher_sha256": fanout.FROZEN_SECONDARY_LAUNCHER_SHA256,
        "launch_argv": fanout.build_launch_argv(
            python, launcher, checkout, manifest, receipt_root
        ),
        "control_root": str(control_root.resolve()),
        "launch_ledger_root": str(ledger_root.resolve()),
        "control_attempt": 1,
        "previous_launch_terminal_sha256": "GENESIS",
        "consumer_output_path": str(consumer_output.resolve()),
        "consumer_cache_path": str(consumer_cache.resolve()),
        "archive_sha256": "a" * 64,
        "archive_manifest_sha256": "b" * 64,
        "resume_existing": False,
        "effect_metrics_read": False,
    }
    return draft, checkout, manifest, launcher


def _activated(draft: dict[str, object]) -> dict[str, object]:
    local_root = Path(str(draft["receipt_root"])).parent / "local"
    archive = (local_root / "P12.tar.gz").resolve()
    manifest = (local_root / "P12.files.csv").resolve()
    checksum = (local_root / "P12.sha256").resolve()
    extraction_receipt_path = (local_root / "P12.extract.json").resolve()
    extraction = {
        "schema_version": "task4_snapshot_extraction_receipt_v1",
        "status": "verified",
        "archive": str(archive),
        "manifest": str(manifest),
        "checksum": str(checksum),
        "manifest_sha256": draft["archive_manifest_sha256"],
        "checksum_verification": {
            "verified": True,
            "observed_sha256": draft["archive_sha256"],
        },
        "archive_verification": {"verified": True},
        "extracted_verification": {"verified": True},
    }
    value = dict(draft)
    value.update(
        {
            "schema_version": fanout.ACTIVATION_SCHEMA,
            "active": True,
            "activated_from_sha256": fanout._sha256_bytes(fanout._json_bytes(draft)),
            "local_archive_path": str(archive),
            "local_archive_sha256": draft["archive_sha256"],
            "local_manifest_path": str(manifest),
            "local_manifest_sha256": draft["archive_manifest_sha256"],
            "local_checksum_path": str(checksum),
            "local_checksum_sha256": "c" * 64,
            "local_extraction_receipt_path": str(extraction_receipt_path),
            "local_extraction_receipt_file_sha256": "d" * 64,
            "local_extraction_receipt_sha256": fanout._sha256_bytes(
                fanout._json_bytes(extraction)
            ),
            "local_extraction_receipt": extraction,
            "activation_controller_sha256": fanout._sha256_source(
                Path(fanout.__file__).resolve()
            ),
            "local_per_file_hash_verified": True,
        }
    )
    return value


def _patch_launch_dependencies(
    monkeypatch: pytest.MonkeyPatch, checkout: Path
) -> tuple[Path, Path, list[dict[str, Path]]]:
    monkeypatch.setattr(
        fanout,
        "_load_manifest",
        lambda *args, **kwargs: {
            "worker_id": "S12",
            "role": "secondary",
            "instance_identity": {
                "output_path": "outputs/formal/task8b/S12/2026090112",
                "cache_namespace": "task8b/S12/2026090112",
            },
        },
    )
    state_path = checkout / "probe_state.tsv"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("status\nrunning\n", encoding="utf-8")
    monkeypatch.setattr(
        fanout,
        "_probe_secondary_running",
        lambda *args, **kwargs: (state_path, "e" * 64),
    )
    signature_path = checkout.parent / "activation.sig"
    allowed_signers_path = checkout.parent / "allowed_signers"
    signature_path.write_bytes(b"placeholder-signature")
    allowed_signers_path.write_bytes(b"placeholder-allowed-signers")
    verification_calls: list[dict[str, Path]] = []

    def _record_signature_verification(**kwargs: Path) -> None:
        verification_calls.append(kwargs)

    monkeypatch.setattr(
        fanout, "_verify_activation_signature", _record_signature_verification
    )
    return signature_path, allowed_signers_path, verification_calls
    monkeypatch.setattr(
        fanout,
        "_assert_fresh_consumer_paths",
        lambda *args, **kwargs: {
            "output": checkout / "outputs/formal/task8b/S12/2026090112",
            "cache": checkout / "task8b/S12/2026090112",
        },
    )


def test_launch_rejects_activation_not_bound_to_rebuilt_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft, checkout, _manifest, _cli = _release(tmp_path)
    activated = _activated(draft)
    activated_path = tmp_path / "activated.json"
    _write_json(activated_path, activated)
    rebuilt = dict(draft)
    rebuilt["release_id"] = "changed-release"
    signature, allowed, _calls = _patch_launch_dependencies(monkeypatch, checkout)

    with pytest.raises(
        fanout.RollingFanoutError,
        match="activated release does not match rebuilt current bindings",
    ):
        fanout.launch_secondary(
            activated_path=activated_path,
            activation_signature_path=signature,
            allowed_signers_path=allowed,
            rebuilt_draft=rebuilt,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
        )
    assert not (tmp_path / "bundle" / "control").exists()


def test_launch_requires_active_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft, checkout, _manifest, _cli = _release(tmp_path)
    inactive_path = tmp_path / "inactive.json"
    _write_json(inactive_path, draft)
    signature, allowed, _calls = _patch_launch_dependencies(monkeypatch, checkout)

    with pytest.raises(fanout.RollingFanoutError, match="activated release is not active"):
        fanout.launch_secondary(
            activated_path=inactive_path,
            activation_signature_path=signature,
            allowed_signers_path=allowed,
            rebuilt_draft=draft,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
        )
    assert not (tmp_path / "bundle" / "control").exists()


def test_launch_uses_exclusive_claim_and_never_resume_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft, checkout, _manifest, _cli = _release(tmp_path)
    activated_path = tmp_path / "activated.json"
    _write_json(activated_path, _activated(draft))
    signature, allowed, signature_calls = _patch_launch_dependencies(
        monkeypatch, checkout
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    class _Process:
        pid = 4242

    def _popen(argv: list[str], **kwargs: object) -> _Process:
        calls.append((argv, kwargs))
        return _Process()

    launched = fanout.launch_secondary(
        activated_path=activated_path,
        activation_signature_path=signature,
        allowed_signers_path=allowed,
        rebuilt_draft=draft,
        stdout_path=tmp_path / "first.stdout.log",
        stderr_path=tmp_path / "first.stderr.log",
        popen_factory=_popen,
    )

    assert launched["pid"] == 4242
    assert launched["resume_existing"] is False
    assert launched["state_tail_row_sha256"] == "e" * 64
    assert signature_calls == [
        {
            "activated_path": activated_path.resolve(),
            "signature_path": signature.resolve(),
            "allowed_signers_path": allowed.resolve(),
        }
    ]
    assert "--resume-existing" not in launched["launch_argv"]
    assert calls[0][0] == draft["launch_argv"]
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["cwd"] == str(checkout.resolve())
    assert calls[0][1]["env"]["PYTHONPATH"] == ""
    assert calls[0][1]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    claim = (
        tmp_path
        / "bundle"
        / "control"
        / "task8b"
        / "rolling_fanout"
        / "H12"
        / "S12"
        / "attempt_0001.claim.json"
    )
    launched_receipt = claim.with_name("attempt_0001.launched.json")
    assert claim.is_file()
    assert launched_receipt.is_file()
    assert Path(str(draft["launch_ledger_root"])) == claim.parent.resolve()
    terminal_staging = (
        tmp_path
        / "bundle"
        / "control"
        / "task8b"
        / "rolling_fanout"
        / ".s"
    )
    staged = list(terminal_staging.iterdir())
    assert len(staged) == 1
    assert len(staged[0].name) == 64
    assert all(character in "0123456789abcdef" for character in staged[0].name)
    assert fanout._sha256_file(staged[0]) == fanout._sha256_file(launched_receipt)

    with pytest.raises(fanout.RollingFanoutError, match="secondary was already launched"):
        fanout.launch_secondary(
            activated_path=activated_path,
            activation_signature_path=signature,
            allowed_signers_path=allowed,
            rebuilt_draft=draft,
            stdout_path=tmp_path / "second.stdout.log",
            stderr_path=tmp_path / "second.stderr.log",
            popen_factory=_popen,
        )
    assert len(calls) == 1


def test_launch_failure_publishes_terminal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft, checkout, _manifest, _cli = _release(tmp_path)
    activated_path = tmp_path / "activated.json"
    _write_json(activated_path, _activated(draft))
    signature, allowed, signature_calls = _patch_launch_dependencies(
        monkeypatch, checkout
    )

    def _failed_popen(argv: list[str], **kwargs: object) -> object:
        raise OSError("bounded launch failure")

    with pytest.raises(OSError, match="bounded launch failure"):
        fanout.launch_secondary(
            activated_path=activated_path,
            activation_signature_path=signature,
            allowed_signers_path=allowed,
            rebuilt_draft=draft,
            stdout_path=tmp_path / "failed.stdout.log",
            stderr_path=tmp_path / "failed.stderr.log",
            popen_factory=_failed_popen,
        )

    ledger_root = Path(str(draft["launch_ledger_root"]))
    claim = ledger_root / "attempt_0001.claim.json"
    failed = ledger_root / "attempt_0001.failed.json"
    assert claim.is_file()
    assert failed.is_file()
    terminal = fanout._read_json(failed)
    assert terminal["schema_version"] == fanout.FAILED_SCHEMA
    assert terminal["claim_sha256"] == fanout._sha256_file(claim)
    assert terminal["failure_stage"] == "process-launch"
    assert terminal["safe_retry_allowed"] is True
    assert len(signature_calls) == 1


def test_launch_rejects_non_authoritative_ledger_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft, checkout, _manifest, _cli = _release(tmp_path)
    rogue_root = tmp_path / "rogue-control" / "H12" / "S12"
    draft["launch_ledger_root"] = str(rogue_root.resolve())
    activated_path = tmp_path / "activated.json"
    _write_json(activated_path, _activated(draft))
    signature, allowed, _calls = _patch_launch_dependencies(monkeypatch, checkout)

    with pytest.raises(
        fanout.RollingFanoutError,
        match="authoritative launch ledger root mismatch",
    ):
        fanout.launch_secondary(
            activated_path=activated_path,
            activation_signature_path=signature,
            allowed_signers_path=allowed,
            rebuilt_draft=draft,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
        )
    assert not rogue_root.exists()


def test_frozen_launcher_rejects_wrong_runner_sha_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    runner = checkout / "src" / "agentmemeval" / "experiments" / "formal_runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_bytes(b"not-the-frozen-runner")
    import_called = False

    def _unexpected_import(name: str) -> object:
        nonlocal import_called
        import_called = True
        raise AssertionError(name)

    monkeypatch.setattr(frozen_launcher.importlib, "import_module", _unexpected_import)

    with pytest.raises(
        frozen_launcher.FrozenSecondaryLaunchError,
        match="frozen formal runner SHA-256 mismatch",
    ):
        frozen_launcher.run_frozen_secondary(
            checkout,
            tmp_path / "S12.json",
            tmp_path / "bundle",
        )
    assert import_called is False


def _write_state_chain(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    previous = "GENESIS"
    for index, (status, detail) in enumerate(
        (
            ("planned", "manifest admitted"),
            ("validating", "identity verified"),
            ("running", "experiment_configs"),
        ),
        start=1,
    ):
        row = {
            "schema_version": "task8-worker-state-v1",
            "created_at_utc": f"2026-07-22T18:00:0{index}+00:00",
            "status": status,
            "detail": detail,
            "previous_sha256": previous,
        }
        row["row_sha256"] = fanout._sha256_json(row)
        previous = row["row_sha256"]
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_probe_secondary_running_verifies_real_state_hash_chain(tmp_path: Path) -> None:
    output = tmp_path / "S12"
    rows = _write_state_chain(output / "state.tsv")

    class _RunningProcess:
        @staticmethod
        def poll() -> None:
            return None

    state_path, tail_sha = fanout._probe_secondary_running(
        _RunningProcess(), output, timeout_seconds=0.1
    )

    assert state_path == (output / "state.tsv").resolve()
    assert tail_sha == rows[-1]["row_sha256"]


@pytest.mark.parametrize(
    ("tamper_field", "message"),
    [
        ("previous_sha256", "secondary state previous hash mismatch"),
        ("detail", "secondary state row SHA-256 mismatch"),
    ],
)
def test_probe_secondary_running_rejects_tampered_state_chain(
    tmp_path: Path, tamper_field: str, message: str
) -> None:
    output = tmp_path / "S12"
    state_path = output / "state.tsv"
    rows = _write_state_chain(state_path)
    target = 1 if tamper_field == "previous_sha256" else 2
    rows[target][tamper_field] = "tampered"
    with state_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    class _RunningProcess:
        @staticmethod
        def poll() -> None:
            return None

    with pytest.raises(fanout.RollingFanoutError, match=message):
        fanout._probe_secondary_running(_RunningProcess(), output, timeout_seconds=0.1)


def test_activation_signature_rejects_fake_signature_without_mocked_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    activated = tmp_path / "activated.json"
    signature = tmp_path / "activated.sig"
    allowed = tmp_path / "allowed_signers"
    _write_json(activated, {"active": True})
    signature.write_text("not-an-ssh-signature\n", encoding="utf-8")
    allowed.write_text("task8b-release ssh-ed25519 AAAA\n", encoding="utf-8")
    monkeypatch.setattr(fanout, "ALLOWED_SIGNERS_SHA256", fanout._sha256_source(allowed))

    with pytest.raises(
        fanout.RollingFanoutError,
        match="activation signature verification failed",
    ):
        fanout._verify_activation_signature(
            activated_path=activated,
            signature_path=signature,
            allowed_signers_path=allowed,
        )
