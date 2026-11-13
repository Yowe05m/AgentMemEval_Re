"""Durable service exit evidence; never a capability to signal a recorded PID.

The producer must run beside the live service owner. The reader may run after
allocation termination and requires the independently configured authoritative
service and worker stores. JSON receipts alone are insufficient. This is an
explicit library integration: no production E1 lifecycle installs it by default.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentmemeval.path1_v1.operations.closure import (
    ClosureBinding,
    ServiceReleaseEvidence,
    _bound_json,
    _cleanup_evidence,
    _json,
    _path,
    _require,
    _sha,
    _write,
)
from agentmemeval.path1_v1.operations.contracts import content_digest
from agentmemeval.path1_v1.operations.state import CampaignStore


def _worker_terminal(binding: ClosureBinding, store: CampaignStore, path: Path) -> str:
    _require(store.campaign_id == binding.campaign_id, "worker store campaign mismatch")
    terminal = _json(path)
    for key in (
        "campaign_id",
        "intent_id",
        "worker_request_sha256",
        "run_id",
        "service_instance_id",
        "core_manifest_sha256",
        "operations_manifest_sha256",
        "config_sha256",
        "model_manifest_sha256",
        "protocol_sha256",
    ):
        _require(terminal.get(key) == getattr(binding, key), "service worker identity mismatch")
    _require(
        terminal.get("slurm_job_id") == binding.job_id
        and terminal.get("status") == "completed"
        and terminal.get("primary_analysis_admitted") is False,
        "service closure requires a completed worker terminal",
    )
    claims = [row for row in store.read_registry().claims if row.run_id == binding.run_id]
    _require(len(claims) == 1, "worker authoritative terminal claim missing")
    claim = claims[0]
    _require(
        claim.intent_id == binding.intent_id
        and claim.status == "completed"
        and _path(Path(claim.run_directory), directory=True) == path.parent
        and content_digest(claim.terminal_receipt) == content_digest(terminal),
        "service release worker terminal differs from authoritative claim",
    )
    return _sha(path)


class ServiceReleaseJournal:
    """Consume a separate claim before service spawn; seal real owner evidence after exit.

    Call begin BEFORE creating the owner, record_launch with its actual live
    snapshot, then publish AFTER worker terminal commit and exact owner cleanup.
    Crashes leave a started claim; no API reopens it or regenerates missing data.
    """

    @classmethod
    def begin(
        cls,
        service_store: CampaignStore,
        *,
        run_id: str,
        binding: ClosureBinding,
        directory: Path,
        worker_store: CampaignStore,
        worker_terminal_path: Path,
    ) -> ServiceReleaseJournal:
        _require(
            service_store.campaign_id == worker_store.campaign_id == binding.campaign_id,
            "service/worker campaign differs",
        )
        _require(run_id != binding.run_id, "service and worker claims must be distinct")
        directory = _path(directory, directory=True, exists=False)
        terminal_path = _path(worker_terminal_path, exists=False)
        _require(
            directory.parent == terminal_path.parent.parent and directory != terminal_path.parent,
            "service journal must be inside job root",
        )
        _require(not directory.exists(), "service journal output already exists")
        with service_store.lock():
            current = service_store.read_registry()
            claim = service_store.claim_run(
                expected_generation=current.generation,
                run_id=run_id,
                intent_id=binding.intent_id,
                run_directory=directory,
                metadata={"kind": "service_release", "binding": asdict(binding)},
            )
            current = service_store.read_registry()
            service_store.start_claim(
                expected_generation=current.generation, run_id=run_id, token=claim.token
            )
        directory.mkdir(exist_ok=False)
        instance = cls()
        instance.store, instance.worker_store = service_store, worker_store
        instance.binding, instance.directory = binding, directory
        instance.claim, instance.worker_terminal_path = claim, terminal_path
        instance._owner, instance._snapshot = None, None
        return instance

    def record_launch(self, owner: Any, snapshot: Any) -> None:
        _require(
            self._owner is None and owner.verify_snapshot(snapshot) is True,
            "service launch snapshot has no live owner provenance",
        )
        _require(
            snapshot.binding == owner.binding and snapshot.identity == owner.identity,
            "service launch snapshot differs from owner",
        )
        expected = {
            "campaign_id": self.binding.campaign_id,
            "intent_id": self.binding.intent_id,
            "job_id": self.binding.job_id,
            "source_sha256": self.binding.source_sha256,
            "role": "model_service",
            "service_instance_id": self.binding.service_instance_id,
        }
        _require(asdict(snapshot.binding) == expected, "service launch binding differs")
        _require(
            snapshot.state not in {"Z", "X"} and bool(snapshot.cmdline),
            "service launch was not live",
        )
        # ProcessSnapshot's typed tuples are JSON arrays in the published file.
        # Compare that exact serialized representation when checking publication.
        self._launch_document = json.loads(json.dumps(asdict(snapshot), allow_nan=False))
        _write(self.directory / "launch.json", self._launch_document)
        self._launch_sha256 = _sha(self.directory / "launch.json")
        self._owner, self._snapshot = owner, snapshot

    def publish(self, owner: Any, cleanup: Any) -> dict:
        _require(
            owner is self._owner
            and self._snapshot is not None
            and owner.verify_snapshot(self._snapshot) is True
            and owner.verify_cleanup(cleanup) is True,
            "service terminal requires exact launch and cleanup owner objects",
        )
        evidence = _cleanup_evidence(self.binding, asdict(cleanup))
        _require(
            cleanup.identity == self._snapshot.identity,
            "service launch and cleanup kernel identities differ",
        )
        launch = _bound_json(self.directory / "launch.json", self._launch_sha256)
        _require(
            content_digest(launch) == content_digest(self._launch_document),
            "recorded live snapshot changed before service exit",
        )
        worker_sha = _worker_terminal(self.binding, self.worker_store, self.worker_terminal_path)
        _write(self.directory / "cleanup.json", evidence.cleanup)
        receipt = {
            "schema_version": 1,
            "kind": "path1_service_release",
            "binding": asdict(self.binding),
            "service_run_id": self.claim.run_id,
            "worker_terminal_sha256": worker_sha,
            "launch_sha256": self._launch_sha256,
            "cleanup_sha256": _sha(self.directory / "cleanup.json"),
            "primary_analysis_admitted": False,
        }
        _write(self.directory / "service_terminal.json", receipt)
        with self.store.lock():
            current = self.store.read_registry()
            self.store.finish_claim(
                expected_generation=current.generation,
                run_id=self.claim.run_id,
                token=self.claim.token,
                status="completed",
                receipt=receipt,
            )
        return receipt


class PersistedServiceReleaseVerifier:
    """Read trusted producer/store facts after process exit; cannot cancel or attach."""

    def __init__(
        self,
        *,
        service_store: CampaignStore,
        run_id: str,
        directory: Path,
        worker_store: CampaignStore,
        worker_terminal_path: Path,
    ):
        self.store, self.run_id, self.directory = service_store, run_id, directory
        self.worker_store, self.worker_terminal_path = worker_store, worker_terminal_path

    def verify(self, binding: ClosureBinding) -> ServiceReleaseEvidence:
        directory = _path(self.directory, directory=True)
        terminal_path = _path(self.worker_terminal_path)
        _require(
            directory.parent == terminal_path.parent.parent and directory != terminal_path.parent,
            "service evidence outside recovered job root",
        )
        _require(self.store.campaign_id == binding.campaign_id, "service store campaign differs")
        receipt = _json(directory / "service_terminal.json")
        claims = [row for row in self.store.read_registry().claims if row.run_id == self.run_id]
        _require(len(claims) == 1, "service authoritative claim missing")
        claim = claims[0]
        _require(
            claim.intent_id == binding.intent_id
            and claim.status == "completed"
            and _path(Path(claim.run_directory), directory=True) == directory
            and claim.metadata == {"kind": "service_release", "binding": asdict(binding)}
            and content_digest(claim.terminal_receipt) == content_digest(receipt),
            "service terminal differs from authoritative producer claim",
        )
        _require(
            type(receipt.get("schema_version")) is int
            and receipt["schema_version"] == 1
            and receipt.get("kind") == "path1_service_release"
            and receipt.get("service_run_id") == self.run_id
            and receipt.get("binding") == asdict(binding)
            and receipt.get("primary_analysis_admitted") is False,
            "service terminal identity mismatch",
        )
        _require(
            _worker_terminal(binding, self.worker_store, terminal_path)
            == receipt["worker_terminal_sha256"],
            "service worker terminal changed",
        )
        launch = _bound_json(directory / "launch.json", receipt["launch_sha256"])
        cleanup = _bound_json(directory / "cleanup.json", receipt["cleanup_sha256"])
        _require(
            launch.get("binding") == cleanup.get("binding")
            and launch.get("identity") == cleanup.get("identity")
            and type(launch.get("cmdline")) is list
            and bool(launch["cmdline"])
            and type(launch.get("state")) is str
            and launch["state"] not in {"Z", "X"},
            "service launch facts differ from cleanup",
        )
        return _cleanup_evidence(binding, cleanup)
