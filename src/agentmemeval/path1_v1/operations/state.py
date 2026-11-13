"""Transactional campaign state and non-reusable run claims.

The SQLite database is the sole authority. JSON summaries are disposable views,
never input to a state transition. Every mutation requires a process-owned OS
lock and an expected generation. Locks have no expiry and are released by the OS
on process death; their files are retained. Use a local durable filesystem whose
SQLite and locking semantics have been validated, not an untested shared mount.

A worker must consume a reserved claim with ``start_claim`` before creating its
output directory or doing work. A crash after consumption leaves a started claim
which cannot be consumed again. This provides at-most-once admission, not an
exactly-once guarantee for external side effects such as sbatch or model calls.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import socket
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = 1


class CampaignStoreError(RuntimeError):
    """An operation failed closed without granting execution permission."""


class LockUnavailable(CampaignStoreError):
    pass


class LockRequired(CampaignStoreError):
    pass


class GenerationConflict(CampaignStoreError):
    pass


class ClaimConflict(CampaignStoreError):
    pass


class StoreIntegrityError(CampaignStoreError):
    pass


@dataclass(frozen=True)
class RunClaim:
    run_id: str
    intent_id: str
    run_directory: str
    token: str
    status: str
    created_generation: int
    updated_generation: int
    metadata: dict[str, Any]
    terminal_receipt: dict[str, Any] | None


@dataclass(frozen=True)
class RegistrySnapshot:
    campaign_id: str
    generation: int
    registry: dict[str, Any]
    state: dict[str, Any]
    payload_sha256: str
    chain_sha256: str
    claims: tuple[RunClaim, ...]


@dataclass(frozen=True)
class _Lease:
    pid: int
    thread_id: int
    token: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_json(value: Any, ancestors: set[int] | None = None) -> None:
    """Reject lossy JSON coercions, non-finite numbers, and circular containers."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not valid campaign state")
        return
    if type(value) not in (dict, list):
        raise ValueError("campaign state must contain only JSON types")
    ancestors = set() if ancestors is None else ancestors
    if id(value) in ancestors:
        raise ValueError("circular campaign state")
    ancestors.add(id(value))
    try:
        items = value.values() if isinstance(value, dict) else value
        if isinstance(value, dict) and any(type(key) is not str for key in value):
            raise ValueError("campaign state object keys must be strings")
        for item in items:
            _validate_json(item, ancestors)
    finally:
        ancestors.remove(id(value))


def _canonical(value: Any) -> str:
    _validate_json(value)
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _object_json(value: Any) -> str:
    if type(value) is not dict:
        raise ValueError("registry, state, and receipt payloads must be JSON objects")
    return _canonical(value)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identifier(value: str, name: str) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or len(value) > 256
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"invalid {name}")
    return value


def _lock_fd(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_fd(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


class CampaignStore:
    """Own one campaign database; JSON snapshots cannot authorize mutations.

    ``read_registry`` returns a consistent status snapshot without requiring the
    writer lock. Dispatchers must call it again *inside* ``lock`` before deciding
    to submit. All writes and claim consumption enforce the lock and CAS anyway.
    A stale caller receives ``GenerationConflict`` instead of overwriting state.
    """

    def __init__(self, directory: str | Path, campaign_id: str) -> None:
        self.directory = Path(directory).absolute().resolve()
        self.campaign_id = _identifier(campaign_id, "campaign_id")
        self.database_path = self.directory / "campaign.sqlite3"
        self.lock_path = self.directory / "campaign.writer.lock"
        self._lease: _Lease | None = None

    @contextmanager
    def lock(self, *, timeout: float = 0.0) -> Iterator[CampaignStore]:
        """Acquire an OS lifetime lock; never infer availability from file age."""
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("lock timeout must be a finite non-negative number")
        if self._lease is not None:
            raise LockUnavailable("this store already has a lock context")
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.lock_path.is_symlink():
            raise StoreIntegrityError("lock path must not be a symlink")
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
        acquired = False
        owner_pid = os.getpid()
        lease: _Lease | None = None
        deadline = time.monotonic() + timeout
        try:
            while True:
                try:
                    _lock_fd(fd)
                    acquired = True
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if time.monotonic() >= deadline:
                        raise LockUnavailable("campaign writer is already locked") from exc
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            lease = _Lease(owner_pid, threading.get_ident(), uuid4().hex)
            self._lease = lease
            owner = _canonical(
                {
                    "pid": lease.pid,
                    "thread_id": lease.thread_id,
                    "token": lease.token,
                    "host": socket.gethostname(),
                    "acquired_at_utc": _utc_now(),
                }
            ).encode("utf-8")
            # Byte zero is the permanent lock region, including on Windows.
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, b"\0" + owner)
            os.ftruncate(fd, len(owner) + 1)
            os.fsync(fd)
            yield self
        finally:
            if lease is not None and self._lease is lease:
                self._lease = None
            try:
                if acquired and os.getpid() == owner_pid:
                    _unlock_fd(fd)
            finally:
                os.close(fd)

    def _require_lock(self) -> _Lease:
        lease = self._lease
        if lease is None or lease.pid != os.getpid() or lease.thread_id != threading.get_ident():
            raise LockRequired("mutation requires this process and thread's writer lock")
        return lease

    def _connect(self, *, create: bool = False) -> sqlite3.Connection:
        if self.database_path.is_symlink():
            raise StoreIntegrityError("campaign database must not be a symlink")
        mode = "rwc" if create else "rw"
        try:
            connection = sqlite3.connect(
                self.database_path.as_uri() + f"?mode={mode}",
                uri=True,
                timeout=0.0,
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            raise CampaignStoreError("campaign database cannot be opened") from exc
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
        except BaseException:
            connection.close()
            raise
        return connection

    def _commit_transaction(self, connection: sqlite3.Connection) -> None:
        connection.commit()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._require_lock()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            self._commit_transaction(connection)
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(
        self,
        *,
        registry: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
    ) -> RegistrySnapshot:
        """Create a new database; existing or partial stores are never overwritten."""
        self._require_lock()
        registry_json = _object_json({} if registry is None else registry)
        state_json = _object_json({} if state is None else state)
        try:
            with self.database_path.open("xb"):
                pass
        except FileExistsError as exc:
            raise CampaignStoreError(
                "campaign database already exists; refusing initialization"
            ) from exc
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=PERSIST")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""CREATE TABLE campaign (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                schema_version INTEGER NOT NULL, campaign_id TEXT NOT NULL,
                generation INTEGER NOT NULL CHECK(generation>=0),
                registry_json TEXT NOT NULL, state_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL, chain_sha256 TEXT NOT NULL)""")
            connection.execute("""CREATE TABLE transitions (
                generation INTEGER PRIMARY KEY, event_json TEXT NOT NULL,
                previous_chain_sha256 TEXT NOT NULL, chain_sha256 TEXT NOT NULL)""")
            connection.execute("""CREATE TABLE claims (
                run_id TEXT PRIMARY KEY, intent_id TEXT NOT NULL UNIQUE,
                path_key TEXT NOT NULL UNIQUE, claim_json TEXT NOT NULL,
                claim_sha256 TEXT NOT NULL)""")
            payload_hash = self._payload_hash(0, registry_json, state_json)
            event = self._event("initialize", "initialize campaign", payload_hash, None)
            chain = _digest("" + event)
            connection.execute(
                "INSERT INTO campaign VALUES(1,?,?,?,?,?,?,?)",
                (
                    SCHEMA_VERSION,
                    self.campaign_id,
                    0,
                    registry_json,
                    state_json,
                    payload_hash,
                    chain,
                ),
            )
            connection.execute("INSERT INTO transitions VALUES(?,?,?,?)", (0, event, "", chain))
            self._commit_transaction(connection)
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.read_registry()

    def _payload_hash(self, generation: int, registry_json: str, state_json: str) -> str:
        return _digest(
            _canonical(
                {
                    "campaign_id": self.campaign_id,
                    "generation": generation,
                    "registry": json.loads(registry_json),
                    "state": json.loads(state_json),
                }
            )
        )

    def _read_row(self, connection: sqlite3.Connection) -> sqlite3.Row:
        try:
            row = connection.execute("SELECT * FROM campaign WHERE singleton=1").fetchone()
        except sqlite3.Error as exc:
            raise StoreIntegrityError("campaign schema is unavailable") from exc
        if (
            row is None
            or row["schema_version"] != SCHEMA_VERSION
            or row["campaign_id"] != self.campaign_id
        ):
            raise StoreIntegrityError("campaign identity or schema mismatch")
        try:
            actual = self._payload_hash(row["generation"], row["registry_json"], row["state_json"])
        except (ValueError, TypeError) as exc:
            raise StoreIntegrityError("invalid stored campaign JSON") from exc
        if actual != row["payload_sha256"]:
            raise StoreIntegrityError("campaign payload digest mismatch")
        latest = connection.execute(
            "SELECT * FROM transitions WHERE generation=?",
            (row["generation"],),
        ).fetchone()
        if (
            latest is None
            or latest["chain_sha256"] != row["chain_sha256"]
            or _digest(latest["previous_chain_sha256"] + latest["event_json"])
            != latest["chain_sha256"]
        ):
            raise StoreIntegrityError("campaign transition digest mismatch")
        return row

    def _check_generation(self, connection: sqlite3.Connection, expected: int) -> sqlite3.Row:
        if type(expected) is not int or expected < 0:
            raise ValueError("expected_generation must be a non-negative integer")
        row = self._read_row(connection)
        if row["generation"] != expected:
            raise GenerationConflict(
                f"expected generation {expected}, observed {row['generation']}"
            )
        return row

    def _event(self, action: str, reason: str, payload_hash: str, claim: dict | None) -> str:
        lease = self._require_lock()
        return _canonical(
            {
                "action": action,
                "reason": _identifier(reason, "reason"),
                "owner_pid": lease.pid,
                "owner_token": lease.token,
                "at_utc": _utc_now(),
                "payload_sha256": payload_hash,
                "claim": claim,
            }
        )

    def _advance(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        action: str,
        reason: str,
        registry_json: str | None = None,
        state_json: str | None = None,
        claim: dict[str, Any] | None = None,
    ) -> int:
        generation = row["generation"] + 1
        registry_json = row["registry_json"] if registry_json is None else registry_json
        state_json = row["state_json"] if state_json is None else state_json
        payload_hash = self._payload_hash(generation, registry_json, state_json)
        event = self._event(action, reason, payload_hash, claim)
        chain = _digest(row["chain_sha256"] + event)
        changed = connection.execute(
            """UPDATE campaign SET generation=?,registry_json=?,state_json=?,
                payload_sha256=?,chain_sha256=? WHERE singleton=1 AND generation=?""",
            (generation, registry_json, state_json, payload_hash, chain, row["generation"]),
        ).rowcount
        if changed != 1:
            raise GenerationConflict("campaign generation changed during transaction")
        connection.execute(
            "INSERT INTO transitions VALUES(?,?,?,?)",
            (
                generation,
                event,
                row["chain_sha256"],
                chain,
            ),
        )
        return generation

    @staticmethod
    def _claim(row: sqlite3.Row) -> RunClaim:
        if _digest(row["claim_json"]) != row["claim_sha256"]:
            raise StoreIntegrityError("run claim digest mismatch")
        try:
            value = json.loads(row["claim_json"])
            if value["run_id"] != row["run_id"] or value["intent_id"] != row["intent_id"]:
                raise StoreIntegrityError("run claim identity mismatch")
            return RunClaim(**value)
        except (TypeError, ValueError, KeyError) as exc:
            raise StoreIntegrityError("invalid run claim record") from exc

    def read_registry(self) -> RegistrySnapshot:
        """Read one consistent SQLite snapshot; no JSON summary is consulted."""
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = self._read_row(connection)
            claims = tuple(
                self._claim(item)
                for item in connection.execute(
                    "SELECT * FROM claims ORDER BY run_id",
                )
            )
            return RegistrySnapshot(
                self.campaign_id,
                row["generation"],
                json.loads(row["registry_json"]),
                json.loads(row["state_json"]),
                row["payload_sha256"],
                row["chain_sha256"],
                claims,
            )
        finally:
            connection.rollback()
            connection.close()

    def commit_transition(
        self,
        *,
        expected_generation: int,
        registry: dict[str, Any],
        state: dict[str, Any],
        reason: str,
    ) -> RegistrySnapshot:
        """Atomically replace registry and state with a compare-and-swap guard."""
        registry_json, state_json = _object_json(registry), _object_json(state)
        with self._transaction() as connection:
            row = self._check_generation(connection, expected_generation)
            self._advance(
                connection,
                row,
                action="state_transition",
                reason=reason,
                registry_json=registry_json,
                state_json=state_json,
            )
        return self.read_registry()

    def claim_run(
        self,
        *,
        expected_generation: int,
        run_id: str,
        intent_id: str,
        run_directory: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> RunClaim:
        """Reserve an ID, intent and absent output path permanently, without creating it."""
        run_id, intent_id = _identifier(run_id, "run_id"), _identifier(intent_id, "intent_id")
        path = Path(run_directory)
        if not path.is_absolute():
            raise ValueError("run_directory must be absolute")
        path = path.resolve()
        if (
            path == self.directory
            or self.directory in path.parents
            or path in self.directory.parents
        ):
            raise ValueError("run output and campaign control directories must be separate")
        metadata = json.loads(_object_json({} if metadata is None else metadata))
        with self._transaction() as connection:
            row = self._check_generation(connection, expected_generation)
            if path.exists():
                raise ClaimConflict("run directory already exists, including an empty directory")
            generation = row["generation"] + 1
            claim = {
                "run_id": run_id,
                "intent_id": intent_id,
                "run_directory": str(path),
                "token": uuid4().hex,
                "status": "reserved",
                "created_generation": generation,
                "updated_generation": generation,
                "metadata": metadata,
                "terminal_receipt": None,
            }
            serialized = _canonical(claim)
            try:
                connection.execute(
                    "INSERT INTO claims VALUES(?,?,?,?,?)",
                    (
                        run_id,
                        intent_id,
                        os.path.normcase(str(path)),
                        serialized,
                        _digest(serialized),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ClaimConflict("run ID, intent or output path was already claimed") from exc
            self._advance(
                connection, row, action="claim_reserved", reason="reserve run", claim=claim
            )
        return RunClaim(**claim)

    def _change_claim(
        self,
        *,
        expected_generation: int,
        run_id: str,
        token: str,
        status: str,
        terminal_receipt: dict[str, Any] | None,
    ) -> RunClaim:
        _identifier(run_id, "run_id")
        _identifier(token, "claim token")
        with self._transaction() as connection:
            row = self._check_generation(connection, expected_generation)
            stored = connection.execute("SELECT * FROM claims WHERE run_id=?", (run_id,)).fetchone()
            if stored is None:
                raise ClaimConflict("run has no reservation")
            current = self._claim(stored)
            if current.token != token:
                raise ClaimConflict("claim token mismatch")
            if status == "started":
                if current.status != "reserved" or Path(current.run_directory).exists():
                    raise ClaimConflict("claim was consumed or its output path already exists")
            elif current.status not in ("reserved", "started"):
                raise ClaimConflict("terminal claim cannot be changed or rerun")
            elif status == "completed" and current.status != "started":
                raise ClaimConflict("an unstarted run cannot be completed")
            claim = json.loads(stored["claim_json"])
            claim.update(
                status=status,
                updated_generation=row["generation"] + 1,
                terminal_receipt=terminal_receipt,
            )
            serialized = _canonical(claim)
            connection.execute(
                "UPDATE claims SET claim_json=?,claim_sha256=? WHERE run_id=?",
                (
                    serialized,
                    _digest(serialized),
                    run_id,
                ),
            )
            self._advance(
                connection,
                row,
                action="claim_" + status,
                reason="change run claim status",
                claim=claim,
            )
        return RunClaim(**claim)

    def start_claim(self, *, expected_generation: int, run_id: str, token: str) -> RunClaim:
        """Consume a reservation once, before any worker output or external work."""
        return self._change_claim(
            expected_generation=expected_generation,
            run_id=run_id,
            token=token,
            status="started",
            terminal_receipt=None,
        )

    def finish_claim(
        self,
        *,
        expected_generation: int,
        run_id: str,
        token: str,
        status: str,
        receipt: dict[str, Any],
    ) -> RunClaim:
        """Record worker completion/failure, not scientific acceptance; never release a claim."""
        if status not in ("completed", "failed"):
            raise ValueError("terminal status must be completed or failed")
        receipt = json.loads(_object_json(receipt))
        if not receipt:
            raise ValueError("a terminal receipt is required")
        return self._change_claim(
            expected_generation=expected_generation,
            run_id=run_id,
            token=token,
            status=status,
            terminal_receipt=receipt,
        )

    def export_state_summary(self, destination: str | Path) -> dict[str, Any]:
        """Write a clearly non-authoritative view; this store never imports summaries."""
        self._require_lock()
        target = Path(destination).absolute()
        protected_paths = {
            self.database_path,
            self.lock_path,
            *(
                self.directory / (self.database_path.name + suffix)
                for suffix in ("-journal", "-wal", "-shm")
            ),
        }
        if target.resolve() in protected_paths:
            raise ValueError("summary cannot replace store authority or lock")
        if target.is_symlink():
            raise ValueError("summary must not replace a symlink")
        snapshot = self.read_registry()
        counts: dict[str, int] = {}
        for claim in snapshot.claims:
            counts[claim.status] = counts.get(claim.status, 0) + 1
        view = {
            "schema_version": "campaign-derived-summary-v1",
            "authoritative": False,
            "submission_authority": False,
            "campaign_id": self.campaign_id,
            "generation": snapshot.generation,
            "payload_sha256": snapshot.payload_sha256,
            "chain_sha256": snapshot.chain_sha256,
            "state": snapshot.state,
            "claim_status_counts": counts,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + "." + uuid4().hex + ".partial")
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical(view) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        return view
