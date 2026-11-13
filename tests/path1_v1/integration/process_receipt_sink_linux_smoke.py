"""Stdlib-only W1H Linux cases: SCRIPT REPOSITORY NEW_OUTPUT_DIRECTORY.

Only new sink cases run here; W1G's twenty process cases are not repeated.
No model, scheduler, network, inherited environment dump or permission recovery.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

SAFE_ENV = {
    "PATH": os.defpath,
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
}
REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=False)
source = REPO / "src/agentmemeval/path1_v1/operations/process_ownership.py"
spec = importlib.util.spec_from_file_location("sink_smoke_ownership", source)
ownership = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ownership
spec.loader.exec_module(ownership)
results = []


def binding(name):
    return ownership.ProcessBinding("sink_smoke", name, None, "a" * 64, "model_service", name)


def persist(path, owner, receipt):
    if owner is None:
        assert isinstance(receipt, ownership.SpawnFailureReceipt)
        assert not receipt.group_cleanup_verified
    else:
        assert owner.verify_cleanup(receipt) is True
    with path.open("x", encoding="utf-8") as stream:
        json.dump(asdict(receipt), stream)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def dead(pid):
    try:
        return ownership._stat(pid)[4] in {"Z", "X"}
    except FileNotFoundError:
        return True


sentinel = ownership.OwnedProcess.spawn(
    [sys.executable, "-c", "import time;time.sleep(30)"],
    binding=binding("sentinel"),
    cwd=OUT,
    env=SAFE_ENV,
)
try:
    for mode in ("creation_term", "spawn_failure"):
        for failing in (False, True):
            name = mode + ("_sink_failure" if failing else "_persisted")
            path = OUT / (name + ".json")
            if failing:
                path.write_bytes(b"existing receipt must not be overwritten")
            real_kernel = ownership._kernel
            created = []
            delivered = []

            def fault_at_kernel(
                pid, *, real_kernel=real_kernel, created=created, mode=mode, **kwargs
            ):
                observed = real_kernel(pid, **kwargs)
                if pid != os.getpid() and not created:
                    created.append(pid)
                    if mode == "spawn_failure":
                        raise PermissionError("controlled startup identity fault")
                    os.kill(os.getpid(), signal.SIGTERM)
                return observed

            def sink(owner, receipt, *, created=created, delivered=delivered, path=path):
                # Linux has already reaped the direct child before any file write.
                assert dead(created[0])
                delivered.append((owner, receipt))
                persist(path, owner, receipt)

            ownership._kernel = fault_at_kernel
            try:
                try:
                    with ownership.OwnedProcess.managed(
                        [sys.executable, "-c", "import time;time.sleep(30)"],
                        binding=binding(name),
                        cwd=OUT,
                        env=SAFE_ENV,
                        term_grace_seconds=0.05,
                        receipt_sink=sink,
                    ):
                        raise AssertionError("fault must occur before managed yield")
                except ownership.ReceiptSinkError as exc:
                    assert failing and not exc.receipt_persisted
                    assert exc.sink_error_type == "FileExistsError"
                    assert exc.receipt is delivered[0][1]
                    assert exc.cleanup_owner is delivered[0][0]
                    assert exc.original_failure_type == (
                        "ProcessInterrupted" if mode == "creation_term" else "SpawnOwnershipError"
                    )
                    assert path.read_bytes() == b"existing receipt must not be overwritten"
                    results.append(
                        {
                            "case": name,
                            "receipt_persisted": False,
                            "receipt": asdict(exc.receipt),
                            "sink_error_type": exc.sink_error_type,
                            "original_failure_type": exc.original_failure_type,
                        }
                    )
                except (ownership.ProcessInterrupted, ownership.SpawnOwnershipError) as exc:
                    assert not failing
                    receipt = delivered[0][1]
                    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(
                        json.dumps(asdict(receipt))
                    )
                    results.append(
                        {
                            "case": name,
                            "receipt_persisted": True,
                            "receipt": asdict(receipt),
                            "propagated_error": type(exc).__name__,
                        }
                    )
                else:
                    raise AssertionError("startup fault did not propagate")
                assert len(delivered) == 1 and created and dead(created[0])
                sentinel.verify_live()
            finally:
                ownership._kernel = real_kernel

    delivered = []
    path = OUT / "finish_receipt.json"

    def finish_sink(owner, receipt):
        delivered.append(receipt)
        persist(path, owner, receipt)

    with ownership.OwnedProcess.managed(
        [sys.executable, "-c", "import time;time.sleep(30)"],
        binding=binding("finish_once"),
        cwd=OUT,
        env=SAFE_ENV,
        term_grace_seconds=0.05,
        receipt_sink=finish_sink,
    ) as owner:
        receipt = owner.finish(deadline_monotonic=time.monotonic() + 0.02, term_grace_seconds=0.05)
        assert not delivered
    assert delivered == [receipt] and owner.close() is receipt
    results.append(
        {
            "case": "finish_before_exit_sink_once",
            "receipt_persisted": True,
            "receipt": asdict(receipt),
        }
    )
    sentinel.verify_live()
finally:
    sentinel_cleanup = sentinel.close(term_grace_seconds=0.05)
    assert sentinel.verify_cleanup(sentinel_cleanup) is True

report = {
    "platform": sys.platform,
    "python": sys.version,
    "case_count": len(results),
    "cases": results,
    "sentinel_cleanup": asdict(sentinel_cleanup),
    "scope": "Local Linux receipt persistence only; no Slurm/GPU or restart kill authority",
}
(OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "passed", "case_count": len(results), "output": str(OUT)}))
