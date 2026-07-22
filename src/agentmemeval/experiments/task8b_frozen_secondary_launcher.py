"""Standard-library-only launcher for a frozen TASK8B secondary worker."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

FROZEN_CODE_SHA = "a1d1eb97efb41d52585057ab7c9594dcd19227ae"
FROZEN_RUNNER_SHA256 = "c4b601ff0de2c27a57ee246efcf91d21f502f27c652d20fd6fa7cfd925a17d5e"


class FrozenSecondaryLaunchError(RuntimeError):
    """Frozen checkout or runner binding failed."""


def canonicalize_resolved_config_identity(config: dict[str, object]) -> dict[str, object]:
    """Drop runner-dynamic fields without stringifying integer mapping keys."""

    value = copy.deepcopy(config)
    value.pop("_config_path", None)
    experiment = dict(value.get("experiment", {}))
    for field in ("output_root", "run_id", "initial_memory_snapshots", "admission_audit"):
        experiment.pop(field, None)
    value["experiment"] = experiment
    agent = dict(value.get("agent", {}))
    agent.pop("embedding_cache_path", None)
    value["agent"] = agent
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_checkout(checkout: Path) -> Path:
    checkout = checkout.resolve()
    runner_path = checkout / "src" / "agentmemeval" / "experiments" / "formal_runner.py"
    if (
        not runner_path.is_file()
        or runner_path.is_symlink()
        or _sha256_file(runner_path) != FROZEN_RUNNER_SHA256
    ):
        raise FrozenSecondaryLaunchError("frozen formal runner SHA-256 mismatch")
    try:
        head = subprocess.run(
            ["git", "-c", f"safe.directory={checkout.as_posix()}", "rev-parse", "HEAD"],
            cwd=str(checkout),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={checkout.as_posix()}",
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            cwd=str(checkout),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FrozenSecondaryLaunchError("unable to verify frozen checkout") from exc
    if head != FROZEN_CODE_SHA or dirty:
        raise FrozenSecondaryLaunchError("frozen checkout HEAD or tracked-clean gate failed")
    return runner_path


def run_frozen_secondary(checkout: Path, manifest: Path, receipt_root: Path) -> dict[str, object]:
    runner_path = _verify_checkout(checkout)
    if any(name == "agentmemeval" or name.startswith("agentmemeval.") for name in sys.modules):
        raise FrozenSecondaryLaunchError("agentmemeval imported before checkout verification")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str((checkout.resolve() / "src").resolve()))
    runner = importlib.import_module("agentmemeval.experiments.formal_runner")
    if Path(str(runner.__file__)).resolve() != runner_path:
        raise FrozenSecondaryLaunchError("loaded formal runner escaped frozen checkout")
    if _sha256_file(Path(str(runner.__file__)).resolve()) != FROZEN_RUNNER_SHA256:
        raise FrozenSecondaryLaunchError("loaded formal runner changed after import")
    if not hasattr(runner, "_semantic_config"):
        raise FrozenSecondaryLaunchError("frozen runner canonicalizer hook missing")
    original_cwd = Path.cwd()
    original_canonicalizer = runner._semantic_config
    try:
        runner._semantic_config = canonicalize_resolved_config_identity
        os.chdir(checkout.resolve())
        result = runner.run_worker_manifest(
            manifest.resolve(), receipt_root=receipt_root.resolve(), resume_existing=False
        )
    finally:
        runner._semantic_config = original_canonicalizer
        os.chdir(original_cwd)
    if not isinstance(result, dict):
        raise FrozenSecondaryLaunchError("frozen runner returned a non-object result")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_frozen_secondary(args.checkout, args.manifest, args.receipt_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
