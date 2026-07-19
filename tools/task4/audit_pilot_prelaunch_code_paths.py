"""Create an immutable code-path audit before Campaign E Pilot starts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from agentmemeval.evaluation.pilot import build_pilot_prelaunch_code_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--campaign-p-code-sha", required=True)
    parser.add_argument("--campaign-e-code-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    changed_paths = _git_changed_paths(
        repo,
        args.campaign_p_code_sha,
        args.campaign_e_code_sha,
    )
    audit = build_pilot_prelaunch_code_audit(
        args.campaign_p_code_sha,
        args.campaign_e_code_sha,
        changed_paths,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), **audit}, ensure_ascii=False, indent=2))
    return 0 if audit["status"].startswith("verified_") else 2


def _git_changed_paths(repo: Path, p_sha: str, e_sha: str) -> list[str]:
    process = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo.as_posix()}",
            "diff",
            "--name-only",
            p_sha,
            e_sha,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line for line in process.stdout.splitlines() if line]


if __name__ == "__main__":
    raise SystemExit(main())
