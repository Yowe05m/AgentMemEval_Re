"""Create an immutable Pilot-only runtime-equivalence audit from P/E aggregates."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from agentmemeval.evaluation.pilot import (
    build_pilot_runtime_equivalence_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--campaign-p", required=True)
    parser.add_argument("--campaign-e", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    campaign_p = _read_json(Path(args.campaign_p).resolve())
    campaign_e = _read_json(Path(args.campaign_e).resolve())
    p_sha = _runtime_commit(campaign_p)
    e_sha = _runtime_commit(campaign_e)
    changed_paths = _git_changed_paths(repo, p_sha, e_sha)
    audit = build_pilot_runtime_equivalence_audit(
        campaign_p,
        campaign_e,
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
        ["git", "-c", f"safe.directory={repo.as_posix()}", "diff", "--name-only", p_sha, e_sha],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line for line in process.stdout.splitlines() if line]


def _runtime_commit(aggregate: dict[str, object]) -> str:
    identity = dict(aggregate.get("runtime_homogeneity", {})).get("identity", {})
    code = dict(identity).get("code", {}) if isinstance(identity, dict) else {}
    if isinstance(code, list):
        code = {str(key): value for key, value in code}
    commit = str(dict(code).get("commit", ""))
    if not commit:
        raise ValueError("aggregate runtime identity lacks code commit")
    return commit


def _read_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
