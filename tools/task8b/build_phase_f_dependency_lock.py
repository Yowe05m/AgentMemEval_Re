"""Build a deterministic TASK8B Phase F lock from the active virtualenv."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import tomllib

PIN_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s]+)$")


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _run_pip(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pip {' '.join(args)} failed with exit code {result.returncode}")
    return result.stdout.strip()


def build_lock(repository_root: Path) -> str:
    repo = repository_root.resolve()
    pyproject = repo / "pyproject.toml"
    if not pyproject.is_file():
        raise RuntimeError(f"missing pyproject.toml: {pyproject}")
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    project_name = _canonical_name(str(project["name"]))
    freeze_output = _run_pip("freeze", "--all")
    pip_version = _run_pip("--version")
    pins: dict[str, str] = {}
    excluded_editable: list[str] = []
    for raw_line in freeze_output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e "):
            editable_raw = line[3:].strip()
            editable_path = Path(editable_raw).resolve()
            if editable_path != repo:
                raise RuntimeError(f"unexpected editable dependency: {editable_raw}")
            excluded_editable.append(f"{project_name} @ {editable_path}")
            continue
        match = PIN_RE.fullmatch(line)
        if match is None:
            raise RuntimeError(f"non-exact or direct dependency is forbidden: {line}")
        name = _canonical_name(match.group(1))
        if name == project_name:
            raise RuntimeError("local project must be editable and excluded, not pinned as a wheel")
        normalized = f"{name}=={match.group(2)}"
        previous = pins.setdefault(name, normalized)
        if previous != normalized:
            raise RuntimeError(f"conflicting installed versions for {name}")
    if len(excluded_editable) != 1:
        raise RuntimeError("expected exactly one excluded local editable project")
    if not pins:
        raise RuntimeError("pip freeze returned no exact third-party pins")
    command = f'"{Path(sys.executable).resolve()}" -m pip freeze --all'
    lines = [
        "# TASK8B Phase F exact dependency lock",
        "# source: current project .venv installed environment",
        f"# generator_command: {command}",
        f"# interpreter_executable: {Path(sys.executable).resolve()}",
        f"# python_version: {sys.version.split()[0]}",
        f"# pip_version: {pip_version}",
        f"# excluded_local_editable: {excluded_editable[0]}",
        "# policy: every included dependency is an exact name==version pin; "
        "direct URLs are forbidden",
        "",
        *(pins[name] for name in sorted(pins)),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    content = build_lock(Path(args.repository_root))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8", newline="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
