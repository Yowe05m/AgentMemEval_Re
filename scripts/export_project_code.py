"""Export relevant source and configuration files into one text file.

Run from the project root:
    python scripts/export_project_code.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


# These extensions describe source or configuration maintained with this project.
INCLUDED_SUFFIXES = {".py", ".yaml", ".yml", ".toml"}
EXCLUDED_DIR_NAMES = {
    ".git",
    ".ruff_cache",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "docs",  # Markdown documentation is exported by merge_docs.py instead.
    "outputs",
    "tmp",
    "build",
    "dist",
    "*.egg-info",
}


def is_excluded(path: Path, project_root: Path) -> bool:
    """Return whether *path* is in a generated, secret, or irrelevant area."""
    relative_parts = path.relative_to(project_root).parts
    return any(
        part in EXCLUDED_DIR_NAMES or part.endswith(".egg-info")
        for part in relative_parts[:-1]
    )


def iter_project_files(project_root: Path, output_path: Path) -> list[Path]:
    """Find relevant, text-based project files in deterministic order."""
    files: list[Path] = []
    for path in project_root.rglob("*"):
        if not path.is_file() or is_excluded(path, project_root):
            continue
        if path.resolve() == output_path.resolve():
            continue
        if path.suffix.lower() in INCLUDED_SUFFIXES:
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(project_root).as_posix())


def export_code(project_root: Path, output_path: Path) -> int:
    files = iter_project_files(project_root, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as destination:
        destination.write("# AgentMemEval rebuild: source and configuration export\n")
        destination.write("# Files are separated by their project-relative path.\n\n")
        for path in files:
            relative_path = path.relative_to(project_root).as_posix()
            destination.write(f"{'=' * 88}\n")
            destination.write(f"FILE: {relative_path}\n")
            destination.write(f"{'=' * 88}\n")
            content = path.read_text(encoding="utf-8-sig")
            destination.write(content)
            if not content.endswith("\n"):
                destination.write("\n")
            destination.write("\n")

    print(f"Exported {len(files)} files to {output_path}")
    return len(files)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "code.txt",
        help="Destination path (default: <project root>/code.txt).",
    )
    args = parser.parse_args()
    output_path = args.output if args.output.is_absolute() else project_root / args.output
    export_code(project_root, output_path)


if __name__ == "__main__":
    main()
