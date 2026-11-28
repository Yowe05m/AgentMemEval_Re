"""One content-pinned preload, checked before importing the application.

This module is deliberately stdlib-only and can run by its absolute file path.
It is an environment repair, never a GPU/backend/scientific admission producer.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import runpy
import stat
import sys
from pathlib import Path


def validate_binding(value: dict) -> dict:
    if (type(value) is not dict or set(value) != {"path", "sha256"}
            or type(value["path"]) is not str or type(value["sha256"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
            or not Path(value["path"]).is_absolute()
            or any(c.isspace() or c in "\x00:" for c in value["path"]
                   if not (os.name == "nt" and c == ":"))):
        raise ValueError("native runtime requires one absolute path and SHA-256")
    return dict(value)


def verify_file(binding: dict) -> dict:
    binding = validate_binding(binding)
    path = Path(binding["path"])
    if path.resolve(strict=True) != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("native runtime path must be canonical without symlinks")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("native runtime is not a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("native runtime changed before reading")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after_open = os.fstat(stream.fileno())
    after = path.stat()
    def signature(s):
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if (signature(opened) != signature(after_open) or signature(before) != signature(after)
            or digest.hexdigest() != binding["sha256"]):
        raise ValueError("native runtime contents changed or hash differs")
    return {**binding, "size": before.st_size}


def environment(binding: dict, base: dict[str, str]) -> dict[str, str]:
    verify_file(binding)
    result = dict(base)
    result.pop("LD_LIBRARY_PATH", None)
    result.pop("LD_AUDIT", None)
    result["LD_PRELOAD"] = binding["path"]
    return result


def verify_loaded(binding: dict, *, pid: int | None = None) -> dict:
    """Check a live Linux mapping; a PID is observation, never ownership authority."""
    record = verify_file(binding)
    if not sys.platform.startswith("linux"):
        raise ValueError("native runtime loaded identity requires Linux /proc")
    if pid is None:
        pid = os.getpid()
        if os.environ.get("LD_PRELOAD") != binding["path"]:
            raise ValueError("running process lacks the exact preload setting")
    if type(pid) is not int or pid <= 0:
        raise ValueError("invalid native runtime process")
    current = Path(binding["path"]).stat()
    matching = []
    for line in Path(f"/proc/{pid}/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        mapped = fields[5]
        if mapped == binding["path"]:
            major, minor = (int(x, 16) for x in fields[3].split(":"))
            if os.makedev(major, minor) != current.st_dev or int(fields[4]) != current.st_ino:
                raise ValueError("native runtime mapped inode differs")
            matching.append(fields[1])
        elif "libstdc++.so" in mapped and "libstdc++.so" in binding["path"]:
            raise ValueError("another C++ runtime is mapped")
    if not matching or not any("x" in perms for perms in matching):
        raise ValueError("native runtime was not actually loaded")
    verify_file(binding)
    return {**record, "pid": pid, "mapped_device": current.st_dev,
            "mapped_inode": current.st_ino, "executable_mapping": True,
            "runtime_admission": False}


def bootstrap_argv(python: Path, repository: Path, binding: dict, *,
                   module: str | None = None, arguments: list[str] | None = None) -> list[str]:
    binding = validate_binding(binding)
    script = repository / "src/agentmemeval/path1_v1/operations/native_runtime.py"
    result = [str(python), "-I", "-B", str(script), "--path", binding["path"],
              "--sha256", binding["sha256"]]
    result += ["--module", module] if module is not None else ["--stdin"]
    return result + ["--", *(arguments or [])]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--sha256", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--module")
    modes.add_argument("--stdin", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    binding = {"path": args.path, "sha256": args.sha256}
    env = environment(binding, os.environ)
    if not sys.platform.startswith("linux"):
        raise ValueError("native preload bootstrap requires Linux")
    # No stdin has been consumed. The same venv entry and wrapper are preserved.
    if any(os.environ.get(key) != env.get(key)
           for key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT")):
        os.execve(sys.executable, [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                                  *(sys.argv[1:] if argv is None else argv)], env)
        raise RuntimeError("execve returned")
    verify_loaded(binding)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    if args.stdin:
        sys.argv = ["-", *arguments]
        exec(compile(sys.stdin.buffer.read(), "<stdin>", "exec"),
             {"__name__": "__main__", "__file__": "<stdin>"})
    else:
        sys.argv = [args.module, *arguments]
        runpy.run_module(args.module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
