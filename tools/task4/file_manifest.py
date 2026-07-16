"""Build or verify a Task4 file-level archive manifest."""

from __future__ import annotations

import argparse
import json

from agentmemeval.storage.archive import build_file_manifest, verify_file_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--root", required=True)
    build.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--root", required=True)
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--allow-extra-files", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        result = build_file_manifest(args.root, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    result = verify_file_manifest(
        args.root,
        args.manifest,
        reject_extra_files=not args.allow_extra_files,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
