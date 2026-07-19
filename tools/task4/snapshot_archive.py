"""Build or checksum-verify append-only Task4 snapshot archives."""

from __future__ import annotations

import argparse
import json

from agentmemeval.storage.snapshot_archive import (
    build_snapshot_archive,
    extract_snapshot_archive,
    verify_archive_checksum,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--root", required=True)
    build.add_argument("--archive", required=True)
    build.add_argument("--manifest", required=True)
    build.add_argument("--checksum", required=True)
    build.add_argument("--receipt", required=True)
    verify = sub.add_parser("verify-checksum")
    verify.add_argument("--archive", required=True)
    verify.add_argument("--checksum", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--archive", required=True)
    extract.add_argument("--checksum", required=True)
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--output-dir", required=True)
    extract.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_snapshot_archive(
            args.root,
            args.archive,
            args.manifest,
            args.checksum,
            args.receipt,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "verify-checksum":
        result = verify_archive_checksum(args.archive, args.checksum)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["verified"] else 2
    result = extract_snapshot_archive(
        args.archive,
        args.checksum,
        args.manifest,
        args.output_dir,
        args.receipt,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
