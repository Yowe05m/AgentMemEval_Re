"""Reverify a sealed Campaign P snapshot before Campaign E starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentmemeval.storage.campaign_seal import audit_campaign_archive_handoff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--seal-readiness", required=True)
    parser.add_argument("--snapshot-receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = audit_campaign_archive_handoff(
        args.campaign_dir,
        seal_readiness_path=args.seal_readiness,
        snapshot_receipt_path=args.snapshot_receipt,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), **audit}, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "verified_campaign_archive_handoff" else 2


if __name__ == "__main__":
    raise SystemExit(main())
