"""Check/seal an exact historical rejection; never submit or expose a fixture bypass."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from agentmemeval.path1_v1.operations.state import CampaignStore
from agentmemeval.path1_v1.operations.submission_rejection import (
    reconcile_rejection,
    resolve_rejection,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "close", "verify"))
    parser.add_argument("--control-directory", required=True, type=Path)
    parser.add_argument("--store-campaign-id", required=True)
    parser.add_argument("--intent-id")
    for name in ("descriptor", "baseline-manifest", "rejection-evidence"):
        parser.add_argument("--" + name, type=Path)
        parser.add_argument("--" + name + "-sha256")
    parser.add_argument("--original-code-sha")
    args = parser.parse_args(argv)
    store = CampaignStore(args.control_directory, args.store_campaign_id)
    if args.action == "verify":
        if not args.intent_id:
            parser.error("verify requires --intent-id")
        value = resolve_rejection(store, args.intent_id)
    else:
        arguments = dict(
            descriptor_path=args.descriptor,
            descriptor_sha256=args.descriptor_sha256,
            original_code_sha=args.original_code_sha,
            baseline_manifest=args.baseline_manifest,
            baseline_manifest_sha256=args.baseline_manifest_sha256,
            rejection_evidence_path=args.rejection_evidence,
            rejection_evidence_sha256=args.rejection_evidence_sha256,
        )
        if any(item is None for item in arguments.values()):
            parser.error(
                "check/close require original commit and all three pinned evidence paths/SHA"
            )
        value = reconcile_rejection(store, **arguments, check_only=args.action == "check")
    result = (
        value
        if type(value) is dict
        else {
            item.name: getattr(value, item.name)
            for item in fields(value)
            if not item.name.startswith("_")
        }
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
