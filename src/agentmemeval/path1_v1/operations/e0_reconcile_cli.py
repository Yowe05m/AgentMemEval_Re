"""Linux-only live E0 reconciliation; no fixture or JSON admission switch."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from agentmemeval.path1_v1.operations.e0_reconcile import (
    E0TerminalResolution,
    check_unstarted_deadline,
    reconcile_unstarted_deadline,
    resolve_e0_terminal_intent,
)
from agentmemeval.path1_v1.operations.state import CampaignStore


def resolution_payload(value: E0TerminalResolution) -> dict:
    return {
        field.name: getattr(value, field.name)
        for field in fields(value)
        if not field.name.startswith("_")
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "close", "verify"))
    parser.add_argument("--control-directory", required=True, type=Path)
    parser.add_argument("--store-campaign-id", required=True)
    parser.add_argument("--descriptor", type=Path)
    parser.add_argument("--descriptor-sha256")
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--baseline-manifest-sha256")
    parser.add_argument("--intent-id")
    args = parser.parse_args(argv)
    store = CampaignStore(args.control_directory, args.store_campaign_id)
    if args.action == "verify":
        if not args.intent_id:
            parser.error("verify requires --intent-id")
        result = resolution_payload(resolve_e0_terminal_intent(store, args.intent_id))
    else:
        if any(
            value is None
            for value in (
                args.descriptor,
                args.descriptor_sha256,
                args.baseline_manifest,
                args.baseline_manifest_sha256,
            )
        ):
            parser.error("check/close require descriptor and pinned baseline paths and SHA-256")
        operation = (
            check_unstarted_deadline if args.action == "check" else reconcile_unstarted_deadline
        )
        value = operation(
            store,
            descriptor_path=args.descriptor,
            descriptor_sha256=args.descriptor_sha256,
            baseline_manifest=args.baseline_manifest,
            baseline_manifest_sha256=args.baseline_manifest_sha256,
        )
        result = value if type(value) is dict else resolution_payload(value)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
