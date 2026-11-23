"""Check, close or verify a service E0 technical claim; no submission or science."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentmemeval.path1_v1.operations.e0_reconcile import resolve_e0_terminal_intent
from agentmemeval.path1_v1.operations.e0_reconcile_cli import resolution_payload
from agentmemeval.path1_v1.operations.service_terminal import (
    ServiceTerminalResolution,
    check_service_terminal,
    reconcile_service_terminal,
)
from agentmemeval.path1_v1.operations.state import CampaignStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "close", "verify"))
    parser.add_argument("--control-directory", required=True, type=Path)
    parser.add_argument("--store-campaign-id", required=True)
    parser.add_argument("--descriptor", type=Path)
    parser.add_argument("--descriptor-sha256")
    parser.add_argument("--intent-id")
    args = parser.parse_args(argv)
    store = CampaignStore(args.control_directory, args.store_campaign_id)
    if args.action == "verify":
        if args.intent_id is None:
            parser.error("verify requires --intent-id")
        resolved = resolve_e0_terminal_intent(store, args.intent_id)
        if type(resolved) is not ServiceTerminalResolution:
            raise ValueError("not a service technical terminal")
        result = resolution_payload(resolved)
    else:
        if args.descriptor is None or args.descriptor_sha256 is None:
            parser.error("check/close require descriptor and its SHA-256")
        operation = check_service_terminal if args.action == "check" else reconcile_service_terminal
        value = operation(
            store, descriptor_path=args.descriptor, descriptor_sha256=args.descriptor_sha256
        )
        result = value if type(value) is dict else resolution_payload(value)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
