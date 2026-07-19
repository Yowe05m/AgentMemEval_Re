"""Write an immutable campaign seal-readiness audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentmemeval.storage.campaign_seal import audit_campaign_seal_readiness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--minimum-quiet-seconds", type=int, default=120)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_campaign_seal_readiness(
        args.campaign_dir,
        minimum_quiet_seconds=args.minimum_quiet_seconds,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), **result}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ready_to_seal" else 2


if __name__ == "__main__":
    raise SystemExit(main())
