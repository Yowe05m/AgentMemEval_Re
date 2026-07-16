"""Build an immutable campaign resource audit JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentmemeval.evaluation.resource_audit import build_campaign_resource_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_campaign_resource_audit(args.campaign_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
