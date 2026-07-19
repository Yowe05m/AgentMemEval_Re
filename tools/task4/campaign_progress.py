"""Print protocol-aware, read-only campaign progress as JSON."""

from __future__ import annotations

import argparse
import json

from agentmemeval.evaluation.campaign_progress import build_campaign_progress


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    args = parser.parse_args()
    progress = build_campaign_progress(args.campaign_dir)
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    return 0 if progress["status"] == "consistent" else 2


if __name__ == "__main__":
    raise SystemExit(main())
