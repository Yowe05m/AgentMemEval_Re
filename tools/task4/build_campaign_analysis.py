"""Build a non-overwriting Task4 campaign analysis bundle."""

from __future__ import annotations

import argparse
import json

from agentmemeval.evaluation.campaign_reporting import build_campaign_analysis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = build_campaign_analysis(args.aggregate, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
