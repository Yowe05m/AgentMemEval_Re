"""Build server_run_map.csv and a formal-main exclusion list."""

from __future__ import annotations

import argparse
import json

from agentmemeval.storage.run_map import build_run_map


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--exclusion-json", required=True)
    args = parser.parse_args()
    result = build_run_map(
        args.campaign_dir, args.output_csv, args.exclusion_json
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
