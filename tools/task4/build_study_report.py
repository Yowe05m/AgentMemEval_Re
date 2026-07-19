"""Build a fail-closed Task4 Chinese study report bundle."""

from __future__ import annotations

import argparse
import json

from agentmemeval.evaluation.study_reporting import build_task4_study_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = build_task4_study_report(args.study_spec, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
