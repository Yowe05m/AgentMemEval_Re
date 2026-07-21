"""CLI for the TASK4 V8 decision-point true-service smoke gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentmemeval.evaluation.decision_point_gate import (
    build_decision_point_smoke_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--expected-train-hands", required=True, type=int)
    parser.add_argument("--expected-test-hands", required=True, type=int)
    args = parser.parse_args()

    audit = build_decision_point_smoke_gate(
        args.run_dir,
        expected_code_sha=args.expected_code_sha,
        expected_train_hands=args.expected_train_hands,
        expected_test_hands=args.expected_test_hands,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), **audit}, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "ready_to_start_v8_calibration_pilot" else 2


if __name__ == "__main__":
    raise SystemExit(main())
