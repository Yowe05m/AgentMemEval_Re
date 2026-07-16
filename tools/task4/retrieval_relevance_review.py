"""Create a blind retrieval review pack or audit completed human labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from agentmemeval.evaluation.relevance_review import (
    audit_relevance_labels,
    build_relevance_review_pack,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--campaign-dir", action="append", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--sample-size", type=int, default=240)
    build.add_argument("--sample-seed", type=int, default=20260717)
    audit = sub.add_parser("audit")
    audit.add_argument("--review-key", required=True)
    audit.add_argument("--labels", required=True)
    audit.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "build":
        pack = build_relevance_review_pack(
            args.campaign_dir, sample_size=args.sample_size, sample_seed=args.sample_seed
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        key_path = output_dir / "review_key.json"
        blind_path = output_dir / "blind_review.jsonl"
        labels_path = output_dir / "human_labels.tsv"
        key_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with blind_path.open("x", encoding="utf-8") as handle:
            for row in pack["blind_rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with labels_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "row_id",
                    "label",
                    "reviewer_id",
                    "reviewer_type",
                    "comment",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            for row in pack["blind_rows"]:
                writer.writerow(
                    {
                        "row_id": row["row_id"],
                        "label": "",
                        "reviewer_id": "",
                        "reviewer_type": "human",
                        "comment": "",
                    }
                )
        summary = {
            "status": pack["status"],
            "output_dir": str(output_dir),
            "sampled_row_count": pack["sampled_row_count"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    with Path(args.labels).open("r", encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle, delimiter="\t"))
    pack = json.loads(Path(args.review_key).read_text(encoding="utf-8"))
    result = audit_relevance_labels(pack, labels)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), **result}, ensure_ascii=False, indent=2))
    return 0 if result["retrieval_threshold_status"] == "frozen" else 2


if __name__ == "__main__":
    raise SystemExit(main())
