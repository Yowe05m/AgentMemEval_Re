"""Build a post-hoc, development-only retrieval relevance diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from agentmemeval.evaluation.relevance_diagnostics import (
    analyze_relevance_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-key", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    review_key = Path(args.review_key)
    labels_path = Path(args.labels)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    pack = json.loads(review_key.read_text(encoding="utf-8"))
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle, delimiter="\t"))
    result = analyze_relevance_diagnostics(pack, labels)
    result["input_evidence"] = {
        "review_key_path": str(review_key.resolve()),
        "review_key_sha256": _sha256(review_key),
        "labels_path": str(labels_path.resolve()),
        "labels_sha256": _sha256(labels_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), **result}, ensure_ascii=False, indent=2))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
