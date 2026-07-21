"""Rebuild a Campaign aggregate after proving isolated run-local cache paths."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from agentmemeval.evaluation.aggregation import validate_runtime_homogeneity
from agentmemeval.experiments.campaign import build_campaign_aggregate_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--original-campaign-dir", required=True)
    parser.add_argument("--observed-aggregate", required=True)
    parser.add_argument("--corrected-aggregate-output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--expected-run-count", type=int, required=True)
    args = parser.parse_args()

    audit, corrected = build_correction(
        Path(args.campaign_dir),
        original_campaign_dir=args.original_campaign_dir,
        observed_aggregate_path=Path(args.observed_aggregate),
        expected_code_sha=args.expected_code_sha,
        expected_run_count=args.expected_run_count,
    )
    audit_output = Path(args.audit_output).resolve()
    corrected_output = Path(args.corrected_aggregate_output).resolve()
    audit["corrected_aggregate_output"] = str(corrected_output)
    if audit["status"] == "verified_isolated_cache_path_correction":
        with corrected_output.open("x", encoding="utf-8") as handle:
            json.dump(corrected, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        audit["corrected_aggregate_file_sha256"] = _sha256(corrected_output)
    else:
        audit["corrected_aggregate_file_sha256"] = None
    with audit_output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"audit_output": str(audit_output), **audit}, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "verified_isolated_cache_path_correction" else 2


def build_correction(
    campaign_dir: Path,
    *,
    original_campaign_dir: str,
    observed_aggregate_path: Path,
    expected_code_sha: str,
    expected_run_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign_dir = campaign_dir.resolve()
    original_root = PurePosixPath(str(original_campaign_dir).replace("\\", "/"))
    observed_aggregate_path = observed_aggregate_path.resolve()
    observed = _read_json(observed_aggregate_path)
    manifest = _read_json(campaign_dir / "campaign_manifest.json")
    blockers: list[str] = []
    completed = _completed_local_rows(campaign_dir, blockers)
    if len(completed) != expected_run_count:
        blockers.append(f"complete run mismatch: {len(completed)}/{expected_run_count}")

    manifests = []
    cache_namespaces = {}
    for row in completed:
        run_id = str(row["run_id"])
        run_dir = Path(row["run_dir"])
        run_manifest = _read_json(run_dir / "manifest.json")
        manifests.append(run_manifest)
        code = run_manifest.get("metadata", {}).get("code", {})
        if code.get("commit") != expected_code_sha or code.get("dirty") is not False:
            blockers.append(f"{run_id} code identity mismatch: {code}")
        observed_namespace = str(
            run_manifest.get("metadata", {})
            .get("embedding", {})
            .get("cache_namespace_template", "")
        )
        expected_namespace = str(
            original_root
            / "runs"
            / run_id
            / "embedding_cache"
            / "{agent_id}.json"
        )
        cache_namespaces[run_id] = observed_namespace
        if observed_namespace != expected_namespace:
            blockers.append(
                f"{run_id} cache namespace is not canonical run-local: "
                f"{observed_namespace}/{expected_namespace}"
            )

    observed_homogeneity = observed.get("runtime_homogeneity", {})
    observed_mismatch_fields = sorted(
        dict(observed_homogeneity).get("mismatches", {})
    )
    if observed_homogeneity.get("homogeneous") is not False:
        blockers.append("observed aggregate is not blocked by runtime heterogeneity")
    if observed_mismatch_fields != ["embedding"]:
        blockers.append(
            "observed runtime mismatch is not isolated to embedding: "
            f"{observed_mismatch_fields}"
        )

    normalized_homogeneity = validate_runtime_homogeneity(manifests)
    if normalized_homogeneity.get("formal_aggregation_allowed") is not True:
        blockers.append(
            "runtime remains heterogeneous after canonical cache exclusion: "
            f"{normalized_homogeneity.get('mismatches')}"
        )
    corrected = build_campaign_aggregate_payload(
        manifest["campaign"],
        manifest["base_config"],
        completed,
    )
    corrected_homogeneity = corrected.get("runtime_homogeneity", {})
    if corrected_homogeneity != normalized_homogeneity:
        blockers.append("corrected aggregate runtime identity differs from leaf audit")

    observed_with_corrected_identity = copy.deepcopy(observed)
    observed_with_corrected_identity["runtime_homogeneity"] = corrected_homogeneity
    payload_unchanged = observed_with_corrected_identity == corrected
    if not payload_unchanged:
        blockers.append(
            "aggregate payload changed outside runtime_homogeneity correction"
        )

    return (
        {
            "schema_version": "task4_run_local_cache_aggregate_correction_v1",
            "classification": "offline_evidence_correction_not_runtime_mutation",
            "campaign_dir": str(campaign_dir),
            "original_campaign_dir": str(original_root),
            "observed_aggregate_path": str(observed_aggregate_path),
            "observed_aggregate_sha256": _sha256(observed_aggregate_path),
            "expected_code_sha": expected_code_sha,
            "completed_run_ids": sorted(str(row["run_id"]) for row in completed),
            "cache_namespaces": cache_namespaces,
            "observed_runtime_mismatch_fields": observed_mismatch_fields,
            "normalized_runtime_homogeneity": normalized_homogeneity,
            "aggregate_payload_unchanged_outside_runtime_homogeneity": payload_unchanged,
            "observed_aggregate_metrics_sha256": _json_sha256(
                observed.get("aggregate_metrics")
            ),
            "corrected_aggregate_metrics_sha256": _json_sha256(
                corrected.get("aggregate_metrics")
            ),
            "corrected_aggregate_content_sha256": _json_sha256(corrected),
            "blockers": blockers,
            "status": (
                "verified_isolated_cache_path_correction"
                if not blockers
                else "no_go"
            ),
        },
        corrected,
    )


def _completed_local_rows(
    campaign_dir: Path, blockers: list[str]
) -> list[dict[str, str]]:
    with (campaign_dir / "state.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row.get("condition_id", "")), int(row.get("seed", ""))),
            [],
        ).append(row)
    completed = []
    for identity, identity_rows in sorted(grouped.items()):
        latest_attempt = max(int(row["attempt"]) for row in identity_rows)
        latest_rows = [
            row
            for row in identity_rows
            if int(row["attempt"]) == latest_attempt
        ]
        latest = latest_rows[-1]
        if latest.get("status") != "complete":
            blockers.append(
                f"latest state is not complete for {identity}: "
                f"{latest.get('status')}"
            )
            continue
        run_id = str(latest["run_id"])
        completed.append(
            {
                **latest,
                "run_dir": str((campaign_dir / "runs" / run_id).resolve()),
            }
        )
    return completed


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
