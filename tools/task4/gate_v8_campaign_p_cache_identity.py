"""Resolve the V8 Campaign-P run-local cache namespace false heterogeneity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from agentmemeval.evaluation.aggregation import validate_runtime_homogeneity
from agentmemeval.experiments.campaign import build_campaign_aggregate_payload


EXPECTED_RAW_BLOCKERS = {
    "aggregate runtime homogeneity is not verified",
    "runtime identity is not homogeneous: 2 identities",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--original-campaign-dir", required=True)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--raw-gate", required=True)
    parser.add_argument("--decision-gate", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--expected-run-count", type=int, required=True)
    args = parser.parse_args()

    audit = build_gate(
        Path(args.campaign_dir),
        original_campaign_dir=Path(args.original_campaign_dir),
        aggregate_path=Path(args.aggregate),
        raw_gate_path=Path(args.raw_gate),
        decision_gate_paths=[Path(value) for value in args.decision_gate],
        expected_code_sha=args.expected_code_sha,
        expected_run_count=args.expected_run_count,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output.resolve()), **audit}, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "ready_to_start_campaign_e" else 2


def build_gate(
    campaign_dir: Path,
    *,
    original_campaign_dir: Path,
    aggregate_path: Path,
    raw_gate_path: Path,
    decision_gate_paths: list[Path],
    expected_code_sha: str,
    expected_run_count: int,
) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    original_campaign_dir_posix = PurePosixPath(str(original_campaign_dir).replace("\\", "/"))
    aggregate_path = aggregate_path.resolve()
    raw_gate_path = raw_gate_path.resolve()
    blockers: list[str] = []
    raw_gate = _read_json(raw_gate_path)
    aggregate = _read_json(aggregate_path)
    campaign_manifest = _read_json(campaign_dir / "campaign_manifest.json")

    raw_blockers = {str(value) for value in raw_gate.get("blockers", [])}
    if raw_gate.get("status") != "no_go" or raw_blockers != EXPECTED_RAW_BLOCKERS:
        blockers.append(f"raw gate blockers are not the expected isolated false positive: {sorted(raw_blockers)}")
    if raw_gate.get("behavior_freeze_preview", {}).get("status") != "frozen":
        blockers.append("raw gate behavior freeze did not pass")
    if raw_gate.get("aggregate_metrics_match_canonical_leaf_rebuild") is not True:
        blockers.append("raw gate aggregate metrics did not match leaf rebuild")
    power_blockers = {
        str(value)
        for value in raw_gate.get("campaign_p_power_diagnostic", {}).get("blockers", [])
    }
    if power_blockers != {"aggregate runtime homogeneity is not verified"}:
        blockers.append(f"unexpected raw power blockers: {sorted(power_blockers)}")

    completed = _completed_local_rows(campaign_dir, blockers)
    if len(completed) != expected_run_count:
        blockers.append(f"complete run mismatch: {len(completed)}/{expected_run_count}")
    manifests = []
    run_ids = []
    for row in completed:
        run_id = str(row["run_id"])
        run_ids.append(run_id)
        run_dir = Path(row["run_dir"])
        manifest = _read_json(run_dir / "manifest.json")
        manifests.append(manifest)
        code = manifest.get("metadata", {}).get("code", {})
        if code.get("commit") != expected_code_sha or code.get("dirty") is not False:
            blockers.append(f"{run_id} code identity mismatch: {code}")
        observed_namespace = (
            manifest.get("metadata", {})
            .get("embedding", {})
            .get("cache_namespace_template")
        )
        expected_namespace = str(
            original_campaign_dir_posix
            / "runs"
            / run_id
            / "embedding_cache"
            / "{agent_id}.json"
        )
        if str(observed_namespace) != expected_namespace:
            blockers.append(
                f"{run_id} cache namespace is not canonical run-local: "
                f"{observed_namespace}/{expected_namespace}"
            )

    normalized_homogeneity = validate_runtime_homogeneity(manifests)
    if normalized_homogeneity.get("homogeneous") is not True:
        blockers.append(
            "runtime remains heterogeneous after excluding canonical run-local "
            f"cache namespaces: {normalized_homogeneity.get('mismatches')}"
        )
    corrected_aggregate = build_campaign_aggregate_payload(
        campaign_manifest["campaign"],
        campaign_manifest["base_config"],
        completed,
    )
    corrected_homogeneity = corrected_aggregate.get("runtime_homogeneity", {})
    if corrected_homogeneity.get("formal_aggregation_allowed") is not True:
        blockers.append("corrected leaf aggregate is not runtime homogeneous")
    metrics_match = _json_sha256(aggregate.get("aggregate_metrics")) == _json_sha256(
        corrected_aggregate.get("aggregate_metrics")
    )
    if not metrics_match:
        blockers.append("corrected rebuild changed aggregate metrics")

    decision_gates = [_read_json(path.resolve()) for path in decision_gate_paths]
    decision_run_ids = sorted(
        Path(str(item.get("run_dir", ""))).name for item in decision_gates
    )
    if decision_run_ids != sorted(run_ids):
        blockers.append(f"decision gate run ids mismatch: {decision_run_ids}/{sorted(run_ids)}")
    for item in decision_gates:
        if item.get("status") != "ready_to_start_v8_calibration_pilot" or item.get("blockers"):
            blockers.append(
                f"decision-point gate failed for {Path(str(item.get('run_dir', ''))).name}"
            )

    return {
        "schema_version": "task4_v8_campaign_p_cache_identity_gate_v1",
        "classification": "pilot_not_for_paper",
        "campaign_dir": str(campaign_dir),
        "original_campaign_dir": str(original_campaign_dir_posix),
        "expected_code_sha": expected_code_sha,
        "completed_run_ids": sorted(run_ids),
        "raw_gate_path": str(raw_gate_path),
        "raw_gate_sha256": _sha256(raw_gate_path),
        "raw_gate_status": raw_gate.get("status"),
        "raw_gate_blockers": sorted(raw_blockers),
        "normalization_rule": (
            "exclude only metadata.embedding.cache_namespace_template from "
            "cross-run runtime identity after verifying each value equals its "
            "canonical run-local embedding_cache/{agent_id}.json path"
        ),
        "normalized_runtime_homogeneity": normalized_homogeneity,
        "corrected_runtime_homogeneity": corrected_homogeneity,
        "aggregate_metrics_unchanged": metrics_match,
        "aggregate_metrics_sha256": _json_sha256(aggregate.get("aggregate_metrics")),
        "corrected_aggregate_sha256": _json_sha256(corrected_aggregate),
        "decision_gate_sha256": {
            path.name: _sha256(path.resolve()) for path in decision_gate_paths
        },
        "blockers": blockers,
        "status": "ready_to_start_campaign_e" if not blockers else "no_go",
    }


def _completed_local_rows(campaign_dir: Path, blockers: list[str]) -> list[dict[str, str]]:
    with (campaign_dir / "state.tsv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((str(row["condition_id"]), int(row["seed"])), []).append(row)
    completed = []
    for identity, identity_rows in sorted(grouped.items()):
        latest_attempt = max(int(row["attempt"]) for row in identity_rows)
        latest = [row for row in identity_rows if int(row["attempt"]) == latest_attempt][-1]
        if latest.get("status") != "complete":
            blockers.append(f"latest state is not complete for {identity}: {latest.get('status')}")
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
