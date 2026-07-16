"""Aggregate measured latency and explicitly estimated token/resource evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


def build_campaign_resource_audit(campaign_dir: str | Path) -> dict[str, Any]:
    """Audit completed leaves without treating heuristic token counts as API usage."""

    root = Path(campaign_dir).resolve()
    state_path = root / "state.tsv"
    with state_path.open("r", encoding="utf-8", newline="") as handle:
        states = list(csv.DictReader(handle, delimiter="\t"))
    completed = [row for row in states if row.get("status") == "complete"]
    latencies: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    action_fallbacks = 0
    revisions: set[tuple[str, str, str, str]] = set()
    revision_fallbacks: set[tuple[str, str, str, str]] = set()
    gpu_identities: set[tuple[str, str, str]] = set()
    leaf_sources = []
    for state in completed:
        run_id = str(state["run_id"])
        local = root / "runs" / run_id
        run_dir = local if local.is_dir() else Path(str(state["run_dir"]))
        events_path = run_dir / "events.jsonl"
        manifest_path = run_dir / "manifest.json"
        for event in _read_jsonl(events_path):
            llm = event.get("llm")
            if isinstance(llm, dict) and llm.get("elapsed_ms") is not None:
                latencies.append(float(llm["elapsed_ms"]))
                prompt_tokens += int(llm.get("prompt_tokens", 0))
                completion_tokens += int(llm.get("completion_tokens", 0))
            action_fallbacks += int(bool(event.get("fallback_used")))
            context = event.get("memory_context")
            experience = context.get("experience") if isinstance(context, dict) else None
            metadata = experience.get("metadata") if isinstance(experience, dict) else None
            if isinstance(metadata, dict):
                key = (
                    run_id,
                    str(event.get("agent_id", "")),
                    str(experience.get("version", "")),
                    str(metadata.get("prompt_sha256", "")),
                )
                revisions.add(key)
                if metadata.get("fallback_used"):
                    revision_fallbacks.add(key)
        manifest = _read_json(manifest_path)
        devices = (
            manifest.get("metadata", {}).get("gpu", {}).get("devices", [])
        )
        for device in devices:
            if isinstance(device, dict):
                gpu_identities.add(
                    (
                        str(device.get("name", "")),
                        str(device.get("driver", "")),
                        str(device.get("pci_bus_id", "")),
                    )
                )
        leaf_sources.append(
            {
                "run_id": run_id,
                "events_sha256": _sha256(events_path),
                "manifest_sha256": _sha256(manifest_path),
            }
        )
    times = [
        _parse_time(row["event_utc"])
        for row in states
        if row.get("event_utc") and row.get("status") in {"running", "complete"}
    ]
    wall_seconds = (max(times) - min(times)).total_seconds() if len(times) >= 2 else 0.0
    return {
        "schema_version": "task4_campaign_resource_audit_v1",
        "campaign_dir": str(root),
        "state_tsv_sha256": _sha256(state_path),
        "completed_leaf_count": len(completed),
        "campaign_wall_seconds": wall_seconds,
        "campaign_wall_hours": wall_seconds / 3600.0,
        "action_request_count": len(latencies),
        "action_latency_ms": _summary(latencies),
        "action_requests_per_wall_second": (
            len(latencies) / wall_seconds if wall_seconds > 0 else None
        ),
        "token_accounting": {
            "status": "heuristic_estimate_not_provider_usage",
            "method": "whitespace_split_prompt_and_short_structured_completion_proxy",
            "estimated_prompt_tokens": prompt_tokens,
            "estimated_completion_tokens": completion_tokens,
            "estimated_total_tokens": prompt_tokens + completion_tokens,
        },
        "experience_revision_count": len(revisions),
        "action_fallback_count": action_fallbacks,
        "experience_revision_fallback_count": len(revision_fallbacks),
        "gpu_identities": [
            {"name": name, "driver": driver, "pci_bus_id": bus}
            for name, driver, bus in sorted(gpu_identities)
        ],
        "gpu_utilization_status": "requires_external_service_heartbeat_evidence",
        "monetary_cost_status": "unavailable_local_service_has_no_provider_invoice",
        "leaf_sources": leaf_sources,
    }


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[index],
        "max": ordered[-1],
    }


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
