"""Read-only, protocol-aware progress accounting for append-only campaigns."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_COMPLETE_ARTIFACTS = (
    "resolved_config.yaml",
    "manifest.json",
    "hand_summaries.jsonl",
    "metrics.json",
    "protocol_audit.json",
    "checkpoint_generalization.json",
    "report.md",
    "experiment_result.json",
)


def build_campaign_progress(campaign_dir: str | Path) -> dict[str, Any]:
    """Rebuild current progress without mutating a campaign."""

    root = Path(campaign_dir).resolve()
    manifest_path = root / "campaign_manifest.json"
    state_path = root / "state.tsv"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "agentmemeval_campaign_v1":
        raise ValueError(
            f"unsupported campaign manifest schema: "
            f"{manifest.get('schema_version')}"
        )
    campaign = manifest.get("campaign")
    base_config = manifest.get("base_config")
    if not isinstance(campaign, dict) or not isinstance(base_config, dict):
        raise ValueError("campaign manifest lacks campaign/base_config")
    conditions = _conditions(campaign)
    seeds = [int(seed) for seed in campaign.get("seeds", [])]
    expected_identities = [
        (str(condition["condition_id"]), seed)
        for condition in conditions
        for seed in seeds
    ]
    state_rows = _read_state(state_path)
    grouped_state_rows: dict[tuple[str, int], list[dict[str, str]]] = {}
    latest_by_identity: dict[tuple[str, int], dict[str, str]] = {}
    state_anomalies: list[str] = []
    for row in state_rows:
        try:
            condition_id = str(row["condition_id"]).strip()
            identity = (condition_id, int(row["seed"]))
            attempt = int(row["attempt"])
            if not condition_id or attempt < 1:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            state_anomalies.append(f"malformed state row: {row}")
            continue
        grouped_state_rows.setdefault(identity, []).append(row)
    superseded_failed_state_rows = 0
    for identity, identity_rows in grouped_state_rows.items():
        maximum_attempt = max(int(row["attempt"]) for row in identity_rows)
        latest_attempt_rows = [
            row
            for row in identity_rows
            if int(row["attempt"]) == maximum_attempt
        ]
        latest = latest_attempt_rows[-1]
        latest_by_identity[identity] = latest
        completed_attempts = {
            int(row["attempt"])
            for row in identity_rows
            if row.get("status") == "complete"
        }
        if len(completed_attempts) > 1:
            state_anomalies.append(
                f"multiple completed attempts for {identity}: "
                f"{sorted(completed_attempts)}"
            )
        if (
            any(row.get("status") == "failed" for row in latest_attempt_rows)
            and latest.get("status") == "complete"
        ):
            state_anomalies.append(
                f"failed state precedes completion within latest attempt for "
                f"{identity}"
            )
        superseded_failed_state_rows += sum(
            row.get("status") == "failed"
            and int(row["attempt"]) < maximum_attempt
            for row in identity_rows
        )
    for identity in sorted(set(grouped_state_rows) - set(expected_identities)):
        state_anomalies.append(f"unexpected state identity: {identity}")

    default_budget = _budget_from_config(base_config)
    units: list[dict[str, Any]] = []
    observed_hands_total = 0
    expected_hands_total = 0
    status_counts: dict[str, int] = {}
    for condition_id, seed in expected_identities:
        identity = (condition_id, seed)
        row = latest_by_identity.get(identity)
        run_id = str(row.get("run_id", "")) if row else ""
        run_dir = root / "runs" / run_id if run_id else None
        config_path = run_dir / "resolved_config.yaml" if run_dir else None
        config = (
            _read_yaml(config_path)
            if config_path is not None and config_path.is_file()
            else base_config
        )
        budget = _budget_from_config(config)
        expected_hands = int(budget["total_hand_summaries_per_run"])
        observed_hands = (
            _count_lines(run_dir / "hand_summaries.jsonl")
            if run_dir is not None
            and (run_dir / "hand_summaries.jsonl").is_file()
            else 0
        )
        status = str(row.get("status", "pending")) if row else "pending"
        status_counts[status] = status_counts.get(status, 0) + 1
        anomalies: list[str] = []
        if observed_hands > expected_hands:
            anomalies.append(
                f"observed hand summaries exceed protocol budget: "
                f"{observed_hands}/{expected_hands}"
            )
        if status == "complete" and observed_hands != expected_hands:
            anomalies.append(
                f"complete unit hand count mismatch: "
                f"{observed_hands}/{expected_hands}"
            )
        if status == "complete" and run_dir is not None:
            missing = [
                name
                for name in REQUIRED_COMPLETE_ARTIFACTS
                if not (run_dir / name).is_file()
                or (run_dir / name).stat().st_size < 1
            ]
            if missing:
                anomalies.append(f"complete unit missing artifacts: {missing}")
            protocol_path = run_dir / "protocol_audit.json"
            if protocol_path.is_file():
                protocol_budget = _read_json(protocol_path).get(
                    "checkpoint_cost_budget"
                )
                if (
                    isinstance(protocol_budget, dict)
                    and protocol_budget != budget["checkpoint_cost_budget"]
                ):
                    anomalies.append(
                        "protocol checkpoint_cost_budget differs from rebuilt budget"
                    )
        observed_hands_total += observed_hands
        expected_hands_total += expected_hands
        units.append(
            {
                "condition_id": condition_id,
                "seed": seed,
                "attempt": int(row["attempt"]) if row else None,
                "status": status,
                "run_id": run_id or None,
                "observed_hand_summaries": observed_hands,
                "expected_hand_summaries": expected_hands,
                "progress_fraction": round(
                    min(observed_hands / expected_hands, 1.0), 6
                ),
                "stage": _stage(
                    status,
                    observed_hands,
                    int(budget["train_hands"]),
                    expected_hands,
                ),
                "budget": budget,
                "anomalies": anomalies,
            }
        )
    anomalies = [
        *state_anomalies,
        *[
            f"{unit['condition_id']}/{unit['seed']}: {anomaly}"
            for unit in units
            for anomaly in unit["anomalies"]
        ],
    ]
    return {
        "schema_version": "agentmemeval_campaign_progress_v2",
        "campaign_id": campaign.get("campaign_id"),
        "campaign_dir": str(root),
        "design": campaign.get("design"),
        "matrix_order": campaign.get("matrix_order", "condition_major"),
        "expected_matrix_units": len(expected_identities),
        "status_counts": status_counts,
        "observed_hand_summaries_total": observed_hands_total,
        "expected_hand_summaries_total": expected_hands_total,
        "progress_fraction": round(
            (
                observed_hands_total / expected_hands_total
                if expected_hands_total
                else 0.0
            ),
            6,
        ),
        "default_budget": default_budget,
        "state_audit": {
            "state_row_count": len(state_rows),
            "latest_attempt_matrix_units": len(latest_by_identity),
            "failed_state_rows": sum(
                row.get("status") == "failed" for row in state_rows
            ),
            "superseded_failed_state_rows": superseded_failed_state_rows,
            "latest_failed_matrix_units": sum(
                row.get("status") == "failed"
                for row in latest_by_identity.values()
            ),
        },
        "units": units,
        "anomalies": anomalies,
        "status": "consistent" if not anomalies else "inconsistent",
        "paper_eligibility_not_assessed": True,
    }


def _budget_from_config(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("config lacks experiment mapping")
    train_hands = int(experiment.get("train_hands", 0))
    checkpoint_interval = int(experiment.get("checkpoint_interval", 0))
    checkpoint_test_hands = int(
        experiment.get(
            "checkpoint_test_hands",
            experiment.get("test_hands", 0),
        )
    )
    if train_hands <= 0 or checkpoint_interval <= 0:
        checkpoint_count = 1
    else:
        checkpoint_count = (
            train_hands + checkpoint_interval - 1
        ) // checkpoint_interval
    evaluation_targets = experiment.get("evaluation_target_ids")
    if isinstance(evaluation_targets, list) and evaluation_targets:
        evaluation_target_count = len(evaluation_targets)
    elif experiment.get("evaluate_all_train_agents") is True:
        roster = experiment.get("agent_roster")
        if not isinstance(roster, list) or not roster:
            raise ValueError("evaluate_all_train_agents requires agent_roster")
        evaluation_target_count = len(roster)
    else:
        evaluation_target_count = 1
    checkpoint_evaluations = checkpoint_count * evaluation_target_count
    checkpoint_hands = checkpoint_evaluations * checkpoint_test_hands
    checkpoint_budget = {
        "checkpoint_count_per_seed": checkpoint_count,
        "evaluation_target_count": evaluation_target_count,
        "checkpoint_evaluations_per_seed": checkpoint_evaluations,
        "checkpoint_generalization_hands_per_seed": checkpoint_hands,
        "seed_count": 1,
        "checkpoint_generalization_hands_all_seeds": checkpoint_hands,
    }
    return {
        "train_hands": train_hands,
        "checkpoint_interval": checkpoint_interval,
        "checkpoint_test_hands": checkpoint_test_hands,
        "checkpoint_cost_budget": checkpoint_budget,
        "total_hand_summaries_per_run": train_hands + checkpoint_hands,
    }


def _stage(
    status: str,
    observed_hands: int,
    train_hands: int,
    expected_hands: int,
) -> str:
    if status == "complete":
        return "complete"
    if status == "failed":
        return "failed"
    if observed_hands < train_hands:
        return "training" if observed_hands else "pending_or_initializing"
    if observed_hands < expected_hands:
        return "checkpoint_generalization"
    return "finalizing"


def _conditions(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    configured = campaign.get("conditions")
    if configured is None and campaign.get("design") == "mixed_table":
        return [{"condition_id": "mixed_table", "target_mechanism": "mixed"}]
    if not isinstance(configured, list) or not configured:
        raise ValueError("campaign conditions are missing")
    if not all(
        isinstance(item, dict) and item.get("condition_id")
        for item in configured
    ):
        raise ValueError("campaign condition is malformed")
    return [dict(item) for item in configured]


def _read_state(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)
