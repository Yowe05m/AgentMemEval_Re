from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agentmemeval.core.errors import ConfigError
from agentmemeval.experiments.formal_protocol import sha256_json
from agentmemeval.experiments.formal_runner import (
    generate_worker_manifests,
    run_worker_manifest,
    validate_worker_manifest,
    validate_worker_manifest_set,
)

TASK8B_SEEDS = list(range(2026090101, 2026090113))
FROZEN_EXPEDITED_STATUS = "frozen/expedited-formal-candidate"


def _matrix(path: Path) -> Path:
    path.write_text(
        "experiment_id,status,checkpoint_set,heldout_tables,memory_mode\n"
        "R1-E1-I,required,30|75|150|300,3,Frozen\n"
        "R1-E1-M,required,,,Frozen\n"
        "R1-E2,required,,,Frozen\n"
        "R1-E3,required,,,Frozen\n"
        "R1-E4,required,,,Online\n"
        "R1-E5,required,,,Without\n"
        "R1-E6,required,,,derived\n",
        encoding="utf-8",
    )
    return path


def _legacy_common_identity() -> dict[str, str]:
    return {
        "code_sha": "a" * 40,
        "resolved_config_sha256": "b" * 64,
        "prompt_sha256": "c" * 64,
        "model_fingerprint": "qwen-frozen-v1",
        "embedding_fingerprint": "bge-m3-frozen-v1",
        "protocol_sha256": "e" * 64,
        "runtime_image_fingerprint": "f" * 64,
        "schedule_sha256": "d" * 64,
    }


def _generate(
    tmp_path: Path,
    *,
    seeds: list[int] | None = None,
    protocol_status: str = FROZEN_EXPEDITED_STATUS,
    execution_mode: str = "formal_candidate",
) -> tuple[Path, list[dict[str, object]]]:
    output = tmp_path / "manifests"
    selected_seeds = TASK8B_SEEDS if seeds is None else seeds
    task_configs_by_worker: dict[str, list[dict[str, object]]] = {}
    for index, seed in enumerate(selected_seeds, start=1):
        for prefix in ("P", "S"):
            worker_id = f"{prefix}{index:02d}"
            task_configs_by_worker[worker_id] = [
                {
                    "task_id": f"{worker_id.lower()}_fixture_task",
                    "planned_hands": 1,
                    "expected_identity": {
                        "schedule_sha256": sha256_json(
                            {"seed": seed, "worker_id": worker_id}
                        )
                    },
                }
            ]
    generate_worker_manifests(
        matrix_path=_matrix(tmp_path / "matrix.csv"),
        seeds=selected_seeds,
        common_identity=_legacy_common_identity(),
        output_dir=output,
        output_root="outputs/formal/task8b-expedited",
        cache_root="task8b-expedited",
        protocol_status=protocol_status,
        execution_mode=execution_mode,
        task_configs_by_worker=task_configs_by_worker,
    )
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output.glob("[PS][0-9][0-9].json"))
    ]
    return output, manifests


def _apply_layered_schedule_identity(
    manifests: list[dict[str, object]],
) -> dict[int, str]:
    schedule_by_seed = {
        seed: sha256_json({"seed": seed, "tables": ["H01", "H02", "H03"]})
        for seed in TASK8B_SEEDS
    }
    for manifest in manifests:
        common = manifest["common_identity"]
        assert isinstance(common, dict)
        common.pop("schedule_sha256", None)
        seed = int(manifest["seed_bundle"])
        manifest["seed_pod_identity"] = {
            "seed_bundle": seed,
            "schedule_sha256": schedule_by_seed[seed],
        }
        classification = manifest["identity_classification"]
        assert isinstance(classification, dict)
        classification["fleet_common"] = [
            "code_sha",
            "prompt_sha256",
            "model_fingerprint",
            "embedding_fingerprint",
        ]
        classification["seed_pod"] = ["seed_bundle", "schedule_sha256"]
        classification.pop("cross_instance_same", None)
    return schedule_by_seed


def test_expedited_generator_requires_exact_preregistered_seed_order(
    tmp_path: Path,
) -> None:
    wrong = TASK8B_SEEDS.copy()
    wrong[-1] = 2026090999

    with pytest.raises(ConfigError, match="2026090101.*2026090112|seed"):
        _generate(tmp_path, seeds=wrong)


def test_expedited_bundle_is_exactly_12_by_2_with_closed_dependencies_and_unique_io(
    tmp_path: Path,
) -> None:
    _, manifests = _generate(tmp_path)

    assert len(manifests) == 24
    primaries = sorted(
        (item for item in manifests if item["role"] == "primary"),
        key=lambda item: str(item["worker_id"]),
    )
    assert [item["seed_bundle"] for item in primaries] == TASK8B_SEEDS
    assert {item["role"] for item in manifests} == {"primary", "secondary"}
    assert len({item["worker_id"] for item in manifests}) == 24
    assert len(
        {item["instance_identity"]["output_path"] for item in manifests}
    ) == 24
    assert len(
        {item["instance_identity"]["cache_namespace"] for item in manifests}
    ) == 24

    by_worker = {str(item["worker_id"]): item for item in manifests}
    for index, seed in enumerate(TASK8B_SEEDS, start=1):
        primary = by_worker[f"P{index:02d}"]
        secondary = by_worker[f"S{index:02d}"]
        assert primary["seed_bundle"] == secondary["seed_bundle"] == seed
        assert secondary["depends_on"] == primary["worker_id"]
        assert (
            secondary["dependency_output_path"]
            == primary["instance_identity"]["output_path"]
        )


def test_schedule_identity_is_equal_within_pod_but_not_globally_across_seeds(
    tmp_path: Path,
) -> None:
    _, manifests = _generate(tmp_path)
    schedule_by_seed = _apply_layered_schedule_identity(manifests)

    validate_worker_manifest_set(manifests, expected_seed_count=12)

    assert len(set(schedule_by_seed.values())) == 12
    for seed in TASK8B_SEEDS:
        pod = [item for item in manifests if int(item["seed_bundle"]) == seed]
        assert len(pod) == 2
        assert {
            item["seed_pod_identity"]["schedule_sha256"] for item in pod
        } == {schedule_by_seed[seed]}
        assert all(
            "schedule_sha256" not in item["common_identity"] for item in pod
        )


def test_schedule_identity_mismatch_inside_seed_pod_fails_closed(tmp_path: Path) -> None:
    _, manifests = _generate(tmp_path)
    _apply_layered_schedule_identity(manifests)
    broken = copy.deepcopy(manifests)
    secondary = next(item for item in broken if item["worker_id"] == "S01")
    secondary["seed_pod_identity"]["schedule_sha256"] = "f" * 64

    with pytest.raises(ConfigError, match="schedule|seed.pod|identity"):
        validate_worker_manifest_set(broken, expected_seed_count=12)


@pytest.mark.parametrize(
    "status",
    [
        "expedited-formal-candidate",
        "candidate/not-frozen/not-authorized-to-run",
        "formal",
        "frozen/formal",
    ],
)
def test_only_explicit_frozen_expedited_status_is_admitted(
    tmp_path: Path, status: str
) -> None:
    _, manifests = _generate(tmp_path)
    manifest = manifests[0]
    manifest["protocol_status"] = status

    with pytest.raises(ConfigError, match="candidate|frozen|expedited|授权"):
        validate_worker_manifest(manifest)


def test_explicit_frozen_expedited_status_is_admitted(tmp_path: Path) -> None:
    _, manifests = _generate(tmp_path)

    validate_worker_manifest(manifests[0])


def test_real_canary_accepts_at_most_100_total_hands(tmp_path: Path) -> None:
    _, manifests = _generate(
        tmp_path,
        seeds=[101],
        protocol_status="mock/not-for-paper/model-substituted",
        execution_mode="mock_seed_pod",
    )
    manifest = manifests[0]
    manifest["protocol_status"] = "canary/not-for-paper"
    manifest["canary_total_hands"] = 100

    validate_worker_manifest(manifest)


def test_real_canary_rejects_more_than_100_total_hands(tmp_path: Path) -> None:
    _, manifests = _generate(
        tmp_path,
        seeds=[101],
        protocol_status="mock/not-for-paper/model-substituted",
        execution_mode="mock_seed_pod",
    )
    manifest = manifests[0]
    manifest["protocol_status"] = "canary/not-for-paper"
    manifest["canary_total_hands"] = 101

    with pytest.raises(ConfigError, match="100|canary.*hands"):
        validate_worker_manifest(manifest)


def test_real_canary_completion_receipt_is_permanently_not_for_paper(
    tmp_path: Path,
) -> None:
    output, _ = _generate(
        tmp_path,
        seeds=[2026090199],
        protocol_status="mock/not-for-paper/model-substituted",
        execution_mode="mock_seed_pod",
    )
    manifest_path = output / "P01.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol_status"] = "canary/not-for-paper"
    manifest["canary_total_hands"] = 2
    manifest["worker_planned_hands"] = 1
    manifest["instance_identity"]["output_path"] = str(tmp_path / "canary-output")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="",
    )

    result = run_worker_manifest(manifest_path, receipt_root=tmp_path / "receipts")

    completion = json.loads(
        (Path(result["run_dir"]) / "completion_receipt.json").read_text(encoding="utf-8")
    )
    assert completion["status"] == "complete"
    assert completion["not_for_paper"] is True
