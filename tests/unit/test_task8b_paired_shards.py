from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agentmemeval.config.loader import ConfigError
from agentmemeval.experiments import formal_runner
from agentmemeval.experiments import task8b_paired_shards as shards


@pytest.fixture(autouse=True)
def _fixed_amendment_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        shards,
        "_verify_amendment",
        lambda _path, declared: (
            None
            if declared == shards.EXPECTED_AMENDMENT_SHA256
            else (_ for _ in ()).throw(ConfigError("amendment SHA binding"))
        ),
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _failed_state(path: Path) -> str:
    body = {
        "schema_version": "task8-worker-state-v1",
        "created_at_utc": "2026-07-24T00:00:00Z",
        "status": "failed",
        "detail": "ConfigError: resolved config identity mismatch",
        "previous_sha256": "GENESIS",
    }
    row = {**body, "row_sha256": formal_runner.sha256_json(body)}
    path.write_text(
        "\t".join(row) + "\n"
        + "\t".join(str(row[field]) for field in row)
        + "\n",
        encoding="utf-8",
    )
    return str(row["row_sha256"])


def _identity(suffix: str) -> dict[str, str]:
    return {
        "code_sha": "a" * 40,
        "prompt_sha256": "1" * 64,
        "model_fingerprint": "2" * 64,
        "embedding_fingerprint": "3" * 64,
        "protocol_sha256": "4" * 64,
        "runtime_image_fingerprint": "5" * 64,
        "resolved_config_sha256": "6" * 64,
        "schedule_sha256": hashlib.sha256(suffix.encode()).hexdigest(),
    }


def _canonical_manifest(
    tmp_path: Path,
    *,
    role: str,
    seed: int = 2026090107,
) -> Path:
    worker = ("P" if role == "primary" else "S") + f"{seed - 2026090100:02d}"
    tasks = []
    for task_id, hands in shards.EXPECTED_TASKS[role].items():
        config = tmp_path / "configs" / f"{task_id}.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"task: {task_id}\n", encoding="utf-8")
        identity = _identity(task_id)
        task = {
            "task_id": task_id,
            "planned_hands": hands,
            "config_path": str(config),
            "config_sha256": _sha(config),
            "schedule_sha256": identity["schedule_sha256"],
            "expected_identity": identity,
            "covers": (
                ["R1-E1-I", "R1-E2", "R1-E3"]
                if role == "primary"
                else (
                    ["R1-E1-M"]
                    if task_id == "mixed_ecological"
                    else ["R1-E4" if task_id.endswith("online") else "R1-E5"]
                )
            ),
            "memory_mode": (
                "Frozen"
                if role == "primary" or task_id == "mixed_ecological"
                else ("Online" if task_id.endswith("online") else "Without")
            ),
        }
        if role == "secondary":
            task["dependency_mode"] = (
                "standalone" if task_id == "mixed_ecological" else "checkpoint"
            )
            if task["dependency_mode"] == "checkpoint":
                task["checkpoint_bindings"] = {
                    "agent_00": (
                        "runs/isolation_expr/snapshots/"
                        "isolation_expr_checkpoint_0300.json"
                    )
                }
        tasks.append(task)
    common = _identity("common")
    common.pop("resolved_config_sha256")
    common.pop("schedule_sha256")
    receipt_identity = _identity("receipt")
    manifest = {
        "schema_version": formal_runner.WORKER_SCHEMA_VERSION,
        "protocol_status": formal_runner.TASK8B_EXPEDITED_STATUS,
        "execution_mode": "experiment_configs",
        "worker_id": worker,
        "role": role,
        "pod_id": f"pod{seed - 2026090100:02d}",
        "seed_bundle": seed,
        "experiment_families": [],
        "checkpoint_set": [30, 75, 150, 300],
        "heldout_table_set": ["H01", "H02", "H03"],
        "memory_modes": [],
        "matrix_sha256": "7" * 64,
        "common_identity": common,
        "instance_identity": {
            "worker_id": worker,
            "cache_namespace": f"task8b/{worker}/{seed}",
            "output_path": f"outputs/formal/task8b/{worker}/{seed}",
        },
        "depends_on": None if role == "primary" else f"P{seed - 2026090100:02d}",
        "dependency_output_path": (
            None
            if role == "primary"
            else f"outputs/formal/task8b/P{seed - 2026090100:02d}/{seed}"
        ),
        "receipt_relative_path": f"receipts/P{seed - 2026090100:02d}.json",
        "receipt_identity": receipt_identity if role == "primary" else None,
        "dependency_receipt_identity": (
            receipt_identity if role == "secondary" else None
        ),
        "seed_pod_identity": {
            "seed_bundle": seed,
            "schedule_sha256": "8" * 64,
        },
        "task_configs": tasks,
    }
    path = tmp_path / "canonical" / f"{worker}.json"
    _write_json(path, manifest)
    return path


def _authorization(
    tmp_path: Path,
    canonical_path: Path,
    *,
    selected: list[str],
    shard_id: str,
    seal_mode: str = "execution",
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(canonical_path.read_text(encoding="utf-8"))
    worker = manifest["worker_id"]
    seed = manifest["seed_bundle"]
    role = manifest["role"]
    amendment = tmp_path / "amendment.md"
    amendment.write_text("paired shard amendment\n", encoding="utf-8")
    scientific_checkout = tmp_path / "scientific"
    scientific_checkout.mkdir(exist_ok=True)
    staging_root = tmp_path / "approved_staging"
    receipt_root = tmp_path / "approved_receipts"
    staging_root.mkdir(exist_ok=True)
    receipt_root.mkdir(exist_ok=True)
    selected_set = set(selected)
    shard_role = next(
        side
        for side in ("high", "low")
        if selected_set.issubset(shards.SIDE_TASKS[(role, side)])
    )
    bridge_root = None
    bridge_receipt_relative = None
    bridge_receipt_sha256 = None
    if role == "secondary":
        bridge_root_path = staging_root / "primary_bridge" / worker
        bridge_root_path.mkdir(parents=True, exist_ok=True)
        relative = (
            "runs/isolation_expr/snapshots/"
            "isolation_expr_checkpoint_0300.json"
        )
        file_path = bridge_root_path / relative
        _write_json(file_path, {"checkpoint": 300, "task_id": "isolation_expr"})
        checkpoint_files = [relative]
        _write_json(
            bridge_root_path / "bridge_manifest.json",
            {
                "schema_version": shards.PRIMARY_BRIDGE_SCHEMA,
                "status": "files_sealed_receipt_pending",
                "worker_id": manifest["depends_on"],
                "seed": seed,
                "pair_id": shards.PAIR_IDS[seed],
                "physical_mapping": shards.PHYSICAL_MAPPING[seed],
                "amendment_id": shards.AMENDMENT_ID,
                "amendment_sha256": shards.EXPECTED_AMENDMENT_SHA256,
                "frozen_code_sha": shards.FROZEN_CODE_SHA,
                "frozen_formal_runner_sha256": (
                    shards.FROZEN_FORMAL_RUNNER_SHA256
                ),
                "engineering_controller_sha256": _sha(
                    Path(shards.__file__).resolve()
                ),
                "task_union": list(shards.EXPECTED_TASKS["primary"]),
                "effect_fields_read": False,
            },
        )
        bridge_receipt_relative = manifest["receipt_relative_path"]
        bridge_receipt_path = receipt_root / bridge_receipt_relative
        if not bridge_receipt_path.exists():
            formal_runner.publish_checkpoint_receipt(
                checkpoint_root=bridge_root_path,
                checkpoint_files=checkpoint_files,
                receipt_path=bridge_receipt_path,
                producer_worker_id=manifest["depends_on"],
                seed_bundle=seed,
                checkpoint_hand=300,
                identity=manifest["dependency_receipt_identity"],
            )
        bridge_root = str(bridge_root_path.resolve())
        bridge_receipt_sha256 = _sha(bridge_receipt_path)
    value = {
        "schema_version": shards.AUTHORIZATION_SCHEMA,
        "active": True,
        "authorization_id": "",
        "amendment_id": shards.AMENDMENT_ID,
        "amendment_path": str(amendment),
        "amendment_sha256": shards.EXPECTED_AMENDMENT_SHA256,
        "scientific_checkout": str(scientific_checkout),
        "frozen_code_sha": shards.FROZEN_CODE_SHA,
        "frozen_formal_runner_sha256": shards.FROZEN_FORMAL_RUNNER_SHA256,
        "engineering_controller_sha256": _sha(Path(shards.__file__).resolve()),
        "approved_staging_root": str(staging_root.resolve()),
        "approved_receipt_root": str(receipt_root.resolve()),
        "denied_partial_root": str(
            (
                scientific_checkout
                / manifest["instance_identity"]["output_path"]
            ).resolve()
        ),
        "pair_id": shards.PAIR_IDS[seed],
        "physical_slot": shards.PHYSICAL_MAPPING[seed][shard_role],
        "shard_role": shard_role,
        "partition_id": f"partition-{worker}",
        "canonical_manifest_sha256": _sha(canonical_path),
        "worker_id": worker,
        "seed": seed,
        "shard_id": shard_id,
        "seal_mode": seal_mode,
        "selected_task_ids": selected,
        "derived_output_path": str(
            (staging_root / "outputs" / worker / str(seed) / shard_id).resolve()
        ),
        "derived_cache_namespace": str(
            (staging_root / "cache" / worker / str(seed) / shard_id).resolve()
        ),
        "derived_receipt_relative_path": (
            f"paired_shards/receipts/{worker}/{shard_id}.json"
            if role == "primary"
            else manifest["receipt_relative_path"]
        ),
        "primary_bridge_root": bridge_root,
        "primary_bridge_receipt_relative_path": bridge_receipt_relative,
        "primary_bridge_receipt_sha256": bridge_receipt_sha256,
        "effect_fields_read": False,
        "scientific_protocol_changed": False,
    }
    value["authorization_id"] = shards._authorization_id(value, role=role)
    path = tmp_path / "authorizations" / f"{worker}-{shard_id}.json"
    _write_json(path, value)
    return path


def _completed_shard(
    tmp_path: Path,
    canonical_path: Path,
    *,
    selected: list[str],
    shard_id: str,
    hand_delta: int = 0,
    valid_completion_hash: bool = True,
) -> Path:
    authorization = _authorization(
        tmp_path,
        canonical_path,
        selected=selected,
        shard_id=shard_id,
    )
    derived = shards.derive_authorized_manifest(canonical_path, authorization)
    manifest_path = (
        tmp_path
        / "approved_staging"
        / "derived"
        / f"{derived['worker_id']}-{shard_id}.json"
    )
    _write_json(manifest_path, derived)
    root = (
        tmp_path
        / "approved_staging"
        / "runs"
        / f"{derived['worker_id']}-{shard_id}"
    )
    for task in derived["task_configs"]:
        task_id = task["task_id"]
        child = root / "runs" / task_id
        _write_json(child / "experiment_result.json", {"status": "complete"})
        _write_json(
            child / "snapshots" / f"{task_id}_checkpoint_0300.json",
            {"checkpoint": 300, "task_id": task_id},
        )
        _write_json(child / "task_identity_audit.json", task["expected_identity"])
        (child / "hand_summaries.jsonl").write_text(
            "{}\n" * (int(task["planned_hands"]) + hand_delta),
            encoding="utf-8",
        )
        marker = {
            "schema_version": "task8-worker-task-receipt-v1",
            "task_id": task_id,
            "config_sha256": task["config_sha256"],
            "run_dir": f"runs/{task_id}",
            "task_row": {
                "task_id": task_id,
                "status": "complete",
                "identity_audit": {
                    **task["expected_identity"],
                    "code_dirty": False,
                },
            },
            "files": shards._directory_manifest(child),
        }
        _write_json(root / "task_receipts" / f"{task_id}.json", marker)
    files_tsv = root / "files.tsv"
    manifest_rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {"state.tsv", "files.tsv", "completion_receipt.json"}:
            continue
        manifest_rows.append(f"{relative}\t{path.stat().st_size}\t{_sha(path)}\n")
    files_tsv.write_text(
        "relative_path\tsize\tsha256\n" + "".join(manifest_rows),
        encoding="utf-8",
    )
    _write_json(
        root / "completion_receipt.json",
        {
            "schema_version": "task8-worker-completion-v1",
            "status": "complete",
            "worker_id": derived["worker_id"],
            "files_tsv_sha256": (
                _sha(files_tsv) if valid_completion_hash else "0" * 64
            ),
        },
    )
    receipt_path = (
        tmp_path
        / "approved_receipts"
        / "shard_receipts"
        / f"{derived['worker_id']}-{shard_id}.json"
    )
    shards.build_shard_receipt(manifest_path, root, receipt_path)
    return receipt_path


def test_p07_sync_only_never_falls_through_to_async(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical_manifest(tmp_path, role="primary")
    authorization = _authorization(
        tmp_path,
        canonical,
        selected=["isolation_sync"],
        shard_id="sync-only",
    )
    observed = {}
    original_cwd = Path.cwd()
    monkeypatch.setattr(
        shards,
        "_verify_scientific_checkout",
        lambda checkout: checkout / "src/agentmemeval/experiments/formal_runner.py",
    )

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        manifest_path = Path(argv[3])
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        observed["tasks"] = [task["task_id"] for task in manifest["task_configs"]]
        observed["publish"] = [
            task["publish_checkpoint_after"] for task in manifest["task_configs"]
        ]
        observed["cwd"] = str(kwargs["cwd"])
        assert Path(argv[4]) == tmp_path / "approved_receipts"
        return SimpleNamespace(
            stdout='{"status":"complete","run_dir":"staging"}\n'
        )

    monkeypatch.setattr(shards.subprocess, "run", fake_run)
    derived_output = tmp_path / "approved_staging" / "derived.json"
    result = shards.run_authorized_shard(
        canonical,
        authorization,
        derived_manifest_output=derived_output,
        receipt_root=tmp_path / "approved_receipts",
    )
    assert result["status"] == "complete"
    assert observed == {
        "tasks": ["isolation_sync"],
        "publish": [True],
        "cwd": str((tmp_path / "scientific").resolve()),
    }
    assert Path.cwd() == original_cwd
    with pytest.raises(ConfigError, match="原子 reservation"):
        shards.run_authorized_shard(
            canonical,
            authorization,
            derived_manifest_output=derived_output,
            receipt_root=tmp_path / "approved_receipts",
        )


def test_derived_manifest_is_byte_deterministic_and_fails_closed(
    tmp_path: Path,
) -> None:
    canonical = _canonical_manifest(tmp_path, role="primary")
    authorization = _authorization(
        tmp_path,
        canonical,
        selected=["isolation_sync"],
        shard_id="a",
    )
    first = shards.derive_authorized_manifest(canonical, authorization)
    second = shards.derive_authorized_manifest(canonical, authorization)
    assert shards._json_bytes(first) == shards._json_bytes(second)
    original = json.loads(canonical.read_text(encoding="utf-8"))
    derived_task = first["task_configs"][0]
    original_task = next(
        task
        for task in original["task_configs"]
        if task["task_id"] == "isolation_sync"
    )
    assert {
        key: value
        for key, value in derived_task.items()
        if key != "publish_checkpoint_after"
    } == {
        key: value
        for key, value in original_task.items()
        if key != "publish_checkpoint_after"
    }

    bad = json.loads(authorization.read_text(encoding="utf-8"))
    bad["selected_task_ids"] = ["isolation_async", "isolation_sync"]
    _write_json(authorization, bad)
    with pytest.raises(ConfigError, match="顺序"):
        shards.derive_authorized_manifest(canonical, authorization)


def test_output_overlap_and_inactive_authorization_fail_before_run(
    tmp_path: Path,
) -> None:
    canonical = _canonical_manifest(tmp_path, role="primary")
    authorization = _authorization(
        tmp_path,
        canonical,
        selected=["isolation_sync"],
        shard_id="a",
    )
    value = json.loads(authorization.read_text(encoding="utf-8"))
    value["derived_output_path"] = str(
        (
            tmp_path
            / "scientific"
            / "outputs/formal/task8b/P07/2026090107/shard-a"
        ).resolve()
    )
    _write_json(authorization, value)
    with pytest.raises(ConfigError, match="逃逸 approved root|output.*重叠"):
        shards.derive_authorized_manifest(canonical, authorization)
    value["derived_output_path"] = str(
        (tmp_path / "approved_staging" / "outputs/P07/a").resolve()
    )
    value["active"] = False
    _write_json(authorization, value)
    with pytest.raises(ConfigError, match="active gate"):
        shards.derive_authorized_manifest(canonical, authorization)


def test_composition_exact_union_hashes_and_determinism(tmp_path: Path) -> None:
    primary = _canonical_manifest(tmp_path, role="primary")
    secondary = _canonical_manifest(tmp_path, role="secondary")
    receipts = [
        _completed_shard(
            tmp_path,
            primary,
            selected=[
                "isolation_no_memory",
                "isolation_fact",
                "isolation_expr",
            ],
            shard_id="p-a",
        ),
        _completed_shard(
            tmp_path,
            primary,
            selected=["isolation_sync"],
            shard_id="p-sync",
        ),
        _completed_shard(
            tmp_path,
            primary,
            selected=["isolation_async"],
            shard_id="p-async",
        ),
        _completed_shard(
            tmp_path,
            secondary,
            selected=["mixed_ecological"],
            shard_id="s-a",
        ),
        _completed_shard(
            tmp_path,
            secondary,
            selected=[
                "expr_online",
                "expr_without",
                "async_online",
                "async_without",
            ],
            shard_id="s-b",
        ),
    ]
    first_path = tmp_path / "approved_receipts" / "composition-1.json"
    second_path = tmp_path / "approved_receipts" / "composition-2.json"
    first = shards.compose_seed_pair(primary, secondary, receipts, first_path)
    second = shards.compose_seed_pair(primary, secondary, receipts, second_path)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first == second
    assert first["planned_hands"] == {"primary": 6750, "secondary": 5100}
    assert [row["task_id"] for row in first["task_union"]["primary"]] == list(
        shards.EXPECTED_TASKS["primary"]
    )

    with pytest.raises(ConfigError, match="high/low execution|task union"):
        shards.compose_seed_pair(
            primary,
            secondary,
            receipts[:-1],
            tmp_path / "approved_receipts" / "composition-gap.json",
        )


def test_composition_rejects_duplicate_and_post_receipt_tamper(
    tmp_path: Path,
) -> None:
    primary = _canonical_manifest(tmp_path, role="primary")
    secondary = _canonical_manifest(tmp_path, role="secondary")
    p_a = _completed_shard(
        tmp_path,
        primary,
        selected=["isolation_no_memory", "isolation_fact", "isolation_expr"],
        shard_id="p-a",
    )
    p_b = _completed_shard(
        tmp_path,
        primary,
        selected=["isolation_sync"],
        shard_id="p-sync",
    )
    p_low = _completed_shard(
        tmp_path,
        primary,
        selected=["isolation_async"],
        shard_id="p-async",
    )
    s_a = _completed_shard(
        tmp_path,
        secondary,
        selected=["mixed_ecological"],
        shard_id="s-a",
    )
    s_b = _completed_shard(
        tmp_path,
        secondary,
        selected=["expr_online", "expr_without", "async_online", "async_without"],
        shard_id="s-b",
    )
    with pytest.raises(ConfigError, match="shard 重复"):
        shards.compose_seed_pair(
            primary,
            secondary,
            [p_a, p_b, p_low, s_a, s_b, p_a],
            tmp_path / "approved_receipts" / "duplicate.json",
        )

    receipt = json.loads(p_b.read_text(encoding="utf-8"))
    attempt = Path(receipt["attempt_root"])
    child = attempt / "runs" / "isolation_sync"
    (child / "hand_summaries.jsonl").write_text(
        "{}\n" * 1349,
        encoding="utf-8",
    )
    marker_path = attempt / "task_receipts" / "isolation_sync.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["files"] = shards._directory_manifest(child)
    _write_json(marker_path, marker)
    receipt["tasks"][0]["files"] = marker["files"]
    receipt["tasks"][0]["task_receipt_sha256"] = _sha(marker_path)
    _write_json(p_b, receipt)
    with pytest.raises(ConfigError, match="hash/hands|files.tsv"):
        shards.compose_seed_pair(
            primary,
            secondary,
            [p_a, p_b, p_low, s_a, s_b],
            tmp_path / "approved_receipts" / "tampered.json",
        )


def test_pair_mapping_is_fixed_to_six_formal_seeds() -> None:
    assert shards.PAIR_MAPPING == {
        2026090107: ("P07", "S07"),
        2026090108: ("P08", "S08"),
        2026090109: ("P09", "S09"),
        2026090110: ("P10", "S10"),
        2026090111: ("P11", "S11"),
        2026090112: ("P12", "S12"),
    }
    assert shards.PAIR_IDS == {
        seed: f"pair_{seed - 2026090100:02d}"
        for seed in range(2026090107, 2026090113)
    }
    assert shards.PHYSICAL_MAPPING[2026090107] == {
        "low": "H01",
        "high": "H07",
    }
    assert shards.PHYSICAL_MAPPING[2026090112] == {
        "low": "H06",
        "high": "H12",
    }


def test_authorization_rejects_wrong_side_physical_slot_and_amendment(
    tmp_path: Path,
) -> None:
    canonical = _canonical_manifest(tmp_path, role="primary")
    authorization = _authorization(
        tmp_path,
        canonical,
        selected=["isolation_sync"],
        shard_id="sync",
    )
    value = json.loads(authorization.read_text(encoding="utf-8"))
    value["shard_role"] = "low"
    value["physical_slot"] = shards.PHYSICAL_MAPPING[2026090107]["low"]
    _write_json(authorization, value)
    with pytest.raises(ConfigError, match="high/low task mapping"):
        shards.derive_authorized_manifest(canonical, authorization)

    value["shard_role"] = "high"
    value["physical_slot"] = shards.PHYSICAL_MAPPING[2026090107]["high"]
    value["amendment_sha256"] = "0" * 64
    _write_json(authorization, value)
    with pytest.raises(ConfigError, match="amendment SHA"):
        shards.derive_authorized_manifest(canonical, authorization)


@pytest.mark.parametrize("failure", ["wrong_head", "dirty", "runner_hash"])
def test_scientific_checkout_gate_fails_before_scientific_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    canonical = _canonical_manifest(tmp_path, role="primary")
    authorization = _authorization(
        tmp_path,
        canonical,
        selected=["isolation_sync"],
        shard_id=failure,
    )
    runner_path = (
        tmp_path
        / "scientific"
        / "src"
        / "agentmemeval"
        / "experiments"
        / "formal_runner.py"
    )
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("# deliberately not frozen\n", encoding="utf-8")

    def fake_git(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        if "rev-parse" in argv:
            stdout = (
                "0" * 40
                if failure == "wrong_head"
                else shards.FROZEN_CODE_SHA
            )
        else:
            stdout = " M tracked.py\n" if failure == "dirty" else ""
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(shards.subprocess, "run", fake_git)
    derived_output = (
        tmp_path / "approved_staging" / f"derived-{failure}.json"
    )
    with pytest.raises(ConfigError, match="HEAD/clean|formal_runner SHA"):
        shards.run_authorized_shard(
            canonical,
            authorization,
            derived_manifest_output=derived_output,
            receipt_root=tmp_path / "approved_receipts",
        )
    authorization_value = json.loads(authorization.read_text(encoding="utf-8"))
    assert not Path(authorization_value["derived_output_path"]).exists()
    assert not Path(authorization_value["derived_cache_namespace"]).exists()
    assert not derived_output.exists()


def test_build_receipt_rejects_actual_hands_and_completion_hash(
    tmp_path: Path,
) -> None:
    canonical = _canonical_manifest(tmp_path, role="primary")
    with pytest.raises(ConfigError, match="task receipt/hash"):
        _completed_shard(
            tmp_path,
            canonical,
            selected=["isolation_sync"],
            shard_id="bad-hands",
            hand_delta=-1,
        )

    other = tmp_path / "other"
    other.mkdir()
    canonical_other = _canonical_manifest(other, role="primary")
    with pytest.raises(ConfigError, match="completion/files.tsv"):
        _completed_shard(
            other,
            canonical_other,
            selected=["isolation_sync"],
            shard_id="bad-completion",
            valid_completion_hash=False,
        )


def _historical_adoption(
    tmp_path: Path,
    canonical_path: Path,
    *,
    selected: list[str],
    shard_id: str,
) -> tuple[Path, Path]:
    authorization = _authorization(
        tmp_path,
        canonical_path,
        selected=selected,
        shard_id=shard_id,
        seal_mode="historical_adoption",
    )
    derived = shards.derive_authorized_manifest(canonical_path, authorization)
    root = Path(derived["paired_shard"]["denied_partial_root"])
    for task in derived["task_configs"]:
        task_id = task["task_id"]
        child = root / "runs" / task_id
        _write_json(child / "experiment_result.json", {"status": "complete"})
        _write_json(
            child / "snapshots" / f"{task_id}_checkpoint_0300.json",
            {"checkpoint": 300, "task_id": task_id},
        )
        _write_json(child / "task_identity_audit.json", task["expected_identity"])
        (child / "hand_summaries.jsonl").write_text(
            "{}\n" * int(task["planned_hands"]),
            encoding="utf-8",
        )
        marker = {
            "schema_version": "task8-worker-task-receipt-v1",
            "task_id": task_id,
            "config_sha256": task["config_sha256"],
            "run_dir": f"runs/{task_id}",
            "task_row": {
                "task_id": task_id,
                "status": "complete",
                "identity_audit": {**task["expected_identity"], "code_dirty": False},
            },
            "files": shards._directory_manifest(child),
        }
        _write_json(root / "task_receipts" / f"{task_id}.json", marker)
    receipt_path = (
        tmp_path / "approved_receipts" / "historical" / f"{shard_id}.json"
    )
    shards.build_historical_adoption_receipt(
        canonical_path,
        authorization,
        root,
        receipt_path,
    )
    return receipt_path, authorization


def test_historical_adoption_accepts_complete_tasks_without_overall_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical_manifest(tmp_path, role="primary", seed=2026090112)
    receipt_path, authorization = _historical_adoption(
        tmp_path,
        canonical,
        selected=["isolation_no_memory", "isolation_fact"],
        shard_id="h12-history",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["source_kind"] == "historical_adoption"
    assert receipt["physical_slot"] == "H12"
    assert receipt["selected_task_ids"] == [
        "isolation_no_memory",
        "isolation_fact",
    ]
    assert not (Path(receipt["attempt_root"]) / "completion_receipt.json").exists()
    monkeypatch.setattr(
        shards,
        "_verify_scientific_checkout",
        lambda checkout: checkout / "src/agentmemeval/experiments/formal_runner.py",
    )
    with pytest.raises(ConfigError, match="拒绝 historical adoption"):
        shards.run_authorized_shard(
            canonical,
            authorization,
            derived_manifest_output=tmp_path / "approved_staging" / "adopt.json",
            receipt_root=tmp_path / "approved_receipts",
        )


def test_pair12_h12_history_h06_async_composes_primary_bridge_and_binds_s(
    tmp_path: Path,
) -> None:
    primary = _canonical_manifest(tmp_path, role="primary", seed=2026090112)
    secondary = _canonical_manifest(tmp_path, role="secondary", seed=2026090112)
    history, _authorization_path = _historical_adoption(
        tmp_path,
        primary,
        selected=["isolation_no_memory", "isolation_fact"],
        shard_id="h12-history",
    )
    h12_new = _completed_shard(
        tmp_path,
        primary,
        selected=["isolation_expr", "isolation_sync"],
        shard_id="h12-new",
    )
    h06_async = _completed_shard(
        tmp_path,
        primary,
        selected=["isolation_async"],
        shard_id="h06-async",
    )
    bridge_root = tmp_path / "approved_staging" / "bridges" / "pair12"
    bridge_receipt = tmp_path / "approved_receipts" / "receipts" / "P12.json"
    result = shards.compose_primary_checkpoint(
        primary,
        [history, h12_new, h06_async],
        bridge_root,
        bridge_receipt,
    )
    assert result["checkpoint_receipt"]["producer_worker_id"] == "P12"
    assert result["checkpoint_receipt"]["seed_bundle"] == 2026090112
    assert {row["shard_role"] for row in result["bridge_manifest"]["source_receipts"]} == {
        "high",
        "low",
    }
    checkpoint_paths = {
        row["relative_path"]
        for row in result["checkpoint_receipt"]["checkpoint_files"]
    }
    secondary_value = json.loads(secondary.read_text(encoding="utf-8"))
    for task in secondary_value["task_configs"]:
        for binding in task.get("checkpoint_bindings", {}).values():
            assert binding in checkpoint_paths
    formal_runner.verify_checkpoint_receipt(
        bridge_receipt,
        bridge_root,
        expected_identity=json.loads(
            primary.read_text(encoding="utf-8")
        )["receipt_identity"],
        expected_producer_worker_id="P12",
        expected_seed_bundle=2026090112,
        expected_checkpoint_hand=300,
    )

    authorization = _authorization(
        tmp_path / "secondary_auth",
        secondary,
        selected=["mixed_ecological"],
        shard_id="h06-mixed",
    )
    value = json.loads(authorization.read_text(encoding="utf-8"))
    value["approved_staging_root"] = str(
        (tmp_path / "approved_staging").resolve()
    )
    value["approved_receipt_root"] = str(
        (tmp_path / "approved_receipts").resolve()
    )
    value["derived_output_path"] = str(
        (tmp_path / "approved_staging" / "outputs/S12/h06-mixed").resolve()
    )
    value["derived_cache_namespace"] = str(
        (tmp_path / "approved_staging" / "cache/S12/h06-mixed").resolve()
    )
    value["derived_receipt_relative_path"] = "receipts/P12.json"
    value["primary_bridge_root"] = str(bridge_root.resolve())
    value["primary_bridge_receipt_relative_path"] = "receipts/P12.json"
    value["primary_bridge_receipt_sha256"] = _sha(bridge_receipt)
    _write_json(authorization, value)
    derived = shards.derive_authorized_manifest(secondary, authorization)
    assert derived["dependency_output_path"] == str(bridge_root.resolve())
    assert derived["receipt_relative_path"] == "receipts/P12.json"
    assert derived["paired_shard"]["physical_slot"] == "H06"


def test_reservation_rechecks_authorization_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical_manifest(tmp_path, role="primary")
    authorization = _authorization(
        tmp_path,
        canonical,
        selected=["isolation_sync"],
        shard_id="toctou",
    )
    original_reserve = shards._reserve_execution

    def reserve_then_mutate(root: Path, identity: dict[str, object]) -> Path:
        result = original_reserve(root, identity)
        value = json.loads(authorization.read_text(encoding="utf-8"))
        value["selected_task_ids"] = ["isolation_no_memory"]
        _write_json(authorization, value)
        return result

    monkeypatch.setattr(shards, "_reserve_execution", reserve_then_mutate)
    with pytest.raises(ConfigError, match="authorization_id identity binding"):
        shards.run_authorized_shard(
            canonical,
            authorization,
            derived_manifest_output=tmp_path / "approved_staging" / "derived.json",
            receipt_root=tmp_path / "approved_receipts",
        )


def test_frozen_runner_uses_isolated_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical_manifest(tmp_path, role="primary")
    authorization = _authorization(
        tmp_path,
        canonical,
        selected=["isolation_sync"],
        shard_id="isolated",
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        shards,
        "_verify_scientific_checkout",
        lambda checkout: checkout / "src/agentmemeval/experiments/formal_runner.py",
    )

    def fake_subprocess(argv: list[str], **kwargs: object) -> SimpleNamespace:
        observed["argv"] = argv
        observed["cwd"] = kwargs["cwd"]
        observed["env"] = kwargs["env"]
        return SimpleNamespace(stdout='{"status":"complete","run_dir":"exact"}\n')

    monkeypatch.setattr(shards.subprocess, "run", fake_subprocess)
    result = shards.run_authorized_shard(
        canonical,
        authorization,
        derived_manifest_output=tmp_path / "approved_staging" / "derived.json",
        receipt_root=tmp_path / "approved_receipts",
    )
    assert result["status"] == "complete"
    assert observed["argv"][0] == sys.executable
    assert observed["cwd"] == str((tmp_path / "scientific").resolve())
    assert "PYTHONPATH" not in observed["env"]
    assert observed["env"]["PYTHONNOUSERSITE"] == "1"
    bootstrap = Path(observed["argv"][1])
    assert "formal_runner import escaped" in bootstrap.read_text(encoding="utf-8")


def test_approved_path_rejects_symlink_component_before_resolve(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Windows symlink privilege is unavailable")
    with pytest.raises(ConfigError, match="symlink"):
        shards._inside_approved(
            root.resolve(),
            link / "child.json",
            label="test target",
        )


@pytest.mark.parametrize(
    ("seed", "physical_slot"),
    [
        (2026090108, "H02"),
        (2026090111, "H05"),
        (2026090112, "H06"),
    ],
)
def test_render_authorization_is_deterministic_for_low_primary_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
    physical_slot: str,
) -> None:
    canonical = _canonical_manifest(tmp_path, role="primary", seed=seed)
    amendment = tmp_path / "amendment.md"
    amendment.write_text("frozen amendment\n", encoding="utf-8")
    checkout = tmp_path / "scientific"
    staging = tmp_path / "staging"
    receipts = tmp_path / "receipts"
    checkout.mkdir()
    staging.mkdir()
    receipts.mkdir()
    monkeypatch.setattr(
        shards,
        "_verify_scientific_checkout",
        lambda path: Path(path) / "formal_runner.py",
    )
    first = tmp_path / "auth" / "first.json"
    second = tmp_path / "auth" / "second.json"
    kwargs = {
        "physical_slot": physical_slot,
        "shard_role": "low",
        "selected_task_ids": ["isolation_async"],
        "scientific_checkout": checkout,
        "approved_staging_root": staging,
        "approved_receipt_root": receipts,
        "shard_id": "primary-low-async-v1",
    }
    rendered = shards.render_authorization(
        canonical,
        amendment,
        output_path=first,
        **kwargs,
    )
    shards.render_authorization(
        canonical,
        amendment,
        output_path=second,
        **kwargs,
    )
    assert first.read_bytes() == second.read_bytes()
    assert rendered["worker_id"] == f"P{seed - 2026090100:02d}"
    assert rendered["physical_slot"] == physical_slot
    assert rendered["selected_task_ids"] == ["isolation_async"]
    assert rendered["effect_fields_read"] is False
    with pytest.raises(ConfigError, match="拒绝覆盖"):
        shards.render_authorization(
            canonical,
            amendment,
            output_path=first,
            **kwargs,
        )


def test_render_high_mapping_and_preflight_never_runs_hands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical_manifest(
        tmp_path,
        role="primary",
        seed=2026090107,
    )
    amendment = tmp_path / "amendment.md"
    amendment.write_text("frozen amendment\n", encoding="utf-8")
    checkout = tmp_path / "scientific"
    staging = tmp_path / "staging"
    receipts = tmp_path / "receipts"
    checkout.mkdir()
    staging.mkdir()
    receipts.mkdir()
    monkeypatch.setattr(
        shards,
        "_verify_scientific_checkout",
        lambda path: Path(path) / "formal_runner.py",
    )
    authorization = tmp_path / "auth" / "high.json"
    shards.render_authorization(
        canonical,
        amendment,
        physical_slot="H07",
        shard_role="high",
        selected_task_ids=["isolation_sync"],
        scientific_checkout=checkout,
        approved_staging_root=staging,
        approved_receipt_root=receipts,
        shard_id="primary-high-sync-rerun-v1",
        output_path=authorization,
    )
    called = False

    def _forbidden_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("preflight must not invoke run_authorized_shard")

    monkeypatch.setattr(shards, "run_authorized_shard", _forbidden_run)
    result = shards.preflight_authorization(canonical, authorization)
    assert result["status"] == "preflight_pass"
    assert result["physical_slot"] == "H07"
    assert result["selected_task_ids"] == ["isolation_sync"]
    assert result["hands_started"] == 0
    assert called is False
    assert not Path(result["derived_output_path"]).exists()
    assert not Path(result["derived_cache_namespace"]).exists()


def test_render_authorization_rejects_wrong_physical_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical_manifest(
        tmp_path,
        role="primary",
        seed=2026090108,
    )
    amendment = tmp_path / "amendment.md"
    amendment.write_text("frozen amendment\n", encoding="utf-8")
    checkout = tmp_path / "scientific"
    staging = tmp_path / "staging"
    receipts = tmp_path / "receipts"
    checkout.mkdir()
    staging.mkdir()
    receipts.mkdir()
    monkeypatch.setattr(
        shards,
        "_verify_scientific_checkout",
        lambda path: Path(path) / "formal_runner.py",
    )
    with pytest.raises(ConfigError, match="physical_slot"):
        shards.render_authorization(
            canonical,
            amendment,
            physical_slot="H05",
            shard_role="low",
            selected_task_ids=["isolation_async"],
            scientific_checkout=checkout,
            approved_staging_root=staging,
            approved_receipt_root=receipts,
            shard_id="wrong-slot",
            output_path=tmp_path / "auth.json",
        )


def _completed_recovery_fixture(
    tmp_path: Path,
    *,
    seed: int = 2026090107,
    selected: list[str] | None = None,
    recovered_task_id: str = "isolation_async",
    hands: int = 1350,
    fallback_count: int = 0,
    include_integer_key: bool = True,
    mapping_keys: dict[object, int] | None = None,
    extra_health_field: bool = False,
    extra_validity_field: bool = False,
    invalid_health_detail: bool = False,
    invalid_validity_detail: bool = False,
    extra_expected_field: bool = False,
    extra_receipt_identity_field: bool = False,
    receipt_config_override: str | None = None,
    legacy_controller: bool = False,
    archive_prefix: str | None = None,
) -> dict[str, Path]:
    selected = selected or [recovered_task_id]
    canonical = _canonical_manifest(tmp_path, role="primary", seed=seed)
    manifest = json.loads(canonical.read_text(encoding="utf-8"))
    prompts = {"system": "frozen"}
    model = {"model": "Qwen-frozen"}
    embedding = {"model": "BGE-frozen"}
    schedule_sha = hashlib.sha256(recovered_task_id.encode()).hexdigest()
    config: dict[str, object] = {
        "experiment": {"seed": seed},
        "agent": {},
    }
    if include_integer_key:
        config["experiment"]["checkpoint_test_hands_by_checkpoint"] = {  # type: ignore[index]
            30: 1,
            75: 1,
            150: 1,
            300: 1,
        }
    if mapping_keys is not None:
        config["experiment"]["checkpoint_test_hands_by_checkpoint"] = mapping_keys  # type: ignore[index]
    try:
        resolved_config_sha256 = formal_runner.sha256_json(
            shards._canonicalize_preserving_mapping_keys(config)
        )
    except TypeError:
        resolved_config_sha256 = "0" * 64
    expected = {
        "code_sha": shards.FROZEN_CODE_SHA,
        "prompt_sha256": formal_runner.sha256_json(prompts),
        "model_fingerprint": formal_runner.sha256_json(model),
        "embedding_fingerprint": formal_runner.task8b_embedding_fingerprint(
            embedding
        ),
        "resolved_config_sha256": resolved_config_sha256,
        "schedule_sha256": schedule_sha,
    }
    for field in formal_runner.FLEET_COMMON_IDENTITY_FIELDS:
        manifest["common_identity"][field] = expected[field]
    manifest["receipt_identity"] = {
        field: expected[field]
        for field in formal_runner.REQUIRED_IDENTITY_FIELDS
    }
    manifest["receipt_identity"]["resolved_config_sha256"] = (
        receipt_config_override or "6" * 64
    )
    if extra_receipt_identity_field:
        manifest["receipt_identity"]["unexpected"] = "forbidden"
    if extra_expected_field:
        expected["unexpected_identity_field"] = "forbidden"
    for task in manifest["task_configs"]:
        if task["task_id"] == recovered_task_id:
            task["expected_identity"] = expected
    _write_json(canonical, manifest)
    authorization = _authorization(
        tmp_path,
        canonical,
        selected=selected,
        shard_id=f"recover-{recovered_task_id}",
    )
    if legacy_controller:
        auth_value = json.loads(authorization.read_text(encoding="utf-8"))
        auth_value["engineering_controller_sha256"] = (
            shards.RECOVERY_SOURCE_CONTROLLER_SHA256
        )
        auth_value["authorization_id"] = shards._authorization_id(
            auth_value,
            role=manifest["role"],
        )
        _write_json(authorization, auth_value)
        derived = shards.derive_authorized_manifest(
            canonical,
            authorization,
            _recovery_source_controller_sha256=(
                shards.RECOVERY_SOURCE_CONTROLLER_SHA256
            ),
        )
    else:
        derived = shards.derive_authorized_manifest(canonical, authorization)
    derived_path = tmp_path / "derived" / "worker.json"
    _write_json(derived_path, derived)
    root = Path(derived["instance_identity"]["output_path"])
    child = root / "runs" / recovered_task_id
    child.mkdir(parents=True)
    _write_json(
        root / "worker_manifest.json",
        derived,
    )
    failed_hash = _failed_state(root / "state.tsv")
    _write_json(
        child / "manifest.json",
        {
            "metadata": {
                "code": {"commit": shards.FROZEN_CODE_SHA, "dirty": False},
                "prompts": prompts,
                "model": model,
                "embedding": embedding,
            }
        },
    )
    (child / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True),
        encoding="utf-8",
    )
    _write_json(
        child / "schedule_manifest.json",
        {"schedule_sha256": schedule_sha},
    )
    _write_json(
        child / "experiment_result.json",
        {
            "run_id": f"fixture-{recovered_task_id}",
            "scenario": "checkpoint_generalization",
            "metrics": {"effect_direction": "must-not-read"},
            "aggregate_metrics": {},
            "artifacts": {},
            "notes": [],
        },
    )
    execution_health = {
        "valid": True,
        **{
            field: (
                fallback_count if field == "fallback_count" else 0
            )
            for field in shards.HEALTH_ZERO_FIELDS
        },
        "reward_conservation_violation_hand_ids": (
            ["fixture-violation"] if invalid_health_detail else []
        ),
        "stack_conservation_violation_hand_ids": [],
        "status": "passed",
    }
    if extra_health_field:
        execution_health["unexpected_counter"] = 0
    _write_json(
        child / "protocol_audit.json",
        {
            "run_validity": {
                "execution_valid": True,
                "behavior_valid": True,
                "paper_eligible": True,
                "run_mode": "formal",
                "status": (
                    "not_for_main_table"
                    if invalid_validity_detail
                    else "valid_for_main_table"
                ),
                **({"effect_like_extra": 1} if extra_validity_field else {}),
            },
            "execution_health": execution_health,
            "forbidden_effect_value": 999,
        },
    )
    (child / "events.jsonl").write_bytes(b"{}\n")
    (child / "hand_summaries.jsonl").write_bytes(b"{}\n" * hands)
    _write_json(child / "metrics.json", {"effect_direction": "must-not-read"})
    archive = tmp_path / "pre-recovery.tar.gz"
    with tarfile.open(archive, mode="w:gz") as handle:
        for row in shards._directory_manifest(root):
            relative = row["relative_path"]
            handle.add(
                root / relative,
                arcname=(
                    f"{archive_prefix}/{relative}"
                    if archive_prefix
                    else relative
                ),
                recursive=False,
            )
    auth = json.loads(authorization.read_text(encoding="utf-8"))
    receipt_root = Path(auth["approved_receipt_root"])
    baseline = receipt_root / "recovery" / "baseline.json"
    baseline_body = {
        "schema_version": shards.COMPLETED_RECOVERY_BASELINE_SCHEMA,
        "status": "activated",
        "recovery_id": f"recovery-{seed}-{recovered_task_id}",
        "reason": shards.COMPLETED_RECOVERY_REASON,
        "created_at_utc": "2026-07-24T00:01:00Z",
        "worker_id": manifest["worker_id"],
        "seed": seed,
        "task_id": recovered_task_id,
        "attempt_root": str(root),
        "derived_manifest_sha256": _sha(derived_path),
        "authorization_sha256": _sha(authorization),
        "recovery_tool_sha256": _sha(Path(shards.__file__).resolve()),
        "pre_recovery_files": shards._directory_manifest(root),
        "pre_recovery_archive_path": str(archive.resolve()),
        "pre_recovery_archive_sha256": _sha(archive),
        "failed_state_row_sha256": failed_hash,
        "effect_fields_read": False,
    }
    _write_json(baseline, baseline_body)
    return {
        "canonical": canonical,
        "authorization": authorization,
        "derived": derived_path,
        "root": root,
        "archive": archive,
        "baseline": baseline,
        "certificate": receipt_root / "recovery" / "certificate.json",
        "ledger": receipt_root / "recovery" / "ledger-entry.json",
        "shard_receipt": receipt_root / "recovery" / "shard.json",
    }


def _run_completed_recovery(paths: dict[str, Path]) -> dict[str, object]:
    return shards.recover_completed_execution(
        paths["derived"],
        paths["root"],
        paths["baseline"],
        paths["archive"],
        paths["certificate"],
        paths["ledger"],
    )


def _mock_frozen_standard_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    def resume(
        manifest_path: Path,
        *,
        scientific_checkout: Path,
        receipt_root: Path,
        legacy_audit_path: Path | None = None,
        legacy_audit_sha256: str | None = None,
        legacy_receipt_config_sha256: str | None = None,
        corrected_receipt_config_sha256: str | None = None,
    ) -> dict[str, object]:
        del (
            scientific_checkout,
            receipt_root,
            legacy_audit_path,
            legacy_audit_sha256,
            legacy_receipt_config_sha256,
            corrected_receipt_config_sha256,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = Path(manifest["instance_identity"]["output_path"])
        if (root / "completion_receipt.json").exists():
            return {"status": "complete", "resumed": True}
        for task in manifest["task_configs"]:
            task_id = task["task_id"]
            marker_path = root / "task_receipts" / f"{task_id}.json"
            if marker_path.exists():
                continue
            child = root / "runs" / task_id
            child.mkdir(parents=True)
            _write_json(child / "experiment_result.json", {"status": "complete"})
            (child / "hand_summaries.jsonl").write_bytes(
                b"{}\n" * int(task["planned_hands"])
            )
            _write_json(
                child / "execution_provenance.json",
                {"executed_by": "frozen-runner-resume", "task_id": task_id},
            )
            identity = {**task["expected_identity"], "code_dirty": False}
            row = {
                "task_id": task_id,
                "memory_mode": task.get("memory_mode"),
                "run_dir": f"runs/{task_id}",
                "cache_namespace": "frozen-resume",
                "identity_audit": identity,
                "status": "complete",
            }
            _write_json(
                marker_path,
                {
                    "schema_version": "task8-worker-task-receipt-v1",
                    "task_id": task_id,
                    "config_sha256": task["config_sha256"],
                    "run_dir": row["run_dir"],
                    "task_row": row,
                    "files": shards._directory_manifest(child),
                },
            )
        states = shards._state_rows(root / "state.tsv")
        previous = states[-1]["row_sha256"]
        with (root / "state.tsv").open("ab") as handle:
            for status in ("validating", "running", "finalizing", "complete"):
                body = {
                    "schema_version": states[-1]["schema_version"],
                    "created_at_utc": "2026-07-24T00:02:00Z",
                    "status": status,
                    "detail": "frozen runner resume",
                    "previous_sha256": previous,
                }
                row = {**body, "row_sha256": formal_runner.sha256_json(body)}
                handle.write(
                    (
                        "\t".join(str(row[field]) for field in row) + "\n"
                    ).encode("utf-8")
                )
                previous = row["row_sha256"]
        (root / "files.tsv").write_bytes(shards._files_tsv_bytes(root))
        _write_json(
            root / "completion_receipt.json",
            {
                "schema_version": "task8-worker-completion-v1",
                "worker_id": manifest["worker_id"],
                "status": "complete",
                "files_tsv_sha256": _sha(root / "files.tsv"),
            },
        )
        return {"status": "complete", "resumed": True}

    monkeypatch.setattr(shards, "_resume_recovered_execution", resume)


@pytest.mark.parametrize(
    ("seed", "task_id"),
    [
        (2026090107, "isolation_async"),
        (2026090107, "isolation_sync"),
    ],
)
def test_recover_completed_execution_is_idempotent_and_sealable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
    task_id: str,
) -> None:
    _mock_frozen_standard_resume(monkeypatch)
    paths = _completed_recovery_fixture(
        tmp_path,
        seed=seed,
        recovered_task_id=task_id,
    )
    first = _run_completed_recovery(paths)
    task_audit_path = (
        paths["root"] / "runs" / task_id / "task_identity_audit.json"
    )
    task_audit = json.loads(task_audit_path.read_text(encoding="utf-8"))
    marker = json.loads(
        (
            paths["root"] / "task_receipts" / f"{task_id}.json"
        ).read_text(encoding="utf-8")
    )
    expected_standard_audit = {
        "schema_version": "task8-task-identity-audit-v1",
        "task_id": task_id,
        "protocol_status": json.loads(
            paths["derived"].read_text(encoding="utf-8")
        )["protocol_status"],
        "actual": marker["task_row"]["identity_audit"],
        "status": "verified",
    }
    assert task_audit == expected_standard_audit
    assert set(marker["task_row"]["identity_audit"]) == {
        *formal_runner.REQUIRED_IDENTITY_FIELDS,
        "code_dirty",
    }
    recovery_audit_path = paths["certificate"].with_name(
        f"{paths['certificate'].stem}.identity_correction_audit.json"
    )
    recovery_audit = json.loads(
        recovery_audit_path.read_text(encoding="utf-8")
    )
    certificate = json.loads(
        paths["certificate"].read_text(encoding="utf-8")
    )
    assert recovery_audit["status"] == "verified-recovered"
    assert recovery_audit["effect_fields_read"] is False
    assert certificate["identity_correction_audit_path"] == str(
        recovery_audit_path
    )
    assert certificate["identity_correction_audit_sha256"] == _sha(
        recovery_audit_path
    )
    checkpoint_correction = {
        "applied": True,
        "old_resolved_config_sha256": "6" * 64,
        "new_resolved_config_sha256": marker["task_row"][
            "identity_audit"
        ]["resolved_config_sha256"],
        "only_authorized_field": "resolved_config_sha256",
    }
    assert (
        recovery_audit["checkpoint_receipt_identity_correction"]
        == checkpoint_correction
    )
    assert (
        certificate["checkpoint_receipt_identity_correction"]
        == checkpoint_correction
    )
    frozen = {
        path: path.read_bytes()
        for path in (
            paths["certificate"],
            paths["ledger"],
            recovery_audit_path,
            task_audit_path,
            paths["root"] / "state.tsv",
            paths["root"] / "files.tsv",
            paths["root"] / "completion_receipt.json",
        )
    }
    second = _run_completed_recovery(paths)
    assert first == second
    assert all(path.read_bytes() == content for path, content in frozen.items())
    assert first["shard_closed"] is True
    state = (paths["root"] / "state.tsv").read_text(encoding="utf-8")
    assert "\tfailed\t" in state
    assert "\tvalidating\t" in state
    assert "\tcomplete\t" in state
    receipt = shards.build_shard_receipt(
        paths["derived"],
        paths["root"],
        paths["shard_receipt"],
    )
    assert receipt["selected_task_ids"] == [task_id]
    assert receipt["effect_fields_read"] is False


def test_recovery_preserves_exact_legacy_custom_identity_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_frozen_standard_resume(monkeypatch)
    paths = _completed_recovery_fixture(tmp_path)
    manifest = json.loads(paths["derived"].read_text(encoding="utf-8"))
    task = next(
        row
        for row in manifest["task_configs"]
        if row["task_id"] == "isolation_async"
    )
    child = paths["root"] / "runs" / "isolation_async"
    actual, health, correction = shards._recovery_identity_and_health(
        manifest,
        task,
        child,
    )
    legacy = {
        "schema_version": "task8-task-identity-audit-v1",
        "task_id": "isolation_async",
        "protocol_status": manifest["protocol_status"],
        "actual": actual,
        "expected": task["expected_identity"],
        "identity_correction": correction,
        "health": health,
        "status": "verified-recovered",
        "effect_fields_read": False,
    }
    audit_path = child / "task_identity_audit.json"
    _write_json(audit_path, legacy)
    legacy_bytes = audit_path.read_bytes()
    baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
    baseline["pre_recovery_files"] = shards._directory_manifest(paths["root"])
    with tarfile.open(paths["archive"], mode="w:gz") as handle:
        for row in baseline["pre_recovery_files"]:
            relative = row["relative_path"]
            handle.add(
                paths["root"] / relative,
                arcname=relative,
                recursive=False,
            )
    baseline["pre_recovery_archive_sha256"] = _sha(paths["archive"])
    _write_json(paths["baseline"], baseline)

    result = _run_completed_recovery(paths)

    assert result["shard_closed"] is True
    assert audit_path.read_bytes() == legacy_bytes
    marker = json.loads(
        (
            paths["root"]
            / "task_receipts"
            / "isolation_async.json"
        ).read_text(encoding="utf-8")
    )
    assert set(marker["task_row"]["identity_audit"]) == {
        *formal_runner.REQUIRED_IDENTITY_FIELDS,
        "code_dirty",
    }


def test_recover_completed_execution_rejects_incomplete_or_unhealthy_data(
    tmp_path: Path,
) -> None:
    short = _completed_recovery_fixture(tmp_path / "short", hands=1349)
    with pytest.raises(ConfigError, match="hands"):
        _run_completed_recovery(short)
    unhealthy = _completed_recovery_fixture(
        tmp_path / "unhealthy",
        fallback_count=1,
    )
    with pytest.raises(ConfigError, match="health counter"):
        _run_completed_recovery(unhealthy)
    no_key_difference = _completed_recovery_fixture(
        tmp_path / "no-key-difference",
        include_integer_key=False,
    )
    with pytest.raises(ConfigError, match="canonicalization"):
        _run_completed_recovery(no_key_difference)


def test_recover_completed_execution_rejects_baseline_or_scientific_tamper(
    tmp_path: Path,
) -> None:
    paths = _completed_recovery_fixture(tmp_path)
    baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
    baseline["authorization_sha256"] = "0" * 64
    _write_json(paths["baseline"], baseline)
    with pytest.raises(ConfigError, match="authorization binding"):
        _run_completed_recovery(paths)

    paths = _completed_recovery_fixture(tmp_path / "scientific")
    with (paths["root"] / "runs" / "isolation_async" / "events.jsonl").open(
        "ab"
    ) as handle:
        handle.write(b"{}\n")
    with pytest.raises(ConfigError, match="file manifest"):
        _run_completed_recovery(paths)


def test_h12_expr_recovery_does_not_forge_sync_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _completed_recovery_fixture(
        tmp_path,
        seed=2026090112,
        selected=["isolation_expr", "isolation_sync"],
        recovered_task_id="isolation_expr",
    )
    def interrupted_resume(
        manifest_path: Path,
        *,
        scientific_checkout: Path,
        receipt_root: Path,
        legacy_audit_path: Path | None = None,
        legacy_audit_sha256: str | None = None,
        legacy_receipt_config_sha256: str | None = None,
        corrected_receipt_config_sha256: str | None = None,
    ) -> dict[str, object]:
        del (
            manifest_path,
            scientific_checkout,
            receipt_root,
            legacy_audit_path,
            legacy_audit_sha256,
            legacy_receipt_config_sha256,
            corrected_receipt_config_sha256,
        )
        raise ConfigError("simulated interruption before frozen Sync execution")

    monkeypatch.setattr(
        shards,
        "_resume_recovered_execution",
        interrupted_resume,
    )
    with pytest.raises(ConfigError, match="before frozen Sync"):
        _run_completed_recovery(paths)
    assert (
        paths["root"] / "task_receipts" / "isolation_expr.json"
    ).is_file()
    assert not (
        paths["root"] / "task_receipts" / "isolation_sync.json"
    ).exists()
    assert not (paths["root"] / "completion_receipt.json").exists()
    assert "resume_existing=True" in shards._FROZEN_RECOVERY_RESUME_BOOTSTRAP


@pytest.mark.parametrize(
    "mapping_keys",
    [
        {True: 1},
        {1.5: 1},
        {1: 1, "1": 2},
    ],
)
def test_recovery_rejects_non_integer_or_colliding_mapping_keys(
    tmp_path: Path,
    mapping_keys: dict[object, int],
) -> None:
    paths = _completed_recovery_fixture(
        tmp_path,
        mapping_keys=mapping_keys,
    )
    with pytest.raises(ConfigError, match="mapping key|collision"):
        _run_completed_recovery(paths)


def test_recovery_requires_exact_identity_and_health_field_sets(
    tmp_path: Path,
) -> None:
    identity = _completed_recovery_fixture(
        tmp_path / "identity",
        extra_expected_field=True,
    )
    with pytest.raises(ConfigError, match="expected_identity 字段集合"):
        _run_completed_recovery(identity)
    receipt_identity = _completed_recovery_fixture(
        tmp_path / "receipt-identity",
        extra_receipt_identity_field=True,
    )
    with pytest.raises(ConfigError, match="receipt_identity 字段集合"):
        _run_completed_recovery(receipt_identity)
    non_hex_receipt = _completed_recovery_fixture(
        tmp_path / "non-hex-receipt",
        receipt_config_override="z" * 64,
    )
    with pytest.raises(ConfigError, match="receipt config identity"):
        _run_completed_recovery(non_hex_receipt)
    health = _completed_recovery_fixture(
        tmp_path / "health",
        extra_health_field=True,
    )
    with pytest.raises(ConfigError, match="execution_health exact-key"):
        _run_completed_recovery(health)
    validity = _completed_recovery_fixture(
        tmp_path / "validity",
        extra_validity_field=True,
    )
    with pytest.raises(ConfigError, match="run_validity exact-key"):
        _run_completed_recovery(validity)
    invalid_validity = _completed_recovery_fixture(
        tmp_path / "invalid-validity",
        invalid_validity_detail=True,
    )
    with pytest.raises(ConfigError, match="run_validity gate"):
        _run_completed_recovery(invalid_validity)
    invalid_health = _completed_recovery_fixture(
        tmp_path / "invalid-health",
        invalid_health_detail=True,
    )
    with pytest.raises(ConfigError, match="execution_health detail"):
        _run_completed_recovery(invalid_health)


@pytest.mark.parametrize("mode", ["missing", "wrong"])
def test_recovery_baseline_requires_current_recovery_tool_sha(
    tmp_path: Path,
    mode: str,
) -> None:
    paths = _completed_recovery_fixture(tmp_path)
    baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
    if mode == "missing":
        baseline.pop("recovery_tool_sha256")
    else:
        baseline["recovery_tool_sha256"] = "0" * 64
    _write_json(paths["baseline"], baseline)
    with pytest.raises(ConfigError, match="字段集合|authorization binding"):
        _run_completed_recovery(paths)


def test_recovery_archive_rejects_path_escape_and_symlink(
    tmp_path: Path,
) -> None:
    for kind in ("escape", "symlink"):
        paths = _completed_recovery_fixture(tmp_path / kind)
        with tarfile.open(paths["archive"], mode="w:gz") as handle:
            info = tarfile.TarInfo(
                "../escape" if kind == "escape" else "state.tsv"
            )
            if kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "../outside"
                handle.addfile(info)
            else:
                payload = b"x"
                info.size = len(payload)
                handle.addfile(info, io.BytesIO(payload))
        baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
        baseline["pre_recovery_archive_sha256"] = _sha(paths["archive"])
        _write_json(paths["baseline"], baseline)
        with pytest.raises(ConfigError, match="archive member"):
            _run_completed_recovery(paths)


def test_recovery_archive_accepts_only_frozen_output_root_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_frozen_standard_resume(monkeypatch)
    frozen = _completed_recovery_fixture(
        tmp_path / "output-prefix",
        archive_prefix="output",
    )
    assert _run_completed_recovery(frozen)["shard_closed"] is True

    other = _completed_recovery_fixture(
        tmp_path / "other-prefix",
        archive_prefix="other",
    )
    with pytest.raises(ConfigError, match="archive 与 baseline files 不闭合"):
        _run_completed_recovery(other)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("mixed", "root prefix 不一致"),
        ("normalized-duplicate", "type/duplicate 非法"),
    ],
)
def test_recovery_archive_rejects_mixed_or_normalized_duplicate_paths(
    tmp_path: Path,
    mode: str,
    message: str,
) -> None:
    paths = _completed_recovery_fixture(tmp_path / mode)
    rows = shards._directory_manifest(paths["root"])
    first = rows[0]["relative_path"]
    second = rows[1]["relative_path"]
    with tarfile.open(paths["archive"], mode="w:gz") as handle:
        handle.add(
            paths["root"] / first,
            arcname=f"output/{first}",
            recursive=False,
        )
        if mode == "mixed":
            handle.add(
                paths["root"] / second,
                arcname=second,
                recursive=False,
            )
        else:
            handle.add(
                paths["root"] / first,
                arcname=f"output/{first}",
                recursive=False,
            )
    baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
    baseline["pre_recovery_archive_sha256"] = _sha(paths["archive"])
    _write_json(paths["baseline"], baseline)
    with pytest.raises(ConfigError, match=message):
        _run_completed_recovery(paths)


def test_recovery_accepts_only_fixed_legacy_controller_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_frozen_standard_resume(monkeypatch)
    paths = _completed_recovery_fixture(
        tmp_path,
        legacy_controller=True,
    )
    result = _run_completed_recovery(paths)
    assert result["shard_closed"] is True
    with pytest.raises(ConfigError, match="identity/active"):
        shards.derive_authorized_manifest(
            paths["canonical"],
            paths["authorization"],
        )
    with pytest.raises(ConfigError, match="source controller"):
        shards.derive_authorized_manifest(
            paths["canonical"],
            paths["authorization"],
            _recovery_source_controller_sha256="0" * 64,
        )


def test_frozen_recovery_resume_runs_in_isolated_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "frozen"
    package = checkout / "src" / "agentmemeval" / "experiments"
    package.mkdir(parents=True)
    (checkout / "src" / "agentmemeval" / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    trace = tmp_path / "resume-trace.json"
    (package / "formal_runner.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "_semantic_config = None\n"
        "def _write_json_same_or_new(path, value):\n"
        "    del path, value\n"
        "def _receipt_identity(manifest, *, consumer=False):\n"
        "    del consumer\n"
        "    return dict(manifest.get('receipt_identity', {}))\n"
        "def run_worker_manifest(path, *, receipt_root, resume_existing):\n"
        "    manifest = json.loads(path.read_text(encoding='utf-8'))\n"
        f"    Path({str(trace)!r}).write_text("
        "json.dumps({'manifest': str(path), "
        "'receipt_root': str(receipt_root), "
        "'resume_existing': resume_existing, "
        "'receipt_config': _receipt_identity(manifest)"
        ".get('resolved_config_sha256'), "
        "'canonicalizer_patched': callable(_semantic_config)}), "
        "encoding='utf-8')\n"
        "    return {'status': 'subprocess-pass'}\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "derived.json"
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    _write_json(
        manifest,
        {
            "worker_id": "P12",
            "receipt_identity": {"resolved_config_sha256": "6" * 64},
        },
    )
    monkeypatch.setattr(
        shards,
        "_verify_scientific_checkout",
        lambda _path: package / "formal_runner.py",
    )
    manifest_before = manifest.read_bytes()
    result = shards._resume_recovered_execution(
        manifest,
        scientific_checkout=checkout.resolve(),
        receipt_root=receipt_root.resolve(),
        legacy_receipt_config_sha256="6" * 64,
        corrected_receipt_config_sha256="7" * 64,
    )
    observed = json.loads(trace.read_text(encoding="utf-8"))
    assert result == {"status": "subprocess-pass"}
    assert observed["resume_existing"] is True
    assert observed["canonicalizer_patched"] is True
    assert observed["receipt_config"] == "7" * 64
    assert Path(observed["manifest"]) == manifest
    assert manifest.read_bytes() == manifest_before

    already_new = json.loads(manifest.read_text(encoding="utf-8"))
    already_new["receipt_identity"]["resolved_config_sha256"] = "7" * 64
    _write_json(manifest, already_new)
    already_new_bytes = manifest.read_bytes()
    result = shards._resume_recovered_execution(
        manifest,
        scientific_checkout=checkout.resolve(),
        receipt_root=receipt_root.resolve(),
        legacy_receipt_config_sha256="6" * 64,
        corrected_receipt_config_sha256="7" * 64,
    )
    observed = json.loads(trace.read_text(encoding="utf-8"))
    assert result == {"status": "subprocess-pass"}
    assert observed["receipt_config"] == "7" * 64
    assert manifest.read_bytes() == already_new_bytes

    third = json.loads(manifest.read_text(encoding="utf-8"))
    third["receipt_identity"]["resolved_config_sha256"] = "8" * 64
    _write_json(manifest, third)
    with pytest.raises(ConfigError, match="frozen resume failed"):
        shards._resume_recovered_execution(
            manifest,
            scientific_checkout=checkout.resolve(),
            receipt_root=receipt_root.resolve(),
            legacy_receipt_config_sha256="6" * 64,
            corrected_receipt_config_sha256="7" * 64,
        )

    with pytest.raises(ConfigError, match="bridge SHA 非法"):
        shards._resume_recovered_execution(
            manifest,
            scientific_checkout=checkout.resolve(),
            receipt_root=receipt_root.resolve(),
            legacy_receipt_config_sha256="7" * 64,
            corrected_receipt_config_sha256="7" * 64,
        )


def test_frozen_resume_bridge_preserves_exact_legacy_recovery_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "frozen"
    package = checkout / "src" / "agentmemeval" / "experiments"
    package.mkdir(parents=True)
    (checkout / "src" / "agentmemeval" / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "formal_runner.py").write_text(
        "import json\n"
        "_semantic_config = None\n"
        "def _receipt_identity(manifest, *, consumer=False):\n"
        "    del consumer\n"
        "    return dict(manifest.get('receipt_identity', {}))\n"
        "def _write_json_same_or_new(path, value):\n"
        "    content = json.dumps(value, ensure_ascii=False, "
        "sort_keys=True, separators=(',', ':')).encode('utf-8') + b'\\n'\n"
        "    if path.exists() and path.read_bytes() != content:\n"
        "        raise RuntimeError('resume refused evidence rewrite')\n"
        "    if not path.exists():\n"
        "        path.write_bytes(content)\n"
        "def run_worker_manifest(path, *, receipt_root, resume_existing):\n"
        "    del receipt_root\n"
        "    payload = json.loads(path.read_text(encoding='utf-8'))\n"
        "    from pathlib import Path\n"
        "    audit_path = Path(payload['audit_path'])\n"
        "    if payload.get('tamper_before_write'):\n"
        "        audit_path.write_text("
        "json.dumps({'tampered': True}), encoding='utf-8')\n"
        "    _write_json_same_or_new("
        "audit_path, payload['standard_audit'])\n"
        "    return {'status': 'bridge-pass', "
        "'resume_existing': resume_existing}\n",
        encoding="utf-8",
    )
    actual = {
        "code_sha": shards.FROZEN_CODE_SHA,
        "code_dirty": False,
        "resolved_config_sha256": "1" * 64,
        "prompt_sha256": "2" * 64,
        "model_fingerprint": "3" * 64,
        "embedding_fingerprint": "4" * 64,
        "schedule_sha256": "5" * 64,
    }
    standard = {
        "schema_version": "task8-task-identity-audit-v1",
        "task_id": "isolation_sync",
        "protocol_status": "frozen/expedited-formal-candidate",
        "actual": actual,
        "status": "verified",
    }
    legacy = {
        **standard,
        "expected": {
            field: actual[field]
            for field in formal_runner.REQUIRED_IDENTITY_FIELDS
        },
        "identity_correction": {"only": "mapping-keys"},
        "health": {"execution_valid": True},
        "status": "verified-recovered",
        "effect_fields_read": False,
    }
    audit_path = tmp_path / "task_identity_audit.json"
    _write_json(audit_path, legacy)
    legacy_bytes = audit_path.read_bytes()
    manifest = tmp_path / "derived.json"
    _write_json(
        manifest,
        {"audit_path": str(audit_path), "standard_audit": standard},
    )
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    monkeypatch.setattr(
        shards,
        "_verify_scientific_checkout",
        lambda _path: package / "formal_runner.py",
    )

    result = shards._resume_recovered_execution(
        manifest,
        scientific_checkout=checkout.resolve(),
        receipt_root=receipt_root.resolve(),
        legacy_audit_path=audit_path.resolve(),
        legacy_audit_sha256=_sha(audit_path),
    )

    assert result == {"status": "bridge-pass", "resume_existing": True}
    assert audit_path.read_bytes() == legacy_bytes

    _write_json(
        manifest,
        {
            "audit_path": str(audit_path),
            "standard_audit": standard,
            "tamper_before_write": True,
        },
    )
    with pytest.raises(ConfigError, match="frozen resume failed"):
        shards._resume_recovered_execution(
            manifest,
            scientific_checkout=checkout.resolve(),
            receipt_root=receipt_root.resolve(),
            legacy_audit_path=audit_path.resolve(),
            legacy_audit_sha256=_sha(audit_path),
        )

    _write_json(audit_path, legacy)
    other_audit = tmp_path / "other_task_identity_audit.json"
    _write_json(other_audit, legacy)
    _write_json(
        manifest,
        {"audit_path": str(other_audit), "standard_audit": standard},
    )
    with pytest.raises(ConfigError, match="frozen resume failed"):
        shards._resume_recovered_execution(
            manifest,
            scientific_checkout=checkout.resolve(),
            receipt_root=receipt_root.resolve(),
            legacy_audit_path=audit_path.resolve(),
            legacy_audit_sha256=_sha(audit_path),
        )
