from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        "authorization_id": f"auth-{worker}-{shard_id}",
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
    with pytest.raises(ConfigError, match="changed after reservation"):
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
