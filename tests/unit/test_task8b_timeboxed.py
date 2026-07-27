from pathlib import Path

from agentmemeval.config.loader import load_config
from agentmemeval.experiments.task8b_timeboxed import (
    FORMAL_SEEDS,
    _bgem3_health_candidates,
    build_task_config,
    render_worker_manifest,
    worker_plan,
)

BASE = Path("configs/formal/task8b_timeboxed_3seed_qwen_bgem3.yaml")


def test_bgem3_health_probe_supports_scoring_route_base() -> None:
    assert _bgem3_health_candidates("http://127.0.0.1:8001/v1/bgem3") == [
        "http://127.0.0.1:8001/v1/bgem3/health",
        "http://127.0.0.1:8001/health",
    ]


def test_timeboxed_shards_preserve_order_and_documented_budgets(tmp_path: Path) -> None:
    shard_a = worker_plan(seed=FORMAL_SEEDS[0], shard="A")
    shard_b = worker_plan(seed=FORMAL_SEEDS[0], shard="B")

    assert [row["task_id"] for row in shard_a] == [
        "isolation_fact",
        "isolation_expr",
        "expr_online",
        "expr_without",
        "isolation_no_memory",
    ]
    assert [row["task_id"] for row in shard_b] == [
        "isolation_sync",
        "isolation_async",
        "async_online",
        "async_without",
        "mixed_ecological",
    ]
    assert sum(row["planned_hands"] for row in shard_a) == 2520
    assert sum(row["planned_hands"] for row in shard_b) == 2550

    manifest = render_worker_manifest(
        base_config_path=BASE,
        output_root=tmp_path,
        host_id="AME1",
        seed=FORMAL_SEEDS[0],
        shard="A",
    )
    assert manifest["decision_model"] == "qwen/qwen3.5-9b"
    assert manifest["embedding_model"] == "BAAI/bge-m3"
    assert manifest["planned_hands"] == 2520


def test_timeboxed_configs_use_local_qwen_and_bgem3(tmp_path: Path) -> None:
    base = load_config(BASE)
    isolation = build_task_config(
        base,
        task_id="isolation_expr",
        seed=FORMAL_SEEDS[1],
        worker_root=tmp_path,
        run_id="isolation_expr",
        source_run_dir=None,
    )
    assert isolation["provider"]["model"] == "qwen/qwen3.5-9b"
    assert isolation["provider"]["base_url_env"] == "LOCAL_LLM_BASE_URL"
    assert isolation["agent"]["embedding_backend"] == "bgem3_hybrid_http"
    assert isolation["agent"]["embedding_model"] == "BAAI/bge-m3"
    assert isolation["experiment"]["checkpoint_set"] == [30, 90, 150, 270]
    assert isolation["experiment"]["checkpoint_test_hands_by_checkpoint"] == {
        30: 30,
        90: 30,
        150: 30,
        270: 60,
    }


def test_timeboxed_comparison_uses_only_local_checkpoint_270(tmp_path: Path) -> None:
    base = load_config(BASE)
    source = tmp_path / "runs" / "isolation_expr"
    snapshots = source / "memory_snapshots"
    snapshots.mkdir(parents=True)
    checkpoint = snapshots / "agent_00_checkpoint_0270.json"
    checkpoint.write_text("{}", encoding="utf-8")

    config = build_task_config(
        base,
        task_id="expr_online",
        seed=FORMAL_SEEDS[2],
        worker_root=tmp_path,
        run_id="expr_online",
        source_run_dir=source,
    )
    experiment = config["experiment"]
    assert experiment["train_hands"] == 0
    assert experiment["initial_checkpoint_hand"] == 270
    assert experiment["memory_mode"] == "Online"
    assert experiment["update_memory_test"] is True
    assert experiment["initial_memory_snapshots"] == {"agent_00": str(checkpoint)}


def test_timeboxed_mixed_uses_one_table_full_roster(tmp_path: Path) -> None:
    base = load_config(BASE)
    config = build_task_config(
        base,
        task_id="mixed_ecological",
        seed=FORMAL_SEEDS[0],
        worker_root=tmp_path,
        run_id="mixed_ecological",
        source_run_dir=None,
    )
    experiment = config["experiment"]
    assert experiment["heldout_table_set"] == ["H01"]
    assert experiment["checkpoint_set"] == [270]
    assert experiment["checkpoint_test_hands_by_checkpoint"] == {270: 60}
    assert experiment["evaluate_all_train_agents"] is True
    assert len(experiment["agent_roster"]) == 8
