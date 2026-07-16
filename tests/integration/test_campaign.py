from pathlib import Path

import yaml

from agentmemeval.experiments.campaign import aggregate_campaign, run_campaign


def test_campaign_e_runs_append_only_matrix_and_resumes(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "provider": {
                    "provider": "mock",
                    "model": "mock-deterministic-v1",
                    "structured_output_mode": "json_object",
                },
                "table": {
                    "starting_stack": 1000,
                    "small_blind": 1,
                    "big_blind": 2,
                    "max_raises_per_street": 4,
                    "lifecycle": "continuous_rebuy",
                },
                "agent": {
                    "mechanism": "fact",
                    "memory_scope": "per_agent",
                    "embedding_cache_path": str(tmp_path / "shared" / "{agent_id}.json"),
                },
                "opponent_agent": {
                    "mechanism": "no_memory",
                    "memory_scope": "per_agent",
                },
                "heldout_agent": {
                    "mechanism": "no_memory",
                    "memory_scope": "per_agent",
                },
                "experiment": {
                    "scenario": "fixed_evolving_table",
                    "run_mode": "smoke",
                    "seed": 1,
                    "train_hands": 1,
                    "test_hands": 1,
                    "checkpoint_interval": 1,
                    "checkpoint_test_hands": 1,
                    "update_memory_train": True,
                    "update_memory_test": False,
                    "primary_endpoint": "final_test_bb_per_100",
                    "multiple_comparison_method": "holm",
                    "statistical_plan_status": "pending_pilot_power_calibration",
                    "behavior_threshold_status": "pending_pilot",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        yaml.safe_dump(
            {
                "campaign": {
                    "campaign_id": "campaign_e_smoke",
                    "design": "target_vs_seven_no_memory",
                    "protocol_label": "not_for_paper_smoke",
                    "base_experiment_config": "base.yaml",
                    "output_root": str(tmp_path / "campaigns"),
                    "max_parallel_runs": 2,
                    "seeds": [101],
                    "baseline_condition_id": "no_memory_target",
                    "conditions": [
                        {
                            "condition_id": "no_memory_target",
                            "target_mechanism": "no_memory",
                        },
                        {"condition_id": "fact_target", "target_mechanism": "fact"},
                        {"condition_id": "expr_target", "target_mechanism": "expr"},
                        {
                            "condition_id": "sync_target",
                            "target_mechanism": "fact_expr_sync",
                        },
                        {
                            "condition_id": "async_target",
                            "target_mechanism": "fact_expr_async",
                        },
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    first = run_campaign(campaign_path)
    assert first["completed_this_invocation"] == 5
    assert first["failed_this_invocation"] == 0
    assert first["completed_matrix_units"] == 5
    assert first["aggregate_status"] == "descriptive_only"
    assert first["max_parallel_runs"] == 2

    campaign_dir = Path(first["campaign_dir"])
    run_dirs = sorted((campaign_dir / "runs").iterdir())
    assert len(run_dirs) == 5
    cache_paths = []
    for run_dir in run_dirs:
        resolved = yaml.safe_load(
            (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        cache_path = Path(resolved["agent"]["embedding_cache_path"])
        assert cache_path == run_dir / "embedding_cache" / "{agent_id}.json"
        cache_paths.append(cache_path)
    assert len(set(cache_paths)) == 5
    state_lines_before = (campaign_dir / "state.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(state_lines_before) == 11
    aggregate = yaml.safe_load(Path(first["aggregate_path"]).read_text(encoding="utf-8"))
    assert aggregate["estimand"] == (
        "same_seed_cross_condition_target_effect_vs_no_memory"
    )
    assert set(aggregate["paired_comparisons"]) == {
        "fact_target",
        "expr_target",
        "sync_target",
        "async_target",
    }
    rebuilt = aggregate_campaign(campaign_dir)
    assert rebuilt["status"] == "descriptive_only"
    assert rebuilt["completed_run_count"] == 5
    assert Path(rebuilt["aggregate_path"]).is_file()

    resumed = run_campaign(campaign_path, resume=True)
    assert resumed["completed_this_invocation"] == 0
    assert resumed["failed_this_invocation"] == 0
    assert resumed["skipped_valid_completed"] == 5
    assert len(sorted((campaign_dir / "runs").iterdir())) == 5
    assert (campaign_dir / "state.tsv").read_text(
        encoding="utf-8"
    ).splitlines() == state_lines_before
