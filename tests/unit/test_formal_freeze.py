from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from agentmemeval.config.loader import load_config
from agentmemeval.core.errors import ConfigError
from agentmemeval.evaluation.formal_freeze import generate_formal_freeze_bundle


def _proposal(required: int = 4) -> dict[str, object]:
    return {
        "schema_version": "agentmemeval_pilot_freeze_proposal_v1",
        "status": "ready_to_generate_immutable_formal_configs",
        "blockers": [],
        "required_seed_pairs": required,
        "power_plan": {
            "blockers": [],
            "no_silent_resource_cap": True,
            "required_seed_pairs_primary_max_across_p_and_e": required,
        },
        "behavior_freeze": {
            "status": "frozen",
            "blockers": [],
            "thresholds": {
                "min_vpip": 0.02,
                "max_fold_rate": 0.90,
                "min_voluntary_participation_hands": 1,
                "max_all_in_hand_rate": 0.20,
                "max_bust_hand_rate": 0.20,
                "max_single_hand_reward_activity_share": 0.60,
                "max_empty_retrieval_rate": 0.90,
                "max_structural_signature_share": 0.80,
            },
        },
        "retrieval_freeze": {
            "retrieval_threshold_status": "frozen",
            "minimum_retrieval_score": 0.0,
        },
        "campaign_p_evidence": {"completed_seeds": [2026071701, 2026071702]},
        "campaign_e_evidence": {"completed_seeds": [2026071701, 2026071702]},
    }


def _runtime_lock() -> dict[str, str]:
    return {
        "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "gpu_driver": "575.57.08",
        "service_torch_cuda_version": "12.8",
        "vllm_version": "0.10.2",
    }


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _generate(tmp_path: Path, proposal: dict[str, object] | None = None) -> dict[str, object]:
    proposal_path = tmp_path / "proposal.json"
    runtime_path = tmp_path / "runtime.json"
    _write_json(proposal_path, proposal or _proposal())
    _write_json(runtime_path, _runtime_lock())
    root = Path(__file__).resolve().parents[2]
    return generate_formal_freeze_bundle(
        proposal_path=proposal_path,
        runtime_lock_path=runtime_path,
        campaign_p_template_path=root
        / "configs/campaigns/task4_campaign_p_pilot_parallel_v2.yaml",
        campaign_e_template_path=root
        / "configs/campaigns/task4_campaign_e_pilot_parallel_v2.yaml",
        formal_p_template_path=root
        / "configs/experiments/task4_campaign_p_robust_formal_template.yaml",
        formal_e_template_path=root
        / "configs/experiments/task4_campaign_e_robust_formal_template.yaml",
        output_dir=tmp_path / "bundle",
        freeze_id="pilot_20260717_v1",
        seed_start=2026071801,
        preflight_seed=2026071799,
    )


def test_formal_freeze_generates_self_contained_valid_p_and_e_bundle(
    tmp_path: Path,
) -> None:
    result = _generate(tmp_path)
    output = Path(str(result["output_dir"]))
    assert result["required_seed_pairs"] == 4
    assert result["seeds"] == [2026071801, 2026071802, 2026071803, 2026071804]
    for label in ("formal_p", "formal_e"):
        config_path = output / result["files"][label]
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "extends" not in raw
        loaded = load_config(config_path)
        assert loaded["experiment"]["protocol_readiness"] == "ready"
        assert loaded["experiment"]["required_seed_pairs"] == 4
        assert loaded["experiment"]["formal_runtime_lock"] == _runtime_lock()
        assert loaded["agent"]["minimum_retrieval_score"] == 0.0
    for label in ("campaign_p", "campaign_e"):
        campaign = yaml.safe_load(
            (output / result["files"][label]).read_text(encoding="utf-8")
        )["campaign"]
        assert campaign["seeds"] == result["seeds"]
        assert campaign["protocol_label"] == "paper_robust_formal_frozen"
        assert campaign["max_parallel_runs"] == 4
        assert (output / campaign["base_experiment_config"]).is_file()
    for label in ("preflight_p", "preflight_e"):
        preflight = load_config(output / result["files"][label])
        assert preflight["experiment"]["run_mode"] == "pilot"
        assert preflight["experiment"]["seed"] == 2026071799
        assert preflight["experiment"]["frozen_config_preflight"] is True
        formal_label = label.removeprefix("preflight_")
        formal = load_config(output / result["files"][f"formal_{formal_label}"])
        preflight_experiment = dict(preflight["experiment"])
        formal_experiment = dict(formal["experiment"])
        for key in ("seed", "run_mode", "frozen_config_preflight"):
            preflight_experiment.pop(key, None)
            formal_experiment.pop(key, None)
        assert preflight_experiment == formal_experiment
        assert preflight["agent"] == formal["agent"]
        assert preflight["provider"] == formal["provider"]
    for label in ("preflight_campaign_p", "preflight_campaign_e"):
        campaign = yaml.safe_load(
            (output / result["files"][label]).read_text(encoding="utf-8")
        )["campaign"]
        assert campaign["seeds"] == [2026071799]
        assert campaign["protocol_label"] == "frozen_config_preflight_not_for_paper"


def test_formal_freeze_rejects_blocked_proposal_before_creating_output(
    tmp_path: Path,
) -> None:
    proposal = copy.deepcopy(_proposal())
    proposal["status"] = "no_go_pilot_freeze_blocked"
    with pytest.raises(ConfigError, match="not ready"):
        _generate(tmp_path, proposal)
    assert not (tmp_path / "bundle").exists()


def test_formal_freeze_rejects_missing_runtime_field(tmp_path: Path) -> None:
    proposal_path = tmp_path / "proposal.json"
    runtime_path = tmp_path / "runtime.json"
    _write_json(proposal_path, _proposal())
    runtime = _runtime_lock()
    runtime.pop("vllm_version")
    _write_json(runtime_path, runtime)
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ConfigError, match="vllm_version"):
        generate_formal_freeze_bundle(
            proposal_path=proposal_path,
            runtime_lock_path=runtime_path,
            campaign_p_template_path=root
            / "configs/campaigns/task4_campaign_p_pilot_parallel_v2.yaml",
            campaign_e_template_path=root
            / "configs/campaigns/task4_campaign_e_pilot_parallel_v2.yaml",
            formal_p_template_path=root
            / "configs/experiments/task4_campaign_p_robust_formal_template.yaml",
            formal_e_template_path=root
            / "configs/experiments/task4_campaign_e_robust_formal_template.yaml",
            output_dir=tmp_path / "bundle",
            freeze_id="pilot_20260717_v1",
            preflight_seed=2026071799,
        )
    assert not (tmp_path / "bundle").exists()


def test_formal_freeze_refuses_to_overwrite_existing_bundle(tmp_path: Path) -> None:
    _generate(tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        _generate(tmp_path)


def test_formal_freeze_rejects_preflight_seed_overlap(tmp_path: Path) -> None:
    proposal_path = tmp_path / "proposal.json"
    runtime_path = tmp_path / "runtime.json"
    _write_json(proposal_path, _proposal())
    _write_json(runtime_path, _runtime_lock())
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ConfigError, match="calibration Pilot"):
        generate_formal_freeze_bundle(
            proposal_path=proposal_path,
            runtime_lock_path=runtime_path,
            campaign_p_template_path=root
            / "configs/campaigns/task4_campaign_p_pilot_parallel_v2.yaml",
            campaign_e_template_path=root
            / "configs/campaigns/task4_campaign_e_pilot_parallel_v2.yaml",
            formal_p_template_path=root
            / "configs/experiments/task4_campaign_p_robust_formal_template.yaml",
            formal_e_template_path=root
            / "configs/experiments/task4_campaign_e_robust_formal_template.yaml",
            output_dir=tmp_path / "bundle",
            freeze_id="pilot_20260717_v1",
            seed_start=2026071801,
            preflight_seed=2026071701,
        )
