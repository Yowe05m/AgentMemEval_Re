from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from agentmemeval.config.loader import load_config
from agentmemeval.core.errors import ConfigError
from agentmemeval.evaluation.aggregation import validate_runtime_homogeneity
from agentmemeval.evaluation.formal_freeze import generate_formal_freeze_bundle
from agentmemeval.evaluation.pilot import build_pilot_freeze_proposal_from_paths
from agentmemeval.evaluation.runtime_lock import (
    build_formal_runtime_lock_from_manifest,
)
from agentmemeval.prompts.decision import BASE_SYSTEM_PROMPT, PROMPT_TEMPLATE_VERSION
from agentmemeval.prompts.experience_update import EXPERIENCE_UPDATE_PROMPT
from tests.unit.test_pilot_power_plan import (
    _e,
    _execution_health,
    _metrics,
    _p,
    _review,
)


def _proposal(
    tmp_path: Path,
    *,
    blocked: bool = False,
    runtime_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    evidence_root = tmp_path / "pilot-evidence"
    evidence_root.mkdir(exist_ok=True)
    p_path = evidence_root / "p.json"
    e_path = evidence_root / "e.json"
    review_path = evidence_root / "review.json"
    p_dir = evidence_root / "campaign-p"
    e_dir = evidence_root / "campaign-e"
    p_dir.mkdir(exist_ok=True)
    e_dir.mkdir(exist_ok=True)
    runtime_manifest = runtime_manifest or _runtime_manifest()
    runtime_homogeneity = validate_runtime_homogeneity([runtime_manifest])
    p_aggregate = _p()
    e_aggregate = _e()
    p_aggregate["runtime_homogeneity"] = runtime_homogeneity
    e_aggregate["runtime_homogeneity"] = runtime_homogeneity
    _write_json(p_path, p_aggregate)
    _write_json(e_path, e_aggregate)
    _write_json(review_path, _review())
    header = (
        "event_utc\tcondition_id\ttarget_mechanism\tseed\tattempt\tstatus\t"
        "run_id\trun_dir\tfailure_class\tmessage"
    )
    p_rows = [header]
    for index, seed in enumerate((1, 2, 3)):
        run_id = f"p{index}"
        run_dir = p_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "manifest.json", runtime_manifest)
        _write_json(run_dir / "metrics.json", _metrics())
        health = (
            _execution_health(fallback_count=1)
            if blocked and index == 0
            else _execution_health()
        )
        _write_json(
            run_dir / "protocol_audit.json",
            {
                "evaluation_target_ids": ["fact_00"],
                "execution_health": health,
            },
        )
        p_rows.append(
            f"t\tmixed\tmixed\t{seed}\t1\tcomplete\t{run_id}\t{run_dir}\t\t"
        )
    (p_dir / "state.tsv").write_text(
        "\n".join(p_rows) + "\n",
        encoding="utf-8",
    )
    e_rows = [header]
    for index in range(15):
        seed = index % 3 + 1
        condition = index // 3
        run_id = f"e{index}"
        run_dir = e_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "manifest.json", runtime_manifest)
        _write_json(run_dir / "metrics.json", _metrics())
        _write_json(
            run_dir / "protocol_audit.json",
            {
                "evaluation_target_ids": ["fact_00"],
                "execution_health": _execution_health(),
            },
        )
        e_rows.append(
            f"t\tcondition-{condition}\ttarget\t{seed}\t1\tcomplete\t"
            f"{run_id}\t{run_dir}\t\t"
        )
    (e_dir / "state.tsv").write_text(
        "\n".join(e_rows) + "\n",
        encoding="utf-8",
    )
    return build_pilot_freeze_proposal_from_paths(
        p_path,
        e_path,
        p_dir,
        e_dir,
        review_path,
    )


def _runtime_manifest() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    config = load_config(
        root / "configs/experiments/task4_campaign_p_robust_formal_template.yaml"
    )
    provider = config["provider"]
    agent = config["agent"]
    return {
        "run_id": "pilot__s1__a01",
        "metadata": {
            "protocol": {
                "admission_audit": {
                    "checks": {
                        "decision_model_identity_complete": True,
                        "embedding_identity_complete": True,
                        "decision_service_smoke_passed": True,
                        "embedding_service_smoke_passed": True,
                    }
                }
            },
            "code": {"commit": "a" * 40, "dirty": False},
            "model": {
                "name": provider["model"],
                "revision": provider["model_revision"],
                "weights_hash": provider["model_weights_hash"],
                "served_model_name": provider["served_model_name"],
            },
            "embedding": {
                "backend": agent["embedding_backend"],
                "name": agent["embedding_model"],
                "revision": agent["embedding_revision"],
                "weights_hash": agent["embedding_weights_hash"],
                "service_startup_parameters": agent[
                    "embedding_service_startup_parameters"
                ],
            },
            "service": {
                "service_startup_parameters": provider[
                    "service_startup_parameters"
                ]
            },
            "model_service_runtime": {
                "status": "verified",
                "torch_cuda_version": "13.0",
                "vllm_version": provider["service_startup_parameters"][
                    "vllm_version"
                ],
            },
            "prompts": {
                "decision_version": PROMPT_TEMPLATE_VERSION,
                "decision_system_sha256": hashlib.sha256(
                    BASE_SYSTEM_PROMPT.encode("utf-8")
                ).hexdigest(),
                "experience_update_sha256": hashlib.sha256(
                    EXPERIENCE_UPDATE_PROMPT.encode("utf-8")
                ).hexdigest(),
            },
            "gpu": {
                "devices": [
                    {
                        "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                        "driver": "575.57.08",
                    }
                ]
            },
        },
    }


def _runtime_lock(tmp_path: Path) -> dict[str, object]:
    source = (
        tmp_path
        / "pilot-evidence"
        / "campaign-p"
        / "runs"
        / "p0"
        / "manifest.json"
    )
    return build_formal_runtime_lock_from_manifest(source)


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _generate(
    tmp_path: Path,
    proposal: dict[str, object] | None = None,
    runtime_artifact: dict[str, object] | None = None,
) -> dict[str, object]:
    proposal_path = tmp_path / "proposal.json"
    runtime_path = tmp_path / "runtime.json"
    _write_json(proposal_path, proposal or _proposal(tmp_path))
    _write_json(runtime_path, runtime_artifact or _runtime_lock(tmp_path))
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
    assert result["schema_version"] == "agentmemeval_formal_freeze_bundle_v2"
    assert result["proposal_source_rebuild"]["verified"] is True
    assert result["required_seed_pairs"] == 3
    assert result["seeds"] == [2026071801, 2026071802, 2026071803]
    for label in ("formal_p", "formal_e"):
        config_path = output / result["files"][label]
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "extends" not in raw
        loaded = load_config(config_path)
        assert loaded["experiment"]["protocol_readiness"] == "ready"
        assert loaded["experiment"]["required_seed_pairs"] == 3
        assert loaded["experiment"]["formal_runtime_lock"] == _runtime_lock(
            tmp_path
        )["formal_runtime_lock"]
        assert loaded["agent"]["minimum_retrieval_score"] == 0.42
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
    proposal = _proposal(tmp_path, blocked=True)
    with pytest.raises(ConfigError, match="not ready"):
        _generate(tmp_path, proposal)
    assert not (tmp_path / "bundle").exists()


def test_formal_freeze_rejects_hand_edited_ready_proposal(
    tmp_path: Path,
) -> None:
    proposal = _proposal(tmp_path)
    proposal["required_seed_pairs"] = 99
    with pytest.raises(ConfigError, match="differs from deterministic source rebuild"):
        _generate(tmp_path, proposal)
    assert not (tmp_path / "bundle").exists()


def test_formal_freeze_rejects_changed_source_evidence(tmp_path: Path) -> None:
    proposal = _proposal(tmp_path)
    review_path = Path(str(proposal["retrieval_review_evidence"]["path"]))
    review_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="evidence hash mismatch"):
        _generate(tmp_path, proposal)
    assert not (tmp_path / "bundle").exists()


def test_formal_freeze_rejects_runtime_source_outside_pilot_leaves(
    tmp_path: Path,
) -> None:
    proposal = _proposal(tmp_path)
    outside = tmp_path / "outside-manifest.json"
    _write_json(outside, _runtime_manifest())
    runtime = build_formal_runtime_lock_from_manifest(outside)
    with pytest.raises(ConfigError, match="not a completed P/E Pilot leaf"):
        _generate(tmp_path, proposal, runtime)
    assert not (tmp_path / "bundle").exists()


def test_formal_freeze_rejects_runtime_identity_different_from_pilot(
    tmp_path: Path,
) -> None:
    proposal = _proposal(tmp_path)
    source = (
        tmp_path
        / "pilot-evidence"
        / "campaign-p"
        / "runs"
        / "p0"
        / "manifest.json"
    )
    changed = _runtime_manifest()
    changed["metadata"]["model"]["revision"] = "different-revision"
    _write_json(source, changed)
    runtime = build_formal_runtime_lock_from_manifest(source)
    with pytest.raises(ConfigError, match="differs from P/E Pilot runtime"):
        _generate(tmp_path, proposal, runtime)
    assert not (tmp_path / "bundle").exists()


def test_formal_freeze_rejects_template_identity_different_from_runtime_lock(
    tmp_path: Path,
) -> None:
    changed = _runtime_manifest()
    changed["metadata"]["model"]["revision"] = "different-revision"
    proposal = _proposal(tmp_path, runtime_manifest=changed)
    runtime = _runtime_lock(tmp_path)
    with pytest.raises(ConfigError, match="formal template identity differs"):
        _generate(tmp_path, proposal, runtime)
    assert not (tmp_path / "bundle").exists()


def test_formal_freeze_rejects_legacy_four_field_runtime_lock(
    tmp_path: Path,
) -> None:
    proposal_path = tmp_path / "proposal.json"
    runtime_path = tmp_path / "runtime.json"
    _write_json(proposal_path, _proposal(tmp_path))
    _write_json(
        runtime_path,
        {
            "gpu_name": "gpu",
            "gpu_driver": "driver",
            "service_torch_cuda_version": "12.8",
            "vllm_version": "0.10.2",
        },
    )
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ConfigError, match="schema is not V2"):
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
    _write_json(proposal_path, _proposal(tmp_path))
    _write_json(runtime_path, _runtime_lock(tmp_path))
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
            preflight_seed=1,
        )
