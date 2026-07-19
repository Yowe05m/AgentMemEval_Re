from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import yaml

from agentmemeval.prompts.decision import BASE_SYSTEM_PROMPT, PROMPT_TEMPLATE_VERSION
from agentmemeval.prompts.experience_update import EXPERIENCE_UPDATE_PROMPT


def _gate_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "task4"
        / "gate_counterfactual_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("task4_counterfactual_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path, *, violate_revision: bool = False) -> Path:
    import hashlib

    run = tmp_path / "run"
    snapshots = run / "memory_snapshots"
    snapshots.mkdir(parents=True)
    config = {
        "experiment": {
            "run_mode": "smoke",
            "protocol_label": "counterfactual_smoke_not_for_main_table",
        }
    }
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    agents = [
        "fact_00",
        "fact_01",
        "expr_00",
        "expr_01",
        "sync_00",
        "sync_01",
        "async_00",
        "async_01",
    ]
    stage_agents = {
        agent_id: {
            "hands": 10,
            "vpip": 0.3,
            "fold_rate": 0.4,
            "voluntary_participation_hands": 3,
            "all_in_hand_rate": 0.0,
            "bust_hand_rate": 0.0,
            "hand_reward_sensitivity": {
                "share_of_absolute_reward_activity": 0.2
            },
            "memory": {
                "empty_retrieval_rate": 0.0,
                "max_structural_signature_share": 0.0,
                "revision_fallback_count": 0,
            },
        }
        for agent_id in agents
    }
    prompts = {
        "decision_version": PROMPT_TEMPLATE_VERSION,
        "decision_system_sha256": hashlib.sha256(
            BASE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "experience_update_sha256": hashlib.sha256(
            EXPERIENCE_UPDATE_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    json_files = {
        "manifest.json": {
            "metadata": {
                "code": {"commit": "expected-sha", "dirty": False},
                "prompts": prompts,
            }
        },
        "metrics.json": {
            "primary_metrics": {"stage_per_agent": {"test": stage_agents}},
            "exploratory_metrics": {
                "decision_quality": {"combined": {"fallback_count": 0}}
            },
        },
        "protocol_audit.json": {
            "evaluation_target_ids": agents,
            "execution_health": {"valid": True},
        },
        "checkpoint_generalization.json": {"results": []},
        "experiment_result.json": {"status": "complete"},
    }
    for name, data in json_files.items():
        (run / name).write_text(json.dumps(data), encoding="utf-8")
    hand = {
        "hand_id": "h1",
        "starting_stacks": {"a": 100, "b": 100},
        "final_stacks": {"a": 101, "b": 99},
        "rewards": {"a": 1, "b": -1},
    }
    (run / "hand_summaries.jsonl").write_text(
        json.dumps(hand) + "\n", encoding="utf-8"
    )
    (run / "events.jsonl").write_text(
        json.dumps({"event": "action", "fallback_used": False}) + "\n",
        encoding="utf-8",
    )
    (run / "report.md").write_text("complete\n", encoding="utf-8")
    for agent_id in agents:
        if agent_id.startswith("expr_"):
            mechanism = "expr"
        elif agent_id.startswith("sync_"):
            mechanism = "fact_expr_sync"
        elif agent_id.startswith("async_"):
            mechanism = "fact_expr_async"
        else:
            mechanism = "fact"
        revision = {
            "keep": not violate_revision,
            "new_md": (
                "应优先在 preflop 果断弃牌"
                if violate_revision
                else "# 我的经验\n（证据不足，保持不变）"
            ),
            "calibration_note": "样本量不足" if violate_revision else "保持观察",
            "self_check": "无法直接验证" if violate_revision else "未作反事实推断",
        }
        expr = {"revision_log": [revision]}
        payload = expr if mechanism == "expr" else {"expr": expr}
        snapshot = {
            "mechanism": mechanism,
            "agent_id": agent_id,
            "scope": "per_agent",
            "payload": payload,
        }
        (snapshots / f"{agent_id}_final.json").write_text(
            json.dumps(snapshot), encoding="utf-8"
        )
    return run


def test_counterfactual_smoke_gate_accepts_clean_evidence(tmp_path: Path) -> None:
    audit = _gate_module().build_gate(
        _run(tmp_path),
        expected_code_sha="expected-sha",
    )
    assert audit["status"] == "ready_to_start_campaign_p_v7_pilot"
    assert audit["blockers"] == []
    assert audit["evidence"]["revision_audit"]["experiential_snapshot_count"] == 6


def test_counterfactual_smoke_gate_rejects_self_contradictory_policy_revision(
    tmp_path: Path,
) -> None:
    audit = _gate_module().build_gate(
        _run(tmp_path, violate_revision=True),
        expected_code_sha="expected-sha",
    )
    assert audit["status"] == "no_go"
    assert any("revision violations" in blocker for blocker in audit["blockers"])
