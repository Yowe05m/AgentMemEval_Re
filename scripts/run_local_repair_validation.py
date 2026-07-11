"""Run a bounded real-model matrix for prompt, guard, persona, and audit validation."""

from __future__ import annotations

import copy
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from agentmemeval.config.loader import load_config
from agentmemeval.experiments.runner import run_resolved_config

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS: list[tuple[str, dict[str, Any]]] = [
    ("no_memory", {"mechanism": "no_memory", "memory_scope": "per_agent"}),
    (
        "fact",
        {
            "mechanism": "fact",
            "memory_scope": "per_agent",
            "top_k": 6,
            "max_records": 300,
        },
    ),
    (
        "expr",
        {"mechanism": "expr", "memory_scope": "per_agent", "window_size": 6},
    ),
    (
        "fact_expr_sync",
        {
            "mechanism": "fact_expr_sync",
            "memory_scope": "per_agent",
            "top_k": 6,
            "window_size": 6,
        },
    ),
    (
        "fact_expr_async",
        {
            "mechanism": "fact_expr_async",
            "memory_scope": "per_agent",
            "top_k": 6,
            "window_size": 6,
            "sweep_every": 2,
            "evidence_k": 4,
        },
    ),
    *[
        (
            f"persona_{persona.lower()}",
            {
                "mechanism": "fact_expr_sync",
                "memory_scope": "per_agent",
                "top_k": 6,
                "window_size": 6,
                "persona": persona,
            },
        )
        for persona in ("INTJ", "ENFP", "ISTP", "ESFJ")
    ],
]


def main() -> int:
    os.chdir(ROOT)
    _load_env(ROOT / ".env")
    base = load_config(ROOT / "configs/experiments/local_base_small.yaml")
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_root = ROOT / "outputs" / f"local_repair_validation_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "post-repair real-model validation; not mechanism ranking",
        "model": base["provider"]["model"],
        "seed": 20260711,
        "train_hands": 6,
        "test_hands": 2,
        "table_size": 4,
        "conditions": {},
    }
    summary_path = output_root / "summary.json"

    for slug, agent_config in CONDITIONS:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] START {slug}", flush=True)
        config = copy.deepcopy(base)
        config["provider"]["max_output_tokens"] = 256
        config["agent"] = {
            **agent_config,
            "raise_sizing_policy": str(
                base["agent"].get("raise_sizing_policy", "native_no_limit")
            ),
        }
        config["experiment"].update(
            {
                "seed": 20260711,
                "output_root": str(output_root),
                "run_id": slug,
                "train_hands": 6,
                "test_hands": 2,
                "table_size": 4,
                "target_agent_id": "agent_00",
                "update_memory_train": True,
                "update_memory_test": False,
                "all_agents_same_mechanism": False,
            }
        )
        result = run_resolved_config(config)
        run_dir = Path(result.artifacts["run_dir"])
        summary["conditions"][slug] = _summarize_run(result.to_dict(), run_dir)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{datetime.now().isoformat(timespec='seconds')}] DONE {slug}", flush=True)

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["acceptance"] = _acceptance(summary["conditions"])
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY {summary_path}", flush=True)
    return 0


def _summarize_run(result: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    metrics = result["metrics"]
    primary = metrics["primary_metrics"]
    target = primary["per_agent"]["agent_00"]
    stage = primary["stage_per_agent"]
    events = _read_jsonl(run_dir / "events.jsonl")
    hands = _read_jsonl(run_dir / "hand_summaries.jsonl")
    target_events = [
        event
        for event in events
        if event.get("event") == "action" and event.get("agent_id") == "agent_00"
    ]
    target_decisions = len(target_events)
    target_fallbacks = sum(bool(event.get("fallback_used")) for event in target_events)
    target_repairs = sum(bool(event.get("guard_repaired")) for event in target_events)
    target_changed = sum(
        (event.get("raw_decision") or {}).get("action_type") != event.get("action_type")
        for event in target_events
    )
    prompt_versions = sorted(
        {
            str((event.get("prompt") or {}).get("template_version"))
            for event in target_events
        }
    )
    return {
        "run_dir": str(run_dir),
        "target_decisions": target_decisions,
        "target_repaired_count": target_repairs,
        "target_repaired_rate": target_repairs / target_decisions if target_decisions else 0.0,
        "target_fallback_count": target_fallbacks,
        "target_fallback_rate": target_fallbacks / target_decisions if target_decisions else 0.0,
        "target_action_type_changed_count": target_changed,
        "target_action_type_changed_rate": (
            target_changed / target_decisions if target_decisions else 0.0
        ),
        "target_combined_fold_rate": target["fold_rate"],
        "target_train": stage.get("train", {}).get("agent_00", {}),
        "target_test": stage.get("test", {}).get("agent_00", {}),
        "target_memory": target.get("memory", {}),
        "combined_decision_quality": metrics["exploratory_metrics"]["decision_quality"],
        "dealer_counts": dict(Counter(hand.get("dealer_agent_id") for hand in hands)),
        "small_blind_counts": dict(
            Counter(hand.get("small_blind_agent_id") for hand in hands)
        ),
        "big_blind_counts": dict(Counter(hand.get("big_blind_agent_id") for hand in hands)),
        "prompt_versions": prompt_versions,
        "protocol_audit": json.loads(
            (run_dir / "protocol_audit.json").read_text(encoding="utf-8")
        ),
    }


def _acceptance(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    personas = {
        slug: item for slug, item in conditions.items() if slug.startswith("persona_")
    }
    return {
        "all_target_fallback_zero": all(
            item["target_fallback_count"] == 0 for item in conditions.values()
        ),
        "all_target_action_type_changed_zero": all(
            item["target_action_type_changed_count"] == 0 for item in conditions.values()
        ),
        "all_prompt_versions_current": all(
            item["prompt_versions"] == ["2026-07-11-v3"] for item in conditions.values()
        ),
        "persona_fold_rates": {
            slug: item["target_combined_fold_rate"] for slug, item in personas.items()
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


if __name__ == "__main__":
    raise SystemExit(main())
