"""Run a real-model A/B comparison of native and local discrete raise sizing."""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from agentmemeval.config.loader import load_config
from agentmemeval.experiments.runner import run_resolved_config

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("native_no_limit", "local_discrete")
CONDITIONS: dict[str, dict[str, Any]] = {
    "no_memory": {"mechanism": "no_memory", "memory_scope": "per_agent"},
    "fact": {
        "mechanism": "fact",
        "memory_scope": "per_agent",
        "top_k": 6,
        "max_records": 300,
    },
    "persona_enfp": {
        "mechanism": "fact_expr_sync",
        "memory_scope": "per_agent",
        "top_k": 6,
        "window_size": 6,
        "persona": "ENFP",
    },
    "persona_istp": {
        "mechanism": "fact_expr_sync",
        "memory_scope": "per_agent",
        "top_k": 6,
        "window_size": 6,
        "persona": "ISTP",
    },
}


def main() -> int:
    os.chdir(ROOT)
    _load_env(ROOT / ".env")
    base = load_config(ROOT / "configs/experiments/local_base_small.yaml")
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_root = ROOT / "outputs" / f"raise_sizing_ab_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "same-seed native_no_limit vs local_discrete raise sizing A/B",
        "model": base["provider"]["model"],
        "seed": 20260711,
        "train_hands": 6,
        "test_hands": 2,
        "runs": {},
    }
    summary_path = output_root / "summary.json"

    for policy in POLICIES:
        for slug, condition in CONDITIONS.items():
            run_id = f"{policy}__{slug}"
            print(f"[{datetime.now().isoformat(timespec='seconds')}] START {run_id}", flush=True)
            config = copy.deepcopy(base)
            config["provider"]["max_output_tokens"] = 256
            config["agent"] = {**condition, "raise_sizing_policy": policy}
            config["opponent_agent"]["raise_sizing_policy"] = policy
            config["heldout_agent"]["raise_sizing_policy"] = policy
            config["experiment"].update(
                {
                    "seed": 20260711,
                    "output_root": str(output_root),
                    "run_id": run_id,
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
            summary["runs"][run_id] = _summarize(result.to_dict(), run_dir, policy)
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[{datetime.now().isoformat(timespec='seconds')}] DONE {run_id}", flush=True)

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["comparison"] = _compare(summary["runs"])
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY {summary_path}", flush=True)
    return 0


def _summarize(
    result: dict[str, Any],
    run_dir: Path,
    policy: str,
) -> dict[str, Any]:
    events = _read_jsonl(run_dir / "events.jsonl")
    action_events = [event for event in events if event.get("event") == "action"]
    target_events = [event for event in action_events if event.get("agent_id") == "agent_00"]
    target_raises = [event for event in target_events if event.get("action_type") == "raise"]
    target = result["metrics"]["primary_metrics"]["per_agent"]["agent_00"]
    quality = result["metrics"]["exploratory_metrics"]["decision_quality"]["combined"]
    return {
        "policy": policy,
        "run_dir": str(run_dir),
        "target_decisions": len(target_events),
        "target_fold_rate": target["fold_rate"],
        "target_chip_delta": target["chip_delta"],
        "target_raise_count": len(target_raises),
        "target_max_raise_to": max(
            (int(event.get("amount") or 0) for event in target_raises),
            default=0,
        ),
        "target_raise_to_at_least_500": sum(
            int(event.get("amount") or 0) >= 500 for event in target_raises
        ),
        "all_agent_quality": quality,
        "observed_raise_sizing_policies": sorted(
            {
                str((event.get("raise_sizing") or {}).get("policy"))
                for event in action_events
            }
        ),
        "discrete_enum_violations": sum(
            _violates_discrete_amount(event) for event in action_events
        ),
    }


def _compare(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs = {}
    for slug in CONDITIONS:
        native = runs[f"native_no_limit__{slug}"]
        discrete = runs[f"local_discrete__{slug}"]
        pairs[slug] = {
            "native_target_max_raise_to": native["target_max_raise_to"],
            "discrete_target_max_raise_to": discrete["target_max_raise_to"],
            "native_target_large_raise_count": native["target_raise_to_at_least_500"],
            "discrete_target_large_raise_count": discrete["target_raise_to_at_least_500"],
            "native_target_chip_delta": native["target_chip_delta"],
            "discrete_target_chip_delta": discrete["target_chip_delta"],
        }
    return pairs


def _violates_discrete_amount(event: dict[str, Any]) -> bool:
    if event.get("action_type") != "raise":
        return False
    sizing = event.get("raise_sizing") or {}
    allowed = sizing.get("allowed_amounts") if isinstance(sizing, dict) else None
    if not isinstance(allowed, list):
        return False
    return int(event.get("amount") or 0) not in {int(amount) for amount in allowed}


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
