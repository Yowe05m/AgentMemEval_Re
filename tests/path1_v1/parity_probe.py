"""Identical fixtures for isolated and immutable-source subprocesses.

The fixture clock and UUID sources are deterministic; time fields stay in the comparison.
Only ArtifactManager path returns and its directory-conflict message use a relative root.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


def collect(prefix: str, root: Path, helper_path: Path) -> dict:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 13, 0, 0, 0, tzinfo=UTC).astimezone(tz)

    # Fix the fixture clock before snapshots are hashed, rather than discarding
    # their content hashes from the comparison.
    domain = importlib.import_module(prefix + ".core.domain")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(domain, "datetime", FixedDateTime)
        patch.setattr(importlib.import_module(prefix + ".llm.providers.openai_compatible"),
                      "datetime", FixedDateTime)
        patch.setattr(importlib.import_module(prefix + ".agents.base").time,
                      "perf_counter", lambda: 0.0)
        return _collect(prefix, root, helper_path)


def _collect(prefix: str, root: Path, helper_path: Path) -> dict:
    def load(name):
        return importlib.import_module(prefix + "." + name)

    factory = load("agents.llm_agent")
    domain = load("core.domain")
    artifacts = load("storage.artifacts")
    profiles = load("prompts.profiles")
    path1 = load("experiments.path1")
    requests = []

    class Client:
        provider = "offline_fixture"
        model = "fixture"

        def generate_structured(self, request, schema):
            assert schema is domain.ActionDecision
            requests.append(asdict(request))
            legal = request.observation.legal_actions.types()
            action = "check" if "check" in legal else "call" if "call" in legal else "fold"
            return domain.ActionDecision(action, reason_summary="fixture", intent="fixture")

    def normalize(value):
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        return value

    base = {"mechanism": "expr", "window_size": 30,
            "experience_revision_prompt_window_size": 2,
            "experience_revision_strategy": "llm", "raise_sizing_policy": "local_discrete"}
    cases = [{}, {"mechanism": "none"}, {"type": "naive"}, {"type": "ExprAgent"}, base]
    variants = {
        "scope": ["global", "per_table", "per_agent", "unsupported"],
        "memory_scope": ["global", "per_table", "per_agent"],
        "experience_prompt_token_budget": [1, 65536, 0, True, "32768"],
        "experience_revision_prompt_window_size": [None, 1, 30, 0, 31, True, "2"],
        "window_size": ["30", 1, 0, None],
        "experience_max_chars": [2400, 4096, "bad"],
        "experience_update_period": [1, 2, "bad"],
        "experience_semantics_mode": ["quality_filtered_v2", "paper_faithful_v1", "bad"],
        "experience_revision_contract": ["structured_patch_v2", "full_document_v1", "bad"],
        "experience_document_contract": ["five_section_actionable_v1", "bad"],
        "experience_failure_policy": ["fail_closed_keep", "bad"],
        "experience_semantic_review_mode": ["disabled", "bad"],
        "experience_revision_strategy": ["deterministic", "llm", "bad"],
        "top_k": ["bad"], "max_records": [None, "bad"],
        "minimum_retrieval_score": ["bad"], "semantic_weight": [None, "bad"],
        "feature_weight": [None, "bad"], "persona": [True],
        "type": ["ExprAgent", "no_memory"], "raise_sizing_policy": ["bad"],
    }
    cases.extend({**base, key: value} for key, values in variants.items() for value in values)
    constructions = []
    for config in cases:
        row = {"config": config}
        for function in ("build_memory", "build_agent"):
            try:
                obj = getattr(factory, function)(
                    "hero", config, Client(), "fixture",
                    experience_reviewer_client=None, experience_reviewer_model="reviewer",
                )
                memory = obj.memory if function == "build_agent" else obj
                row[function] = {
                    "type": type(obj).__name__, "snapshot": memory.snapshot().to_dict(),
                }
                if function == "build_agent":
                    row[function]["raise_sizing_policy"] = obj.raise_sizing_policy
                    row[function]["model"] = obj.model
            except Exception as exc:
                row[function] = {"error": type(exc).__name__, "message": str(exc)}
        constructions.append(row)

    storage = artifacts.ArtifactManager(root / "files", "run", {"unicode": "经验", "z": [1, True]})
    directories = [storage.agent_memory_directory(value, namespace=("episode/1", "retained"))
                   for value in ("hero", "../escape", "", "中文", "a" * 90)]
    storage.log_event({"event": "decision", "value": [1, True, None, "中文"]})
    storage.log_hand({"hand": 1, "rewards": {"hero": -2, "villain": 2}})
    storage.write_json("nested/data.json", {"b": 1, "a": "经验"})
    storage.write_text("nested/raw.txt", "literal\n中文\n")
    try:
        artifacts.ArtifactManager(root / "files", "run", {})
    except Exception as exc:
        conflict = {"error": type(exc).__name__,
                    "message": str(exc).replace(str(root), "<fixture-root>")}
    else:
        raise AssertionError("non-empty directory accepted")
    files = {p.relative_to(storage.run_dir).as_posix(): p.read_text(encoding="utf-8")
             for p in sorted(storage.run_dir.rglob("*")) if p.is_file()}
    storage_result = {"files": files,
                      "directories": [str(p.relative_to(root)) for p in directories],
                      "conflict": conflict,
                      "safe_components": [artifacts._safe_path_component(v)
                                          for v in ("hero", "../escape", "", "中文", "a" * 90)]}

    parent = factory.build_agent("agent_00", base, Client(), "fixture").snapshot_memory()
    receipt = path1.run_evaluation_pair(
        output_dir=root / "pair", design=path1.Path1Design(
            "parity", "artifact_00", "offline", 18429, evaluation_episode_hands=2),
        hero_config=base, model="fixture", client_factory=lambda *_: Client(),
        snapshot=parent, episode_index=0,
    )
    scientific_files = {}
    for name in ("hand_lifecycle.jsonl", "settled_hands.jsonl"):
        scientific_files[name] = [json.loads(line) for line in (root / "pair" / name).read_text(
            encoding="utf-8").splitlines()]
    for arm in ("retained", "removed"):
        directory = root / "pair" / f"evaluation_0000_{arm}"
        scientific_files[arm] = [json.loads(line) for line in
                                 (directory / "hand_summaries.jsonl").read_text(
                                     encoding="utf-8").splitlines()]

    spec = importlib.util.spec_from_file_location("parity_fixture_helpers", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    policies = {}
    for name in ("keep", "install", "reject", "contract_failure", "action_repair"):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(helper.provider, "uuid4", lambda: SimpleNamespace(hex="f" * 32))
            patch.setattr(load("llm.terminal_policy"), "uuid4",
                          lambda: SimpleNamespace(hex="e" * 32))
            (root / name).mkdir()
            client = helper.client(patch, root / name)
            replies = {"keep": [helper.response(helper.keep())],
                       "install": [helper.response({
                           **helper.keep(), "operation": "patch", "patches": [{
                               "section": "Starting Hands", "op": "add", "rule_id": "",
                               "when": ["position is favorable and the entry price is low"],
                               "guidance": "choose a call or the minimum legal raise",
                               "invalidators": [
                                   "an earlier player re-raises or the stack is shallow",
                               ],
                               "basis": "model_prior", "status": "tentative",
                               "supporting_trajectory_ids": [], "supporting_fact_ids": [],
                           }],
                       })],
                       "reject": [helper.response(helper.invalid_patch())] * 2,
                       "contract_failure": [helper.response("{bad")] * 2,
                       "action_repair": [helper.response(helper.action("")),
                                         helper.response(helper.action())]}[name]
            calls = helper.mock_http(patch, replies)
            if name == "action_repair":
                result = client.generate_structured(
                    helper.request(), helper.ActionDecision,
                ).to_dict()
                snapshot = None
            else:
                memory = helper.memory(client)
                result = memory._revise_from_window([helper.trajectory()])
                memory.commit_revision([helper.trajectory()], result)
                snapshot = memory.snapshot().to_dict()
                if name == "install":
                    assert memory.current.version == 2, result
            policies[name] = {"calls": calls, "result": result, "snapshot": snapshot}

    identity = load("llm.request_identity")
    context = identity.EpisodeRequestContext(
        "parity", "E1", "artifact_00", "identity_fixture", 918,
        "formation", 0, "formation", "agent_00", 2**62 + 273,
    )
    ledger = identity.SamplerSeedLedger()
    binding = identity.RequestIdentityBinding(context, ledger)
    request = helper.request()
    request.observation.agent_id = "agent_00"
    request.observation.table_id = "artifact_00_formation_0000"
    sampler_records = []
    for purpose in ("action", "revision"):
        for index in (0, 1):
            budget = load("llm.terminal_policy").LogicalRequestBudget(
                17, purpose, logical_request_id=f"fixture_{purpose}_{index}",
            )
            sampler_records.append(binding.issue(request, budget, "primary"))
            if index == 0:
                sampler_records.append(binding.issue(request, budget, "contract_repair"))

    return normalize({"factory": constructions, "storage": storage_result,
                      "profiles": [profiles.active_experience_section_labels(),
                                   profiles.active_prompt_localization()],
                      "engine": {"receipt": receipt, "files": scientific_files,
                                 "actual_requests": requests}, "terminal": policies,
                      "http_identity": {"records": sampler_records, "streams": ledger.entries}})
