from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from agentmemeval.path1_v1.storage.jsonl_store import JsonlStore


def dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=True)


class ArtifactManager:
    def __init__(self, root: str | Path, run_id: str, config: dict[str, Any]) -> None:

        self.root = Path(root)
        self.run_id = run_id
        self.run_dir = self.root / run_id
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(f"run directory already exists and is not empty: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "memory_snapshots").mkdir(exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)
        self.config = config
        self.events = JsonlStore(self.run_dir / "events.jsonl")
        self.hand_summaries = JsonlStore(self.run_dir / "hand_summaries.jsonl")
        self.write_text("resolved_config.yaml", dump_yaml(config))

    def log_event(self, record: dict[str, Any]) -> None:

        self.events.append(record)

    def log_hand(self, record: dict[str, Any]) -> None:

        self.hand_summaries.append(record)

    def agent_memory_directory(
        self,
        agent_id: str,
        *,
        namespace: tuple[str, ...] = (),
    ) -> Path:
        """Return an isolated, traversal-safe directory for one Agent's live memory files."""

        components = ["agent_memory"]
        components.extend(_safe_path_component(value) for value in namespace)
        components.append(_safe_path_component(agent_id))
        path = self.run_dir.joinpath(*components)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative_path: str, data: dict[str, Any]) -> str:

        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def write_text(self, relative_path: str, text: str) -> str:

        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)


def _safe_path_component(value: str) -> str:
    """Create a readable unique path component without accepting traversal syntax."""

    raw = str(value)
    clean = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw).strip("._")
    clean = clean[:80] or "agent"
    if clean == raw:
        return clean
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{clean}_{digest}"
