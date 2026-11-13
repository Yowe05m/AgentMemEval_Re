from __future__ import annotations

import json
from pathlib import Path

from agentmemeval.path1_v1.core.domain import MemorySnapshot


def save_snapshot(path: str | Path, snapshot: MemorySnapshot) -> None:

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_snapshot(path: str | Path) -> MemorySnapshot:

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MemorySnapshot(**data)
