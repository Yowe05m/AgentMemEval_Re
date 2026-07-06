"""
模块说明：本模块负责记忆快照的文件读写。
核心职责：将 MemorySnapshot 保存为 JSON，并从 JSON 恢复领域对象。
输入与输出：输入快照对象或路径，输出 JSON 文件或 MemorySnapshot。
依赖边界：依赖核心领域对象和标准库 json。
不负责：不解释具体记忆 payload，不决定何时快照。
"""

from __future__ import annotations

import json
from pathlib import Path

from agentmemeval.core.domain import MemorySnapshot


def save_snapshot(path: str | Path, snapshot: MemorySnapshot) -> None:
    """
    功能：保存记忆快照。
    参数：
        path：输出路径。
        snapshot：记忆快照。
    返回：无。
    副作用：写入 JSON 文件。
    异常：文件写入失败时抛出标准异常。
    设计说明：快照 JSON 化，避免 pickle 绑定代码版本。
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_snapshot(path: str | Path) -> MemorySnapshot:
    """
    功能：读取记忆快照。
    参数：
        path：快照路径。
    返回：MemorySnapshot。
    副作用：读取文件。
    异常：文件或 JSON 错误由标准库抛出。
    设计说明：恢复时由具体记忆机制解释 payload。
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MemorySnapshot(**data)
