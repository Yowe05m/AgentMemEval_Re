"""
模块说明：本模块提供 JSONL 追加与读取工具。
核心职责：用 UTF-8 JSON Lines 保存事件和手牌摘要。
输入与输出：输入字典记录，输出文件或记录列表。
依赖边界：只依赖标准库 json/pathlib。
不负责：不定义记录 schema，不做指标计算。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlStore:
    """
    功能：管理一个 JSONL 文件。
    参数：
        path：文件路径。
    返回：存储对象。
    副作用：创建父目录。
    异常：文件读写错误由标准库抛出。
    设计说明：实验事件采用 append-only，方便中途排查。
    """

    def __init__(self, path: str | Path) -> None:
        """
        功能：初始化 JSONL 存储。
        参数：
            path：文件路径。
        返回：无。
        副作用：创建父目录。
        异常：无。
        设计说明：不自动清空文件，由 ArtifactManager 控制 run 目录唯一性。
        """

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        """
        功能：追加一条记录。
        参数：
            record：JSON 可序列化字典。
        返回：无。
        副作用：写入文件。
        异常：序列化失败时抛出标准异常。
        设计说明：逐行写入便于流式处理。
        """

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        """
        功能：读取全部记录。
        参数：无。
        返回：记录列表。
        副作用：读取文件。
        异常：JSON 解析失败时抛出标准异常。
        设计说明：report 命令从原始工件重算指标。
        """

        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
