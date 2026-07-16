"""
模块说明：本模块管理一次实验运行的输出工件目录。
核心职责：创建 run 目录，写 manifest、resolved config、事件、指标、快照和报告。
输入与输出：输入配置和记录，输出标准文件树。
依赖边界：依赖存储工具、配置 dump 和核心领域对象。
不负责：不运行实验，不计算指标。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentmemeval.config.loader import dump_yaml
from agentmemeval.core.domain import ExperimentResult, MemorySnapshot, RunManifest
from agentmemeval.prompts.decision import BASE_SYSTEM_PROMPT, PROMPT_TEMPLATE_VERSION
from agentmemeval.prompts.experience_update import EXPERIENCE_UPDATE_PROMPT
from agentmemeval.storage.jsonl_store import JsonlStore
from agentmemeval.storage.snapshots import save_snapshot


class ArtifactManager:
    """
    功能：管理标准输出工件。
    参数：
        root：输出根目录。
        run_id：运行 ID。
        config：resolved 配置。
    返回：工件管理器。
    副作用：创建目录和基础文件。
    异常：文件系统错误由标准库抛出。
    设计说明：所有场景共享同一文件树，便于 report 命令重建。
    """

    def __init__(self, root: str | Path, run_id: str, config: dict[str, Any]) -> None:
        """
        功能：初始化工件目录。
        参数：
            root：输出根目录。
            run_id：运行 ID。
            config：resolved 配置。
        返回：无。
        副作用：创建目录、resolved_config.yaml 和 JSONL store。
        异常：文件系统错误由标准库抛出。
        设计说明：run_id 唯一时不会覆盖其他运行。
        """

        self.root = Path(root)
        self.run_id = run_id
        self.run_dir = self.root / run_id
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(
                f"运行目录已存在且非空，拒绝追加或覆盖既有工件：{self.run_dir}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "memory_snapshots").mkdir(exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)
        self.config = config
        self.events = JsonlStore(self.run_dir / "events.jsonl")
        self.hand_summaries = JsonlStore(self.run_dir / "hand_summaries.jsonl")
        self.write_text("resolved_config.yaml", dump_yaml(config))

    def manifest(self) -> RunManifest:
        """
        功能：创建运行清单对象。
        参数：无。
        返回：RunManifest。
        副作用：无。
        异常：无。
        设计说明：清单记录 seed、provider、model 和代码版本。
        """

        experiment = self.config["experiment"]
        provider = self.config["provider"]
        return RunManifest(
            run_id=self.run_id,
            scenario=str(experiment["scenario"]),
            seed=int(experiment["seed"]),
            config_snapshot_path=str(self.run_dir / "resolved_config.yaml"),
            output_dir=str(self.run_dir),
            code_version=get_code_version(Path.cwd()),
            provider=str(provider.get("provider", "mock")),
            model=str(provider.get("model", "")),
            metadata=collect_runtime_metadata(self.config, Path.cwd()),
        )

    def write_manifest(self) -> None:
        """
        功能：写入 manifest.json。
        参数：无。
        返回：无。
        副作用：写 JSON 文件。
        异常：文件系统错误由标准库抛出。
        设计说明：在实验开始时写入，失败运行也能追踪配置。
        """

        self.write_json("manifest.json", self.manifest().to_dict())

    def log_event(self, record: dict[str, Any]) -> None:
        """
        功能：追加事件日志。
        参数：
            record：事件记录。
        返回：无。
        副作用：写入 events.jsonl。
        异常：序列化错误由标准库抛出。
        设计说明：动作、换桌、rebuy 和报告重建事件都走统一日志。
        """

        self.events.append(record)

    def log_hand(self, record: dict[str, Any]) -> None:
        """
        功能：追加手牌摘要。
        参数：
            record：手牌摘要。
        返回：无。
        副作用：写入 hand_summaries.jsonl。
        异常：序列化错误由标准库抛出。
        设计说明：指标尽量从手牌摘要重建，降低事件日志体积依赖。
        """

        self.hand_summaries.append(record)

    def save_snapshot(self, agent_id: str, snapshot: MemorySnapshot, suffix: str = "final") -> str:
        """
        功能：保存 Agent 记忆快照。
        参数：
            agent_id：Agent 标识。
            snapshot：快照。
            suffix：文件后缀标签。
        返回：快照路径字符串。
        副作用：写入 JSON 文件。
        异常：文件系统错误由标准库抛出。
        设计说明：训练后和最终快照可以共存。
        """

        path = self.run_dir / "memory_snapshots" / f"{agent_id}_{suffix}.json"
        save_snapshot(path, snapshot)
        return str(path)

    def write_json(self, relative_path: str, data: dict[str, Any]) -> str:
        """
        功能：写入 JSON 文件。
        参数：
            relative_path：相对 run 目录路径。
            data：JSON 字典。
        返回：文件路径字符串。
        副作用：写文件。
        异常：文件系统或序列化错误由标准库抛出。
        设计说明：统一 ensure_ascii=False，保证中文文档可读。
        """

        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def write_text(self, relative_path: str, text: str) -> str:
        """
        功能：写入文本文件。
        参数：
            relative_path：相对路径。
            text：文本内容。
        返回：文件路径字符串。
        副作用：写文件。
        异常：文件系统错误由标准库抛出。
        设计说明：报告、配置快照和说明文档统一 UTF-8。
        """

        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

    def finish(self, result: ExperimentResult) -> None:
        """
        功能：写入最终结果摘要。
        参数：
            result：实验结果。
        返回：无。
        副作用：写 experiment_result.json。
        异常：文件系统错误由标准库抛出。
        设计说明：CLI 可以读取该文件快速显示运行结果。
        """

        self.write_json("experiment_result.json", result.to_dict())


def make_run_id(scenario: str, seed: int) -> str:
    """
    功能：生成运行 ID。
    参数：
        scenario：场景名称。
        seed：根 seed。
    返回：运行 ID 字符串。
    副作用：读取系统时间。
    异常：无。
    设计说明：包含场景和 seed，便于人工浏览 outputs。
    """

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    clean = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in scenario)
    return f"{stamp}_{clean}_seed{seed}"


def get_code_version(cwd: Path) -> str:
    """
    功能：获取当前 Git commit。
    参数：
        cwd：工作目录。
    返回：commit hash 或 unknown。
    副作用：调用 git 子进程。
    异常：内部捕获所有失败并返回 unknown。
    设计说明：本工作区可能不是标准 Git 仓库，必须显式标记 unknown。
    """

    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={cwd.resolve().as_posix()}", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def collect_runtime_metadata(config: dict[str, Any], cwd: Path) -> dict[str, Any]:
    """Collect reproducibility metadata without persisting API keys or other secrets."""

    provider = dict(config.get("provider", {}))
    runtime: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "code": {
            "commit": get_code_version(cwd),
            "dirty": _git_dirty(cwd),
        },
        "model": {
            "name": provider.get("model"),
            "revision": provider.get("model_revision"),
            "weights_hash": provider.get("model_weights_hash"),
        },
        "service": {
            key: provider.get(key)
            for key in (
                "provider", "temperature", "max_output_tokens", "timeout_seconds",
                "max_retries", "structured_output_mode", "rate_limit_policy",
                "service_startup_parameters",
            )
            if key in provider
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
        "packages": {
            name: _package_version(name) for name in ("torch", "vllm", "treys")
        },
        "gpu": _gpu_metadata(),
    }
    try:
        import torch

        runtime["cuda"] = {
            "torch_cuda_version": torch.version.cuda,
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
    except Exception as exc:  # noqa: BLE001
        runtime["cuda"] = {"available": False, "collection_error": type(exc).__name__}
    return runtime


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_dirty(cwd: Path) -> bool | None:
    try:
        result = subprocess.run(
            [
                "git", "-c", f"safe.directory={cwd.resolve().as_posix()}",
                "status", "--porcelain",
            ],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def _gpu_metadata() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        devices = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                devices.append({"name": parts[0], "driver": parts[1], "pci_bus_id": parts[2]})
        return {"devices": devices}
    except Exception as exc:  # noqa: BLE001
        return {"devices": [], "collection_error": type(exc).__name__}
