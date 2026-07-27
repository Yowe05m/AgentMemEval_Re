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


def _safe_path_component(value: str) -> str:
    """Create a readable unique path component without accepting traversal syntax."""

    raw = str(value)
    clean = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw).strip("._")
    clean = clean[:80] or "agent"
    if clean == raw:
        return clean
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{clean}_{digest}"


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

    from agentmemeval.memory.experiential import (
        EXPERIENCE_DOCUMENT_CONTRACT_VERSION,
        EXPERIENCE_FULL_DOCUMENT_PROMPT_VERSION,
        EXPERIENCE_FULL_DOCUMENT_REVISION_SCHEMA_VERSION,
        EXPERIENCE_LEGACY_FULL_DOCUMENT_PROMPT_VERSION,
        EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION,
        EXPERIENCE_PAPER_FULL_DOCUMENT_PROMPT_VERSION,
        EXPERIENCE_PAPER_REVISION_SYSTEM_TEMPLATE,
        EXPERIENCE_PATCH_PROMPT_VERSION,
        EXPERIENCE_PATCH_REVISION_SCHEMA_VERSION,
        EXPERIENCE_PATCH_REVISION_SYSTEM_TEMPLATE,
        EXPERIENCE_REVISION_SYSTEM_TEMPLATE,
        EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION,
        EXPERIENCE_TRANSFERABLE_REVISION_SYSTEM_TEMPLATE,
    )
    from agentmemeval.prompts.experience_update import (
        EXPERIENCE_PAPER_UPDATE_PROMPT,
        EXPERIENCE_PATCH_UPDATE_PROMPT,
        EXPERIENCE_TRANSFERABLE_QUALITY_PROMPT,
        EXPERIENCE_UPDATE_PROMPT,
    )

    provider = dict(config.get("provider", {}))
    agent = dict(config.get("agent", {}))
    experiment = dict(config.get("experiment", {}))
    experience_semantics_mode = str(
        agent.get(
            "experience_semantics_mode",
            agent.get("memory_semantics_mode", "quality_filtered_v2"),
        )
    )
    revision_contract = str(
        agent.get(
            "experience_revision_contract",
            (
                "full_document_v1"
                if experience_semantics_mode == "paper_faithful_v1"
                else "structured_patch_v2"
            ),
        )
    )
    full_document_revision = revision_contract == "full_document_v1"
    document_contract = str(
        agent.get(
            "experience_document_contract",
            EXPERIENCE_DOCUMENT_CONTRACT_VERSION,
        )
    )
    if not full_document_revision:
        active_experience_prompt = EXPERIENCE_PATCH_UPDATE_PROMPT
        active_revision_prompt_version = EXPERIENCE_PATCH_PROMPT_VERSION
        active_revision_system_template = EXPERIENCE_PATCH_REVISION_SYSTEM_TEMPLATE
    elif document_contract == EXPERIENCE_PAPER_DOCUMENT_CONTRACT_VERSION:
        active_experience_prompt = EXPERIENCE_PAPER_UPDATE_PROMPT
        active_revision_prompt_version = EXPERIENCE_PAPER_FULL_DOCUMENT_PROMPT_VERSION
        active_revision_system_template = EXPERIENCE_PAPER_REVISION_SYSTEM_TEMPLATE
    elif document_contract == EXPERIENCE_TRANSFERABLE_DOCUMENT_CONTRACT_VERSION:
        # This is the exact common update instruction assembled at runtime for
        # transferable full-document revisions.  The quality prompt is not a
        # generic advisory: it is always appended after the base instruction.
        active_experience_prompt = (
            f"{EXPERIENCE_UPDATE_PROMPT}\n\n{EXPERIENCE_TRANSFERABLE_QUALITY_PROMPT}"
        )
        active_revision_prompt_version = EXPERIENCE_FULL_DOCUMENT_PROMPT_VERSION
        active_revision_system_template = EXPERIENCE_TRANSFERABLE_REVISION_SYSTEM_TEMPLATE
    else:
        # Preserve the original five-section full-document contract and its
        # historical prompt identity for legacy configurations.
        active_experience_prompt = EXPERIENCE_UPDATE_PROMPT
        active_revision_prompt_version = EXPERIENCE_LEGACY_FULL_DOCUMENT_PROMPT_VERSION
        active_revision_system_template = EXPERIENCE_REVISION_SYSTEM_TEMPLATE
    active_revision_schema_version = (
        EXPERIENCE_FULL_DOCUMENT_REVISION_SCHEMA_VERSION
        if full_document_revision
        else EXPERIENCE_PATCH_REVISION_SCHEMA_VERSION
    )
    behavior_thresholds = experiment.get("behavior_thresholds")
    behavior_threshold_sha256 = (
        hashlib.sha256(
            json.dumps(
                behavior_thresholds,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if isinstance(behavior_thresholds, dict)
        else None
    )
    runtime: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "protocol": {
            "run_mode": experiment.get("run_mode", "smoke"),
            "protocol_readiness": experiment.get("protocol_readiness"),
            "admission_audit": experiment.get("admission_audit", {}),
            "behavior_threshold_status": experiment.get("behavior_threshold_status"),
            "behavior_thresholds": behavior_thresholds,
            "behavior_threshold_sha256": behavior_threshold_sha256,
            "behavior_threshold_source": experiment.get("behavior_threshold_source"),
            "behavior_threshold_freeze_label": experiment.get(
                "behavior_threshold_freeze_label"
            ),
        },
        "code": {
            "commit": get_code_version(cwd),
            "dirty": _git_dirty(cwd),
        },
        "model": {
            "name": provider.get("model"),
            "revision": provider.get("model_revision"),
            "weights_hash": provider.get("model_weights_hash"),
            "tokenizer_revision": provider.get("model_tokenizer_revision"),
            "served_model_name": provider.get("served_model_name"),
        },
        "embedding": {
            "backend": agent.get("embedding_backend"),
            "name": agent.get("embedding_model"),
            "revision": agent.get("embedding_revision"),
            "weights_hash": agent.get("embedding_weights_hash"),
            "tokenizer_revision": agent.get("embedding_tokenizer_revision"),
            "query_policy": agent.get("embedding_query_policy"),
            "hybrid_weights": agent.get("embedding_hybrid_weights"),
            "candidate_depth": agent.get("embedding_candidate_depth"),
            "colbert_rerank_depth": agent.get("embedding_colbert_rerank_depth"),
            "final_top_k_policy": agent.get("embedding_final_top_k_policy"),
            "cache_schema_version": agent.get("embedding_cache_schema_version"),
            "cache_namespace_template": agent.get("embedding_cache_path"),
            "service_startup_parameters": agent.get(
                "embedding_service_startup_parameters"
            ),
        },
        "service": {
            key: provider.get(key)
            for key in (
                "provider", "temperature", "max_output_tokens",
                "experience_max_output_tokens",
                "experience_repair_max_output_tokens", "timeout_seconds",
                "max_retries", "structured_output_mode",
                "experience_structured_output_mode", "thinking_mode",
                "rate_limit_policy",
                "service_startup_parameters", "served_model_name",
                "runtime_probe_python",
            )
            if key in provider
        },
        "prompts": {
            "decision_version": PROMPT_TEMPLATE_VERSION,
            "decision_system_sha256": hashlib.sha256(
                BASE_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "experience_update_sha256": hashlib.sha256(
                active_experience_prompt.encode("utf-8")
            ).hexdigest(),
            "experience_revision_prompt_version": active_revision_prompt_version,
            "experience_revision_schema_version": active_revision_schema_version,
            "experience_document_contract_version": document_contract,
            "experience_revision_system_template_sha256": hashlib.sha256(
                active_revision_system_template.encode("utf-8")
            ).hexdigest(),
            "experience_revision_semantics": (
                "keep_authoritative_fail_closed_no_truncation"
            ),
            "experience_dynamic_request_sha256_recorded": True,
            "experience_dynamic_request_sha256_location": (
                "memory_snapshot.revision_log[].prompt_sha256"
            ),
        },
        "experience": {
            "revision_strategy": agent.get(
                "experience_revision_strategy", "deterministic"
            ),
            "window_size": int(agent.get("window_size", 8)),
            "semantics_mode": experience_semantics_mode,
            "revision_contract": revision_contract,
            "document_contract": document_contract,
            "failure_policy": agent.get(
                "experience_failure_policy", "fail_closed_keep"
            ),
            "max_chars": int(agent.get("experience_max_chars", 2400)),
            "prompt_token_budget": int(
                agent.get("experience_prompt_token_budget", 32768)
            ),
            "update_period": int(agent.get("experience_update_period", 1)),
            "max_output_tokens": int(
                provider.get("experience_max_output_tokens", 3072)
            ),
            "repair_max_output_tokens": int(
                provider.get("experience_repair_max_output_tokens", 2048)
            ),
        },
        "packages": {
            name: _package_version(name) for name in ("torch", "vllm", "treys")
        },
        "gpu": _gpu_metadata(),
        "model_service_runtime": _service_runtime_metadata(provider),
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


def _service_runtime_metadata(provider: dict[str, Any]) -> dict[str, Any]:
    """Probe the Python environment that actually hosts the model service."""

    executable = provider.get("runtime_probe_python")
    if not executable:
        return {"status": "not_configured"}
    probe = (
        "import importlib.metadata as metadata,json,platform,torch;"
        "print(json.dumps({"
        "'python':platform.python_version(),"
        "'torch_version':torch.__version__,"
        "'torch_cuda_version':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),"
        "'cuda_device_count':torch.cuda.device_count(),"
        "'vllm_version':metadata.version('vllm')"
        "},sort_keys=True))"
    )
    try:
        result = subprocess.run(
            [str(executable), "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        data = json.loads(result.stdout)
        if not isinstance(data, dict):
            raise ValueError("service runtime probe did not return a JSON object")
        return {"status": "verified", "python_executable": str(executable), **data}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "python_executable": str(executable),
            "collection_error": type(exc).__name__,
        }


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
