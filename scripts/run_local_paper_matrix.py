"""
运行本地 LM Studio 论文规模实验矩阵，并在每个实验结束后写结果文档。

用法：
    python scripts/run_local_paper_matrix.py

说明：
    - 从 .env 读取 LOCAL_LLM_BASE_URL / LOCAL_LLM_API_KEY。
    - 顺序执行 configs/experiments/local_paper_*.yaml。
    - 每个实验完成或失败都会写 docs/0710_xx_*.md。
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from agentmemeval.config.loader import load_config
from agentmemeval.experiments.runner import run_config
from agentmemeval.llm.router import provider_health

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CONFIGS: list[tuple[str, str, str]] = [
    (
        "Exp1-NoMemory",
        "local_paper_exp1_no_memory",
        "configs/experiments/local_paper_exp1_no_memory.yaml",
    ),
    (
        "Exp1-FactAgent",
        "local_paper_exp1_fact_agent",
        "configs/experiments/local_paper_exp1_fact_agent.yaml",
    ),
    (
        "Exp1-ExprAgent",
        "local_paper_exp1_expr_agent",
        "configs/experiments/local_paper_exp1_expr_agent.yaml",
    ),
    (
        "Exp1-FactExprSync",
        "local_paper_exp1_fact_expr_sync",
        "configs/experiments/local_paper_exp1_fact_expr_sync.yaml",
    ),
    (
        "Exp1-FactExprAsync",
        "local_paper_exp1_fact_expr_async",
        "configs/experiments/local_paper_exp1_fact_expr_async.yaml",
    ),
    (
        "Exp2-Persona-INTJ",
        "local_paper_exp2_persona_intj",
        "configs/experiments/local_paper_exp2_persona_intj.yaml",
    ),
    (
        "Exp2-Persona-ENFP",
        "local_paper_exp2_persona_enfp",
        "configs/experiments/local_paper_exp2_persona_enfp.yaml",
    ),
    (
        "Exp2-Persona-ISTP",
        "local_paper_exp2_persona_istp",
        "configs/experiments/local_paper_exp2_persona_istp.yaml",
    ),
    (
        "Exp2-Persona-ESFJ",
        "local_paper_exp2_persona_esfj",
        "configs/experiments/local_paper_exp2_persona_esfj.yaml",
    ),
]


def main() -> int:
    os.chdir(ROOT)
    _load_env(ROOT / ".env")
    DOCS.mkdir(exist_ok=True)
    _write_controller_doc("running", [])
    completed: list[dict[str, Any]] = []
    for index, (title, slug, config_path) in enumerate(CONFIGS, start=2):
        started_at = datetime.now().isoformat(timespec="seconds")
        print(f"[{started_at}] START {title} -> {config_path}", flush=True)
        try:
            config = load_config(ROOT / config_path)
            result = run_config(ROOT / config_path)
            payload = {
                "title": title,
                "slug": slug,
                "config_path": config_path,
                "status": "success",
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "resolved_config": config,
                "result": result.to_dict(),
            }
            completed.append(payload)
            _write_experiment_doc(index, payload)
            _write_controller_doc("running", completed)
            print(f"[{payload['finished_at']}] DONE {title}", flush=True)
        except Exception as exc:  # noqa: BLE001
            payload = {
                "title": title,
                "slug": slug,
                "config_path": config_path,
                "status": "failed",
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            completed.append(payload)
            _write_experiment_doc(index, payload)
            _write_controller_doc("blocked", completed)
            print(f"[{payload['finished_at']}] FAILED {title}: {exc!r}", flush=True)
            return 2
    _write_controller_doc("complete", completed)
    return 0


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _write_controller_doc(status: str, completed: list[dict[str, Any]]) -> None:
    health = _safe_health()
    lines = [
        "# 本地论文规模实验总控记录",
        "",
        f"更新时间：{datetime.now().isoformat(timespec='seconds')}",
        f"状态：{status}",
        "",
        "## Provider",
        "",
        "```json",
        json.dumps(health, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 实验矩阵",
        "",
        "| 序号 | 实验 | 配置 | 状态 | 结果文档 |",
        "| ---: | --- | --- | --- | --- |",
    ]
    by_slug = {item["slug"]: item for item in completed}
    for index, (title, slug, config_path) in enumerate(CONFIGS, start=2):
        item = by_slug.get(slug)
        item_status = item["status"] if item else "pending"
        doc_name = f"0710_{index:02d}_{slug}_结果记录.md"
        lines.append(
            f"| {index - 1} | {title} | `{config_path}` | {item_status} | `{doc_name}` |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 本矩阵使用本地 LM Studio OpenAI-compatible 服务，不等价于原论文 API 模型。",
            "- 规模按论文规格设置为 train=150、test=25；Exp1 泛化对手数为 7，Exp2 泛化对手数为 3。",
            "- 若某项失败，脚本会停止，并在对应结果文档中记录错误栈。",
            "",
        ]
    )
    (DOCS / "0710_01_本地论文规模实验总控记录.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_experiment_doc(index: int, payload: dict[str, Any]) -> None:
    path = DOCS / f"0710_{index:02d}_{payload['slug']}_结果记录.md"
    lines = [
        f"# {payload['title']} 结果记录",
        "",
        f"开始时间：{payload['started_at']}",
        f"结束时间：{payload['finished_at']}",
        f"状态：{payload['status']}",
        f"配置：`{payload['config_path']}`",
        "",
    ]
    if payload["status"] == "success":
        result = payload["result"]
        metrics = result["metrics"]
        target = metrics.get("primary_metrics", {}).get("per_agent", {}).get("agent_00", {})
        lines.extend(
            [
                "## 输出工件",
                "",
                f"- run_dir：`{result['artifacts'].get('run_dir')}`",
                f"- report：`{result['artifacts'].get('report')}`",
                f"- plot：`{result['artifacts'].get('plot')}`",
                "",
                "## 核心指标",
                "",
                f"- run_id：`{result['run_id']}`",
                f"- scenario：`{result['scenario']}`",
                f"- hands：`{metrics.get('run_counters', {}).get('hands')}`",
                f"- actions：`{metrics.get('run_counters', {}).get('actions')}`",
                f"- agents：`{metrics.get('run_counters', {}).get('agents')}`",
                f"- agent_00 chip_delta：`{target.get('chip_delta')}`",
                f"- agent_00 bb_per_100：`{target.get('bb_per_100')}`",
                f"- agent_00 win_rate：`{target.get('win_rate')}`",
                f"- agent_00 vpip：`{target.get('vpip')}`",
                f"- agent_00 raise_rate：`{target.get('raise_rate')}`",
                "",
                "## Memory 指标",
                "",
                "```json",
                json.dumps(target.get("memory", {}), ensure_ascii=False, indent=2),
                "```",
                "",
                "## 运行摘要",
                "",
                "```json",
                json.dumps(result["aggregate_metrics"], ensure_ascii=False, indent=2),
                "```",
                "",
                "## 本次代码或配置变化记录",
                "",
                "- 本轮实验使用 `configs/experiments/local_paper_*.yaml` 论文规模本地配置。",
                "- 本地 Provider 使用 `structured_output_mode: json_schema` 兼容 LM Studio。",
                "- 运行脚本：`scripts/run_local_paper_matrix.py`。",
            ]
        )
    else:
        lines.extend(
            [
                "## 失败信息",
                "",
                f"- error：`{payload.get('error')}`",
                "",
                "## Traceback",
                "",
                "```text",
                payload.get("traceback", ""),
                "```",
                "",
                "## 待处理",
                "",
                "- 该失败已阻塞后续矩阵运行，需要先修复再继续。",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _safe_health() -> dict[str, object]:
    try:
        config = load_config(ROOT / "configs/experiments/local_base_small.yaml")
        return provider_health(config["provider"])
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}


if __name__ == "__main__":
    raise SystemExit(main())
