# AgentMemEval Rebuild

论文 *An Empirical Study of Memory Mechanisms in Agentic Systems* 复现 + 扩展中。

## 项目状态

| 能力 | 当前状态 |
| --- | --- |
| Python 包安装 | `pyproject.toml` + `src/` 布局，支持 editable install |
| CLI | `doctor`、`run`、`report` 三个命令 |
| 离线 Provider | 默认 `mock`，无需密钥即可跑实验和测试 |
| 真实 Provider | 提供 `openai_compatible` 骨架，通过环境变量接入 |
| 本地扑克环境 | 覆盖核心 Hold'em 流程、合法动作、摊牌和结算 |
| 记忆机制 | NoMemory、Fact、Expr、FactExprSync、FactExprAsync |
| 人格机制 | 内置 INTJ、ENFP、ISTP、ESFJ 示例，可配置扩展 |
| 换桌实验 | 支持 20 Agent 轮换桌、暴露统计和 pairwise histogram |
| 报告闭环 | 每次运行写入 JSONL、快照、指标、图表和 `report.md` |

## 快速开始

需要 Python 3.10 或更高版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

检查离线 Provider 和开发环境：

```powershell
python -m agentmemeval doctor --provider mock
python -m pytest
ruff check src tests
```

不在 Windows PowerShell 中时，只需要把虚拟环境激活命令替换为本机平台对应命令，其余命令保持一致。

## 运行离线实验

固定桌 FactAgent smoke：

```powershell
python -m agentmemeval run --config configs/experiments/paper_fact_agent_mock.yaml
```

固定桌 ExprAgent smoke：

```powershell
python -m agentmemeval run --config configs/experiments/paper_expr_agent_mock.yaml
```

带人格 wrapper 的 INTJ 示例：

```powershell
python -m agentmemeval run --config configs/experiments/persona_intj_mock.yaml
```

20 Agent 换桌实验：

```powershell
python -m agentmemeval run --config configs/experiments/rotating_20_agents_mock.yaml
```

固定换桌对照配置：

```powershell
python -m agentmemeval run --config configs/experiments/rotating_20_agents_fixed_control_mock.yaml
```

从已有输出重新生成指标、图表和报告：

```powershell
python -m agentmemeval report --input outputs/<run_id>
```

## 输出工件

每次 `run` 会创建一个独立的 `outputs/<run_id>/` 目录：

```text
outputs/<run_id>/
|- manifest.json
|- resolved_config.yaml
|- events.jsonl
|- hand_summaries.jsonl
|- memory_snapshots/
|- metrics.json
|- aggregate_metrics.json
|- plots/
`- report.md
```

换桌场景会额外写入 `exposure_stats.json`，用于记录对手暴露次数、暴露熵、pairwise histogram 和不均衡度。`report` 命令会从这些原始工件重建报告，不需要重新跑实验。

## 架构概览

```text
src/agentmemeval/
|- agents/        # Agent 构建、LLM 决策管线和 persona 配置
|- analysis/      # 图表生成
|- cli/           # doctor/run/report 命令
|- config/        # YAML 加载、继承和解析
|- core/          # 领域对象、协议、异常和 seed
|- environment/   # 本地 Hold'em 环境、动作保护、可见性边界
|- evaluation/    # 指标、聚合和报告生成
|- experiments/   # 固定桌、泛化、换桌和 runner
|- llm/           # Provider 抽象、mock、openai-compatible
|- memory/        # 事实记忆、经验记忆、同步/异步组合机制
|- prompts/       # 决策、记忆筛选、经验更新提示模板
`- storage/       # JSONL、快照和标准工件写入
```

## 使用真实 Provider

目前项目处于 `mock` 阶段。如需接入 OpenAI-compatible 服务，需先提供环境变量，再运行 doctor 或实验配置。

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
python -m agentmemeval doctor --provider openai_compatible --config configs/providers/openai_compatible.yaml
```

`configs/providers/openai_compatible.yaml` 使用 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 读取密钥与服务地址。真实 Provider 运行会受到模型、速率限制、费用和结构化输出稳定性的影响；没有密钥时，本项目不会把离线路径伪装成真实在线复现。

## 复现边界

- `configs/base.yaml` 保留论文默认语义；smoke 配置故意使用较小手数，便于快速离线验证。
- 本地 Hold'em 环境覆盖核心评估流程，但还不是完整赌场级扑克引擎；复杂边池、淘汰赛结构等属于后续增强。
- 经验记忆更新当前采用确定性摘要，便于测试稳定；后续可以替换为真实 LLM 修订器。
- 换桌 scheduler 只声明尽量均衡，不声明严格 pairwise balanced；实际均衡程度以 `exposure_stats.json` 和报告为准。

## 开发检查清单

提交或分享实验结果前，建议至少运行：

```powershell
python -m agentmemeval doctor --provider mock
python -m pytest
ruff check src tests
python -m agentmemeval run --config configs/experiments/paper_fact_agent_mock.yaml
python -m agentmemeval run --config configs/experiments/rotating_20_agents_mock.yaml
```

如果只修改文档，可以跳过实验命令；如果修改了环境、记忆、实验调度或报告生成，建议同时验证固定桌和换桌路径。
