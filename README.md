# AgentMemEval 重构版

这是一个与官方仓库物理隔离的 AgentMemEval 独立实现，根目录为 `agentmemeval_rebuild/`。目标是提供可安装、可运行、可测试、可扩展的 LLM Agent Memory 实验平台。

## 已实现能力

- NoMemory、Fact、Expr、FactExprSync、FactExprAsync。
- 人格驱动 wrapper，内置 INTJ / ENFP / ISTP / ESFJ 示例，可配置任意 persona。
- 本地 Texas Hold'em 环境：盲注、轮流行动、合法动作、发公共牌、摊牌和结算。
- 结构化 LLM Provider 抽象、离线 mock、openai-compatible 骨架和真实厂商占位。
- 固定桌训练、训练后快照、泛化测试。
- 20 Agent 换桌 smoke、轮空记录、对手暴露次数、暴露熵和 pairwise histogram。
- 标准工件、报告重建、图表和 pytest 闭环。

## 安装与检查

```bash
python -m pip install -e ".[dev]"
python -m agentmemeval doctor --provider mock
pytest
ruff check src tests
```

## 运行实验

固定桌训练与泛化测试：

```bash
python -m agentmemeval run --config configs/experiments/paper_fact_agent_mock.yaml
```

20 Agent 换桌实验：

```bash
python -m agentmemeval run --config configs/experiments/rotating_20_agents_mock.yaml
```

固定桌对照式换桌配置：

```bash
python -m agentmemeval run --config configs/experiments/rotating_20_agents_fixed_control_mock.yaml
```

从已有工件重建报告：

```bash
python -m agentmemeval report --input outputs/<run_id>
```

## 目录结构

```text
agentmemeval_rebuild/
├── configs/
├── docs/
├── src/agentmemeval/
│   ├── agents/
│   ├── analysis/
│   ├── cli/
│   ├── config/
│   ├── core/
│   ├── environment/
│   ├── evaluation/
│   ├── experiments/
│   ├── llm/
│   ├── memory/
│   ├── prompts/
│   └── storage/
├── tests/
├── outputs/
└── tmp/
```

## 输出工件

每次 run 生成：

```text
manifest.json
resolved_config.yaml
events.jsonl
hand_summaries.jsonl
memory_snapshots/
metrics.json
aggregate_metrics.json
plots/
report.md
```

换桌场景额外生成 `exposure_stats.json`。

## 真实 Provider

默认测试只使用 `mock`。`openai_compatible` 可通过环境变量接入兼容接口：

```bash
set OPENAI_API_KEY=...
set OPENAI_BASE_URL=https://api.openai.com/v1
python -m agentmemeval doctor --provider openai_compatible --config configs/providers/openai_compatible.yaml
```

OpenAI、Anthropic、Google、xAI、DeepSeek、Qwen 已保留注册位，但当前未用真实密钥验证。

## 当前限制

- 本地扑克环境覆盖核心实验流程，不是完整赌场级引擎；复杂边池规则待增强。
- 经验更新使用确定性摘要，真实 LLM 经验修订器属于后续扩展。
- smoke 配置使用小手数；论文默认训练 150 手、测试 25 手可通过配置放大。
