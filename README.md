# AgentMemEval Rebuild

论文 *An Empirical Study of Memory Mechanisms in Agentic Systems* 复现 + 扩展中。

## 项目状态

| 能力 | 当前状态 |
| --- | --- |
| Python 包安装 | `pyproject.toml` + `src/` 布局，支持 editable install |
| CLI | `doctor`、`run`、`campaign`、`campaign-aggregate`、`pilot-plan`、`pilot-freeze`、`formal-freeze`、`report` |
| 离线 Provider | 默认 `mock`，无需密钥即可跑实验和测试 |
| 真实 Provider | 提供 `openai_compatible` 骨架，通过环境变量接入 |
| 本地扑克环境 | 覆盖 no-limit Hold'em 合法动作、加注重开、all-in、边池、摊牌和筹码守恒 |
| 记忆机制 | NoMemory、Fact、Expr、FactExprSync、FactExprAsync |
| 人格机制 | 内置 INTJ、ENFP、ISTP、ESFJ 示例，可配置扩展 |
| 换桌实验 | 支持 20 Agent 轮换桌、暴露统计和 pairwise histogram |
| Campaign | append-only `state.tsv`、唯一 attempt、断点续跑、隔离并行 leaf、同质性与配对聚合 |
| 报告闭环 | 每次运行写入 JSONL、快照、指标、图表和 `report.md` |

论文实验运行分为三个显式级别：`smoke` 仅验证工程链路并永久标记
`not_for_paper`；`pilot` 用于独立阈值校准，但要求真实模型身份和语义
embedding；`formal` 在创建 run 目录前对代码、模型、硬件、服务、检索阈值、
行为阈值和统计方案执行 fail-closed 准入。

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

论文 Exp1 混合桌协议 smoke（Fact/Expr/Sync/Async 各 2 个 Agent、checkpoint 泛化）：

```powershell
python -m agentmemeval run --config configs/experiments/paper_exp1_mixed_mock.yaml
```

本地 Qwen 决策与 LLM experience revision 配置见
`configs/experiments/paper_exp1_mixed_local.yaml`。正式约束已经固定为
Qwen3-Embedding-4B、每个 checkpoint 50 个 heldout hands、禁用策略风险门控；
在 embedding 服务 smoke 和统一硬件验证完成前仍不能直接进入论文主表。
当前该配置是 `pilot`，检索阈值和行为阈值仍等待独立 pilot 校准。审核表
A7-R 已预注册为同 seed 的 table/run 级机制配对效应：同桌同机制 Agent
先取算术平均，再与 `fact` 基线比较；主要终点为 final-test BB/100，
机制族比较使用 Holm 校正。`statistical_plan_status` 仍保持
`pending_pilot_power_calibration`，需由独立 pilot 冻结 seed 数后才可进入正式运行。

Fact 系记忆采用分级写入准入：fallback、动作类型改写和无信息的零收益单次
翻前弃牌不会进入事实库，但拒绝原因会保存在记忆审计中；近期相同结构签名只
累计重复计数。检索 top-k 是上限，支持最低分数、空检索和结构签名多样性。
阈值未通过独立 pilot 冻结前不得运行 formal。

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

运行 TASK4 多 seed campaign（身份、真实服务 smoke、阈值或统计计划未满足时会在
创建 run 目录前 fail-closed）：

```powershell
python -m agentmemeval campaign --config configs/campaigns/task4_campaign_p_pilot.yaml
python -m agentmemeval campaign --config configs/campaigns/task4_campaign_e_pilot.yaml
```

事实记忆防自强化修复必须先经过真实 decision/embedding 双服务短 smoke。smoke 完成后
使用排他 gate 核对代码与 prompt 身份、execution health、fallback、事实快照 schema、
单次 preflop fold 拒收策略、旧模型理由泄漏和系统性行为坍塌：

```powershell
python tools/task4/gate_memory_debias_smoke.py `
  --run-dir outputs/task4_memory_debias_smoke/task4_campaign_p_memory_debias_smoke_s2026071899 `
  --output outputs/task4_memory_debias_smoke/memory_debias_smoke_gate_<utc>.json `
  --expected-code-sha <full-commit-sha>
```

只有 gate 状态为 `ready_to_start_v5_pilot` 才能启动对应 V5 calibration Pilot；该状态
不授予 formal 或 paper eligibility，Pilot 仍需独立行为、检索人工审查和功效冻结。

断点续跑只处理未完成/失败条件；标准工件完整的 completed run 不会重跑：

```powershell
python -m agentmemeval campaign --config configs/campaigns/task4_campaign_e_pilot.yaml --resume
```

只从 immutable manifest、append-only state 和 run 工件重建新的版本化聚合：

```powershell
python -m agentmemeval campaign-aggregate --input outputs/campaigns/<campaign_id>
```

完整独立 Pilot 后，先从版本化 P/E aggregate 生成固定 MDE 5 BB/100、敏感性
3/5/10、alpha 0.05、power 0.80 的功效计划；计划会对所有 P/E 对比取最大所需
seed，且不会按资源静默截断：

```powershell
python -m agentmemeval pilot-plan `
  --campaign-p outputs/campaigns/<p>/campaign_aggregate_<utc>.json `
  --campaign-e outputs/campaigns/<e>/campaign_aggregate_<utc>.json `
  --output outputs/campaigns/pilot_power_plan_<utc>.json
```

任一 versioned campaign aggregate 都可一键重建不覆盖旧产物的统计包，包含
`main_table.csv`、seed 级 `paired_effects.csv`、带 n 和 bootstrap 95% CI 的主终点图、
中文分析报告及逐文件 SHA-256 manifest。V2 会把资格三态写入 manifest、全部 CSV、
报告和图标题：`pilot_descriptive_only`、`formal_inference_ready`、
`blocked_or_underpowered`；只有 aggregate status 为 `ready` 时
`paper_inference_eligible=true`，其余状态显式禁止论文推断：

```powershell
python tools/task4/build_campaign_analysis.py `
  --aggregate outputs/campaigns/<campaign>/campaign_aggregate_<utc>.json `
  --output-dir outputs/campaigns/<campaign>/analysis_<utc>
```

服务器打包前和本地解压后使用同一文件级 SHA-256 manifest 工具。manifest 必须放在
被归档根目录之外；默认同时拒绝 header/行格式错误、缺失、大小/哈希不符、危险或重复
相对路径、root/manifest symlink、目录内任意 symlink 和额外文件。失败以结构化 V2
verification 报告返回，不把 malformed manifest 异常误作已验证归档：

```powershell
python tools/task4/file_manifest.py build `
  --root outputs/campaigns/<campaign> `
  --output outputs/campaigns/<campaign>_files.tsv

python tools/task4/file_manifest.py verify `
  --root outputs/R_<date>/<snapshot>/extracted/<campaign> `
  --manifest outputs/R_<date>/<snapshot>/<campaign>_files.tsv
```

已完成且停止写入的 campaign 可用单一 append-only 命令生成 tar.gz、外置文件 manifest、
GNU 风格 `.sha256` 和最后写入的 verified receipt。工具会重新校验源目录、逐 tar member
读取大小/SHA-256、拒绝危险或特殊成员，并在本地下载后提供严格 checksum 验证：

```bash
python tools/task4/snapshot_archive.py build \
  --root outputs/campaigns/<campaign> \
  --archive /root/autodl-tmp/backups/<snapshot>.tar.gz \
  --manifest /root/autodl-tmp/backups/<snapshot>.files.tsv \
  --checksum /root/autodl-tmp/backups/<snapshot>.tar.gz.sha256 \
  --receipt /root/autodl-tmp/backups/<snapshot>.receipt.json
```

```powershell
python tools/task4/snapshot_archive.py verify-checksum `
  --archive outputs/R_<date>/<snapshot>/<snapshot>.tar.gz `
  --checksum outputs/R_<date>/<snapshot>/<snapshot>.tar.gz.sha256

python tools/task4/snapshot_archive.py extract `
  --archive outputs/R_<date>/<snapshot>/<snapshot>.tar.gz `
  --checksum outputs/R_<date>/<snapshot>/<snapshot>.tar.gz.sha256 `
  --manifest outputs/R_<date>/<snapshot>/<snapshot>.files.tsv `
  --output-dir outputs/R_<date>/<snapshot>/extracted `
  --receipt outputs/R_<date>/<snapshot>/<snapshot>.extract.receipt.json
```

只有 receipt `status=verified`、下载后 checksum 通过、解压后 file-manifest V2 也通过的
快照，才能成为本地分析输入。`extract` 只写全新目录，预先拒绝路径穿越、symlink、
hardlink、special/额外/缺失成员，并在解压后再次逐文件校验。打包或解压失败时已有
partial 输出永久保留，重试必须换新名称。

回收多个 campaign 后生成一份 `server_run_map.csv` 和 formal 主表排除清单。工具按
campaign/condition/seed/attempt 折叠 lifecycle，只把完整、formal、execution valid、
paper eligible 且非 model-substituted 的 leaf 标为候选；Pilot、失败、partial 和敏感性
实验保留但分层排除。V2 只读取每个回收 campaign 自身 `runs/<run_id>` 的 canonical
leaf，不回退到 `state.tsv` 中仍可访问的服务器绝对路径；同时交叉绑定 state、manifest、
resolved config 与 experiment result 身份，要求八项标准工件、clean code、零 fallback/
守恒违规，并把每个叶工件的 SHA-256 写入 run map：

```powershell
python tools/task4/build_run_map.py `
  --campaign-dir outputs/campaigns/<p> `
  --campaign-dir outputs/campaigns/<e> `
  --output-csv outputs/R_<date>/server_run_map.csv `
  --exclusion-json outputs/R_<date>/formal_main_exclusions.json
```

资源审计从完整 leaf 的事件、state 和 manifest 重建实测动作延迟、墙钟吞吐、GPU
身份、fallback 与经验修订次数。当前事件中的 token 是 whitespace proxy，工具会强制
标为估算而非 provider usage；本地服务没有账单时货币成本也明确标为 unavailable：

```powershell
python tools/task4/build_resource_audit.py `
  --campaign-dir outputs/campaigns/<campaign> `
  --output outputs/campaigns/<campaign>/resource_audit_<utc>.json
```

先从恰好一个完整 P Pilot 和一个完整 E Pilot 的真实语义检索事件生成 240 条分层、
结果盲化的人工审查表。`blind_review.jsonl` 不含检索分数或牌局收益；review key
会绑定两个 Campaign 的 manifest、state 和每个 leaf 的 events SHA-256。人工填写
`human_labels.tsv` 后，审计命令才会按预注册精度下界和空检索率约束冻结阈值，并把
review key、标签文件和匿名化 reviewer identity 哈希写入 V2 审计。模型标签不能
冒充人工标签：

```powershell
python tools/task4/retrieval_relevance_review.py build `
  --campaign-dir outputs/campaigns/<p> `
  --campaign-dir outputs/campaigns/<e> `
  --output-dir outputs/campaigns/retrieval_review_<utc>

python tools/task4/retrieval_relevance_review.py audit `
  --review-key outputs/campaigns/retrieval_review_<utc>/review_key.json `
  --labels outputs/campaigns/retrieval_review_<utc>/human_labels.tsv `
  --output outputs/campaigns/retrieval_review_<utc>/relevance_audit.json
```

行为、执行、检索和功效的联合冻结提案只读取 P/E campaign 中 state 为
`complete` 的 leaf；partial、failed 或 interrupted run 会被排除。行为门槛只检查
`protocol_audit.json` 中预注册的 evaluation targets，不把 heldout 对手纳入目标
样本。VPIP、弃牌、主动参与、all-in、bust、检索和结构集中度采用预注册分位数裕量
及领域退化上限；单手绝对收益活动占比继续逐 target 报告，但作为结果敏感性诊断，
不把正常扑克收益方差误作策略退化。人工相关性审计缺失或任何行为硬门槛越界仍会
NO-GO：

```powershell
python -m agentmemeval pilot-freeze `
  --campaign-p outputs/campaigns/<p>/campaign_aggregate_<utc>.json `
  --campaign-e outputs/campaigns/<e>/campaign_aggregate_<utc>.json `
  --campaign-p-dir outputs/campaigns/<p> `
  --campaign-e-dir outputs/campaigns/<e> `
  --retrieval-review-audit outputs/campaigns/retrieval_review_<utc>/relevance_audit.json `
  --output outputs/campaigns/pilot_freeze_proposal_<utc>.json
```

冻结提案状态必须为 `ready_to_generate_immutable_formal_configs`。随后结合双服务
runtime lock，用 P/E campaign 的设计骨架和 robust formal 实验模板生成一个全新的、
拒绝覆盖的配置包。runtime lock 必须从已通过真实双服务准入的 Pilot run manifest
生成；旧四字段手填 lock 不再接受。它会锁定 GPU/驱动/CUDA/vLLM、decision 与
embedding revision/weights/启动参数哈希，以及两个 prompt 哈希：

```powershell
python tools/task4/build_formal_runtime_lock.py `
  --manifest outputs/campaigns/<complete_p_or_e>/runs/<run_id>/manifest.json `
  --output outputs/campaigns/formal_runtime_lock_<utc>.json
```

`formal-freeze` 会重新读取上述源 manifest 并重建 runtime lock，也会从原始 P/E
aggregate、leaf、retrieval audit 重新构建 Pilot proposal；任一手工改写都会
fail-closed。`seed-start` 必须来自预注册且不得与 Pilot seeds 重叠：

```powershell
python -m agentmemeval formal-freeze `
  --proposal outputs/campaigns/pilot_freeze_proposal_<utc>.json `
  --runtime-lock outputs/campaigns/formal_runtime_lock_<utc>.json `
  --campaign-p-template configs/campaigns/task4_campaign_p_pilot_parallel_v7_counterfactual_calibrated.yaml `
  --campaign-e-template configs/campaigns/task4_campaign_e_pilot_parallel_v7_counterfactual_calibrated.yaml `
  --formal-p-template configs/experiments/task4_campaign_p_robust_formal_template.yaml `
  --formal-e-template configs/experiments/task4_campaign_e_robust_formal_template.yaml `
  --strict-p-template configs/experiments/task4_campaign_p_strict_model_substituted.yaml `
  --strict-p-campaign-template configs/campaigns/task4_campaign_p_strict_model_substituted.yaml `
  --output-dir configs/frozen/task4_<freeze_id> `
  --freeze-id <freeze_id> `
  --seed-start <preregistered_formal_seed_start> `
  --preflight-seed <independent_extra_pilot_seed>
```

`preflight-seed` 必须与 calibration Pilot 和 formal seeds 都不重叠。输出目录包含
P/E 两份自包含 formal YAML、两份仅改变 seed、`run_mode` 与
`frozen_config_preflight` 标记的 not-for-paper preflight YAML、对应四份完整 campaign
YAML，以及一组与 robust formal 使用相同 seeds 的 strict paper-protocol/
model-substituted sensitivity experiment+campaign。strict sensitivity 保持
`run_mode=pilot`、paper-exact unthresholded retrieval 和 not-for-paper 标签，
禁止进入 robust 主表。freeze manifest 会记录全部源文件 SHA-256、runtime lock、
seed 规则和 strict 隔离策略。先完整运行
两份 preflight campaign；它们复用 formal 的 protocol/runtime 准入检查，但永久不进入
论文主表。目录已存在、提案未 ready、人工检索审计未冻结、seed 重叠、runtime lock
缺字段或模板验证失败时，命令都会 fail-closed，不能原地修改模板绕过。

Formal manifest 通过 `runtime_probe_python` 从实际 vLLM 服务环境采集 torch CUDA
和 vLLM 版本；项目运行环境无需重复安装 torch。探针缺失或与 frozen runtime
lock 不一致时，Formal 会在创建 run 目录前 fail-closed。

`task4_campaign_p_strict_model_substituted.yaml` 尽量复现论文的 150/25、每 10
手 checkpoint、elimination 和“全部事实写入”，但当前 decision model 是
Qwen3.5-9B，不能冒充原论文两模型的完全复现，也不会进入 robust formal 主表。
`task4_campaign_*_robust_formal_template.yaml` 是故意保持 NO-GO 的模板；pilot 后会
生成新的 immutable frozen config，而不是原地把模板改成“已冻结”。

Robust/Pilot 的事实记忆默认拒收仅含一次 preflop fold 的低信息轨迹，并且可检索事实
只保留可观察状态、已提交动作和结果，不把模型自己的旧 `reason_summary` 当成策略证据。
`task4_campaign_p_strict_model_substituted.yaml` 显式关闭该过滤，以保留作者协议“全部事实
写入”的敏感性边界；strict 结果不得与 robust 主表混合。

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
|- protocol_audit.json
|- async_evidence_review_queue.json
|- plots/
`- report.md
```

混合桌 checkpoint 协议还会生成 `checkpoint_generalization.json`，并在
`memory_snapshots/` 中按 `checkpoint_XXXX` 保存每个目标 Agent 的独立快照。

换桌场景会额外写入 `exposure_stats.json`，用于记录对手暴露次数、暴露熵、pairwise histogram 和不均衡度。`report` 命令会从这些原始工件重建报告，不需要重新跑实验。

## 架构概览

```text
src/agentmemeval/
|- agents/        # Agent 构建、LLM 决策管线和 persona 配置
|- analysis/      # 图表生成
|- cli/           # doctor/run/campaign/campaign-aggregate/report 命令
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

项目同时支持 mock 与 OpenAI-compatible decision/embedding 服务。真实运行需先提供
环境变量，并完成模型身份、权重指纹、启动参数和服务 smoke 记录。

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
python -m agentmemeval doctor --provider openai_compatible --config configs/providers/openai_compatible.yaml
```

`configs/providers/openai_compatible.yaml` 使用 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 读取密钥与服务地址。真实 Provider 运行会受到模型、速率限制、费用和结构化输出稳定性的影响；没有密钥时，本项目不会把离线路径伪装成真实在线复现。

## 复现边界

- `configs/base.yaml` 保留论文默认语义；smoke 配置故意使用较小手数，便于快速离线验证。
- 本地 Hold'em 环境已覆盖本项目需要的边池、all-in、加注重开、摊牌和筹码守恒；这不等于对所有赌场规则变体作出通用完备性声明。
- 经验记忆支持结构化 LLM revision 与带审计的 deterministic fallback；paper pilot/formal 要求真实 LLM 路径且未知 fallback 会使结果失效。
- 原论文使用 DeepSeek-V4-Flash 与 Qwen3.6-Flash；当前 Qwen3.5-9B 结果必须标记模型替代，不能称完全模型复现。
- hash embedding、mock、deterministic substitute 和 persona smoke 永久不进入 TASK4 论文主表。
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
