# AgentMemEval 实验与版本管理工作规范

版本：v1.1

适用范围：本地工作区、GitHub `Yowe05m/AgentMemEval_Re`、AutoDL 实验服务器

定位：本项目唯一有效的长期工作规范，也是 Codex 仓库级持久指令

## 1. 基本原则

1. 代码只通过 Git 在本地、GitHub 和服务器之间同步。
2. GitHub `origin/main` 是代码版本中心；AutoDL 只负责拉取代码和运行实验，不直接开发。
3. 实验结果不进入 Git，通过压缩包、文件清单、SHA-256 和 `scp` 回收到本地。
4. 不删除已有本地或服务器文件。移动、复制或重命名前先确认目标不存在；确需删除必须获得用户单独明确授权。
5. 代码、结果、文档、凭据、环境和缓存严格分离。
6. 程序可运行不等于论文实验可准入。始终区分 `verified`、`implemented-only`、`blocked/deferred`。

## 2. 仓库和三端职责

| 位置 | 职责 | 权威性 |
|---|---|---|
| 本地 `agentmemeval_rebuild/` | 开发、审阅、测试、提交 | 开发工作树 |
| GitHub `origin/main` | 版本控制、提交历史、服务器同步源 | 代码权威版本 |
| AutoDL `/root/autodl-tmp/agentmemeval_rebuild/` | 拉取已提交代码、运行实验 | 运行工作树 |
| AutoDL `outputs/` | 原始实验结果 | 原始结果源 |
| 本地根级 `outputs/R_YYYYMMDD/` | 下载、校验和归档后的结果 | 本地分析证据库 |

- `AgentMemEval-main/` 是论文作者原始快照，默认只读，只用于协议和语义校准。
- `agentmemeval_rebuild/` 是当前产品代码仓库。
- 禁止把源码压缩包覆盖服务器作为日常同步方式。源码压缩包仅用于灾备或明确批准的紧急恢复。

## 3. 目录职责

### 3.1 工作区根目录

```text
03_AgentMemEval/
├─ AGENTS.md                    # 仅用于指向本规范
├─ README.md                    # 工作区导航
├─ AgentMemEval-main/           # 论文作者原始快照，默认只读
├─ agentmemeval_rebuild/        # Git 产品仓库和本规范所在地
├─ docs/
│  ├─ project-background/       # 项目背景和历史需求
│  ├─ task-specifications/      # 原始任务合同
│  ├─ task-records/             # 执行、审计和验证记录
│  └─ private/                  # 私密连接资料，禁止提交
├─ references/papers/           # 论文原文
├─ outputs/R_YYYYMMDD/          # 本地结果归档
├─ scripts/                     # 工作区级运维脚本
└─ tmp/                         # 可再生成临时产物
```

### 3.2 rebuild Git 仓库

```text
agentmemeval_rebuild/
├─ AGENTS.md              # 唯一权威工作规范，随 Git 同步
├─ README.md              # 安装、运行和用户入口
├─ pyproject.toml         # 包、依赖和工具配置
├─ .env.example           # 无真实凭据的环境模板
├─ configs/
│  ├─ agents/             # Agent 和记忆机制配置
│  ├─ experiments/        # 实验协议配置
│  └─ providers/          # Provider 配置
├─ src/agentmemeval/      # 产品源码
├─ tests/                 # 单元和集成测试
├─ outputs/               # 运行结果，Git ignored
├─ docs/                  # 本地生成和审阅材料，Git ignored
├─ scripts/               # 本地运行与维护脚本，Git ignored
└─ tmp/                   # 测试和诊断临时目录，Git ignored
```

- 本仓库只追踪产品代码、配置、测试、README、包元数据和 `AGENTS.md`。
- `docs/`、`scripts/`、`outputs/`、`.env`、`.venv`、缓存和 `tmp/` 保持 Git ignored。
- 核心运行能力必须进入 `src/agentmemeval/`；正式实验参数必须进入 `configs/`。
- 产品不可依赖只存在于 ignored `scripts/` 的关键逻辑。

### 3.3 AutoDL

```text
/root/autodl-tmp/
├─ agentmemeval_rebuild/   # Git 工作树和 outputs
├─ backups/                # 源码/结果快照与校验清单
└─ envs/
   ├─ agentmemeval/        # 项目 Python 环境
   └─ vllm/                # 模型服务环境
```

- 项目 Python：`/root/autodl-tmp/envs/agentmemeval/bin/python`。
- 不得把虚拟环境、模型缓存或服务器结果移动进 Git 可追踪目录。
- 整理服务器目录前检查活跃实验和模型服务，不干扰正在写入的进程。

## 4. 代码同步流程

### 4.1 本地提交前

```powershell
cd C:\Users\YO\Documents\Codex\03_AgentMemEval\agentmemeval_rebuild
git status -sb
git fetch origin main
git rev-list --left-right --count origin/main...main
.\.venv\Scripts\python.exe -m pytest -q --basetemp tmp\pytest_basetemp_<unique>
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q src tests
```

- 修改前检查 dirty worktree，区分已有用户改动和本轮改动。
- 检查完整 diff，不能覆盖未知改动。
- 修改扑克规则、记忆、指标、统计或实验生命周期时必须增加回归测试。
- 测试失败不得推送；环境性跳过必须记录原因和影响边界。

### 4.2 提交与推送

```powershell
git add -A
git diff --cached --name-only
git commit -m "<用户约定的提交说明>"
git push origin main
```

- staged 清单不得出现 `.env`、`outputs/`、`docs/`、`scripts/`、`tmp/`、缓存或虚拟环境。
- 只有用户明确授权时才可直接推送 `main`。
- 提交和同步记录使用完整 commit SHA。

### 4.3 服务器拉取

```bash
cd /root/autodl-tmp/agentmemeval_rebuild
git status -sb
git fetch origin main
git pull --ff-only origin main
git rev-parse HEAD
```

- 服务器只允许 fast-forward；禁止自动 merge、rebase 或 `reset --hard`。
- 若服务器存在 tracked 修改、远端分叉或活跃代码写入，立即停止并生成差异/备份记录。
- 拉取后核对本地、GitHub、服务器完整 SHA 一致。
- 按风险运行 compileall、Ruff、smoke 或全量 pytest。

## 5. 服务器实验结果回收

### 5.1 回收前

- 确认 run 已完成，或明确标记为 `partial/incomplete`。
- 不在实验仍写入同一目录时制作最终快照。
- 记录服务器源路径、run 数、文件数、大小、代码 SHA、配置和协议状态。

### 5.2 快照命名

```text
agentmemeval_outputs_snapshot_YYYYMMDDTHHMMSS.tar.gz
agentmemeval_outputs_snapshot_YYYYMMDDTHHMMSS.files.tsv
agentmemeval_outputs_snapshot_YYYYMMDDTHHMMSS.sha256
```

- 快照和清单放 `/root/autodl-tmp/backups/`。
- 不删除或修改原始 `outputs/`。
- 清单至少包含相对路径、文件大小和 SHA-256。

### 5.3 下载和校验

- 使用 `scp` 下载到本地 `outputs/R_YYYYMMDD/NN_server_snapshot/`。
- 解压到独立 `extracted/` 子目录，不覆盖其他归档。
- 先校验压缩包 SHA-256，再按服务器清单逐文件复核。
- Windows 长路径使用 `\\?\` 前缀复核，不能把长路径 API 失败误报为文件缺失。
- partial run 单独保留和标注，不混入正式分析。

## 6. 命名规范

### 6.1 代码和配置

- Python 模块、函数、变量：`snake_case`。
- 类名：`PascalCase`。
- 常量：`UPPER_SNAKE_CASE`。
- YAML：小写 `snake_case.yaml`，名称体现 scenario、mechanism 和用途。
- 测试：`test_<被测行为>.py`，测试函数描述预期行为。

### 6.2 实验 run

推荐格式：

```text
YYYYMMDDTHHMMSSZ_<scenario>_seed<seed>
seed_<seed>__<mechanism>
<slug>__attempt_02
```

- 时间统一使用 UTC 并带 `Z`。
- 非空 run 目录不得复用；重试使用 `__attempt_NN`。
- 未完成目录保留原名并在清单中标记 `partial`。

### 6.3 结果和文档

- 本地日期归档：`outputs/R_YYYYMMDD/`。
- 日期内子目录：`NN_<source>_<purpose>`，如 `01_server_snapshot`。
- 任务/审计文档：`MMDD_NN_<中文标题>.md`。
- 同日编号递增，不复用编号，不使用 `final_final`、`new_folder`、`新建文件夹` 等名称。
- 历史派生物放 `generated/legacy/`，不得作为当前权威来源。

## 7. 凭据与安全

- 私密连接资料只放工作区根 `docs/private/`。
- `.env` 只留本地和服务器；Git 仅保留无真实值的 `.env.example`。
- 日志、报告、命令和提交不得包含密码、Token 或完整密钥。
- SSH 辅助脚本只能运行时读取凭据，禁止嵌入密码。

## 8. 论文实验准入与完成定义

正式实验开始前必须检查：

- 本地、GitHub、服务器代码 SHA 一致；
- resolved config、roster、seed、模型和 embedding 后端明确；
- run 目录隔离，无未知 fallback，筹码守恒；
- 统计单位、主要终点、多 seed/配对方案明确；
- 已知方法学 P0/P1 已解决，否则保持 `NO-GO`。

一次实验只有同时满足以下条件才算完成：

1. 代码 SHA 和完整 resolved config 已记录；
2. manifest、事件、手牌、快照、指标和报告齐全；
3. 失败、跳过、fallback 和协议偏差有明确记录；
4. 结果已回收到本地并通过逐文件哈希；
5. 报告区分 `verified`、`implemented-only`、`blocked/deferred`。

- 禁止把 hash embedding、deterministic revision、旧 roster 或 skipped cross-check 表述为论文级验证完成。
- 测试通过只证明已覆盖工程路径未发生已知回归，不等于论文协议正确。

## 9. Codex 执行规则

- 开始任务先读本文件、任务合同和真实仓库状态。
- 用户指定从 `TASK.md`、`TASK2.md` 或其他权威文档开始时，先读该文档再行动。
- 路径重命名必须同步代码、配置、活跃文档和服务器运行脚本中的引用。
- 优先使用本地已有论文、结果、归档和数据，不默认重新下载。
- 未经明确授权不删除文件、不推送 GitHub、不启动昂贵长跑。
- 出现服务器 tracked 修改、远端分叉、活跃写入或校验不一致时停止并报告。
- 每次交付给出准确路径、验证命令与结果，并说明尚未验证的部分。

## 10. 规范维护

- 本文件是唯一权威规范，不再另建同用途的“工作规范”文档。
- 重复错误或流程变化时直接更新本文件，并通过 Git 同步到 GitHub 和服务器。
- 工作区根 `AGENTS.md` 只作为发现本文件的加载器，不复制规则。
- 更新后新开 Codex 任务，确认已加载本文件；Codex 已运行中的任务不会自动重建指令链。
