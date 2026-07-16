# AgentMemEval rebuild repository instructions

## Scope and tracked content

- 本仓库只承载产品代码、配置、测试、README、包元数据和本文件。
- `docs/`、`scripts/`、`outputs/`、`.env`、`.venv`、缓存和 `tmp/` 保持 Git ignored。
- 核心运行逻辑必须进入 `src/agentmemeval/`；正式实验参数必须进入 `configs/`；不可依赖仅存在于 ignored `scripts/` 的关键逻辑。
- `AgentMemEval-main` 是仓库外的只读语义校准基线，不在本仓库内复制或修改。

## Git workflow

- 本地是开发与验证入口，GitHub `origin/main` 是权威版本中心，AutoDL 只通过 Git 拉取代码。
- 提交前检查 `git status -sb`、完整 diff 和 staged 文件；保护已有用户改动。
- 推送前运行相关 pytest、`ruff check .` 和 `compileall`。测试失败不得推送。
- 只有用户明确授权时才直接推送 `main`。
- AutoDL 使用 `git fetch origin main` 和 `git pull --ff-only origin main`；禁止自动 merge、rebase、`reset --hard` 或日常压缩包覆盖。
- 同步后核对本地、GitHub 和服务器完整 commit SHA 一致。

## Server and results

- AutoDL 项目路径：`/root/autodl-tmp/agentmemeval_rebuild`。
- 项目环境：`/root/autodl-tmp/envs/agentmemeval/bin/python`；vLLM 使用独立环境。
- 实验结果保留在服务器 `outputs/`，不进 Git。
- 结果回收使用服务器 `.tar.gz` + 文件清单 + 逐文件 SHA-256，再通过 `scp` 下载到本地日期归档。
- 不删除服务器原始结果；partial/incomplete run 必须保留并单独标记。

## Correctness and validation

- pytest 使用项目内 `--basetemp tmp/<unique_name>`，避免 Windows 临时目录问题。
- 修改扑克规则、记忆、指标、统计或实验生命周期时必须增加回归测试。
- 禁止把 hash embedding、deterministic revision、旧 roster 或 skipped cross-check 描述为论文级验证完成。
- 报告明确区分 verified、implemented-only、blocked/deferred；未解决方法学 P0/P1 时保持 NO-GO。
- run 目录必须隔离；非空目录不得复用，重试命名为 `__attempt_NN`。

## Naming

- Python 模块、函数、变量使用 snake_case；类使用 PascalCase；常量使用 UPPER_SNAKE_CASE。
- YAML 使用小写 snake_case，并体现 scenario、mechanism 和用途。
- run 名包含 UTC 时间、scenario、seed 或 mechanism；不得使用 `final_final`、`new_folder` 等临时名称。
