# TASK8B Phase F 冻结分析合同

合同 ID：`task8b-phase-f-v1`  
冻结时点：正式矩阵结果揭盲前  
适用数据：TASK8B 12-seed 完整必做矩阵中通过工程与证据门禁的 R1-E1-I/M、E2、E3、E4、E5、E6

## 1. 主要终点和统计单位

主要单位为 BB/100，同时报告 chip/hand 与 raw chips。Big blind=2 时，换算必须记录公式和舍入规则，分析内部保留未舍入值。

独立统计单位是 seed。固定 confirmatory seeds 为 `2026090101–2026090112`。一个 seed 内的 hands、events、同桌 agents、H01/H02/H03、checkpoints 和 modes 是重复测量，不得增加 n。

Seed 内聚合顺序固定为：

1. 同一 run/condition/table 的多个目标 Agent 先取预注册算术均值；
2. 每个 table 先生成 table-level BB/100；
3. H01/H02/H03 三个 held-out tables 等权平均，得到 seed × mechanism × checkpoint × mode 的 Heldout；
4. Source 使用同 seed、mechanism、checkpoint 的预注册 source evaluation；若存在多个预注册 source replicates，先各自形成 replicate-level BB/100，再等权平均；
5. 先在 seed 内计算 gap 和 interaction，再跨 seed 计算均值、CI 和检验。

主要 confirmatory checkpoint=`300`，design=`R1-E1-I`，mode=`Frozen`：

```text
Gap(seed,m) = Heldout(seed,m,300,Frozen) - Source(seed,m,300,Frozen)

PrimaryInteraction(seed,m) =
    Gap(seed,m) - Gap(seed,Fact),  m ∈ {Expr, Async}
```

数值越负表示相对 Fact 的 held-out 损失更大。主要报告为跨 eligible seeds 的 PrimaryInteraction 算术均值、seed-cluster bootstrap 95% CI、两侧 raw p 和 Holm-adjusted p。

NoMemory、Sync、AdaptationGain、GeneralizationDrop、CheckpointSlope、mixed ecological、Online、Without 和行为/稳健性均为 secondary/exploratory，不能替换主要检验族。

Table 4 的时间语义在揭盲前固定如下：每个 held-out table 先对预注册 target agents 等权，
`initial_transfer_bb100` 使用该 mode 每桌按 `hand_number, hand_id` 排序后的前 50 hands，
`final_bb100` 使用最后 50 hands；再对 H01/H02/H03 等权、最后跨 seed 汇总。Online 的
`recovery_slope_bb100_per_100_hands` 使用连续、不重叠的 50-hand blocks，先按 table 等权形成
block BB/100，再以 block 起点间隔 `0.5 × 100 hands` 的横轴做含截距 OLS；少于两个完整
blocks 标为 `NA_NOT_ESTIMABLE`。Actual Frozen 与 Without 的 slope 固定为
`NA_NOT_APPLICABLE`。不得在看到结果后改变窗口、block 大小或回归形式。

## 2. Seed-cluster bootstrap 95% CI

- Cluster=seed；在完整 eligible seed-level rows 上有放回抽样。
- 每次抽样抽取 `n_effective` 个 seed clusters；同一 seed 内全部相关重复测量整体保留。
- 重复次数固定为 10,000；PRNG 固定为 PCG64；bootstrap seed 固定为 `2026090199`。
- 区间固定为 percentile CI：bootstrap 分布的 2.5th 与 97.5th percentiles。
- 点估计不取 bootstrap 均值，仍用原始 eligible seed-level effect 的算术均值。
- 每个 estimand 必须记录 `n_planned=12`、`n_effective`、cluster IDs、replicate count、PRNG、seed 和 quantile method。
- 禁止按 CI 好看程度改用 BCa、studentized、正态近似或 hand-level bootstrap。
- `n_effective < 2` 时 CI 标为 `NA_NOT_ESTIMABLE`，不得生成伪区间。
- Figure 2–4 的 ribbon/error bar 必须来自同一 seed-cluster bootstrap 实现，不能使用 hand-level standard error。

## 3. Confirmatory p 值与 Holm

每个主要 interaction 使用 seed-level effects 对零的精确双侧 sign-flip test：

- n≤20 时枚举全部 `2^n` 符号组合；
- 统计量为 seed-level effect 的算术均值；
- 双侧极端性为 `abs(permuted_mean) >= abs(observed_mean)`；
- raw p 使用包含 observed assignment 的精确比例；
- 不按观察方向改成单侧。

Primary Family A 固定为两个假设，顺序标识为 `Expr_vs_Fact`、`Async_vs_Fact`。对两个 raw p 使用 Holm step-down 控制 family-wise two-sided alpha=0.05；报告 raw p、Holm adjusted p、排序和 reject 标志。若任一主要 p 因数据门禁不可估计，则 family 状态为 `UNRESOLVED`，不得把缺失 p 当 1 后宣称另一项通过。

CI 是 effect uncertainty；Holm 只作用于上述两个主要 p。Secondary families 必须与 Family A 分表，不得混入其 Holm 分母。

## 4. Missing、exclusion、retry 与 attempt

### 4.1 Missing

- 不插补、不用 hand/event 补 n、不临时借用其他 seed。
- 一个机制在某 seed/checkpoint/mode 下必须具有预注册的完整 source evaluation 和 H01/H02/H03；缺任一 table 时该 seed-mechanism cell 为 missing。
- 一个 paired contrast 只有 Fact 与目标机制在同 seed 的相关 cells 都完整且通过门禁时才进入。
- 每个单元记录 `n_planned`、`n_effective`、missing seed IDs 和 reason codes。
- n=12 固定；不得因方差、方向、显著性或缺失临时增加 seed。

### 4.2 Exclusion

排除只基于预注册盲态工程门禁：

- `IDENTITY_MISMATCH`
- `CRN_MISMATCH`
- `FALLBACK_NONZERO`
- `REVISION_FALLBACK_NONZERO`
- `REWARD_CONSERVATION_VIOLATION`
- `STACK_CONSERVATION_VIOLATION`
- `ARTIFACT_INCOMPLETE_OR_HASH_MISMATCH`
- `OUTPUT_PATH_COLLISION`
- `INVALID_RECEIPT_OR_DEPENDENCY`
- `EXECUTION_INVALID`
- `ELIGIBLE_INFRA_FAILURE`

收益差、方向相反、行为退化、极端牌、CI 跨零或不显著都不是排除理由。所有 excluded/partial/failed evidence 原样保留并进入 exclusion ledger。

### 4.3 Retry 与 authoritative attempt

- 默认 attempt 为 `attempt_01`。
- 只有上列预注册工程失败且在查看科学效应前记录，才允许新建 `__attempt_NN`。
- retry 继承同 seed、schedule、代码、Prompt、模型、embedding 与冻结 config；任何必要身份变化必须触发新的协议审计，不能静默等价。
- 旧 attempt 不删除、不覆盖；失败证据和 retry 关联必须机器可审计。
- authoritative attempt 是按 attempt number 从小到大，第一个同时满足 complete、identity、CRN、fallback、守恒、工件和 hash 门禁的 attempt。
- 一旦出现 authoritative valid attempt，不得再为改善结果追加 retry。若误产生多个 valid attempts，自动分析 fail closed，进入独立复核，不得择优。
- `failed/partial` attempt 永不与 valid attempt 拼接 hands/events。
- latest 文件时间、最高收益或“最新目录”都不是 authoritative selection 规则。

### 4.4 揭盲前 verifier-only false-positive 同 attempt 恢复

本节是用户在正式效应揭盲前批准的窄范围 expedited protocol amendment，只适用于
P01–P11、seeds `2026090101–2026090111` 的首任务 `isolation_no_memory`。保留的原始预算固定为
`11 × 1,350 = 14,850 hands`。本节不改变主要终点、统计单位、paired interaction、bootstrap、
Holm、missing、exclusion 或任何 Table/Figure 统计口径。

- 恢复 disposition 固定为 `VERIFIER_FALSE_POSITIVE_SAME_ATTEMPT_RECOVERY`。它不是 retry；
  outer attempt 仍为 `attempt_01`，不得创建或选择新的 outer `__attempt_NN`，不得跨 attempt
  拼接 hands/events/metrics。
- 原 `failed` 行、state hash chain、worker manifest、task1 raw artifacts、预恢复 archive 和 file
  manifest 必须永久保留且不得覆盖。只有 recovery certificate、amendment、archive/file-manifest
  hashes、原 terminal-state hash 和 canonicalization equivalence audit 全部通过时，才允许在同一
  state chain 追加 `failed → validating → running → finalizing → complete`。
- Task1 adoption 必须在不读取效应值的前提下逐 worker 验证：恰有 1,350 hands；原始
  hands/events/metrics 存在且逐文件 hash 与预恢复证据一致；scientific code、Prompt、模型、
  embedding、schedule、CRN、health、fallback 和守恒门禁通过；唯一 identity 差异是已审计的
  `resolved_config_sha256` canonicalization 表示差异。任一门禁失败即 fail closed。
- Task1 不得重跑、不得修改原始字节。adoption 通过后从 manifest 的第二个 task 继续。
  P01–P11 后续所有 GPU hands 必须由 clean scientific checkout
  `a1d1eb97efb41d52585057ab7c9594dcd19227ae` 生成。
- `raw_artifact_reused` 按 source execution stage 冻结：`pre_recovery_task1=true`；
  `post_adoption_task2_plus=false`。后续新生成的 task2+ 原始工件不得误标为复用 task1 工件。
- verifier 必须使用独立、clean、在激活前冻结的完整 40-hex `verifier_code_sha`；verifier 只可
  验证、adopt、生成 receipt 和 sealing 工件，不得生成 GPU hands。`experiment_code_sha` 和
  `scientific_execution_code_sha` 均保持为 `a1d1eb97efb41d52585057ab7c9594dcd19227ae`。
- 普通 failed→complete resurrection 仍然拒绝。不得仅因 run 目录内出现自报 certificate 而
  接受；certificate 必须与可信 pre-unblind amendment SHA、verifier SHA、worker、seed、
  `attempt_01`、run path、原 state SHA 和预恢复 archive hashes 绑定并重新校验。
- `authoritative_attempt` 在全部 completion、identity、CRN、health、artifact 和 hash 门禁通过前
  必须为 `PENDING_VERIFIER_RECOVERY`；通过后才可写为 `attempt_01`。
- exclusion/retry ledger 使用 `reason_code=IDENTITY_MISMATCH`、`eligible_retry=false`、空
  `retry_attempt` 和上述 recovery disposition。`excluded_from_confirmatory=false` 只有在全部恢复
  门禁通过后才成立。
- 全过程固定 `formal_effects_read=false`、`effect_metrics_read=false`、
  `task1_rerun_performed=false`。不得读取收益方向、p 值或 CI，不得追加 seed、改变 endpoint、
  改变排除规则或据此启动 conditional/deferred 实验。

受影响的 lineage 与 ledger 必须记录独立 verifier SHA、amendment SHA、recovery certificate SHA、
预恢复 archive/file-manifest SHA、原 terminal-state SHA、原/修正 config SHA、canonicalization
equivalence audit SHA、task1 adoption attestation SHA、scientific execution SHA、source execution
stage、raw artifact reuse 和 task1 rerun 状态。未发生恢复的行使用 `recovery_disposition=NONE`。
其中 `original_terminal_state_sha256` 专指原 terminal state row 的规范化序列化 SHA-256，
不是整个 state 文件的 SHA-256；如需记录整份 state 文件哈希，必须使用独立字段，不能复用本字段。

## 5. Robustness 和报告边界

必须同时报告 raw paired mean；median、10% trimmed、5/95 winsorized、leave-largest-absolute-pot-out、leave-one-seed-out 和行为率是稳健性，不得替换 raw mean。方向冲突必须写为“对极端手牌/聚合选择敏感”。

所有 Table/Figure cell 必须有 data lineage。数值在 lineage、exclusion ledger、analysis manifest 和独立复核通过前只能标 `TBD` 或 `UNVERIFIED`。

固定 n=12 来自用户在结果揭盲前批准的 expedited protocol amendment。由于新 Power Pilot 被跳过，MDE=100 BB/100 只保留为历史规划候选；报告必须披露功效不确定性，不能声称 n=12 已由本轮方差 Pilot 证实。
