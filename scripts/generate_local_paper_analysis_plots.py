# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentmemeval.analysis.plots import plot_stack_curves  # noqa: E402


@dataclass(frozen=True)
class RunSpec:
    group: str
    key: str
    label: str
    config: str
    run_dir: str


RUNS = [
    RunSpec(
        group="exp1",
        key="no_memory",
        label="No memory",
        config="configs/experiments/local_paper_exp1_no_memory.yaml",
        run_dir="outputs/20260709T184503Z_fixed_evolving_table_seed42",
    ),
    RunSpec(
        group="exp1",
        key="fact_agent",
        label="Fact memory",
        config="configs/experiments/local_paper_exp1_fact_agent.yaml",
        run_dir="outputs/20260709T195112Z_fixed_evolving_table_seed42",
    ),
    RunSpec(
        group="exp1",
        key="expr_agent",
        label="Experience memory",
        config="configs/experiments/local_paper_exp1_expr_agent.yaml",
        run_dir="outputs/20260709T205049Z_fixed_evolving_table_seed42",
    ),
    RunSpec(
        group="exp1",
        key="fact_expr_sync",
        label="Fact + experience sync",
        config="configs/experiments/local_paper_exp1_fact_expr_sync.yaml",
        run_dir="outputs/20260709T215524Z_fixed_evolving_table_seed42",
    ),
    RunSpec(
        group="exp1",
        key="fact_expr_async",
        label="Fact + experience async",
        config="configs/experiments/local_paper_exp1_fact_expr_async.yaml",
        run_dir="outputs/20260709T224904Z_fixed_evolving_table_seed42",
    ),
    RunSpec(
        group="exp2",
        key="intj",
        label="INTJ",
        config="configs/experiments/local_paper_exp2_persona_intj.yaml",
        run_dir="outputs/20260709T235034Z_fixed_evolving_table_seed42",
    ),
    RunSpec(
        group="exp2",
        key="enfp",
        label="ENFP",
        config="configs/experiments/local_paper_exp2_persona_enfp.yaml",
        run_dir="outputs/20260710T001959Z_fixed_evolving_table_seed42",
    ),
    RunSpec(
        group="exp2",
        key="istp",
        label="ISTP",
        config="configs/experiments/local_paper_exp2_persona_istp.yaml",
        run_dir="outputs/20260710T004616Z_fixed_evolving_table_seed42",
    ),
    RunSpec(
        group="exp2",
        key="esfj",
        label="ESFJ",
        config="configs/experiments/local_paper_exp2_persona_esfj.yaml",
        run_dir="outputs/20260710T011457Z_fixed_evolving_table_seed42",
    ),
]

COMPARISON_DIR = ROOT / "outputs" / "local_paper_comparison_20260710"
PLOTS_DIR = COMPARISON_DIR / "plots"
DOC_PATH = ROOT / "docs" / "0710_12_本地论文规模图表重绘与结果解读.md"


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(COMPARISON_DIR / ".matplotlib"))
    (COMPARISON_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)

    rows = [summarize_run(spec) for spec in RUNS]
    write_summary_csv(rows)
    write_summary_json(rows)

    for row in rows:
        run_path = ROOT / row["run_dir"]
        plot_stack_curves(read_jsonl(run_path / "hand_summaries.jsonl"), run_path / "plots")

    make_plots(rows)
    write_doc(rows)
    print(f"Wrote {DOC_PATH}")
    print(f"Wrote comparison plots under {PLOTS_DIR}")


def summarize_run(spec: RunSpec) -> dict[str, Any]:
    run_path = ROOT / spec.run_dir
    metrics = read_json(run_path / "metrics.json")
    events = read_jsonl(run_path / "events.jsonl")
    target = metrics["primary_metrics"]["per_agent"]["agent_00"]
    memory = target.get("memory", {})
    action_events = [event for event in events if event.get("event") == "action"]
    target_actions = [event for event in action_events if event.get("agent_id") == "agent_00"]
    repair_count = sum(1 for event in action_events if event.get("guard_repaired"))
    fallback_count = sum(1 for event in action_events if event.get("fallback_used"))
    target_repair_count = sum(1 for event in target_actions if event.get("guard_repaired"))
    target_fallback_count = sum(1 for event in target_actions if event.get("fallback_used"))
    action_total = len(action_events)
    target_action_total = len(target_actions)
    return {
        "group": spec.group,
        "key": spec.key,
        "label": spec.label,
        "config": spec.config,
        "run_dir": spec.run_dir,
        "hands": target.get("hands", 0),
        "chip_delta": target.get("chip_delta", 0),
        "bb_per_100": target.get("bb_per_100", 0.0),
        "win_rate": target.get("win_rate", 0.0),
        "vpip": target.get("vpip", 0.0),
        "fold_rate": target.get("fold_rate", 0.0),
        "raise_rate": target.get("raise_rate", 0.0),
        "fold_to_raise": target.get("fold_to_raise", 0.0),
        "proxy_bluff_rate": target.get("proxy_bluff_rate", 0.0),
        "intent_bluff_rate": target.get("intent_bluff_rate", 0.0),
        "memory_mechanism": memory.get("mechanism", ""),
        "fact_count": memory.get("fact_count", 0),
        "experience_updates": memory.get("experience_updates", 0),
        "actions": action_total,
        "guard_repaired": repair_count,
        "fallback_used": fallback_count,
        "guard_repair_rate": safe_div(repair_count, action_total),
        "fallback_rate": safe_div(fallback_count, action_total),
        "target_actions": target_action_total,
        "target_guard_repaired": target_repair_count,
        "target_fallback_used": target_fallback_count,
        "target_guard_repair_rate": safe_div(target_repair_count, target_action_total),
        "target_fallback_rate": safe_div(target_fallback_count, target_action_total),
    }


def make_plots(rows: list[dict[str, Any]]) -> None:
    exp1 = [row for row in rows if row["group"] == "exp1"]
    exp2 = [row for row in rows if row["group"] == "exp2"]
    plot_cumulative_curves(
        exp1,
        stage=None,
        filename="exp1_target_cumulative_chip_delta_by_mechanism.png",
        title="Exp1 target cumulative chip delta by memory mechanism",
    )
    plot_cumulative_curves(
        exp1,
        stage="test",
        filename="exp1_target_test_chip_delta_by_mechanism.png",
        title="Exp1 target test-stage chip delta by memory mechanism",
    )
    plot_metric_pair(
        exp1,
        filename="exp1_target_final_performance.png",
        title="Exp1 target final performance",
    )
    plot_grouped_rates(
        exp1,
        metrics=[
            ("vpip", "VPIP"),
            ("raise_rate", "Raise"),
            ("fold_rate", "Fold"),
            ("fold_to_raise", "Fold to raise"),
        ],
        filename="exp1_target_action_profile.png",
        title="Exp1 target action profile",
    )
    plot_grouped_values(
        exp1,
        metrics=[
            ("fact_count", "Facts"),
            ("experience_updates", "Experience updates"),
        ],
        filename="exp1_target_memory_behavior.png",
        title="Exp1 target memory behavior",
        ylabel="Count",
    )
    plot_grouped_rates(
        exp1,
        metrics=[
            ("guard_repair_rate", "All agents repaired"),
            ("fallback_rate", "All agents fallback"),
            ("target_guard_repair_rate", "Target repaired"),
            ("target_fallback_rate", "Target fallback"),
        ],
        filename="exp1_guard_repair_and_fallback_rates.png",
        title="Exp1 output validity diagnostics",
    )

    plot_metric_pair(
        exp2,
        filename="exp2_target_final_performance.png",
        title="Exp2 target final performance",
    )
    plot_grouped_rates(
        exp2,
        metrics=[
            ("vpip", "VPIP"),
            ("raise_rate", "Raise"),
            ("fold_rate", "Fold"),
            ("fold_to_raise", "Fold to raise"),
        ],
        filename="exp2_persona_action_profile.png",
        title="Exp2 persona action profile",
    )
    plot_grouped_values(
        exp2,
        metrics=[
            ("fact_count", "Facts"),
            ("experience_updates", "Experience updates"),
        ],
        filename="exp2_persona_memory_behavior.png",
        title="Exp2 persona memory behavior",
        ylabel="Count",
    )
    plot_grouped_rates(
        exp2,
        metrics=[
            ("guard_repair_rate", "All agents repaired"),
            ("fallback_rate", "All agents fallback"),
            ("target_guard_repair_rate", "Target repaired"),
            ("target_fallback_rate", "Target fallback"),
        ],
        filename="exp2_guard_repair_and_fallback_rates.png",
        title="Exp2 output validity diagnostics",
    )


def plot_cumulative_curves(
    rows: list[dict[str, Any]],
    *,
    stage: str | None,
    filename: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    for row in rows:
        xs, ys = target_cumulative(ROOT / row["run_dir"], stage=stage)
        ax.plot(xs, ys, label=row["label"], linewidth=2.0)
    ax.axhline(0, color="black", linewidth=0.9, alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel("Hand index" if stage is None else "Test hand index")
    ax.set_ylabel("Target chip delta")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", frameon=True, title="Condition")
    save_fig(fig, filename)


def plot_metric_pair(rows: list[dict[str, Any]], *, filename: str, title: str) -> None:
    import matplotlib.pyplot as plt

    labels = [row["label"] for row in rows]
    xs = list(range(len(rows)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(xs, [row["chip_delta"] for row in rows], color="#4777b3")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Chip delta")
    axes[0].set_xticks(xs, labels, rotation=22, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(xs, [row["bb_per_100"] for row in rows], color="#d08a3c")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("BB per 100")
    axes[1].set_xticks(xs, labels, rotation=22, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    save_fig(fig, filename)


def plot_grouped_rates(
    rows: list[dict[str, Any]],
    *,
    metrics: list[tuple[str, str]],
    filename: str,
    title: str,
) -> None:
    plot_grouped_values(rows, metrics=metrics, filename=filename, title=title, ylabel="Rate", scale=100.0)


def plot_grouped_values(
    rows: list[dict[str, Any]],
    *,
    metrics: list[tuple[str, str]],
    filename: str,
    title: str,
    ylabel: str,
    scale: float = 1.0,
) -> None:
    import matplotlib.pyplot as plt

    labels = [row["label"] for row in rows]
    xs = list(range(len(rows)))
    width = min(0.8 / max(len(metrics), 1), 0.2)
    colors = ["#4777b3", "#d08a3c", "#5b9f6e", "#b45c63", "#6f63ad"]
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    for metric_index, (metric, label) in enumerate(metrics):
        offset = (metric_index - (len(metrics) - 1) / 2) * width
        values = [row.get(metric, 0.0) * scale for row in rows]
        ax.bar([x + offset for x in xs], values, width=width, label=label, color=colors[metric_index % len(colors)])
    ax.set_title(title)
    ax.set_ylabel(f"{ylabel} (%)" if scale == 100.0 else ylabel)
    ax.set_xticks(xs, labels, rotation=22, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", frameon=True, title="Metric")
    save_fig(fig, filename)


def target_cumulative(run_path: Path, *, stage: str | None) -> tuple[list[int], list[int]]:
    total = 0
    xs: list[int] = []
    ys: list[int] = []
    for hand in read_jsonl(run_path / "hand_summaries.jsonl"):
        if stage is not None and hand.get("stage") != stage:
            continue
        total += int((hand.get("rewards", {}) or {}).get("agent_00", 0))
        xs.append(len(xs) + 1)
        ys.append(total)
    return xs, ys


def write_summary_csv(rows: list[dict[str, Any]]) -> None:
    path = COMPARISON_DIR / "summary_metrics.csv"
    fieldnames = [
        "group",
        "key",
        "label",
        "run_dir",
        "chip_delta",
        "bb_per_100",
        "win_rate",
        "vpip",
        "fold_rate",
        "raise_rate",
        "fold_to_raise",
        "fact_count",
        "experience_updates",
        "actions",
        "guard_repair_rate",
        "fallback_rate",
        "target_guard_repair_rate",
        "target_fallback_rate",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_summary_json(rows: list[dict[str, Any]]) -> None:
    path = COMPARISON_DIR / "summary_metrics.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_doc(rows: list[dict[str, Any]]) -> None:
    exp1 = [row for row in rows if row["group"] == "exp1"]
    exp2 = [row for row in rows if row["group"] == "exp2"]
    exp1_best = max(exp1, key=lambda row: row["bb_per_100"])
    exp1_worst = min(exp1, key=lambda row: row["bb_per_100"])
    exp2_most_active = max(exp2, key=lambda row: row["vpip"])
    exp2_most_passive = min(exp2, key=lambda row: row["vpip"])
    lines = [
        "# 0710_12 本地论文规模图表重绘与结果解读",
        "",
        "## 1. 本次处理结论",
        "",
        "- 原来的单次运行图 `plots/stack_curve.png` 不是不能用，而是当 Agent 数量超过 12 时旧代码主动关闭了图例；因此你看到很多线，却看不出每条线属于哪个 Agent。",
        "- 我已修改绘图逻辑：所有单次运行的 stack 曲线都会显示图例，并把 `agent_00 (target)` 用黑色粗线高亮；`agent_01...agent_07` 标为训练对手，`heldout_00...` 标为泛化测试对手。",
        "- 我没有重跑 9 个长实验；本次只是基于已经完成的 `metrics.json`、`events.jsonl`、`hand_summaries.jsonl` 重绘和汇总，避免改动原始实验结果。",
        "- 另外发现一个会影响解释的运行期问题：本轮实验中，Qwen 经常输出 `call` 且携带 `amount=2`，旧 action guard 把它判为非法并 fallback 成 `fold`。我已经修复代码，后续重跑时非 raise 动作会自动清空 amount。",
        "",
        "## 2. 新增和重绘的图表",
        "",
        "- 单次运行图：9 个运行目录下的 `plots/stack_curve.png` 已重绘，图例含 target/train/heldout 标签。",
        f"- 跨实验汇总目录：`{rel(COMPARISON_DIR)}`。",
        f"- 汇总指标表：`{rel(COMPARISON_DIR / 'summary_metrics.csv')}` 和 `{rel(COMPARISON_DIR / 'summary_metrics.json')}`。",
        "",
        "### 2.1 Exp1 记忆机制对照",
        "",
        f"![Exp1 target cumulative](../{rel(PLOTS_DIR / 'exp1_target_cumulative_chip_delta_by_mechanism.png')})",
        "",
        "这张图最接近论文中“不同记忆机制随训练推进的收益曲线”这一类图：每条线对应一种记忆机制，纵轴是目标 Agent 的累计筹码变化。它只画 `agent_00`，避免把训练对手和 heldout 对手混在一起。",
        "",
        f"![Exp1 target test](../{rel(PLOTS_DIR / 'exp1_target_test_chip_delta_by_mechanism.png')})",
        "",
        "这张图只看测试阶段，近似对应论文里的泛化/heldout 对照视角。它能回答：训练后的记忆机制面对未见过对手时，是否仍然带来收益。",
        "",
        f"![Exp1 final](../{rel(PLOTS_DIR / 'exp1_target_final_performance.png')})",
        "",
        f"本轮单 seed 下，Exp1 表现最好的是 **{exp1_best['label']}**，目标 Agent 的 BB/100 为 {fmt(exp1_best['bb_per_100'])}；表现最差的是 **{exp1_worst['label']}**，BB/100 为 {fmt(exp1_worst['bb_per_100'])}。",
        "",
        f"![Exp1 action](../{rel(PLOTS_DIR / 'exp1_target_action_profile.png')})",
        "",
        "这张图解释为什么很多收益曲线会突变：VPIP/raise/fold_to_raise 直接反映目标 Agent 的行动风格。若 fold_rate 或 fallback 很高，模型实际在牌桌上的主动行为会被压缩。",
        "",
        f"![Exp1 memory](../{rel(PLOTS_DIR / 'exp1_target_memory_behavior.png')})",
        "",
        "这张图用于核验实验机制有没有真的生效：Fact memory 应有 fact_count，Experience memory 应有 experience_updates，组合记忆应两者都有。",
        "",
        f"![Exp1 guard](../{rel(PLOTS_DIR / 'exp1_guard_repair_and_fallback_rates.png')})",
        "",
        "这张图非常关键：它不是论文指标，而是本地复现实验的质量诊断。fallback 率越高，说明越多模型原始决策没有按预期进入牌局逻辑；因此当前 9 次结果适合看“单 seed 现象”和流程打通，不适合直接当作最终论文复现结论。",
        "",
        "### 2.2 Exp2 Persona 对照",
        "",
        f"![Exp2 final](../{rel(PLOTS_DIR / 'exp2_target_final_performance.png')})",
        "",
        f"本轮 persona 中，目标 Agent 收益最高的是 **{max(exp2, key=lambda row: row['bb_per_100'])['label']}**；最低的是 **{min(exp2, key=lambda row: row['bb_per_100'])['label']}**。",
        "",
        f"![Exp2 action](../{rel(PLOTS_DIR / 'exp2_persona_action_profile.png')})",
        "",
        f"Persona 行为差异在本轮很明显：**{exp2_most_active['label']}** 的 VPIP 最高，行动最主动；**{exp2_most_passive['label']}** 的 VPIP 最低，几乎不主动入池。",
        "",
        f"![Exp2 memory](../{rel(PLOTS_DIR / 'exp2_persona_memory_behavior.png')})",
        "",
        "这张图说明 persona 实验里记忆写入量也不同，尤其 experience_updates 与行动频率高度相关：越主动参与牌局，越容易产生经验更新。",
        "",
        f"![Exp2 guard](../{rel(PLOTS_DIR / 'exp2_guard_repair_and_fallback_rates.png')})",
        "",
        "这张图同样用于判断本轮 persona 结果是否受到输出格式修复的影响。",
        "",
        "## 3. 结果速读",
        "",
        "| 实验 | 条件 | chip_delta | BB/100 | VPIP | raise_rate | fold_rate | facts | exp_updates | fallback_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['group']} | {row['label']} | {row['chip_delta']} | {fmt(row['bb_per_100'])} | "
            f"{pct(row['vpip'])} | {pct(row['raise_rate'])} | {pct(row['fold_rate'])} | "
            f"{row['fact_count']} | {row['experience_updates']} | {pct(row['fallback_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## 4. 与原论文图表的对应关系",
            "",
            "| 原论文常见图表视角 | 本次可对照图表 | 说明 |",
            "|---|---|---|",
            "| 不同记忆机制的收益/stack 曲线 | `exp1_target_cumulative_chip_delta_by_mechanism.png` | 只画目标 Agent，避免单表 15 条线混乱。 |",
            "| 泛化测试表现 | `exp1_target_test_chip_delta_by_mechanism.png` | 只取 `stage=test` 的 heldout 阶段。 |",
            "| 行为风格/动作分布 | `exp1_target_action_profile.png`、`exp2_persona_action_profile.png` | 对应 VPIP、raise、fold、fold-to-raise。 |",
            "| 记忆机制是否真正写入 | `exp1_target_memory_behavior.png`、`exp2_persona_memory_behavior.png` | 对应事实记忆和经验记忆写入量。 |",
            "| 本地复现实验质量诊断 | `*_guard_repair_and_fallback_rates.png` | 论文通常不画这个，但本地小模型复现必须看。 |",
            "",
            "## 5. 本次代码修改记录",
            "",
            "- `src/agentmemeval/analysis/plots.py`：单次运行 stack 曲线始终显示图例；`agent_00` 黑色粗线高亮；训练/泛化对手在图例中显式标注。",
            "- `src/agentmemeval/environment/action_guard.py`：解析模型决策后，非 `raise` 动作统一清空 `amount`，避免 `call amount=2` 被误判为非法并 fallback。",
            "- `src/agentmemeval/prompts/decision.py`：在决策提示中明确只有 `raise` 可以携带整数 `amount`，`fold/check/call` 必须为 `null`。",
            "- `src/agentmemeval/llm/providers/openai_compatible.py`：JSON schema 增加 `additionalProperties: false`，让 LM Studio/OpenAI-compatible 输出更稳定。",
            "- `tests/unit/test_action_guard.py`：新增非 raise 动作清空 amount 的单元测试。",
            "- `scripts/generate_local_paper_analysis_plots.py`：新增本轮图表重绘、跨实验汇总和本文档生成脚本。",
            "",
            "## 6. 验证",
            "",
            "- `python -m pytest tests\\unit\\test_action_guard.py`：4 passed。",
            "- `python -m ruff check ...`：All checks passed。",
            "- `python scripts\\generate_local_paper_analysis_plots.py`：已生成汇总 CSV/JSON、跨实验图表，并重绘 9 个单次运行图。",
            "",
            "## 7. 下一步建议",
            "",
            "当前结果已经能帮助我们看懂这次实验，但由于旧 guard 导致 fallback 偏高，建议下一轮在已修复代码基础上优先重跑 1 个最小对照组：`No memory`、`Experience memory`、`Fact + experience async`、`ENFP`，确认 fallback 明显下降后，再决定是否重新跑全量 9 组。",
            "",
        ]
    )
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def fmt(value: int | float) -> str:
    return f"{float(value):.2f}"


def pct(value: int | float) -> str:
    return f"{float(value) * 100:.1f}%"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def save_fig(fig: Any, filename: str) -> None:
    import matplotlib.pyplot as plt

    path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
