"""Rebuild auditable campaign tables, plot, and report from one aggregate JSON."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TABLE_FIELDS = (
    "design",
    "contrast",
    "endpoint",
    "baseline",
    "n_seed_pairs",
    "mean_effect",
    "median_effect",
    "std_effect",
    "ci95_low",
    "ci95_high",
    "bootstrap_ci95_low",
    "bootstrap_ci95_high",
    "raw_p_value",
    "holm_adjusted_p_value",
)


def build_campaign_analysis(
    aggregate_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Create a non-overwriting analysis bundle whose rows retain seed-level effects."""

    source = Path(aggregate_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    aggregate = _read_json(source)
    rows, paired = _extract_rows(aggregate)
    if not rows:
        raise ValueError("campaign aggregate contains no paired contrast rows")
    table_path = output / "main_table.csv"
    paired_path = output / "paired_effects.csv"
    plot_data_path = output / "primary_effects_plot_data.csv"
    plot_path = output / "primary_effects_plot.png"
    report_path = output / "campaign_analysis_report.md"
    _write_csv(table_path, TABLE_FIELDS, rows)
    _write_csv(
        paired_path,
        ("design", "contrast", "endpoint", "seed", "effect"),
        paired,
    )
    _write_csv(
        plot_data_path,
        (
            "contrast",
            "n_seed_pairs",
            "mean_effect",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
        ),
        rows,
    )
    _plot_primary_effects(rows, plot_path, plot_data_path)
    report_path.write_text(
        _report_text(aggregate, source, rows, paired, plot_data_path), encoding="utf-8"
    )
    outputs = [table_path, paired_path, plot_data_path, plot_path, report_path]
    manifest = {
        "schema_version": "task4_campaign_analysis_bundle_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_aggregate": str(source),
        "source_aggregate_sha256": _sha256(source),
        "campaign_status": aggregate.get("status"),
        "analysis_is_descriptive_only": aggregate.get("status") == "descriptive_only",
        "table_row_count": len(rows),
        "paired_effect_row_count": len(paired),
        "outputs": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in outputs
        },
    }
    manifest_path = output / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"output_dir": str(output), "manifest": str(manifest_path), **manifest}


def _extract_rows(
    aggregate: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    design = str(aggregate.get("design", ""))
    rows: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    if design == "mixed_table":
        metrics = dict(aggregate.get("aggregate_metrics", {}))
        family = (
            metrics.get("main_table")
            if aggregate.get("status") != "descriptive_only"
            else metrics.get("paired_estimand_descriptive")
        )
        family = family if isinstance(family, dict) else {}
        endpoint = str(family.get("endpoint", "final_test_bb_per_100"))
        baseline = str(family.get("baseline_mechanism", "fact"))
        seeds = [int(value) for value in family.get("matched_seeds", [])]
        effects = dict(family.get("effects_by_mechanism", {}))
        summaries = dict(family.get("metrics", {}))
        for mechanism, values in sorted(effects.items()):
            contrast = f"{mechanism}_vs_{baseline}"
            summary = dict(summaries.get(mechanism, {}))
            rows.append(_table_row(design, contrast, endpoint, baseline, summary))
            paired.extend(
                {
                    "design": design,
                    "contrast": contrast,
                    "endpoint": endpoint,
                    "seed": seed,
                    "effect": float(effect),
                }
                for seed, effect in zip(seeds, values, strict=True)
            )
        return rows, paired
    if design == "target_vs_seven_no_memory":
        endpoint = str(aggregate.get("primary_endpoint", "final_test_bb_per_100"))
        baseline = str(aggregate.get("baseline_condition_id", "no_memory_target"))
        for condition, comparison in sorted(
            dict(aggregate.get("paired_comparisons", {})).items()
        ):
            primary = dict(dict(comparison.get("metrics", {})).get(endpoint, {}))
            summary = dict(primary.get("summary", {}))
            summary["raw_p_value"] = comparison.get("primary_raw_p_value")
            summary["adjusted_p_value"] = comparison.get(
                "primary_holm_adjusted_p_value"
            )
            rows.append(_table_row(design, condition, endpoint, baseline, summary))
            paired.extend(
                {
                    "design": design,
                    "contrast": condition,
                    "endpoint": endpoint,
                    "seed": int(seed),
                    "effect": float(effect),
                }
                for seed, effect in zip(
                    primary.get("matched_seeds", []),
                    primary.get("effects", []),
                    strict=True,
                )
            )
    return rows, paired


def _table_row(
    design: str,
    contrast: str,
    endpoint: str,
    baseline: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "design": design,
        "contrast": contrast,
        "endpoint": endpoint,
        "baseline": baseline,
        "n_seed_pairs": int(float(summary.get("n", 0))),
        "mean_effect": summary.get("mean"),
        "median_effect": summary.get("median"),
        "std_effect": summary.get("std"),
        "ci95_low": summary.get("ci95_low"),
        "ci95_high": summary.get("ci95_high"),
        "bootstrap_ci95_low": summary.get("bootstrap_ci95_low"),
        "bootstrap_ci95_high": summary.get("bootstrap_ci95_high"),
        "raw_p_value": summary.get("raw_p_value"),
        "holm_adjusted_p_value": summary.get("adjusted_p_value"),
    }


def _plot_primary_effects(
    rows: list[dict[str, Any]], path: Path, data_path: Path
) -> None:
    import matplotlib.pyplot as plt

    means = [float(row["mean_effect"]) for row in rows]
    low = [float(row["bootstrap_ci95_low"]) for row in rows]
    high = [float(row["bootstrap_ci95_high"]) for row in rows]
    labels = [f"{row['contrast']}\n(n={row['n_seed_pairs']})" for row in rows]
    positions = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(max(8.0, len(rows) * 2.2), 5.5))
    ax.errorbar(
        positions,
        means,
        yerr=[
            [mean - bound for mean, bound in zip(means, low, strict=True)],
            [bound - mean for mean, bound in zip(means, high, strict=True)],
        ],
        fmt="o",
        capsize=5,
        label="Mean effect with bootstrap 95% CI",
    )
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Paired effect (BB/100)")
    ax.set_xlabel("Preregistered contrast and independent seed-pair count")
    ax.set_title("AgentMemEval primary paired effects")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)


def _report_text(
    aggregate: dict[str, Any],
    source: Path,
    rows: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    plot_data_path: Path,
) -> str:
    lines = [
        "# Campaign 统计分析报告",
        "",
        f"- 设计：`{aggregate.get('design')}`",
        f"- 聚合状态：`{aggregate.get('status')}`",
        f"- 完成矩阵：{aggregate.get('completed_run_count')}/"
        f"{aggregate.get('expected_run_count')}",
        f"- 输入：`{source}`",
        f"- 输入 SHA-256：`{_sha256(source)}`",
        "- 独立统计单位：seed 下的一次完整 table/run 或 target condition run",
        f"- 主表行数：{len(rows)}；seed 配对效应行数：{len(paired)}",
        f"- 图表数据：`{plot_data_path}`",
        "",
        "## 解释边界",
        "",
    ]
    if aggregate.get("status") == "descriptive_only":
        lines.append("该 Campaign 是独立 Pilot，只用于描述、阈值与功效规划，不进入 formal 推断。")
    else:
        lines.append("推断资格以 aggregate status、冻结统计计划和运行同质性审计为准。")
    lines.extend(
        [
            "手牌、checkpoint 或同桌 Agent 未被当成独立样本；p 值仅在聚合器准入时报告。",
            "",
            "## 主终点配对结果",
            "",
            "| 对比 | n | 均值 BB/100 | bootstrap 95% CI | Holm p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        adjusted = row["holm_adjusted_p_value"]
        p_text = "NA" if adjusted is None else f"{float(adjusted):.6g}"
        lines.append(
            f"| {row['contrast']} | {row['n_seed_pairs']} | "
            f"{float(row['mean_effect']):.6f} | "
            f"[{float(row['bootstrap_ci95_low']):.6f}, "
            f"{float(row['bootstrap_ci95_high']):.6f}] | {p_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
