"""
模块说明：本模块实现命令行界面。
核心职责：提供 doctor、run 和 report 三个用户可直接执行的入口。
输入与输出：输入命令行参数，输出终端摘要和退出码。
依赖边界：调用 Provider 路由、实验 runner 和报告重建函数。
不负责：不实现具体实验逻辑，不在默认命令中调用真实在线 API。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentmemeval.config.loader import load_raw_config
from agentmemeval.core.errors import AgentMemEvalError
from agentmemeval.evaluation.formal_freeze import generate_formal_freeze_bundle
from agentmemeval.evaluation.pilot import (
    build_pilot_freeze_proposal_from_paths,
    build_pilot_power_plan,
)
from agentmemeval.evaluation.reporting import rebuild_report
from agentmemeval.experiments.campaign import aggregate_campaign, run_campaign
from agentmemeval.experiments.runner import run_config
from agentmemeval.llm.router import provider_health


def main(argv: list[str] | None = None) -> int:
    """
    功能：CLI 主入口。
    参数：
        argv：可选命令行参数列表。
    返回：进程退出码。
    副作用：打印终端输出，可能写实验工件。
    异常：内部捕获可预期领域异常并返回 2。
    设计说明：保持错误中文可读，方便用户定位配置或 Provider 问题。
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "run":
            return _run(args)
        if args.command == "campaign":
            return _campaign(args)
        if args.command == "campaign-aggregate":
            return _campaign_aggregate(args)
        if args.command == "pilot-plan":
            return _pilot_plan(args)
        if args.command == "pilot-freeze":
            return _pilot_freeze(args)
        if args.command == "formal-freeze":
            return _formal_freeze(args)
        if args.command == "report":
            return _report(args)
        parser.print_help()
        return 1
    except AgentMemEvalError as exc:
        print(f"错误：{exc}")
        return 2


def build_parser() -> argparse.ArgumentParser:
    """
    功能：构建命令行解析器。
    参数：无。
    返回：ArgumentParser。
    副作用：无。
    异常：无。
    设计说明：命令结构与 README 中的可复制命令保持一致。
    """

    parser = argparse.ArgumentParser(prog="agentmemeval", description="AgentMemEval 重构版 CLI")
    sub = parser.add_subparsers(dest="command")
    doctor = sub.add_parser("doctor", help="检查 Provider 和离线环境")
    doctor.add_argument("--provider", help="Provider 名称；不提供时优先使用配置文件中的 provider")
    doctor.add_argument("--config", help="可选配置文件；提供后优先读取 provider 段")
    run = sub.add_parser("run", help="运行实验配置")
    run.add_argument("--config", required=True, help="YAML 配置路径")
    campaign = sub.add_parser("campaign", help="运行或续跑多 seed campaign")
    campaign.add_argument("--config", required=True, help="campaign YAML 配置路径")
    campaign.add_argument(
        "--resume",
        action="store_true",
        help="仅续跑缺失/失败矩阵；已验证完成的 run 不会重跑",
    )
    campaign_aggregate = sub.add_parser(
        "campaign-aggregate", help="不重跑实验，从 campaign 原始工件重建聚合"
    )
    campaign_aggregate.add_argument("--input", required=True, help="campaign 目录")
    pilot_plan = sub.add_parser(
        "pilot-plan", help="从完整 Campaign P/E pilot aggregate 生成审计功效计划"
    )
    pilot_plan.add_argument("--campaign-p", required=True, help="Campaign P aggregate JSON")
    pilot_plan.add_argument("--campaign-e", required=True, help="Campaign E aggregate JSON")
    pilot_plan.add_argument(
        "--runtime-equivalence-audit",
        help="仅当 P/E commit 不同时使用的 Pilot-only 执行等价审计 JSON",
    )
    pilot_plan.add_argument("--output", required=True, help="新功效计划 JSON；拒绝覆盖")
    pilot_freeze = sub.add_parser(
        "pilot-freeze", help="从完整 Pilot 工件生成行为、功效、执行和检索冻结提案"
    )
    pilot_freeze.add_argument("--campaign-p", required=True, help="Campaign P aggregate JSON")
    pilot_freeze.add_argument("--campaign-e", required=True, help="Campaign E aggregate JSON")
    pilot_freeze.add_argument("--campaign-p-dir", required=True, help="Campaign P 目录")
    pilot_freeze.add_argument("--campaign-e-dir", required=True, help="Campaign E 目录")
    pilot_freeze.add_argument(
        "--retrieval-review-audit",
        required=True,
        help="独立人工相关性标签生成的检索阈值审计 JSON",
    )
    pilot_freeze.add_argument(
        "--runtime-equivalence-audit",
        help="仅当 P/E commit 不同时使用的 Pilot-only 执行等价审计 JSON",
    )
    pilot_freeze.add_argument("--output", required=True, help="新冻结提案 JSON；拒绝覆盖")
    formal_freeze = sub.add_parser(
        "formal-freeze", help="从 ready Pilot 提案生成不可变 P/E 正式配置包"
    )
    formal_freeze.add_argument("--proposal", required=True, help="ready Pilot freeze JSON")
    formal_freeze.add_argument("--runtime-lock", required=True, help="双服务运行时锁 JSON")
    formal_freeze.add_argument("--campaign-p-template", required=True)
    formal_freeze.add_argument("--campaign-e-template", required=True)
    formal_freeze.add_argument("--formal-p-template", required=True)
    formal_freeze.add_argument("--formal-e-template", required=True)
    formal_freeze.add_argument(
        "--strict-p-template",
        required=True,
        help="strict paper-protocol/model-substituted sensitivity experiment template",
    )
    formal_freeze.add_argument(
        "--strict-p-campaign-template",
        required=True,
        help="strict sensitivity Campaign P template",
    )
    formal_freeze.add_argument("--output-dir", required=True, help="全新输出目录；拒绝覆盖")
    formal_freeze.add_argument("--freeze-id", required=True, help="不可变冻结标识")
    formal_freeze.add_argument("--seed-start", type=int, default=2026071801)
    formal_freeze.add_argument(
        "--preflight-seed",
        type=int,
        required=True,
        help="与 calibration Pilot 和 formal seeds 均不重叠的冻结预检 seed",
    )
    report = sub.add_parser("report", help="从 run 目录重建报告")
    report.add_argument("--input", required=True, help="outputs/<run_id> 目录")
    report.add_argument("--big-blind", type=int, default=2, help="重算 BB/100 使用的大盲")
    return parser


def _doctor(args: argparse.Namespace) -> int:
    """
    功能：执行 Provider 健康检查。
    参数：
        args：命令参数。
    返回：退出码。
    副作用：打印 JSON。
    异常：Provider 配置错误向上抛出。
    设计说明：doctor 不做真实生成，避免无密钥环境误触在线 API。
    """

    if args.config:
        config = load_raw_config(args.config)
        provider_config = dict(config.get("provider", config))
    else:
        provider_name = args.provider or "mock"
        provider_config = {"provider": provider_name}
        if provider_name == "mock":
            provider_config["model"] = "mock-deterministic-v1"
    if args.provider:
        provider_config["provider"] = args.provider
    provider_config["provider"] = provider_config.get("provider", "mock")
    print(json.dumps(provider_health(provider_config), ensure_ascii=False, indent=2))
    return 0


def _run(args: argparse.Namespace) -> int:
    """
    功能：运行实验。
    参数：
        args：命令参数。
    返回：退出码。
    副作用：写工件并打印结果路径。
    异常：实验错误向上抛出。
    设计说明：结果摘要来自 ExperimentResult，用户无需翻找目录。
    """

    result = run_config(args.config)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _campaign(args: argparse.Namespace) -> int:
    """运行 append-only campaign，并打印本次矩阵摘要。"""

    result = run_campaign(args.config, resume=bool(args.resume))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _campaign_aggregate(args: argparse.Namespace) -> int:
    """从 append-only campaign 工件生成新的版本化聚合。"""

    result = aggregate_campaign(args.input)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _pilot_plan(args: argparse.Namespace) -> int:
    """Generate one immutable pilot power plan from versioned P/E aggregates."""

    campaign_p = json.loads(Path(args.campaign_p).read_text(encoding="utf-8"))
    campaign_e = json.loads(Path(args.campaign_e).read_text(encoding="utf-8"))
    runtime_equivalence = (
        json.loads(
            Path(args.runtime_equivalence_audit).read_text(encoding="utf-8")
        )
        if args.runtime_equivalence_audit
        else None
    )
    plan = build_pilot_power_plan(
        campaign_p,
        campaign_e,
        runtime_equivalence,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(output), **plan}, ensure_ascii=False, indent=2))
    return (
        0
        if plan["status"]
        == "power_plan_ready_requires_behavior_execution_and_runtime_freeze"
        else 2
    )


def _pilot_freeze(args: argparse.Namespace) -> int:
    """Generate an immutable fail-closed freeze proposal from completed Pilot evidence."""

    proposal = build_pilot_freeze_proposal_from_paths(
        args.campaign_p,
        args.campaign_e,
        args.campaign_p_dir,
        args.campaign_e_dir,
        args.retrieval_review_audit,
        args.runtime_equivalence_audit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(proposal, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(output), **proposal}, ensure_ascii=False, indent=2))
    return (
        0
        if proposal["status"] == "ready_to_generate_immutable_formal_configs"
        else 2
    )


def _formal_freeze(args: argparse.Namespace) -> int:
    """Generate self-contained formal configs only from a ready frozen proposal."""

    result = generate_formal_freeze_bundle(
        proposal_path=args.proposal,
        runtime_lock_path=args.runtime_lock,
        campaign_p_template_path=args.campaign_p_template,
        campaign_e_template_path=args.campaign_e_template,
        formal_p_template_path=args.formal_p_template,
        formal_e_template_path=args.formal_e_template,
        strict_p_template_path=args.strict_p_template,
        strict_p_campaign_template_path=args.strict_p_campaign_template,
        output_dir=args.output_dir,
        freeze_id=args.freeze_id,
        seed_start=args.seed_start,
        preflight_seed=args.preflight_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _report(args: argparse.Namespace) -> int:
    """
    功能：从原始工件重建报告。
    参数：
        args：命令参数。
    返回：退出码。
    副作用：重写 metrics、aggregate、plots 和 report.md。
    异常：文件错误由标准库向上抛出。
    设计说明：满足不用重跑实验即可重新聚合和绘图的要求。
    """

    output = rebuild_report(Path(args.input), big_blind=args.big_blind)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0
