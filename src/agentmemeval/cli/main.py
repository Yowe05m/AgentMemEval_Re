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
from agentmemeval.evaluation.task8b_analysis import (
    build_task8b_analysis_input,
    build_task8b_preunlock_manifest,
    run_task8b_analysis,
)
from agentmemeval.experiments.campaign import aggregate_campaign, run_campaign
from agentmemeval.experiments.formal_runner import (
    generate_worker_manifests,
    run_worker_manifest,
    summarize_worker_states,
    verify_checkpoint_receipt,
)
from agentmemeval.experiments.runner import run_config
from agentmemeval.experiments.task8b_bundle import build_task8b_executable_bundle
from agentmemeval.experiments.task8b_transport import (
    archive_completed_worker,
    transfer_checkpoint_bundle,
)
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
        if args.command == "formal-generate-manifests":
            return _formal_generate_manifests(args)
        if args.command == "formal-worker":
            return _formal_worker(args)
        if args.command == "formal-verify-receipt":
            return _formal_verify_receipt(args)
        if args.command == "formal-status":
            return _formal_status(args)
        if args.command == "task8b-build-bundle":
            return _task8b_build_bundle(args)
        if args.command == "task8b-transfer-checkpoint":
            return _task8b_transfer_checkpoint(args)
        if args.command == "task8b-archive-worker":
            return _task8b_archive_worker(args)
        if args.command == "task8b-analyze":
            return _task8b_analyze(args)
        if args.command == "task8b-freeze-phase-f":
            return _task8b_freeze_phase_f(args)
        if args.command == "task8b-build-analysis-input":
            return _task8b_build_analysis_input(args)
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
    formal_generate = sub.add_parser(
        "formal-generate-manifests", help="从 TASK8 matrix 与 seed list 生成中央 worker manifests"
    )
    formal_generate.add_argument("--matrix", required=True, help="冻结或候选 experiment matrix")
    formal_generate.add_argument("--seeds", required=True, help="逗号分隔、顺序冻结的整数 seed")
    formal_generate.add_argument("--identity", required=True, help="common identity JSON")
    formal_generate.add_argument("--output-dir", required=True, help="全新 manifest 输出目录")
    formal_worker = sub.add_parser(
        "formal-worker", help="验证并运行单个 worker manifest；candidate 默认拒绝"
    )
    formal_worker.add_argument("--manifest", required=True)
    formal_worker.add_argument("--receipt-root", required=True)
    formal_worker.add_argument("--resume-existing", action="store_true")
    formal_receipt = sub.add_parser(
        "formal-verify-receipt", help="只读验证 checkpoint receipt、身份与逐文件哈希"
    )
    formal_receipt.add_argument("--receipt", required=True)
    formal_receipt.add_argument("--checkpoint-root", required=True)
    formal_receipt.add_argument("--identity", help="可选 expected common identity JSON")
    formal_status = sub.add_parser("formal-status", help="只读汇总 worker append-only state")
    formal_status.add_argument("--root", required=True)
    task8b_bundle = sub.add_parser(
        "task8b-build-bundle", help="生成 TASK8B 可执行 canary 或 12x2 expedited bundle"
    )
    task8b_bundle.add_argument("--matrix", required=True)
    task8b_bundle.add_argument("--base-config", required=True)
    task8b_bundle.add_argument("--fleet-identity", required=True)
    task8b_bundle.add_argument("--output-dir", required=True)
    task8b_bundle.add_argument("--runtime-bundle-root", required=True)
    task8b_bundle.add_argument("--canary-seed", type=int)
    task8b_transfer = sub.add_parser(
        "task8b-transfer-checkpoint", help="逐文件验证并以 receipt-last 传输 checkpoint"
    )
    task8b_transfer.add_argument("--source-receipt", required=True)
    task8b_transfer.add_argument("--source-root", required=True)
    task8b_transfer.add_argument("--destination-root", required=True)
    task8b_transfer.add_argument("--destination-receipt", required=True)
    task8b_transfer.add_argument("--expected-identity", required=True)
    task8b_transfer.add_argument("--producer-worker-id", required=True)
    task8b_transfer.add_argument("--seed", required=True, type=int)
    task8b_transfer.add_argument("--checkpoint", required=True, type=int)
    task8b_archive = sub.add_parser(
        "task8b-archive-worker", help="门禁并封存一个已完成 TASK8B worker"
    )
    task8b_archive.add_argument("--run-dir", required=True)
    task8b_archive.add_argument("--output-dir", required=True)
    task8b_analysis = sub.add_parser("task8b-analyze", help="运行冻结的 TASK8B Phase F 确定性分析")
    task8b_analysis.add_argument("--input-manifest", required=True)
    task8b_analysis.add_argument("--exclusion-ledger", required=True)
    task8b_analysis.add_argument("--output-dir", required=True)
    task8b_freeze = sub.add_parser(
        "task8b-freeze-phase-f", help="在正式结果揭盲前冻结 Phase F 文件、代码和依赖锁"
    )
    task8b_freeze.add_argument("--phase-f-dir", required=True)
    task8b_freeze.add_argument("--dependency-lock", required=True)
    task8b_freeze.add_argument("--output", required=True)
    task8b_freeze.add_argument("--repository-root")
    task8b_freeze.add_argument("--frozen-at-utc")
    task8b_analysis_input = sub.add_parser(
        "task8b-build-analysis-input", help="从冻结 manifests 与本地回收 attempts 构建 Phase F 输入"
    )
    task8b_analysis_input.add_argument("--worker-manifest-dir", required=True)
    task8b_analysis_input.add_argument("--snapshot-root", required=True)
    task8b_analysis_input.add_argument("--pre-unlock-manifest", required=True)
    task8b_analysis_input.add_argument("--output", required=True)
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
        json.loads(Path(args.runtime_equivalence_audit).read_text(encoding="utf-8"))
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
        if plan["status"] == "power_plan_ready_requires_behavior_execution_and_runtime_freeze"
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
    return 0 if proposal["status"] == "ready_to_generate_immutable_formal_configs" else 2


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


def _formal_generate_manifests(args: argparse.Namespace) -> int:
    identity = json.loads(Path(args.identity).read_text(encoding="utf-8"))
    seeds = [int(value.strip()) for value in str(args.seeds).split(",") if value.strip()]
    result = generate_worker_manifests(
        matrix_path=args.matrix,
        seeds=seeds,
        common_identity=identity,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _formal_worker(args: argparse.Namespace) -> int:
    result = run_worker_manifest(
        args.manifest,
        receipt_root=args.receipt_root,
        resume_existing=bool(args.resume_existing),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _formal_verify_receipt(args: argparse.Namespace) -> int:
    identity = (
        json.loads(Path(args.identity).read_text(encoding="utf-8")) if args.identity else None
    )
    result = verify_checkpoint_receipt(
        args.receipt,
        args.checkpoint_root,
        expected_identity=identity,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _formal_status(args: argparse.Namespace) -> int:
    print(json.dumps(summarize_worker_states(args.root), ensure_ascii=False, indent=2))
    return 0


def _task8b_build_bundle(args: argparse.Namespace) -> int:
    result = build_task8b_executable_bundle(
        matrix_path=args.matrix,
        base_config_path=args.base_config,
        fleet_identity_path=args.fleet_identity,
        output_dir=args.output_dir,
        runtime_bundle_root=args.runtime_bundle_root,
        canary_seed=args.canary_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _task8b_transfer_checkpoint(args: argparse.Namespace) -> int:
    identity = json.loads(Path(args.expected_identity).read_text(encoding="utf-8"))
    result = transfer_checkpoint_bundle(
        source_receipt=args.source_receipt,
        source_checkpoint_root=args.source_root,
        destination_checkpoint_root=args.destination_root,
        destination_receipt=args.destination_receipt,
        expected_identity=identity,
        expected_producer_worker_id=args.producer_worker_id,
        expected_seed_bundle=args.seed,
        expected_checkpoint_hand=args.checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _task8b_archive_worker(args: argparse.Namespace) -> int:
    result = archive_completed_worker(args.run_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _task8b_analyze(args: argparse.Namespace) -> int:
    result = run_task8b_analysis(
        args.input_manifest,
        args.exclusion_ledger,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _task8b_freeze_phase_f(args: argparse.Namespace) -> int:
    result = build_task8b_preunlock_manifest(
        args.phase_f_dir,
        args.dependency_lock,
        args.output,
        repository_root=args.repository_root,
        frozen_at_utc=args.frozen_at_utc,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _task8b_build_analysis_input(args: argparse.Namespace) -> int:
    result = build_task8b_analysis_input(
        args.worker_manifest_dir,
        args.snapshot_root,
        args.output,
        args.pre_unlock_manifest,
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
