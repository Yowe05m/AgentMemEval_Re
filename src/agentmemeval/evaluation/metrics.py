"""
模块说明：本模块计算运行级行为、收益和记忆指标。
核心职责：从手牌摘要和事件日志中生成 metrics.json 所需字段。
输入与输出：输入 JSONL 记录列表，输出指标字典。
依赖边界：只依赖标准库和统计工具，不依赖实验场景。
不负责：不绘图，不写文件。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

BLUFF_KEYWORDS = ("bluff", "诈唬", "虚张", "唬人", "诈")
POSTFLOP_PHASES = {"flop", "turn", "river"}


def compute_metrics(
    hand_summaries: list[dict[str, Any]],
    events: list[dict[str, Any]],
    big_blind: int,
    memory_metrics: dict[str, dict[str, Any]] | None = None,
    exposure_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    功能：计算实验主要指标和探索性指标。
    参数：
        hand_summaries：手牌摘要记录。
        events：事件记录。
        big_blind：大盲数值。
        memory_metrics：按 Agent 的记忆指标。
        exposure_stats：换桌暴露统计。
    返回：指标字典。
    副作用：无。
    异常：无。
    设计说明：指标从原始工件重建，避免场景代码内散落统计逻辑。
    """

    rewards_by_agent: dict[str, int] = defaultdict(int)
    hands_by_agent: dict[str, int] = defaultdict(int)
    wins_by_agent: dict[str, int] = defaultdict(int)
    stage_rewards: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for hand in hand_summaries:
        stage = str(hand.get("stage", "unknown"))
        rewards = hand.get("rewards", {}) or {}
        for agent_id, reward in rewards.items():
            value = int(reward)
            rewards_by_agent[agent_id] += value
            stage_rewards[stage][agent_id] += value
            hands_by_agent[agent_id] += 1
            if value > 0:
                wins_by_agent[agent_id] += 1
    action_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    vpip_hands: dict[str, set[str]] = defaultdict(set)
    faced_raise = defaultdict(int)
    folded_to_raise = defaultdict(int)
    postflop_raise_hands: dict[str, set[str]] = defaultdict(set)
    intent_bluff_hands: dict[str, set[str]] = defaultdict(set)
    for event in events:
        if event.get("event") != "action":
            continue
        agent_id = str(event.get("agent_id"))
        action = str(event.get("action_type"))
        hand_id = str(event.get("hand_id"))
        action_counts[agent_id][action] += 1
        if action in {"call", "raise"}:
            vpip_hands[agent_id].add(hand_id)
        if int(event.get("to_call") or 0) > 0:
            faced_raise[agent_id] += 1
            if action == "fold":
                folded_to_raise[agent_id] += 1
        effective_raise = bool(event.get("effective_raise", False))
        if action == "raise" and str(event.get("phase")) in POSTFLOP_PHASES and effective_raise:
            postflop_raise_hands[agent_id].add(hand_id)
            raw_decision = event.get("raw_decision", {}) or {}
            if _has_bluff_intent(raw_decision):
                intent_bluff_hands[agent_id].add(hand_id)
    showdown_ranks_by_hand: dict[str, dict[str, str]] = {}
    for hand in hand_summaries:
        ranks = hand.get("showdown_ranks", {}) or {}
        if ranks:
            showdown_ranks_by_hand[str(hand.get("hand_id"))] = {
                str(agent_id): str(rank) for agent_id, rank in dict(ranks).items()
            }
    proxy_bluff = _compute_proxy_bluff_rates(postflop_raise_hands, showdown_ranks_by_hand)
    intent_bluff = {
        agent_id: (
            len(intent_bluff_hands.get(agent_id, set())) / len(hands)
            if hands
            else 0.0
        )
        for agent_id, hands in postflop_raise_hands.items()
    }
    per_agent: dict[str, Any] = {}
    for agent_id in sorted(hands_by_agent):
        hands = max(1, hands_by_agent[agent_id])
        reward = rewards_by_agent[agent_id]
        counts = dict(action_counts.get(agent_id, {}))
        total_actions = max(1, sum(counts.values()))
        per_agent[agent_id] = {
            "hands": hands,
            "chip_delta": reward,
            "bb_per_100": (reward / max(1, big_blind)) / hands * 100,
            "win_rate": wins_by_agent[agent_id] / hands,
            "vpip": len(vpip_hands.get(agent_id, set())) / hands,
            "fold_rate": counts.get("fold", 0) / total_actions,
            "check_rate": counts.get("check", 0) / total_actions,
            "call_rate": counts.get("call", 0) / total_actions,
            "raise_rate": counts.get("raise", 0) / total_actions,
            "fold_to_raise": (
                folded_to_raise[agent_id] / faced_raise[agent_id]
                if faced_raise[agent_id]
                else 0.0
            ),
            "proxy_bluff_rate": proxy_bluff.get(agent_id, 0.0),
            "intent_bluff_rate": intent_bluff.get(agent_id, 0.0),
            "action_counts": counts,
            "memory": (memory_metrics or {}).get(agent_id, {}),
        }
    train = stage_rewards.get("train", {})
    test = stage_rewards.get("test", {})
    generalization_gap = {
        agent_id: train.get(agent_id, 0) - test.get(agent_id, 0)
        for agent_id in sorted(set(train) | set(test))
    }
    return {
        "primary_metrics": {
            "per_agent": per_agent,
            "generalization_gap_chip_delta": generalization_gap,
        },
        "exploratory_metrics": {
            "action_behavior": {
                agent_id: dict(counts) for agent_id, counts in action_counts.items()
            },
            "opponent_diversity": exposure_stats or {},
            "bluff_rate": {
                "proxy_high_card_showdown_after_postflop_raise": proxy_bluff,
                "intent_keyword_after_postflop_raise": intent_bluff,
            },
        },
        "run_counters": {
            "hands": len(hand_summaries),
            "actions": sum(sum(counts.values()) for counts in action_counts.values()),
            "agents": len(per_agent),
        },
    }


def _compute_proxy_bluff_rates(
    postflop_raise_hands: dict[str, set[str]],
    showdown_ranks_by_hand: dict[str, dict[str, str]],
) -> dict[str, float]:
    """
    功能：计算原版 proxy 诈唬率。
    参数：
        postflop_raise_hands：每个 Agent 有效翻后 raise 的手牌集合。
        showdown_ranks_by_hand：每手摊牌牌型。
    返回：按 Agent 的诈唬率。
    副作用：无。
    异常：无。
    设计说明：分母为有效翻后 raise 且进入摊牌，分子为摊牌 High Card。
    """

    rates: dict[str, float] = {}
    for agent_id, hand_ids in postflop_raise_hands.items():
        qualifies = 0
        bluffs = 0
        for hand_id in hand_ids:
            ranks = showdown_ranks_by_hand.get(hand_id, {})
            rank = ranks.get(agent_id)
            if not rank:
                continue
            qualifies += 1
            if "high card" in rank.lower():
                bluffs += 1
        rates[agent_id] = bluffs / qualifies if qualifies else 0.0
    return rates


def _has_bluff_intent(raw_decision: object) -> bool:
    """
    功能：从结构化动作摘要中识别诈唬 intent。
    参数：
        raw_decision：事件里的 raw_decision 字段。
    返回：是否包含诈唬关键词。
    副作用：无。
    异常：无。
    设计说明：对应原版 intent-based bluff rate，离线 mock 用 reason_summary 替代自述 intent。
    """

    if not isinstance(raw_decision, dict):
        return False
    text = " ".join(
        str(raw_decision.get(key, ""))
        for key in ("intent", "reason", "reason_summary")
    ).lower()
    return any(keyword.lower() in text for keyword in BLUFF_KEYWORDS)
