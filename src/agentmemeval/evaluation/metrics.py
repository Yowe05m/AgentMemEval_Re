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

    per_agent = _compute_per_agent(
        hand_summaries,
        events,
        big_blind=big_blind,
        memory_metrics=memory_metrics,
    )
    stages = sorted(
        {str(hand.get("stage", "unknown")) for hand in hand_summaries}
        | {
            str(event.get("stage", "unknown"))
            for event in events
            if event.get("event") == "action"
        }
    )
    stage_per_agent = {
        stage: _compute_per_agent(
            [hand for hand in hand_summaries if str(hand.get("stage", "unknown")) == stage],
            [event for event in events if str(event.get("stage", "unknown")) == stage],
            big_blind=big_blind,
        )
        for stage in stages
    }
    train = stage_per_agent.get("train", {})
    test = stage_per_agent.get("test", {})
    comparable_agents = sorted(set(train) & set(test))
    generalization_gap_chip_delta = {
        agent_id: train[agent_id]["chip_delta"] - test[agent_id]["chip_delta"]
        for agent_id in comparable_agents
    }
    generalization_gap_bb_per_100 = {
        agent_id: train[agent_id]["bb_per_100"] - test[agent_id]["bb_per_100"]
        for agent_id in comparable_agents
    }
    quality = _decision_quality(events)
    stage_quality = {
        stage: _decision_quality(
            [event for event in events if str(event.get("stage", "unknown")) == stage]
        )
        for stage in stages
    }
    return {
        "primary_metrics": {
            "per_agent": per_agent,
            "per_agent_scope": "all stages combined; use stage_per_agent for analysis",
            "stage_per_agent": stage_per_agent,
            "generalization_gap_chip_delta": generalization_gap_chip_delta,
            "generalization_gap_bb_per_100": generalization_gap_bb_per_100,
            "generalization_gap_definition": "train minus test; only agents present in both stages",
        },
        "exploratory_metrics": {
            "action_behavior": {
                agent_id: dict(metrics["action_counts"])
                for agent_id, metrics in per_agent.items()
            },
            "opponent_diversity": exposure_stats or {},
            "bluff_rate": {
                "proxy_high_card_showdown_after_postflop_raise": {
                    agent_id: metrics["proxy_bluff_rate"]
                    for agent_id, metrics in per_agent.items()
                },
                "intent_keyword_after_postflop_raise": {
                    agent_id: metrics["intent_bluff_rate"]
                    for agent_id, metrics in per_agent.items()
                },
            },
            "decision_quality": {"combined": quality, "by_stage": stage_quality},
            "raise_sizing": _raise_sizing_quality(events),
            "call_risk": {
                "combined": _call_risk_quality(hand_summaries, events),
                "by_stage": {
                    stage: _call_risk_quality(
                        [
                            hand
                            for hand in hand_summaries
                            if str(hand.get("stage", "unknown")) == stage
                        ],
                        [
                            event
                            for event in events
                            if str(event.get("stage", "unknown")) == stage
                        ],
                    )
                    for stage in stages
                },
            },
        },
        "run_counters": {
            "hands": len(hand_summaries),
            "actions": quality["decision_count"],
            "agents": len(per_agent),
        },
    }


def _compute_per_agent(
    hand_summaries: list[dict[str, Any]],
    events: list[dict[str, Any]],
    big_blind: int,
    memory_metrics: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute one internally consistent metric table for one stage or all stages."""

    rewards_by_agent: dict[str, int] = defaultdict(int)
    hands_by_agent: dict[str, int] = defaultdict(int)
    wins_by_agent: dict[str, int] = defaultdict(int)
    for hand in hand_summaries:
        for agent_id, reward in (hand.get("rewards", {}) or {}).items():
            value = int(reward)
            rewards_by_agent[agent_id] += value
            hands_by_agent[agent_id] += 1
            if value > 0:
                wins_by_agent[agent_id] += 1

    action_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    vpip_hands: dict[str, set[str]] = defaultdict(set)
    faced_raise: dict[str, int] = defaultdict(int)
    folded_to_raise: dict[str, int] = defaultdict(int)
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
        if (
            action == "raise"
            and str(event.get("phase")) in POSTFLOP_PHASES
            and bool(event.get("effective_raise", False))
        ):
            postflop_raise_hands[agent_id].add(hand_id)
            if _has_bluff_intent(event.get("raw_decision", {}) or {}):
                intent_bluff_hands[agent_id].add(hand_id)

    showdown_ranks_by_hand = {
        str(hand.get("hand_id")): {
            str(agent_id): str(rank)
            for agent_id, rank in dict(hand.get("showdown_ranks", {}) or {}).items()
        }
        for hand in hand_summaries
        if hand.get("showdown_ranks")
    }
    proxy_bluff = _compute_proxy_bluff_rates(postflop_raise_hands, showdown_ranks_by_hand)
    intent_bluff = {
        agent_id: len(intent_bluff_hands.get(agent_id, set())) / len(hand_ids)
        if hand_ids
        else 0.0
        for agent_id, hand_ids in postflop_raise_hands.items()
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
    return per_agent


def _decision_quality(events: list[dict[str, Any]]) -> dict[str, int | float]:
    """Report how often provider output required normalization or semantic fallback."""

    actions = [event for event in events if event.get("event") == "action"]
    repaired = sum(bool(event.get("guard_repaired")) for event in actions)
    fallback = sum(bool(event.get("fallback_used")) for event in actions)
    changed = sum(
        isinstance(event.get("raw_decision"), dict)
        and event["raw_decision"].get("action_type") != event.get("action_type")
        for event in actions
    )
    count = len(actions)
    return {
        "decision_count": count,
        "repaired_count": repaired,
        "repaired_rate": repaired / count if count else 0.0,
        "fallback_count": fallback,
        "fallback_rate": fallback / count if count else 0.0,
        "action_type_changed_count": changed,
        "action_type_changed_rate": changed / count if count else 0.0,
    }


def _raise_sizing_quality(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit policy use, native all-in selections, and discrete enum compliance."""

    policy_action_counts: dict[str, int] = defaultdict(int)
    policy_raise_counts: dict[str, int] = defaultdict(int)
    native_max_selected: dict[str, int] = defaultdict(int)
    enum_violations: dict[str, int] = defaultdict(int)
    for event in events:
        if event.get("event") != "action":
            continue
        sizing = event.get("raise_sizing") or {}
        policy = str(sizing.get("policy", "unknown")) if isinstance(sizing, dict) else "unknown"
        policy_action_counts[policy] += 1
        if event.get("action_type") != "raise":
            continue
        policy_raise_counts[policy] += 1
        amount = int(event.get("amount") or 0)
        native_max = sizing.get("native_max_amount") if isinstance(sizing, dict) else None
        if native_max is not None and amount == int(native_max):
            native_max_selected[policy] += 1
        allowed = sizing.get("allowed_amounts") if isinstance(sizing, dict) else None
        if isinstance(allowed, list) and allowed and amount not in {
            int(candidate) for candidate in allowed
        }:
            enum_violations[policy] += 1
    policies = sorted(set(policy_action_counts) | set(policy_raise_counts))
    return {
        "by_policy": {
            policy: {
                "action_count": policy_action_counts[policy],
                "raise_count": policy_raise_counts[policy],
                "native_max_selected_count": native_max_selected[policy],
                "discrete_enum_violation_count": enum_violations[policy],
            }
            for policy in policies
        }
    }


def _call_risk_quality(
    hand_summaries: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Audit calls that commit at least half the remaining stack or go all-in."""

    rewards = {
        (str(hand.get("hand_id")), str(agent_id)): int(reward)
        for hand in hand_summaries
        for agent_id, reward in (hand.get("rewards", {}) or {}).items()
    }
    calls = [
        event
        for event in events
        if event.get("event") == "action" and event.get("action_type") == "call"
    ]
    by_agent_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in calls:
        by_agent_events[str(event.get("agent_id"))].append(event)

    def summarize(agent_calls: list[dict[str, Any]]) -> dict[str, Any]:
        high_risk = [
            event
            for event in agent_calls
            if float((event.get("call_risk") or {}).get("stack_fraction", 0.0)) >= 0.5
        ]
        all_in = [
            event
            for event in agent_calls
            if bool((event.get("call_risk") or {}).get("is_all_in", False))
        ]
        high_risk_hands = {
            (str(event.get("hand_id")), str(event.get("agent_id"))) for event in high_risk
        }
        made_hands: dict[str, int] = defaultdict(int)
        for event in high_risk:
            made_hand = str((event.get("call_risk") or {}).get("made_hand_class", "unknown"))
            made_hands[made_hand] += 1
        count = len(agent_calls)
        return {
            "call_count": count,
            "high_risk_call_count": len(high_risk),
            "high_risk_call_rate": len(high_risk) / count if count else 0.0,
            "all_in_call_count": len(all_in),
            "high_risk_hand_count": len(high_risk_hands),
            "high_risk_hand_net_reward": sum(
                rewards.get(hand_agent, 0) for hand_agent in high_risk_hands
            ),
            "high_risk_made_hand_counts": dict(sorted(made_hands.items())),
            "missing_call_risk_metadata": sum(
                not isinstance(event.get("call_risk"), dict) for event in agent_calls
            ),
        }

    return {
        "all_agents": summarize(calls),
        "by_agent": {
            agent_id: summarize(agent_calls)
            for agent_id, agent_calls in sorted(by_agent_events.items())
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
