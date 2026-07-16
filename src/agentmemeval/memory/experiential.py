"""
模块说明：本模块实现 ExprAgent 使用的版本化经验文档。
核心职责：按最近轨迹窗口更新经验摘要，保留版本历史并控制长度。
输入与输出：输入观察与可见轨迹，输出 MemoryContext 和 MemorySnapshot。
依赖边界：不依赖事实库或具体 LLM Provider；mock 场景使用确定性更新。
不负责：不检索事实，不决定动作。
"""

from __future__ import annotations

import hashlib
import json

from agentmemeval.core.domain import (
    AgentObservation,
    DecisionEvent,
    ExperienceDocument,
    HandTrajectory,
    MemoryContext,
    MemoryScope,
    MemorySnapshot,
    utc_now_iso,
)
from agentmemeval.core.protocols import LLMClient
from agentmemeval.llm.schemas import LLMRequest
from agentmemeval.memory.base import trajectory_quality
from agentmemeval.prompts.experience_update import EXPERIENCE_UPDATE_PROMPT

EXPERIENCE_REVISION_SCHEMA_VERSION = "experience_revision_v1"
EXPERIENCE_PROMPT_VERSION = "2026-07-17-v2-length-bounded"

INITIAL_EXPERIENCE = """# 我的经验

## 起手牌
（暂无）

## 翻牌后
（暂无）

## 转牌 / 河牌
（暂无）

## 对手类型应对
（暂无）

## 注码与位置
（暂无）
"""


class ExperientialMemory:
    """
    功能：维护可版本化的经验文档。
    参数：
        agent_id：所属 Agent。
        scope：记忆作用域。
        window_size：更新时使用的最近轨迹窗口。
        max_chars：经验正文长度上限。
        update_period：每多少手更新一次。
    返回：经验记忆机制。
    副作用：手牌结束时可能追加经验版本。
    异常：无。
    设计说明：初版用确定性摘要保证离线测试，真实 LLM 修订可作为替换策略接入。
    """

    name = "expr"

    def __init__(
        self,
        agent_id: str,
        scope: MemoryScope = "per_agent",
        window_size: int = 8,
        max_chars: int = 1600,
        update_period: int = 1,
        revision_strategy: str = "deterministic",
        llm_client: LLMClient | None = None,
        model: str = "",
    ) -> None:
        """
        功能：初始化经验记忆。
        参数：
            agent_id：所属 Agent。
            scope：记忆作用域。
            window_size：最近轨迹窗口。
            max_chars：长度上限。
            update_period：更新周期。
        返回：无。
        副作用：创建初始经验版本。
        异常：无。
        设计说明：经验文档从固定模板开始，便于快照比较。
        """

        self.agent_id = agent_id
        self.scope: MemoryScope = scope
        self.window_size = window_size
        self.max_chars = max_chars
        self.update_period = max(1, update_period)
        if revision_strategy not in {"deterministic", "llm"}:
            raise ValueError(f"未知 experience revision strategy：{revision_strategy}")
        if revision_strategy == "llm" and llm_client is None:
            raise ValueError("LLM experience revision 需要 llm_client")
        self.revision_strategy = revision_strategy
        self.llm_client = llm_client
        self.model = model
        self.trajectories: list[HandTrajectory] = []
        self.skipped_trajectory_hand_ids: list[str] = []
        self.revision_log: list[dict[str, object]] = []
        self.history: list[ExperienceDocument] = [
            ExperienceDocument(
                version=1,
                body=INITIAL_EXPERIENCE,
                source_hand_ids=[],
                updated_at=utc_now_iso(),
                scope=scope,
                metadata={
                    "reason": "initial",
                    "structure": "five_section_experience_doc",
                },
            )
        ]

    @property
    def current(self) -> ExperienceDocument:
        """
        功能：返回当前经验版本。
        参数：无。
        返回：ExperienceDocument。
        副作用：无。
        异常：无。
        设计说明：使用属性避免外部直接依赖 history 列表结构。
        """

        return self.history[-1]

    def build_context(self, observation: AgentObservation) -> MemoryContext:
        """
        功能：构造包含经验文档的记忆上下文。
        参数：
            observation：当前观察。
        返回：MemoryContext。
        副作用：无。
        异常：无。
        设计说明：经验作用域由快照和配置记录，当前版本不因观察即时变化。
        """

        return MemoryContext(
            experience=self.current,
            metadata={
                "mechanism": self.name,
                "scope": self.scope,
                "experience_version": self.current.version,
            },
        )

    def on_decision_committed(self, event: DecisionEvent) -> None:
        """
        功能：接收已提交决策事件。
        参数：
            event：决策事件。
        返回：无。
        副作用：无。
        异常：无。
        设计说明：经验更新在手牌结束时进行，避免使用未结算回报。
        """

    def on_hand_finished(self, trajectory: HandTrajectory) -> None:
        """
        功能：根据最近轨迹窗口更新经验文档。
        参数：
            trajectory：可见轨迹。
        返回：无。
        副作用：追加轨迹，必要时追加经验版本。
        异常：无。
        设计说明：每次更新记录来源手牌和窗口大小，支持复现实验审计。
        """

        if not trajectory_quality(trajectory)["memory_eligible"]:
            self.skipped_trajectory_hand_ids.append(trajectory.hand_id)
            return
        self.trajectories.append(trajectory)
        if len(self.trajectories) % self.update_period != 0:
            return
        recent = self.trajectories[-self.window_size :]
        revision = self._revise_from_window(recent)
        body = str(revision["new_md"])
        self.revision_log.append(revision)
        if body == self.current.body:
            return
        self.history.append(
            ExperienceDocument(
                version=self.current.version + 1,
                body=body,
                source_hand_ids=[item.hand_id for item in recent],
                updated_at=utc_now_iso(),
                scope=self.scope,
                metadata={
                    "window_size": len(recent),
                    "update_index": len(self.history),
                    "strategy": revision["revision_strategy"],
                    "schema_version": revision.get("schema_version"),
                    "prompt_version": revision.get("prompt_version"),
                    "prompt_sha256": revision.get("prompt_sha256"),
                    "fallback_used": revision.get("fallback_used", False),
                    "failure": revision.get("failure"),
                    "calibration_note": revision["calibration_note"],
                    "self_check": revision["self_check"],
                    "supporting_fact_ids": revision["supporting_fact_ids"],
                    "contradicting_fact_ids": revision["contradicting_fact_ids"],
                    "noise_fact_ids": revision["noise_fact_ids"],
                },
            )
        )

    def snapshot(self) -> MemorySnapshot:
        """
        功能：导出经验记忆快照。
        参数：无。
        返回：MemorySnapshot。
        副作用：无。
        异常：无。
        设计说明：保存完整版本历史，便于分析经验如何演化。
        """

        return MemorySnapshot(
            mechanism=self.name,
            agent_id=self.agent_id,
            scope=self.scope,
            payload={
                "window_size": self.window_size,
                "max_chars": self.max_chars,
                "update_period": self.update_period,
                "revision_strategy": self.revision_strategy,
                "revision_model": self.model,
                "history": [doc.to_dict() for doc in self.history],
                "revision_log": list(self.revision_log),
                "skipped_trajectory_hand_ids": list(self.skipped_trajectory_hand_ids),
            },
        )

    def restore(self, snapshot: MemorySnapshot) -> None:
        """
        功能：从快照恢复经验历史。
        参数：
            snapshot：经验快照。
        返回：无。
        副作用：替换当前经验版本历史。
        异常：无。
        设计说明：恢复后 trajectories 清空，避免训练窗口泄露到泛化测试。
        """

        self.scope = snapshot.scope
        self.window_size = int(snapshot.payload.get("window_size", self.window_size))
        self.max_chars = int(snapshot.payload.get("max_chars", self.max_chars))
        self.update_period = int(snapshot.payload.get("update_period", self.update_period))
        self.revision_strategy = str(
            snapshot.payload.get("revision_strategy", self.revision_strategy)
        )
        self.model = str(snapshot.payload.get("revision_model", self.model))
        history = snapshot.payload.get("history", [])
        self.history = [ExperienceDocument(**doc) for doc in history] or self.history
        self.revision_log = list(snapshot.payload.get("revision_log", []))
        self.skipped_trajectory_hand_ids = list(
            snapshot.payload.get("skipped_trajectory_hand_ids", [])
        )
        self.trajectories = []

    def metrics(self) -> dict[str, object]:
        """
        功能：返回经验记忆指标。
        参数：无。
        返回：指标字典。
        副作用：无。
        异常：无。
        设计说明：报告关注更新次数和正文长度，观察过拟合风险。
        """

        return {
            "mechanism": self.name,
            "fact_count": 0,
            "experience_updates": max(0, len(self.history) - 1),
            "experience_version": self.current.version,
            "experience_chars": len(self.current.body),
            "revision_count": len(self.revision_log),
            "revision_strategy": self.revision_strategy,
            "revision_model": self.model,
            "revision_fallback_count": sum(
                bool(item.get("fallback_used")) for item in self.revision_log
            ),
            "last_revision": self.revision_log[-1] if self.revision_log else {},
            "skipped_fallback_trajectories": len(self.skipped_trajectory_hand_ids),
        }

    def _revise_from_window(
        self,
        recent: list[HandTrajectory],
        evidence_records: list[object] | None = None,
    ) -> dict[str, object]:
        """Run the selected revision strategy with an audited deterministic fallback."""

        deterministic = self._deterministic_revision(recent)
        if self.revision_strategy == "deterministic":
            return deterministic
        assert self.llm_client is not None
        trajectory_ids = [item.hand_id for item in recent]
        fact_payload = [
            {
                "record_id": str(getattr(item, "record_id", "")),
                "state_summary": str(getattr(item, "state_summary", "")),
                "action_summary": str(getattr(item, "action_summary", "")),
                "final_reward": int(getattr(item, "final_reward", 0)),
            }
            for item in (evidence_records or [])
        ]
        fact_ids = [str(item["record_id"]) for item in fact_payload if item["record_id"]]
        evidence_ids = [*trajectory_ids, *fact_ids]
        trajectory_payload = [
            {
                "hand_id": item.hand_id,
                "summary": item.summary,
                "final_reward": item.final_reward,
            }
            for item in recent
        ]
        system_prompt = (
            "你负责修订可迁移的德州扑克经验文档。只使用给定轨迹与事实证据，不写具体玩家身份，"
            "输出 JSON 对象，必须包含 keep,new_md,calibration_note,self_check,"
            "supporting_fact_ids,contradicting_fact_ids,noise_fact_ids。"
            f"new_md 最多 {self.max_chars} 个字符；calibration_note 与 self_check "
            "各最多 240 个字符；只保留简洁结论，不输出思维过程。"
        )
        user_prompt = (
            f"{EXPERIENCE_UPDATE_PROMPT}\n\n旧文档：\n{self.current.body}\n\n"
            f"证据轨迹：\n{json.dumps(trajectory_payload, ensure_ascii=False)}\n\n"
            f"检索事实证据：\n{json.dumps(fact_payload, ensure_ascii=False)}"
        )
        prompt_hash = hashlib.sha256(
            (system_prompt + "\n" + user_prompt).encode("utf-8")
        ).hexdigest()
        observation = next(
            (
                event.observation
                for trajectory in reversed(recent)
                for event in reversed(trajectory.decision_events)
            ),
            None,
        )
        if observation is None:
            return {
                **deterministic,
                "fallback_used": True,
                "failure": "no_decision_observation_for_llm_revision",
                "prompt_version": EXPERIENCE_PROMPT_VERSION,
                "prompt_sha256": prompt_hash,
                "schema_version": EXPERIENCE_REVISION_SCHEMA_VERSION,
            }
        request = LLMRequest(
            observation=observation,
            memory_context=self.build_context(observation),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_config={"model": self.model},
            metadata={
                "response_schema": EXPERIENCE_REVISION_SCHEMA_VERSION,
                "evidence_ids": evidence_ids,
                "experience_max_chars": self.max_chars,
                "experience_aux_max_chars": 240,
            },
        )
        try:
            payload = self.llm_client.generate_structured(request, dict)
            if not isinstance(payload, dict) or not str(payload.get("new_md", "")).strip():
                raise ValueError("LLM revision 缺少非空 new_md")
            new_md = str(payload["new_md"]).strip() + "\n"
            if len(new_md) > self.max_chars:
                new_md = new_md[: self.max_chars - 20].rstrip() + "\n- （因长度上限截断）\n"
            allowed_ids = set(evidence_ids)
            result = {
                "rev": len(self.revision_log) + 1,
                "hand_index": recent[-1].hand_id if recent else "",
                "keep": bool(payload.get("keep", new_md == self.current.body)),
                "old_md": self.current.body,
                "new_md": new_md,
                "calibration_note": str(payload.get("calibration_note", "")),
                "self_check": str(payload.get("self_check", "")),
                "supporting_fact_ids": _filter_evidence_ids(
                    payload, "supporting_fact_ids", allowed_ids
                ),
                "contradicting_fact_ids": _filter_evidence_ids(
                    payload, "contradicting_fact_ids", allowed_ids
                ),
                "noise_fact_ids": _filter_evidence_ids(payload, "noise_fact_ids", allowed_ids),
                "revision_strategy": "llm_structured_revision",
                "schema_version": EXPERIENCE_REVISION_SCHEMA_VERSION,
                "prompt_version": EXPERIENCE_PROMPT_VERSION,
                "prompt_sha256": prompt_hash,
                "model": self.model,
                "fallback_used": False,
                "failure": None,
            }
            return result
        except Exception as exc:  # noqa: BLE001
            return {
                **deterministic,
                "fallback_used": True,
                "failure": f"{type(exc).__name__}: {exc}",
                "prompt_version": EXPERIENCE_PROMPT_VERSION,
                "prompt_sha256": prompt_hash,
                "schema_version": EXPERIENCE_REVISION_SCHEMA_VERSION,
                "requested_strategy": "llm",
            }

    def _deterministic_revision(self, recent: list[HandTrajectory]) -> dict[str, object]:
        """
        功能：基于最近轨迹生成确定性五章节经验修订。
        参数：
            recent：最近轨迹窗口。
        返回：经验修订记录。
        副作用：无。
        异常：无。
        设计说明：模拟原版经验修订 JSON 的 keep/new_md/calibration/self_check 结构。
        """

        rewards = [item.final_reward for item in recent]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        action_counts: dict[str, int] = {}
        postflop_aggression = 0
        cheap_calls = 0
        expensive_losses = 0
        showdown_count = 0
        supporting_fact_ids: list[str] = []
        for item in recent:
            for event in item.decision_events:
                action = event.committed_action.action_type
                action_counts[action] = action_counts.get(action, 0) + 1
                if event.observation.phase in {"flop", "turn", "river"} and action == "raise":
                    postflop_aggression += 1
                if action == "call" and event.observation.to_call <= 2:
                    cheap_calls += 1
                if item.final_reward < 0 and event.observation.to_call >= 8:
                    expensive_losses += 1
            if item.showdown_visible_cards:
                showdown_count += 1
            supporting_fact_ids.append(item.hand_id)
        total_actions = max(1, sum(action_counts.values()))
        raise_rate = action_counts.get("raise", 0) / total_actions
        fold_rate = action_counts.get("fold", 0) / total_actions
        preflop_line = (
            "高对子、同花高张和连张可主动进入底池；边缘牌在高 to_call 下减少跟注。"
            if avg_reward >= 0
            else "近期收益偏弱，起手牌选择收紧，避免用弱踢脚高张支付大额入池成本。"
        )
        flop_line = (
            "翻牌后若已形成强听牌或强成牌，可用小尺度加注争取主动。"
            if postflop_aggression
            else "翻牌后缺少成牌或听牌时优先控制底池，避免把一次性诈唬写成长期规则。"
        )
        late_line = (
            "转牌/河牌面对大额下注时，只有强牌或明确赔率支持的听牌才继续。"
            if expensive_losses
            else "转牌/河牌保持摊牌价值判断，避免因前街投入而自动跟到底。"
        )
        opponent_line = (
            "对持续跟注型对手减少纯诈唬；对频繁弃牌型对手可保留小额持续下注。"
            if showdown_count
            else "暂不绑定具体对手身份，只记录 loose、tight、station 等类型化应对。"
        )
        position_line = (
            "便宜补注可结合位置和底池赔率继续；昂贵补注需要更高牌力阈值。"
            if cheap_calls
            else "先观察位置、to_call 与 pot 的比例，再决定是否主动扩大底池。"
        )
        behavior = (
            ", ".join(f"{key}={value}" for key, value in sorted(action_counts.items()))
            or "无行动"
        )
        body = "\n".join(
            [
                "# 我的经验",
                "",
                "## 起手牌",
                f"- {preflop_line}",
                "",
                "## 翻牌后",
                f"- {flop_line}",
                "",
                "## 转牌 / 河牌",
                f"- {late_line}",
                "",
                "## 对手类型应对",
                f"- {opponent_line}",
                "",
                "## 注码与位置",
                f"- {position_line}",
                f"- 最近 {len(recent)} 手牌平均收益 {avg_reward:.2f}；行为分布：{behavior}。",
                "",
            ]
        )
        body = body.rstrip() + "\n"
        if len(body) > self.max_chars:
            body = body[: self.max_chars - 20].rstrip() + "\n- （因长度上限截断）\n"
        calibration = (
            f"近期平均收益 {avg_reward:.2f}，"
            f"raise_rate={raise_rate:.2f}，fold_rate={fold_rate:.2f}；"
            "若主动线未带来收益则收紧边缘投入。"
        )
        return {
            "rev": len(self.revision_log) + 1,
            "hand_index": recent[-1].hand_id if recent else "",
            "keep": body == self.current.body,
            "old_md": self.current.body,
            "new_md": body,
            "calibration_note": calibration,
            "self_check": "未写入具体玩家身份；经验仅保留跨手、跨桌可迁移规律。",
            "supporting_fact_ids": supporting_fact_ids[-self.window_size :],
            "contradicting_fact_ids": [],
            "noise_fact_ids": [],
            "revision_strategy": "deterministic_five_section_revision",
            "schema_version": EXPERIENCE_REVISION_SCHEMA_VERSION,
            "prompt_version": None,
            "prompt_sha256": None,
            "fallback_used": False,
            "failure": None,
        }


def _filter_evidence_ids(
    payload: dict[str, object], key: str, allowed_ids: set[str]
) -> list[str]:
    values = payload.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value) in allowed_ids]
