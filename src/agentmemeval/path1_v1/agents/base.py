from __future__ import annotations

import hashlib
import time
from pathlib import Path

from agentmemeval.path1_v1.core.domain import (
    ActionDecision,
    AgentId,
    AgentObservation,
    DecisionEvent,
    HandTrajectory,
    MemoryContext,
    MemorySnapshot,
)
from agentmemeval.path1_v1.core.protocols import LLMClient, MemoryMechanism
from agentmemeval.path1_v1.environment.action_guard import ActionGuard
from agentmemeval.path1_v1.environment.raise_sizing import build_raise_sizing_plan
from agentmemeval.path1_v1.llm.schemas import LLMCallStats, LLMRequest
from agentmemeval.path1_v1.prompts.decision import (
    PROMPT_TEMPLATE_VERSION,
    render_system_prompt,
    render_user_prompt,
)


class LLMDecisionAgent:
    def __init__(
        self,
        agent_id: AgentId,
        memory: MemoryMechanism,
        llm_client: LLMClient,
        model: str = "",
        guard: ActionGuard | None = None,
        raise_sizing_policy: str = "native_no_limit",
    ) -> None:

        self.agent_id = agent_id
        self.memory = memory
        self.llm_client = llm_client
        self.model = model
        self.guard = guard or ActionGuard()
        self.raise_sizing_policy = raise_sizing_policy

    def decide(
        self,
        observation: AgentObservation,
    ) -> tuple[ActionDecision, MemoryContext, dict[str, object]]:

        context = self.memory.build_context(observation)
        raise_sizing = build_raise_sizing_plan(observation, self.raise_sizing_policy)
        system_prompt = render_system_prompt(context)
        user_prompt = render_user_prompt(observation, context, raise_sizing=raise_sizing)
        request = LLMRequest(
            observation=observation,
            memory_context=context,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_config={"model": self.model},
            metadata={
                "seed": observation.seed,
                "raise_sizing": raise_sizing.to_dict(),
            },
        )
        started = time.perf_counter()
        raw_decision = self.llm_client.generate_structured(request, ActionDecision)
        elapsed_ms = (time.perf_counter() - started) * 1000
        guard_result = self.guard.guard(
            raw_decision,
            observation.legal_actions,
            strict=False,
            allowed_raise_amounts=raise_sizing.allowed_amounts,
        )
        stats = LLMCallStats(
            provider=getattr(self.llm_client, "provider", "unknown"),
            model=getattr(self.llm_client, "model", self.model),
            elapsed_ms=elapsed_ms,
            prompt_tokens=max(1, len(system_prompt.split()) + len(user_prompt.split())),
            completion_tokens=max(
                1,
                len(raw_decision.reason_summary.split()) + len(raw_decision.intent.split()) + 5,
            ),
        )
        metadata = {
            "raw_decision": raw_decision.to_dict(),
            "guard_repaired": guard_result.repaired,
            "guard_errors": list(guard_result.errors),
            "fallback_used": guard_result.fallback_used,
            "llm": stats.to_dict(),
            "raise_sizing": raise_sizing.to_dict(),
            "prompt": {
                "template_version": PROMPT_TEMPLATE_VERSION,
                "system_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
                "user_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
                "system_chars": len(system_prompt),
                "user_chars": len(user_prompt),
            },
        }
        return guard_result.action, context, metadata

    def observe_decision(self, event: DecisionEvent) -> None:

        self.memory.on_decision_committed(event)

    def observe_hand_finished(self, trajectory: HandTrajectory) -> None:

        self.memory.on_hand_finished(trajectory)

    def snapshot_memory(self) -> MemorySnapshot:

        return self.memory.snapshot()

    def restore_memory(self, snapshot: MemorySnapshot) -> None:

        self.memory.restore(snapshot)

    def bind_memory_persistence(
        self,
        directory: str | Path,
        *,
        reference_root: str | Path,
    ) -> None:
        """Bind file-backed memory artifacts when the selected mechanism supports them."""

        binder = getattr(self.memory, "bind_persistence", None)
        if callable(binder):
            binder(Path(directory), reference_root=Path(reference_root))

    def set_memory_frozen(self, frozen: bool) -> None:
        """Switch a memory clone into or out of its read-only lifecycle."""

        setter = getattr(self.memory, "set_frozen", None)
        if callable(setter):
            setter(bool(frozen))

    def experience_reference(self) -> dict[str, object] | None:
        """Return a compact pointer to the canonical experience document, if present."""

        reference = getattr(self.memory, "experience_reference", None)
        if not callable(reference):
            return None
        payload = reference()
        return dict(payload) if isinstance(payload, dict) else None

    def memory_metrics(self) -> dict[str, object]:

        return self.memory.metrics()
