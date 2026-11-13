from __future__ import annotations

from dataclasses import dataclass, field

from agentmemeval.path1_v1.core.domain import AgentObservation, JsonDict, MemoryContext


@dataclass(slots=True)
class LLMRequest:
    observation: AgentObservation
    memory_context: MemoryContext
    system_prompt: str
    user_prompt: str
    provider_config: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class LLMCallStats:
    provider: str
    model: str
    retries: int = 0
    elapsed_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_saved: bool = False

    def to_dict(self) -> JsonDict:

        return {
            "provider": self.provider,
            "model": self.model,
            "retries": self.retries,
            "elapsed_ms": self.elapsed_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "raw_saved": self.raw_saved,
        }
