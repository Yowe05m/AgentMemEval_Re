class AgentMemEvalError(Exception):
    pass


class ConfigError(AgentMemEvalError):
    pass


class ActionValidationError(AgentMemEvalError):
    pass


class EnvironmentError(AgentMemEvalError):
    pass


class ProviderError(AgentMemEvalError):
    def __init__(
        self,
        message: str,
        *,
        diagnostics: dict[str, object] | None = None,
        retryable: bool = True,
    ) -> None:
        """Carry safe structured diagnostics without embedding them in error text."""

        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})
        self.retryable = bool(retryable)
