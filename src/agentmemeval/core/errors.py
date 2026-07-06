"""
模块说明：本模块定义重构平台的领域异常。
核心职责：为配置、动作、环境、Provider 和实验失败提供可区分的错误类型。
输入与输出：输入为错误上下文，输出为带中文说明的异常对象。
依赖边界：只依赖 Python 标准异常体系。
不负责：不记录日志，不吞掉异常，不执行恢复策略。
"""


class AgentMemEvalError(Exception):
    """
    功能：作为本项目所有可预期领域异常的基类。
    参数：
        message：面向用户或开发者的中文错误说明。
    返回：异常实例。
    副作用：无。
    异常：无。
    设计说明：统一基类便于 CLI 将可预期错误格式化为清晰输出。
    """


class ConfigError(AgentMemEvalError):
    """
    功能：表示配置加载、合并或校验失败。
    参数：
        message：错误说明。
    返回：异常实例。
    副作用：无。
    异常：无。
    设计说明：配置错误通常应在实验开始前暴露，避免产出半截工件。
    """


class ActionValidationError(AgentMemEvalError):
    """
    功能：表示 LLM 或 Agent 给出的动作不满足当前合法动作集合。
    参数：
        message：错误说明。
    返回：异常实例。
    副作用：无。
    异常：无。
    设计说明：动作合法性是防止脆弱文本解析污染环境状态的最后边界。
    """


class EnvironmentError(AgentMemEvalError):
    """
    功能：表示扑克环境推进时发生不可恢复的状态错误。
    参数：
        message：错误说明。
    返回：异常实例。
    副作用：无。
    异常：无。
    设计说明：环境是真相源，遇到非法推进应立即失败而不是静默修正。
    """


class ProviderError(AgentMemEvalError):
    """
    功能：表示 LLM Provider 初始化或调用失败。
    参数：
        message：错误说明。
    返回：异常实例。
    副作用：无。
    异常：无。
    设计说明：Provider 错误与 Agent 逻辑解耦，便于替换真实或 mock 后端。
    """
