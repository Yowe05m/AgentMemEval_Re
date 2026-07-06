"""
模块说明：本模块负责 `python -m agentmemeval` 的命令入口。
核心职责：把模块执行转发到 CLI 主函数。
输入与输出：读取命令行参数并返回进程退出码。
依赖边界：仅依赖 CLI 层，不直接依赖实验实现细节。
不负责：不解析具体实验配置，不直接创建环境或 Agent。
"""

from agentmemeval.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
