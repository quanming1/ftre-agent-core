"""
内置工具定义

ReActRunner 需要的内置工具集中在此定义，与执行引擎解耦。
每个函数返回一个 Tool 实例，由 runner 注册到 agent 的 ToolRegistry。

工具清单：
- think:  内部思维工具，支持推理(think)和反思(reflect)两种模式
"""
from .base import Tool, ToolParameter


def create_think_tool() -> Tool:
    """创建 think 工具：内部思维工具，支持推理和反思两种模式"""

    def think(type: str, thought: str) -> str:
        return thought

    return Tool(
        name="think",
        description=(
            "你的内部思维空间，这是你对自己说话的地方，内容不会展示给用户。"
            "支持两种模式：\n"
            "- type=\"think\"：深度思考。收到用户消息时，先用 think 理解用户的真实意图，"
            "快速梳理需求要点：用户想做什么？涉及哪些模块？我已有的上下文够不够？需要先搜索还是可以直接动手？"
            "面对复杂问题时，用来拆解分析、头脑风暴、制定策略、规划下一步行动。在动手之前先想清楚。\n"
            "- type=\"reflect\"：自我反思。在完成一项任务或一系列操作之后，回顾刚才做了什么，审视是否有遗漏或错误。"
            "问自己：方案是最优的吗？边界情况考虑了吗？代码风格和项目一致吗？有没有多余的改动？\n"
            "要求：保持客观，不要自夸，做冷静的分析和判断。"
        ),
        parameters=[
            ToolParameter(
                name="type",
                type="string",
                description="思维模式：think=推理思考，reflect=回顾反思",
                required=True,
                enum=["think", "reflect"],
            ),
            ToolParameter(
                name="thought",
                type="string",
                description="你的内心独白",
                required=True,
            ),
        ],
        func=think,
    )


# 所有内置工具的工厂列表，方便批量注册
BUILTIN_TOOL_FACTORIES = [
    create_think_tool,
]
