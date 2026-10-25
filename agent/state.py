# agent/state.py
# ============================================================
# Agent 状态定义：节点间传递的共享数据
# ============================================================
from typing import Annotated, List, TypedDict, Optional
import operator


class AgentState(TypedDict, total=False):
    """
    Agent 全局状态。

    使用 total=False 表示所有键都是可选的，
    避免节点返回时必须包含所有字段。
    """
    # 用户原始问题
    query: str
    # 对话历史，格式为 [(role, content), ...]，使用 add 运算符累加
    messages: Annotated[List, operator.add]
    # 工具返回结果列表，累加
    tool_results: Annotated[List, operator.add]
    # 当前执行步数（用于限制循环，防死循环）
    step: int
    # 规划节点输出的结构化指令
    action_data: dict
    # 反思节点决定是否继续调用工具
    need_continue: bool
    # 最终回答
    final_answer: str
    # 错误信息（如果有）
    error: str
    # 执行轨迹（用于 UI 展示思考过程）
    trace: Annotated[List, operator.add]
