# agent/graph.py
# ============================================================
# LangGraph 工作流图：节点 + 条件路由
# ============================================================
from langgraph.graph import END, START, StateGraph

from agent.nodes import final_node, planner_node, reflection_node, tool_node
from agent.state import AgentState
from config import config


def route_after_planner(state: AgentState) -> str:
    """planner 之后：需要工具则走 tools，否则直接走 final 收尾"""
    action = state.get("action_data", {}).get("action")
    if action == "call_tool":
        return "tools"
    return "final"


def route_after_reflection(state: AgentState) -> str:
    """reflection 之后：信息不足则回到 planner 继续，否则走 final 收尾"""
    if state.get("need_continue", False):
        return "planner"
    return "final"


def build_graph():
    """构建 Agent 工作流图"""
    builder = StateGraph(AgentState)

    # 注册节点
    builder.add_node("planner", planner_node)
    builder.add_node("tools", tool_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("final", final_node)

    # 入口
    builder.add_edge(START, "planner")

    # planner 条件路由：工具 or 直接回答
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"tools": "tools", "final": "final"},
    )

    # 工具执行后进入反思
    builder.add_edge("tools", "reflection")

    # 反思条件路由：继续 or 结束
    builder.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {"planner": "planner", "final": "final"},
    )

    # 最终节点后结束
    builder.add_edge("final", END)

    return builder.compile()


# 全局图实例
graph = build_graph()


def run_agent(query: str, messages: list = None) -> dict:
    """
    便捷函数：运行 Agent，返回完整结果。

    Args:
        query: 用户问题
        messages: 历史对话消息列表 [(role, content), ...]

    Returns:
        dict: 包含 final_answer, trace, tool_results 等
    """
    init_state = {
        "query": query,
        "messages": messages or [],
        "tool_results": [],
        "step": 0,
        "action_data": {},
        "need_continue": True,
        "final_answer": "",
        "trace": [],
    }

    final_state = graph.invoke(init_state)
    return final_state


def stream_agent(query: str, messages: list = None):
    """
    流式运行 Agent，逐个产出节点事件。

    Yields:
        dict: 每个节点的执行结果
    """
    init_state = {
        "query": query,
        "messages": messages or [],
        "tool_results": [],
        "step": 0,
        "action_data": {},
        "need_continue": True,
        "final_answer": "",
        "trace": [],
    }

    for event in graph.stream(init_state):
        yield event
