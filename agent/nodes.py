# agent/nodes.py
# ============================================================
# Agent 节点实现
#   - planner_node:    规划节点，决定直接回答 or 调用工具
#   - tool_node:       工具执行节点
#   - reflection_node: 反思节点，判断是否继续
#   - final_node:      收尾节点，整合输出最终答案
# ============================================================
import json
import time
from typing import List, Tuple

from config import config
from agent.state import AgentState
from agent.tools import call_tool, get_tool_descriptions
from llm import get_llm
from utils.helpers import clean_json_response, try_parse_json, truncate_text
from utils.logger import get_logger

logger = get_logger("agent.nodes")

# 最大执行步数（从配置读取）
MAX_STEP = config.agent.max_steps


def _build_system_prompt() -> str:
    """构建规划器的系统提示词（优化版，适配小模型）"""
    tool_desc = get_tool_descriptions()
    return f"""你是一个智能助手规划器。你的任务是分析用户问题，决定直接回答还是调用工具。

可用工具列表：
{tool_desc}

规则：
1. 仔细分析用户问题，判断是否需要调用工具
2. 如果问题可以直接回答（如问候、简单常识），直接回答
3. 如果问题需要计算、查文件、处理文本等，选择最合适的工具
4. 严格按照 JSON 格式输出，不要输出任何多余文字

输出格式（二选一）：
直接回答：{{"action":"direct_answer","answer":"你的回答"}}
调用工具：{{"action":"call_tool","tool_name":"工具名","args":{{"参数名":"参数值"}}}}"""


def _build_reflection_prompt(query: str, tool_results: list) -> str:
    """构建反思节点的提示词"""
    results_str = "\n".join(tool_results[-3:])  # 只取最近3个结果，节省上下文
    return f"""用户原始问题: {query}

已调用工具得到的结果:
{results_str}

请判断：这些信息是否已经足够回答用户的问题？
- 如果信息足够回答问题，输出 need_continue: false
- 如果信息不足，还需要继续调用工具，输出 need_continue: true

严格只输出 JSON：{{\"need_continue\": true 或 false}}"""


def _build_final_prompt(query: str, tool_results: list) -> str:
    """构建最终回答的提示词"""
    results_str = "\n".join(tool_results[-5:])  # 取最近5个结果
    return f"""用户问题: {query}

工具执行结果:
{results_str}

请根据工具执行结果，给用户一个完整、自然、友好的最终回答。
要求：
1. 直接回答用户问题，不要提到"工具"或"调用"等内部过程
2. 如果有多个步骤，整合结果后给出连贯的回答
3. 回答简洁明了，重点突出
4. 用中文回答"""


def _safe_llm_call(
    messages: List[Tuple[str, str]],
    max_retries: int = 2,
    temperature: float = None,
    max_tokens: int = None,
) -> str:
    """
    安全调用 LLM，带自动重试和错误处理。

    Args:
        messages: 消息列表
        max_retries: 最大重试次数
        temperature: 温度（None 使用默认）
        max_tokens: 最大 token 数（None 使用默认）

    Returns:
        模型输出文本
    """
    llm = get_llm()
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            resp = llm.invoke(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.content
        except Exception as e:
            last_error = e
            logger.warning(
                f"LLM 调用失败（第 {attempt + 1}/{max_retries + 1} 次）: {e}"
            )
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))  # 指数退避

    # 所有重试都失败
    logger.error(f"LLM 调用最终失败: {last_error}")
    raise RuntimeError(f"大模型调用失败: {last_error}")


def _manage_context(messages: list) -> list:
    """
    上下文管理：当消息总字符数超过限制时，
    保留系统提示和最近的消息，截断较早的对话。
    """
    max_chars = config.agent.max_context_chars
    total_chars = sum(len(content) for _, content in messages)

    if total_chars <= max_chars:
        return messages

    # 保留第一条（system）和最后几条
    kept = []
    if messages and messages[0][0] == "system":
        kept.append(messages[0])
        remaining = messages[1:]
    else:
        remaining = messages[:]

    # 从后往前加，直到接近上限
    result = kept[:]
    current_chars = sum(len(content) for _, content in kept)

    for role, content in reversed(remaining):
        if current_chars + len(content) > max_chars * 0.9:
            break
        result.append((role, content))
        current_chars += len(content)

    # 如果有 system，保持 system 在最前
    if kept:
        final = [kept[0]] + [m for m in result if m[0] != "system"]
    else:
        final = list(reversed(result))

    logger.debug(
        f"上下文截断: {total_chars} -> {sum(len(c) for _, c in final)} 字符"
    )
    return final


def planner_node(state: AgentState):
    """
    规划节点：判断直接回答 or 调用工具，输出结构化 JSON 指令。

    优化点：
    - 改进的提示词工程（更适合小模型）
    - 上下文管理（防止 4K 溢出）
    - JSON 解析容错
    - 自动重试
    """
    query = state.get("query", "")
    messages = state.get("messages", [])
    tool_results = state.get("tool_results", [])
    step = state.get("step", 0)
    trace = state.get("trace", [])

    logger.info(f"[规划节点] 第 {step + 1} 步，问题: {query[:50]}...")

    system_prompt = _build_system_prompt()

    # 构建用户消息（包含工具结果上下文）
    user_msg_parts = [f"用户问题: {query}"]
    if tool_results:
        # 只包含最近的工具结果，节省上下文
        recent_results = tool_results[-3:]
        user_msg_parts.append(
            f"已执行的工具结果:\n" + "\n".join(recent_results)
        )
    user_msg_parts.append(f"当前步数: {step + 1}/{MAX_STEP}")
    user_msg = "\n\n".join(user_msg_parts)

    # 组装消息并管理上下文
    full_messages = [("system", system_prompt)]
    # 加入历史对话（最近几轮）
    if messages:
        full_messages.extend(messages[-4:])  # 只保留最近2轮对话
    full_messages.append(("user", user_msg))

    # 上下文管理
    full_messages = _manage_context(full_messages)

    try:
        resp_content = _safe_llm_call(full_messages, max_retries=1)
    except RuntimeError as e:
        # LLM 调用失败，降级处理
        logger.error(f"规划节点 LLM 调用失败: {e}")
        action_data = {
            "action": "direct_answer",
            "answer": f"抱歉，大模型服务暂时不可用（{str(e)}），请稍后再试。",
        }
        trace.append({"node": "planner", "status": "error", "detail": str(e)})
        return {
            "action_data": action_data,
            "step": step + 1,
            "trace": trace,
        }

    # 解析 JSON
    action_data = try_parse_json(resp_content)
    if action_data is None:
        # 解析失败，降级为直接回答
        logger.warning(f"规划节点 JSON 解析失败，降级为直接回答: {resp_content[:100]}")
        action_data = {"action": "direct_answer", "answer": resp_content.strip()}

    # 校验 action 字段
    if action_data.get("action") not in ("direct_answer", "call_tool"):
        action_data = {"action": "direct_answer", "answer": resp_content.strip()}

    trace.append({
        "node": "planner",
        "status": "ok",
        "action": action_data.get("action"),
        "detail": (
            f"工具: {action_data.get('tool_name')}"
            if action_data.get("action") == "call_tool"
            else "直接回答"
        ),
    })

    logger.info(
        f"[规划节点] 决策: {action_data.get('action')}"
        + (f", 工具: {action_data.get('tool_name')}" if action_data.get("action") == "call_tool" else "")
    )

    return {
        "action_data": action_data,
        "step": step + 1,
        "trace": trace,
    }


def tool_node(state: AgentState):
    """
    执行工具节点：根据 planner 输出的 action_data 调用对应工具。

    优化点：
    - 统一的工具调用入口 call_tool()
    - 参数校验
    - 完整的错误处理和日志
    """
    action_data = state.get("action_data", {})
    tool_name = action_data.get("tool_name", "")
    args = action_data.get("args", {}) or {}
    trace = state.get("trace", [])

    logger.info(f"[工具节点] 调用工具: {tool_name}, 参数: {args}")

    if not tool_name:
        result = "[工具执行失败] 未指定工具名"
        trace.append({"node": "tools", "status": "error", "detail": "未指定工具名"})
        return {"tool_results": [result], "trace": trace}

    # 调用工具（统一错误处理）
    result = call_tool(tool_name, args)

    # 截断过长的工具结果，防止撑爆上下文
    if len(result) > config.tools.max_file_chars:
        result = truncate_text(result, config.tools.max_file_chars)

    trace.append({
        "node": "tools",
        "status": "ok" if "成功" in result else "error",
        "tool": tool_name,
        "detail": f"结果长度: {len(result)} 字符",
    })

    logger.debug(f"[工具节点] 执行完成，结果长度: {len(result)}")

    return {
        "tool_results": [result],
        "trace": trace,
    }


def reflection_node(state: AgentState):
    """
    反思节点：判断已有信息是否足够回答，或达到步数上限强制终止。

    优化点：
    - 步数上限强制终止（防死循环）
    - 优化提示词
    - 解析失败默认结束（保守策略，防死循环）
    """
    step = state.get("step", 0)
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    trace = state.get("trace", [])

    logger.info(f"[反思节点] 第 {step} 步 / 最大 {MAX_STEP} 步")

    # 达到最大步数，强制终止
    if step >= MAX_STEP:
        logger.warning(f"[反思节点] 达到最大步数 {MAX_STEP}，强制终止")
        trace.append({
            "node": "reflection",
            "status": "max_step",
            "detail": f"达到最大步数 {MAX_STEP}",
        })
        return {"need_continue": False, "trace": trace}

    # 如果没有工具结果，不需要反思
    if not tool_results:
        trace.append({"node": "reflection", "status": "skip", "detail": "无工具结果"})
        return {"need_continue": False, "trace": trace}

    # 检查最近一个工具结果是否失败，如果失败了不继续
    last_result = tool_results[-1] if tool_results else ""
    if "失败" in last_result and step > 1:
        logger.info("[反思节点] 工具执行失败，终止继续")
        trace.append({
            "node": "reflection",
            "status": "tool_failed",
            "detail": "工具失败，终止继续",
        })
        return {"need_continue": False, "trace": trace}

    prompt = _build_reflection_prompt(query, tool_results)

    try:
        resp_content = _safe_llm_call(
            [("user", prompt)],
            max_retries=1,
            temperature=0.1,
            max_tokens=50,  # 反思输出很短，限制 token 节省时间
        )
    except RuntimeError as e:
        logger.error(f"反思节点 LLM 调用失败: {e}")
        trace.append({"node": "reflection", "status": "error", "detail": str(e)})
        return {"need_continue": False, "trace": trace}

    # 解析 JSON
    data = try_parse_json(resp_content)
    if data is None:
        # 解析失败，默认结束（保守策略，防止死循环）
        logger.warning(f"反思节点 JSON 解析失败，默认结束: {resp_content[:50]}")
        need_continue = False
    else:
        need_continue = bool(data.get("need_continue", False))

    trace.append({
        "node": "reflection",
        "status": "ok",
        "need_continue": need_continue,
    })

    logger.info(f"[反思节点] 是否继续: {need_continue}")

    return {
        "need_continue": need_continue,
        "trace": trace,
    }


def final_node(state: AgentState):
    """
    收尾节点：整合工具结果，输出最终回答。

    优化点：
    - 如果规划器已直接回答，直接采用（省一次 LLM 调用）
    - 优化最终回答提示词
    - 工具结果整合更智能
    """
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    action_data = state.get("action_data", {})
    trace = state.get("trace", [])

    logger.info("[最终节点] 生成最终回答")

    # 规划器已直接回答的场景：直接用它的回答
    if action_data.get("action") == "direct_answer" and action_data.get("answer"):
        answer = action_data["answer"]
        trace.append({
            "node": "final",
            "status": "direct",
            "detail": "使用规划器直接回答",
        })
        logger.debug("[最终节点] 使用规划器直接回答")
        return {"final_answer": answer, "trace": trace}

    # 没有工具结果的异常情况
    if not tool_results:
        answer = "抱歉，我没有获取到足够的信息来回答你的问题。"
        trace.append({"node": "final", "status": "no_result", "detail": "无工具结果"})
        return {"final_answer": answer, "trace": trace}

    # 基于工具结果整合最终回答
    prompt = _build_final_prompt(query, tool_results)

    try:
        resp_content = _safe_llm_call(
            [("user", prompt)],
            max_retries=1,
            temperature=0.3,
        )
        final_answer = resp_content.strip()
    except RuntimeError as e:
        logger.error(f"最终节点 LLM 调用失败: {e}")
        # 降级：直接返回工具结果
        final_answer = "抱歉，生成回答时出现问题。以下是工具执行的原始结果：\n\n" + "\n".join(tool_results)
        trace.append({"node": "final", "status": "error", "detail": str(e)})
        return {"final_answer": final_answer, "trace": trace}

    trace.append({
        "node": "final",
        "status": "ok",
        "detail": f"回答长度: {len(final_answer)} 字符",
    })

    logger.debug(f"[最终节点] 回答生成完成，长度: {len(final_answer)}")

    return {
        "final_answer": final_answer,
        "trace": trace,
    }
