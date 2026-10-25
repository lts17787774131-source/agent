# app.py
# ============================================================
# Gradio 网页交互界面（企业级优化版）
# 功能：
#   - 流式对话体验
#   - Agent 思考过程实时展示
#   - 文件上传与处理
#   - 工具列表展示
#   - 对话历史管理
#   - 系统状态监控
# ============================================================
import datetime
import json
import os
import shutil
import time
from typing import List, Tuple

import gradio as gr

from config import config, apply_llm_preset, current_preset_name
from agent.graph import stream_agent
from agent.tools import list_tools
from llm.factory import reset_llm
from llm_presets import LLM_PRESETS
from utils.helpers import safe_filename
from utils.logger import get_logger, setup_logging

# 初始化日志
setup_logging()
logger = get_logger("app")

# 对话日志目录
_LOG_DIR = config.get_abs_path(config.logging.log_dir)
_UPLOAD_DIR = config.get_abs_path("uploads")

# 确保目录存在
os.makedirs(_LOG_DIR, exist_ok=True)
os.makedirs(_UPLOAD_DIR, exist_ok=True)


# ============================================================
# 辅助函数
# ============================================================

def _parse_history(history) -> List[Tuple[str, str]]:
    """
    把 Gradio 历史消息统一转成 (role, content) 列表。
    兼容 Gradio 5/6 的消息格式。
    """
    messages = []
    for h in (history or []):
        if isinstance(h, dict):
            role = h.get("role", "user")
            content = h.get("content", "")
            if role not in ("user", "assistant"):
                role = "user"
            messages.append((role, content))
        elif isinstance(h, (tuple, list)) and len(h) >= 2:
            role = str(h[0])
            if role not in ("user", "assistant"):
                role = "user"
            messages.append((role, str(h[1])))
    return messages


def _save_chat_log(query: str, trace: list, final_answer: str):
    """保存对话日志到 jsonl 文件"""
    if not config.logging.enable_chat_history:
        return
    try:
        fname = datetime.datetime.now().strftime("chat_%Y%m%d.jsonl")
        entry = {
            "time": datetime.datetime.now().isoformat(),
            "query": query,
            "trace": trace,
            "final_answer": final_answer,
        }
        with open(os.path.join(_LOG_DIR, fname), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"保存对话日志失败: {e}")


def _format_trace(trace: list) -> str:
    """将执行轨迹格式化为可读文本"""
    if not trace:
        return "（暂无执行记录）"

    node_labels = {
        "planner": "🧠 规划",
        "tools": "🛠 工具",
        "reflection": "💭 反思",
        "final": "✅ 回答",
    }

    lines = []
    for i, item in enumerate(trace, 1):
        node = item.get("node", "unknown")
        status = item.get("status", "")
        detail = item.get("detail", "")

        label = node_labels.get(node, node)
        status_icon = {
            "ok": "✓",
            "error": "✗",
            "max_step": "⚠",
            "skip": "→",
            "direct": "→",
            "tool_failed": "✗",
        }.get(status, "")

        lines.append(f"[{i}] {label} {status_icon}")
        if detail:
            # 限制详情长度
            if len(detail) > 200:
                detail = detail[:200] + "..."
            lines.append(f"    {detail}")

    return "\n".join(lines)


def _process_uploaded_file(files) -> str:
    """
    处理上传的文件，保存到 uploads 目录，返回文件信息说明。
    """
    if not files:
        return ""

    saved_files = []
    for file_info in files:
        try:
            # file_info 可能是路径字符串或 Gradio 文件对象
            if isinstance(file_info, str):
                src_path = file_info
                orig_name = os.path.basename(file_info)
            elif hasattr(file_info, "name"):
                src_path = file_info.name
                orig_name = os.path.basename(file_info.name)
            elif isinstance(file_info, dict):
                src_path = file_info.get("path", "")
                orig_name = file_info.get("name", "uploaded_file")
            else:
                continue

            if not src_path or not os.path.exists(src_path):
                continue

            # 生成安全文件名
            base, ext = os.path.splitext(orig_name)
            safe_name = safe_filename(base) + ext
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_name = f"{timestamp}_{safe_name}"
            dest_path = os.path.join(_UPLOAD_DIR, dest_name)

            shutil.copy2(src_path, dest_path)
            saved_files.append(f"uploads/{dest_name}")

            logger.info(f"文件已上传: {dest_name}")

        except Exception as e:
            logger.error(f"处理上传文件失败: {e}")

    if saved_files:
        file_list = "\n".join(f"  - {f}" for f in saved_files)
        return (
            f"📎 已上传 {len(saved_files)} 个文件到 uploads/ 目录：\n"
            f"{file_list}\n\n"
            f"你可以对这些文件进行操作，例如：\n"
            f"• 读取文件内容：`读取 uploads/{os.path.basename(saved_files[0])} 并摘要`\n"
            f"• 统计字数：`统计 uploads/{os.path.basename(saved_files[0])} 的字数`"
        )
    return ""


# ============================================================
# 核心对话函数
# ============================================================

def chat_with_agent(query: str, history: list, upload_files=None):
    """
    处理用户输入，流式返回 Agent 执行过程和最终回答。

    使用生成器模式，逐个 yield 更新 UI，实现实时效果。
    """
    # 处理上传文件
    upload_info = ""
    if upload_files:
        upload_info = _process_uploaded_file(upload_files)
        if upload_info:
            # 先返回一条系统消息提示文件上传
            history = (history or []) + [
                {"role": "assistant", "content": upload_info},
            ]
            yield history, "📎 文件上传处理中...", "", upload_files

    if not query or not query.strip():
        return history, "", "", upload_files

    # 解析历史消息
    messages = _parse_history(history)

    # 初始化返回值
    trace_lines = []
    final_answer = ""
    full_trace = []

    # 添加用户消息到历史
    history = (history or []) + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "🤔 思考中..."},
    ]
    yield history, "🧠 Agent 正在规划...", "", upload_files

    try:
        # 流式执行 Agent
        for event in stream_agent(query, messages):
            for node_name, value in event.items():
                if node_name == "planner":
                    action_data = value.get("action_data", {})
                    action = action_data.get("action", "unknown")
                    if action == "call_tool":
                        tool = action_data.get("tool_name", "?")
                        args = action_data.get("args", {})
                        trace_lines.append(f"🧠 [规划] 决定调用工具: {tool}")
                        trace_lines.append(f"   参数: {json.dumps(args, ensure_ascii=False)}")
                    else:
                        trace_lines.append("🧠 [规划] 决定直接回答")
                    trace_text = "\n".join(trace_lines)
                    history[-1]["content"] = "🛠 正在执行工具..." if action == "call_tool" else "✍️ 生成回答中..."
                    yield history, trace_text, "", upload_files

                elif node_name == "tools":
                    results = value.get("tool_results", [])
                    for r in results:
                        # 只显示前 200 字符的结果摘要
                        preview = r[:200] + "..." if len(r) > 200 else r
                        trace_lines.append(f"🛠 [工具] {preview}")
                    trace_text = "\n".join(trace_lines)
                    history[-1]["content"] = "💭 反思中..."
                    yield history, trace_text, "", upload_files

                elif node_name == "reflection":
                    need_continue = value.get("need_continue", False)
                    if need_continue:
                        trace_lines.append("💭 [反思] 信息不足，继续规划...")
                        history[-1]["content"] = "🧠 继续规划中..."
                    else:
                        trace_lines.append("💭 [反思] 信息足够，准备输出回答")
                        history[-1]["content"] = "✍️ 生成最终回答..."
                    trace_text = "\n".join(trace_lines)
                    yield history, trace_text, "", upload_files

                elif node_name == "final":
                    final_answer = value.get("final_answer", "")
                    full_trace = value.get("trace", [])
                    trace_lines.append("✅ [完成] 回答已生成")

                # 收集 trace
                if "trace" in value:
                    full_trace = value["trace"]

        # 更新最终回答
        if final_answer:
            history[-1]["content"] = final_answer
        else:
            history[-1]["content"] = "（未能生成回答）"

        trace_text = "\n".join(trace_lines)

        # 保存对话日志
        _save_chat_log(query, full_trace, final_answer)

        yield history, trace_text, "", upload_files

    except Exception as e:
        logger.error(f"对话处理异常: {e}", exc_info=True)
        error_msg = f"❌ 处理出错: {str(e)}\n\n请检查 Ollama 服务是否正常运行，或查看日志获取详细信息。"
        history[-1]["content"] = error_msg
        trace_lines.append(f"❌ [错误] {e}")
        yield history, "\n".join(trace_lines), "", upload_files


# ============================================================
# 界面构建
# ============================================================

# 全局样式
_CSS = """
:root {
    --brand-primary: #6366f1;
    --brand-secondary: #8b5cf6;
    --brand-accent: #a855f7;
    --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* 页头样式 */
#app-header {
    background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    color: #fff;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25);
}

#app-header h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 0.3px;
}

#app-header p {
    margin: 6px 0 0;
    font-size: 14px;
    opacity: 0.9;
}

.header-badges {
    margin-top: 12px;
}

.badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    backdrop-filter: blur(10px);
    border-radius: 999px;
    padding: 4px 12px;
    margin: 4px 6px 0 0;
    font-size: 12px;
    font-weight: 500;
}

/* 状态指示器 */
.status-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    font-size: 13px;
    color: #166534;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* 工具卡片 */
.tool-card {
    padding: 12px;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    margin-bottom: 8px;
    transition: all 0.2s;
    background: #fff;
}

.tool-card:hover {
    border-color: #6366f1;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.1);
}

.tool-name {
    font-weight: 600;
    color: #1f2937;
    font-size: 14px;
}

.tool-desc {
    font-size: 12px;
    color: #6b7280;
    margin-top: 4px;
}

.tool-cat {
    display: inline-block;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    background: #eef2ff;
    color: #4f46e5;
    margin-top: 6px;
}

/* 隐藏 footer */
footer { display: none !important; }

/* 思考日志样式 */
.trace-log {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 12px;
    line-height: 1.6;
}
"""

# 全局主题（Gradio 6 中 theme/css 需在 launch() 传入）
_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)


def _get_tool_cards_html() -> str:
    """生成工具列表的 HTML 卡片"""
    tools = list_tools()
    cat_colors = {
        "file": ("#eff6ff", "#1d4ed8"),
        "math": ("#fef3c7", "#b45309"),
        "text": ("#f0fdf4", "#15803d"),
        "code": ("#faf5ff", "#7e22ce"),
        "utility": ("#f3f4f6", "#4b5563"),
        "general": ("#f3f4f6", "#4b5563"),
    }

    cards = []
    for tool in tools:
        cat = tool["category"]
        bg, color = cat_colors.get(cat, ("#f3f4f6", "#4b5563"))
        params = ", ".join(
            f"{p}" for p in tool["parameters"].keys()
        ) or "无参数"

        cards.append(f"""
        <div class="tool-card">
            <div class="tool-name">🔧 {tool['name']}({params})</div>
            <div class="tool-desc">{tool['description']}</div>
            <span class="tool-cat" style="background:{bg};color:{color};">{cat}</span>
        </div>
        """)

    return "\n".join(cards)


def _get_status_html() -> str:
    """生成系统状态的 HTML"""
    provider = "Ollama 本地" if config.llm.is_ollama else "云端 API"
    model = config.llm.model
    tools_count = len(list_tools())

    return f"""
    <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 12px;">
        <div class="status-indicator">
            <div class="status-dot"></div>
            <span>模型: {model}</span>
        </div>
        <div class="status-indicator" style="background: #eff6ff; border-color: #bfdbfe; color: #1e40af;">
            <span>📡 {provider}</span>
        </div>
        <div class="status-indicator" style="background: #faf5ff; border-color: #e9d5ff; color: #7e22ce;">
            <span>🔧 {tools_count} 个工具</span>
        </div>
        <div class="status-indicator" style="background: #fef3c7; border-color: #fde68a; color: #b45309;">
            <span>⚡ 最大 {config.agent.max_steps} 步</span>
        </div>
    </div>
    """


def build_ui():
    """构建 Gradio 界面"""
    with gr.Blocks(
        title="多工具调用智能助手 Agent",
    ) as demo:

        # ===== 页头 =====
        gr.HTML(
            """<div id="app-header">
                <h1>🤖 多工具调用智能助手</h1>
                <p>基于 LangGraph 的 Agent · 任务规划 · 工具调用 · 结果反思 · 多轮对话</p>
                <div class="header-badges">
                    <span class="badge">📁 文件操作</span>
                    <span class="badge">🔢 数学计算</span>
                    <span class="badge">📝 文本处理</span>
                    <span class="badge">💻 代码执行</span>
                    <span class="badge">⏰ 时间日期</span>
                    <span class="badge">🔄 多步推理</span>
                </div>
            </div>"""
        )

        # ===== 系统状态 =====
        status_html = gr.HTML(_get_status_html())

        # ===== 模型/API 一键切换 =====
        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=list(LLM_PRESETS.keys()),
                value=current_preset_name() or list(LLM_PRESETS.keys())[0],
                label="🔀 模型 / API（下拉选择，即时生效）",
                interactive=True,
                scale=4,
            )
            switch_status = gr.HTML(scale=1)

        with gr.Row():
            # ===== 左列：对话区 =====
            with gr.Column(scale=3):
                # 聊天机器人
                chatbot = gr.Chatbot(
                    label="对话",
                    height=520,
                    show_label=False,
                    avatar_images=(None, None),
                )

                # 文件上传
                with gr.Row():
                    upload_btn = gr.UploadButton(
                        "📎 上传文件",
                        file_count="multiple",
                        file_types=["text", ".txt", ".md", ".json", ".csv", ".py"],
                        size="sm",
                        variant="secondary",
                    )
                    clear_btn = gr.Button("🗑 清空对话", size="sm", variant="secondary")

                # 输入区
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="输入你的问题，例如：计算 123 + 456 * 7，或上传文件后询问内容...",
                        show_label=False,
                        scale=5,
                        container=False,
                        lines=1,
                    )
                    send_btn = gr.Button(
                        "发送",
                        scale=1,
                        variant="primary",
                    )

                # 示例引导
                gr.Examples(
                    examples=[
                        "计算 123 + 456 * 7 等于多少",
                        "读取 data/test.txt 文件的内容",
                        "现在几点了？今天星期几",
                        "帮我写一个 Python 代码计算斐波那契数列",
                        "列出 output 目录下的所有文件",
                        "把 'Hello World' 翻译成中文",
                    ],
                    inputs=msg,
                    label="💡 快速开始",
                )

            # ===== 右列：信息面板 =====
            with gr.Column(scale=2):
                # Agent 思考日志
                with gr.Accordion("🧠 Agent 思考过程", open=True):
                    trace_box = gr.Textbox(
                        label="",
                        lines=16,
                        max_lines=20,
                        show_label=False,
                        placeholder="Agent 的实时思考过程会显示在这里...",
                        elem_classes=["trace-log"],
                    )

                # 工具列表
                with gr.Accordion("🔧 可用工具", open=False):
                    gr.HTML(_get_tool_cards_html())

                # 使用说明
                with gr.Accordion("📖 使用说明", open=False):
                    gr.Markdown(
                        """
### 🚀 快速上手

1. **直接提问** - 在输入框输入问题，按回车或点击发送
2. **上传文件** - 点击「上传文件」按钮，支持 txt/md/json/csv/py 等文本文件
3. **查看过程** - 右侧面板实时展示 Agent 的思考过程

### 🔧 工具说明

- **文件操作**: 读取、写入、列出文件、查看文件信息
- **数学计算**: 安全的表达式计算（加减乘除幂模）
- **文本处理**: 摘要、翻译、JSON 格式化、字数统计
- **代码执行**: Python 代码沙箱执行（受限环境）
- **实用工具**: 时间查询、日期计算、Agent 信息查询

### ⚠️ 注意事项

- 本地模型（Ollama）响应速度取决于电脑配置
- 文件操作仅限 `data/`、`output/`、`uploads/` 目录
- Agent 最多执行 {max_steps} 步，防止无限循环
- 对话历史会自动保存到 `logs/` 目录
                        """.format(max_steps=config.agent.max_steps)
                    )

        # ===== 交互逻辑 =====

        # 模型/API 切换
        def on_preset_change(name):
            """切换模型预设：更新 config.llm + 重置 LLM 单例 + 刷新状态栏"""
            if not name:
                return gr.update(), gr.update()
            ok = apply_llm_preset(name)
            if ok:
                reset_llm()
                logger.info(f"已切换模型预设: {name} -> 模型 {config.llm.model}")
                return (
                    gr.update(value=_get_status_html()),
                    gr.update(
                        value=(
                            f"<div style='color:#16a34a;margin:6px 0;font-size:13px;'>"
                            f"✅ 已切换到：{name}（模型：{config.llm.model}）</div>"
                        )
                    ),
                )
            return (
                gr.update(),
                gr.update(
                    value=(
                        f"<div style='color:#dc2626;margin:6px 0;font-size:13px;'>"
                        f"❌ 未找到预设：{name}</div>"
                    )
                ),
            )

        preset_dropdown.change(
            on_preset_change,
            inputs=preset_dropdown,
            outputs=[status_html, switch_status],
        )

        # 发送消息
        def respond(user_input, chat_history, upload_files):
            if not user_input or not user_input.strip():
                yield chat_history, "", "", upload_files
                return

            yield from chat_with_agent(user_input, chat_history, upload_files)

        # 文本框提交
        msg.submit(
            respond,
            [msg, chatbot, upload_btn],
            [chatbot, trace_box, msg, upload_btn],
        )

        # 发送按钮点击
        send_btn.click(
            respond,
            [msg, chatbot, upload_btn],
            [chatbot, trace_box, msg, upload_btn],
        )

        # 清空对话
        def clear_chat():
            return [], "", ""

        clear_btn.click(
            clear_chat,
            outputs=[chatbot, trace_box, msg],
        )

        # 上传文件后自动提示
        def on_upload(files, history):
            if not files:
                return history, ""
            info = _process_uploaded_file(files)
            if info:
                history = (history or []) + [
                    {"role": "assistant", "content": info},
                ]
            return history, info

        upload_btn.upload(
            on_upload,
            [upload_btn, chatbot],
            [chatbot, trace_box],
        )

    return demo


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("启动多工具调用智能助手 Agent")
    logger.info(f"模型: {config.llm.model}")
    logger.info(f"提供商: {'Ollama 本地' if config.llm.is_ollama else '云端 API'}")
    logger.info(f"工具数量: {len(list_tools())}")
    logger.info("=" * 50)

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=_THEME,
        css=_CSS,
    )
