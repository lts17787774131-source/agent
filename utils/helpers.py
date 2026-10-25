# utils/helpers.py
# ============================================================
# 通用工具函数
# ============================================================
import json
import os
import re
from typing import Any, Optional

from config import config


def clean_json_response(raw: str) -> str:
    """
    清理大模型输出的 JSON 字符串：
    - 去除 markdown 代码块包裹（```json ... ```）
    - 去除首尾空白
    - 处理可能的前置解释文字（取第一个 { 到最后一个 } 之间的内容）
    """
    s = raw.strip()

    # 去除 markdown 代码块
    if s.startswith("```"):
        # 去掉开头的 ```
        s = s[3:]
        # 去掉语言标记（如 json）
        if "\n" in s:
            first_line_end = s.index("\n")
            first_line = s[:first_line_end].strip().lower()
            if first_line in ("json", "python", "text", "markdown", "md"):
                s = s[first_line_end + 1:]
        # 去掉结尾的 ```
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

    # 如果仍然不是纯 JSON，尝试提取花括号内容
    if not s.startswith("{"):
        brace_start = s.find("{")
        brace_end = s.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            s = s[brace_start:brace_end + 1]

    return s.strip()


def try_parse_json(raw: str) -> Optional[dict]:
    """尝试解析 JSON，失败返回 None"""
    # 第一次尝试：标准清洗后直接解析
    cleaned = clean_json_response(raw)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # 第二次尝试：容错修复小模型常见的 JSON 畸形
    # （键缺右引号 "key:{、键未加引号 {key:、多余尾逗号 等）
    repaired = repair_json(cleaned)
    if repaired != cleaned:
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def repair_json(s: str) -> str:
    """
    容错修复大模型输出的常见畸形 JSON：
    - 字符串键缺右引号："args:{ -> "args":{
    - 键未加引号：{action: -> {"action":
    - 多余尾逗号：{,} / {,] 等
    仅做保守修复，无法修复时原样返回。
    """
    s = s.strip()

    # 1. 键缺右引号："key:{ / "key":[ / "key":" / "key":123 -> "key":...
    s = re.sub(r'"([A-Za-z_][\w]*)\s*:(?=\s*[{["\d\-])', r'"\1":', s)

    # 2. 键未加引号：{key: -> {"key":
    s = re.sub(r'([{,]\s*)([A-Za-z_][\w]*)\s*:', r'\1"\2":', s)

    # 3. 去除多余尾逗号
    s = re.sub(r',\s*([}\]])', r'\1', s)

    return s.strip()


def truncate_text(text: str, max_chars: int = None, suffix: str = "...") -> str:
    """
    智能截断文本，保留前后信息。
    默认使用配置中的 max_context_chars。
    """
    if max_chars is None:
        max_chars = config.agent.max_context_chars

    if len(text) <= max_chars:
        return text

    # 保留前 60% 和后 40%，中间用省略号代替
    head_len = int(max_chars * 0.6)
    tail_len = max_chars - head_len - len(suffix)

    if tail_len <= 0:
        return text[:max_chars - len(suffix)] + suffix

    return text[:head_len] + suffix + text[-tail_len:]


def safe_filename(name: str) -> str:
    """清理文件名中的非法字符"""
    # Windows 非法字符: \ / : * ? " < > |
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    # 去除首尾空格和点
    name = name.strip(" .")
    # 限制长度
    if len(name) > 200:
        name = name[:200]
    return name or "unnamed"


def format_tool_result(tool_name: str, result: str, success: bool = True) -> str:
    """格式化工具执行结果，便于 Agent 识别"""
    status = "成功" if success else "失败"
    return f"[{tool_name} {status}] {result}"


def is_safe_path(file_path: str) -> bool:
    """
    检查文件路径是否在安全目录内。
    用于 write_file / read_file 等文件操作工具的安全校验。
    """
    full_path = os.path.abspath(os.path.join(config.project_root, file_path))
    for safe_dir in config.tools.safe_dirs:
        safe_abs = os.path.abspath(os.path.join(config.project_root, safe_dir))
        # 路径等于安全目录或在安全目录下
        if full_path == safe_abs or full_path.startswith(safe_abs + os.sep):
            return True
    return False


def get_safe_abs_path(file_path: str) -> str:
    """获取安全的绝对路径（前提是路径在安全目录内）"""
    return os.path.abspath(os.path.join(config.project_root, file_path))
