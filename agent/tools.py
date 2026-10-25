# agent/tools.py
# ============================================================
# 工具系统：装饰器注册 + Schema 描述 + 安全执行
#
# 新增工具只需：
#   1. 写函数 + 类型注解 + docstring
#   2. 加 @register_tool 装饰器
#   3. 自动纳入 tool_map 和工具描述列表
# ============================================================
import ast
import datetime
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

from config import config
from llm import get_llm
from utils.helpers import (
    format_tool_result,
    is_safe_path,
    get_safe_abs_path,
    truncate_text,
)
from utils.logger import get_logger

logger = get_logger("tools")

# ============================================================
# 工具注册表
# ============================================================

# 工具信息字典：name -> {func, description, parameters, category}
_tool_registry: Dict[str, dict] = {}


def register_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "general",
):
    """
    工具注册装饰器。

    Args:
        name: 工具名（默认取函数名）
        description: 工具描述（默认取函数 docstring 第一行）
        category: 工具分类，便于展示和管理
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__

        # 从 docstring 提取描述
        if description:
            desc = description
        elif func.__doc__:
            desc = func.__doc__.strip().split("\n")[0].strip()
        else:
            desc = tool_name

        # 从函数签名提取参数信息
        import inspect

        sig = inspect.signature(func)
        params = {}
        for pname, param in sig.parameters.items():
            param_info = {"type": "string", "required": True}
            if param.annotation != inspect.Parameter.empty:
                ann = param.annotation
                if ann == int:
                    param_info["type"] = "integer"
                elif ann == float:
                    param_info["type"] = "number"
                elif ann == bool:
                    param_info["type"] = "boolean"
                elif ann == str:
                    param_info["type"] = "string"

            if param.default != inspect.Parameter.empty:
                param_info["required"] = False
                param_info["default"] = param.default

            params[pname] = param_info

        _tool_registry[tool_name] = {
            "func": func,
            "description": desc,
            "parameters": params,
            "category": category,
        }

        logger.debug(f"工具已注册: {tool_name} ({category})")
        return func

    return decorator


def get_tool_map() -> Dict[str, Callable]:
    """获取 工具名 -> 函数 的映射表"""
    return {name: info["func"] for name, info in _tool_registry.items()}


def get_tool_descriptions() -> str:
    """
    获取所有工具的文本描述，用于构建系统提示词。
    格式为编号列表，便于模型理解。
    """
    lines = []
    for idx, (name, info) in enumerate(_tool_registry.items(), 1):
        params = info["parameters"]
        param_str = ", ".join(
            f"{pname}: {pinfo['type']}" for pname, pinfo in params.items()
        )
        lines.append(f"{idx}. {name}({param_str}) - {info['description']}")
    return "\n".join(lines)


def list_tools() -> List[dict]:
    """列出所有工具的详细信息（用于 UI 展示）"""
    return [
        {
            "name": name,
            "description": info["description"],
            "category": info["category"],
            "parameters": info["parameters"],
        }
        for name, info in _tool_registry.items()
    ]


def call_tool(tool_name: str, args: Optional[dict] = None) -> str:
    """
    安全调用工具，统一错误处理。

    Args:
        tool_name: 工具名称
        args: 参数字典

    Returns:
        格式化的工具执行结果字符串
    """
    args = args or {}

    if tool_name not in _tool_registry:
        msg = f"未知工具: {tool_name}"
        logger.warning(msg)
        return format_tool_result(tool_name, msg, success=False)

    try:
        logger.info(f"调用工具: {tool_name}, 参数: {args}")
        result = _tool_registry[tool_name]["func"](**args)
        # 确保返回字符串
        if not isinstance(result, str):
            result = str(result)
        logger.debug(f"工具 {tool_name} 执行完成，结果长度: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"工具 {tool_name} 执行异常: {e}", exc_info=True)
        return format_tool_result(tool_name, f"执行异常: {str(e)}", success=False)


# ============================================================
# 工具实现
# ============================================================

# ---------- 文件操作类 ----------

@register_tool(category="file")
def read_file(file_path: str, max_lines: int = None) -> str:
    """读取本地文本文件，支持大文件分片读取。

    Args:
        file_path: 相对项目根的文件路径
        max_lines: 最多读取行数（默认由配置决定）
    """
    if max_lines is None:
        max_lines = config.tools.default_max_lines

    try:
        full_path = get_safe_abs_path(file_path)

        # 安全检查
        if not is_safe_path(file_path):
            return format_tool_result(
                "read_file",
                f"路径不在安全目录内，仅允许访问: {config.tools.safe_dirs}",
                success=False,
            )

        if not os.path.isfile(full_path):
            return format_tool_result(
                "read_file", f"文件不存在: {file_path}", success=False
            )

        # 检查文件大小
        file_size = os.path.getsize(full_path)
        max_size = int(config.tools.max_file_size_mb * 1024 * 1024)
        if file_size > max_size:
            return format_tool_result(
                "read_file",
                f"文件过大（{file_size / 1024 / 1024:.1f}MB），超过上限 {config.tools.max_file_size_mb}MB",
                success=False,
            )

        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines(max_lines)

        content = "".join(lines)

        # 字符数截断
        if len(content) > config.tools.max_file_chars:
            content = truncate_text(content, config.tools.max_file_chars)
            suffix = f"\n...(内容已截断，共 {len(lines)} 行，{file_size} 字节)"
        else:
            suffix = f"（{len(lines)} 行，{len(content)} 字符）"

        return format_tool_result(
            "read_file", f"{suffix}\n{content}", success=True
        )

    except Exception as e:
        return format_tool_result("read_file", str(e), success=False)


@register_tool(category="file")
def write_file(file_path: str, content: str) -> str:
    """将内容写入本地文本文件，仅限安全目录。

    Args:
        file_path: 相对项目根的文件路径
        content: 要写入的内容
    """
    try:
        full_path = get_safe_abs_path(file_path)

        # 安全检查
        if not is_safe_path(file_path):
            return format_tool_result(
                "write_file",
                f"路径越界，仅允许写入: {config.tools.safe_dirs}",
                success=False,
            )

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        return format_tool_result(
            "write_file",
            f"已写入 {file_path}（{len(content)} 字符）",
            success=True,
        )

    except Exception as e:
        return format_tool_result("write_file", str(e), success=False)


@register_tool(category="file")
def list_files(dir_path: str = ".") -> str:
    """列出指定目录下的文件和子目录。

    Args:
        dir_path: 相对项目根的目录路径，默认为项目根目录
    """
    try:
        full_path = get_safe_abs_path(dir_path)

        # 安全检查
        if not is_safe_path(dir_path) and dir_path != ".":
            return format_tool_result(
                "list_files",
                f"路径不在安全目录内，仅允许访问: {config.tools.safe_dirs}",
                success=False,
            )

        if not os.path.isdir(full_path):
            return format_tool_result(
                "list_files", f"目录不存在: {dir_path}", success=False
            )

        entries = []
        for entry in sorted(os.listdir(full_path)):
            entry_path = os.path.join(full_path, entry)
            if os.path.isdir(entry_path):
                entries.append(f"[目录] {entry}/")
            else:
                size = os.path.getsize(entry_path)
                entries.append(f"[文件] {entry} ({size} 字节)")

        result = "\n".join(entries) if entries else "（空目录）"
        return format_tool_result(
            "list_files", f"目录 {dir_path} 内容：\n{result}", success=True
        )

    except Exception as e:
        return format_tool_result("list_files", str(e), success=False)


@register_tool(category="file")
def file_info(file_path: str) -> str:
    """获取文件的详细信息（大小、创建时间、修改时间等）。

    Args:
        file_path: 相对项目根的文件路径
    """
    try:
        full_path = get_safe_abs_path(file_path)

        if not is_safe_path(file_path):
            return format_tool_result(
                "file_info",
                f"路径不在安全目录内",
                success=False,
            )

        if not os.path.exists(full_path):
            return format_tool_result(
                "file_info", f"文件/目录不存在: {file_path}", success=False
            )

        stat = os.stat(full_path)
        is_dir = os.path.isdir(full_path)

        info_lines = [
            f"路径: {file_path}",
            f"类型: {'目录' if is_dir else '文件'}",
            f"大小: {stat.st_size} 字节 ({stat.st_size / 1024:.2f} KB)",
            f"创建时间: {datetime.datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}",
            f"修改时间: {datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        return format_tool_result(
            "file_info", "\n".join(info_lines), success=True
        )

    except Exception as e:
        return format_tool_result("file_info", str(e), success=False)


# ---------- 计算类 ----------

@register_tool(category="math")
def calculator(expr: str) -> str:
    """安全的数学表达式计算器，支持加减乘除取模幂运算。

    Args:
        expr: 数学表达式，如 "123 + 456 * 7"
    """
    try:
        # AST 白名单过滤，防止代码注入
        allowed_nodes = {
            ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
            ast.Pow, ast.FloorDiv, ast.USub, ast.UAdd,
        }

        # 预处理：去除空白
        expr_clean = expr.strip().replace(" ", "")

        tree = ast.parse(expr_clean, mode="eval")
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                return format_tool_result(
                    "calculator",
                    f"表达式包含非法运算符或函数: {type(node).__name__}",
                    success=False,
                )

        result = eval(compile(tree, filename="<ast>", mode="eval"))
        return format_tool_result("calculator", f"{expr} = {result}", success=True)

    except ZeroDivisionError:
        return format_tool_result("calculator", "除数不能为零", success=False)
    except SyntaxError as e:
        return format_tool_result("calculator", f"表达式语法错误: {e}", success=False)
    except Exception as e:
        return format_tool_result("calculator", str(e), success=False)


# ---------- 时间类 ----------

@register_tool(category="utility")
def get_current_time() -> str:
    """获取当前日期时间、星期和时区信息。"""
    now = datetime.datetime.now()
    week_days = ["一", "二", "三", "四", "五", "六", "日"]
    week_day = week_days[now.weekday()]

    # 尝试获取时区
    try:
        tz_name = datetime.datetime.now(datetime.timezone.utc).astimezone().tzname()
    except Exception:
        tz_name = "本地时间"

    result = (
        f"{now.strftime('%Y-%m-%d %H:%M:%S')} "
        f"星期{week_day} "
        f"({tz_name})"
    )
    return format_tool_result("get_current_time", result, success=True)


@register_tool(category="utility")
def date_calculate(date_str: str, days: int = 0) -> str:
    """日期计算：给定日期加减天数。

    Args:
        date_str: 日期字符串，格式如 '2024-01-15' 或 '2024/1/15'
        days: 要加减的天数（正数加，负数减）
    """
    try:
        # 尝试多种日期格式
        formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%m-%d-%Y", "%m/%d/%Y"]
        dt = None
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(date_str.strip(), fmt)
                break
            except ValueError:
                continue

        if dt is None:
            return format_tool_result(
                "date_calculate",
                f"无法解析日期: {date_str}，支持格式: YYYY-MM-DD, YYYY/MM/DD",
                success=False,
            )

        result_date = dt + datetime.timedelta(days=days)
        week_days = ["一", "二", "三", "四", "五", "六", "日"]
        week_day = week_days[result_date.weekday()]

        operation = "加" if days >= 0 else "减"
        result = (
            f"{date_str} {operation} {abs(days)} 天 = "
            f"{result_date.strftime('%Y-%m-%d')} 星期{week_day}"
        )
        return format_tool_result("date_calculate", result, success=True)

    except Exception as e:
        return format_tool_result("date_calculate", str(e), success=False)


# ---------- 文本处理类 ----------

@register_tool(category="text")
def text_summarize(text: str, max_length: int = 150) -> str:
    """对文本进行摘要提炼，输出简洁概括。

    Args:
        text: 需要摘要的文本内容
        max_length: 摘要最大字数（默认 150 字）
    """
    try:
        llm = get_llm()
        # 先截断输入，防止超长
        truncated = truncate_text(text, config.tools.max_file_chars)

        prompt = (
            f"请对以下文本进行精炼摘要，控制在 {max_length} 字以内，"
            f"保留核心信息，不要添加原文没有的内容。\n\n"
            f"待摘要文本：\n{truncated}\n\n"
            f"请直接输出摘要内容："
        )

        resp = llm.invoke([("user", prompt)], temperature=0.3)
        summary = resp.content.strip()

        return format_tool_result(
            "text_summarize",
            f"摘要（约{len(summary)}字）：\n{summary}",
            success=True,
        )

    except Exception as e:
        return format_tool_result("text_summarize", str(e), success=False)


@register_tool(category="text")
def text_translate(text: str, target_lang: str = "中文") -> str:
    """文本翻译，支持中英互译及其他语言。

    Args:
        text: 需要翻译的文本
        target_lang: 目标语言，如 '中文'、'英文'、'日文' 等
    """
    try:
        llm = get_llm()
        truncated = truncate_text(text, config.tools.max_file_chars)

        prompt = (
            f"请将以下文本翻译成{target_lang}，只输出翻译结果，不要添加解释。\n\n"
            f"原文：\n{truncated}\n\n"
            f"翻译结果："
        )

        resp = llm.invoke([("user", prompt)], temperature=0.3)
        translation = resp.content.strip()

        return format_tool_result(
            "text_translate",
            f"翻译结果（{target_lang}）：\n{translation}",
            success=True,
        )

    except Exception as e:
        return format_tool_result("text_translate", str(e), success=False)


@register_tool(category="text")
def json_format(json_str: str) -> str:
    """格式化 JSON 字符串，美化输出并校验语法。

    Args:
        json_str: 需要格式化的 JSON 字符串
    """
    try:
        data = json.loads(json_str)
        formatted = json.dumps(data, indent=2, ensure_ascii=False)
        return format_tool_result(
            "json_format",
            f"JSON 格式化成功：\n{formatted}",
            success=True,
        )
    except json.JSONDecodeError as e:
        return format_tool_result(
            "json_format",
            f"JSON 语法错误: {e}",
            success=False,
        )


@register_tool(category="text")
def word_count(text: str) -> str:
    """统计文本的字数、行数、字符数等信息。

    Args:
        text: 需要统计的文本
    """
    try:
        lines = text.split("\n")
        # 中文字数（不含空白）
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        # 英文单词数
        english_words = len(re.findall(r"[a-zA-Z]+", text))
        # 总字符数（不含空白）
        non_whitespace = len(re.sub(r"\s", "", text))

        result = (
            f"总行数: {len(lines)}\n"
            f"总字符数: {len(text)}\n"
            f"非空白字符: {non_whitespace}\n"
            f"中文字数: {chinese_chars}\n"
            f"英文单词数: {english_words}"
        )
        return format_tool_result("word_count", result, success=True)

    except Exception as e:
        return format_tool_result("word_count", str(e), success=False)


# ---------- 代码执行类 ----------

@register_tool(category="code")
def run_python(code: str) -> str:
    """执行 Python 代码并返回输出结果（受限沙箱）。

    安全限制：仅允许基础运算和 math/json/datetime 标准库，
    禁止文件系统写入、网络请求、系统调用等危险操作。

    Args:
        code: Python 代码字符串
    """
    try:
        import io
        import contextlib

        # 安全检查：禁止危险模块和操作
        dangerous_patterns = [
            r"import\s+os\b", r"import\s+sys\b", r"import\s+subprocess",
            r"import\s+shutil", r"import\s+socket", r"import\s+requests",
            r"import\s+urllib", r"import\s+http",
            r"\bopen\s*\(", r"\beval\s*\(", r"\bexec\s*\(",
            r"__import__", r"\bos\.", r"\bsys\.",
            r"__builtins__", r"globals\(\)", r"locals\(\)",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return format_tool_result(
                    "run_python",
                    "代码包含受限操作（文件/网络/系统调用），为安全起见已拒绝执行。仅允许 math、json、datetime 等安全模块。",
                    success=False,
                )

        # 捕获 stdout
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            # 使用受限的 globals
            safe_globals = {
                "__builtins__": {
                    "print": print,
                    "range": range,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "bool": bool,
                    "abs": abs,
                    "round": round,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "sorted": sorted,
                    "reversed": reversed,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "type": type,
                    "isinstance": isinstance,
                    "chr": chr,
                    "ord": ord,
                    "pow": pow,
                    "divmod": divmod,
                    "any": any,
                    "all": all,
                    "None": None,
                    "True": True,
                    "False": False,
                },
                "math": __import__("math"),
                "json": __import__("json"),
                "datetime": __import__("datetime"),
                "re": __import__("re"),
            }

            exec(code, safe_globals, {})

        result = output.getvalue()
        if not result.strip():
            result = "(代码执行完成，无输出)"

        return format_tool_result(
            "run_python",
            f"执行结果：\n{result}",
            success=True,
        )

    except Exception as e:
        return format_tool_result("run_python", f"执行错误: {str(e)}", success=False)


# ---------- 系统/信息类 ----------

@register_tool(category="utility")
def get_agent_info() -> str:
    """获取当前 Agent 的配置信息和可用工具列表。"""
    tools_info = list_tools()
    categories = {}
    for tool in tools_info:
        cat = tool["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(tool["name"])

    lines = [
        "=== Agent 信息 ===",
        f"模型: {config.llm.model}",
        f"提供商: {'Ollama 本地' if config.llm.is_ollama else '云端 API'}",
        f"最大步数: {config.agent.max_steps}",
        f"上下文限制: {config.agent.max_context_chars} 字符",
        "",
        "可用工具（按分类）:",
    ]

    cat_names = {
        "file": "文件操作",
        "math": "数学计算",
        "text": "文本处理",
        "code": "代码执行",
        "utility": "实用工具",
        "general": "通用",
    }

    for cat, tools in sorted(categories.items()):
        cat_label = cat_names.get(cat, cat)
        lines.append(f"  [{cat_label}]: {', '.join(tools)}")

    lines.append(f"\n共 {len(tools_info)} 个工具可用")

    return format_tool_result("get_agent_info", "\n".join(lines), success=True)


# ---------- 联网搜索 ----------

@register_tool(category="general")
def web_search(query: str, max_results: int = 5) -> str:
    """
    联网搜索（Bing 中国版，免费无需 key）。输入关键词，返回搜索结果标题、链接和摘要。
    适合查询实时新闻、百科、技术资料等网络信息。
    """
    if not query or not query.strip():
        return "[搜索失败] 关键词不能为空"

    try:
        import requests
    except ImportError:
        return "[搜索失败] 环境缺少 requests 库，请先安装 requests"

    url = "https://cn.bing.com/search"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    params = {"q": query, "count": str(max_results)}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            return f"[搜索失败] Bing 返回状态码 {resp.status_code}"
        html = resp.text
    except Exception as e:
        return f"[搜索失败] 网络错误: {e}"

    # 解析 Bing 结果块 <li class="b_algo">
    results = []
    blocks = re.findall(r'<li class="b_algo".*?</li>', html, re.S)
    for blk in blocks[:max_results]:
        title_m = re.search(
            r'<h2[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', blk, re.S
        )
        if not title_m:
            continue
        link = title_m.group(1)
        title = re.sub(r"<[^>]+>", "", title_m.group(2)).strip()

        snip_m = re.search(r"<p[^>]*>(.*?)</p>", blk, re.S)
        snippet = re.sub(r"<[^>]+>", "", snip_m.group(1)).strip() if snip_m else ""

        results.append(f"• {title}\n  链接: {link}\n  摘要: {snippet}")

    if not results:
        return f"[搜索完成] 未找到与“{query}”相关的结果"

    head = f"“{query}” 的搜索结果（{len(results)} 条）:\n"
    return head + "\n\n".join(results)


# ============================================================
# 向后兼容：保留 tool_map 变量名
# ============================================================
tool_map = get_tool_map()
