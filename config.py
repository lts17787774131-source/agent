# config.py
# ============================================================
# 企业级配置管理：类型安全、环境校验、默认值合理
# ============================================================
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件
load_dotenv()


def _get_bool(name: str, default: bool = False) -> bool:
    """安全读取布尔型环境变量"""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _get_int(name: str, default: int) -> int:
    """安全读取整型环境变量"""
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def _get_float(name: str, default: float) -> float:
    """安全读取浮点型环境变量"""
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


@dataclass
class LLMConfig:
    """大模型连接配置"""
    base_url: str = "http://127.0.0.1:11434/v1"
    api_key: str = "dummy"
    model: str = "qwen3:4b"
    temperature: float = 0.1
    max_tokens: int = 512
    timeout: int = 120          # 请求超时（秒）
    max_retries: int = 2        # 自动重试次数
    streaming: bool = True      # 是否启用流式输出

    # 针对 Ollama 本地模型的特殊参数
    ollama_think: bool = False  # 是否开启 qwen3 思考模式

    @property
    def is_ollama(self) -> bool:
        """判断是否为本地 Ollama 服务"""
        return "11434" in self.base_url or "ollama" in self.base_url.lower()


@dataclass
class AgentConfig:
    """Agent 运行时配置"""
    max_steps: int = 6              # 最大执行步数（防死循环）
    max_context_chars: int = 3000   # 上下文窗口字符上限（4K token 约等于 3000 中文字符）
    reflection_enabled: bool = True # 是否启用反思节点
    enable_tool_history: bool = True  # 是否在消息中保留工具调用历史


@dataclass
class ToolConfig:
    """工具安全配置"""
    safe_dirs: list = field(default_factory=lambda: ["data", "output", "uploads"])
    max_file_chars: int = 3000      # 单文件读取字符上限（适配 4K 上下文）
    default_max_lines: int = 200    # 单文件读取行数上限
    max_file_size_mb: float = 5.0   # 单文件大小上限（MB）


@dataclass
class LogConfig:
    """日志配置"""
    log_dir: str = "logs"
    enable_chat_history: bool = True
    log_level: str = "INFO"


@dataclass
class AppConfig:
    """应用总配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    logging: LogConfig = field(default_factory=LogConfig)

    # 项目根目录（本文件位于项目根）
    project_root: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))

    def get_abs_path(self, rel_path: str) -> str:
        """获取相对项目根的绝对路径"""
        return os.path.abspath(os.path.join(self.project_root, rel_path))


def load_config() -> AppConfig:
    """从 .env 加载配置，返回 AppConfig 实例"""
    llm = LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "dummy"),
        model=os.getenv("LLM_MODEL", "qwen3:4b"),
        temperature=_get_float("LLM_TEMPERATURE", 0.1),
        max_tokens=_get_int("LLM_MAX_TOKENS", 512),
        timeout=_get_int("LLM_TIMEOUT", 120),
        max_retries=_get_int("LLM_MAX_RETRIES", 2),
        streaming=_get_bool("LLM_STREAMING", True),
        ollama_think=_get_bool("OLLAMA_THINK", False),
    )

    agent = AgentConfig(
        max_steps=_get_int("AGENT_MAX_STEPS", 6),
        max_context_chars=_get_int("AGENT_MAX_CONTEXT_CHARS", 3000),
        reflection_enabled=_get_bool("AGENT_REFLECTION_ENABLED", True),
    )

    tools = ToolConfig(
        safe_dirs=[d.strip() for d in os.getenv("TOOL_SAFE_DIRS", "data,output,uploads").split(",") if d.strip()],
        max_file_chars=_get_int("TOOL_MAX_FILE_CHARS", 3000),
        default_max_lines=_get_int("TOOL_DEFAULT_MAX_LINES", 200),
    )

    logging_cfg = LogConfig(
        log_dir=os.getenv("LOG_DIR", "logs"),
        enable_chat_history=_get_bool("LOG_CHAT_HISTORY", True),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

    return AppConfig(
        llm=llm,
        agent=agent,
        tools=tools,
        logging=logging_cfg,
    )


# 全局单例配置
config = load_config()


def apply_llm_preset(preset_name: str) -> bool:
    """
    运行时切换模型/API 预设（不改 .env、无需重启）。

    从 llm_presets.LLM_PRESETS 取对应方案，覆盖 config.llm 的字段。
    之后需调用 llm.factory.reset_llm() 清掉单例缓存，下一次请求生效。

    Args:
        preset_name: 预设显示名（对应 llm_presets.LLM_PRESETS 的 key）

    Returns:
        bool: 是否成功找到并应用该预设
    """
    from llm_presets import LLM_PRESETS

    params = LLM_PRESETS.get(preset_name)
    if not params:
        return False

    llm = config.llm
    # api_key 留空 "" 时沿用当前(.env 加载的)值
    if params.get("api_key") is not None and params.get("api_key") != "":
        llm.api_key = params["api_key"]

    mapping = {
        "base_url": "base_url",
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "timeout": "timeout",
        "max_retries": "max_retries",
        "streaming": "streaming",
        "ollama_think": "ollama_think",
    }
    for preset_key, attr in mapping.items():
        if preset_key in params and params[preset_key] is not None:
            setattr(llm, attr, params[preset_key])

    return True


def current_preset_name() -> str:
    """返回当前 config.llm 对应的预设名（按 base_url 匹配），匹配不到返回 None"""
    from llm_presets import LLM_PRESETS

    cur_url = config.llm.base_url.strip().rstrip("/")
    for name, params in LLM_PRESETS.items():
        preset_url = str(params.get("base_url", "")).strip().rstrip("/")
        if preset_url and preset_url == cur_url:
            return name
    return None
