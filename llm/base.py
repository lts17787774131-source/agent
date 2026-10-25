# llm/base.py
# ============================================================
# LLM 抽象基类：统一所有大模型提供商的调用接口
# 后续接入新模型（如通义千问、文心一言等）只需实现此类
# ============================================================
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple


@dataclass
class LLMResponse:
    """LLM 响应统一封装"""
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)  # prompt_tokens, completion_tokens, total_tokens
    raw: Any = None  # 原始响应对象，便于调试


class BaseLLM(ABC):
    """大模型调用抽象基类"""

    @abstractmethod
    def invoke(
        self,
        messages: List[Tuple[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        同步调用大模型，返回完整响应。

        Args:
            messages: 消息列表，格式为 [(role, content), ...]
                     role 可选: "system", "user", "assistant"
            temperature: 采样温度，None 则使用配置默认值
            max_tokens: 最大输出 token 数，None 则使用配置默认值

        Returns:
            LLMResponse 封装对象
        """
        pass

    @abstractmethod
    def stream(
        self,
        messages: List[Tuple[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        流式调用大模型，逐块产出文本。

        Yields:
            文本块（字符串）
        """
        pass

    @abstractmethod
    async def ainvoke(
        self,
        messages: List[Tuple[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """异步版本的 invoke"""
        pass

    @abstractmethod
    async def astream(
        self,
        messages: List[Tuple[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """异步版本的 stream"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """当前使用的模型名称"""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """提供商名称（如 ollama, deepseek, openai 等）"""
        pass
