# llm/providers.py
# ============================================================
# LLM 提供商实现
#   - OpenAICompatibleLLM: 兼容 OpenAI API 的所有服务
#     包括：Ollama 本地、DeepSeek、OpenAI 官方、智谱、月之暗面等
# ============================================================
import time
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm.base import BaseLLM, LLMResponse
from config import LLMConfig


class OpenAICompatibleLLM(BaseLLM):
    """
    OpenAI 兼容接口的 LLM 实现。

    适用范围：
    - Ollama 本地模型（http://localhost:11434/v1）
    - DeepSeek 云端 API
    - OpenAI 官方 API
    - 任何兼容 OpenAI /v1/chat/completions 接口的服务
    """

    def __init__(self, cfg: LLMConfig):
        self._cfg = cfg
        self._extra_body = {}

        # Ollama 特殊处理：关闭思考模式以加快响应。
        # 注意：必须用 extra_body 传给 Ollama。因为 openai 3.x 的
        # Completions.create() 不接受任意顶层关键字，若放进 model_kwargs
        # 会被直接透传成 create(think=...) 而报 TypeError。
        if cfg.is_ollama and not cfg.ollama_think:
            self._extra_body["think"] = False

        # 构造 LangChain ChatOpenAI 实例
        self._llm = ChatOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            extra_body=self._extra_body if self._extra_body else None,
        )

    @staticmethod
    def _convert_messages(messages: List[Tuple[str, str]]) -> list:
        """将 (role, content) 元组列表转换为 LangChain 消息对象列表"""
        converted = []
        for role, content in messages:
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "user":
                converted.append(HumanMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                # 默认按 user 处理
                converted.append(HumanMessage(content=content))
        return converted

    def _build_llm(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatOpenAI:
        """根据临时参数构造新的 LLM 实例（用于单次调用覆盖配置）"""
        if temperature is None and max_tokens is None:
            return self._llm

        kwargs = dict(
            base_url=self._cfg.base_url,
            api_key=self._cfg.api_key,
            model=self._cfg.model,
            temperature=temperature if temperature is not None else self._cfg.temperature,
            max_tokens=max_tokens if max_tokens is not None else self._cfg.max_tokens,
            timeout=self._cfg.timeout,
            max_retries=self._cfg.max_retries,
            extra_body=self._extra_body if self._extra_body else None,
        )
        return ChatOpenAI(**kwargs)

    def invoke(
        self,
        messages: List[Tuple[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        llm = self._build_llm(temperature, max_tokens)
        converted = self._convert_messages(messages)

        resp = llm.invoke(converted)
        usage = {}
        if resp.usage_metadata:
            usage = {
                "prompt_tokens": resp.usage_metadata.get("input_tokens", 0),
                "completion_tokens": resp.usage_metadata.get("output_tokens", 0),
                "total_tokens": resp.usage_metadata.get("total_tokens", 0),
            }

        return LLMResponse(
            content=resp.content,
            model=self._cfg.model,
            usage=usage,
            raw=resp,
        )

    def stream(
        self,
        messages: List[Tuple[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Iterator[str]:
        llm = self._build_llm(temperature, max_tokens)
        converted = self._convert_messages(messages)

        for chunk in llm.stream(converted):
            if chunk.content:
                yield chunk.content

    async def ainvoke(
        self,
        messages: List[Tuple[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        llm = self._build_llm(temperature, max_tokens)
        converted = self._convert_messages(messages)

        resp = await llm.ainvoke(converted)
        usage = {}
        if resp.usage_metadata:
            usage = {
                "prompt_tokens": resp.usage_metadata.get("input_tokens", 0),
                "completion_tokens": resp.usage_metadata.get("output_tokens", 0),
                "total_tokens": resp.usage_metadata.get("total_tokens", 0),
            }

        return LLMResponse(
            content=resp.content,
            model=self._cfg.model,
            usage=usage,
            raw=resp,
        )

    async def astream(
        self,
        messages: List[Tuple[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        llm = self._build_llm(temperature, max_tokens)
        converted = self._convert_messages(messages)

        async for chunk in llm.astream(converted):
            if chunk.content:
                yield chunk.content

    @property
    def model_name(self) -> str:
        return self._cfg.model

    @property
    def provider_name(self) -> str:
        if self._cfg.is_ollama:
            return "ollama"
        if "deepseek" in self._cfg.base_url.lower():
            return "deepseek"
        if "openai" in self._cfg.base_url.lower():
            return "openai"
        return "openai-compatible"
