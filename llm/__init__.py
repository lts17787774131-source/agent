# llm/__init__.py
# LLM 接入层：统一接口，支持多种大模型提供商
from llm.base import BaseLLM, LLMResponse
from llm.factory import get_llm, get_llm_provider

__all__ = ["BaseLLM", "LLMResponse", "get_llm", "get_llm_provider"]
