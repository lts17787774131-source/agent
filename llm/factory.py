# llm/factory.py
# ============================================================
# LLM 工厂：根据配置自动选择并实例化对应的 LLM 提供商
# ============================================================
from typing import Optional

from config import LLMConfig, config
from llm.base import BaseLLM
from llm.providers import OpenAICompatibleLLM

# 全局 LLM 单例缓存
_llm_instance: Optional[BaseLLM] = None


def get_llm_provider(cfg: Optional[LLMConfig] = None) -> BaseLLM:
    """
    根据配置创建 LLM 实例。

    目前所有 OpenAI 兼容接口（Ollama / DeepSeek / OpenAI 等）
    都使用 OpenAICompatibleLLM 实现。
    未来如需接入非 OpenAI 兼容的模型（如 Gemini、Claude 原生接口），
    可在此处根据 cfg 判断并返回对应实现。
    """
    if cfg is None:
        cfg = config.llm
    return OpenAICompatibleLLM(cfg)


def get_llm() -> BaseLLM:
    """获取全局单例 LLM 实例（懒加载）"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = get_llm_provider()
    return _llm_instance


def reset_llm() -> None:
    """重置全局 LLM 单例（配置变更后调用）"""
    global _llm_instance
    _llm_instance = None
