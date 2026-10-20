"""LLM 实例（单例懒加载）"""
from functools import lru_cache

from langchain_openai import ChatOpenAI

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE


@lru_cache(maxsize=4)
def get_llm(streaming: bool = False) -> ChatOpenAI:
    """获取 LLM 实例（懒加载，单例缓存）。

    streaming=True 的实例给 Agent 真流式用：只有开了 streaming，
    invoke 路径才会逐 token 触发 on_llm_new_token 回调。
    """
    return ChatOpenAI(
        base_url=LLM_BASE_URL or None,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        streaming=streaming,
    )