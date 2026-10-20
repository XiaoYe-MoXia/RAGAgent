"""多轮对话（Phase 4）

用 dict 模拟数据库，生产环境应替换为 Redis / PostgreSQL
"""
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from .llm import get_llm

_llm = get_llm()

_session_store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据 session_id 取历史（不存在则创建）"""
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


# 带历史的 prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的技术客服助手，能记住用户之前说过的话，回答简洁。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 基础链
chat_chain = chat_prompt | _llm | StrOutputParser()

# 包装成带记忆的链
chat_with_history = RunnableWithMessageHistory(
    chat_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def chat_session(session_id: str, user_input: str) -> str:
    """多轮对话入口"""
    config = {"configurable": {"session_id": session_id}}
    response = chat_with_history.invoke({"input": user_input}, config=config)
    return response


def clear_session(session_id: str) -> None:
    """清空某个会话的历史"""
    if session_id in _session_store:
        del _session_store[session_id]


__all__ = [
    "chat_session",
    "chat_with_history",
    "get_session_history",
    "clear_session",
]