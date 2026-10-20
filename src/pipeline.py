"""端到端流程编排（Phase 7）

提供 3 个对外接口：
- process_query      同步 invoke（固定流程：清洗→分类→回答→校验）
- process_query_stream  流式输出
- chat_with_agent    智能体模式（自主决定调哪些工具）
- chat_session       多轮对话（带历史）
- rag_query          RAG 问答

每个模式独立可选，互不耦合。
"""
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .cleaning import InputRejected, clean_input
from .router import router_chain
from .validation import validation_chain


# 完整同步 chain：清洗 → 路由 → 校验
full_chain = (
    RunnableLambda(clean_input)
    | router_chain
    | validation_chain
)


def process_query(query: str) -> str:
    """同步处理：清洗 → 分类 → 专家回答 → 校验"""
    try:
        result = full_chain.invoke({"raw_input": query})
    except InputRejected as e:
        print(f"\n====== 处理日志 ======")
        print(f"原始输入: {query}")
        print(f"拦截原因: {e}")
        return f"输入已被过滤：{e}"

    print(f"\n====== 处理日志 ======")
    print(f"原始输入: {query}")
    print(f"清洗后输入: {result['input']}")
    print(f"分类标签: {result['category']}")
    print(f"生成回答: {result['text']}")
    print(f"验证结果: {result['validation_result']}")

    if "有效" in result["validation_result"]:
        return result["text"]
    return "回答需要人工复核，我们将在24小时内联系您"


def process_query_stream(query: str) -> str:
    """流式处理：清洗 → 分类 → 流式回答 → 校验"""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from .experts import get_expert_by_category
    from .router import router_prompt
    from .llm import get_llm
    from .validation import _VALIDATION_TEMPLATE

    llm = get_llm()

    print(f"\n====== 流式处理 ======")
    print(f"原始输入: {query}")

    try:
        cleaned = clean_input({"raw_input": query})
    except InputRejected as e:
        return f"输入已被过滤：{e}"
    print(f"清洗后: {cleaned['input']}")

    # 分类
    classification_chain = router_prompt | llm | StrOutputParser()
    category = classification_chain.invoke({"input": cleaned["input"]}).strip().lower()
    print(f"分类标签: {category}")

    chosen = get_expert_by_category(category)

    # 流式回答
    print(f"\n回答（流式输出）: ", end="", flush=True)
    full_text = ""
    for chunk in chosen.stream({"input": cleaned["input"]}):
        print(chunk, end="", flush=True)
        full_text += chunk
    print()

    # 校验
    inner_validation = (
        ChatPromptTemplate.from_template(_VALIDATION_TEMPLATE)
        | llm
        | StrOutputParser()
    )
    validation = inner_validation.invoke({"input": cleaned["input"], "text": full_text})
    print(f"校验结果: {validation}")

    if "有效" in validation:
        return full_text
    return "回答需要人工复核，我们将在24小时内联系您"


def chat_with_agent(question: str) -> str:
    """Agent 模式（懒加载，避免不必要启动）"""
    from .agent import run_agent
    return run_agent(question)


def rag_answer(question: str) -> str:
    """RAG 模式（懒加载）"""
    from .rag import rag_query
    return rag_query(question)


def multi_turn_chat(session_id: str, user_input: str) -> str:
    """多轮对话模式"""
    from .memory import chat_session
    return chat_session(session_id, user_input)


__all__ = [
    "full_chain",
    "process_query",
    "process_query_stream",
    "chat_with_agent",
    "rag_answer",
    "multi_turn_chat",
]