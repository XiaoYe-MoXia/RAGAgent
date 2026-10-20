"""4 个专家子链（Phase 1）"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .llm import get_llm

_llm = get_llm()

technical_chain = (
    ChatPromptTemplate.from_template(
        "作为技术专家，用通俗易懂的方式回答问题：\n"
        "问题：{input}\n"
        "回答（不超过100字）："
    )
    | _llm
    | StrOutputParser()
)

historical_chain = (
    ChatPromptTemplate.from_template(
        "作为历史学家，用严谨的方式回答问题：\n"
        "问题：{input}\n"
        "回答（包含可靠出处）："
    )
    | _llm
    | StrOutputParser()
)

math_chain = (
    ChatPromptTemplate.from_template(
        "作为数学家，用严谨的数学思维方式回答问题：\n"
        "问题：{input}\n"
        "回答："
    )
    | _llm
    | StrOutputParser()
)

default_chain = (
    ChatPromptTemplate.from_template(
        "作为通用助手回答问题：\n问题：{input}\n回答："
    )
    | _llm
    | StrOutputParser()
)


def get_expert_by_category(category: str):
    """根据分类标签选专家链"""
    c = category.strip().lower()
    if "technical" in c or "tech" in c:
        return technical_chain
    if "historical" in c or "history" in c:
        return historical_chain
    if "math" in c or "number" in c:
        return math_chain
    return default_chain


__all__ = [
    "technical_chain",
    "historical_chain",
    "math_chain",
    "default_chain",
    "get_expert_by_category",
]