"""内容校验（Phase 1）"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .llm import get_llm

_llm = get_llm()

_VALIDATION_TEMPLATE = """请验证回答是否符合以下要求：
1. 事实准确性
2. 无主观臆断
3. 无商业广告内容

原问题：{input}
生成回答：{text}

验证结果（只需回答[有效]或[无效]）："""


def get_validation_chain():
    """返回验证链（Runnable）"""
    return RunnablePassthrough.assign(
        validation_result=ChatPromptTemplate.from_template(_VALIDATION_TEMPLATE)
        | _llm
        | StrOutputParser()
    )


# 兼容旧的模块级变量访问
validation_chain = get_validation_chain()


__all__ = ["validation_chain", "get_validation_chain", "_VALIDATION_TEMPLATE"]