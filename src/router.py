"""路由：分类 → 选专家（Phase 1）"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .experts import get_expert_by_category
from .llm import get_llm

_llm = get_llm()

router_prompt = ChatPromptTemplate.from_template(
    """你是一个问题分类器。根据用户问题，只输出一个英文标签（不要任何解释或前缀）：
- 编程、电脑、技术故障等技术类问题 → technical
- 历史人物、朝代、事件等历史类问题 → historical
- 涉及数字运行，大小比对，数字逻辑运算等数学问题 → math
- 其他通用问题 → default
用户问题：{input}
分类："""
)


def pick_and_invoke(state: dict) -> dict:
    """根据分类标签挑专家链，并把回答写到 text 字段"""
    chosen = get_expert_by_category(state["category"])
    state["text"] = chosen.invoke({"input": state["input"]})
    return state


router_chain = (
    RunnablePassthrough.assign(
        category=router_prompt | _llm | StrOutputParser()
    )
    | RunnableLambda(pick_and_invoke)
)


__all__ = ["router_prompt", "router_chain", "pick_and_invoke"]