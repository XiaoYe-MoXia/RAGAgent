"""智能客服助手（Phase 10）

把 5 个模式合并为一个统一入口：用户只管提问，系统自动选路径。

- 意图分类（knowledge / tool / chat）→ 决定走 RAG / Agent / 直答
- 所有路径共享会话记忆（滑动窗口，防上下文无限增长）
- 和 Phase 1 路由的区别：Phase 1 是选"哪个专家回答"（technical/math/...），
  这里是选"走哪条管线"（要不要检索、要不要调工具）
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import HISTORY_WINDOW
from .llm import get_llm
from .memory import get_session_history

_llm = get_llm()


# ---------- 意图路由 ----------
INTENT_KNOWLEDGE = "knowledge"  # 产品/退货/政策 → RAG
INTENT_TOOL = "tool"            # 计算/时间/多步推理 → Agent
INTENT_CHAT = "chat"            # 闲聊/技术/通用 → 直答

assistant_router_prompt = ChatPromptTemplate.from_template(
    """你是电商客服系统的意图分类器。根据用户问题，只输出一个英文标签（不要任何解释或前缀）：
- 关于具体产品（咖啡机/手表/键盘/扫地机器人/筋膜枪等）的规格、使用、故障、保修、退货政策、物流售后 → knowledge
- 需要计算、单位换算、查当前时间、或多步推理才能回答 → tool
- 闲聊、问候、技术/历史/数学等通用问题 → chat

【对话背景（可能为空）】
{history}

【用户问题】
{input}

分类："""
)


def classify_intent(user_input: str, history_text: str = "") -> str:
    """意图分类：返回 knowledge / tool / chat 之一。

    带对话背景是因为"那它保修多久？"这种追问单看问题无法分类。
    分类异常时兜底为 chat（最安全的路径，直答不会错得离谱）。
    """
    chain = assistant_router_prompt | _llm | StrOutputParser()
    try:
        label = chain.invoke({"input": user_input, "history": history_text})
        label = label.strip().lower()
    except Exception as e:
        print(f"[Assistant] 意图分类失败，兜底 chat：{e}")
        return INTENT_CHAT
    for intent in (INTENT_KNOWLEDGE, INTENT_TOOL, INTENT_CHAT):
        if intent in label:
            return intent
    return INTENT_CHAT


INTENT_DESC = {
    INTENT_KNOWLEDGE: "📚 知识库问题，走 RAG 检索",
    INTENT_TOOL: "🔧 需要工具/计算，走 Agent",
    INTENT_CHAT: "💬 通用对话，直接回答",
}


# ---------- 会话记忆（滑动窗口） ----------
def _history_text(session_id: str) -> str:
    """把最近几轮历史拼成纯文本（给分类器用，只要大意）"""
    msgs = get_session_history(session_id).messages[-HISTORY_WINDOW:]
    lines = []
    for m in msgs:
        role = "用户" if isinstance(m, HumanMessage) else "客服"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def windowed_messages(session_id: str, system_text: str, user_input: str) -> list:
    """构造完整 messages：system + 最近 HISTORY_WINDOW 条历史 + 当前输入。

    滑动窗口是关键：历史无限堆积会撑爆 token 上限，
    且模型对中段内容"视而不见"（lost in the middle），质量反而下降。
    """
    history = list(get_session_history(session_id).messages)[-HISTORY_WINDOW:]
    return [
        SystemMessage(content=system_text),
        *history,
        HumanMessage(content=user_input),
    ]


def remember_turn(session_id: str, user_text: str, ai_text: str) -> None:
    """一轮问答写回会话记忆（所有路径统一调用）"""
    history = get_session_history(session_id)
    history.add_message(HumanMessage(content=user_text))
    from langchain_core.messages import AIMessage
    history.add_message(AIMessage(content=ai_text))


def format_history_for_rag(session_id: str) -> str:
    """RAG 路径的对话背景：比 chat 路径更短（只带最近 4 条，够理解追问即可）"""
    msgs = get_session_history(session_id).messages[-4:]
    lines = []
    for m in msgs:
        role = "用户" if isinstance(m, HumanMessage) else "客服"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


__all__ = [
    "INTENT_KNOWLEDGE",
    "INTENT_TOOL",
    "INTENT_CHAT",
    "INTENT_DESC",
    "assistant_router_prompt",
    "classify_intent",
    "windowed_messages",
    "remember_turn",
    "format_history_for_rag",
    "_history_text",
]
