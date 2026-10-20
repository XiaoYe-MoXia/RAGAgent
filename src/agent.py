"""Agent 智能体（Phase 6）

- 3 个 Tool：search_knowledge_base / calculator / get_current_time
- 用 ReAct 风格（百度的 MiniMax-M3 模型不支持标准 OpenAI function calling）
- AgentStreamCallback：把执行过程实时推进 asyncio.Queue（供 SSE 真流式）
"""
import asyncio
import datetime
from functools import lru_cache

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from .config import AGENT_MAX_ITERATIONS, TOOL_RESULT_MAX_CHARS
from .llm import get_llm
from .rag import format_docs, get_retriever


# ---------- Tools ----------
@tool
def search_knowledge_base(query: str) -> str:
    """在电商产品/退货知识库里检索相关信息。输入是用户问题的核心关键词。"""
    retrieved = get_retriever().invoke(query)
    result = format_docs(retrieved)
    # Observation 会整段喂回 ReAct 下一轮的 prompt：全文 5 chunk ≈ 2000 字
    # 是格式噪音的来源（模型曾把'Observation'写进计算器参数），截断够用
    if len(result) > TOOL_RESULT_MAX_CHARS:
        result = result[:TOOL_RESULT_MAX_CHARS] + "\n...(内容过长已截断)"
    return result


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入必须是合法的 Python 算式字符串，例如 '100 * 0.2'、'9 * 14.5'。"""
    try:
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_current_time(_: str = "") -> str:
    """获取当前的日期和时间，无需任何参数。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [search_knowledge_base, calculator, get_current_time]


# ---------- ReAct Prompt ----------
# MiniMax-M3 唯一靠谱的 Agent 模式：禁止"Action 和 Final Answer 同时输出"
REACT_TEMPLATE = """你是星辰电商平台的客服助手。请用中文回答。

【可用工具】
{tools}

【强制规则】
1. 涉及咖啡机/手表/键盘等产品规格、退货政策 → 必须先调 search_knowledge_base
2. 涉及"现在几点"等时间问题 → 必须先调 get_current_time
3. 涉及单位换算/算式 → 必须先调 calculator
4. 【绝对禁止】在同一次响应里同时写 Action 和 Final Answer
5. 只有当你已经看到至少一次 Observation 之后，才能在下一轮写 Final Answer
6. 最终回答【必须】以 "Final Answer:" 开头，不要直接写答案

【严格格式】

Question: 用户问题
Thought: 你在想什么
Action: 要调的工具名，必须是 [{tool_names}] 之一
Action Input: 工具参数
Observation: 工具返回（系统填）
... (可重复 N 轮)
Thought: 我现在可以给出最终答案
Final Answer: 中文最终回答

开始！

Question: {input}
Thought:{agent_scratchpad}"""


# ---------- 实时流式回调 ----------
class AgentStreamCallback(AsyncCallbackHandler):
    """把 Agent 执行过程实时推进 asyncio.Queue（SSE 真流式的核心）。

    事件流（kind, payload）：
    - ("token", "...")   LLM 逐 token 输出（Thought/Action/Final Answer 原文）
    - ("tool",  "...")   工具调用边界（开始调用 / 拿到 Observation）

    前提：executor 用 ainvoke 驱动（异步回调跑在事件循环上，put_nowait 才安全），
    且 LLM 实例 streaming=True（否则 invoke 路径不会触发 on_llm_new_token）。
    """

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self._n_tools = 0  # 第几次工具调用（前端展示用）

    def _put(self, kind: str, payload: str) -> None:
        if payload:
            self.queue.put_nowait((kind, payload))

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 思考模型可能推 reasoning 空片，过滤掉
        if token:
            self._put("token", token)

    async def on_agent_action(self, action, **kwargs) -> None:
        self._n_tools += 1
        tool_input = action.tool_input
        if isinstance(tool_input, dict):
            tool_input = tool_input.get("query") or str(tool_input)
        # 模型偶尔会把换行/下一行提示词写进参数（如 '9*14.5\nObservation'），压平成一行
        tool_input = str(tool_input).replace("\r", " ").replace("\n", " ")
        self._put("tool", f"🔧 第{self._n_tools}步 调用工具 {action.tool} · 参数: {tool_input}")

    async def on_tool_end(self, output, **kwargs) -> None:
        out = str(output).replace("\n", " ")
        preview = out[:150] + ("…" if len(out) > 150 else "")
        self._put("tool", f"📥 第{self._n_tools}步 返回: {preview}")


# ---------- 单例懒加载 ----------
@lru_cache(maxsize=2)
def get_agent_executor(verbose: bool = True) -> AgentExecutor:
    """懒加载 Agent executor。

    LLM 用 streaming=True 的实例：ReAct 每一轮的 Thought/Action
    才能逐 token 触发回调（真流式），最终聚合结果不受影响。
    """
    _llm = get_llm(streaming=True)
    agent_prompt = ChatPromptTemplate.from_template(REACT_TEMPLATE)
    agent = create_react_agent(_llm, TOOLS, agent_prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=AGENT_MAX_ITERATIONS,
    )


# ---------- 入口函数 ----------
def run_agent(question: str) -> str:
    """Agent 问答入口"""
    print(f"\n====== Agent 问答 ======")
    print(f"用户问题: {question}\n")
    executor = get_agent_executor(verbose=False)  # UI 模式不打印 verbose，避免刷屏
    result = executor.invoke({"input": question})
    answer = result["output"]
    print(f"\n✅ Final Answer: {answer}")
    return answer


def run_agent_verbose(question: str) -> str:
    """Agent 问答（带 verbose 输出，适合调试）"""
    executor = get_agent_executor(verbose=True)
    result = executor.invoke({"input": question})
    return result["output"]


__all__ = [
    "search_knowledge_base",
    "calculator",
    "get_current_time",
    "TOOLS",
    "AgentStreamCallback",
    "get_agent_executor",
    "run_agent",
    "run_agent_verbose",
]