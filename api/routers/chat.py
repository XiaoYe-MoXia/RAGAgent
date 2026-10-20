"""/api/chat/* 对话相关接口

- POST /api/chat/chain         固定流程（非流式，保留兼容）
- POST /api/chat/stream        流式（仅 stream 模式）
- POST /api/chat/rag           RAG（非流式，保留兼容）
- POST /api/chat/agent         Agent（非流式，保留兼容）
- POST /api/chat/multi         多轮对话（非流式，保留兼容）
- POST /api/chat/all-stream    ⭐ 全模式统一流式入口（推荐）
"""
import asyncio
import io
import sys
from contextlib import redirect_stdout

from fastapi import APIRouter, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sse_starlette.sse import EventSourceResponse

from api.models import (
    AgentResponse,
    AllStreamRequest,
    ChatRequest,
    ChatResponse,
    MultiTurnRequest,
    MultiTurnResponse,
)
from src.cleaning import InputRejected, clean_input
from src.experts import (
    default_chain,
    historical_chain,
    math_chain,
    technical_chain,
    get_expert_by_category,
)
from src.llm import get_llm
from src.memory import get_session_history
from src.pipeline import (
    chat_with_agent,
    multi_turn_chat,
    process_query,
    rag_answer,
)
from src.rag import format_docs, get_retriever
from src.router import router_prompt
from src.validation import _VALIDATION_TEMPLATE

router = APIRouter(prefix="/api/chat", tags=["chat"])


# ---------- Chain ----------
@router.post("/chain", response_model=ChatResponse)
def chat_chain(req: ChatRequest):
    """Chain 固定流程（清洗→分类→专家→校验）"""
    try:
        # 直接调 pipeline，但元信息从日志里捞不到——这里跑两遍代价大
        # 简单方案：先 invoke chain 拿原始 state，再单独格式化
        from src.cleaning import clean_input as ci
        from src.router import router_chain
        from src.validation import validation_chain as vc

        from langchain_core.runnables import RunnableLambda, RunnablePassthrough

        full_chain = RunnableLambda(ci) | router_chain | vc
        try:
            result = full_chain.invoke({"raw_input": req.query})
        except InputRejected as e:
            return ChatResponse(answer=f"输入已被过滤：{e}", mode="chain", metadata={"rejected": True})

        meta = {
            "cleaned_input": result["input"],
            "category": result["category"],
            "validation": result["validation_result"],
        }
        if "有效" in result["validation_result"]:
            answer = result["text"]
        else:
            answer = "回答需要人工复核，我们将在24小时内联系您"
        return ChatResponse(answer=answer, mode="chain", metadata=meta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Stream (SSE) ----------
@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """流式输出（Server-Sent Events）

    关键修复：
    1. 用 async for + astream 真正异步流式（不是同步 generator 卡死事件循环）
    2. 用 chat_model.astream 直接拿字符串 chunk（绕过 chain 的 AIMessageChunk 包装）
    3. 中间步骤（清洗/分类/校验）放在线程池跑，不阻塞流
    """
    import asyncio
    from langchain_core.messages import HumanMessage

    async def event_generator():
        try:
            # 1. 清洗（CPU 密集，放线程池）
            try:
                cleaned = await asyncio.to_thread(clean_input, {"raw_input": req.query})
            except InputRejected as e:
                yield {"event": "error", "data": f"输入已被过滤：{e}"}
                return
            yield {"event": "step", "data": f"清洗后: {cleaned['input']}"}

            # 2. 分类（CPU 密集，放线程池）
            llm = get_llm()
            classification_chain = router_prompt | llm | StrOutputParser()
            category = await asyncio.to_thread(
                classification_chain.invoke, {"input": cleaned["input"]}
            )
            category = category.strip().lower()
            yield {"event": "step", "data": f"分类: {category}"}

            # 3. 流式生成：拿到 prompt template，自己 format 成 messages，再 astream LLM
            from src.experts import (
                technical_chain, historical_chain, math_chain, default_chain,
            )
            chain_map = {
                "technical": technical_chain,
                "historical": historical_chain,
                "math": math_chain,
                "default": default_chain,
            }
            chosen = chain_map.get(category, default_chain)
            # chosen 是 RunnableSequence(prompt | llm | parser)
            # 解包拿 prompt template
            expert_prompt = chosen.first  # ChatPromptTemplate
            messages = expert_prompt.format_messages(input=cleaned["input"])

            full_text = ""
            async for chunk in llm.astream(messages):
                # chunk 是 AIMessageChunk，取 .content（字符串）
                piece = chunk.content if hasattr(chunk, "content") else str(chunk)
                if piece:
                    full_text += piece
                    yield {"event": "chunk", "data": piece}

            # 4. 校验（CPU 密集，放线程池）
            inner = ChatPromptTemplate.from_template(_VALIDATION_TEMPLATE) | llm | StrOutputParser()
            validation = await asyncio.to_thread(
                inner.invoke, {"input": cleaned["input"], "text": full_text}
            )
            yield {"event": "step", "data": f"校验: {validation.strip()}"}
            yield {"event": "done", "data": full_text}
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())


# ---------- RAG ----------
@router.post("/rag", response_model=ChatResponse)
def chat_rag(req: ChatRequest):
    """RAG 模式（返回检索条数作为元信息）"""
    try:
        # 先检索拿元信息
        retrieved = get_retriever().invoke(req.query)
        sources = [
            {
                "category": doc.metadata.get("category", ""),
                "source": doc.metadata.get("source", ""),
                "preview": doc.page_content[:100],
            }
            for doc in retrieved
        ]
        answer = rag_answer(req.query)
        return ChatResponse(
            answer=answer,
            mode="rag",
            metadata={"sources": sources, "retrieved_count": len(retrieved)},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Agent ----------
@router.post("/agent", response_model=AgentResponse)
def chat_agent(req: ChatRequest):
    """Agent 模式（返回思考日志）"""
    try:
        # 抓 verbose 输出
        from src.agent import get_agent_executor

        executor = get_agent_executor(verbose=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = executor.invoke({"input": req.query})
        log = [l for l in buf.getvalue().splitlines() if l.strip()]
        return AgentResponse(
            answer=result["output"],
            mode="agent",
            tool_log=log[-30:],  # 只取最后 30 行
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Multi-turn ----------
@router.post("/multi", response_model=MultiTurnResponse)
def chat_multi(req: MultiTurnRequest):
    """多轮对话"""
    try:
        answer = multi_turn_chat(req.session_id, req.query)

        from src.memory import get_session_history
        hist_len = len(get_session_history(req.session_id).messages)

        return MultiTurnResponse(
            answer=answer,
            mode="multi_turn",
            session_id=req.session_id,
            history_length=hist_len,
            metadata={},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ⭐ 全模式统一流式入口
# ============================================================
# 之前 5 个模式里只有 /stream 是流式，其它都是一次性返回。
# 改造后：前端只调 /api/chat/all-stream 一个接口，mode 作为参数决定走哪种逻辑。
# 所有模式都走 SSE 流式，前端体验一致（打字机效果）。
# ============================================================

CHAIN_TO_LLM = {
    "technical": technical_chain,
    "historical": historical_chain,
    "math": math_chain,
    "default": default_chain,
}

RAG_PROMPT = ChatPromptTemplate.from_template(
    """你是电商客服助手。请基于以下【知识库内容】回答用户问题。

【知识库内容】
{context}

【用户问题】
{input}

要求：
1. 必须基于知识库内容回答，不要编造
2. 找不到答案时说"知识库中没有相关信息"
3. 回答简洁明了"""
)

# auto 模式的 RAG 变体：多一段对话背景，支持"那它保修多久？"这类追问
ASSISTANT_RAG_PROMPT = ChatPromptTemplate.from_template(
    """你是电商客服助手。请基于【知识库内容】和【对话背景】回答用户问题。

【知识库内容】
{context}

【对话背景（可能为空）】
{history}

【用户问题】
{input}

要求：
1. 必须基于知识库内容回答，不要编造
2. 用户追问时，代词（它/这个/那个产品）指对话背景里最近提到的产品
3. 找不到答案时说"知识库中没有相关信息"
4. 回答简洁明了"""
)


def _stream_llm_text(llm, messages):
    """异步生成器：把 AIMessageChunk 流式转成纯字符串 chunk

    Yields:
        str: 每次返回一段增量文本
    """
    async def gen():
        async for chunk in llm.astream(messages):
            piece = chunk.content if hasattr(chunk, "content") else str(chunk)
            if piece:
                yield piece
    return gen()


def _agent_events(user_input: str):
    """Agent 真流式（SSE 事件生成器）。agent 模式和 auto 的 tool 路径共用。

    AgentStreamCallback 把每个 token / 工具调用实时推进 queue，
    这里边收边 yield —— 前端能看着 Agent 一步步思考。
    """
    async def gen():
        from src.agent import AgentStreamCallback, get_agent_executor

        executor = get_agent_executor(verbose=False)
        queue: asyncio.Queue = asyncio.Queue()

        async def run_agent():
            try:
                result = await executor.ainvoke(
                    {"input": user_input},
                    config={"callbacks": [AgentStreamCallback(queue)]},
                )
                return result.get("output", "")
            except Exception as e:
                return f"__AGENT_ERROR__{e}"
            finally:
                queue.put_nowait(None)  # 哨兵：通知消费端结束

        task = asyncio.create_task(run_agent())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                kind, payload = item
                if kind == "token":
                    yield {"event": "agent_token", "data": payload}
                else:
                    yield {"event": "tool_log", "data": payload}
        except GeneratorExit:
            task.cancel()  # 客户端断开，别让 Agent 在后台白跑
            raise

        final = await task
        if final.startswith("__AGENT_ERROR__"):
            yield {"event": "error", "data": final[len("__AGENT_ERROR__"):]}
            return
        yield {"event": "final", "data": final}
        yield {"event": "done", "data": final}
    return gen()


@router.post("/all-stream")
async def chat_all_stream(req: AllStreamRequest):
    """⭐ 全模式统一流式入口

    - mode=auto    : ⭐ 智能客服（Phase 10）——意图路由：knowledge→RAG / tool→Agent / chat→直答
    - mode=chain   : 清洗 → 分类 → 专家流式回答 → 校验
    - mode=rag     : 检索知识库 → 流式回答（同时返回 sources 元信息）
    - mode=agent   : Agent 自主调用工具（流式推 Thought/Action/Observation）
    - mode=multi   : 多轮对话（带 session_id 记忆 + 流式回答）
    - mode=stream  : 等同 chain 但强制流式（保留兼容）
    """
    llm = get_llm()

    async def event_generator():
        try:
            mode = req.mode

            # ---- 公共：清洗 ----
            try:
                cleaned = await asyncio.to_thread(clean_input, {"raw_input": req.query})
            except InputRejected as e:
                yield {"event": "error", "data": f"输入已被过滤：{e}"}
                return
            yield {"event": "step", "data": f"清洗后: {cleaned['input']}"}

            # ============================================================
            # ⭐ AUTO 模式（Phase 10 智能客服：意图路由统一入口）
            # ============================================================
            if mode == "auto":
                import json as _json

                from src.assistant import (
                    INTENT_DESC,
                    INTENT_KNOWLEDGE,
                    INTENT_TOOL,
                    _history_text,
                    classify_intent,
                    format_history_for_rag,
                    remember_turn,
                    windowed_messages,
                )

                session_id = req.session_id

                # ---- 1. 意图分类（LLM 调用，放线程池）----
                yield {"event": "step", "data": "正在理解问题..."}
                intent = await asyncio.to_thread(
                    classify_intent, cleaned["input"], _history_text(session_id)
                )
                # route 事件：告诉前端走了哪条路（json.dumps 同 sources 的教训）
                yield {"event": "route", "data": _json.dumps(
                    {"path": intent, "desc": INTENT_DESC[intent]}, ensure_ascii=False
                )}
                yield {"event": "step", "data": f"智能路由: {INTENT_DESC[intent]}"}

                # ---- 2a. knowledge 路径：RAG（带对话背景，支持追问）----
                if intent == INTENT_KNOWLEDGE:
                    retriever = get_retriever()
                    retrieved = await asyncio.to_thread(retriever.invoke, cleaned["input"])
                    sources = [
                        {
                            "category": d.metadata.get("category", ""),
                            "source": d.metadata.get("source", ""),
                            "preview": d.page_content[:100],
                        }
                        for d in retrieved
                    ]
                    yield {"event": "sources", "data": _json.dumps(sources, ensure_ascii=False)}

                    messages = ASSISTANT_RAG_PROMPT.format_messages(
                        context=format_docs(retrieved),
                        history=format_history_for_rag(session_id),
                        input=cleaned["input"],
                    )
                    full_text = ""
                    async for piece in _stream_llm_text(llm, messages):
                        full_text += piece
                        yield {"event": "chunk", "data": piece}
                    remember_turn(session_id, cleaned["input"], full_text)
                    yield {"event": "done", "data": full_text}
                    return

                # ---- 2b. tool 路径：Agent 真流式 ----
                if intent == INTENT_TOOL:
                    final_text = ""
                    async for ev in _agent_events(cleaned["input"]):
                        yield ev
                        if ev.get("event") == "final":
                            final_text = ev["data"]
                    if final_text:
                        remember_turn(session_id, cleaned["input"], final_text)
                    return

                # ---- 2c. chat 路径：直答（滑动窗口历史 + 流式 + 校验）----
                messages = windowed_messages(
                    session_id,
                    "你是友善的电商客服助手。基于对话历史简洁回答用户；"
                    "涉及具体产品或售后时，建议用户描述具体产品名称。",
                    cleaned["input"],
                )
                full_text = ""
                async for piece in _stream_llm_text(llm, messages):
                    full_text += piece
                    yield {"event": "chunk", "data": piece}

                # 校验（流式结束再跑，不阻塞打字机）
                validation = await asyncio.to_thread(
                    (ChatPromptTemplate.from_template(_VALIDATION_TEMPLATE)
                     | llm | StrOutputParser()).invoke,
                    {"input": cleaned["input"], "text": full_text},
                )
                yield {"event": "step", "data": f"校验: {validation.strip()}"}
                remember_turn(session_id, cleaned["input"], full_text)
                yield {"event": "done", "data": full_text}
                return

            # ============ CHAIN / STREAM 模式 ============
            if mode in ("chain", "stream"):
                # 分类
                classification_chain = router_prompt | llm | StrOutputParser()
                category = await asyncio.to_thread(
                    classification_chain.invoke, {"input": cleaned["input"]}
                )
                category = category.strip().lower()
                yield {"event": "step", "data": f"分类: {category}"}

                # 选专家 → 流式
                chain = CHAIN_TO_LLM.get(category, default_chain)
                expert_prompt = chain.first  # ChatPromptTemplate
                messages = expert_prompt.format_messages(input=cleaned["input"])

                full_text = ""
                async for piece in _stream_llm_text(llm, messages):
                    full_text += piece
                    yield {"event": "chunk", "data": piece}

                # 校验
                inner = ChatPromptTemplate.from_template(_VALIDATION_TEMPLATE) | llm | StrOutputParser()
                validation = await asyncio.to_thread(
                    inner.invoke, {"input": cleaned["input"], "text": full_text}
                )
                yield {"event": "step", "data": f"校验: {validation.strip()}"}
                yield {"event": "done", "data": full_text}
                return

            # ============ RAG 模式 ============
            if mode == "rag":
                # 1. 检索（CPU 密集，放线程池）
                retriever = get_retriever()
                retrieved = await asyncio.to_thread(retriever.invoke, cleaned["input"])

                sources = [
                    {
                        "category": d.metadata.get("category", ""),
                        "source": d.metadata.get("source", ""),
                        "preview": d.page_content[:100],
                    }
                    for d in retrieved
                ]
                # 先推 sources 给前端（必须 json.dumps：直接塞 list 会被
                # sse-starlette str() 成 Python repr 单引号，不是合法 JSON）
                import json as _json
                yield {"event": "sources", "data": _json.dumps(sources, ensure_ascii=False)}

                # 2. 流式生成
                context_str = format_docs(retrieved)
                messages = RAG_PROMPT.format_messages(
                    context=context_str, input=cleaned["input"]
                )
                full_text = ""
                async for piece in _stream_llm_text(llm, messages):
                    full_text += piece
                    yield {"event": "chunk", "data": piece}

                yield {"event": "done", "data": full_text}
                return

            # ============ AGENT 模式（真流式，共用 _agent_events） ============
            if mode == "agent":
                async for ev in _agent_events(cleaned["input"]):
                    yield ev
                return

            # ============ MULTI 模式 ============
            if mode == "multi":
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

                session_id = req.session_id
                history = get_session_history(session_id)
                existing_msgs = list(history.messages)  # 现有历史
                yield {"event": "step", "data": f"会话 {session_id}，历史 {len(existing_msgs)} 条"}

                # 构造完整 messages：可选 system + 历史 + 当前输入
                all_msgs = [
                    SystemMessage(content="你是友善的客服助手。基于对话历史简洁回答用户。"),
                    *existing_msgs,
                    HumanMessage(content=cleaned["input"]),
                ]

                # 流式生成
                full_text = ""
                async for piece in _stream_llm_text(llm, all_msgs):
                    full_text += piece
                    yield {"event": "chunk", "data": piece}

                # 写回历史
                history.add_message(HumanMessage(content=cleaned["input"]))
                history.add_message(AIMessage(content=full_text))

                yield {"event": "done", "data": full_text}
                return

            # 兜底
            yield {"event": "error", "data": f"未知 mode: {mode}"}

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())