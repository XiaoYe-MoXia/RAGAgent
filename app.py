"""app.py - Streamlit Web 界面

启动：
    streamlit run app.py

提供 5 种模式：
- Chain：固定流程（清洗→分类→专家→校验）
- Stream：流式输出
- RAG：纯知识库问答
- Agent：智能体（自主调工具）
- 多轮对话：带记忆
"""
import streamlit as st

from src.pipeline import (
    process_query,
    process_query_stream,
    rag_answer,
    chat_with_agent,
    multi_turn_chat,
)
from src.config import DOCS_DIR
from src.rag import load_documents_from_folder


# ============== 页面配置 ==============
st.set_page_config(
    page_title="AIAgent - LangChain 学习 Demo",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AIAgent - LangChain 学习 Demo")
st.caption("电商客服智能体 ｜ 清洗 / 分类 / RAG / Agent / 多轮对话 一站体验")


# ============== 侧边栏：模式选择 + 配置 ==============
with st.sidebar:
    st.header("⚙️ 设置")

    mode = st.radio(
        "选择对话模式",
        options=["Chain（固定流程）", "Stream（流式）", "RAG（知识库）", "Agent（智能体）", "多轮对话"],
        index=0,
        help="Chain=固定管线 ｜ Stream=打字机 ｜ RAG=只查知识库 ｜ Agent=自主决策 ｜ 多轮=带记忆",
    )

    # 多轮对话需要 session_id
    session_id = "default"
    if mode == "多轮对话":
        session_id = st.text_input("会话 ID", value="default", help="不同 ID 历史独立")

    st.divider()

    # 知识库信息
    with st.expander("📚 知识库信息", expanded=False):
        try:
            docs = load_documents_from_folder(DOCS_DIR)
            from collections import Counter
            cnt = Counter(d.metadata["category"] for d in docs)
            st.write(f"**文档块总数**: {len(docs)}")
            for cat, n in cnt.most_common():
                st.write(f"- {cat}: {n} 块")
        except Exception as e:
            st.error(f"加载失败: {e}")

    st.divider()
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        if "multi_turn_session" in st.session_state:
            from src.memory import clear_session
            clear_session(st.session_state.multi_turn_session)
        st.rerun()


# ============== 状态管理 ==============
if "messages" not in st.session_state:
    st.session_state.messages = []

if "multi_turn_session" not in st.session_state:
    st.session_state.multi_turn_session = session_id


# ============== 历史消息渲染 ==============
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Agent 模式：附加工具调用日志
        if msg.get("tool_log"):
            with st.expander("🔧 工具调用记录", expanded=False):
                for line in msg["tool_log"]:
                    st.code(line, language="text")


# ============== 用户输入处理 ==============
if user_input := st.chat_input("请输入您的问题..."):
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. 根据模式调用对应 pipeline
    with st.chat_message("assistant"):
        placeholder = st.empty()
        tool_log: list[str] = []

        try:
            if mode == "Chain（固定流程）":
                with st.spinner("Chain 流程处理中（清洗→分类→专家→校验）..."):
                    answer = process_query(user_input)
                placeholder.markdown(answer)

            elif mode == "Stream（流式）":
                placeholder.markdown("▌")
                chunks = []
                # process_query_stream 内部用 print 输出，无法直接 capture
                # 所以直接调 experts.stream 自己实现流式
                from src.cleaning import clean_input, InputRejected
                from src.router import router_prompt
                from src.experts import get_expert_by_category
                from src.validation import _VALIDATION_TEMPLATE
                from langchain_core.output_parsers import StrOutputParser
                from langchain_core.prompts import ChatPromptTemplate
                from src.llm import get_llm

                try:
                    cleaned = clean_input({"raw_input": user_input})
                except InputRejected as e:
                    answer = f"输入已被过滤：{e}"
                    placeholder.markdown(answer)
                else:
                    llm = get_llm()
                    classification_chain = router_prompt | llm | StrOutputParser()
                    category = classification_chain.invoke({"input": cleaned["input"]}).strip().lower()
                    chosen = get_expert_by_category(category)

                    full_text = ""
                    for chunk in chosen.stream({"input": cleaned["input"]}):
                        full_text += chunk
                        chunks.append(chunk)
                        placeholder.markdown(full_text + "▌")
                    placeholder.markdown(full_text)
                    answer = full_text

                    # 校验
                    inner = ChatPromptTemplate.from_template(_VALIDATION_TEMPLATE) | llm | StrOutputParser()
                    validation = inner.invoke({"input": cleaned["input"], "text": full_text})
                    if "有效" not in validation:
                        answer = f"{full_text}\n\n⚠️ 校验未通过：{validation.strip()}"
                        placeholder.markdown(answer)

            elif mode == "RAG（知识库）":
                with st.spinner("RAG 检索中..."):
                    retrieved = _show_rag_steps(user_input)
                    tool_log = retrieved
                    answer = rag_answer(user_input)
                placeholder.markdown(answer)

            elif mode == "Agent（智能体）":
                with st.spinner("Agent 思考 + 调工具中..."):
                    # 抓 Agent verbose 输出
                    import io
                    import sys as _sys
                    from contextlib import redirect_stdout

                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        answer = chat_with_agent(user_input)
                    tool_log = buf.getvalue().splitlines()
                    tool_log = [l for l in tool_log if l.strip()]
                placeholder.markdown(answer)
                if tool_log:
                    with st.expander("🔧 Agent 思考 + 工具调用记录", expanded=True):
                        for line in tool_log[-30:]:  # 只显示最后 30 行，避免太长
                            st.code(line, language="text")

            elif mode == "多轮对话":
                sid = st.session_state.multi_turn_session
                with st.spinner(f"多轮对话中（session={sid}）..."):
                    answer = multi_turn_chat(sid, user_input)
                placeholder.markdown(answer)
                # 显示当前 session 历史消息数
                from src.memory import get_session_history
                hist_len = len(get_session_history(sid).messages)
                st.caption(f"📝 当前历史消息数: {hist_len}")

        except Exception as e:
            answer = f"❌ 处理失败: {e}"
            placeholder.error(answer)
            tool_log.append(f"EXCEPTION: {e}")

    # 3. 保存助手消息
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer if 'answer' in dir() else "",
        "tool_log": tool_log if tool_log else None,
    })


# ============== 辅助：RAG 检索过程可视化 ==============
def _show_rag_steps(question: str) -> list[str]:
    """把 RAG 检索的中间步骤显示出来"""
    from src.rag import get_retriever
    log = []
    retrieved = get_retriever().invoke(question)
    log.append(f"检索到 {len(retrieved)} 条相关文档：")
    for i, doc in enumerate(retrieved, 1):
        log.append(f"[{i}] [{doc.metadata.get('category', '')}] {doc.page_content[:100]}...")
    return log


# ============== 底部说明 ==============
st.divider()
with st.expander("📖 关于这个项目", expanded=False):
    st.markdown("""
**AIAgent** - LangChain 1.x 学习项目（7 个阶段）

| Phase | 内容 | 入口 |
|---|---|---|
| 1 | LCEL 基础（清洗→分类→专家）| `process_query()` |
| 2 | 流式输出 | `process_query_stream()` |
| 3 | 结构化输出（Pydantic）| `analyze_comment()` |
| 4 | 多轮对话 | `multi_turn_chat()` |
| 5 | RAG 检索增强 | `rag_answer()` |
| 6 | Agent 智能体 | `chat_with_agent()` |
| 7 | 项目重构 + Web 界面 | 当前页 |

**架构**：
```
src/
├── config.py    # 配置
├── llm.py       # LLM 单例
├── cleaning.py  # 5 层清洗
├── experts.py   # 4 专家链
├── router.py    # 路由
├── validation.py # 校验
├── memory.py    # 多轮对话
├── rag.py       # RAG
├── agent.py     # Agent + Tools
└── pipeline.py  # 端到端编排
```

试试这些问题：
- 💻 "我的电脑蓝屏怎么办？" → 走 Chain 流程
- ☕ "咖啡机的水泵压力是多少？" → 走 RAG 检索
- 🤔 "咖啡机压力换算成 PSI 多少？" → 走 Agent（自动调多个工具）
- 💬 "我叫张三" → 多轮对话，AI 会记住
    """)