"""wisdom.py - 命令行入口（Phase 7 重构后变成薄门面）

所有实现已迁移到 src/ 包：
- src.config        配置
- src.llm           LLM 实例
- src.cleaning      5 层清洗
- src.experts       4 专家链
- src.router        路由
- src.validation     校验
- src.memory        多轮对话
- src.rag           RAG 检索
- src.agent         智能体
- src.pipeline      端到端编排

这个文件仅保留命令行演示入口 + 向后兼容的旧 API 导出。
Web 界面请用 app.py。
"""
import sys

# Windows GBK → UTF-8
sys.stdout.reconfigure(encoding="utf-8")

# 旧 API 兼容：从 src 包 re-export
from src.cleaning import (
    InputRejected,
    clean_input,
    normalize_text,
    remove_noise,
    filter_sensitive,
    quality_check,
    business_rules,
)
from src.experts import (
    technical_chain,
    historical_chain,
    math_chain,
    default_chain,
    get_expert_by_category,
)
from src.router import router_prompt, router_chain, pick_and_invoke
from src.validation import validation_chain
from src.memory import (
    chat_session,
    chat_with_history,
    get_session_history,
    clear_session,
)
from src.rag import (
    rag_query,
    rag_prompt,
    get_retriever,
    get_rag_chain,
    get_vector_store,
    format_docs,
    load_documents_from_folder,
)
from src.agent import (
    run_agent,
    run_agent_verbose,
    search_knowledge_base,
    calculator,
    get_current_time,
)
from src.pipeline import (
    process_query,
    process_query_stream,
    chat_with_agent,
    rag_answer,
    multi_turn_chat,
    full_chain,
)


# ================= Phase 1 练习 =================
def run_phase1_exercises():
    """同步 + 流式各跑几个 case"""
    print("\n" + "=" * 60)
    print("【Phase 1 同步版本】")
    print("=" * 60)
    sync_cases = [
        "广告微商加微信我的电脑频繁蓝屏怎么办？",
        "介绍一下中国的唐朝",
        "123 * 456 等于多少",
        "你好",
    ]
    for q in sync_cases:
        process_query(q)
        print("-" * 50)

    print("\n" + "=" * 60)
    print("【Phase 2 流式版本】")
    print("=" * 60)
    for q in sync_cases:
        process_query_stream(q)
        print("-" * 50)


# ================= Phase 4 练习 =================
def run_phase4_exercises():
    """多轮对话练习"""
    print("\n" + "=" * 60)
    print("【Phase 4 多轮对话】")
    print("=" * 60)

    # 练习 1：自我介绍
    print("\n--- 练习 1：让 AI 自我介绍并验证 ---")
    chat_session("test", "我叫张三，今年 30 岁，是程序员")
    print(f"AI: {chat_session('test', '我叫什么名字？')}")
    print(f"AI: {chat_session('test', '我多大了？')}")

    # 练习 2：会话隔离
    print("\n--- 练习 2：会话隔离 ---")
    chat_session("A", "我喜欢吃苹果")
    print(f"B（应不知道）: {chat_session('B', '我喜欢吃什么？')}")
    print(f"A（应记得）: {chat_session('A', '我喜欢吃什么？')}")

    # 练习 3：长对话 token 消耗
    print("\n--- 练习 3：长对话 ---")
    for i in range(1, 11):
        resp = chat_session("long", f"这是第{i}轮")
        hist = len(get_session_history("long").messages)
        print(f"轮次 {i}: 历史={hist}  AI: {resp[:30]}...")


# ================= Main =================
if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("""
╔════════════════════════════════════════════════════════════╗
║            AIAgent - LangChain 学习项目（Phase 7）            ║
║                                                              ║
║  运行:                                                        ║
║    python wisdom.py       ← 命令行演示（这个文件）            ║
║    streamlit run app.py  ← Web 界面（推荐）                 ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Phase 1-2 同步 + 流式
    run_phase1_exercises()

    # Phase 4 多轮对话
    run_phase4_exercises()

    # Phase 5 RAG
    print("\n\n" + "=" * 60)
    print(">>> Phase 5：RAG 检索增强生成")
    print("=" * 60)
    rag_test_questions = [
        "咖啡机的水泵是什么品牌？萃取压力多少？",
        "智能手表的电池续航多久？支持哪些运动模式？",
        "退货的流程是什么？需要哪些材料？",
    ]
    for q in rag_test_questions:
        rag_query(q)
        print("\n" + "-" * 50)

    # Phase 6 Agent
    print("\n\n" + "=" * 60)
    print(">>> Phase 6：Agent 智能体")
    print("=" * 60)
    agent_test_questions = [
        "咖啡机的萃取压力是多少？换算成 PSI 大约多少？（提示：1 Bar ≈ 14.5 PSI）",
        "现在几点了？",
        "退货流程审核需要多久？审核后几天内必须寄回？",
    ]
    for q in agent_test_questions:
        try:
            run_agent(q)
        except Exception as e:
            print(f"[Agent 出错]: {e}")
        print("\n" + "-" * 50)