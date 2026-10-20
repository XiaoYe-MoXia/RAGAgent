# 📘 Phase 7 学习文档：项目整合 + Web 界面

> **本阶段目标**：把前 6 个 Phase 串成一个**结构清晰、可维护**的项目，并加上 **Web 界面**让普通用户也能用。

---

## 🎯 学习目标 Checklist

完成本阶段后，你应该能：
- [ ] 解释为什么单体脚本要拆成多文件
- [ ] 识别一个项目的"模块边界"（按职责拆）
- [ ] 用 `functools.lru_cache` 实现"懒加载单例"
- [ ] 用 Streamlit 快速搭一个 Web demo
- [ ] 区分"内部 API"和"对外接口"

---

## 📚 核心概念（5 个）

### 概念 1：为什么要拆模块？

**单体脚本的痛点**（`wisdom.py` 之前 796 行的状态）：
- ❌ 所有代码挤在一个文件，找东西靠 grep
- ❌ 改一个 bug 可能影响 6 个无关模块
- ❌ 测试时得跑整个脚本（启动慢）
- ❌ 团队其他人不知道从哪下手

**拆模块的好处**：
- ✅ 每个文件职责单一（一个 Phase 一个文件）
- ✅ 改 cleaning 不会碰 agent
- ✅ 单元测试可以单独跑某个模块
- ✅ 新人接手只看对应模块即可

---

### 概念 2：怎么拆？（按"职责"拆）

我们按 **6 个 Phase + 基础设施** 拆：

| 模块 | 职责 | 依赖 |
|---|---|---|
| `config.py` | 环境变量、路径、常量 | — |
| `llm.py` | LLM 实例（单例）| config |
| `cleaning.py` | 5 层清洗 | config |
| `experts.py` | 4 专家链 | llm |
| `router.py` | 分类路由 | experts, llm |
| `validation.py` | 校验链 | llm |
| `memory.py` | 多轮对话 | llm |
| `rag.py` | 文档加载 + 向量库 + RAG chain | config, llm |
| `agent.py` | Tools + Agent executor | llm, rag |
| `pipeline.py` | 端到端编排（对外接口）| 上面所有 |

**依赖方向是单向的**：底层（config/llm）→ 中层（cleaning/experts）→ 高层（agent/pipeline）

**绝不能反向依赖**：比如 cleaning 不能 import agent！

---

### 概念 3：`@lru_cache` 懒加载单例

**是什么**：用装饰器让"重对象"只创建一次。

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_vector_store():
    """首次调用才建向量库（避免启动慢）"""
    return FAISS.from_documents(load_docs(), get_embeddings())
```

**好处**：
- 启动时不会把 100MB 的 embedding 模型加载进来
- 跑命令行 demo 时不依赖 RAG，不会触发 FAISS
- 跑 Streamlit 时才用到 RAG，第一次访问才建库

**对比 eager loading**：
```python
# ❌ 启动就加载（启动慢，浪费内存）
vector_store = FAISS.from_documents(load_docs(), get_embeddings())

# ✅ 按需加载（启动快，按需付费）
@lru_cache(maxsize=1)
def get_vector_store():
    return FAISS.from_documents(load_docs(), get_embeddings())
```

---

### 概念 4：`wisdom.py` = 薄门面（Facade）

**是什么**：保留旧的 `wisdom.py`，但里面啥都不实现，**只是把所有模块 re-export 出来**。

```python
# wisdom.py（精简后 100 行）
from src.cleaning import clean_input, InputRejected
from src.rag import rag_query
from src.agent import run_agent
from src.pipeline import process_query, process_query_stream
# ...

if __name__ == "__main__":
    # 命令行测试入口
    process_query("...")
    rag_query("...")
    run_agent("...")
```

**好处**：
- 旧的测试代码、笔记、别人写的 import 都不会失效
- 真正干活的是 `src/`，新代码都从那里 import
- `wisdom.py` 只剩"演示"角色

---

### 概念 5：Streamlit 极简 Web 框架

**是什么**：用 Python 写 Web 界面，不用 HTML/CSS/JS。

**最小例子**：
```python
import streamlit as st

st.title("我的 App")
msg = st.chat_input("请输入")
if msg:
    st.write(f"你说: {msg}")
```

**我们的 App 结构**：
```
app.py
├── 侧边栏: 模式选择 + 知识库信息 + 清空按钮
├── 主区:   历史消息渲染 + 输入框 + 处理
└── 底部:   项目说明（折叠面板）
```

**5 个模式共享同一个 chat_input**：
```python
if user_input := st.chat_input("..."):
    if mode == "Chain（固定流程）":
        answer = process_query(user_input)
    elif mode == "Stream（流式）":
        for chunk in process_query_stream(user_input): ...
    elif mode == "RAG（知识库）":
        answer = rag_answer(user_input)
    elif mode == "Agent（智能体）":
        answer = chat_with_agent(user_input)
    elif mode == "多轮对话":
        answer = multi_turn_chat(session_id, user_input)
```

**启动方式**：
```bash
streamlit run app.py
# 自动打开 http://localhost:8501
```

---

## 🏗️ 新项目结构

```
AIAgent/
├── app.py                    # Web 入口（Streamlit）
├── wisdom.py                 # 命令行演示（薄门面）
├── src/                      # 核心代码包
│   ├── __init__.py
│   ├── config.py             # 配置
│   ├── llm.py                # LLM 单例
│   ├── cleaning.py           # 5 层清洗
│   ├── experts.py            # 4 专家链
│   ├── router.py             # 路由
│   ├── validation.py         # 校验
│   ├── memory.py             # 多轮对话
│   ├── rag.py                # RAG
│   ├── agent.py              # Agent
│   └── pipeline.py           # 端到端编排
├── tests/                    # 单元测试（占位）
├── docs/                     # 学习文档（PHASE1-7）
├── vectorDocs/               # 知识库（6 个 .txt）
├── .env                      # 配置（API key）
├── requirements.txt          # 依赖清单
└── README.md                 # 项目说明
```

---

## 🚀 使用方式

### 命令行

```bash
conda activate AIAgent
python wisdom.py
```

跑完 6 个 Phase 的所有 demo。

### Web 界面（推荐）

```bash
conda activate AIAgent
streamlit run app.py
```

打开 http://localhost:8501，左边切模式，右边输入问题。

### 当 Python 包用

```python
from src.pipeline import process_query, rag_answer, chat_with_agent

# 同步处理
result = process_query("咖啡机的水泵压力？")

# RAG 问答
answer = rag_answer("退货几天内必须寄回？")

# Agent
answer = chat_with_agent("咖啡机压力换算成 PSI 多少？")
```

---

## 📝 设计原则总结

1. **单一职责**：每个文件只干一件事
2. **单向依赖**：底层不依赖高层
3. **懒加载**：重资源按需加载
4. **向后兼容**：旧 API 不破坏（薄门面）
5. **对外接口清晰**：`pipeline.py` 是唯一入口

---

## 🎓 本阶段总结

| 学到的 | 怎么检验 |
|---|---|
| 模块拆分 | 能说出按什么维度拆（职责/层次）|
| 懒加载 | 能用 `@lru_cache` 改造一个 eager 模块 |
| Facade 模式 | 能解释为啥 wisdom.py 现在只剩 re-export |
| Streamlit | 能独立搭一个带侧边栏+聊天的 demo |
| 单向依赖 | 能识别循环依赖并修复 |

---

## ➡️ 收尾

**7 个 Phase 全部完成！** 这是一个真正"可运行、可维护、可扩展"的项目。

可选的进阶方向：
- 🅰 加 FastAPI 后端 + 前端分离
- 🅱 写单元测试（`tests/` 目录）
- 🅲 接真实业务（替换 mock 数据为生产数据）
- 🅓 上 Docker / 部署到云
- 🅔 LangGraph 重构 Agent（LangChain 1.x 推荐方向）

你想继续哪个？还是告一段落？