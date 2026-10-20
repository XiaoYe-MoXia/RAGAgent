# 🤖 AIAgent - LangChain 学习项目

> 一个端到端的电商客服 AI，从 5 层清洗到智能体决策、REST API 后端、Web 前端，最终合并为智能客服助手，分 10 个 Phase 渐进构建。

## ✨ 功能

| Phase | 能力 | 入口 |
|---|---|---|
| 1 | LCEL 基础（清洗→分类→专家→校验）| `process_query()` |
| 2 | 流式输出（打字机效果）| `process_query_stream()` |
| 3 | 结构化输出（Pydantic 校验）| `analyze_comment()` |
| 4 | 多轮对话（带记忆）| `multi_turn_chat()` |
| 5 | RAG 检索增强（基于 vectorDocs/）| `rag_answer()` |
| 6 | Agent 智能体（自主调工具）| `chat_with_agent()` |
| 7 | 项目重构（拆 src/ 包 + Streamlit 界面）| `streamlit run app.py` |
| 8 | FastAPI 后端（5 模式 + 会话管理 + SSE 流式）| `uvicorn api.main:app` |
| 9 | **单文件 Web 前端**（零依赖，浏览器直接打开）| 打开 `web/index.html` |
| 10 | **⭐ 智能客服助手**（意图路由统一入口 + 跨路径记忆）| 前端默认模式 / `mode=auto` |

## 📦 安装

```bash
# 1. 创建 conda 环境
conda create -n AIAgent python=3.12 -y
conda activate AIAgent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 .env
#    LLM_API_KEY=your_key
#    LLM_BASE_URL=https://your-endpoint/v1
#    LLM_MODEL=your-model
```

## 🚀 运行

### A. Web 前端（Phase 9，最推荐）

```bash
# 1. 启后端
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 2. 浏览器打开 web/index.html
#    或命令行：start web/index.html   (Windows)
#             open web/index.html    (macOS)
```

仿 ChatGPT 风格，左侧选模式 + 会话，右侧聊天。支持 RAG 来源标签、Agent 思考日志、SSE 打字机效果。
**默认就是 ✨ 智能客服（auto）模式**：自动路由知识库/工具/对话，不用手选模式；其余 5 个模式保留作调试对比。

### B. FastAPI 后端（Phase 8）

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

### C. Streamlit Web 界面（Phase 7）

```bash
streamlit run app.py
```
打开 http://localhost:8501，左侧切模式，右侧问问题。

### D. 命令行

```bash
python wisdom.py
```
跑完所有 Phase 的 demo。

### E. 当 Python 包用

```python
from src.pipeline import process_query, rag_answer, chat_with_agent

result = process_query("咖啡机的水泵压力？")
answer = rag_answer("退货几天内必须寄回？")
answer = chat_with_agent("咖啡机压力换算成 PSI 多少？")
```

### E. 当 HTTP API 用

```bash
# RAG 问答
curl -X POST http://localhost:8000/api/chat/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "咖啡机的水泵是什么品牌？"}'

# 多轮对话（带 session_id）
curl -X POST http://localhost:8000/api/chat/multi \
  -H "Content-Type: application/json" \
  -d '{"query": "我叫张三", "session_id": "u1"}'

# 查会话历史
curl http://localhost:8000/api/sessions/u1/history
```

## 📁 项目结构

```
AIAgent/
├── app.py                   # Streamlit Web 入口（Phase 7）
├── wisdom.py                # 命令行演示
├── weather.py / travel.py   # Phase 6 的实战 demo
├── api/                     # FastAPI 后端（Phase 8）
│   ├── main.py              # FastAPI 应用入口
│   ├── models.py            # Pydantic 请求/响应模型
│   └── routers/
│       ├── health.py        # GET  /api/health
│       ├── chat.py          # POST /api/chat/{chain,stream,rag,agent,multi}
│       └── sessions.py      # GET/DELETE /api/sessions/{id}/...
├── web/                     # 单文件 Web 前端（Phase 9）← 新增
│   └── index.html           # 浏览器直接打开
├── src/                     # 核心业务代码
│   ├── config.py            # 配置（路径、模型参数、清洗词表）
│   ├── llm.py               # LLM 单例懒加载
│   ├── cleaning.py          # 5 层清洗
│   ├── experts.py           # 4 专家链
│   ├── router.py            # 分类路由
│   ├── validation.py        # 回答校验
│   ├── memory.py            # 多轮对话（基于 RunnableWithMessageHistory）
│   ├── rag.py               # RAG（Chroma 持久化 + BGE 中文 embedding）
│   ├── assistant.py         # ⭐ 智能客服（Phase 10：意图路由 + 滑动窗口记忆）
│   ├── agent.py             # Agent（ReAct 文本格式）
│   └── pipeline.py          # 端到端编排（4 个高层函数）
├── docs/                    # 10 个学习文档（PHASE1~PHASE10.md）
├── vectorDocs/              # 知识库（11 份 md/txt）
├── chroma_db/               # Chroma 向量库落盘目录（首次启动自动生成）
├── tests/                   # 测试脚本
│   └── test_api.py          # API 端到端测试
├── .env                     # API 配置
└── requirements.txt         # 依赖清单
```

## 🧪 推荐试玩的问题

| 模式 | 问题 |
|---|---|
| **✨ 智能客服** | 同样的问题随便问，看路由徽章切换路径："咖啡机水泵什么品牌"（→RAG）→"换算成PSI"（→Agent）→"你好你会啥"（→直答） |
| **Chain** | "我的电脑蓝屏怎么办？" "123 × 456 等于多少" |
| **Stream** | "介绍一下 Python 装饰器" |
| **RAG** | "咖啡机的水泵是什么品牌？" "退货几天内必须寄回？" |
| **Agent** | "咖啡机压力换算成 PSI 多少？" "退货审核要多久？" |
| **多轮** | "我叫张三" → "我叫什么？" → "我多大了？" |

## 🛠️ 技术栈

- **LangChain 1.x**（LCEL 表达式）
- **Pydantic v2**（数据校验）
- **Chroma** + **BGE-small-zh**（向量数据库持久化 + 中文 embedding，支持 metadata 过滤）
- **FastAPI** + **Uvicorn**（REST API 后端）
- **SSE-Starlette**（流式输出）
- **Streamlit**（Web 界面）
- **Python 3.12**

## 📚 学习路径

按 `docs/PHASE1.md` → `PHASE10.md` 顺序读，每个文档都有：
- 核心概念讲解
- 代码模板
- 动手练习
- Bug 速查表
- 自我检测题

## 🤝 贡献

这是个人学习项目，欢迎 fork 和魔改。：）

---

📌 **Note**: 当前项目使用 AIStudio 百度 ernie-5.1 作为 LLM。Phase 6 Agent 使用 ReAct 文本格式（不是 OpenAI Function Calling），因为 ernie-5.1 用专有 tool calling 格式。Phase 8 把所有能力封装成 REST API，业务代码 0 改动；Phase 9 写了一个零依赖的单文件前端 `web/index.html`，浏览器直接打开就能用。
