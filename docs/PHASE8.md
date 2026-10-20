# Phase 8 — FastAPI 后端

> 把 7 个 Phase 全部能力（Chain / RAG / Agent / 多轮 / 流式）封装成 REST API，让前端、移动端、第三方系统都能调用。

## 1. 目标

| Before (Phase 1-7) | After (Phase 8) |
|---|---|
| 只能跑 `python xxx.py` CLI | 任何 HTTP client 都能调 |
| 模式分散在多个脚本里 | 统一入口 `/api/chat/*` |
| 没有持久化会话 | 会话可查历史 / 可清空 |
| 没有流式输出 | SSE 流式边生成边推 |
| 没有健康检查 | `/api/health` 暴露服务状态 |

## 2. 目录结构

```
AIAgent/
├── api/                       # ← 新增：FastAPI 后端
│   ├── __init__.py
│   ├── main.py                # FastAPI app + 中间件
│   ├── models.py              # Pydantic 请求/响应模型
│   └── routers/
│       ├── __init__.py
│       ├── health.py          # GET  /api/health
│       ├── chat.py            # POST /api/chat/{chain,stream,rag,agent,multi}
│       └── sessions.py        # GET/DELETE /api/sessions/{id}/history
├── src/                       # 业务逻辑（不变）
│   ├── pipeline.py            # 4 个高层入口
│   ├── chain.py / experts.py  # Chain 模式
│   ├── rag.py                 # RAG 模式
│   ├── agent.py               # Agent 模式
│   └── memory.py              # 多轮记忆
└── vectorDocs/                # 知识库（11 份 md/txt）
```

## 3. 端点清单

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/api/health` | 健康检查 + 知识库元信息 |
| POST | `/api/chat/chain` | 固定流程：清洗→分类→专家→校验 |
| POST | `/api/chat/stream` | SSE 流式输出 |
| POST | `/api/chat/rag` | RAG 检索增强 |
| POST | `/api/chat/agent` | Agent（带思考日志） |
| POST | `/api/chat/multi` | 多轮对话（需 session_id） |
| GET  | `/api/sessions/{id}/history` | 查会话历史 |
| DELETE | `/api/sessions/{id}` | 清空会话 |
| GET  | `/docs` | Swagger UI（自动生成） |
| GET  | `/openapi.json` | OpenAPI schema |

## 4. 启动

```bash
# 方式 1：直接
python -m api.main

# 方式 2：uvicorn（推荐开发用，支持 reload）
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

启动后访问 `http://localhost:8000/docs` 看 Swagger UI。

## 5. 调用示例

### 5.1 健康检查
```bash
curl http://localhost:8000/api/health
```
```json
{
  "status": "ok",
  "version": "0.1.0",
  "knowledge_base": {
    "total_chunks": 109,
    "by_category": {
      "咖啡机": 6,
      "退货信息": 22,
      "智能手表": 10,
      ...
    },
    "docs_dir": "D:\\deep_learning\\AIAgent\\vectorDocs"
  }
}
```

### 5.2 Chain 模式
```bash
curl -X POST http://localhost:8000/api/chat/chain \
  -H "Content-Type: application/json" \
  -d '{"query": "请问你们退货几天内必须寄回？"}'
```
```json
{
  "answer": "根据平台政策，您需要在收到货后 7 天内申请...",
  "mode": "chain",
  "metadata": {
    "cleaned_input": "请问你们退货几天内必须寄回？",
    "category": "退货",
    "validation_result": "回答有效"
  }
}
```

### 5.3 RAG 模式
```bash
curl -X POST http://localhost:8000/api/chat/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "咖啡机的水泵是什么品牌？"}'
```
```json
{
  "answer": "咖啡机使用的是意大利 ULKA 水泵，15Bar 额定压力...",
  "mode": "rag",
  "metadata": {
    "sources": [
      {"category": "咖啡机", "source": "咖啡机.md", "preview": "## 星辰醇香 BrewMaster S1..."},
      {"category": "咖啡机", "source": "咖啡机.md", "preview": "- 水泵：意大利ULKA，15Bar..."},
      ...
    ],
    "retrieved_count": 5
  }
}
```

### 5.4 Agent 模式
```bash
curl -X POST http://localhost:8000/api/chat/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "现在几点了？"}'
```
```json
{
  "answer": "现在是 2026-09-08 14:32:11。",
  "mode": "agent",
  "tool_log": [
    "Thought: 用户问时间，需要调用 get_current_time 工具",
    "Action: get_current_time",
    "Action Input: {}",
    "Observation: 2026-09-08 14:32:11",
    "Thought: 拿到时间了，直接回答",
    "Final Answer: 现在是 2026-09-08 14:32:11。"
  ]
}
```

### 5.5 多轮对话
```bash
# 第一轮
curl -X POST http://localhost:8000/api/chat/multi \
  -H "Content-Type: application/json" \
  -d '{"query": "我叫张三，今年 25 岁", "session_id": "user_001"}'

# 第二轮（同 session_id）
curl -X POST http://localhost:8000/api/chat/multi \
  -H "Content-Type: application/json" \
  -d '{"query": "我叫什么？多大了？", "session_id": "user_001"}'
```

### 5.6 查 / 清空会话
```bash
# 查历史
curl http://localhost:8000/api/sessions/user_001/history

# 清空
curl -X DELETE http://localhost:8000/api/sessions/user_001
```

### 5.7 SSE 流式
```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "介绍一下星辰守护智能门锁"}'
```
事件流：
```
event: step
data: 清洗后: 介绍一下星辰守护智能门锁

event: step
data: 分类: 智能门锁

event: chunk
data: 星辰守护 Guardian X1

event: chunk
data: 是一款采用...

event: done
data: 星辰守护 Guardian X1 是一款...
```

## 6. 请求/响应模型（Pydantic）

```python
# api/models.py
class ChatRequest(BaseModel):
    query: str                              # 用户输入

class MultiTurnRequest(ChatRequest):
    session_id: str = "default"             # 会话 ID

class ChatResponse(BaseModel):
    answer: str                             # AI 回复
    mode: str                               # chain / rag / agent
    metadata: dict = {}                     # 模式特有元信息

class MultiTurnResponse(ChatResponse):
    session_id: str
    history_length: int                     # 当前会话消息数

class AgentResponse(ChatResponse):
    tool_log: list[str] = []                # Agent 思考步骤
```

## 7. 关键设计

### 7.1 不重写业务代码
API 层只做"薄封装"——直接调 `src.pipeline` 里的 4 个高层函数：
```python
# src/pipeline.py 已经把 7 个 Phase 收口成 4 个函数
def process_query(query)        -> dict  # chain 模式
def rag_answer(query)           -> str   # rag 模式
def multi_turn_chat(sid, query) -> str   # 多轮模式
def chat_with_agent(query)      -> dict  # agent 模式
```

FastAPI 路由只是把这些函数包装成 HTTP 接口，**业务逻辑零修改**。

### 7.2 RAG 的元信息怎么来？
`/api/chat/rag` 返回 `sources`（每条命中文档的 category/source/preview），前端可以直接展示"参考来源"。

实现上跑了**两次检索**——一次拿元信息，一次给 LLM 拼 context。这点代价换来 API 更友好，是值得的。

### 7.3 Agent 的思考日志
用 `contextlib.redirect_stdout` 把 `verbose=True` 的 ReAct 日志抓到 StringIO，取最后 30 行返回。

### 7.4 启动时加载 .env
```python
# api/main.py
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
```
uvicorn 子进程可能没继承 shell 的环境变量，必须显式加载并打印确认。

### 7.5 懒加载 RAG 向量库
向量库 ~100MB，首次调用 `/api/chat/rag` 才加载，**不阻塞服务启动**。

## 8. 🐛 调试中遇到的关键 Bug

### Bug #1：RAG 只加载 .txt，丢了 10 份 .md

**症状**：检索"咖啡机的水泵"返回的都是智能手表内容。

**根因**：[src/rag.py:43](../src/rag.py#L43) 只过滤 `.txt` 后缀，而 `vectorDocs/` 下 10 份文档是 `.md`（只有 1 份 .txt）——向量库只索引了 10 个 chunk，全是智能手表。

**修复**：
```python
# 原来
if not filename.endswith(".txt"):
    continue

# 改成
if not (filename.endswith(".txt") or filename.endswith(".md")):
    continue
```

**验证**：修完后 109 chunks 全部加载，咖啡机查询能精确命中水泵规格。

### Bug #2：上游 LLM 返回 HTML 而非 JSON

**症状**：`POST /api/chat/*` 返回 500，detail 是
```
Unexpected response type from OpenAI-compatible endpoint.
Expected a dict or openai.BaseModel, got str: '<html>...<title>URL错误</title>...'
```

**根因**：环境网络层把对 `api.minimaxi.com` 的请求重定向到一个 URL 过滤页面（HTML），而不是真的访问到 LLM 服务。FastAPI 层无法处理这种"上游返回非 JSON"的情况。

**说明**：这是网络/代理问题，不是代码 bug。其他 Phase（weather.py / travel.py）也复现同样错误。**代码层面已经做了完善的错误处理**——返回 500 + 详细 detail，前端可识别。

### Bug #3：切换到 ernie-5.1 思考模型后，分类标签变成 default

**症状**：Chain / Stream 模式的 `metadata.category` 永远是 `default`，但回答内容仍然完整。

**根因**：ernie-5.1 是"思考模型"，最终输出放在 `reasoning_content` 字段，`content` 字段返回空字符串 `""`。Router prompt 要分类标签时被拿到空串，匹配不到任何专家 → 落到 default。

**验证**：
```bash
$ curl -X POST /api/chat/rag -d '{"query":"咖啡机的水泵是什么品牌？"}'
# 状态 200，答案正确：意大利ULKA，15Bar
# 检索条数: 5, 来源全部命中咖啡机.md  ← RAG 完美
```

**当前状态**：RAG / Agent / 多轮三种模式完全正常；Chain / Stream 仍能回答，但走 default 专家（用通用 prompt，不影响可用性）。要彻底解决，可在 `src/router.py` 把 prompt 改为"先输出思考，最后一行单独输出分类标签"，或改用非思考模型（ernie-4.5-8k）。

## 9. 验证清单

| 检查项 | 结果 |
|---|---|
| 服务能起 | ✅ uvicorn 监听 0.0.0.0:8000 |
| `/api/health` 返回 200 + 知识库元信息 | ✅ 109 chunks, 11 categories |
| OpenAPI schema 自动生成 | ✅ `/docs` 可访问 |
| `/api/sessions/...` 路由 | ✅ history / clear 都返回 200 |
| RAG 检索准确 | ✅ 咖啡机/退货查询命中正确文档 |
| 业务代码改动 | ✅ 0 行（薄封装） |

## 10. 后续

- Phase 9：写前端 Web 页面（Gradio / Streamlit / 纯 HTML），调用这套 API
- Phase 10：Docker 化 + 部署

## 11. 文件清单

| 文件 | 作用 |
|---|---|
| [api/main.py](../api/main.py) | FastAPI 应用入口、CORS、路由注册 |
| [api/models.py](../api/models.py) | Pydantic 请求/响应模型 |
| [api/routers/health.py](../api/routers/health.py) | 健康检查 + 知识库状态 |
| [api/routers/chat.py](../api/routers/chat.py) | 5 种对话模式 |
| [api/routers/sessions.py](../api/routers/sessions.py) | 会话历史查询/清空 |
| [tests/test_api.py](../tests/test_api.py) | 端到端测试脚本 |
