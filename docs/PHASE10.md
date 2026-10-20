# Phase 10 — 智能客服助手（合并所有模式）

## 1. 这一阶段解决什么问题

Phase 1-9 做出来的是 **5 个模式**：chain / rag / agent / multi / stream。
用户要自己挑模式——这像 5 个 demo，不像一个产品。

Phase 10 把它们合并成**一个客服 AI 助手**：

```
用户提问（不用选模式）
   ↓
清洗（敏感词/长度，Phase 1）
   ↓
意图分类（LLM，三选一）──┬── knowledge → RAG 检索知识库（Phase 5）
                          ├── tool      → Agent 调工具（Phase 6，真流式）
                          └── chat      → LLM 直答（带记忆，Phase 4）
   ↓
统一：SSE 流式 + 会话记忆（所有路径共享）+ route 事件告知前端
```

## 2. 和 Phase 1 路由的区别（面试常问）

| | Phase 1 路由 | Phase 10 意图路由 |
|---|---|---|
| 分什么 | 选**哪个专家**回答（technical/math/...） | 选**走哪条管线**（要不要检索、要不要调工具） |
| 输出 | 4 类 | 3 类（knowledge / tool / chat） |
| 位置 | chain 模式内部的一步 | 整个系统的总入口 |

## 3. 文件改动

| 文件 | 改动 |
|---|---|
| `src/assistant.py`（新增） | 意图分类 + 滑动窗口记忆 + 写回历史 |
| `src/config.py` | `HISTORY_WINDOW=10`、`TOOL_RESULT_MAX_CHARS=1500` |
| `api/models.py` | mode 增加 `"auto"` 并设为默认 |
| `api/routers/chat.py` | all-stream 加 auto 分支；Agent 真流式抽成 `_agent_events()` 共用 |
| `src/agent.py` | 检索工具返回截断 1500 字（防 Observation 噪音） |
| `web/index.html` | ✨ auto 模式置顶默认；route 徽章展示路由结果 |

## 4. 关键实现

### 4.1 意图分类（src/assistant.py）

```python
def classify_intent(user_input, history_text="") -> str:
    chain = assistant_router_prompt | _llm | StrOutputParser()
    label = chain.invoke({"input": user_input, "history": history_text})
    ...
```

两个设计点：
- **带对话背景**："那它保修多久？"单看问题没法分类，必须看上一轮
- **兜底 chat**：分类失败/输出非法标签时走直答——最安全的路径，不会错得离谱

### 4.2 滑动窗口记忆（解决"上下文太长质量下降"）

```python
def windowed_messages(session_id, system_text, user_input):
    history = list(get_session_history(session_id).messages)[-HISTORY_WINDOW:]
    return [SystemMessage(...), *history, HumanMessage(user_input)]
```

为什么不带全部历史：
1. **token 上限**：聊 30 轮后必然超限报错
2. **lost in the middle**：模型对超长上下文的中段内容"视而不见"，还会被无关旧话题带偏

### 4.3 RAG 路径支持追问

`ASSISTANT_RAG_PROMPT` 多了一段【对话背景】：

```
用户：咖啡机多少钱？
客服：BrewMaster S1 售价 1899 元。
用户：那它保修多久？     ← "它"=咖啡机，靠对话背景消解
```

### 4.4 跨路径记忆

所有路径结束后统一 `remember_turn()` 写回历史。所以可以：
"咖啡机的水泵什么品牌"（走了 RAG）→ "换算成 PSI 多少"（走了 Agent）→
两轮之间模型是"记得"第一轮的。

### 4.5 Agent 检索截断

`search_knowledge_base` 工具返回截断到 1500 字。Observation 会整段拼进
ReAct 下一轮 prompt，全文 2000 字是格式噪音来源（曾导致模型把
'Observation' 写进计算器参数）。RAG 主路径不受影响（不走这个工具）。

## 5. SSE 事件协议（auto 模式）

| 事件 | 时机 | 内容 |
|---|---|---|
| step | 清洗/分类/校验各阶段 | 进度文本 |
| **route** | 意图分类完成 | `{"path":"knowledge","desc":"..."}` |
| sources | knowledge 路径检索完成 | JSON 数组 |
| agent_token / tool_log | tool 路径实时 | ReAct 原文 / 工具边界 |
| chunk | 流式回答 | 增量文本 |
| done | 结束 | 完整回答 |

## 6. 端到端验证

```bash
# 1. 知识库问题 → 应 route=knowledge
curl -X POST http://127.0.0.1:8000/api/chat/all-stream \
  -H "Content-Type: application/json" \
  -d '{"mode":"auto","query":"咖啡机的水泵是什么品牌","session_id":"t1"}'

# 2. 计算问题 → 应 route=tool
curl -X POST ... -d '{"mode":"auto","query":"咖啡机压力换算成PSI是多少","session_id":"t1"}'

# 3. 闲聊 → 应 route=chat
curl -X POST ... -d '{"mode":"auto","query":"你好，你会做什么","session_id":"t1"}'

# 4. 跨路径记忆：先问 1，再问"我刚问的产品叫什么名字"
```

## 7. 已知限制（诚实清单）

1. **Agent 路径无对话记忆**：ReAct 的 prompt 里没拼历史（拼了会破坏格式约束），
   追问类问题靠意图分类落到 RAG/chat 路径兜住
2. **意图分类增加 5-8s 首响**：多一次完整 LLM 往返。优化方向：关键词规则前置
   （含产品名→直接 knowledge），只有不确定时才问 LLM
3. **记忆还在内存里**：重启丢失，生产要换 Redis（见 product 路线图）

## 8. 面试一句话

> "最后我把 5 个独立模式合并成一个智能客服：LLM 意图分类决定走 RAG、
> Agent 还是直答，所有路径共享滑动窗口记忆和流式输出。其中滑动窗口
> 就是解决长对话上下文膨胀——不限历史会撑爆 token 且模型会忽略中段内容。"
