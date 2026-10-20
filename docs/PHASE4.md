# 📘 Phase 4 学习文档：多轮对话（Memory）

> **本阶段目标**：让 LLM **记住之前聊过什么**，实现真正的"对话"——而不是每次都"第一次见面"。

---

## 🎯 学习目标 Checklist

完成本阶段后，你应该能：

- [ ] 解释为什么 LLM 默认没有记忆
- [ ] 用 `InMemoryChatMessageHistory` 存对话历史
- [ ] 用 `MessagesPlaceholder` 在 prompt 里插入历史
- [ ] 用 `RunnableWithMessageHistory` 包装带记忆的链
- [ ] 理解 `session_id` 的作用（同一会话连续 / 不同会话隔离）
- [ ] 区分 `RunnableWithMessageHistory`（已弃用）和 LangGraph（推荐方向）

---

## 📚 核心概念（5 个）

### 概念 1：LLM 是"金鱼记忆"

**为什么没记忆？**  
每次 `chain.invoke({"input": "..."})`，**整个调用是独立的**——模型看不到之前的输入和输出。就像你和**金鱼**对话：你说啥它都热情接，但根本记不住上一秒说过啥。

**对比**：

| 场景 | 第一次调用 | 第二次调用 |
|---|---|---|
| 无记忆 | "我电脑蓝屏了" | "重启没用" |
| 模型回答 | 给出建议 | **一脸懵**："什么重启没用？" |
| 有记忆 | "我电脑蓝屏了" | "重启没用" |
| 模型回答 | 给出建议 | **知道上文是蓝屏** → 进阶建议 |

---

### 概念 2：`InMemoryChatMessageHistory` —— 历史记录本

**是什么**：把对话消息存在内存里（一个 Python list）。

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

# 创建一个"聊天本"
history = InMemoryChatMessageHistory()

# 添加消息
history.add_user_message("你好")  # 用户说
history.add_ai_message("你好呀！")  # AI 答

# 拿到全部消息
print(history.messages)
# [HumanMessage(content='你好'), AIMessage(content='你好呀！')]
```

**生产环境注意**：内存是临时的，进程重启就没了。真实项目用 **Redis / PostgreSQL / MongoDB**。

---

### 概念 3：`MessagesPlaceholder` —— Prompt里的"历史插槽"

**是什么**：prompt 模板里专门留个"位置"，运行时把历史消息自动塞进去。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手"),    # 系统提示（固定）
    MessagesPlaceholder(variable_name="history"),  # ← 历史插槽（动态）
    ("human", "{input}"),                        # 当前用户消息
])

# 运行时：
chain.invoke({
    "history": [
        HumanMessage(content="之前说过的话"),
        AIMessage(content="之前的回答"),
    ],
    "input": "现在的问题"
})
```

**关键**：变量名要和后面 `RunnableWithMessageHistory` 的 `history_messages_key` 对应。

---

### 概念 4：`RunnableWithMessageHistory` —— "记忆魔盒"

**是什么**：把任意普通链"包装"成"自动管理历史"的链。

**最小代码模板**：

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. 先有普通链
chain = prompt | llm | StrOutputParser()

# 2. 写一个函数：给定 session_id，返回对应的 history 对象
def get_session_history(session_id):
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]

# 3. 包装
with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",      # 用户输入在字典里的 key
    history_messages_key="history",   # prompt 模板里 MessagesPlaceholder 的 variable_name
)

# 4. 调用时必须传 session_id
with_history.invoke(
    {"input": "你好"},
    config={"configurable": {"session_id": "user_001"}}
)
```

**4 个关键参数**：
| 参数 | 作用 |
|---|---|
| `chain` | 被包装的原始链 |
| `get_session_history` | 取历史的函数（按 session_id）|
| `input_messages_key` | 用户输入的字典 key（一般是 "input"） |
| `history_messages_key` | 历史在 prompt 里占位符的变量名 |

---

### 概念 5：`session_id` —— 谁在和谁聊

**核心思想**：每个"用户"或"会话"用一个 ID 区分，ID 相同 →共享历史，ID 不同 →独立历史。

```python
# 用户 A 第 1 轮
with_history.invoke({"input": "我叫小明"}, config={"configurable": {"session_id": "A"}})

# 用户 A 第 2 轮（接上文）
with_history.invoke({"input": "我叫什么？"}, config={"configurable": {"session_id": "A"}})
# AI: "你叫小明" ✅

# 用户 B 第 1 轮（不同 session_id，历史隔离）
with_history.invoke({"input": "我叫什么？"}, config={"configurable": {"session_id": "B"}})
# AI: "我不知道你叫什么" ✅
```

**实战中**：
- Web 应用用 session ID 或 user ID
- 客服系统用工单 ID
- 多租户系统用 tenant ID

---

## 🏗️ 实战项目：wisdom.py 的演化

### 本阶段改了哪些代码

#### ① 加导入

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
```

#### ② 会话存储

```python
_session_store: dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]
```

> 💡 **小技巧**：用 dict 模拟数据库。生产用 Redis 时，把 dict 换成 Redis 客户端即可。

#### ③ 带历史的 prompt

```python
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的技术客服助手，能记住用户之前说过的话..."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chat_chain = chat_prompt | llm | StrOutputParser()
```

#### ④ 包装成"带记忆"的链

```python
chat_with_history = RunnableWithMessageHistory(
    chat_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

#### ⑤ 封装成易用的函数

```python
def chat_session(session_id: str, user_input: str) -> str:
    config = {"configurable": {"session_id": session_id}}
    response = chat_with_history.invoke(
        {"input": user_input},
        config=config
    )
    return response
```

### 跑通结果（user1 的 4 轮对话）

```
轮 1: "我的电脑蓝屏了，怎么办？" → 一般建议
轮 2: "重启没用"               → 知道上文，进阶建议
轮 3: "那我该怎么办？"         → 接住前两轮
轮 4: "那我该准备多少钱维修？" → 记得是修电脑，报维修费
```

**完美证明**：AI "记住"了上文。

---

## 🧪 动手练习

### 必做（3 个）

#### 练习 1：让 AI 自我介绍并验证
**目标**：体会"AI 真的记住"和"AI 装记住"的区别。

```python
chat_session("test", "我叫张三，今年 30 岁，是程序员")
chat_session("test", "我叫什么名字？")  # 应该答"张三"
chat_session("test", "我多大了？")      # 应该答"30"
```

#### 练习 2：会话隔离验证
**目标**：确认不同 session_id 不串台。

```python
chat_session("A", "我喜欢吃苹果")
chat_session("B", "我喜欢吃什么？")  # 应该答不知道
chat_session("A", "我喜欢吃什么？")  # 应该答"苹果"
```

#### 练习 3：故意超长对话看 token 消耗
**目标**：体验历史过长时的成本问题。

```python
# 连问 20 轮，看 token 是不是越来越多
for i in range(20):
    chat_session("long", f"这是第{i+1}轮，我说了些啥？")
```

**思考**：第 20 轮的 prompt 比第 1 轮**长 20 倍**，token 费用也是 20 倍。这引出了**生产环境的优化话题**（截断、摘要）。

### 选做（2 个）

#### 选做 1：用 Redis 替换内存存储
**目标**：让历史"重启进程不丢"。

```python
# pip install redis
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_session_history(session_id):
    return RedisChatMessageHistory(session_id, url="redis://localhost:6379")
```

#### 选做 2：限制历史长度
**目标**：防止 token 爆炸。

```python
from langchain_core.messages import trim_messages

trimmer = trim_messages(max_tokens=500, strategy="last")  # 只保留最近 500 token
# 在 prompt 前 trim 前加 trimmer
```

---

## ⚠️ 常见 Bug 速查

| 现象 | 原因 | 修法 |
|---|---|---|
| `KeyError: 'session_id'` | invoke 时没传 `config` | 必须传 `config={"configurable": {"session_id": "xxx"}}` |
| AI 不"记住" | `MessagesPlaceholder` 变量名和 `history_messages_key` 不一致 | 名字要对上 |
| 不同 session 串台 | 没区分 session_id | 每次 invoke 传不同的 session_id |
| `name 'ChatMessageHistory' is not defined` | 导入路径错了 | 用 `from langchain_core.chat_history import InMemoryChatMessageHistory` |
| DeprecationWarning: RunnableWithMessageHistory | 已被弃用 | 仍可使用；生产推荐用 LangGraph |
| 历史无限增长，token 暴涨 | 没限制长度 | 用 `trim_messages` 截断 |

---

## ✅ 自我检测（8 题）

### 判断题（4 题）

1. ⬜ LLM 默认记得所有之前的对话
2. ⬜ 同一个 `session_id` 的多次 invoke 会共享历史
3. ⬜ `MessagesPlaceholder` 必须在 prompt 里有 `human` 消息之前
4. ⬜ 进程重启后内存里的历史会丢失

### 代码题（4 题）

**题 1**：补全 get_session_history
```python
_store = {}
def get_session_history(session_id):
    if session_id ??? in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[???]
```

**题 2**：补全 RunnableWithMessageHistory 配置
```python
with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key=???,
    history_messages_key=???,
)
```

**题 3**：补全 invoke 调用
```python
response = with_history.invoke(
    {"input": "你好"},
    config={"configurable": {"???": "user1"}}
)
```

**题 4**：下面代码有什么问题？
```python
# 用户 A
with_history.invoke({"input": "我喜欢红色"}, config={"configurable": {"session_id": "A"}})
# 用户 B
response = with_history.invoke(
    {"input": "我喜欢什么颜色？"},
    # ← 漏了 config
)
# 期望：B 答"不知道"；实际：会报错
```

---

## 📌 速查卡

### 3 个核心组件
```python
from langchain_core.chat_history import InMemoryChatMessageHistory      # 历史存储
from langchain_core.prompts import MessagesPlaceholder                  # 历史插槽
from langchain_core.runnables.history import RunnableWithMessageHistory # 包装器
```

### 4 步搭建多轮对话
```python
# 1. 历史存储
_store = {}
def get_session_history(sid):
    if sid not in _store:
        _store[sid] = InMemoryChatMessageHistory()
    return _store[sid]

# 2. 带历史插槽
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是..."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 3. 基础链
chain = prompt | llm | StrOutputParser()

# 4. 包装
with_history = RunnableWithMessageHistory(
    chain, get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

### 调用模板
```python
response = with_history.invoke(
    {"input": "用户消息"},
    config={"configurable": {"session_id": "user_001"}}
)
```

---

## 🎓 本阶段总结

| 学到的 | 怎么检验 |
|---|---|
| LLM 无记忆的本质 | 能说出为什么 `invoke` 默认不记历史 |
| 三个核心组件 | 能独立写出 `MessagesPlaceholder` + `InMemoryChatMessageHistory` + `RunnableWithMessageHistory` 的组合 |
| session_id 隔离 | 能解释同一 session 共享、不同 session 隔离的原因 |
| invoke 必须传 config | 漏掉 `config` 时能预判会报错 |

---

## ⚠️ 关于 deprecation 警告

你跑代码时会看到：
```
LangChainDeprecationWarning: RunnableWithMessageHistory is deprecated.
Use LangGraph's built-in persistence instead.
```

**含义**：LangChain 团队已经把多轮对话的官方推荐实现迁到了 **LangGraph**（一个状态图框架）。

**怎么办**：
- ✅ **学习阶段**：继续用 `RunnableWithMessageHistory`，代码简单易懂
- 🚀 **生产项目**：考虑迁移到 LangGraph（更强大但学起来也复杂）

我们 Phase 4 用了"简单路线"，先学会多轮对话的核心思想。后续可选 Phase 4.5 介绍 LangGraph 写法。

---

## ➡️ Phase 5 预告：RAG 入门（检索增强生成）

**下一阶段目标**：让 LLM 能回答"**自己训练数据里没有的内容**"——比如公司内部文档、最新新闻、私人笔记。

**会学到的**：
- **Embedding**：把文字转成"语义坐标"（数字向量）
- **向量数据库**：按相似度搜索（FAISS / Chroma）
- **Retriever**：从库中捞相关文档
- RAG 链路：问题 → 检索 → 拼 prompt → 答

**Phase 5 文档生成条件**：你完成 Phase 5 的代码改造并跑通后，告诉我"开搞 Phase 5"，我会生成 `docs/PHASE5.md`。

---

## 📂 文件清单

本阶段相关文件：
- `wisdom.py`：新增 `_session_store`、`get_session_history`、`chat_prompt`、`chat_chain`、`chat_with_history`、`chat_session`
- `docs/PHASE4.md`：本学习文档