# 📘 Phase 2 学习文档：流式输出（Streaming）

> **本阶段目标**：把 `wisdom.py` 从"等半天整段出来"升级到"一字一字流出来"，体验 ChatGPT 那种"打字机效果"。

---

## 🎯 学习目标 Checklist

完成本阶段后，你应该能：

- [ ] 说出 `invoke()` 和 `stream()` 的本质区别
- [ ] 用 `chain.stream()` 自己写一个流式循环
- [ ] 理解为什么**不是所有链都该流式**
- [ ] 解释 `flush=True` 在流式输出中的作用
- [ ] 独立判断一个场景该用 `invoke` 还是 `stream`

---

## 📚 核心概念（4 个）

### 概念 1：`invoke()` vs `stream()` 的本质

**是什么**：LangChain 的链都支持 4 种调用方式：`invoke` / `stream` / `batch` / `ainvoke`，本质区别是**"返回节奏"**。

| 方式 | 返回 | 适用场景 |
|---|---|---|
| `invoke(input)` | 等全部完成，返回最终结果 | 短文本、批处理、内部判断 |
| `stream(input)` | 立刻开始，边生成边吐块 | 长文本、用户交互 |
| `batch([input1, input2])` | 多个一起跑，返回列表 | 离线批处理 |
| `ainvoke(input)` | 异步版本 | Web 服务器高并发 |

**类比**：
- `invoke` = 餐厅做好菜再上桌
- `stream` = 火锅（边涮边吃）
- `batch` = 一次性点10 桌
- `ainvoke` = 多个服务员同时干活

---

### 概念 2：`stream()` 返回迭代器

**是什么**：`chain.stream()` 返回的是一个**迭代器**（Iterator），每次 `next()` 拿到一个 chunk（模型新生成的一段文字）。

```python
chain = prompt | llm | StrOutputParser()

# stream 返回迭代器
text_stream = chain.stream({"input": "..."})

# 用 for 循环逐块取出
for chunk in text_stream:
    print(chunk, end="", flush=True)
```

**关键点**：
- 不是先返回完整字符串再切片
- 是真的**模型每生成一点，迭代器就吐一点**
- 你可以用 `flush=True` 立刻把内容刷到屏幕上（否则 Python 会"攒"够一行才显示）

---

### 概念 3：`flush=True` 的作用

**是什么**：Python 默认把 `print` 的内容**攒在内存缓冲区**，攒够一行或缓冲区满了才显示。流式输出场景下，这会让"打字机效果"变成"突然一坨"。

**对比**：

```python
# ❌ 错误：默认行为，最后才看到
for chunk in chain.stream({"input": "..."}):
    print(chunk, end="")   # 没 flush，可能最后才一次性显示

# ✅ 正确：每个 chunk 都立刻刷出来
for chunk in chain.stream({"input": "..."}):
    print(chunk, end="", flush=True)   # 立刻显示
```

**Web 场景下**（Flask / FastAPI）还需要 `yield` + `Content-Type: text/event-stream`。

---

### 概念 4：不是所有环节都该流式

**核心原则**：**只有"用户要看的长文本"才流式**，其它内部判断同步即可。

在我们的 `wisdom.py` 里：

| 环节 | 输出长度 | 是否流式 | 原因 |
|---|---|---|---|
| 清洗 | 无 LLM 调用 | 同步 | 纯函数，瞬间完成 |
| 分类 | 一个单词（"technical"） | 同步 | 太短，不值得流式 |
| **专家回答** | 长文本 | ✅ **流式** | 用户最关心的 |
| 校验 | "[有效]" | 同步 | 太短，且需要完整文本 |

**为什么全链一起流式不行？**
- 分类必须先完成才知道选哪个专家
- 校验必须先有完整回答才能判断
- 强行流式会出现"先吐分类、再吐专家、再吐校验"的混乱

> 💡 **架构原则**：把"需要用户等待的大段输出"单独 stream，其它保持 invoke。

---

## 🏗️ 实战项目：wisdom.py 的演化

### 本阶段改了哪些代码

#### ① 抽取校验模板（避免重复）

**改之前**：`validation_chain` 里硬编码 prompt，stream 版没法复用。

**改之后**：模板提取为模块常量：
```python
_VALIDATION_TEMPLATE = """请验证回答是否符合以下要求：
1. 事实准确性
2. 无主观臆断
3. 无商业广告内容
原问题：{input}
生成回答：{text}
验证结果（只需回答[有效]或[无效]）："""

# invoke 版用
validation_chain = RunnablePassthrough.assign(
    validation_result=ChatPromptTemplate.from_template(_VALIDATION_TEMPLATE)
    | llm | StrOutputParser()
)

# stream 版也用同一个模板
inner_validation = (
    ChatPromptTemplate.from_template(_VALIDATION_TEMPLATE)
    | llm
    | StrOutputParser()
)
```

> 💡 **小技巧**：模板字符串**单独提取**，多场景共用，避免复制粘贴。

---

#### ② 新增 `process_query_stream` 函数

**核心代码**（建议背下来）：

```python
def process_query_stream(query: str) -> str:
    print(f"\n====== 流式处理 ======")
    print(f"原始输入: {query}")

    # 1. 清洗（同步）
    try:
        cleaned = clean_input({"raw_input": query})
    except InputRejected as e:
        return f"输入已被过滤：{e}"
    print(f"清洗后: {cleaned['input']}")

    # 2. 分类（同步）
    classification_chain = router_prompt | llm | StrOutputParser()
    category = classification_chain.invoke({"input": cleaned["input"]}).strip().lower()
    print(f"分类标签: {category}")

    # 3. 挑专家
    if "technical" in category or "tech" in category:
        chosen = technical_chain
    elif "historical" in category or "history" in category:
        chosen = historical_chain
    elif "math" in category:
        chosen = math_chain
    else:
        chosen = default_chain

    # 4. 流式调用 ← 本节重点！
    print(f"\n回答（流式输出）: ", end="", flush=True)
    full_text = ""
    for chunk in chosen.stream({"input": cleaned["input"]}):
        print(chunk, end="", flush=True)  # ← flush=True 立刻刷屏
        full_text += chunk                  # ← 累积起来给校验用
    print()

    # 5. 校验（同步，因为需要完整文本）
    inner_validation = (
        ChatPromptTemplate.from_template(_VALIDATION_TEMPLATE)
        | llm
        | StrOutputParser()
    )
    validation = inner_validation.invoke({"input": cleaned["input"], "text": full_text})
    print(f"校验结果: {validation}")

    if "有效" in validation:
        return full_text
    else:
        return "回答需要人工复核，我们将在24小时内联系您"
```

---

## 🧪 动手练习

### 必做（3 个）

#### 练习 1：观察 chunk 长度
**目标**：理解每个 chunk 大概是多大。

**步骤**：
1. 把 `process_query_stream` 里改成：
   ```python
   for chunk in chosen.stream({"input": cleaned["input"]}):
       print(f"[chunk长度={len(chunk)}, 内容={chunk!r}]", flush=True)
       full_text += chunk
   ```
2. 跑一次，看每个 chunk 长啥样（一个汉字？一个词？一句话？）
3. **恢复原代码**

**思考题**：为什么不是每个汉字一个 chunk？

#### 练习 2：故意不加 `flush=True`
**目标**：理解 `flush=True` 的必要性。

**步骤**：
1. 把 `flush=True` 删掉，跑一次
2. 观察：是不是看到"突然一大坨"出来，而不是一字一字？

#### 练习 3：让分类也流式
**目标**：体会"不该流式的也流式"会怎样。

**步骤**：
1. 把分类那步也改成 stream，看效果
2. 思考：分类输出太短，stream 有什么意义？
3. **恢复原代码**

### 选做（2 个）

#### 选做 1：模拟打字机效果
**目标**：每个 chunk 之间加微小延迟，让"打字机"更逼真。

```python
import time
for chunk in chosen.stream({"input": ...}):
    print(chunk, end="", flush=True)
    time.sleep(0.05)  # 每个字停 0.05 秒
```

#### 选做 2：累计字数显示进度
**目标**：实时显示"已生成 100 字"，让用户知道模型还在干活。

```python
for chunk in chosen.stream({"input": ...}):
    full_text += chunk
    print(f"\r已生成 {len(full_text)} 字", end="", flush=True)
```

---

## ⚠️ 常见 Bug 速查

| 现象 | 原因 | 修法 |
|---|---|---|
| 看不到打字机效果，整段一起出 | 没加 `flush=True` | `print(chunk, end="", flush=True)` |
| `for chunk in chain.stream()` 报 `AttributeError` | 这个 Runnable 不支持 stream（比如某些自定义 RunnableLambda） | 用 `invoke()` 替代，或改用支持 stream 的组件 |
| 流式输出有"残影" | 之前 print 的内容没清掉 | 流式前加 `\r` 覆盖或单独打印 |
| 想给前端用 stream | `print` 是写到终端的 | 改用 Web 框架的 `yield` + SSE |
| 流式过程中异常中断 | 中途网络错误 | 加 `try/except` 捕获 + 重试逻辑 |

---

## ✅ 自我检测（8 题）

### 判断题（4 题）

1. ⬜ `chain.stream()` 会先返回完整字符串再切片
2. ⬜ `flush=True` 是 Python `print` 的默认行为
3. ⬜ 所有链都应该用 stream 才有好的用户体验
4. ⬜ LangChain 链支持 `invoke / stream / batch / ainvoke` 四种调用方式

### 代码题（4 题）

**题 1**：补全流式输出代码
```python
chain = prompt | llm | StrOutputParser()
for chunk in chain.???({"input": "..."}):
    print(chunk, ???="", ???=True)
```

**题 2**：指出下面代码的问题
```python
text = ""
for chunk in chain.stream({"input": "..."}):
    print(chunk)
    text = chunk  # ❌ 错！
```

**题 3**：判断下面哪种场景适合用 `stream()`？
- A. 客服系统给用户回复长答案
- B. 批量给 1000 条评论打分
- C. 让 AI 扮演路由器判断问题属于哪个分类
- D. 让 AI 写一篇 800 字的文章

**题 4**：补全代码，让它流式打印且累计文本
```python
full_text = ""
for chunk in chosen.stream(???):
    ???(chunk, end="", flush=True)
    ??? += chunk
```

---

## 📌 速查卡

### 4 种调用方式
```python
chain.invoke(input)            # 同步：等全部完成返回
chain.stream(input)            # 流式：迭代器，每个 chunk 一段
chain.batch([input1, input2])  # 批量：列表进，列表出
chain.ainvoke(input)           # 异步（async/await 场景）
chain.astream(input)           # 异步流式
```

### 流式最小代码模板
```python
for chunk in chain.stream({"input": "..."}):
    print(chunk, end="", flush=True)
print()  # 结束换行
```

### 流式 + 累计
```python
full_text = ""
for chunk in chain.stream({"input": "..."}):
    print(chunk, end="", flush=True)
    full_text += chunk
# full_text 现在是完整字符串，可以传给下游做校验/入库
```

---

## 🎓 本阶段总结

| 学到的 | 怎么检验 |
|---|---|
| stream vs invoke | 能说出本质区别（迭代器 vs 一次性返回）|
| flush 的作用 | 不加 `flush=True` 时能说出会出现什么现象 |
| 选择性流式 | 能判断哪些环节该流式、哪些不该 |
| 抽取模板复用 | 能独立把硬编码字符串改成模块常量 |

---

## ➡️ Phase 3 预告：结构化输出（Structured Output）

**下一阶段目标**：让 LLM **直接返回 Python 对象**（如 `{score: 8, tags: [...]}`），而不是一段字符串。

**会学到的**：
- `Pydantic` 模型（带类型校验的数据类）
- `chain.with_structured_output(Pydantic模型)`
- JSON 模式 vs 工具调用模式

**Phase 3 文档生成条件**：你完成 Phase 3 的代码改造并跑通后，告诉我"开搞 Phase 3"，我会生成 `docs/PHASE3.md`。

---

## 📂 文件清单

本阶段相关文件：
- `wisdom.py`：新增 `process_query_stream` 函数，抽取 `_VALIDATION_TEMPLATE`
- `docs/PHASE2.md`：本学习文档

**运行方式**：
```bash
# 跑全部测试（既有 invoke 也有 stream）
python wisdom.py

# 只跑流式部分：在 main 块的 stream_test_cases 里加测试问题
```