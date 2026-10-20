# 📘 Phase 1 学习文档：LangChain 基础与 LCEL 入门

> **本阶段目标**：从零开始，用 `wisdom.py` 这个项目，掌握 LangChain **最核心的 5 个概念**，能做"清洗→分类→专家回答→校验"的完整 AI 应用。

---

## 🎯 学习目标 Checklist

完成本阶段后，你应该能：

- [ ] 用一句话解释 LangChain 是什么、解决了什么问题
- [ ] 写出 `prompt | llm | parser` 这条最经典的链
- [ ] 用 `RunnableLambda` 把任意 Python 函数塞进链
- [ ] 用 `RunnablePassthrough.assign` 给数据字典加字段
- [ ] 理解"字典 key 匹配 = 链的接口协议"
- [ ] 独立设计一个 3-4 节的 AI 应用流水线

---

## 📚 核心概念（5 个）

### 概念 1：LCEL（LangChain Expression Language）

**是什么**：LangChain 的"管道操作符"语法，用 `|` 把 Runnable 串起来。

**类比**：和 Linux 的 `|` 完全一样：
```bash
cat file.txt | grep "hello" | wc -l
# 等价于：wc -l(grep("hello", cat(file.txt)))
```

```python
chain = prompt | llm | parser
# 等价于：parser(llm(prompt(input)))
```

**为什么这么设计**：让 AI 应用像写函数管道一样**一眼看出数据流向**。

---

### 概念 2：`prompt | llm | StrOutputParser()`

**是什么**：LangChain **最经典的三段式**，调一次大模型的"标准动作"。

**三段各自干啥的**：

| 段 | 组件 | 输入 | 输出 |
|---|---|---|---|
| 1 | `ChatPromptTemplate` | 字典 `{"input": "..."}` | `PromptValue`（拼好的提示词） |
| 2 | `llm`（大模型） | `PromptValue` | `AIMessage`（带元数据的复杂对象） |
| 3 | `StrOutputParser` | `AIMessage` | `str`（纯文字） |

**关键代码**：
```python
chain = (
    ChatPromptTemplate.from_template("问题：{input}\n回答：")
    | llm
    | StrOutputParser()
)

# 用法
chain.invoke({"input": "你好"})  # → "你好！有什么可以帮您？"
```

---

### 概念 3：Runnable 家族

**是什么**：LangChain 把所有可调用对象（Chain、函数、模型）都抽象成**统一接口 `Runnable`**。

**最常用的 4 种 Runnable**：

| 名字 | 干啥 | 用法 |
|---|---|---|
| `ChatPromptTemplate` | 模板填字 | 接字典，输出 PromptValue |
| `llm`（ChatOpenAI 等） | 调大模型 | 接 PromptValue，输出 AIMessage |
| `StrOutputParser` | 摘纯文本 | 接 AIMessage，输出 str |
| `RunnableLambda(函数)` | 包普通函数 | 接任意输入，输出函数返回值 |
| `RunnablePassthrough.assign(k=v)` | 数据字典加字段 | 保留原字典 + 加一个新 key |

**关键代码**：
```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# 把普通函数包装成 Runnable
def double(x):
    return x * 2

double_runnable = RunnableLambda(double)

# 给字典加新字段
chain = RunnablePassthrough.assign(doubled=lambda x: x["n"] * 2)
chain.invoke({"n": 5})  # → {"n": 5, "doubled": 10}
```

---

### 概念 4：数据流转 = 字典 key 匹配

**核心规则**：链之间传递的是**字典**（`dict`），下游想用啥字段，就在 prompt 模板里写对应的 `{字段名}` 占位符。

**示例**：
```python
chain = RunnablePassthrough.assign(greeting="你好")
# 输出 {"input": "xxx", "greeting": "你好"}

# 下游 prompt 模板里写 {input} 和 {greeting} 都能拿到
template = ChatPromptTemplate.from_template("{greeting}！{input}")
```

**常见坑**：
- 上游输出 key 是 `text`，下游模板写 `{answer}` → 拿不到值
- 模板有 `{input}`，上游字典没 `input` key → `KeyError`

---

### 概念 5：异常处理（用 `try/except` 短路）

**为什么需要**：有时候**我们根本不想让 LLM 跑**（比如输入是垃圾评论），应该在清洗阶段直接拦截，省 token、省时间。

**关键代码**：
```python
class InputRejected(Exception):
    pass

def clean_input(inputs):
    text = inputs["raw_input"]
    if not text:
        raise InputRejected("输入为空")
    return {"input": text}

def process_query(query):
    try:
        result = full_chain.invoke({"raw_input": query})
    except InputRejected as e:
        return f"已被过滤：{e}"
    return result["text"]
```

---

## 🏗️ 实战项目：从 `wisdom.py` 看完整 AI 应用

### 演化路线

```
v1: 原始版本（老 API，会报错）
        ↓
v2: LCEL 重写（核心架构）
        ↓
v3: 加 4 位专家子链（technical/historical/math/default）
        ↓
v4: 加路由分发（先分类，再调专家）
        ↓
v5: 加内容校验
        ↓
v6: 5 层清洗模块（标准化/去噪/敏感词/质量/业务）
        ↓
v7: 异常兜底 + .env 配置外置
```

### 最终架构图

```
用户输入 {"raw_input": "..."}
   ↓
[clean_input] 5层清洗 → 拦垃圾 / 输出 {"input": "..."}
   ↓
[router_chain]
   ├─ RunnablePassthrough.assign(category=LLM分类)
   └─ RunnableLambda(pick_and_invoke)
       ├─ category="technical" → technical_chain
       ├─ category="historical" → historical_chain
       ├─ category="math" → math_chain
       └─ 其他 → default_chain
   ↓
{"input": "...", "category": "...", "text": "专家回答"}
   ↓
[validation_chain]
   └─ RunnablePassthrough.assign(validation_result=LLM校验)
   ↓
{"input": "...", "category": "...", "text": "...", "validation_result": "[有效]"}
   ↓
[process_query] 业务封装：返回 text 或人工兜底话术
```

### 关键代码片段（建议背下来）

#### ① 基础链写法
```python
technical_chain = (
    ChatPromptTemplate.from_template(
        "作为技术专家回答：{input}"
    )
    | llm
    | StrOutputParser()
)
```

#### ② 路由链写法
```python
router_chain = (
    RunnablePassthrough.assign(
        category=router_prompt | llm | StrOutputParser()
    )
    | RunnableLambda(pick_and_invoke)
)
```

#### ③ 拼接完整流程
```python
full_chain = (
    RunnableLambda(clean_input)
    | router_chain
    | validation_chain
)
```

#### ④ 业务封装
```python
def process_query(query: str) -> str:
    try:
        result = full_chain.invoke({"raw_input": query})
    except InputRejected as e:
        return f"输入已被过滤：{e}"
    return result["text"] if "有效" in result["validation_result"] else "转人工"
```

---

## 🧪 动手练习

### 必做（3 个）

#### 练习 1：加第 5 个专家
**目标**：掌握"复制粘贴一个现有专家，改3 处"的模式。

**步骤**：
1. 在 wisdom.py 里加一条 `translation_chain`（翻译专家）
2. 在 `router_prompt` 里加一条分类规则
3. 在 `pick_and_invoke` 里加一个 elif

**验收**：问"翻译 Hello to Chinese"，分类 = `translation`，回答 = 中文翻译。

#### 练习 2：故意制造 Bug 并修复
**目标**：练排查能力。

**步骤**：
1. 把 `technical_chain` 的 `| llm | StrOutputParser()` 删掉
2. 跑代码，看哪里出错
3. 加 `print(type(result['text']))` 调试
4. 改回来

#### 练习 3：改 temperature 观察效果
**目标**：理解 temperature 参数的实际影响。

**步骤**：
1. 设 `temperature=0`，跑同一个问题 3 次，看回答是否一致
2. 设 `temperature=1.0`，跑同一个问题 3 次，看回答是否变化
3. 记录你的发现

### 选做（2 个）

#### 选做 1：把温度改成 prompt 的一部分
**目标**：把"温度"从代码移到 prompt，体验 prompt 工程的乐趣。

#### 选做 2：拆分清洗模块到独立文件
**目标**：学模块化。把 `clean_input` 那 5 层函数挪到 `cleaner.py`，在 `wisdom.py` 里 `from cleaner import clean_input`。

---

## ⚠️ 常见 Bug 速查

| 报错 | 原因 | 修法 |
|---|---|---|
| `ModuleNotFoundError: langchain.chains` | 用老 API 了 | 改用 LCEL：`prompt \| llm \| parser` |
| `ModuleNotFoundError: langchain_community` | 没装这个包 | `pip install langchain-community` |
| `ImportError: ChatOpenAI from langchain_community.chat_models` | OpenAI 集成被拆走了 | `pip install langchain-openai` + `from langchain_openai import ChatOpenAI` |
| `OpenAIError: Missing credentials` | `.env` 变量名对不上 | 代码读 `.env` 里**实际有的** key 名 |
| `UnicodeEncodeError: 'gbk' codec` | Windows 终端 GBK | 加 `sys.stdout.reconfigure(encoding="utf-8")` |
| 回答总是 `[无效]` | `xxx_chain` 漏了 `\| llm \| parser` | 补全三段 |
| `KeyError: 'input'` | 字典流转断 | 检查上游是否产出 `input` key |

---

## ✅ 自我检测（10 题）

### 判断题（5 题）

1. ⬜ `prompt | llm | parser` 中，`parser` 可以省略，因为 LLM 直接返回的就是字符串
2. ⬜ `RunnableLambda` 可以把任何 Python 函数包装成 Runnable
3. ⬜ RunnablePassthrough.assign 加的新字段**会覆盖**原字典中同名的 key
4. ⬜ SequentialChain 是新版 LangChain 推荐的多链组合方式
5. ⬜ 字典里的 key 名和 prompt 模板里的 `{xxx}` 占位符**必须严格匹配**

### 代码题（5 题，难度递增）

**题 2.1**：补全这条链
```python
chain = (
    ChatPromptTemplate.from_template("...")
    ??? llm
    ??? StrOutputParser()
)
```

**题 2.2**：判断下面代码是否正确并说明原因
```python
def my_func(state):
    return {"output": state["input"] * 2}

chain = (
    {"input": RunnablePassthrough()}
    | RunnableLambda(my_func)
)
# 问：my_func 接收到的 state 是什么类型？输出是什么类型？
```

**题 2.3**：补全代码，让下游能拿到 `category` 字段
```python
chain = (
    RunnablePassthrough.assign(
        ???=router_prompt | llm | StrOutputParser()
    )
)
```

**题 2.4**：指出下面代码的 Bug
```python
technical_chain = (
    ChatPromptTemplate.from_template("问题：{question}")
    | llm
    | StrOutputParser()
)

# 调用时
technical_chain.invoke({"question": "蓝屏怎么办"})
```

**题 2.5**：写出"清洗→分类→专家→校验"的完整流水线（伪代码即可）

---

## 📌 速查卡（贴在桌边随时翻）

### 3 个导入
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
```

### 4 个 Runnable
```python
# 1. 模板
ChatPromptTemplate.from_template("问题：{input}")

# 2. 模型
ChatOpenAI(base_url=..., api_key=..., model=..., temperature=...)

# 3. 解析器
StrOutputParser()

# 4. 函数包装器
RunnableLambda(my_function)

# 5. 字典加字段
RunnablePassthrough.assign(new_key=some_runnable)
```

### 5 个调用方式
```python
chain.invoke(input)            # 同步：等全部完成返回
chain.stream(input)            # 流式：一个块一个块返回
chain.batch([input1, input2])  # 批量：多个一起跑
chain.ainvoke(input)           # 异步：async 函数里用
chain.astream(input)           # 异步流式
```

---

## 🎓 本阶段总结

| 学到的 | 怎么检验 |
|---|---|
| LangChain 定位 | 能用 1 句话说清 LangChain 解决了什么问题 |
| LCEL 写法 | 能不看文档写出 `prompt \| llm \| parser` |
| Runnable 类型 | 能说出 5 个 Runnable 的作用 |
| 数据流转 | 看到 `KeyError` 知道是字典 key 没对上 |
| 异常处理 | 能在链里加自定义异常 + try/except |

---

## ➡️ Phase 2 预告：流式输出

**下一阶段目标**：让 LLM 回答**一个字一个字"流"出来**（像 ChatGPT 那种打字机效果）。

**会学到的**：
- `chain.stream()` 替代 `chain.invoke()`
- Python 异步迭代器（`AsyncIterator`）
- SSE（Server-Sent Events）原理

**Phase 2 文档生成条件**：你完成 Phase 2 的代码改造并跑通后，告诉我"开搞 Phase 2"，我会生成 `docs/PHASE2.md`。

---

## 📂 文件清单

本阶段相关文件：
- `wisdom.py`：实战代码
- `.env`：配置文件（LLM_API_KEY 等）
- `docs/PHASE1.md`：本学习文档

**下一阶段会新增**：
- `docs/PHASE2.md`：流式输出学习文档
- 可能改动 `wisdom.py`（process_query 函数）