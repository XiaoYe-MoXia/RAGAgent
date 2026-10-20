# 📘 Phase 3 学习文档：结构化输出（Structured Output）

> **本阶段目标**：让 LLM **直接返回 Python 对象**（带类型的、可校验的），而不是一段自由文本字符串。

---

## 🎯 学习目标 Checklist

完成本阶段后，你应该能：

- [ ] 用 `BaseModel` 定义一个数据模型
- [ ] 用 `Field()` 给字段加描述、范围、长度等约束
- [ ] 用 `PydanticOutputParser` 让 LLM 返回结构化对象
- [ ] 区分 `with_structured_output()` 和 `PydanticOutputParser` 的适用场景
- [ ] 用 `model_dump_json()` 把 Pydantic 对象转 JSON 字符串
- [ ] 解释为什么推理模型（deepseek-r1）有时不按 JSON 格式输出

---

## 📚 核心概念（6 个）

### 概念 1：为什么需要结构化输出？

LLM 默认返回的是**自由文本**，实际业务经常需要**结构化数据**。

**痛点对比**：

| 场景 | 字符串输出（痛苦） | 结构化输出（舒适） |
|---|---|---|
| 评分 | 解析"评分：5星"正则，可能漏 | `result.score == 5` |
| 分类 | 用 `if "好评" in text` | `result.sentiment == "positive"` |
| 多值 | 字符串解析拆分关键词 | `result.keywords == [...]` |
| 入库 | 写一堆正则 | 直接 `.model_dump()` |

---

### 概念 2：Pydantic 是什么？

**Pydantic** = Python 的"数据建模 + 自动校验"库。LangChain 用它来定义 LLM 的输出 schema。

**两个核心概念**：

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str                              # 必填字符串
    age: int = Field(ge=0, le=150)         # 整数，0-150
    email: str = Field(default="")         # 默认值
    hobbies: list[str] = Field(default=[]) # 默认空列表
```

- `BaseModel` = 数据类的"模板"
- `Field()` = 给字段加约束和说明（`ge`/`le`/`max_length`/`description`）
- 自动校验：传入数据如果不符合类型/约束，会抛 `ValidationError`

---

### 概念 3：`PydanticOutputParser`（经典写法）

**是什么**：LangChain 自带的解析器，把 LLM 的 JSON 输出转成 Pydantic 对象。

**核心三步**：

```python
from langchain_core.output_parsers import PydanticOutputParser

# 1. 定义 schema
parser = PydanticOutputParser(pydantic_object=MyModel)

# 2. 拿到格式说明（自动生成的 JSON schema 文本）
print(parser.get_format_instructions())
# 输出：
# "The output should be formatted as a JSON instance that conforms to
#  the JSON schema below. ..."

# 3. 把格式说明塞进 prompt
chain = (
    ChatPromptTemplate.from_template("分析：{input}\n\n{format_instructions}")
    .partial(format_instructions=parser.get_format_instructions())
    | llm
    | parser
)
```

**关键技巧**：`.partial(format_instructions=...)` 把格式说明**提前注入**，prompt 模板里只用写 `{input}`。

---

### 概念 4：`with_structured_output()`（现代写法）

**是什么**：LangChain 的"便捷方法"，一行代码搞定结构化输出。

```python
# 写法 A：现代方法
chain = prompt | llm.with_structured_output(MyModel)

# 写法 B：经典方法（更稳）
parser = PydanticOutputParser(pydantic_object=MyModel)
chain = prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser
```

**两种写法对比**：

| | `with_structured_output` | `PydanticOutputParser` |
|---|---|---|
| 代码量 | 少一行 | 多两步 |
| 底层 | 优先用 tool calling | 让 LLM 自己输出 JSON |
| 推理模型兼容 | ❌ 经常不灵 | ✅ 更稳 |
| 控制力 | 低 | 高（你能看到完整 prompt）|

> 💡 **实战建议**：用 deepseek-r1 这类**推理模型**时，**优先用经典写法**。

---

### 概念 5：常见 Field 约束

```python
from pydantic import BaseModel, Field
from typing import Literal, List

class ProductReview(BaseModel):
    rating: int = Field(description="评分", ge=1, le=5)              # 范围约束
    title: str = Field(description="标题", max_length=50)             # 最大长度
    tags: List[str] = Field(description="标签", max_length=3)         # 数组长度
    category: Literal["电子", "服装", "食品"] = Field(...)             # 枚举
    description: str = Field(default="暂无", description="默认描述")    # 默认值
```

| 约束 | 作用 |
|---|---|
| `ge=1, le=5` | 数值范围 |
| `max_length=50` | 字符串最大长度 |
| `min_length=1` | 字符串最小长度 |
| `max_length=3` | 列表最大元素数 |
| `Literal["a","b"]` | 枚举值（必须这几个之一） |
| `default="..."` | 默认值（可选字段） |
| `description="..."` | 给 LLM 看的字段说明（关键！）|

---

### 概念 6：`model_dump_json()` 的作用

```python
result = CommentAnalysis(is_ad=False, sentiment="positive", score=5, ...)

# 转字典
result.model_dump()
# {'is_ad': False, 'sentiment': 'positive', 'score': 5, ...}

# 转 JSON 字符串
result.model_dump_json(indent=2, ensure_ascii=False)
# '{\n  "is_ad": false,\n  "sentiment": "positive",\n  "score": 5,\n  ...}'

# 简化版：直接打印对象
print(result)  # 默认输出 JSON 字符串（不一定美观）
```

> `model_dump_json` 是 Pydantic v2 的方法，v1 是 `json()`。

---

## 🏗️ 实战项目：wisdom.py 的演化

### 本阶段改了哪些代码

#### ① 加 Pydantic 导入

```python
from pydantic import BaseModel, Field
from typing import Tuple, List, Literal
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
```

#### ② 定义 `CommentAnalysis` 模型

```python
class CommentAnalysis(BaseModel):
    """用户评论的结构化分析模型"""
    is_ad: bool = Field(description="是否为广告/营销引流类评论")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="情感倾向：positive(好评) / negative(差评) / neutral(中性)"
    )
    score: int = Field(description="用户满意度评分 1-5 星", ge=1, le=5)
    keywords: List[str] = Field(
        description="评论中提到的关键主题词（最多 3 个）",
        max_length=3
    )
    summary: str = Field(description="一句话总结这条评论（不超过 50 字）", max_length=50)
```

#### ③ 构建分析链（**走经典写法，对推理模型更稳**）

```python
parser = PydanticOutputParser(pydantic_object=CommentAnalysis)

analysis_chain = (
    ChatPromptTemplate.from_template(
        "请分析下面这条用户评论。\n\n"
        "评论：{input}\n\n"
        "{format_instructions}"   # ← 关键：手动注入 JSON schema 说明
    ).partial(format_instructions=parser.get_format_instructions())
    | llm
    | parser   # ← Pydantic 解析器
)
```

#### ④ 封装 `analyze_comment` 函数

```python
def analyze_comment(text: str) -> CommentAnalysis:
    print(f"\n====== 评论结构化分析 ======")
    print(f"原始文本: {text}")
    try:
        result = analysis_chain.invoke({"input": text})
        print(result.model_dump_json(indent=2, ensure_ascii=False))
        return result
    except Exception as e:
        print(f"分析失败：{e}")
        raise
```

### 为什么没走 `with_structured_output`？

我们**先试了** `llm.with_structured_output(CommentAnalysis, method="json_mode")`，结果：

```
❌ langchain_core.exceptions.OutputParserException: Invalid json output
   （deepseek-r1 返回了一坨 markdown 表格）
```

**原因**：推理模型（带"思考"过程）不严格遵守"只能输出 JSON"的指令，反而爱用 markdown 表格展示。

**修法**：改用经典 `PydanticOutputParser`，手动把 JSON schema 说明注入 prompt。

> 💡 **经验**：对推理模型（deepseek-r1、QwQ 等），**永远优先用经典写法**。

---

## 🧪 动手练习

### 必做（3 个）

#### 练习 1：加一个新字段
**目标**：体验 `Field` 约束的作用。

**步骤**：
1. 在 `CommentAnalysis` 里加 `urgency: Literal["low", "medium", "high"]` 字段
2. 加字段说明、限制取值
3. 重新跑测试，看 LLM 是否按新字段输出

#### 练习 2：故意违反约束看报错
**目标**：理解 Pydantic 校验的作用。

**步骤**：
1. 临时把 `score` 改成 `Literal[1,2,3,4,5]`（枚举）
2. 让 LLM 分析一个特别糟的评论
3. 如果 LLM 输出 `score=0`，会报 `ValidationError`

#### 练习 3：用 `result.model_dump()` 入库
**目标**：把结构化输出用到实际场景。

**步骤**：
1. 拿到 `result = analyze_comment(text)`
2. `print(result.model_dump())` 看返回的字典
3. 想一想：怎么用 `model_dump()` 把数据存到数据库？

### 选做（2 个）

#### 选做 1：嵌套模型
**目标**：体验"模型套模型"。

```python
class ProductInfo(BaseModel):
    name: str
    category: str

class CommentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    product: ProductInfo   # ← 嵌套！
```

#### 选做 2：跑两条评论做对比
**目标**：把分析结果用 pandas/表格展示。

```python
import pandas as pd
comments = ["...", "..."]
results = [analyze_comment(c) for c in comments]
df = pd.DataFrame([r.model_dump() for r in results])
print(df)
```

---

## ⚠️ 常见 Bug 速查

| 现象 | 原因 | 修法 |
|---|---|---|
| `OutputParserException: Invalid json output` | 推理模型返回 markdown 而非 JSON | 用 `PydanticOutputParser` 经典写法 |
| `ValidationError: value is not a valid integer` | LLM 返回的字段类型不对 | 加强 prompt 或加 `Field` 约束 |
| `Field required` 报错 | 模型漏了某个必填字段 | 把字段改成可选 `default=...`，或增强 prompt |
| 字段值越界（如 score=10） | LLM 没遵守 `ge/le` 约束 | 经典解析器会自动校验，不通过就报错 |
| LLM 输出 `keywords: ["很多","很多"]` | 没限制 `max_length` | 加 `max_length=3` |

---

## ✅ 自我检测（8 题）

### 判断题（4 题）

1. ⬜ Pydantic 的 `BaseModel` 和普通 `@dataclass` 效果一样
2. ⬜ `with_structured_output()` 对所有模型（包括推理模型）都好用
3. ⬜ `Field(description="...")` 是给 **LLM 看的**提示，不是给用户看的
4. ⬜ `Literal["a","b","c"]` 让字段只能是这三个值之一

### 代码题（4 题）

**题 1**：补全 Pydantic 模型
```python
class Order(BaseModel):
    order_id: ???  # 必填字符串
    amount: ???    # 浮点数，必须 ≥0
    status: ???    # 只能是 "pending" / "paid" / "shipped"
```

**题 2**：补全分析链
```python
from langchain_core.output_parsers import PydanticOutputParser

parser = ???(pydantic_object=Order)
chain = (
    ChatPromptTemplate.from_template("分析订单：{input}\n{???}")
    .partial(format_instructions=parser.???())
    | llm
    | parser
)
```

**题 3**：判断下面代码会输出什么类型？
```python
result = chain.invoke({"input": "..."})
print(type(result))
```

**题 4**：补全代码，把 Pydantic 对象转成 JSON 字符串
```python
result = analyze_comment("...")
json_str = result.???(indent=2, ensure_ascii=False)
print(json_str)
```

---

## 📌 速查卡

### 3 个导入
```python
from pydantic import BaseModel, Field
from typing import Literal, List
from langchain_core.output_parsers import PydanticOutputParser
```

### 2 种结构化输出写法
```python
# 写法 A：现代（对推理模型不稳）
chain = prompt | llm.with_structured_output(MyModel)

# 写法 B：经典（对任何模型都稳）
parser = PydanticOutputParser(pydantic_object=MyModel)
chain = (
    ChatPromptTemplate.from_template("...{input}...\n{format_instructions}")
    .partial(format_instructions=parser.get_format_instructions())
    | llm
    | parser
)
```

### 5 个常用 Field 约束
```python
Field(default="未知")          # 默认值
Field(description="给LLM的提示") # 字段说明（重要！）
Field(ge=0, le=100)            # 数值范围
Field(max_length=50)           # 字符串/列表长度上限
Field(min_length=1)            # 字符串/列表长度下限
```

### 3 个对象方法
```python
result.model_dump()        # 转字典
result.model_dump_json()   # 转 JSON 字符串
result.model_dump_json(indent=2, ensure_ascii=False)  # 美观打印
```

---

## 🎓 本阶段总结

| 学到的 | 怎么检验 |
|---|---|
| Pydantic 模型 | 能独立写一个带约束的 BaseModel |
| 字段约束 | 看到 `score: int = Field(ge=1, le=5)` 能说出含义 |
| 两种结构化写法 | 能说出 `with_structured_output` 和 `PydanticOutputParser` 的适用场景 |
| 推理模型兼容 | 能解释为啥对 deepseek-r1 优先用经典写法 |
| 序列化方法 | 知道 `model_dump` / `model_dump_json` 的区别 |

---

## ➡️ Phase 4 预告：多轮对话（Memory）

**下一阶段目标**：让 AI **记住之前聊过什么**，实现真正的对话。

**会学到的**：
- `ChatMessageHistory`：聊天记录管理
- `MessagesPlaceholder`：在 prompt 里插入历史消息
- 同会话/跨会话隔离

**Phase 4 文档生成条件**：你完成 Phase 4 的代码改造并跑通后，告诉我"开搞 Phase 4"，我会生成 `docs/PHASE4.md`。

---

## 📂 文件清单

本阶段相关文件：
- `wisdom.py`：新增 `CommentAnalysis` 模型、`analysis_chain`、`analyze_comment` 函数
- `docs/PHASE3.md`：本学习文档