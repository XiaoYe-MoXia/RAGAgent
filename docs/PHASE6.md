# 📘 Phase 6 学习文档：Agent（智能体）

> **本阶段目标**：让 LLM **自己决定**——查不查、查几次、调哪个工具——而不是被写死的链式流程控制。

---

## 🎯 学习目标 Checklist

完成本阶段后，你应该能：
- [ ] 解释 Chain 和 Agent 的本质区别
- [ ] 用 `@tool` 装饰器把函数变成 LLM 能调的工具
- [ ] 解释 ReAct 模式的 Thought / Action / Observation 循环
- [ ] 用 `create_react_agent` + `AgentExecutor` 搭一个最小 Agent
- [ ] 让 Agent 自主决定"要不要调工具"、"调哪个"、"调几次"
- [ ] 解释为什么推理模型（deepseek-r1）+ ReAct 偶尔会输出解析警告

---

## 📚 核心概念（5 个）

### 概念 1：Chain vs Agent 的本质区别

| | Chain（链） | Agent（智能体） |
|---|---|---|
| 流程 | **固定**：A → B → C | **动态**：LLM 自己挑 |
| 调几次 LLM | 已知（写死在代码里）| 未知（LLM 自己挑）|
| 调哪些工具 | 无 | LLM 在多个工具里选 |
| 适用 | 流程固定的场景 | 复杂、不可预测的问题 |

**类比**：
- Chain = 工厂流水线（每个工位干啥固定）
- Agent = 一个能干的助理（你说"帮我搞定这个问题"，他自己想该查资料还是算一下）

---

### 概念 2：`@tool` 装饰器 —— 让函数变成 LLM 能"调"的东西

**是什么**：LangChain 用装饰器把 Python 函数包成 LLM 能看懂的工具。
LLM 拿到 3 个东西：
- **name**（函数名）
- **description**（函数 docstring）—— **至关重要**！LLM 用它决定"啥时候该调我"
- **args schema**（参数类型，自动从函数签名推断）

**最小代码**：
```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入必须是合法的 Python 算式字符串，例如 '100 * 0.2'"""
    return str(eval(expression))
```

**`@tool` 干了啥？**
- 读你的函数名 → 当工具名
- 读你的 docstring → 当工具描述
- 读你的类型注解 → 当参数 schema
- 包成 LangChain 的 `BaseTool` 对象

> 💡 **写好 docstring = 让 LLM 知道啥时候调你**。如果描述模糊，LLM 永远不知道用它。

---

### 概念 3：ReAct 模式（Reasoning + Acting）

Agent 的思考循环是 **ReAct = Reasoning（推理）+ Acting（行动）**：

```
Question:   咖啡机的萃取压力是多少？换算成 PSI 大约多少？
Thought:    我需要先查知识库拿到压力值（Bar），再用计算器换算成 PSI
Action:     search_knowledge_base
Action Input: 咖啡机萃取压力
Observation: 9 Bar ± 0.5

Thought:    知识库说 9 Bar。1 Bar ≈ 14.5 PSI，我算一下
Action:     calculator
Action Input: 9 * 14.5
Observation: 130.5

Thought:    我现在可以给出最终答案了
Final Answer: 9 Bar ≈ 130.5 PSI
```

**关键**：
- LLM **自己写 Thought**（思考过程）
- LLM **自己选 Action**（调哪个工具、传啥参数）
- LLM **自己看 Observation**（工具返回啥）
- 直到 LLM 觉得够了，自己决定输出 `Final Answer`

---

### 概念 4：`AgentExecutor` —— Agent 的"运行时"

光有 Agent 还不够，得有个执行器负责"调工具 → 把结果喂回 LLM → 判断啥时候停"。

```python
from langchain_classic.agents import create_react_agent, AgentExecutor

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,                 # 打印 Thought/Action/Observation 全过程
    handle_parsing_errors=True,   # LLM 输出格式偶尔出错时，温柔地让它重试
    max_iterations=5,             # 最多调 5 次工具，防止死循环
)

result = agent_executor.invoke({"input": "咖啡机的水泵压力是多少？"})
print(result["output"])
```

**3 个关键参数**：
| 参数 | 作用 |
|---|---|
| `verbose=True` | 打印完整思考过程，调试必备 |
| `handle_parsing_errors=True` | 推理模型偶尔输出格式异常，让 Agent 自己重试 |
| `max_iterations=N` | 最多调几次工具，防止 Agent 死循环烧 token |

> ⚠️ **LangChain 1.x 警告**：`create_react_agent` / `AgentExecutor` 已移到 `langchain_classic` 包，未来 LangGraph 会是推荐方向。

---

### 概念 5：手写 ReAct 模板（vs `hub.pull`）

**方法 A：拉官方模板**
```python
from langchain_classic import hub
prompt = hub.pull("hwchase17/react")  # ⚠️ 默认禁用，要 dangerously_allow_public_pull=True
```

**方法 B：手写模板（推荐，更可控）**
```python
REACT_TEMPLATE = """你是一个中文电商客服助手...

你可以使用以下工具：
{tools}

请严格按以下格式回答：

Question: 用户的问题
Thought: 你在想什么、打算怎么做
Action: 要调用的工具名，必须是 [{tool_names}] 之一
Action Input: 传给工具的参数
Observation: 工具返回的结果（这一步由系统填，你不要写）
... (可以重复 N 轮)
Thought: 我现在可以给出最终答案了
Final Answer: 给用户的最终中文回答

开始！

Question: {input}
Thought:{agent_scratchpad}"""

prompt = ChatPromptTemplate.from_template(REACT_TEMPLATE)
```

**模板里 3 个特殊占位符**：
| 占位符 | 含义 |
|---|---|
| `{tools}` | 自动注入工具列表（含 name + description）|
| `{tool_names}` | 自动注入工具名列表（Action 那行用）|
| `{agent_scratchpad}` | 自动注入历史 Thought/Action/Observation |

---

## 🏗️ 实战项目：wisdom.py 的演化

### 本阶段改了哪些代码

#### ① 加导入

```python
from langchain_core.tools import tool
# LangChain 1.x 把 Agent 移到了 langchain_classic 包
from langchain_classic.agents import create_react_agent, AgentExecutor
import datetime
```

#### ② 定义 3 个 Tool

```python
@tool
def search_knowledge_base(query: str) -> str:
    """在电商产品/退货知识库里检索相关信息。输入是用户问题的核心关键词。"""
    retrieved = retriever.invoke(query)
    return format_docs(retrieved)


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入必须是合法的 Python 算式字符串。"""
    try:
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_current_time(_: str = "") -> str:
    """获取当前的日期和时间，无需任何参数。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

> 💡 **小技巧**：`get_current_time(_: str = "")` 用 `_` 作参数名表示"忽略"，避免 LLM 不知道传啥。

#### ③ 手写 ReAct 模板 + Agent

```python
REACT_TEMPLATE = """...见概念5..."""

agent_prompt = ChatPromptTemplate.from_template(REACT_TEMPLATE)

tools = [search_knowledge_base, calculator, get_current_time]

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
)
```

#### ④ 封装 `run_agent` 函数

```python
def run_agent(question: str) -> str:
    print(f"\n====== Agent 问答 ======")
    print(f"用户问题: {question}\n")
    result = agent_executor.invoke({"input": question})
    print(f"\n✅ Final Answer: {result['output']}")
    return result["output"]
```

---

### 跑通结果

#### Q1：咖啡机的萃取压力是多少？换算成 PSI 大约多少？

```
Thought: 我需要先查知识库拿到压力值（Bar），再用计算器换算
Action: search_knowledge_base
Action Input: 咖啡机萃取压力
Observation: 9 Bar ± 0.5

Thought: 知识库说 9 Bar。1 Bar ≈ 14.5 PSI，算一下
Action: calculator
Action Input: 9 * 14.5
Observation: 130.5

Final Answer: 9 Bar ≈ 130.5 PSI
```

🎯 **亮点**：Agent **自主**做了"两步推理 + 两次工具调用"，Chain 做不到这种动态决策。

#### Q2：现在几点了？

```
Action: get_current_time
Observation: 2026-09-04 14:17:20
```

🎯 Agent 自主选了"时间工具"，没用知识库（因为知识库没时间）。

#### Q3：退货流程审核需要多久？审核后几天内必须寄回？

```
Action: search_knowledge_base
Action Input: 退货审核时间 寄回时限
Observation: 审核 1-3 个工作日，7 天内寄回

Final Answer: 审核 1-3 工作日内完成，7 天内必须寄回，逾期自动关闭
```

🎯 Agent 自主判断"只查知识库就够"，没用计算器。

---

## 🧪 动手练习

### 必做（3 个）

#### 练习 1：故意设计一个"需要多工具"的问题
**目标**：观察 Agent 多工具协同。

```python
# 加到 agent_test_questions：
"咖啡机水箱 2.5 升，能做几杯 30ml 的浓缩？",
# 期望：Agent → search_knowledge_base 查水箱 → calculator 算 2500 / 30
```

#### 练习 2：让 Agent 决定"不调任何工具"
**目标**：理解 Agent 的"零工具调用"路径。

```python
run_agent("你好")
# 期望：Agent 思考后直接 Final Answer，不调工具
```

#### 练习 3：故意问"知识库 + 计算"组合题
**目标**：测 Agent 的多步推理能力。

```python
run_agent("退货审核 3 天 + 寄回 7 天，总共最长多少天？")
# 期望：Agent 自己心算 3 + 7 = 10（或调 calculator）
```

---

### 选做（2 个）

#### 选做 1：加一个真实工具——查订单
```python
@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单状态。输入订单号字符串。"""
    # 真实场景接数据库
    fake_db = {"OD001": "已发货", "OD002": "待付款"}
    return fake_db.get(order_id, "订单不存在")
```

#### 选做 2：把 RAG 检索改成"先过滤 metadata 再搜"
```python
@tool
def search_knowledge_base(query: str, category: str = "") -> str:
    """在电商知识库里检索。category 可选: 咖啡机, 退货信息 等"""
    # 让 LLM 传 category 参数，配合 FAISS 的 filter 功能
    ...
```

---

## ⚠️ 常见 Bug 速查

| 现象 | 原因 | 修法 |
|---|---|---|
| `ImportError: create_react_agent from langchain.agents` | LangChain 1.x 移到 `langchain_classic` | `from langchain_classic.agents import create_react_agent` |
| `hub.pull("hwchase17/react")` 报"disabled by default" | 安全策略变了 | 改手写 ReAct 模板 |
| Agent 输出 `Invalid or incomplete response` | 推理模型同时输出了 Final Answer 和 Action | `handle_parsing_errors=True` 让它自愈；或换 OpenAI tools 风格 |
| Agent 死循环、token 烧光 | 没限制迭代次数 | `max_iterations=5` |
| Tool 被调了但参数不对 | docstring 没写清 | 把 docstring 写具体（举例子）|
| Agent 完全不调工具 | 工具描述太抽象 | docstring 写明"啥时候用我"|
| Agent 调错工具 | 多个工具描述相似 | 让每个工具的"领域"明显区分 |

---

## ✅ 自我检测（8 题）

### 判断题（4 题）

1. ⬜ Chain 和 Agent 本质上一样，只是名字不同
2. ⬜ `@tool` 装饰器会把函数的 docstring 当作工具描述传给 LLM
3. ⬜ ReAct 模式里 LLM 只能调一次工具
4. ⬜ 推理模型（deepseek-r1）和 ReAct 兼容性完美

### 代码题（4 题）

**题 1**：补全 Tool 定义
```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """查某城市天气。输入是城市名，例如"北京""""
    return f"{city}：晴，25°C"

# 这个工具的 name / description / args 分别是啥？
```

**题 2**：补全 Agent 创建
```python
from langchain_classic.agents import create_react_agent, AgentExecutor

agent = ???(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=???,
    ???=True,            # 打印思考过程
    ???=True,            # 自动恢复解析错误
    ???=5,               # 最多 5 次迭代
)
```

**题 3**：判断下面 ReAct 流程是否正确
```
Question: 退货几天？
Thought: 我查知识库
Action: search_knowledge_base
Action Input: 退货
Observation: 7 天
Thought: 好的
Final Answer: 7 天
```
✅ 还是 ❌？哪里有问题？

**题 4**：为什么下面这样写 LLM 永远不调这个工具？
```python
@tool
def my_tool(x: str) -> str:
    """工具"""
    return x
```
应该怎么改？

---

## 📌 速查卡

### 4 个核心组件
```python
from langchain_core.tools import tool                                      # 工具装饰器
from langchain_classic.agents import create_react_agent, AgentExecutor    # Agent + 运行时
from langchain_core.prompts import ChatPromptTemplate                     # ReAct 模板
```

### 3 步搭建 Agent
```python
# 1. 定义工具（每个函数一个工具）
@tool
def my_tool(x: str) -> str:
    """工具描述（LLM 用这个决定啥时候调我）"""
    return "..."

tools = [my_tool, ...]

# 2. 写 ReAct prompt（含 {tools} / {tool_names} / {agent_scratchpad}）
prompt = ChatPromptTemplate.from_template(REACT_TEMPLATE)

# 3. 创建 Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### 5 个常见 Tool 模式
```python
# 1. RAG 检索
@tool
def search_kb(query: str) -> str:
    """检索知识库"""
    return format_docs(retriever.invoke(query))

# 2. 数学计算
@tool
def calc(expr: str) -> str:
    """算数学表达式"""
    return str(eval(expr))

# 3. 查数据库
@tool
def query_db(sql: str) -> str:
    """执行 SQL"""
    return db.execute(sql)

# 4. 调外部 API
@tool
def get_weather(city: str) -> str:
    """查天气"""
    return requests.get(f".../{city}").text

# 5. 无参工具（用 _ 占位）
@tool
def get_time(_: str = "") -> str:
    """获取当前时间"""
    return datetime.now().isoformat()
```

### AgentExecutor 的 3 个救命参数
```python
AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,                  # 调试必备
    handle_parsing_errors=True,    # 推理模型救星
    max_iterations=5,              # 防止 token 爆炸
)
```

---

## 🎓 本阶段总结

| 学到的 | 怎么检验 |
|---|---|
| Chain vs Agent | 能说出"动态决策 vs 固定流程"的区别 |
| `@tool` | 能独立写一个有清晰 docstring 的工具 |
| ReAct 循环 | 能完整说出 Thought/Action/Observation 三步 |
| 手写 ReAct 模板 | 能默写含 3 个特殊占位符的模板 |
| 推理模型兼容 | 能解释为啥要 `handle_parsing_errors=True` |
| max_iterations | 能说出为啥要限制迭代次数（防 token 爆炸）|

---

## ⚠️ 关于 deprecation 警告

跑代码你会看到：
```
LangChainDeprecationWarning: langchain_classic.hub.pull is deprecated.
Use the LangSmith SDK instead.
```

**怎么办**：
- ✅ **学习阶段**：继续用 `langchain_classic` + 手写模板，代码易懂
- 🚀 **生产项目**：考虑迁移到 **LangGraph**（更现代的状态图框架，支持人机协同、多 Agent 协作）

```bash
# 未来路线
pip install langgraph
```

---

## ➡️ Phase 7 预告：项目整合 + 部署

**下一阶段目标**：把前面 6 个 Phase 串起来，做一个**完整可运行**的应用。

**会做**：
- 把 Chain + Agent + RAG + Memory 整合成一条端到端 pipeline
- 用 FastAPI / Streamlit 暴露成 HTTP 服务
- 加日志、错误处理、限流
- 写 README，跑通部署

**Phase 7 文档生成条件**：你完成 Phase 7 的代码改造并跑通后，告诉我"开搞 Phase 7"，我会生成 `docs/PHASE7.md`。

---

## 📂 文件清单

本阶段相关文件：
- `wisdom.py`：新增 3 个 `@tool` 函数、`REACT_TEMPLATE`、`tools` 列表、`agent`、`agent_executor`、`run_agent` 函数
- `docs/PHASE6.md`：本学习文档

**运行方式**：
```bash
python wisdom.py
# 跑全部测试（Phase 1-5 + Phase 6 Agent）
```