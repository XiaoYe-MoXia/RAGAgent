# 📘 Phase 5 学习文档：RAG 检索增强生成

> **本阶段目标**：让 LLM 能回答**自己训练数据里没有的内容**——比如公司内部文档、最新新闻、私人笔记——通过"先查资料，再回答"的方式弥补 LLM 的知识盲区。

---

## 🎯 学习目标 Checklist

完成本阶段后，你应该能：
- [ ] 说出 RAG 是什么、为什么需要它
- [ ] 解释 Embedding / 向量数据库 / Retriever 各自的作用
- [ ] 用 `HuggingFaceEmbeddings` + `FAISS.from_documents()` 搭一个最小 RAG
- [ ] 用 `RunnablePassthrough.assign()` 把"检索结果"注入 prompt
- [ ] 独立判断一个场景该不该用 RAG
- [ ] 说出 RAG 的局限（不是万能的）

---

## 📚 核心概念（7 个）

### 概念 1：为什么需要 RAG？

**LLM 的三大短板**：
| 短板 | 例子 |
|---|---|
| 知识有截止日期 | 训练完之后的新闻/政策不知道 |
| 不知道私有内容 | 公司内部文档、产品手册、用户订单 |
| 会"一本正经地编" | 模型为了"显得专业"会幻觉出假信息 |

**RAG = Retrieval Augmented Generation（检索增强生成）**：
> 在 LLM 回答前，**先从一个知识库里检索相关文档**，把检索到的内容塞进 prompt，让 LLM **"看着资料回答"**。

**类比**：开卷考试——考生就是 LLM，参考资料就是知识库。

---

### 概念 2：Embedding（嵌入）—— "文字 → 数字向量"

**是什么**：把一段文字转成一串固定长度的数字（向量），让"语义相近"的文字在向量空间里也相近。

**示例**：
```
"退货政策"   → [0.12, 0.87, -0.23, 0.55, ..., 0.34]  (512 维)
"怎么退款"   → [0.11, 0.85, -0.21, 0.53, ..., 0.36]  ← 距离很近！
"今天天气好" → [-0.78, 0.02, 0.91, -0.45, ..., 0.11]  ← 距离很远
```

**代码**：
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector = embeddings.embed_query("退货政策")
print(len(vector))  # 512 维
```

**常用中文模型**：
- `BAAI/bge-small-zh-v1.5`（我们用的，约 100MB，中文友好）
- `BAAI/bge-large-zh-v1.5`（更准，约 1.3GB）
- `shibing624/text2vec-base-chinese`（备选）

---

### 概念 3：向量数据库（Vector Store）—— "按相似度搜"

**是什么**：专门存"向量 + 原始文本"的数据库，查询时按**相似度**（如余弦相似度）找最相近的几条。

**为什么不用普通数据库？**
- 普通数据库按"精确匹配"搜（`WHERE content LIKE '%退货%'`）
- 向量数据库按"语义匹配"搜（搜"退货"也能命中"退款"、"七天无理由"）

**常见选择**：

| 向量库 | 特点 |
|---|---|
| FAISS | Meta 出品，轻量、纯本地、内存级 |
| Chroma | 更现代，支持持久化 |
| Milvus | 生产级，支持亿级向量 |
| Pinecone | 云服务，付费 |

我们用 **FAISS**（最简单，零配置）。

**代码**：
```python
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

docs = [Document(page_content="退货政策：7天内...")]
vector_store = FAISS.from_documents(docs, embeddings)  # 自动 embed + 索引
```

---

### 概念 4：Retriever（检索器）—— "把向量库变成一个函数"

**是什么**：把向量库包装成一个 `invoke(question) → [Doc, Doc]` 的可调用对象，**用统一的接口去查**。

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

results = retriever.invoke("怎么退货？")
for doc in results:
    print(doc.page_content)
```

**`k` 参数**：返回最相似的 k 条（k 太小可能漏，k 太大 prompt 会爆）。

**为什么不在向量库上直接搜？**
- 未来可能换向量库（FAISS → Milvus），换库要改一堆代码
- Retriever 是 LangChain 的标准抽象，换库不动业务代码

---

### 概念 5：RAG 链 —— "检索 + 回答"串起来

**思路**：用户问 → 检索相关文档 → 拼成 prompt → LLM 回答。

**最小代码模板**（**建议背下来**）：
```python
from langchain_core.runnables import RunnablePassthrough

rag_prompt = ChatPromptTemplate.from_template("""
你是客服助手。请基于【知识库】回答用户问题，找不到就说不知道。

【知识库】
{context}

用户问题：{input}
""")

rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["input"]))
    )
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

**`RunnablePassthrough.assign` 是什么？**
- `RunnablePassthrough`：原样传 dict 过去
- `.assign({...})`：往 dict 里**加新 key**
- 上面那行 = 在原 `{"input": ...}` 上**加一个 `context` key**，值是检索结果的格式化文本

**之后**：prompt 模板里的 `{input}` 和 `{context}` 都能拿到了。

---

### 概念 6：Document 对象 —— 知识库的"最小单元"

```python
from langchain_core.documents import Document

doc = Document(
    page_content="退货政策：7天内...",   # 必填：实际文本
    metadata={"category": "return"},    # 可选：附加信息（分类、来源、时间...）
)
```

**metadata 的作用**：
- 检索后可以过滤（如"只检索 2024 年的文档"）
- 调试时知道这条文档从哪来
- 前端展示时分类显示

---

### 概念 7：RAG 不是万能的

**RAG 能解决的**：
- ✅ 公司内部 FAQ（我们做的）
- ✅ 产品手册问答
- ✅ 个人笔记检索

**RAG 不能解决的**：
- ❌ 需要**跨文档推理**（"对比 A 和 B 两个产品的区别"）
- ❌ 需要**结构化查询**（"上个季度的销售冠军"）
- ❌ 文档里根本没有的内容（巧妇难为无米之炊）

**进阶方向**：
- 多跳 RAG（先检索 A，再根据 A 检索 B）
- Agent + RAG（让 LLM 决定"查不查"、"查几次"）
- Hybrid Search（关键词 + 语义混合搜）

---

## 🏗️ 实战项目：wisdom.py 的演化

### 本阶段改了哪些代码

#### ① 加导入

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 国内拉不到 huggingface.co，换成镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
```

#### ② 准备知识库（电商 FAQ）

```python
FAQ_DOCUMENTS = [
    Document(
        page_content="退货政策：用户收到商品后7天内可申请无理由退货...",
        metadata={"category": "return"}
    ),
    Document(page_content="换货政策：...", metadata={"category": "exchange"}),
    Document(page_content="运费说明：...", metadata={"category": "shipping"}),
    # ... 共 7 条
]
```

#### ③ 加载 Embedding + 建向量库

```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},          # 没 GPU 也能跑
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = FAISS.from_documents(FAQ_DOCUMENTS, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
```

> 💡 首次运行会下载约 100MB 模型，国内网络记得设 `HF_ENDPOINT`。

#### ④ RAG prompt（强制"基于知识库回答"）

```python
rag_prompt = ChatPromptTemplate.from_template("""
你是电商客服助手。请基于以下【知识库内容】回答用户问题。

【知识库内容】
{context}

【用户问题】
{input}

要求：
1. 必须基于知识库内容回答，不要编造
2. 找不到答案时说"知识库中没有相关信息"
3. 回答简洁明了
""")
```

#### ⑤ 格式化检索结果

```python
def format_docs(docs):
    return "\n\n---\n\n".join(
        [f"[{doc.metadata.get('category', '')}] {doc.page_content}" for doc in docs]
    )
```

把多条 Document 拼成一个字符串，中间用 `---` 分隔。

#### ⑥ 拼成 RAG 链

```python
rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["input"]))
    )
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

**数据流**：
```
{"input": "怎么退货？"}
   ↓ (RunnablePassthrough.assign 加 context)
{"input": "怎么退货？", "context": "[return] 退货政策：..."}
   ↓ (prompt)
ChatPromptValue(messages=[...含 {input} 和 {context}...])
   ↓ (llm)
AIMessage(chunk="您好，...")
   ↓ (StrOutputParser)
"您好，..."
```

#### ⑦ 封装成 `rag_query` 函数

```python
def rag_query(question: str) -> str:
    print(f"\n====== RAG 问答 ======")
    print(f"用户问题: {question}")

    # 第1步：检索（调试用，让你看到检索到了啥）
    retrieved = retriever.invoke(question)
    print(f"检索到 {len(retrieved)} 条相关知识：")
    for i, doc in enumerate(retrieved, 1):
        print(f"  [{i}] {doc.metadata.get('category', '')}: {doc.page_content[:80]}...")

    # 第2步：让 LLM 基于检索内容回答
    answer = rag_chain.invoke({"input": question})
    print(f"AI 回答: {answer}")
    return answer
```

---

### 跑通结果

```
====== RAG 问答 ======
用户问题: 我买了衣服想退货，怎么办？
检索到 2 条相关知识：
  [1] return: 退货政策：用户收到商品后7天内可申请无理由退货...
  [2] exchange: 换货政策：用户收到商品15天内如出现质量问题...
AI 回答: 您好！关于退货，您需要了解以下要点：
1. 退货时限：收到商品后 7天内 可申请无理由退货。
2. 商品要求：需保持完好，不影响二次销售。
3. 申请方式：退货前请联系客服提交退货申请。
```

**完美证明**：AI **没有幻觉**——所有要点都来自知识库。

---

## 🧪 动手练习

### 必做（3 个）

#### 练习 1：故意问知识库里没有的问题
**目标**：验证 prompt 的"找不到就说不知道"约束是否生效。

```python
rag_query("你们支持比特币支付吗？")  # 知识库没这条
```
**期望**：AI 应该回答"知识库中没有相关信息"，而不是编一个比特币支付流程。

#### 练习 2：调整 k 值看变化
**目标**：理解 `k` 对结果的影响。

```python
# 临时把 k=2 改成 k=4
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
rag_query("运费多少？")
```
**思考**：k=4 时检索到的 4 条有什么？是不是有些"不太相关"的也混进来了？

#### 练习 3：把"分类"改成"商品详情"看效果
**目标**：体验 embedding 的语义匹配能力。

```python
# 在 FAQ_DOCUMENTS 里加一条完全不同领域的（比如"手机维修"）
# 然后问："我的手机坏了" 看会不会匹配到
```

---

### 选做（2 个）

#### 选做 1：用更准的 embedding 模型
```python
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
# 模型更大，检索更准，但下载慢、占内存
```

#### 选做 2：把 FAISS 持久化（重启不丢）
```python
# 保存到本地
vector_store.save_local("my_faiss_index")

# 下次加载
vector_store = FAISS.load_local("my_faiss_index", embeddings)
```

#### 选做 3：加"来源标注"（让 AI 回答时引用原文）
在 `rag_prompt` 里加一句：
```
回答时请标注信息来源（如"根据【退货政策】..."）。
```
看看 AI 是否会在回答里说出"根据【退货政策】"这种引用。

---

## ⚠️ 常见 Bug 速查

| 现象 | 原因 | 修法 |
|---|---|---|
| `OSError: We couldn't connect to 'https://huggingface.co'` | 国内网络拉不到模型 | 设 `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"` |
| `KeyError: 'input'` | 链里字段名不一致 | prompt 里 `{input}` 必须和 invoke 的字典 key 一致 |
| 检索到的内容完全不相关 | embedding 模型不对（如用英文模型跑中文）| 用中文 embedding 模型（bge-small-zh） |
| AI 还是幻觉 / 编答案 | prompt 没强调"只基于知识库" | 在 prompt 里写"找不到就说不知道" |
| `k=1` 漏掉正确答案 | k 太小 | 改成 k=2 或 k=3 |
| `k=10` 答案变差 | k 太大，prompt 被噪声淹没 | k 一般 2-5 最佳 |
| 检索慢 | embedding 在 CPU 跑 | 有 GPU 时把 `device="cpu"` 改成 `device="cuda"` |

---

## ✅ 自我检测（8 题）

### 判断题（4 题）

1. ⬜ RAG 是为了让 LLM "记更多东西"
2. ⬜ Embedding 模型能把任意长度文字转成**固定长度**的向量
3. ⬜ 向量数据库按"精确匹配"搜关键词
4. ⬜ RAG 能 100% 解决 LLM 幻觉问题

### 代码题（4 题）

**题 1**：补全最小 RAG 链
```python
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

retriever = vector_store.as_retriever(???={"k": 2})

rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: ???(retriever.invoke(x["input"]))
    )
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

**题 2**：补全 Document 创建
```python
from langchain_core.documents import Document

doc = Document(
    page_content=???,
    metadata=???
)
```

**题 3**：为什么 `RunnablePassthrough.assign(context=...)` 后，prompt 里能同时用 `{input}` 和 `{context}`？

**题 4**：下面这段代码为什么 AI 还是幻觉？
```python
rag_prompt = ChatPromptTemplate.from_template("""
你是助手。{input}
""")
# 缺了 {context}，没注入知识库
```

---

## 📌 速查卡

### 4 个核心组件
```python
from langchain_community.embeddings import HuggingFaceEmbeddings   # 文字 → 向量
from langchain_community.vectorstores import FAISS                  # 向量数据库
from langchain_core.documents import Document                       # 知识库条目
# retriever = vector_store.as_retriever(...)                        # 检索接口
```

### 5 步搭建最小 RAG
```python
# 1. 准备文档
docs = [Document(page_content="...")]

# 2. 加载 embedding
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 3. 建向量库 + retriever
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 4. 写 prompt（含 {context}）
prompt = ChatPromptTemplate.from_template("基于【{context}】回答：{input}")

# 5. 拼链
chain = (
    RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["input"])))
    | prompt | llm | StrOutputParser()
)
```

### 3 个常用 embedding 模型
```python
"BAAI/bge-small-zh-v1.5"      # 小巧（100MB），中文友好 ⭐ 推荐入门
"BAAI/bge-large-zh-v1.5"      # 更大更准（1.3GB）
"shibing624/text2vec-base-chinese"  # 备选
```

---

## 🎓 本阶段总结

| 学到的 | 怎么检验 |
|---|---|
| RAG 的本质 | 能说出"先检索、后回答"的核心流程 |
| Embedding | 能说出"文字 → 向量"的作用 |
| 向量数据库 vs 普通数据库 | 能说出"语义匹配 vs 精确匹配"的区别 |
| Retriever 接口 | 能写出 `vector_store.as_retriever(search_kwargs={"k": 2})` |
| RunnablePassthrough.assign | 能说出为什么 `assign` 后 prompt 能拿到 `{context}` |
| RAG 局限 | 能列出至少 2 个 RAG 不能解决的场景 |

---

## ⚠️ 关于 deprecation 警告

跑代码时你会看到：
```
DeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in
LangChain 0.2.2 and will be removed in 1.0. An updated version of the
class exists in the `langchain-huggingface` package.
```

**怎么办**：
- ✅ **学习阶段**：继续用 `langchain_community` 版本，代码简单
- 🚀 **生产项目**：迁移到 `langchain-huggingface`（更新、API 更干净）

```bash
pip install langchain-huggingface
```
```python
from langchain_huggingface import HuggingFaceEmbeddings  # 新写法
```

---

## ➡️ Phase 6 预告：Agent（智能体）

**下一阶段目标**：让 LLM **自己决定**——"查不查"、"查几次"、"调哪个工具"——而不是被我们写死的链式流程控制。

**会学到的**：
- **Tool（工具）**：把函数包装成 LLM 能"调用"的工具
- **AgentExecutor**：Agent 的运行时
- **ReAct 框架**：思考 → 行动 → 观察 → 再思考
- 多轮工具调用（一次回答可能要查 3 次工具）

**Phase 6 文档生成条件**：你完成 Phase 6 的代码改造并跑通后，告诉我"开搞 Phase 6"，我会生成 `docs/PHASE6.md`。

---

## 📂 文件清单

本阶段相关文件：
- `wisdom.py`：新增 `FAQ_DOCUMENTS`、`embeddings`、`vector_store`、`retriever`、`rag_prompt`、`format_docs`、`rag_chain`、`rag_query`
- `docs/PHASE5.md`：本学习文档

**运行方式**：
```bash
python wisdom.py
# 跑全部测试（既有 Phase 1-4 也有 Phase 5 RAG 演示）
```