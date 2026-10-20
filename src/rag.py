"""RAG 检索增强生成（Phase 5）

- 从 vectorDocs/ 加载文档（.txt / .md）
- 用 BGE 中文 embedding + Chroma 建向量库（持久化到 chroma_db/）
- 支持按 category 元数据过滤检索（product-level 需求）
- 提供 retriever + rag_chain + rag_query + rebuild_vectorstore
"""
import os
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    DOCS_DIR,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    RETRIEVER_K,
)
from .llm import get_llm


# ---------- 文档加载 ----------
def load_documents_from_folder(folder: str | os.PathLike) -> list[Document]:
    """遍历 folder 下所有 .txt / .md，按章节切分成多个 Document"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", " "],
    )
    docs: list[Document] = []
    folder = str(folder)
    if not os.path.isdir(folder):
        print(f"[RAG] 警告：文档目录不存在: {folder}")
        return docs
    for filename in sorted(os.listdir(folder)):
        # 同时支持 .txt 和 .md（项目里大多是 .md）
        if not (filename.endswith(".txt") or filename.endswith(".md")):
            continue
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        category = os.path.splitext(filename)[0]
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": filename, "category": category, "chunk": i},
            ))
    return docs


def format_docs(docs: list[Document]) -> str:
    """Document 列表 → 字符串"""
    return "\n\n---\n\n".join(
        [f"[{doc.metadata.get('category', '')}] {doc.page_content}" for doc in docs]
    )


# ---------- 单例懒加载 ----------
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------- 单例懒加载（Chroma 持久化） ----------
def _chroma_count() -> int:
    """查询已持久化的索引里有多少条向量（0 = 还没建过）"""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(CHROMA_COLLECTION).count()
    except Exception:
        return 0


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """建/加载向量库。

    - 首次：从 vectorDocs/ 读文档 → embedding → 写入 chroma_db/（落盘）
    - 之后：直接加载磁盘索引，不再重算 embedding（这正是替换 FAISS 的意义）
    - 首次会下载约 100MB embedding 模型
    """
    if _chroma_count() > 0:
        print(f"[RAG] 加载已有 Chroma 索引：{_chroma_count()} chunks（{CHROMA_DIR}）")
        return Chroma(
            persist_directory=str(CHROMA_DIR),
            collection_name=CHROMA_COLLECTION,
            embedding_function=get_embeddings(),
        )

    documents = load_documents_from_folder(DOCS_DIR)
    print(f"[RAG] 首次构建 Chroma 索引：{len(documents)} chunks（来自 {DOCS_DIR}）")
    if not documents:
        raise RuntimeError(f"vectorDocs 目录为空或不存在: {DOCS_DIR}")
    return Chroma.from_documents(
        documents,
        embedding=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_name=CHROMA_COLLECTION,
    )


def rebuild_vectorstore() -> int:
    """强制重建索引（vectorDocs/ 文档更新后调用）。

    相比 FAISS 全量重建：Chroma 直接删 collection 重写，且支持以后做增量 add。
    返回重建后的 chunk 数。
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass  # collection 不存在（首次），忽略
    # 清掉所有依赖向量库的懒加载缓存
    get_vector_store.cache_clear()
    get_retriever.cache_clear()
    get_rag_chain.cache_clear()
    vs = get_vector_store()
    return _chroma_count()


@lru_cache(maxsize=8)
def get_retriever(category: str | None = None):
    """获取 retriever。

    Args:
        category: 可选，按文档类别过滤（metadata.category，如 "咖啡机"）。
                  传了就只在该类别内做相似度检索，避免跨产品误召回。
    """
    search_kwargs: dict = {"k": RETRIEVER_K}
    if category:
        search_kwargs["filter"] = {"category": category}
    return get_vector_store().as_retriever(search_kwargs=search_kwargs)


# ---------- RAG 链 ----------
_llm = get_llm()

rag_prompt = ChatPromptTemplate.from_template(
    """你是电商客服助手。请基于以下【知识库内容】回答用户问题。

【知识库内容】
{context}

【用户问题】
{input}

要求：
1. 必须基于知识库内容回答，不要编造
2. 找不到答案时说"知识库中没有相关信息"
3. 回答简洁明了"""
)


def _build_rag_chain():
    retriever = get_retriever()
    return (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | rag_prompt
        | _llm
        | StrOutputParser()
    )


@lru_cache(maxsize=1)
def get_rag_chain():
    """懒加载 RAG chain（首次调用才建向量库，避免启动慢）"""
    return _build_rag_chain()


# ---------- 入口函数 ----------
def rag_query(question: str) -> str:
    """RAG 问答：先检索知识库，再让 LLM 基于知识库回答"""
    print(f"\n====== RAG 问答 ======")
    print(f"用户问题: {question}")

    retrieved = get_retriever().invoke(question)
    print(f"检索到 {len(retrieved)} 条相关知识：")
    for i, doc in enumerate(retrieved, 1):
        print(f"  [{i}] {doc.metadata.get('category', '')}: {doc.page_content[:80]}...")

    answer = get_rag_chain().invoke({"input": question})
    print(f"AI 回答: {answer}")
    return answer


__all__ = [
    "load_documents_from_folder",
    "format_docs",
    "get_embeddings",
    "get_vector_store",
    "get_retriever",
    "get_rag_chain",
    "rag_prompt",
    "rag_query",
    "rebuild_vectorstore",
]