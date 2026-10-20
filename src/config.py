"""统一配置：环境变量、路径、常量"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 必须在所有 huggingface_hub 相关库导入前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 知识库目录
DOCS_DIR = PROJECT_ROOT / "vectorDocs"

# 加载 .env（override=True：覆盖已存在的环境变量，保证 .env 一定生效）
load_dotenv(PROJECT_ROOT / ".env", override=True)


# ===== LLM 配置 =====
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))


# ===== 清洗层配置（Phase 1） =====
SENSITIVE_WORDS = [
    # 营销引流
    "广告", "垃圾词", "微商", "代购", "加微信", "加vx", "+v", "+wx",
    # 辱骂攻击
    "傻逼", "草泥马", "操你", "滚蛋", "打你",
    # 无意义
    "反动", "法轮",
]
MIN_LEN = 5
MAX_LEN = 1000


# ===== RAG 配置（Phase 5） =====
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DEVICE = "cpu"  # 没 GPU 时用 CPU；有 GPU 改成 "cuda"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
RETRIEVER_K = 5

# ===== 向量库配置（Chroma 持久化） =====
# 索引落盘目录：服务重启直接加载，不再重算 embedding
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION = "aia_docs"


# ===== Agent 配置（Phase 6） =====
AGENT_MAX_ITERATIONS = 5

# ===== 智能客服配置（Phase 10） =====
# 拼 prompt 时最多带最近 N 条历史消息（滑动窗口）：
# 历史无限增长会撑爆 token 且让模型"忘掉"重点（lost in the middle）
HISTORY_WINDOW = 10
# Agent 的 search_knowledge_base 工具返回截断长度（Observation 喂回 ReAct 不需要全文）
TOOL_RESULT_MAX_CHARS = 1500