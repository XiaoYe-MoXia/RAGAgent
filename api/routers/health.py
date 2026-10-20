"""/health 健康检查"""
from collections import Counter

from fastapi import APIRouter

from api.models import HealthResponse
from src.rag import load_documents_from_folder
from src.config import DOCS_DIR

router = APIRouter(tags=["health"])


@router.get("/", response_model=HealthResponse)
@router.get("/api/health", response_model=HealthResponse)
def health():
    """服务状态 + 知识库统计"""
    docs = load_documents_from_folder(DOCS_DIR)
    cnt = Counter(d.metadata.get("category", "unknown") for d in docs)
    return HealthResponse(
        status="ok" if docs else "degraded",
        version="0.1.0",
        knowledge_base={
            "total_chunks": len(docs),
            "by_category": dict(cnt),
            "docs_dir": str(DOCS_DIR),
        },
    )