"""请求/响应模型（Pydantic）"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ===== 通用 =====

class ChatRequest(BaseModel):
    """对话请求基类"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")


class ChatResponse(BaseModel):
    """对话响应"""
    answer: str = Field(..., description="AI 回答")
    mode: str = Field(..., description="实际使用的模式")
    metadata: dict = Field(default_factory=dict, description="附加信息（分类、检索条数等）")


# ===== 多轮对话 =====

class MultiTurnRequest(ChatRequest):
    """多轮对话请求"""
    session_id: str = Field(default="default", description="会话 ID（不同 ID 隔离历史）")


# ===== 全模式流式 =====

class AllStreamRequest(ChatRequest):
    """全模式流式请求 - mode 决定走哪种逻辑（auto=智能路由，Phase 10 默认）"""
    mode: Literal["auto", "chain", "rag", "agent", "multi", "stream"] = Field(
        default="auto", description="对话模式"
    )
    session_id: str = Field(default="default", description="会话 ID（auto/multi 模式必填）")


class MultiTurnResponse(ChatResponse):
    """多轮对话响应"""
    session_id: str
    history_length: int = Field(..., description="当前会话历史消息数")


# ===== 会话管理 =====

class HistoryResponse(BaseModel):
    """历史消息"""
    session_id: str
    messages: list[dict] = Field(default_factory=list, description="[{role, content}, ...]")


class ClearResponse(BaseModel):
    """清空会话响应"""
    session_id: str
    cleared: bool


# ===== 健康检查 =====

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    knowledge_base: dict = Field(default_factory=dict, description="知识库文档统计")


# ===== Agent 工具日志 =====

class AgentResponse(ChatResponse):
    """Agent 响应（含思考日志）"""
    tool_log: list[str] = Field(default_factory=list, description="Thought/Action/Observation")