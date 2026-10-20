"""/api/sessions/* 会话管理"""
from fastapi import APIRouter, HTTPException

from api.models import ClearResponse, HistoryResponse
from src.memory import clear_session, get_session_history

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("/{session_id}/history", response_model=HistoryResponse)
def get_history(session_id: str):
    """获取会话历史"""
    history = get_session_history(session_id)
    messages = [
        {"role": m.type if hasattr(m, "type") else "unknown", "content": m.content}
        for m in history.messages
    ]
    return HistoryResponse(session_id=session_id, messages=messages)


@router.delete("/{session_id}", response_model=ClearResponse)
def clear(session_id: str):
    """清空会话历史"""
    try:
        clear_session(session_id)
        return ClearResponse(session_id=session_id, cleared=True)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))