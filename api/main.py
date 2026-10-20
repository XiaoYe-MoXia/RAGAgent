"""FastAPI 应用入口

启动：
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
或：
    python -m api.main
"""
import os
from pathlib import Path

# 关键：uvicorn 子进程里可能没继承 shell 的环境变量，必须显式加载
# override=True：即使环境变量已存在，也用 .env 覆盖
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# 启动时校验环境变量是否真的加载了
import os as _os
_env = {k: _os.getenv(k) for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")}
print("[api.main] 环境变量:", {k: (v[:10] + "..." if v and len(v) > 10 else v) for k, v in _env.items()})

# 同样要确保 HF_ENDPOINT 设置了
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, health, sessions

app = FastAPI(
    title="AIAgent API",
    description="LangChain 学习项目的 REST API - 5 种对话模式 + 会话管理",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS：允许任何来源（开发用；生产要限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由注册
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(sessions.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )