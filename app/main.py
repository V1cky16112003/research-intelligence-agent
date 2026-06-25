from __future__ import annotations
import logging
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)
START_TIME = time.time()
_query_counter = 0

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = ""
    redis_url: str = ""
    # Upstash provides two separate vars — we combine them into redis_url at startup
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    dagshub_token: str = ""
    dagshub_repo: str = ""
    embed_model: str = "nomic-ai/nomic-embed-text-v2-moe"
    embed_dim: int = 768

    def get_redis_url(self) -> str:
        """Return a single Redis URL, combining Upstash vars if needed."""
        if self.redis_url:
            return self.redis_url
        if self.upstash_redis_rest_url and self.upstash_redis_rest_token:
            # Build https://default:{token}@{host} from Upstash's two-var format
            from urllib.parse import urlparse
            parsed = urlparse(self.upstash_redis_rest_url)
            return f"https://default:{self.upstash_redis_rest_token}@{parsed.netloc}"
        return ""

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Autonomous Research Intelligence Agent")
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY not set — LLM calls will fail")
    if not settings.database_url:
        logger.warning("DATABASE_URL not set — DB calls will fail")
    yield
    logger.info("Shutting down")

app = FastAPI(title="Research Intelligence Agent", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    answer: str
    citations: list
    sql_results: list | None
    session_id: str
    provider: str

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global _query_counter
    _query_counter += 1
    session_id = req.session_id or str(uuid.uuid4())
    # LangGraph agent wired in Wave 3 — still stub until that commit lands
    return ChatResponse(
        answer=f"[Stub] You asked: {req.query}. LangGraph agent loading...",
        citations=[],
        sql_results=None,
        session_id=session_id,
        provider="stub",
    )

@app.get("/metrics")
async def metrics():
    return {"total_queries": _query_counter, "uptime_seconds": time.time() - START_TIME}
