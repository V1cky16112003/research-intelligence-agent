from __future__ import annotations
import logging
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
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
    nvidia_nim_api_key: str = ""
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
    # DB pool
    if settings.database_url:
        from db.connection import init_pool, get_connection, apply_schema
        await init_pool(settings.database_url)
        try:
            async with get_connection() as conn:
                await apply_schema(conn)
        except Exception as e:
            logger.warning("Schema apply failed: %s", e)
    else:
        logger.warning("DATABASE_URL not set — DB disabled")

    # Redis
    from agent.redis_client import create_redis_client
    redis_client = await create_redis_client(settings.get_redis_url() or None)

    # LLM gateway
    from agent.gateway import LLMGateway
    from agent.registry import set_gateway
    gateway = LLMGateway(
        groq_api_key=settings.groq_api_key,
        nvidia_api_key=settings.nvidia_nim_api_key,
        gemini_api_key=settings.gemini_api_key,
        redis_client=redis_client,
    )
    set_gateway(gateway)

    # LangGraph agent
    from agent.graph import init_graph
    await init_graph()

    logger.info("Research agent ready")
    yield

    # Shutdown
    if settings.database_url:
        from db.connection import close_pool
        await close_pool()


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
async def chat(req: ChatRequest, request: Request):
    global _query_counter
    _query_counter += 1
    session_id = req.session_id or str(uuid.uuid4())
    start = time.time()

    try:
        from agent.graph import run_agent
        result = await run_agent(
            user_query=req.query,
            session_id=session_id,
        )
        latency_ms = int((time.time() - start) * 1000)

        # Log to audit table
        if settings.database_url:
            try:
                from db.connection import get_connection
                from db.queries import log_query
                async with get_connection() as conn:
                    await log_query(
                        conn,
                        session_id=session_id,
                        user_query=req.query,
                        route="multi",
                        tools_called=result.get("tools_called", []),
                        latency_ms=latency_ms,
                        tokens_in=result.get("tokens_in", 0),
                        tokens_out=result.get("tokens_out", 0),
                        llm_provider=result.get("provider", "unknown"),
                        retrieved_chunk_ids=[
                            c.get("chunk_id")
                            for c in result.get("citations", [])
                            if c.get("chunk_id")
                        ],
                    )
                    await conn.commit()
            except Exception as e:
                logger.warning("Audit log failed: %s", e)

        return ChatResponse(
            answer=result["final_report"],
            citations=result.get("citations", []),
            sql_results=result.get("sql_results"),
            session_id=session_id,
            provider=result.get("provider", "unknown"),
        )

    except Exception as e:
        logger.error("Chat failed: %s", e, exc_info=True)
        return ChatResponse(
            answer=f"I encountered an error: {str(e)}. Please try again.",
            citations=[],
            sql_results=None,
            session_id=session_id,
            provider="error",
        )


@app.get("/metrics")
async def metrics():
    return {"total_queries": _query_counter, "uptime_seconds": time.time() - START_TIME}
