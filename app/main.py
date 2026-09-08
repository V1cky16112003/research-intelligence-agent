from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""
    # Comma-separated origin allowlist. Defaults to "*" to preserve the deployed
    # Vercel frontend; set ALLOWED_ORIGINS to that frontend's URL to lock it down.
    allowed_origins: str = "*"
    # /chat is unauthenticated and each request fans out to ~8 provider calls on
    # personal API keys, so an open endpoint is a direct quota-drain amplifier.
    # Generous enough that no human user notices; low enough to stop a script.
    chat_rate_limit_per_minute: int = 20

    def get_allowed_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()] or ["*"]

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
        from db.connection import apply_schema, get_connection, init_pool
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

    # Neo4j graph driver (optional — graph_query tool degrades gracefully if unset)
    if settings.neo4j_uri:
        from graph.neo4j_client import get_driver
        get_driver(uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password)
        logger.info("Neo4j graph driver ready")
    else:
        logger.warning("NEO4J_URI not set — graph_query tool will error gracefully if invoked")

    # Embedding model — warmed here instead of lazily on first request. Loading
    # nomic-embed-text-v2-moe takes 10-30s on a fresh container. Without this, the
    # first rag_retrieval call after a Hugging Face Space wakes from sleep raced
    # (and lost to) the agent's retry/timeout budget, silently returning zero
    # chunks — the user saw "no information found" for a query the corpus can
    # answer, and it looked fine again on the very next request once warm.
    try:
        from ingestion.embed import get_model
        await asyncio.to_thread(get_model)
        logger.info("Embedding model warmed")
    except Exception as e:
        logger.warning("Embedding model warm-up failed, will lazy-load on first request: %s", e)

    logger.info("Research agent ready")
    yield

    # Shutdown
    if settings.database_url:
        from db.connection import close_pool
        await close_pool()

    if settings.neo4j_uri:
        from graph.neo4j_client import close_driver
        await close_driver()


app = FastAPI(title="Research Intelligence Agent", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


# Fixed-window per-client counter. In-process, so it resets on redeploy and does not
# coordinate across replicas — it is a quota-drain brake, not an access control. The
# endpoint still has no authentication; see docs/security.md.
_rate_window: dict[str, tuple[int, float]] = {}


def _client_key(request: Request) -> str:
    # Hugging Face Spaces terminates TLS upstream, so the socket peer is the proxy.
    # X-Forwarded-For is client-controlled and trivially spoofed, which is precisely
    # why this is a brake and not a control.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limited(request: Request) -> bool:
    limit = settings.chat_rate_limit_per_minute
    if limit <= 0:
        return False
    key = _client_key(request)
    now = time.time()
    count, window_start = _rate_window.get(key, (0, now))
    if now - window_start >= 60:
        count, window_start = 0, now
    count += 1
    _rate_window[key] = (count, window_start)
    if len(_rate_window) > 10_000:  # bound the dict against unique-IP flooding
        for stale, (_, started) in list(_rate_window.items()):
            if now - started >= 60:
                _rate_window.pop(stale, None)
    return count > limit


class ChatRequest(BaseModel):
    # Bounded: the query is embedded, sent to the planner, and echoed into the
    # reporter prompt, so an unbounded string is a cheap way to inflate token spend.
    query: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=200)


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
    session_id = req.session_id or str(uuid.uuid4())

    if _rate_limited(request):
        return ChatResponse(
            answer="Too many requests — please wait a minute and try again.",
            citations=[], sql_results=None, session_id=session_id, provider="rate_limited",
        )

    _query_counter += 1
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
                from db.queries import log_llm_calls, log_query
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
                    await log_llm_calls(conn, session_id=session_id, calls=result.get("llm_calls", []))
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
        # Log the detail, return a generic message. str(e) on the exceptions that
        # reach here comes from psycopg (which puts host/database/user in connection
        # errors) and from provider SDKs (which echo request URLs and account
        # identifiers), so returning it verbatim published internal topology to any
        # caller who could make the request fail.
        logger.error("Chat failed: %s", e, exc_info=True)
        return ChatResponse(
            answer="I encountered an internal error while answering that. Please try again.",
            citations=[],
            sql_results=None,
            session_id=session_id,
            provider="error",
        )


@app.get("/metrics")
async def metrics():
    return {"total_queries": _query_counter, "uptime_seconds": time.time() - START_TIME}


@app.get("/analytics/cost")
async def analytics_cost(days: int = Query(default=30, ge=1, le=3650)):
    """Cost/latency dashboard data: per-node/provider/day aggregates plus retry
    overhead as its own line item. Backed by the llm_cost_latency and
    llm_retry_overhead views over llm_call_log (db/schema.sql)."""
    if not settings.database_url:
        return {"error": "DATABASE_URL not set — cost/latency data unavailable"}

    from db.connection import get_connection
    from db.queries import llm_cost_by_node, retry_overhead_summary

    async with get_connection() as conn:
        by_node = await llm_cost_by_node(conn, days=days)
        retry_overhead = await retry_overhead_summary(conn, days=days)

    return {"by_node": by_node, "retry_overhead": retry_overhead}
