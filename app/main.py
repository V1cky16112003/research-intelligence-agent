from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.background import BackgroundTask

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
    # Comma-separated origin allowlist. Defaults to the production Vercel domain
    # plus the Vite dev server; "*" let any site spend this API's provider quota.
    # Override with ALLOWED_ORIGINS (e.g. a preview URL, or "*" to reopen).
    # Model pins, overridable so a provider retirement (NIM has silently 410'd two
    # pinned models) is a secret change, not a code deploy. Empty = gateway default.
    groq_model: str = ""
    nim_model: str = ""
    allowed_origins: str = "https://frontend-vert-eight-61.vercel.app,http://localhost:5173"
    # /chat is unauthenticated and each request fans out to ~8 provider calls on
    # personal API keys, so an open endpoint is a direct quota-drain amplifier.
    # Generous enough that no human user notices; low enough to stop a script.
    chat_rate_limit_per_minute: int = 20
    # Agent runs admitted at once. Groq's free tier caps at 8000 TPM and one run
    # makes six-to-ten LLM calls, so past ~4 concurrent runs every extra request
    # just queues on 429s inside the gateway (p50 hit 250s at 8-way). Shedding
    # with a 503 + Retry-After is faster for everyone than admitting it.
    max_concurrent_chats: int = 4
    # Hard cap on the answer returned to the client — a runaway generation (or a
    # prompt-injected one) should not ship megabytes to the browser.
    max_answer_chars: int = 20_000
    # Supabase Auth. Setting SUPABASE_URL turns authentication on for /chat and
    # /chat/stream (see app/auth.py); SUPABASE_JWT_SECRET only for legacy HS256 projects.
    supabase_url: str = ""
    supabase_jwt_secret: str = ""
    # Per-user daily cap on agent runs — the per-user budget that an IP limit can't
    # give. ~100 runs is ~70k Groq tokens, a third of the free 200k TPD.
    user_daily_query_limit: int = 100
    # "json" for one JSON object per log line (what log search wants), "text" for local dev.
    log_format: str = "json"

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

# Set per request by the middleware below and stamped onto every log record, so one
# /chat run's planner, executor and gateway lines can be pulled out of the stream.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_RequestIdFilter())
    if settings.log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


# Shared with the gateway's response cache; None when Redis is unconfigured.
_redis = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    configure_logging()
    from agent.tracing import init_tracing
    init_tracing()

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
    _redis = redis_client

    # LLM gateway
    from agent.gateway import LLMGateway
    from agent.registry import set_gateway
    gateway = LLMGateway(
        groq_api_key=settings.groq_api_key,
        nvidia_api_key=settings.nvidia_nim_api_key,
        gemini_api_key=settings.gemini_api_key,
        redis_client=redis_client,
        groq_model=settings.groq_model,
        nim_model=settings.nim_model,
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
    expose_headers=["X-Request-ID", "Retry-After"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    # Honour a caller-supplied ID (bounded) so a frontend error report can be matched
    # to backend logs; otherwise mint one.
    rid = (request.headers.get("x-request-id") or "")[:64] or uuid.uuid4().hex[:16]
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


# Fixed-window per-client counter. Lives in Redis when it is configured, so the window
# survives redeploys and is shared across replicas; falls back to this in-process dict
# when Redis is absent or erroring (fail-open to the local brake, never to no brake).
# It is a quota-drain brake, not an access control; see docs/security.md.
_rate_window: dict[str, tuple[int, float]] = {}


def _client_key(request: Request) -> str:
    # Hugging Face Spaces terminates TLS upstream, so the socket peer is the proxy.
    # X-Forwarded-For is client-controlled and trivially spoofed, which is precisely
    # why this is a brake and not a control.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _rate_limited(request: Request, user_id: str | None = None) -> bool:
    limit = settings.chat_rate_limit_per_minute
    if limit <= 0:
        return False
    # A verified user ID can't be spoofed; the IP is the fallback when auth is off.
    key = f"user:{user_id}" if user_id else _client_key(request)
    return await _over_limit(f"rl:chat:{key}", 60, limit)


async def _over_daily_quota(user_id: str) -> bool:
    limit = settings.user_daily_query_limit
    if limit <= 0:
        return False
    return await _over_limit(f"quota:day:{user_id}", 86_400, limit)


async def _over_limit(key: str, window: int, limit: int) -> bool:
    if _redis is not None:
        try:
            count = await _redis.incr_window(key, window)
            return count > limit
        except Exception as e:
            logger.warning("Redis rate limit unavailable, using in-process window: %s", e)
    return _rate_limited_local(key, limit, window)


def _rate_limited_local(key: str, limit: int, window: int = 60) -> bool:
    now = time.time()
    count, window_start = _rate_window.get(key, (0, now))
    if now - window_start >= window:
        count, window_start = 0, now
    count += 1
    _rate_window[key] = (count, window_start)
    if len(_rate_window) > 10_000:  # bound the dict against unique-IP flooding
        for stale, (_, started) in list(_rate_window.items()):
            if now - started >= 86_400:
                _rate_window.pop(stale, None)
    return count > limit


# Admission control for agent runs; see Settings.max_concurrent_chats.
_active_chats = 0


def _try_admit() -> bool:
    # No await between check and increment, so this is atomic on the event loop.
    global _active_chats
    if settings.max_concurrent_chats > 0 and _active_chats >= settings.max_concurrent_chats:
        return False
    _active_chats += 1
    return True


def _release() -> None:
    global _active_chats
    _active_chats = max(0, _active_chats - 1)


async def _gate(request: Request) -> JSONResponse | None:
    """Auth, then per-user/IP rate limit, then daily quota, then admission. Returns
    the rejection response, or None when the request is admitted (slot taken)."""
    user_id = None
    if settings.supabase_url:
        from app.auth import AuthError, authenticate
        try:
            user = await authenticate(request, settings.supabase_url, settings.supabase_jwt_secret)
        except AuthError as e:
            logger.info("Rejected unauthenticated chat: %s", e)
            return JSONResponse(
                status_code=401, headers={"WWW-Authenticate": "Bearer"},
                content={"detail": "Please sign in to ask questions."},
            )
        user_id = user.id
        request.state.user_id = user.id
    if await _rate_limited(request, user_id):
        return _rejection(request_rate_limited=True)
    if user_id and await _over_daily_quota(user_id):
        return JSONResponse(
            status_code=429, headers={"Retry-After": "3600"},
            content={"detail": "You've reached today's question limit — it resets within 24 hours."},
        )
    if not _try_admit():
        return _rejection(request_rate_limited=False)
    return None


def _scoped_session_id(requested: str | None, user_id: str | None) -> str:
    """session_id is the checkpointer's thread_id, i.e. the key to a conversation's
    stored history. With auth on, it is namespaced by user so a caller who learns
    another user's session ID gets a fresh thread instead of their history."""
    if not user_id:
        return requested or str(uuid.uuid4())
    prefix = f"{user_id}:"
    if requested and requested.startswith(prefix):
        return requested
    return prefix + uuid.uuid4().hex


def _rejection(request_rate_limited: bool) -> JSONResponse:
    if request_rate_limited:
        return JSONResponse(
            status_code=429, headers={"Retry-After": "60"},
            content={"detail": "Too many requests — please wait a minute and try again."},
        )
    return JSONResponse(
        status_code=503, headers={"Retry-After": "15"},
        content={"detail": "The agent is busy with other questions — please retry in a few seconds."},
    )


def _cap_answer(answer: str) -> str:
    limit = settings.max_answer_chars
    if len(answer) <= limit:
        return answer
    return answer[:limit] + "\n\n*[Answer truncated.]*"


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
    """Liveness: the process is up. Cheap, touches nothing external — keep-alive pings hit this."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/ready")
async def ready():
    """Readiness: dependencies answer. 503 when the database is down, since no
    query can be served without it; Redis is reported but only degrades (cache and
    shared rate limit fall back), so it does not fail the probe."""
    checks: dict[str, str] = {}
    healthy = True
    if settings.database_url:
        try:
            from db.connection import get_connection
            async with get_connection() as conn:
                await asyncio.wait_for(conn.execute("SELECT 1"), timeout=5)
            checks["database"] = "ok"
        except Exception as e:
            logger.warning("Readiness: database check failed: %s", e)
            checks["database"] = "error"
            healthy = False
    else:
        checks["database"] = "disabled"
    if _redis is not None:
        try:
            checks["redis"] = "ok" if await asyncio.wait_for(_redis.ping(), timeout=3) else "error"
        except Exception as e:
            logger.warning("Readiness: redis check failed: %s", e)
            checks["redis"] = "error"
    else:
        checks["redis"] = "disabled"
    checks["active_chats"] = str(_active_chats)
    return JSONResponse(status_code=200 if healthy else 503, content={"ready": healthy, "checks": checks})


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    global _query_counter

    if (rejection := await _gate(request)) is not None:
        return rejection
    session_id = _scoped_session_id(req.session_id, getattr(request.state, "user_id", None))

    _query_counter += 1
    start = time.time()

    try:
        from agent.graph import run_agent
        result = await run_agent(
            user_query=req.query,
            session_id=session_id,
        )
        await _write_audit(req.query, session_id, result, int((time.time() - start) * 1000))
        return ChatResponse(
            answer=_cap_answer(result["final_report"]),
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
            answer=_INTERNAL_ERROR_ANSWER,
            citations=[],
            sql_results=None,
            session_id=session_id,
            provider="error",
        )
    finally:
        _release()


_INTERNAL_ERROR_ANSWER = "I encountered an internal error while answering that. Please try again."


async def _write_audit(query: str, session_id: str, result: dict, latency_ms: int) -> None:
    if not settings.database_url:
        return
    try:
        from db.connection import get_connection
        from db.queries import log_llm_calls, log_query
        async with get_connection() as conn:
            await log_query(
                conn,
                session_id=session_id,
                user_query=query,
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


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _describe_node(node: str, update: dict) -> str | None:
    """One human-readable progress line per finished graph node (None = don't show)."""
    if node == "planner":
        tools = [step.get("tool", "?") for step in update.get("plan", [])]
        return f"Planned: {', '.join(tools)}" if tools else "Planned: answer from context"
    if node == "executor":
        return "Gathered evidence"
    if node == "critic":
        return "Refining the answer" if update.get("_critic_verdict") == "RETRY" else "Checked the answer"
    if node == "reporter":
        return "Drafted the answer"
    return None


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """
    Server-Sent Events variant of /chat. Emits `status` events as each agent node
    finishes, then one `answer` event with the same payload /chat returns, then `done`.
    The answer is not token-streamed: the critic can reject a draft and send the
    graph back to the executor, so tokens from a draft may never be the final answer.
    """

    # Rejections happen before the stream opens so the client sees a real 429/503
    # rather than a 200 whose body happens to say "too many requests".
    if (rejection := await _gate(request)) is not None:
        return rejection
    session_id = _scoped_session_id(req.session_id, getattr(request.state, "user_id", None))
    rid = request_id_var.get()
    released = False

    def release_once() -> None:
        nonlocal released
        if not released:
            released = True
            _release()

    async def events():
        global _query_counter
        # The generator runs after the middleware has returned, so restore the ID.
        request_id_var.set(rid)
        _query_counter += 1
        start = time.time()
        try:
            from agent.graph import stream_agent
            result: dict = {}
            async for event in stream_agent(user_query=req.query, session_id=session_id):
                if event["type"] == "result":
                    result = event["data"]
                    continue
                message = _describe_node(event["node"], event["update"])
                if message:
                    yield _sse("status", {"node": event["node"], "message": message})

            await _write_audit(req.query, session_id, result, int((time.time() - start) * 1000))
            yield _sse("answer", ChatResponse(
                answer=_cap_answer(result.get("final_report") or ""),
                citations=result.get("citations", []),
                sql_results=result.get("sql_results"),
                session_id=session_id,
                provider=result.get("provider", "unknown"),
            ).model_dump())
        except Exception as e:
            logger.error("Chat stream failed: %s", e, exc_info=True)
            yield _sse("answer", ChatResponse(
                answer=_INTERNAL_ERROR_ANSWER, citations=[], sql_results=None,
                session_id=session_id, provider="error",
            ).model_dump())
        finally:
            # Also runs when the client disconnects mid-stream (GeneratorExit), so
            # a cancelled request frees its slot.
            release_once()
        yield _sse("done", {})

    # X-Accel-Buffering stops reverse proxies (HF Spaces sits behind one) from
    # holding the whole response until it completes, which would defeat streaming.
    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        # Backstop for a client that disconnects before the generator ever starts,
        # in which case its finally block never runs.
        background=BackgroundTask(release_once),
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
