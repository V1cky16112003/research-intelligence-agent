"""
OpenTelemetry tracing for agent runs: one root span per run, one child span per
graph node, so a slow answer can be attributed to planner vs executor vs a critic
retry loop without reading logs.

Export is opt-in: set OTEL_EXPORTER_OTLP_ENDPOINT (+ OTEL_EXPORTER_OTLP_HEADERS for
auth) to any OTLP/HTTP backend — Grafana Cloud and Honeycomb both have free tiers.
Unset, spans are created against the no-op provider and cost nothing.

The root span is carried in a ContextVar rather than attached as the current span,
because stream_agent is an async generator: attaching there and detaching after a
`yield` raises "token was created in a different Context". LangGraph runs each node
in a task that copies the caller's context, so the node wrapper sees the root.
"""
from __future__ import annotations

import contextvars
import functools
import logging
import os
from contextlib import contextmanager

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("research-agent")

_run_context: contextvars.ContextVar = contextvars.ContextVar("otel_run_context", default=None)


def init_tracing(service_name: str = "research-agent") -> bool:
    """Install an OTLP exporter if configured. Returns True when tracing is live."""
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT set but opentelemetry SDK missing — tracing off")
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    logger.info("OpenTelemetry tracing enabled")
    return True


@contextmanager
def run_span(session_id: str, user_query: str):
    """Root span for one agent run. Ended explicitly, never attached (see module doc)."""
    span = tracer.start_span("agent.run", attributes={
        "session.id": session_id, "query.length": len(user_query),
    })
    token = _run_context.set(trace.set_span_in_context(span))
    try:
        yield span
    except BaseException as e:
        span.record_exception(e)
        span.set_status(trace.Status(trace.StatusCode.ERROR))
        raise
    finally:
        try:
            _run_context.reset(token)
        except ValueError:
            # Generator finalised from a different context (client disconnect).
            pass
        span.end()


def traced_node(name: str, fn):
    """Wrap an async LangGraph node so each invocation is a child span of the run."""
    @functools.wraps(fn)
    async def wrapper(state):
        with tracer.start_as_current_span(f"node.{name}", context=_run_context.get()) as span:
            span.set_attribute("agent.retry_count", state.get("retry_count", 0) or 0)
            update = await fn(state)
            if isinstance(update, dict):
                if "plan" in update:
                    span.set_attribute("agent.plan", [s.get("tool", "?") for s in update["plan"]])
                if "_critic_verdict" in update:
                    span.set_attribute("agent.critic_verdict", update["_critic_verdict"])
                # llm_calls is an accumulator; a node's update holds only its own calls.
                new_calls = update.get("llm_calls") or []
                span.set_attribute("llm.calls", len(new_calls))
                providers = sorted({c.get("provider", "?") for c in new_calls})
                if providers:
                    span.set_attribute("llm.providers", providers)
            return update
    return wrapper
