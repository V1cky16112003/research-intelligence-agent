"""Every agent run is one root span with a child span per graph node."""
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import agent.tracing as tracing


@pytest.fixture
def exporter(monkeypatch):
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(tracing, "tracer", provider.get_tracer("test"))
    return exp


@pytest.mark.asyncio
async def test_node_spans_are_children_of_the_run_span(exporter):
    async def planner(state):
        return {"plan": [{"tool": "sql_analytics"}], "llm_calls": [{"provider": "groq"}]}

    node = tracing.traced_node("planner", planner)
    with tracing.run_span("s1", "how many papers"):
        await node({"retry_count": 0})

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {"agent.run", "node.planner"}
    child, root = spans["node.planner"], spans["agent.run"]
    assert child.parent.span_id == root.context.span_id
    assert child.attributes["agent.plan"] == ("sql_analytics",)
    assert child.attributes["llm.calls"] == 1
    assert root.attributes["session.id"] == "s1"


@pytest.mark.asyncio
async def test_run_span_records_errors(exporter):
    with pytest.raises(RuntimeError):
        with tracing.run_span("s1", "q"):
            raise RuntimeError("boom")
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == trace.StatusCode.ERROR


def test_tracing_is_off_without_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert tracing.init_tracing() is False


@pytest.mark.asyncio
async def test_parenting_survives_langgraph_scheduling_inside_a_generator(exporter):
    """stream_agent opens the run span inside an async generator and LangGraph runs
    nodes in their own tasks; the node spans must still land under the run."""
    from typing import TypedDict

    from langgraph.graph import END, StateGraph

    class S(TypedDict, total=False):
        n: int

    async def step(state):
        return {"n": (state.get("n") or 0) + 1}

    g = StateGraph(S)
    g.add_node("a", tracing.traced_node("a", step))
    g.add_node("b", tracing.traced_node("b", step))
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    graph = g.compile()

    async def gen():
        with tracing.run_span("s", "q"):
            async for chunk in graph.astream({"n": 0}, stream_mode="updates"):
                yield chunk

    assert len([c async for c in gen()]) == 2
    spans = {s.name: s for s in exporter.get_finished_spans()}
    root = spans["agent.run"].context.span_id
    assert spans["node.a"].parent.span_id == root
    assert spans["node.b"].parent.span_id == root
