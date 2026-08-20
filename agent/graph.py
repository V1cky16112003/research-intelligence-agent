from __future__ import annotations

"""
LangGraph state machine: Planner → Executor → Reporter → Critic.

Graph flow:
  START → planner → executor → reporter(draft) → critic → ┬─ RETRY → executor
                                   ↑_________________________| └─ PASS  → END

Reporter writes a real draft before Critic, so Critic evaluates the actual answer
against context. On RETRY, the Critic sets refined_query and the executor re-retrieves
with a better query before re-drafting.
"""
import logging
import os
from typing import Any

from langgraph.graph import END, StateGraph

from agent.nodes import MAX_RETRIES, critic_node, executor_node, planner_node, reporter_node
from agent.state import AgentState

logger = logging.getLogger(__name__)


def _should_retry(state: dict) -> str:
    """Conditional edge: re-execute with refined query, or finish."""
    verdict = state.get("_critic_verdict", "PASS")
    retry_count = state.get("retry_count", 0)
    if verdict == "RETRY" and retry_count < MAX_RETRIES:
        logger.info("Critic says RETRY (attempt %d/%d)", retry_count, MAX_RETRIES)
        return "executor"
    return "end"


def build_graph(checkpointer=None):
    """Build and compile the LangGraph research agent."""
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("reporter", reporter_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "reporter")
    workflow.add_edge("reporter", "critic")
    workflow.add_conditional_edges(
        "critic",
        _should_retry,
        {"executor": "executor", "end": END},
    )

    return workflow.compile(checkpointer=checkpointer)


# Module-level fallback graph — only used when DATABASE_URL is unset (no
# checkpointing possible/needed). When DATABASE_URL is set, run_agent() builds a
# fresh graph with a freshly-opened checkpointer connection on every call; see
# the comment on init_graph() for why.
_graph = None
_checkpointer_schema_ready = False  # avoids re-running the idempotent-but-not-free setup() every call


async def init_graph() -> None:
    """Prepare the graph. Call at app startup.

    When DATABASE_URL is set, the checkpointer connection is intentionally NOT
    opened here and held for the app's lifetime — an earlier version of this
    function did that (entering AsyncPostgresSaver.from_conn_string()'s context
    manager once at startup), and it broke in production: Neon's serverless free
    tier silently drops long-idle connections, and a single held raw psycopg
    connection has no reconnect logic, so the first idle period after a while
    caused every subsequent /chat request to fail instantly with "the connection
    is closed" until the process restarted. AsyncPostgresSaver.from_conn_string()
    is an @asynccontextmanager specifically so it can be entered fresh per use;
    run_agent() does that per call instead. Without DATABASE_URL, falls back to
    a single in-memory (uncheckpointed) graph built once here.
    """
    global _graph
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        _graph = build_graph(checkpointer=None)
        logger.info("LangGraph agent graph compiled (no DATABASE_URL — uncheckpointed)")
    else:
        logger.info("LangGraph agent will use a per-request PostgresSaver checkpointer")


async def run_agent(
    user_query: str,
    session_id: str,
) -> dict[str, Any]:
    """
    Run the agent for a single query.

    Returns:
        {final_report, citations, sql_results, tools_called, provider, tokens_in, tokens_out}
    """
    global _checkpointer_schema_ready
    database_url = os.getenv("DATABASE_URL", "")

    # `retrieved_chunks`, `sql_results`, `citations`, and `previous_user_query` are
    # deliberately absent here. AsyncPostgresSaver checkpoints AgentState per
    # session_id (thread_id); previously every field was reset on every call, which
    # clobbered the checkpoint and made multi-turn conversations stateless in
    # practice. Omitting these four keys lets their prior-turn values flow through
    # from the checkpoint into this invoke: the planner sees what the last query
    # was (so it can recognize a pronoun-style follow-up like "summarize that"
    # instead of launching a fresh, wrong-topic search for it — this was observed
    # live, where such a follow-up searched for "summarize" and silently replaced
    # good context with irrelevant results) and the reporter still has last turn's
    # retrieved context when the planner correctly emits no new tool calls. Every
    # other field is reset because it's specific to this turn's planning/critique
    # cycle, or — for tokens_in/tokens_out/tools_called — because the audit log
    # expects per-request values, not a session-cumulative total.
    initial_state = {
        "user_query": user_query,
        "session_id": session_id,
        "plan": [],
        "current_step": 0,
        "tool_results": [],
        "critique": None,
        "retry_count": 0,
        "refined_query": None,
        "draft_answer": None,
        "final_report": None,
        "tools_called": [],
        "llm_provider": None,
        "tokens_in": 0,
        "tokens_out": 0,
        "_critic_verdict": "PASS",
    }

    config = {"configurable": {"thread_id": session_id}}

    try:
        if database_url:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
                if not _checkpointer_schema_ready:
                    await checkpointer.setup()
                    _checkpointer_schema_ready = True
                graph = build_graph(checkpointer=checkpointer)
                final_state = await graph.ainvoke(initial_state, config=config)
        else:
            if _graph is None:
                raise RuntimeError("Agent graph not initialized. Call init_graph() first.")
            final_state = await _graph.ainvoke(initial_state, config=config)
    except Exception as e:
        logger.error("Agent graph failed: %s", e, exc_info=True)
        raise

    return {
        "final_report": final_state.get("final_report", ""),
        "citations": final_state.get("citations", []),
        "sql_results": final_state.get("sql_results", []) or None,
        "tools_called": final_state.get("tools_called", []),
        "provider": final_state.get("llm_provider", "unknown"),
        "tokens_in": final_state.get("tokens_in", 0),
        "tokens_out": final_state.get("tokens_out", 0),
    }
