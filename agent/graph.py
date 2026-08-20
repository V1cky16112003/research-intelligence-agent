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


# Module-level compiled graph (lazy init — set up in app lifespan)
_graph = None
_checkpointer = None


async def init_graph() -> None:
    """Initialize graph with PostgresSaver checkpointer. Call at app startup."""
    global _graph, _checkpointer
    database_url = os.getenv("DATABASE_URL", "")
    if database_url:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            _checkpointer = AsyncPostgresSaver.from_conn_string(database_url)
            await _checkpointer.setup()
            logger.info("LangGraph PostgresSaver checkpointer initialized")
        except Exception as e:
            logger.warning("Could not init PostgresSaver: %s — using in-memory", e)
            _checkpointer = None
    _graph = build_graph(checkpointer=_checkpointer)
    logger.info("LangGraph agent graph compiled")


async def run_agent(
    user_query: str,
    session_id: str,
) -> dict[str, Any]:
    """
    Run the agent for a single query.

    Returns:
        {final_report, citations, sql_results, tools_called, provider, tokens_in, tokens_out}
    """
    if _graph is None:
        raise RuntimeError("Agent graph not initialized. Call init_graph() first.")

    # `retrieved_chunks`, `sql_results`, and `citations` are deliberately absent here.
    # AsyncPostgresSaver checkpoints AgentState per session_id (thread_id); previously
    # every field was reset on every call, which clobbered the checkpoint and made
    # multi-turn conversations stateless in practice — a follow-up like "summarize
    # that in one sentence" got an empty plan from the planner and then had no
    # context to work with. Omitting these three keys lets their prior-turn values
    # flow through from the checkpoint into this invoke, so the reporter still has
    # last turn's retrieved context even when the planner (correctly) emits no new
    # tool calls for a pure follow-up. Every other field is reset because it's
    # specific to this turn's planning/critique cycle, or — for tokens_in/tokens_out/
    # tools_called — because the audit log expects per-request values, not a
    # session-cumulative total.
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
