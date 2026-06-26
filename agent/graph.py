from __future__ import annotations
"""
LangGraph state machine: Planner → Executor → Critic → Reporter.

Graph flow:
  START → planner → executor → critic → reporter → END
                      ↑___________| (if RETRY and retry_count < MAX_RETRIES)
"""
import logging
import os
from typing import Any

from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import planner_node, executor_node, critic_node, reporter_node, MAX_RETRIES

logger = logging.getLogger(__name__)


def _should_retry(state: dict) -> str:
    """Conditional edge: retry execution or proceed to reporter."""
    verdict = state.get("_critic_verdict", "PASS")
    retry_count = state.get("retry_count", 0)
    if verdict == "RETRY" and retry_count < MAX_RETRIES:
        logger.info("Critic says RETRY (attempt %d/%d)", retry_count, MAX_RETRIES)
        return "executor"
    return "reporter"


def build_graph(checkpointer=None):
    """Build and compile the LangGraph research agent."""
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("reporter", reporter_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "critic")
    workflow.add_conditional_edges(
        "critic",
        _should_retry,
        {"executor": "executor", "reporter": "reporter"},
    )
    workflow.add_edge("reporter", END)

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

    initial_state = {
        "user_query": user_query,
        "session_id": session_id,
        "plan": [],
        "current_step": 0,
        "tool_results": [],
        "retrieved_chunks": [],
        "sql_results": [],
        "critique": None,
        "retry_count": 0,
        "draft_answer": None,
        "final_report": None,
        "citations": [],
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
