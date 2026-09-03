from __future__ import annotations

"""LangGraph shared state schema for the research agent."""
import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    """Shared state passed between all agent nodes."""
    # Input
    user_query: str
    session_id: str
    # Set by reporter_node to this turn's user_query so the *next* invoke's
    # planner (loading this from the checkpoint) can recognize pronoun-style
    # follow-ups ("summarize that") instead of treating them as a fresh topic.
    previous_user_query: str | None

    # Planning
    plan: list[dict]          # [{step: str, tool: str, args: dict}]
    current_step: int

    # Execution
    tool_results: Annotated[list[dict], operator.add]  # accumulated across retries

    # Retrieved context
    retrieved_chunks: list[dict]   # from RAG tool
    sql_results: list[dict]        # from SQL tool

    # Critique
    critique: str | None
    retry_count: int
    refined_query: str | None   # Critic's suggested re-retrieval query on RETRY

    # Output
    draft_answer: str | None
    final_report: str | None
    citations: list[dict]

    # Audit
    tools_called: list[str]
    llm_provider: str | None
    tokens_in: int
    tokens_out: int
    # Per-call cost/latency records, tagged by node and retry status — accumulated
    # across the executor/reporter/critic retry loop for the cost/latency dashboard.
    llm_calls: Annotated[list[dict], operator.add]

    # Internal control (not checkpointed as objects, kept as plain values)
    _critic_verdict: str
