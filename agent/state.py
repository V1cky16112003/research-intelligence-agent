from __future__ import annotations
"""LangGraph shared state schema for the research agent."""
from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    """Shared state passed between all agent nodes."""
    # Input
    user_query: str
    session_id: str

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

    # Output
    draft_answer: str | None
    final_report: str | None
    citations: list[dict]

    # Audit
    tools_called: list[str]
    llm_provider: str | None
    tokens_in: int
    tokens_out: int
