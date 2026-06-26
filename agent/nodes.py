from __future__ import annotations
"""
LangGraph node implementations: Planner, Executor, Critic, Reporter.
"""
import json
import logging

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

# System prompts — keep them tight
PLANNER_SYSTEM = """You are a research planning assistant. Given a user query, create a plan to answer it using available tools.

Available tools:
- rag_retrieval: Search ArXiv ML paper corpus semantically. Use for questions about paper content, methods, findings.
- sql_analytics: Run analytics over the papers database. Use for counting papers, trends, publication stats.
  query_type options: papers_by_month, query_volume, provider_latency, experiments
- web_search: Search the web for recent or out-of-corpus information.

Respond with a JSON array of steps:
[{"step": "description", "tool": "tool_name", "args": {"arg": "value"}}]

For simple queries, 1-2 steps. For complex ones, up to 3 steps. Always end with a synthesis step using rag_retrieval or just answer directly if the query is conversational."""

CRITIC_SYSTEM = """You are a research quality critic. Review the retrieved context and draft answer.
Rate the answer and decide: PASS or RETRY.
- PASS: answer is grounded, cites sources, addresses the question
- RETRY: answer is vague, ungrounded, or misses the key question

Respond with JSON: {"verdict": "PASS" or "RETRY", "reason": "brief reason"}
If RETRY, suggest a better tool or query in the reason."""

REPORTER_SYSTEM = """You are a research report writer. Synthesize the retrieved context into a clear, cited answer.
- Be specific and factual, citing papers by title and arxiv_id when available
- If SQL results are present, include relevant statistics
- Keep the answer focused and under 400 words
- End with a brief "Sources" list if there are citations"""


async def planner_node(state: dict) -> dict:
    """Decompose user query into a tool-execution plan."""
    from agent.registry import get_gateway
    gw = get_gateway()

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": f"Query: {state['user_query']}"},
    ]

    resp = await gw.chat(messages, temperature=0.1, max_tokens=512)
    provider = resp["provider"]
    content = resp.get("content") or "[]"

    # Parse plan JSON — be defensive
    try:
        # Strip markdown code fences if present
        clean = content.strip().strip("```json").strip("```").strip()
        plan = json.loads(clean)
        if not isinstance(plan, list):
            plan = [{"step": "search", "tool": "rag_retrieval", "args": {"query": state["user_query"]}}]
    except (json.JSONDecodeError, ValueError):
        logger.warning("Planner returned non-JSON, using default RAG plan")
        plan = [{"step": "search", "tool": "rag_retrieval", "args": {"query": state["user_query"]}}]

    return {
        "plan": plan,
        "current_step": 0,
        "llm_provider": provider,
        "tokens_in": state.get("tokens_in", 0) + resp.get("tokens_in", 0),
        "tokens_out": state.get("tokens_out", 0) + resp.get("tokens_out", 0),
    }


async def executor_node(state: dict) -> dict:
    """Execute the current plan step by calling the appropriate tool."""
    from agent.tools import TOOL_DISPATCH

    plan = state.get("plan", [])
    step_idx = state.get("current_step", 0)

    if step_idx >= len(plan):
        return {"tool_results": [{"error": "No more steps in plan"}]}

    step = plan[step_idx]
    tool_name = step.get("tool", "rag_retrieval")
    args = step.get("args", {})
    tools_called = list(state.get("tools_called", []))

    logger.info("Executing step %d: %s(%s)", step_idx, tool_name, args)

    if tool_name not in TOOL_DISPATCH:
        tool_name = "rag_retrieval"
        args = {"query": state["user_query"]}

    tool_fn = TOOL_DISPATCH[tool_name]
    result_json = await tool_fn(**args)
    result = json.loads(result_json)

    tools_called.append(tool_name)
    tool_results = [{"step": step_idx, "tool": tool_name, "result": result}]

    # Extract structured results for state
    retrieved_chunks = state.get("retrieved_chunks", [])
    sql_results = state.get("sql_results", [])

    if tool_name == "rag_retrieval" and result.get("results"):
        retrieved_chunks = result["results"]
    elif tool_name == "sql_analytics" and result.get("results"):
        sql_results = result["results"]

    return {
        "tool_results": tool_results,
        "tools_called": tools_called,
        "retrieved_chunks": retrieved_chunks,
        "sql_results": sql_results,
        "current_step": step_idx + 1,
    }


async def critic_node(state: dict) -> dict:
    """Evaluate whether retrieved context is sufficient to answer the query."""
    from agent.registry import get_gateway
    gw = get_gateway()

    # Summarise context for the critic
    chunks = state.get("retrieved_chunks", [])
    sql = state.get("sql_results", [])
    context_summary = f"Retrieved {len(chunks)} chunks. SQL results: {len(sql)} rows."
    if chunks:
        context_summary += f"\nFirst chunk: {chunks[0].get('content', '')[:200]}..."

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": f"Query: {state['user_query']}\nContext: {context_summary}"},
    ]

    resp = await gw.chat(messages, temperature=0.0, max_tokens=128)
    content = resp.get("content") or '{"verdict": "PASS", "reason": "proceeding"}'

    try:
        clean = content.strip().strip("```json").strip("```").strip()
        verdict = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        verdict = {"verdict": "PASS", "reason": "json parse failed, proceeding"}

    retry_count = state.get("retry_count", 0)
    if verdict.get("verdict") == "RETRY" and retry_count < MAX_RETRIES:
        retry_count += 1

    return {
        "critique": verdict.get("reason", ""),
        "retry_count": retry_count,
        "tokens_in": state.get("tokens_in", 0) + resp.get("tokens_in", 0),
        "tokens_out": state.get("tokens_out", 0) + resp.get("tokens_out", 0),
        "_critic_verdict": verdict.get("verdict", "PASS"),
    }


async def reporter_node(state: dict) -> dict:
    """Synthesize retrieved context into a final cited report."""
    from agent.registry import get_gateway
    gw = get_gateway()

    chunks = state.get("retrieved_chunks", [])
    sql = state.get("sql_results", [])

    # Build context string
    context_parts = []
    for i, chunk in enumerate(chunks[:6]):
        title = chunk.get("title", "Unknown")
        arxiv_id = chunk.get("arxiv_id", "")
        content = chunk.get("content", "")[:400]
        context_parts.append(f"[{i+1}] {title} ({arxiv_id})\n{content}")

    if sql:
        context_parts.append(f"\nSQL Analytics Results:\n{json.dumps(sql[:10], default=str, indent=2)}")

    context = "\n\n".join(context_parts) if context_parts else "No relevant context found in corpus."

    messages = [
        {"role": "system", "content": REPORTER_SYSTEM},
        {
            "role": "user",
            "content": f"Query: {state['user_query']}\n\nContext:\n{context}\n\nWrite a comprehensive answer.",
        },
    ]

    resp = await gw.chat(messages, temperature=0.2, max_tokens=1024)
    answer = resp.get("content") or "I was unable to generate an answer."

    # Build citations from retrieved chunks
    citations = []
    seen: set[str] = set()
    for chunk in chunks[:8]:
        arxiv_id = chunk.get("arxiv_id", "")
        if arxiv_id and arxiv_id not in seen:
            seen.add(arxiv_id)
            citations.append({
                "arxiv_id": arxiv_id,
                "title": chunk.get("title", ""),
                "authors": chunk.get("authors", []),
                "content": chunk.get("content", "")[:150],
            })

    return {
        "final_report": answer,
        "citations": citations,
        "tokens_in": state.get("tokens_in", 0) + resp.get("tokens_in", 0),
        "tokens_out": state.get("tokens_out", 0) + resp.get("tokens_out", 0),
    }
