from __future__ import annotations
"""
LangGraph node implementations: Planner, Executor, Reporter, Critic.

Flow: START → planner → executor → reporter(draft) → critic → ┬─ RETRY → executor
                                                               └─ PASS  → END
"""
import json
import logging
import re

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

CRITIC_SYSTEM = """You are a research quality critic. Review the draft answer against the retrieved context.
Rate the answer and decide: PASS or RETRY.
- PASS: the draft is grounded in the context, cites sources, and addresses the question
- RETRY: the draft is vague, contains claims unsupported by the context, or misses the key question

Respond with JSON only (no markdown fences):
{"verdict": "PASS" or "RETRY", "reason": "brief reason", "refined_query": "improved search query if RETRY, else null"}

If RETRY, set refined_query to a more specific search query that would retrieve better evidence."""

REPORTER_SYSTEM = """You are a research report writer. Synthesize the retrieved context into a clear, cited answer.
- Be specific and factual, citing papers by title and arxiv_id when available
- If SQL results are present, include relevant statistics
- Keep the answer focused and under 400 words
- End with a brief "Sources" list if there are citations"""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Extract the first JSON object or array from text.

    Handles markdown code fences (```json ... ```) and bare JSON.
    Falls back to the stripped text if no delimited block is found.
    """
    # Try balanced brace/bracket extraction first
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()


def _build_context(chunks: list, sql: list) -> str:
    """Build a formatted context string from retrieved chunks and SQL results."""
    parts = []
    for i, chunk in enumerate(chunks[:6]):
        title = chunk.get("title", "Unknown")
        arxiv_id = chunk.get("arxiv_id", "")
        content = chunk.get("content", "")[:400]
        parts.append(f"[{i+1}] {title} ({arxiv_id})\n{content}")
    if sql:
        parts.append(f"\nSQL Analytics Results:\n{json.dumps(sql[:10], default=str, indent=2)}")
    return "\n\n".join(parts) if parts else "No relevant context found in corpus."


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

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
        plan = json.loads(_extract_json(content))
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
    """Execute all plan steps, or re-retrieve using refined_query on a critic retry.

    Normal path: runs every step in plan[current_step:], accumulating results.
    Retry path: if refined_query is set, skips the plan and does a single
                rag_retrieval with the refined query, merging new chunks into
                existing ones (deduped by id, capped at 10).
    """
    from agent.tools import TOOL_DISPATCH, rag_retrieval_tool

    user_query = state.get("user_query", "")
    tools_called = list(state.get("tools_called", []))
    tool_results_acc: list[dict] = []
    retrieved_chunks = list(state.get("retrieved_chunks", []))
    sql_results = list(state.get("sql_results", []))

    # --- Retry path: re-retrieve with the critic's refined query ---
    refined_query = state.get("refined_query")
    if refined_query:
        logger.info("Executor retry path — refined query: %s", refined_query[:80])
        try:
            result_json = await rag_retrieval_tool(query=refined_query)
            result = json.loads(result_json)
        except Exception as e:
            logger.warning("Retry retrieval failed: %s", e)
            result = {"results": [], "error": str(e)}

        # Merge new chunks into existing ones, deduped by chunk id, cap at 10
        new_chunks = result.get("results", [])
        existing_ids = {c.get("id") for c in retrieved_chunks if c.get("id")}
        for chunk in new_chunks:
            if chunk.get("id") not in existing_ids:
                retrieved_chunks.append(chunk)
                existing_ids.add(chunk.get("id"))
        retrieved_chunks = retrieved_chunks[:10]

        tools_called.append("rag_retrieval")
        tool_results_acc.append({"step": "retry", "tool": "rag_retrieval", "result": result})

        return {
            "tool_results": tool_results_acc,
            "tools_called": tools_called,
            "retrieved_chunks": retrieved_chunks,
            "sql_results": sql_results,
            "refined_query": None,  # consumed — clear for next pass
            "current_step": state.get("current_step", 0),
        }

    # --- Normal path: run all plan steps ---
    plan = state.get("plan", [])
    step_idx = state.get("current_step", 0)

    if step_idx >= len(plan):
        logger.warning("Executor: no plan steps to run (current_step=%d, plan len=%d)", step_idx, len(plan))
        return {"tool_results": [{"error": "No steps in plan"}]}

    for i, step in enumerate(plan[step_idx:], start=step_idx):
        tool_name = step.get("tool", "rag_retrieval")
        args = step.get("args", {})
        logger.info("Executing step %d: %s(%s)", i, tool_name, args)

        if tool_name not in TOOL_DISPATCH:
            logger.warning("Unknown tool %r, falling back to rag_retrieval", tool_name)
            tool_name = "rag_retrieval"
            args = {"query": user_query}

        tool_fn = TOOL_DISPATCH[tool_name]
        try:
            result_json = await tool_fn(**args)
            result = json.loads(result_json)
        except TypeError as e:
            # LLM sent unexpected arg names — fall back for retrieval tools
            logger.warning("Step %d bad args for %s (%s), using fallback", i, tool_name, e)
            if tool_name == "rag_retrieval":
                try:
                    result_json = await rag_retrieval_tool(query=user_query)
                    result = json.loads(result_json)
                except Exception as fe:
                    result = {"error": str(fe), "results": []}
            else:
                result = {"error": f"Invalid args for {tool_name}: {e}", "results": []}
        except Exception as e:
            logger.warning("Step %d tool %s raised: %s", i, tool_name, e)
            result = {"error": str(e), "results": []}

        tools_called.append(tool_name)
        tool_results_acc.append({"step": i, "tool": tool_name, "result": result})

        if tool_name == "rag_retrieval" and result.get("results"):
            retrieved_chunks = result["results"]
        elif tool_name == "sql_analytics" and result.get("results"):
            sql_results = result["results"]

    return {
        "tool_results": tool_results_acc,
        "tools_called": tools_called,
        "retrieved_chunks": retrieved_chunks,
        "sql_results": sql_results,
        "current_step": len(plan),  # all steps done
    }


async def reporter_node(state: dict) -> dict:
    """Synthesize retrieved context into a draft (and final) answer."""
    from agent.registry import get_gateway
    gw = get_gateway()

    chunks = state.get("retrieved_chunks", [])
    sql = state.get("sql_results", [])
    context = _build_context(chunks, sql)

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
        "draft_answer": answer,    # read by critic next pass
        "final_report": answer,    # served if critic PASSes
        "citations": citations,
        "tokens_in": state.get("tokens_in", 0) + resp.get("tokens_in", 0),
        "tokens_out": state.get("tokens_out", 0) + resp.get("tokens_out", 0),
    }


async def critic_node(state: dict) -> dict:
    """Evaluate draft answer against retrieved context; issue PASS or RETRY.

    On RETRY, sets refined_query so the executor re-retrieves with a better query.
    The draft_answer (written by reporter) is the real artifact being reviewed.
    """
    from agent.registry import get_gateway
    gw = get_gateway()

    chunks = state.get("retrieved_chunks", [])
    sql = state.get("sql_results", [])
    draft = state.get("draft_answer") or ""
    context = _build_context(chunks, sql)

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Query: {state['user_query']}\n\n"
                f"Draft answer:\n{draft[:800]}\n\n"
                f"Retrieved context:\n{context[:1200]}"
            ),
        },
    ]

    resp = await gw.chat(messages, temperature=0.0, max_tokens=256)
    content = resp.get("content") or '{"verdict": "PASS", "reason": "proceeding", "refined_query": null}'

    try:
        verdict = json.loads(_extract_json(content))
    except (json.JSONDecodeError, ValueError):
        verdict = {"verdict": "PASS", "reason": "json parse failed, proceeding", "refined_query": None}

    retry_count = state.get("retry_count", 0)
    new_refined_query: str | None = None

    if verdict.get("verdict") == "RETRY" and retry_count < MAX_RETRIES:
        retry_count += 1
        new_refined_query = verdict.get("refined_query") or state.get("user_query", "")
        logger.info(
            "Critic RETRY (%d/%d): %s — refined query: %s",
            retry_count, MAX_RETRIES, verdict.get("reason", ""), new_refined_query[:80],
        )
    else:
        logger.info("Critic PASS: %s", verdict.get("reason", ""))

    return {
        "critique": verdict.get("reason", ""),
        "retry_count": retry_count,
        "refined_query": new_refined_query,
        "tokens_in": state.get("tokens_in", 0) + resp.get("tokens_in", 0),
        "tokens_out": state.get("tokens_out", 0) + resp.get("tokens_out", 0),
        "_critic_verdict": verdict.get("verdict", "PASS"),
    }
