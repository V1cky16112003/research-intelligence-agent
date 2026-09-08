from __future__ import annotations

"""
LangGraph node implementations: Planner, Executor, Reporter, Critic.

Flow: START → planner → executor → reporter(draft) → critic → ┬─ RETRY → executor
                                                               └─ PASS  → END
"""
import inspect
import json
import logging
import re

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

# System prompts — keep them tight
PLANNER_SYSTEM = """You are a research planning assistant. Given a user query, create a plan to answer it using available tools.

Available tools:
- rag_retrieval: Search ArXiv ML paper corpus semantically. Use for questions about paper content, methods, findings.
- sql_analytics: Run analytics over the papers database. Every result carries a "summary" with the totals already computed — use those numbers, never add up the rows yourself.
  query_type options:
    - corpus_stats: total papers, chunks, authors, categories and the corpus date range — use for "how many papers are there", "how big is the corpus", "what does the corpus cover". Takes no filters.
    - papers_by_category: paper counts per ArXiv category, largest first — use for "which categories", "how many cs.LG papers". Optional "year" and "limit".
    - papers_by_year: publication counts per year — use for "papers per year", "how many were published in 2015". Optional "category".
    - papers_by_month: monthly publication counts — use for month-level publication trends. Optional "category" and "year".
    - top_authors: most prolific authors — use for "who publishes most". Optional "category" and "limit".
    - query_volume: recent daily user query volume from the audit log — use for "how many questions/queries have been asked"
    - provider_latency: LLM provider p95 latency stats — use for "how fast/slow does the system respond"
    - experiments: RAGAS eval metrics summary (faithfulness, relevancy) — use for "how well does the system perform", evaluation quality
    - cost_by_node: $ cost and latency per agent node (planner/executor/critic/reporter) per day — use for "how much does this cost", "which node is slow/expensive"
    - retry_overhead: cost/latency caused by Critic-triggered retries vs the happy path — use for "how much do retries cost"
- web_search: Search the web for recent or out-of-corpus information.
- graph_query: Answer relational questions via the co-authorship/category knowledge graph.
  query_type options (pass "value" arg with the author name or category code):
    - papers_by_author: other papers by a given author — use for "what else has X published"
    - coauthors: an author's co-authors — use for "who has X worked with", "collaborators of X"
    - papers_by_category: papers in a given category (e.g. "cs.LG") — use for "what papers are in category X", "papers about subfield X" when the user names a category code, not when they describe a topic (use rag_retrieval for topic descriptions)

Respond with a JSON array of steps:
[{"step": "description", "tool": "tool_name", "args": {"arg": "value"}}]

Only use argument names listed above for each tool — do not invent extra ones.

For simple queries, 1-2 steps. For complex ones, up to 3 steps. Emit only steps that actually fetch data: a separate report-writing stage already synthesizes the final answer, so do NOT append a trailing "synthesize"/"summarize" step. In particular, never add a rag_retrieval step with empty args — for a pure statistics question a single sql_analytics step is the whole plan.

If the current query is a follow-up to the previous question shown below (uses a pronoun like "that"/"it", or asks to elaborate, clarify, rephrase, or summarize the prior answer) and answering it needs no new information, respond with an empty plan: []. Do NOT search for words in the follow-up itself (e.g. do not treat "summarize that" as a query about summarization) — the already-retrieved context from the previous turn is reused automatically when the plan is empty."""

CRITIC_SYSTEM = """You are a research quality critic. Review the draft answer against the retrieved context.
Rate the answer and decide: PASS or RETRY.
- PASS: the draft is grounded in the context and addresses the question. If the context contains
  SQL Analytics Results, an answer built from those numbers is fully grounded and should PASS even
  though it cites no paper titles — SQL analytics questions (counts, trends, stats) have no papers
  to cite by nature, so "no paper citations" is NOT a reason to RETRY when SQL results are present.
  A figure taken from the SQL Analytics Summary is correct by definition: those totals are computed
  in the database, so PASS a draft that quotes them even if the individual rows shown do not
  visibly add up to it. Never RETRY because you could not re-derive a number by hand.
- RETRY: the draft is vague, contains claims unsupported by the context, or misses the key question.
  If the context is marked as truncated, absent evidence is not contradicting evidence — do not
  RETRY on that basis alone.

Respond with JSON only (no markdown fences):
{"verdict": "PASS" or "RETRY", "reason": "brief reason", "refined_query": "improved search query if RETRY, else null"}

If RETRY, set refined_query to a more specific search query that would retrieve better evidence."""

REPORTER_SYSTEM = """You are a research report writer. Answer the question using ONLY the information in the provided context.
- Do NOT add facts, claims, or details from your training knowledge that are not explicitly present in the context
- If the context does not contain enough information to answer, say so explicitly rather than filling gaps from memory
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


def _filter_tool_args(tool_fn, args: dict) -> dict:
    """Drop planner-invented kwargs the tool does not declare.

    The planner is an LLM, so it emits plausible-but-undeclared argument names
    (observed live: sql_analytics with "year" and "agg"). Calling the tool with those
    raised TypeError, which the executor converted into an empty result — the tool
    reported as "called" while returning nothing, which reads downstream as "the
    corpus has no data" rather than "the call was malformed". Tools that declare
    **kwargs opt out and receive everything.
    """
    if not isinstance(args, dict):
        return {}
    try:
        params = inspect.signature(tool_fn).parameters
    except (TypeError, ValueError):
        return args
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return args
    accepted = {k: v for k, v in args.items() if k in params}
    dropped = set(args) - set(accepted)
    if dropped:
        logger.info("Dropping unsupported args %s for %s", sorted(dropped), getattr(tool_fn, "__name__", tool_fn))
    return accepted


# The critic reviews a bounded slice of the context to keep its prompt small. The
# bound used to be a blind context[:1200], which on any SQL query returning more
# than a handful of rows cut off mid-array — so the critic compared the draft's
# totals against a fragment of the evidence, "found" a mismatch every single pass,
# and burned all MAX_RETRIES cycles on a draft that was correct. Two things fix it:
# keep the authoritative summary whole, and label the cut so absent evidence is not
# mistaken for unsupported claims.
CRITIC_CONTEXT_CHARS = 4000

# Rows shown per aux tool (graph_query, web_search). The graph tool already caps
# its own result sets, but web_search is an external feed with no such guarantee,
# so the context builder bounds it independently rather than trusting the source.
MAX_AUX_ROWS = 25


def _critic_context(context: str, sql_summary: dict | None = None) -> str:
    """Bound the critic's view of the context without cutting away the numbers."""
    if len(context) <= CRITIC_CONTEXT_CHARS:
        return context
    notice = "\n\n[Context truncated for review. Anything above is complete and authoritative; do not treat material missing here as unsupported.]"
    if sql_summary:
        head = (
            "SQL Analytics Summary (authoritative totals, complete):\n"
            + json.dumps(sql_summary, default=str, indent=2)
        )
        separator = "\n\n"
        budget = max(0, CRITIC_CONTEXT_CHARS - len(head) - len(separator) - len(notice))
        return f"{head}{separator}{context[:budget]}{notice}"
    return context[:CRITIC_CONTEXT_CHARS] + notice


def _call_record(resp: dict) -> dict:
    """Extract the cost/latency fields the gateway attached to a chat() response
    into a flat record for the llm_calls accumulator (see state.py)."""
    return {
        "node": resp.get("node", "unknown"),
        "provider": resp.get("provider"),
        "model": resp.get("model"),
        "tokens_in": resp.get("tokens_in", 0),
        "tokens_out": resp.get("tokens_out", 0),
        "cost_usd": resp.get("cost_usd", 0.0),
        "latency_ms": resp.get("latency_ms", 0),
        "is_retry": resp.get("is_retry", False),
        "cached": resp.get("cached", False),
    }


def _build_context(
    chunks: list,
    sql: list,
    sql_summary: dict | None = None,
    aux_results: list | None = None,
) -> str:
    """Build a formatted context string from every tool's output."""
    parts = []
    for i, chunk in enumerate(chunks[:8]):
        title = chunk.get("title", "Unknown")
        arxiv_id = chunk.get("arxiv_id", "")
        content = chunk.get("content", "")[:600]
        parts.append(f"[{i+1}] {title} ({arxiv_id})\n{content}")
    if sql_summary:
        # First, and separately, because it is the answer. Every sql_analytics query
        # type computes its totals in Postgres precisely so the reporter never has to
        # sum a list — the old context put only raw rows here, and the reporter's
        # hand-summing produced 3,098 for a corpus of 50,000.
        parts.append(
            "\nSQL Analytics Summary (authoritative totals — use these numbers directly, "
            f"do not recompute them from the rows below):\n{json.dumps(sql_summary, default=str, indent=2)}"
        )
    if sql:
        # No truncation here — the SQL tool itself already bounds every result set
        # (db.queries.MAX_ANALYTICS_ROWS) and states in its summary when a bound bit.
        parts.append(f"\nSQL Analytics Results:\n{json.dumps(sql, default=str, indent=2)}")
    for entry in aux_results or []:
        # graph_query / web_search. Same shape as the SQL block: summary first
        # (it carries totals and, for author lookups, the stored name variants
        # that were actually matched), then the rows.
        label = {
            "graph_query": "Knowledge Graph Results",
            "web_search": "Web Search Results",
        }.get(entry.get("tool", ""), f"{entry.get('tool', 'Tool')} Results")
        rows = entry.get("results") or []
        if not rows:
            continue
        block = [f"\n{label}:"]
        if entry.get("summary"):
            block.append(json.dumps(entry["summary"], default=str, indent=2))
        block.append(json.dumps(rows[:MAX_AUX_ROWS], default=str, indent=2))
        parts.append("\n".join(block))
    return "\n\n".join(parts) if parts else "No relevant context found in corpus."


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def planner_node(state: dict) -> dict:
    """Decompose user query into a tool-execution plan."""
    from agent.registry import get_gateway
    gw = get_gateway()

    previous_user_query = state.get("previous_user_query")
    user_content = f"Query: {state['user_query']}"
    if previous_user_query:
        user_content = f"Previous question in this conversation: {previous_user_query}\n\n{user_content}"

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    resp = await gw.chat(messages, temperature=0.1, max_tokens=512, node="planner")
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
        "llm_calls": [_call_record(resp)],
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
    sql_summary = state.get("sql_summary") or {}
    aux_results = list(state.get("aux_results", []))

    # --- Retry path: re-retrieve with the critic's refined query ---
    refined_query = state.get("refined_query")
    if refined_query:
        if sql_results:
            # sql_results already answer the question — more semantic search can't
            # improve an SQL-grounded answer, and merging in irrelevant chunks only
            # pollutes the context the reporter uses to draft its next answer.
            logger.info("Executor retry path — sql_results already present, skipping re-retrieval")
            return {
                "tool_results": tool_results_acc,
                "tools_called": tools_called,
                "retrieved_chunks": retrieved_chunks,
                "sql_results": sql_results,
                "sql_summary": sql_summary,
                "aux_results": aux_results,
                "refined_query": None,
                "current_step": state.get("current_step", 0),
            }

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
            "sql_summary": sql_summary,
            "aux_results": aux_results,
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
        args = _filter_tool_args(tool_fn, args)
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
            sql_summary = result.get("summary") or {}
        elif result.get("results"):
            # graph_query, web_search, and anything added later. Without this
            # branch their output reached tool_results (the audit trail) and
            # stopped there, so the reporter was asked to answer from a context
            # that never included the rows the tool had just fetched.
            aux_results.append({
                "tool": tool_name,
                "summary": result.get("summary") or {},
                "results": result["results"],
            })

    return {
        "tool_results": tool_results_acc,
        "tools_called": tools_called,
        "retrieved_chunks": retrieved_chunks,
        "sql_results": sql_results,
        "sql_summary": sql_summary,
        "aux_results": aux_results,
        "current_step": len(plan),  # all steps done
    }


async def reporter_node(state: dict) -> dict:
    """Synthesize retrieved context into a draft (and final) answer."""
    from agent.registry import get_gateway
    gw = get_gateway()

    chunks = state.get("retrieved_chunks", [])
    sql = state.get("sql_results", [])
    context = _build_context(chunks, sql, state.get("sql_summary"), state.get("aux_results"))

    messages = [
        {"role": "system", "content": REPORTER_SYSTEM},
        {
            "role": "user",
            "content": f"Query: {state['user_query']}\n\nContext:\n{context}\n\nAnswer using only the context above. Do not introduce information not present in the context.",
        },
    ]

    is_retry = state.get("retry_count", 0) > 0
    resp = await gw.chat(messages, temperature=0.0, max_tokens=1024, node="reporter", is_retry=is_retry)
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
        "previous_user_query": state.get("user_query"),  # for next turn's planner
        "tokens_in": state.get("tokens_in", 0) + resp.get("tokens_in", 0),
        "tokens_out": state.get("tokens_out", 0) + resp.get("tokens_out", 0),
        "llm_calls": [_call_record(resp)],
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
    sql_summary = state.get("sql_summary")
    context = _build_context(chunks, sql, sql_summary, state.get("aux_results"))

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Query: {state['user_query']}\n\n"
                f"Draft answer:\n{draft[:800]}\n\n"
                # The critic used to see context[:1200] while the reporter saw all of
                # it. On any SQL query returning more than a handful of rows that cut
                # landed mid-array, so the critic "checked" the draft's totals against
                # a fragment, always found a mismatch, and issued RETRY every pass
                # until MAX_RETRIES — three extra reporter+critic round trips per
                # query, guaranteed, for a draft that was fine. Grounding checks need
                # the same evidence the draft was written from.
                f"Retrieved context:\n{_critic_context(context, sql_summary)}"
            ),
        },
    ]

    is_retry = state.get("retry_count", 0) > 0
    resp = await gw.chat(messages, temperature=0.0, max_tokens=256, node="critic", is_retry=is_retry)
    content = resp.get("content") or '{"verdict": "PASS", "reason": "proceeding", "refined_query": null}'

    try:
        verdict = json.loads(_extract_json(content))
    except (json.JSONDecodeError, ValueError):
        verdict = {"verdict": "PASS", "reason": "json parse failed, proceeding", "refined_query": None}

    retry_count = state.get("retry_count", 0)
    new_refined_query: str | None = None
    final_verdict = verdict.get("verdict", "PASS")

    if final_verdict == "RETRY" and retry_count < MAX_RETRIES:
        candidate = (verdict.get("refined_query") or "").strip()
        current_query = (state.get("user_query") or "").strip()
        if candidate and candidate != current_query:
            retry_count += 1
            new_refined_query = candidate
            logger.info(
                "Critic RETRY (%d/%d): %s — refined query: %s",
                retry_count, MAX_RETRIES, verdict.get("reason", ""), candidate[:80],
            )
        else:
            # No distinct refined query — rag_retrieval is deterministic, so
            # retrying with the same (or no) query would repeat the exact same
            # search and can never produce a different result. Treat as PASS
            # instead of burning a full retry cycle for zero possible benefit.
            logger.info(
                "Critic RETRY requested but refined_query is empty or identical "
                "to the original — treating as PASS"
            )
            final_verdict = "PASS"
    else:
        logger.info("Critic PASS: %s", verdict.get("reason", ""))

    return {
        "critique": verdict.get("reason", ""),
        "retry_count": retry_count,
        "refined_query": new_refined_query,
        "tokens_in": state.get("tokens_in", 0) + resp.get("tokens_in", 0),
        "tokens_out": state.get("tokens_out", 0) + resp.get("tokens_out", 0),
        "_critic_verdict": final_verdict,
        "llm_calls": [_call_record(resp)],
    }
