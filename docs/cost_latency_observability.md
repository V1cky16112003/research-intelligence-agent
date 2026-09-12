# Cost/latency observability

Per-LLM-call cost and latency, tagged by LangGraph node (Planner/Executor/Critic/
Reporter) and by whether the call happened during a Critic-triggered retry.

## How it works

- **Instrumentation point:** `LLMGateway.chat()` (`agent/gateway.py`) takes `node`
  and `is_retry` tags from the caller and attaches `cost_usd` (from
  `PRICING_PER_1M_TOKENS`, published per-provider rates) and `latency_ms`
  (wall time, including retries/backoff within that call) to every response.
  This is the single instrumentation point — no logging is sprinkled across nodes.
- **Node tagging:** each of `planner_node`, `reporter_node`, `critic_node` in
  `agent/nodes.py` passes its own name as `node=` and
  `is_retry=state.get("retry_count", 0) > 0` (true once the Critic has issued at
  least one RETRY for this query). `executor_node` issues no direct LLM call
  itself; the LLM-based reranker it invokes via `rag_retrieval_tool` is tagged
  `node="executor"` at the gateway call site but is not yet threaded back into
  `llm_calls` (its result travels through a JSON-string tool boundary) — a known
  gap, not a blocker, since reranker cost is small relative to planner/reporter/
  critic and every rerank call is `cache=True`.
- **Storage:** `AgentState.llm_calls` (`agent/state.py`) is an
  `Annotated[list[dict], operator.add]` field, so it accumulates one record per
  call across the full Planner→Executor→Reporter→Critic(→retry)→... run.
  `run_agent()` returns it; `/chat` in `app/main.py` bulk-inserts it into the new
  `llm_call_log` table (`db/schema.sql`, `db/migrations/002_llm_call_log.sql`)
  right next to the existing `query_audit_log` write, keyed by the same
  `session_id` so per-call rows tie back to the per-query audit row.
- **Dashboard:** `GET /analytics/cost?days=30` (`app/main.py`) returns
  `{"by_node": [...], "retry_overhead": [...]}` from the `llm_cost_latency` and
  `llm_retry_overhead` Postgres views — the same "SQL view + thin read query"
  pattern already used for `experiments` and `provider_latency`. The same data is
  also reachable through the agent itself via `sql_analytics` with
  `query_type: "cost_by_node"` or `"retry_overhead"`, matching how
  `provider_latency`/`experiments` are already exposed to the Planner.

## Reading the numbers (once data accumulates)

No production `llm_call_log` rows exist yet as of this writing — the table is new.
Once `/chat` traffic accumulates, `GET /analytics/cost` and the `experiments` /
`provider_latency` views (see `CLAUDE.md`'s existing incident notes) together
should make these questions answerable:

- **Which node dominates cost/latency?** Reporter and Critic both run on every
  query; Planner runs once; Executor's own node total is near-zero (its cost
  lives inside the tool call, not tracked yet — see gap above). Given gpt-oss-120b
  pricing (`$0.15`/`$0.75` per 1M prompt/completion tokens) and Reporter's larger
  `max_tokens=1024` vs Critic's `max_tokens=256`, Reporter is expected to be the
  single most expensive node per query even before any retries.
- **Retry overhead as its own line:** `llm_retry_overhead` isolates cost/latency
  for `is_retry=true` rows. Since a RETRY re-runs Executor→Reporter→Critic (not
  just Critic), each retry roughly doubles that query's Reporter+Critic cost and
  adds a full round-trip of latency (potentially through the Groq→NIM→Gemini
  fallback chain per call — see `groq-llama-decommissioned` incident history).
  If `retry_overhead` shows retries are a large fraction of total cost, the
  highest-leverage fix is tightening `CRITIC_SYSTEM`'s PASS bar or lowering
  `MAX_RETRIES`, not re-routing models.
- **Model-routing recommendation:** NVIDIA NIM (`openai/gpt-oss-20b`) is
  the known slow tier (prior incidents recorded 50s mean / 234s p95 when calls
  cascaded to it) and Gemini 2.5 Flash has the highest completion-token price in
  `PRICING_PER_1M_TOKENS` ($2.50/1M). If `by_node` shows a meaningful share of
  calls landing on NIM or Gemini rather than Groq, that's a signal to page on —
  it means the cascade's primary tier is failing, not just occasionally
  overflowing, and should be investigated the same way the
  `groq-llama-decommissioned` incident was (check for 404s/model deprecation on
  Groq's primary model before assuming normal fallback volume).
