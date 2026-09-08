# Security posture

What this service does and does not defend against, and why. Written during the
2026-09-03 audit; the mitigations below are in `app/main.py` unless noted.

## Threat model

The app is a public, unauthenticated demo backed by **personal free-tier API keys**
(Groq, NVIDIA NIM, Gemini) and a **512 MB Neon instance**. The realistic attacker is
not after data — the corpus is public ArXiv metadata. They are after *your quota*.
One `/chat` request fans out to roughly eight provider calls, so the endpoint is a
force multiplier: a trivial script can exhaust a day's tokens in minutes and take the
demo offline. Everything below is sized against that, not against data exfiltration.

## Mitigated

| Issue | Mitigation |
|---|---|
| Unbounded fan-out from anonymous callers | Fixed-window rate limit, `CHAT_RATE_LIMIT_PER_MINUTE` (default 20). Set to `0` to disable. |
| `Access-Control-Allow-Origin: *` | `ALLOWED_ORIGINS` allowlist. Still defaults to `*` so the deployed Vercel frontend keeps working — **set it to that frontend's URL**. |
| Unbounded prompt size inflating token spend | `ChatRequest.query` capped at 4,000 chars, `session_id` at 200. |
| Internal topology leaked in errors | `/chat` returns a generic message; the detail goes to the log. Previously it returned `str(e)`, and psycopg connection errors embed host/database/user while provider SDK errors embed request URLs. |
| `GET /analytics/cost?days=` unbounded | `Query(ge=1, le=3650)` at the edge and `_clamp_days()` in `db/queries.py`. An unclamped value raised `interval out of range` — a 500 from a query string. |
| SQL/Cypher injection | Every statement is parameterized. The single f-string in `db/queries.py` interpolates only literal filter fragments chosen by the code, never user text. Neo4j uses a fixed set of Cypher templates (`agent/tools.py`); the LLM picks a template and supplies parameters, it never writes Cypher. |
| BM25 query injection into `to_tsquery` | `_to_tsquery_safe` strips operator punctuation before `plainto_tsquery`. |

## Accepted, with reasons

**No authentication on `/chat`.** Adding it would break the live Vercel frontend,
which ships no credential. The rate limiter is a brake, not an access control. If this
ever holds anything non-public, this is the first thing to fix — an API key checked in
a dependency, with the key added to the frontend's build env.

**The rate limiter is in-process.** It resets on redeploy and does not coordinate
across replicas. `X-Forwarded-For` is client-controlled and trivially spoofed, and on
Hugging Face Spaces the socket peer is the proxy, so there is no trustworthy client
identity available at this layer. Accepted because the goal is stopping a naive script,
not a determined attacker. Upstash Redis is already a dependency if this needs to
become real.

**Prompt injection via `web_search_tool`.** DuckDuckGo snippets are attacker-writable
text that flows untreated into `_build_context` and then into the reporter prompt. A
page crafted to rank for a query can instruct the model. Not mitigated. The blast
radius is bounded — the reporter has no tools and can only emit text — so the worst
case is a wrong or manipulated answer, not an action. Fixing it properly means
delimiting and labelling untrusted spans in the context and instructing the reporter
to treat them as data.

## Verified clean

- `.env` is untracked and absent from git history.
- No `eval`, `exec`, `pickle`, `yaml.load`, `shell=True`, or `subprocess` anywhere.
- No secrets in log statements.
- Large local artifacts (`dataset/`, `graphify-out/`, `.serena/`, `.playwright-mcp/`)
  are now in `.gitignore`; before the audit they were untracked but unignored, one
  `git add -A` away from a 5.3 GB commit.
