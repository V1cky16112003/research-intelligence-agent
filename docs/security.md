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
| Unbounded fan-out from anonymous callers | Fixed-window rate limit, `CHAT_RATE_LIMIT_PER_MINUTE` (default 20), returning `429` + `Retry-After`. Counted in Redis (`rl:chat:<ip>`, one Upstash pipeline call) so it survives redeploys and is shared across replicas; falls back to the in-process window if Redis is down. Set to `0` to disable. |
| Concurrent runs exceeding Groq's free 8000 TPM | Admission control, `MAX_CONCURRENT_CHATS` (default 4). Excess requests get `503` + `Retry-After: 15` immediately instead of queueing on provider 429s. Slots are released on completion, error, or client disconnect. |
| Runaway or injected output size | Answers capped at `MAX_ANSWER_CHARS` (default 20,000) with a visible truncation marker. |
| Frontend XSS / clickjacking | `frontend/vercel.json` sets CSP (`script-src 'self'`, `connect-src` limited to `*.hf.space`), `frame-ancestors 'none'`, HSTS, `nosniff`. Model markdown is rendered by `react-markdown`, which does not render raw HTML. |
| Vulnerable dependencies | `dependency-audit` CI job: `pip-audit` (known, unfixable findings ignored by ID with reasons in `ci.yml`) and `npm audit --audit-level=high`. |
| `Access-Control-Allow-Origin: *` | `ALLOWED_ORIGINS` allowlist. Defaults to the production frontend (`https://frontend-vert-eight-61.vercel.app`) and `http://localhost:5173`; set `ALLOWED_ORIGINS` to override. |
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

## Authentication (Supabase)

With `SUPABASE_URL` set, `/chat` and `/chat/stream` require `Authorization: Bearer
<supabase access token>`, verified locally in `app/auth.py` (JWKS for asymmetric
projects, `SUPABASE_JWT_SECRET` for legacy HS256): signature, `exp`, `aud=authenticated`
and `iss`. `alg: none` and anonymous sessions (`is_anonymous`) are rejected — anonymous
sign-ins would let anyone mint unlimited identities. Once a user is verified:

- the rate limit keys on the user ID, not the spoofable `X-Forwarded-For`;
- `USER_DAILY_QUERY_LIMIT` (default 100) caps each user's agent runs per 24h — the
  per-user token budget;
- `session_id` (the LangGraph checkpoint thread, i.e. stored conversation history) is
  namespaced `<user_id>:<uuid>`. A client-supplied ID with another user's prefix is
  replaced by a fresh one, so knowing someone's session ID does not expose their history.

Unset `SUPABASE_URL` and the API is open again, with the IP limit as the only brake
(the spoofable-XFF caveat then applies). `/health`, `/ready`, `/metrics` and
`/analytics/cost` stay public; the analytics endpoint holds only aggregates.

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

**Hugging Face Spaces caveat (verified 2026-10-07):** the Spaces proxy answers CORS preflights itself and reflects any `Origin` (`https://evil.example` came back allowed after the allowlist deployed), so the app's allowlist is only enforced off-Spaces. CORS never stopped non-browser clients anyway; on Spaces the per-client rate limit is the effective quota guard.
