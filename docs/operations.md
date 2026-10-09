# Operations runbook

Everything here runs on free tiers: GitHub Actions, Vercel Hobby, Hugging Face
Spaces (CPU basic), Neon free, Upstash free, and an optional free OTLP backend.

## Release

1. Merge to `main`. CI runs lint + unit tests, dependency audit, Playwright E2E,
   and the RAGAS gate. Vercel deploys the frontend from `main` automatically.
2. Tag the release so rollback has a named target:
   `git tag -a v1.N.0 -m "..." && git push origin v1.N.0`
3. Deploy the backend: `git push space main`. The Space rebuilds the Docker image
   (several minutes; the embedding model warms during startup).
4. Verify: `curl $HF_SPACE_URL/ready` should return
   `{"ready": true, "checks": {"database": "ok", "redis": "ok", ...}}`.

## Rollback

- **Backend:** `git push --force space v1.(N-1).0:main`, then re-check `/ready`.
  (Force is required because the Space's `main` moves backwards.)
- **Frontend:** in Vercel → Deployments, pick the previous production deployment →
  *Promote to Production*. Instant, no rebuild.
- **Database migrations** are forward-only; each file in `db/migrations/` documents
  its own rollback (see `docs/storage.md` for 003).

## Staging

- **Frontend:** every PR gets a Vercel preview URL automatically. Previews call the
  production API unless `VITE_API_URL` is overridden for the Preview environment.
- **Backend:** duplicate the Space (Hugging Face → *Duplicate this Space*, free),
  point it at the same secrets or a Neon branch (`neon branches create`, free tier
  allows branches), then add the preview origin to `ALLOWED_ORIGINS`. Not set up yet:
  it doubles Groq quota consumption against the same keys.

## Observability

| Signal | Where |
|---|---|
| Liveness / readiness | `GET /health` (no I/O), `GET /ready` (DB + Redis). Polled every 6h by `.github/workflows/scheduled.yml`; a failed run emails the repo owner. |
| Logs | JSON lines on stdout (HF Space → *Logs*). Every line has `request_id`; the same ID is returned as `X-Request-ID`. Set `LOG_FORMAT=text` locally. |
| Traces | One `agent.run` span per request, one `node.<name>` child per planner/executor/critic/reporter step (plan, critic verdict, LLM call count, providers, tokens). Enable by setting `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS` as Space secrets — e.g. Grafana Cloud free tier (OTLP gateway URL, `Authorization=Basic <base64 instanceId:token>`) or Honeycomb free (`https://api.honeycomb.io`, `x-honeycomb-team=<key>`). Unset = no-op. |
| Cost / latency | `GET /analytics/cost` (from `llm_call_log`); see `docs/cost_latency_observability.md`. |
| Model retirement | Daily `scripts/probe_models.py` in `scheduled.yml`; fails on any non-429 error from a pinned model. Fix with the `GROQ_MODEL` / `NIM_MODEL` secrets, no deploy. |

GitHub emails the repo owner on a failed scheduled run by default — that is the
alerting path. Check *Settings → Notifications → Actions* if those emails stop.

## Load shedding

`MAX_CONCURRENT_CHATS` (default 4) returns `503 Retry-After: 15` past capacity, and
`CHAT_RATE_LIMIT_PER_MINUTE` (default 20/IP) returns `429 Retry-After: 60`. Both are
env vars on the Space; changing them is a restart, not a deploy. The frontend shows
the API's `detail` text for either.

## Enabling Supabase Auth (one-time)

Order matters: if the backend requires tokens before the frontend sends them, the
live site 401s every question.

1. Create a free project at supabase.com. *Authentication → Sign In / Providers*:
   enable **GitHub** (create a GitHub OAuth app; callback URL is shown on that page)
   and keep **Email**. Leave **anonymous sign-ins off**.
2. *Authentication → URL Configuration*: Site URL = the Vercel production URL; add
   `http://localhost:5173` to Redirect URLs for dev.
3. Vercel env (Production + Preview): `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`
   (*Project Settings → API*; the anon key is public by design),
   `VITE_SUPABASE_OAUTH_PROVIDERS=github`. Redeploy and confirm the sign-in screen.
4. HF Space secret: `SUPABASE_URL`. Only if the project still uses the legacy HS256
   secret, also `SUPABASE_JWT_SECRET`. Restart, then confirm an unauthenticated
   `curl -X POST $HF_SPACE_URL/chat -d '{"query":"x"}' -H 'content-type: application/json'`
   returns 401 and a signed-in question works.

Rollback: delete `SUPABASE_URL` from the Space (API opens again), then the Vercel vars.
The built-in mailer sends ~2 emails/hour on the free tier; add custom SMTP (e.g. a
free Resend account) if magic links matter.
