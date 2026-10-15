# UI/UX plan — Research Intelligence Agent

Status: implemented 2026-10-15 (uncommitted at time of writing); 43 Playwright + 19 unit + 183 backend tests green. Scope: `frontend/` plus one small backend endpoint. Free-tier only; no new paid services, no new runtime dependencies in the frontend.

## 1. Where we are

`frontend/src/App.jsx` (364 lines) is one inline-styled component: a single centered column, one conversation persisted in `localStorage` (`research-agent-history-v1`), a "New chat" button that **destroys** the conversation, no history, no indication of the daily quota until a `429` lands. Light on structure, dark `#0f0f0f` + blue.

The backend already enforces two limits the user can't see: 20 requests/min (`CHAT_RATE_LIMIT_PER_MINUTE`) and a per-user daily cap of 100 agent runs (`USER_DAILY_QUERY_LIMIT`, Redis key `quota:day:<user_id>`, 24 h fixed window started by the first request). A 429 or 503 is the only signal today.

## 2. Research: what the three products do

Observed live in Chrome (Claude, Gemini) on 2026-10-15; ChatGPT was blocked by site permission in the browser extension, so its notes come from general knowledge of the product and are lower confidence.

| Pattern | Claude | ChatGPT | Gemini | Take |
|---|---|---|---|---|
| Sidebar | Collapsible; New, Search, nav items, Pinned, then recents grouped by date ("Oct 13", "Older"); profile + plan at the bottom | Collapsible; New chat, Search, library; recents; profile at bottom | Collapsible; New chat, nav; one flat "All" list with a filter icon; account at bottom | Collapsible left rail, **date-grouped** recents, pinned, search, account at bottom |
| Usage limit | Settings → Usage: "Current session · resets at 5:20 PM" and "This week · resets Sunday", progress bars, plain-language verdict ("On track…") | Limit surfaces as an inline banner when hit, with reset time | Mostly invisible until hit | **Always-visible, quiet meter** with reset time; verdict text only when it matters (low / out) |
| Empty state | Greeting by name, one centered composer | Centered composer, suggestion chips | "Where should we start?", three suggestion chips with icons | Greeting + composer + example prompts that *show capability* (RAG / SQL / graph) |
| Composer | Rounded, grows with text, model label, mic | Rounded, grows, attach | Pill, attach, mic | Auto-growing, Enter to send, Stop in place of Send, char counter near the 4000 cap |
| Messages | User: soft bubble right. Assistant: **no bubble**, full-width prose | Same | Assistant prose, no bubble | Assistant answers as full-width prose, user as a subdued bubble |
| Mobile | Sidebar becomes a drawer | Drawer | Drawer | Drawer with backdrop, `Esc` and swipe-free close |

What none of them do, and we can: this is a **research** agent whose answers are grounded in 50k arXiv papers. That is our identity.

## 3. Identity: "the lab notebook"

Not another grey ChatGPT clone and not Claude's cream. We lean into research:

- **Palette** — warm near-black "ink" (`#14110f` family), paper-white text, one **amber/oxide accent** (`#e0a458`) for the brand and the usage meter, teal (`#5fb3a1`) for citations/sources. No blue-purple gradient.
- **Type** — a serif for headings and the greeting (Newsreader via Google Fonts, falling back to Georgia), system sans for body, a mono (ui-monospace) for arXiv IDs, counts and metadata. Paper-like, not product-like.
- **Signature elements**
  1. **Run trace** — the agent's four nodes (Planner → Executor → Critic → Reporter) rendered as a compact stepper that fills in during streaming, replacing the generic "Working…" box. It makes the architecture visible and is the one thing the other three can't show. After the answer it collapses to a one-line "Planned: sql_analytics · 3 sources · 2.1s" toggle.
  2. **Source cards** — citations as small cards with the arXiv ID in mono, authors, a link, and a number the answer text can refer to, instead of an undifferentiated list.
  3. **Budget gauge** — the usage meter drawn as a thin amber bar with a tick, mono numerals ("73 / 100 left · resets in 6h 12m"), sitting in the sidebar footer.
- **Motion** — one 150 ms ease for hover/focus, a stepper fill, and `fade-in` on messages; all disabled under `prefers-reduced-motion`.

Non-goals: light theme (dark only for now, tokens make it a later 1-file change), voice, file attachments, sharing/export, server-side history sync (see §6).

## 4. Functionality

### F1 — Multi-conversation history (core)
- Sidebar lists conversations, grouped **Today / Yesterday / Previous 7 days / Older**, newest first. Active one highlighted.
- Click to switch (aborts any in-flight request first). **New chat keeps the old conversation** (today it deletes it).
- Title = first user message, truncated to 48 chars; rename inline (double-click or menu).
- Delete with inline confirm (no `window.confirm`; modal dialogs are hostile to the e2e suite and to a11y).
- Pin/unpin; pinned section above the groups.
- Search box filters by title and message text (client-side, case-insensitive).
- Persistence: `localStorage`, per user. Key `research-agent-conversations-v2:<user_id|anon>`; the existing `research-agent-history-v1` blob is migrated into one conversation on first load and then removed. Caps: 30 conversations, 50 messages each; oldest unpinned evicted. All reads/writes in try/catch (storage can be full or disabled — history stays best-effort, as today).
- Each conversation keeps its own `sessionId` so the backend's LangGraph checkpoint continues the right thread.

### F2 — Usage limit left (core)
- **Backend:** `GET /usage` (auth required when `SUPABASE_URL` is set; same `authenticate` path). Read-only: it must *not* consume quota. Response:
  ```json
  {"enabled": true, "limit": 100, "used": 27, "remaining": 73, "resets_in_seconds": 21540,
   "per_minute_limit": 20}
  ```
  `{"enabled": false}` when the limit is `<= 0` or there is no user (auth off). Reads the same `quota:day:<user_id>` key; needs a read-only `get` (exists) plus a new `ttl(key)` on the Redis clients; falls back to the in-process `_rate_window` entry when Redis is unset or down. Must never 500 — on error return `{"enabled": false}`.
- **Frontend:** `useUsage()` fetches on mount and after every completed or failed `/chat/stream`, and on a 429 sets `remaining = 0` immediately without waiting for the network. Sidebar footer shows the gauge; colour shifts amber → red under 20% left; text "Out of questions · resets in 3h 04m" at 0; composer shows the same inline notice and disables Send when out. Hidden entirely when `enabled: false`.
- The count is a **questions** budget, not tokens — say "questions", never "tokens".

### F3 — Chat surface
- Assistant answers full-width, no bubble; user messages as right-aligned subdued bubble.
- Auto-growing textarea (max ~8 rows), Enter sends, Shift+Enter newline, `Stop` replaces Send in flight, `↑`/`■` icons keep current `aria-label`s (`Send`, `Stop generating`, `Your question`) — the existing e2e tests select on them.
- Run trace (§3) fed by the existing `status` SSE events (`{node, message}`); no backend change.
- Source cards, per-answer provider chip ("via groq"), Copy button, SQL result table restyled with the new tokens (class names `data-table`/`table-wrap` stay).
- Empty state: serif greeting, composer, and four example prompts tagged by capability (Semantic search · SQL analytics · Co-author graph).
- Auto-scroll only while the user is near the bottom, so reading an earlier answer isn't yanked away.

### F4 — Shell & responsiveness
- Layout: CSS grid, 280 px sidebar + fluid main; main column max ~760 px for readable measure.
- Collapse toggle (state persisted); under 768 px the sidebar is an overlay drawer with backdrop, closes on selection, `Esc`, or backdrop click.
- Account block at the bottom of the sidebar: email, Sign out (moved from the header).
- Keyboard: `Cmd/Ctrl+K` focuses sidebar search, `Cmd/Ctrl+Shift+O` new chat, `Esc` closes drawer. Visible focus rings; every control reachable by keyboard.
- Replace inline `styles` object with `theme.css` tokens + per-component CSS. No CSS framework (no new dependency).

## 5. Architecture and file ownership

New modules, so five people can work without touching the same file:

| File | Owner | Contract |
|---|---|---|
| `app/main.py` (+ `agent/redis_client.py`, `tests/test_usage.py`) | **T1 backend** | `GET /usage` as in F2 |
| `frontend/src/conversations.js` + `frontend/src/conversations.test.js` | **T2 state** | Pure functions, no React: `loadStore(userId)`, `saveStore(userId, store)`, `createConversation()`, `appendMessage(store, id, msg)`, `renameConversation`, `deleteConversation`, `togglePin`, `groupByDate(list, now)`, `searchConversations(list, q)`, `deriveTitle(messages)`. Store shape: `{ activeId, items: [{id, title, pinned, createdAt, updatedAt, sessionId, messages}] }`. Test with `node --test` (zero deps). |
| `frontend/src/useUsage.js` | **T2 state** | `useUsage() → {usage, refresh, markExhausted}`; `usage` is the F2 JSON or `null`. Uses `authHeader()` from `auth.js`. |
| `frontend/src/Sidebar.jsx` + `sidebar.css`, `UsageMeter.jsx` | **T3 sidebar** | `<Sidebar groups activeId query onQuery onSelect onNew onRename onDelete onPin collapsed onToggle user onSignOut usage />`; `<UsageMeter usage />`. Pure presentational — no storage, no fetch. |
| `frontend/src/theme.css`, `Message.jsx`, `Composer.jsx`, `RunTrace.jsx`, `Welcome.jsx`, `markdown.css`, `index.html` | **T4 chat surface** | `<Message msg />`, `<Composer value onChange onSend onStop loading usage />`, `<RunTrace steps active />`, `<Welcome onPick />`. Defines the tokens everyone else uses (`--bg`, `--surface`, `--line`, `--text`, `--muted`, `--accent`, `--source`, `--danger`, `--radius`, `--font-*`). |
| `frontend/src/App.jsx` | **Lead** | Wires store + hooks + components; owns SSE and `sendMessage` (logic unchanged). Done in phase 2, once the parts exist. |
| `frontend/e2e/history.spec.js`, `usage.spec.js`, updates to `chat.spec.js`/`auth.spec.js` | **T5 QA** | Phase 2. Playwright with `/chat/stream` and `/usage` mocked via `page.route`; desktop + mobile (drawer) projects. |

Tokens contract first: T4 writes `theme.css` in its first minutes and T3 only consumes `var(--…)`; until then T3 uses the names above.

**Phases**
1. Parallel: T1, T2, T3, T4 (disjoint files).
2. Lead integrates `App.jsx`, runs `npm run build` and the existing e2e; T5 writes new e2e against the integrated app and reviews a11y/mobile; fixes go back to the owning file's author.
3. Lead: docs (`CLAUDE.md` frontend section, `docs/operations.md` note on `/usage`), full verification, commit.

## 6. Decisions and what is deliberately left out

- **History is client-side.** Server-side history would add a table to a Neon DB with 171 MB of headroom, a new authenticated CRUD surface, and a migration, to buy cross-device sync nobody asked for. Trade-off accepted: clearing browser data loses history, and the LangGraph checkpoint on the server is not browsable. Revisit if cross-device matters.
- **Usage is a daily question count**, not tokens, because that is the only thing the backend actually enforces per user. The per-minute limit stays invisible; a 429 for it keeps its existing inline message.
- No light theme, no export/share, no attachments, no streaming of answer tokens (declined earlier — see production-readiness notes).
- No framework or component-library install; no router (one view, the active id lives in state).

## 7. Definition of done

- Existing e2e + new e2e green on desktop and mobile; `ruff` + `pytest` green; `npm run build` green.
- `GET /usage` never consumes quota and never 500s (tests prove both).
- Reload keeps all conversations; a v1 blob migrates; two users on one browser do not see each other's history.
- Sidebar usable at 375 px and with keyboard only.
- No new dependencies in `package.json` or `requirements.txt`.
