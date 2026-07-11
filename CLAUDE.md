# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

kode — a TUI agentic coding tool on OpenRouter (Kimi K2.7 default). Three real modules: `agent.py` (REPL, streaming LLM client, tool loop, modes, sub-agents/swarms, sessions), `tools.py` (tool implementations + the JSON schemas advertised to the model), `checkpoint.py` (shadow-git snapshots). `timer.py` is unrelated local scratch and is gitignored.

## Commands

```bash
python3 -m pytest test_kode.py -q          # full suite (~100 tests), offline — no network or API key
python3 -m pytest test_kode.py -q -k swarm # single test / group by keyword
./kode [workspace]                          # run from the repo; `kode` is on PATH via ~/.local/bin symlink
kode -p "prompt" [--json]                   # one-shot mode (needs a real key; costs money)
```

No linter or build step. Python ≥3.11; on this machine (3.14/CachyOS) install deps with `pip --break-system-packages` or pacman. Bump `__version__` in `agent.py` **and** `version` in `pyproject.toml` together.

Live testing: the OpenRouter key lives in `~/.kode/key` (or `$OPENROUTER_API_KEY`) — never in the repo. Real one-shot runs cost real money; keep test prompts small and prefer `moonshotai/kimi-k2.5` for cheap live checks.

## Architecture

**`agent.py` — everything flows through `Agent.messages`** (OpenAI chat format, the single source of truth; sessions are just this list serialized). `run_turn()` per user turn: optional router pick → `repair_history()` → `prune_context()` → `maybe_compact()` → checkpoint snapshot → stream loop (`_stream()` → execute tool calls via `_run_tool()` → repeat, capped at `MAX_STEPS`).

**Two LLM call paths, both streaming — keep it that way:**
- `_stream()` — the main agent's turn, with live Rich rendering.
- `_complete()` — headless calls (sub-agents, router, swarm planner/synthesis, compaction). It streams under the hood **on purpose**: OpenRouter's non-streaming endpoint intermittently stalls forever (keep-alive bytes defeat the read timeout; observed live with kimi models). Do not "simplify" it back to `stream=False`.

Both funnel through `_post()`, which layers three guards: a hard-budget gate (raises `BudgetExceeded` before spending when `budget_hard` is set), retry with backoff (`_post_retry`), and single-hop provider failover — retries-exhausted raises `_ProviderDown` and the request is retried once on `_fallback_model()` (first different menu entry); hard HTTP errors (400/401/…) raise plain `RuntimeError` and are deliberately **not** failed over.

**Approval modes** (`MODES` = confirm/auto/plan/yolo) gate mutating tools in `_run_tool`: plan mode rejects everything in `tools.MUTATING` outright; auto mode auto-approves `FILE_TOOLS` but not bash; `Agent.yolo` is a read-only property derived from `mode`. Plan mode is deliberately never persisted to config (`_apply_mode`) and injects/removes a system prompt on toggle (`set_mode`). Config stores `"mode"`; a legacy `"yolo"` boolean is still read in `main()`.

**Sub-agents & swarms**: `spawn()` runs a tool loop restricted to `READONLY_TOOLS`, bounded by `SUBAGENT_STEPS` *and* a `SUBAGENT_TIMEOUT` wall-clock deadline (checked between steps; returns partial findings on expiry). `spawn_many()` / `swarm()` fan out threads — token/cost counters are updated under `_acct_lock`, only the main thread touches `Agent.messages`, and completions print live via `as_completed`. `spawn_swarm` = planner call (on `router_model`) → parallel `spawn()` workers (≤ `MAX_SWARM`) → synthesis call; worker findings are clipped to `SWARM_FINDING_CLIP` chars before synthesis, and each stage degrades gracefully (fallback to single agent / raw findings). `spawn_swarm` is intentionally absent from `READONLY_TOOLS` so workers can't nest swarms, and the main agent gets **one swarm per turn** (`_swarms_this_turn`, reset in `run_turn`) — without the cap a thorough model loops "audit more" indefinitely (observed live: 3 swarms/12 workers, no answer delivered).

**Image input**: `expand_mentions()` returns `(text, images)`; image mentions become base64 data-URL parts on the user message (content becomes a parts list). Anything reading user content must go through `_msg_text()` (handles both string and parts-list forms). `prune_context()` strips images from all but the newest user turn so a later switch to a non-vision model can't 400 on old history.

**`MODEL_MENU` order matters**: entry 0 is the default role and `_role_models()` picks the first descriptions matching "reason/think" and "cheap/fast" — keep the three Kimi entries at the top. The menu is filtered against the live catalog at startup; `switch_model` validates against the *full* catalog, not just the menu.

**History invariants**: every assistant `tool_calls` entry must be followed by matching tool results or the next request 400s — `repair_history()` enforces this (stubs for missing results, drops orphaned/unknown-id/duplicate tool results) and must be called after any interrupt path. `prune_context()` elides stale read-tool outputs (they're re-fetchable); `compact()` summarizes everything before the last user turn.

**`tools.py` contract**: tool functions take kwargs matching their JSON schema and report failures as `"ERROR: ..."` strings (a few things still raise — `_safe_path` throws `ValueError` on workspace escape — and `_run_tool`'s catch-all converts those). All paths go through `_safe_path()` (workspace sandbox; absolute paths allowed if inside); guards: secret-file glob block, stale-write mtime check (`_READ_MTIMES`), dangerous-command regex, SSRF re-check on every redirect hop in `_safe_get()` (hand-rolled redirect loop, 6-hop cap, size cap). Fixed-host fetchers (GitHub/PyPI/MDN/…) call `requests` directly *by design*; only arbitrary-URL follows (wayback snapshot, man7 fallback) route through `_safe_get`. Other specifics: `read_file` refuses a *full* read >400KB but allows explicit `offset`/`limit` slices, output clipped at 20k chars; foreground `bash` runs in its own process group with a watchdog timer that `killpg`s the whole tree; background jobs log to `.kode-jobs/job-*.log` (auto-pruned); `fetch_pdf` lazy-imports `pypdf` (deliberately not a hard dep); `undo_last()` backs `/undo` and is **not** advertised to the model. Module-level mutable state (`TODOS`, `JOBS`, `_UNDO_STACK`, `_READ_MTIMES`) is process-global — `set_workspace()` clears only the mtimes.

Adding a tool means: function + `TOOL_FUNCS` entry + `TOOLS_SPEC` schema, plus in `agent.py`: `READONLY_TOOLS` (if sub-agents may use it), `_render_tool_call` detail, `_CONTENT_TOOLS` (if output deserves a multi-line preview), and `MUTATING` in tools.py if it changes disk state (that's what triggers the mode/confirm gate). Tools needing `Agent` state (spawn*, switch_model) are special-cased in `_run_tool`, not registered in `TOOL_FUNCS`.

**`checkpoint.py`**: a shadow git repo (`GIT_DIR` = `~/.kode/shadow/<sha1-of-workspace-path>`, `GIT_WORK_TREE` = the workspace, committer `kode@localhost`) so `/rewind`/`/revert` never touch the user's real git history; the workspace's own `.gitignore` is honored plus coarse excludes (`node_modules`, `.venv`, …). Auto-disabled when the workspace is home or filesystem root — *unless* that directory is itself a git repo (`KODE_ALLOW_BROAD_CKPT=1` also forces it on). `snapshot()` returns `None` when nothing changed; `setup()` runs `git gc --auto` once per session so the shadow repo can't grow unbounded. `reset()` is `git reset --hard`: it removes tracked files added since the snapshot but never touches untracked/excluded ones.

**UI conventions**: the global `console` is `Console(highlight=False)` — all color comes from explicit styles, never rich's auto-highlighter; keep it that way or dim gutter output turns garish. Literal `[...]` in markup strings must be escaped (`\\[n]`) or rich silently eats them. The prompt (`read_message(session, agent)`) is mode-colored and the `PromptSession` has a persistent bottom toolbar from `_toolbar_factory(agent)` — the toolbar callable re-runs on every render and must never raise or block (it reads cached fields like `_ctx_limit` only). Tool calls render with a category glyph from `_tool_ui()`.

**Config precedence** (resolved in `main()`): CLI flag → `.kode.toml`/`.kode.json` in the workspace → `~/.kode/config.json` → built-in default.

## Testing conventions

Tests are fully offline: network paths are covered by monkeypatching `Agent._complete` / `_post` / `_stream` or `requests` objects, never by hitting the API. The `ws` fixture sandboxes `tools.WORKSPACE` into `tmp_path`; config-touching tests monkeypatch `CONFIG_DIR`/`CONFIG_FILE`/`KEY_FILE`. Interactive prompts are monkeypatched via `agent.console.input`. When adding a feature, mirror this: unit-test the logic offline, and if you live-test, do it as a `kode -p` one-shot in a scratch directory.
