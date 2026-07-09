# kode

A full-featured **TUI agentic coding tool** powered by **OpenRouter + Kimi K2.7**.

You give it a task; the model reads/edits files, runs commands, greps and globs
the tree, tracks a plan, and reports back — streaming into a terminal UI, with a
confirmation + diff gate before anything touches disk.

## Setup

```bash
python3 -m pip install --break-system-packages rich prompt_toolkit requests
```

On first run, kode walks you through a **setup wizard** — API key, default model,
and confirm-vs-yolo mode — and saves your choices. Re-run it anytime with `/setup`.

You can also provide the key directly:

```bash
export OPENROUTER_API_KEY=sk-or-...   # env var (takes precedence), or
kode  →  /key                         # prompts hidden, validates, saves to ~/.kode/key (600)
```

## Run

```bash
cd ~/kode
./kode                       # operate on the current directory
./kode /path/to/project      # operate on another directory
./kode --model moonshotai/kimi-k2-thinking
./kode --yolo                # skip all confirmations (auto-approve)
./kode --route               # auto-pick a model per prompt (cheap/thinking/default)
./kode --resume              # resume the most recent session for this workspace
./kode --resume mytask       # resume a named session
./kode --list-sessions       # print saved sessions and exit
```

The launch directory (or the path you pass) is the **workspace**. File tools are
sandboxed to it; `bash` runs with it as CWD.

## Features

- **Streaming** — watch the model type; reasoning is shown dimmed above the answer.
  `bash` commands stream their output live (dimmed) as they run (`KODE_BASH_QUIET=1`
  to silence); a watchdog enforces the timeout even on silent commands.
- **Diff + confirm** — every `write_file` / `edit_file` / `bash` shows a colored
  diff or command and asks `y / always / n / feedback` before running.
  `always` remembers that tool for the session; `/auto` toggles full YOLO mode.
- **Undo** — `/undo` reverts the last file write or edit (deletes newly-created files).
- **Task plan** — the model calls `todo_write`; progress renders as a live checklist.
- **Sessions & resume** — every turn autosaves to `~/.kode/sessions/auto-<workspace>.json`,
  so nothing is lost. Relaunch with `--resume` (most recent for this workspace) or
  `--resume <name>`; inside, `/resume` shows a picker and `/sessions` lists them.
  On resume, the last few messages replay so you have context. `/save [name]`
  and `/load <name>` manage named snapshots.
- **Project context** — `/init` scans the tree, README/manifests, and git status
  into the conversation.
- **Cost** — token counts and estimated $ (pulled from OpenRouter pricing) after
  every turn and via `/cost`.
- **Config** — model + YOLO preference persist to `~/.kode/config.json`.

## In-session commands

| Command         | Action                                   |
|-----------------|------------------------------------------|
| `/help`         | list commands                            |
| `/init`         | load project context into the chat       |
| `/model [id\|filter]` | switch model — blank or a filter browses the full OpenRouter catalog |
| `/route`        | toggle auto model-routing (pick per prompt) |
| `/tools`        | list callable tools                      |
| `/auto`         | toggle YOLO (no confirmations)           |
| `/undo`         | revert last file change                  |
| `/plan`         | re-show the task plan                    |
| `/compact`      | summarize history to reclaim context     |
| `/temp <0-2>`   | set sampling temperature                 |
| `/cost`         | tokens + $ + context size                |
| `/budget <usd>` | warn once session cost passes this       |
| `/export [file]`| write the conversation to a markdown file |
| `/save [name]`  | save conversation                        |
| `/sessions [prune]` | list (or prune old) saved sessions   |
| `/resume [#\|name]` | resume a session (blank = pick from list) |
| `/load <name>`  | restore conversation                     |
| `/clear`        | reset the conversation                   |
| `/exit`, Ctrl-D | quit                                     |

Multi-line message: end a line with `\` to keep typing; a blank line sends.
`!cmd` runs a shell command directly (no model turn, no tokens) — e.g. `!git status`.

## Autonomy: self-spawned agents & model switching

The agent manages its own compute. Given the right task it will, on its own:

- **Fan out sub-agents** — `spawn_agents` runs up to 4 read-only sub-agents in
  parallel (e.g. reviewing three unrelated modules at once), then merges the
  answers. `spawn_agent` handles a single delegated investigation. Each can run
  on its own model (delegate heavy research to a cheaper one, hard sub-problems
  to a thinking model).
- **Switch its own model** — `switch_model` lets it drop to a cheaper/faster model
  for mechanical work and move to a reasoning model for hard debugging or design,
  stating why each time. Validated against the live catalog. The system prompt
  includes a curated menu of recommended models so it picks sensible ids.
- **Router agent** — with `--route` (or `/route`), a dedicated lightweight router
  agent (a cheap/fast model) reads each prompt and chooses the model to run it on,
  balancing capability vs cost — cheap model for mechanical edits, a thinking model
  for hard debugging / concurrency / design, the default otherwise. It returns a
  one-line reason (`⇄ router → <model>  router: …`). Costs a fraction of a cent per
  turn; if the router call fails it falls back to a free keyword heuristic. The
  router model defaults to the cheap menu entry — override with
  `KODE_ROUTER_MODEL`. The main agent can still `switch_model` mid-task.

Both are just tools, so they respect the same cost/budget tracking; parallel
sub-agents share the token/cost accounting (thread-safe). You can still steer the
model yourself with `/model` at any time.

## Model switching

`/model` browses the whole OpenRouter catalog (~340 models):

```
/model              # prompts for a filter, shows a table, pick by number
/model kimi         # filter to matching ids, pick by number
/model free         # e.g. list free models
/model anthropic/claude-sonnet-5   # exact id → switch instantly
```

The table shows context length and $/1M in-out pricing, marks the current model,
and your choice persists to `~/.kode/config.json`. Tab-completes model ids after
`/model ` once the catalog has loaded.

## Safety & efficiency

- **Secret guard** — `read_file` refuses `.env`, `*.pem`, `id_rsa`, etc. so keys
  never get sent to the API (`KODE_ALLOW_SECRETS=1` to override). Sessions are
  written `chmod 600`.
- **Dangerous-command block** — catches common catastrophic patterns (`rm -rf /`,
  `mkfs`, `dd of=/dev/…`, fork bombs) even in YOLO mode. It's a guardrail against
  *accidents*, not a sandbox — a determined command (`bash -c …`, a script) can still
  get through, so don't run YOLO on untrusted input.
- **Stale-write guard** — if a file changed on disk since the model read it, the
  write is refused until it re-reads (no silent clobber).
- **SSRF guard** — `fetch_url` refuses private/loopback addresses
  (`KODE_ALLOW_LOCAL_FETCH=1` to override) and caps download size.
- **Accurate cost** — uses OpenRouter's authoritative per-request `cost`
  (cache + reasoning aware), not an estimate.
- **Prompt caching** — for models billed by cache breakpoints (Anthropic, Google),
  kode marks the static system prefix and the conversation tail with `cache_control`
  so long sessions reuse the cache instead of re-paying full price each turn.
  Auto-caching models (Kimi, DeepSeek) need nothing and are left untouched.
- **Per-model context window** — the auto-compaction threshold follows each model's
  real `context_length` from the catalog, so switching to a smaller model compacts
  in time instead of hitting a hard API error (`KODE_CONTEXT_LIMIT` pins it).
- **Whitespace-tolerant edits** — when an `edit_file` / `multi_edit` `old` string
  fails to match exactly, kode retries a line-aligned match that ignores indentation
  differences, so cheaper models' near-miss edits still land.
- **Cheaper context** — stale/duplicate read outputs are auto-elided each turn so
  full compaction is rare; `grep` uses ripgrep when available and skips vendor dirs.
- **Per-model temperature** — Kimi/reasoning models default to 0.6; `/temp` to tune.

## Everyday workflow

- **Checkpoints & rollback** — kode snapshots the workspace (via a *shadow* git repo
  that never touches your real history) before every turn. `/diff` shows everything
  the session changed; `/rewind [n]` undoes the last n turns — **files and
  conversation together**; `/revert` throws away all of the session's file changes.
- **One-shot / scriptable** — `kode -p "fix the failing test"` runs a single turn and
  exits. `cat error.log | kode -p "explain this"` reads stdin. `--json` emits a
  parseable result (answer, model, cost, files changed) with the UI on stderr.
- **`/retry [model]`** — re-run the last turn (rolls back its files first), optionally
  on a different model.
- **`/jobs`** — list background bash jobs with status; `/jobs kill <pid>` stops one.
- **`/allow <prefix>`** — auto-approve safe bash commands (e.g. `/allow pytest`),
  persisted per project so confirm-mode stops nagging.
- **`/usage`** — cost per day and per model across all your sessions.
- **Per-project config** — drop a `.kode.toml` or `.kode.json` in a repo to set its
  `model`, `yolo`, `budget`, `auto_route`, and `bash_allow`.
- **Project context is cached** — scanned once and refreshed only when key files
  (README, manifests) change, so startup doesn't re-read the tree every time.
- Terminal **bell** after long turns; sessions **auto-titled** from the first prompt.

## Install

```bash
pip install --user -e .        # exposes `kode` on PATH via pyproject entry point
# or just symlink the launcher:
ln -sf ~/kode/kode ~/.local/bin/kode
```
Make sure `~/.local/bin` is on your `PATH`.

## Testing

```bash
python3 -m pytest test_kode.py -q      # 60 tests, no network needed
```

## More features

- **Resilient API** — retry with exponential backoff on 429/5xx/network errors.
- **Auto-compaction** — when a request nears the context limit (`KODE_CONTEXT_LIMIT`,
  default 200k), older turns are summarized into a handoff note; `/compact` forces it.
- **Interrupt-safe** — Ctrl-C during a reply stops it cleanly; history is repaired so
  orphaned tool calls never break the next request. Autosaves on interrupt/crash/exit.
- **@file mentions** — `look at @src/app.py` inlines that file into your message;
  Tab completes both slash-commands and `@paths`.
- **Project context** — `KODE.md` / `AGENTS.md` / `CLAUDE.md` plus OS/date/workspace
  are loaded into the system prompt automatically at startup.
- **Budget guard** — `--budget 5` or `/budget 5` warns once session cost passes $5.
- **Session hygiene** — `/sessions prune [N]` trims old named sessions.

## Tools the model can call

`read_file` · `write_file` · `edit_file` (whitespace-tolerant matching) · `multi_edit`
· `list_dir` · `glob_files` · `grep` · `bash` (live output; `background=true`) ·
`fetch_url` · `web_search` (keyless DuckDuckGo) · `spawn_agent` / `spawn_agents`
(read-only sub-agents, parallel, per-agent model) · `switch_model` (agent changes its
own model) · `todo_write`

## Config via env vars

| Var                  | Default                      |
|----------------------|------------------------------|
| `OPENROUTER_API_KEY` | *(required)*                 |
| `KODE_MODEL`         | `moonshotai/kimi-k2.7-code`  |
| `KODE_WORKSPACE`     | current directory            |

## Files

- `agent.py` — REPL, streaming OpenRouter client, tool loop, diff/confirm, cost, sessions
- `tools.py` — tool implementations + JSON schema advertised to the model
- `kode` — launcher
