# kode

A **TUI agentic coding tool** powered by **OpenRouter** (Kimi K2.7 by default).

You give it a task; the model reads and edits files, runs commands, greps the
tree, tracks a plan, and reports back — streaming into a terminal UI, with a
diff + confirmation gate before anything touches disk. It manages its own
compute: it can switch its own model mid-task, delegate to sub-agents, or fan
a broad goal out to a swarm of parallel workers.

## Quick start

```bash
python3 -m pip install --break-system-packages rich prompt_toolkit requests
pip install --user -e .        # exposes `kode` on PATH via the pyproject entry point
# or just symlink the launcher:  ln -sf ~/kode/kode ~/.local/bin/kode
```

On first run kode walks you through a **setup wizard** — API key, default
model, approval mode — and saves your choices. Re-run it anytime with `/setup`.
The key can come from either place:

```bash
export OPENROUTER_API_KEY=sk-or-...   # env var (takes precedence), or
kode  →  /key                         # hidden prompt, validates, saves to ~/.kode/key (chmod 600)
```

## Running kode

```bash
kode                         # operate on the current directory
kode /path/to/project        # operate on another directory
kode --model moonshotai/kimi-k2-thinking
kode --mode auto             # approval mode: confirm / auto / plan / yolo
kode --plan                  # start in read-only plan mode   (= --mode plan)
kode --auto                  # file edits auto-approved       (= --mode auto)
kode --yolo                  # skip all confirmations (--no-yolo forces it off)
kode --route                 # a router agent picks the model per prompt
kode --budget 5              # warn once the session passes $5
kode --resume                # resume the most recent session for this workspace
kode --resume mytask         # resume a named session
kode --list-sessions         # print saved sessions and exit
kode -p "fix the failing test"        # one-shot: run a single turn and exit
cat error.log | kode -p "explain"     # one-shot from stdin
kode -p "..." --json                  # machine-readable result (answer, cost, files changed)
```

The launch directory (or the path you pass) is the **workspace**. File tools
are sandboxed to it; `bash` runs with it as CWD. Launching in your home
directory or filesystem root gets you a warning — new files would scatter and
per-turn rollback switches off there; make a project folder first.

## Approval modes

How much you review before anything touches disk. Switch with `/mode <name>`,
toggle with `/auto` · `/plan` · `/yolo`, or start with the CLI flags above.

| Mode      | Behaviour                                                          |
|-----------|--------------------------------------------------------------------|
| `confirm` | (default) review every file write / edit / bash command            |
| `auto`    | file edits auto-approved; bash still asks (allowlist via `/allow`) |
| `plan`    | read-only: the model researches and proposes a numbered plan       |
| `yolo`    | everything auto-approved                                           |

In confirm mode every mutation shows a colored diff (rendered through the
*real* edit engine, so whitespace-tolerant matches land exactly as previewed)
and asks `y / always / n / feedback` — `always` remembers that tool for the
session, and typed feedback goes straight back to the model.

In **plan mode** mutating tools are rejected, so the model investigates with
read-only tools and presents an implementation plan. After each plan turn kode
asks: `e` executes it (dropping back to your previous mode), `x` executes in
auto mode, Enter keeps planning. Plan mode is never saved as your default —
`confirm` / `auto` / `yolo` persist to config, plan is per-session.

## Autonomy

The agent manages its own compute. Given the right task it will, on its own:

- **Fan out sub-agents** — `spawn_agent` delegates a single read-only
  investigation; `spawn_agents` runs up to 4 in parallel when the questions are
  independent. Each can run on its own model (heavy research on a cheap model,
  a hard sub-problem on a thinking model). Sub-agents can read/grep/glob/
  search/fetch but never write.
- **Spin up swarms** — `spawn_swarm` fans a broad goal out to up to **10
  parallel read-only workers**: a cheap planner model splits the goal into
  independent angles, the workers investigate concurrently (optionally all on
  a cheap model for breadth), and a synthesis pass merges everything into one
  de-duplicated report. Planning or synthesis failures degrade gracefully
  (single sub-agent / raw findings). Use it for codebase audits, system
  mapping, or surveying a design space; `spawn_agents` is better when you
  already know the exact 2–4 questions.
- **Switch its own model** — `switch_model` drops to a cheaper model for
  mechanical work or escalates to a reasoning model for hard debugging,
  stating why each time. Ids are validated against the live catalog; the
  system prompt carries a curated ~20-model menu (Kimi, Claude, GPT, Gemini,
  Grok, DeepSeek, Qwen, GLM, and more) so it picks sensible ones.
- **Route per prompt** — with `--route` (or `/route`), a lightweight router
  agent on a cheap model reads each prompt and picks the model to run it on,
  with a one-line reason (`⇄ router → <model>`). Falls back to a free keyword
  heuristic if the router call fails; bare follow-ups (`ok`, `continue`) skip
  re-routing. Override the router's own model with `KODE_ROUTER_MODEL`.

These are all ordinary tools, so they respect the same budget tracking;
parallel workers share thread-safe token/cost accounting. You can steer with
`/model` at any time.

## Models

`/model` browses the whole OpenRouter catalog (~340 models):

```
/model              # prompts for a filter, shows a table, pick by number
/model kimi         # filter to matching ids
/model free         # e.g. list free models
/model anthropic/claude-sonnet-5   # exact id → switch instantly
```

The table shows context length and $/1M pricing and marks the current model;
your choice persists to `~/.kode/config.json`. Model ids tab-complete after
`/model ` once the catalog has loaded, and a failed catalog fetch is cached
briefly so a flaky network doesn't stall every lookup. The auto-compaction
threshold follows each model's real context window, and temperature defaults
per model family (0.6 for Kimi/reasoning models; `/temp` to tune).

## In-session commands

| Command             | Action                                                        |
|---------------------|---------------------------------------------------------------|
| `/help`             | list commands                                                 |
| `/setup`            | re-run the first-time setup wizard                            |
| `/key [sk-or-…]`    | set/validate your OpenRouter key (saved to `~/.kode/key`)     |
| `/init`             | load project context into the chat                            |
| `/model [id\|filter]` | switch model — blank or a filter browses the full catalog   |
| `/route`            | toggle auto model-routing                                     |
| `/mode [name]`      | show / set approval mode: confirm · auto · plan · yolo        |
| `/auto` `/plan` `/yolo` | toggle that approval mode                                 |
| `/tools`            | list callable tools                                           |
| `/undo`             | revert last file change                                       |
| `/diff`             | show all file changes this session                            |
| `/rewind [n]`       | undo last n turns — files **and** conversation                |
| `/revert`           | discard all of this session's file changes                    |
| `/retry [model]`    | re-run the last turn (optionally on another model)            |
| `/jobs [kill <pid>]`| list / kill background bash jobs                              |
| `/allow <prefix>`   | auto-approve bash commands starting with prefix               |
| `/todos`            | re-show the task todo list                                    |
| `/compact`          | summarize history to reclaim context                          |
| `/temp <0-2>`       | set sampling temperature                                      |
| `/cost`             | tokens + $ + context size this session                        |
| `/usage`            | cost per day / per model across all sessions                  |
| `/budget <usd>`     | warn once session cost passes this                            |
| `/export [file]`    | write the conversation to a markdown file                     |
| `/save [name]`      | save conversation                                             |
| `/sessions [prune]` | list (or prune old) saved sessions                            |
| `/resume [#\|name]` | resume a session (blank = pick from a list)                   |
| `/load <name>`      | restore conversation                                          |
| `/clear`            | reset the conversation                                        |
| `/exit`, Ctrl-D     | quit                                                          |

Also: `@src/app.py` in a message inlines that file (Tab completes paths);
`!cmd` runs a shell command directly with no model turn; end a line with `\`
for multi-line input; Ctrl-C stops a reply cleanly.

## Checkpoints, sessions & workflow

- **Checkpoints & rollback** — kode snapshots the workspace via a *shadow* git
  repo (never touches your real history) before every turn. `/diff` shows
  everything the session changed; `/rewind [n]` undoes the last n turns —
  files and conversation together; `/revert` discards all the session's file
  changes; `/retry` rolls back the last turn and re-runs it. Disabled in the
  home/root dir; the shadow repo repacks once per session so it can't grow
  unbounded.
- **Sessions** — every turn autosaves to `~/.kode/sessions/` (chmod 600), so a
  crash or Ctrl-C loses nothing. `--resume` reopens the latest for the
  workspace, `/resume` shows a picker, `/save` + `/load` manage named
  snapshots, `/sessions prune` trims old ones. Sessions are auto-titled from
  the first prompt, and the last few messages replay on resume.
- **Context economy** — stale/duplicate read outputs are elided each turn;
  near the model's context limit, older turns are summarized into a handoff
  note automatically (`/compact` forces it). Anthropic/Google models get
  `cache_control` breakpoints so long sessions reuse the prompt cache;
  auto-caching models (Kimi, DeepSeek) are left untouched.
- **Task plan** — the model tracks multi-step work via `todo_write`, rendered
  as a live checklist (`/todos` re-shows it).
- **Cost** — OpenRouter's authoritative per-request cost (cache- and
  reasoning-aware) shown after every turn; `/cost`, `/usage`, and `/budget`
  for tracking. Project docs (`KODE.md` / `AGENTS.md` / `CLAUDE.md`) load into
  the system prompt automatically; `/init` scans the tree and git status in.
  Terminal bell after long turns.

## Safety guards

- **Secret guard** — `read_file` refuses `.env`, `*.pem`, `id_rsa`, etc. so
  keys never reach the API (`KODE_ALLOW_SECRETS=1` to override).
- **Dangerous-command block** — catches common catastrophic patterns
  (`rm -rf /`, `mkfs`, `dd of=/dev/…`, fork bombs) even in yolo mode. A
  guardrail against *accidents*, not a sandbox — don't run yolo on untrusted
  input.
- **Scoped auto-approve** — `/allow <prefix>` only green-lights a *single*
  command with that prefix; anything with `;`, `&&`, `|`, backticks,
  redirects, or `$(…)` still stops for confirmation.
- **Stale-write guard** — if a file changed on disk since the model read it,
  the write is refused until it re-reads (no silent clobber).
- **SSRF guard** — `fetch_url` refuses private/loopback/metadata addresses and
  re-checks every redirect hop (`KODE_ALLOW_LOCAL_FETCH=1` to override);
  download size is capped.
- **Path sandbox** — file tools can't escape the workspace; big-file reads are
  refused with a pointer to slice with `offset`/`limit`.
- **Resilient API** — retry with backoff on 429/5xx/network errors; internal
  completions (sub-agents, router, compaction) stream under the hood because
  OpenRouter's non-streaming endpoint can stall indefinitely on some
  providers. Ctrl-C is interrupt-safe: history is repaired so orphaned tool
  calls never break the next request.

## Tools the model can call

`read_file` · `write_file` · `edit_file` (whitespace-tolerant matching) ·
`multi_edit` · `list_dir` · `glob_files` · `grep` (ripgrep when available,
skips vendor dirs) · `bash` (live output, `background=true`, `/jobs` to
manage) · `fetch_url` · `web_search` (keyless DuckDuckGo) · `todo_write` ·
`spawn_agent` / `spawn_agents` / `spawn_swarm` (read-only sub-agents & swarms,
per-agent model) · `switch_model`

### Fetchers

Targeted retrieval from well-known sources (read-only; usable by sub-agents).
Fixed-host API calls go direct; anything that chases an arbitrary URL is
routed through the same SSRF-guarded helper as `fetch_url`:

- `fetch_github` — repo file contents, issues, PRs, or releases (`owner/repo`); uses `gh auth token` when present
- `fetch_docs` — package metadata from PyPI / npm / crates.io (ecosystem auto-detected)
- `fetch_error` — Stack Overflow search with links + accepted answer for the top hit
- `fetch_readme` — README for a GitHub `owner/repo` or a PyPI package name
- `fetch_json` — fetch JSON and drill in with a key path like `items[0].name`
- `fetch_mdn` — MDN Web Docs search (JS/CSS/HTTP references)
- `fetch_wayback` — closest Internet Archive snapshot of a URL, stripped to text
- `fetch_rfc` — plain text of an IETF RFC by number (optional line slice)
- `fetch_manpage` — Unix man page (local `man` first, then man7.org)
- `fetch_pdf` — download a PDF and extract text (optional page range; needs `pypdf`)

## Configuration

**Precedence:** explicit CLI flag → per-project `.kode.toml` / `.kode.json` →
global `~/.kode/config.json` (what the wizard writes) → built-in default. So
`kode --model X` always overrides the saved default, and `--no-yolo` overrides
a config that turned yolo on.

Per-project config can set `model`, `mode`, `budget`, `auto_route`, and
`bash_allow` (the `/allow` list is saved there automatically).

| Env var                  | Default          | Effect                                              |
|--------------------------|------------------|-----------------------------------------------------|
| `OPENROUTER_API_KEY`     | *(or ~/.kode/key)* | API key (env overrides the saved file)            |
| `KODE_WORKSPACE`         | current dir      | workspace root the tools are sandboxed to           |
| `KODE_CONTEXT_LIMIT`     | per-model        | pins the context window / compaction threshold      |
| `KODE_ROUTER_MODEL`      | cheap menu entry | model the router + swarm planner run on             |
| `KODE_BASH_QUIET`        | off              | `=1` silences live `bash` output echo               |
| `KODE_ALLOW_SECRETS`     | off              | `=1` lets `read_file` open `.env`/key files         |
| `KODE_ALLOW_LOCAL_FETCH` | off              | `=1` lets `fetch_url` hit private/loopback hosts    |
| `KODE_ALLOW_BROAD_CKPT`  | off              | `=1` enables checkpoints in the home/root dir       |

## Development

```bash
python3 -m pytest test_kode.py -q      # 99 tests, no network needed
```

- `agent.py` — REPL, streaming OpenRouter client, tool loop, modes, routing, swarms, sessions
- `tools.py` — tool implementations + JSON schemas advertised to the model
- `checkpoint.py` — shadow-git checkpoints for `/diff`, `/rewind`, `/revert`
- `test_kode.py` — offline test suite
- `kode` — launcher
