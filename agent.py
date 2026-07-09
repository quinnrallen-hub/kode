#!/usr/bin/env python3
"""kode — a full-featured TUI agentic coding tool (OpenRouter + Kimi K2.7).

Features
  · streaming responses (reasoning shown dimmed); Ctrl-C stops a reply cleanly
  · diff previews + confirmation before any write / edit / multi_edit / shell command
  · /undo, per-tool "always allow", or full auto (--yolo) mode
  · tools: read/write/edit/multi_edit/list/glob/grep/bash(+background)/fetch_url/
    todo_write/spawn_agent/spawn_agents (parallel read-only sub-agents)/switch_model
  · autonomy: spawns its own sub-agents and switches its own model when the work warrants
  · retry+backoff on transient API errors; auto-compaction near the context limit
  · @file mentions inline file contents; Tab completes commands and paths
  · project docs (KODE.md/AGENTS.md/CLAUDE.md) + live env auto-loaded into context
  · continuous autosave, resume, save/load; token + $ cost + budget tracking

Usage
  export OPENROUTER_API_KEY=sk-or-...
  ./kode [workspace] [--model ID] [--yolo] [--budget USD] [--resume [name]]

Type /help inside for commands.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import json
import os
import platform
import random
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import checkpoint
import tools

__version__ = "0.3.0"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_MODEL = "moonshotai/kimi-k2.7-code"
MAX_STEPS = 50
SUBAGENT_STEPS = 20
MAX_RETRIES = 5
_CTX_ENV = os.environ.get("KODE_CONTEXT_LIMIT")  # explicit pin overrides the catalog
CONTEXT_LIMIT = int(_CTX_ENV or "200000")        # fallback when the catalog is unknown
SESSION_KEEP = 40  # cap on timestamped autosaves kept by prune
CONFIG_DIR = Path(os.path.expanduser("~/.kode"))
SESSION_DIR = CONFIG_DIR / "sessions"
CONFIG_FILE = CONFIG_DIR / "config.json"
KEY_FILE = CONFIG_DIR / "key"
PROJECT_DOCS = ("KODE.md", "AGENTS.md", "CLAUDE.md")
READONLY_TOOLS = {"read_file", "grep", "glob_files", "list_dir", "fetch_url",
                  "web_search"}
MAX_PARALLEL_AGENTS = 4

# Curated models the agent is told it can switch to / delegate to. Any that
# aren't in the live OpenRouter catalog are filtered out before use.
MODEL_MENU = [
    ("moonshotai/kimi-k2.7-code",  "strong coding default"),
    ("moonshotai/kimi-k2-thinking", "deep reasoning — hard debugging / design"),
    ("moonshotai/kimi-k2.5",        "cheaper & faster — simple mechanical edits"),
    ("anthropic/claude-sonnet-5",   "strong general-purpose alternative"),
    ("openai/gpt-5.1",              "alternative frontier model"),
    ("google/gemini-2.5-pro",       "long-context alternative"),
]

console = Console()

BASE_PROMPT = """You are kode, a terminal coding agent operating inside the user's \
workspace directory. You solve real software tasks using the provided tools.

Guidelines:
- Inspect the project with read_file/grep/glob_files/list_dir before changing anything. Never guess file contents.
- When you need current documentation, an error message's meaning, or a reference you don't know the URL for, use web_search, then fetch_url to read the best result.
- For multi-step work, call todo_write with a short plan and keep it updated as you finish steps.
- Prefer edit_file for surgical changes; multi_edit for several edits to one file; write_file for new files.
- Verify your work by running tests or commands via bash when it makes sense. Use bash background=true for long-running processes.
- Delegate large read-only investigations to spawn_agent to keep your own context small.
  When a task splits into several INDEPENDENT questions, use spawn_agents to run them in parallel.
- Manage your own model. Call switch_model when the work changes character: drop to a cheaper/
  faster model for mechanical edits and boilerplate; move to a 'thinking' model for hard
  debugging, tricky design, or when you're stuck. Always say why in one line.
- You may run a sub-agent on a different model than yourself (e.g. delegate heavy research to a
  cheaper model, or a hard sub-problem to a thinking model).
- Keep prose tight. Report what you did, not what you're about to do.
- When the task is complete, stop calling tools and give a one-paragraph summary.
"""


def available_menu() -> list[tuple[str, str]]:
    """MODEL_MENU filtered to ids that actually exist in the catalog."""
    ids = {m["id"] for m in fetch_models()}
    menu = [(mid, desc) for mid, desc in MODEL_MENU if not ids or mid in ids]
    return menu or MODEL_MENU


def _menu_block() -> str:
    lines = "\n".join(f"  - {mid}  — {desc}" for mid, desc in available_menu())
    return ("\nModels you can switch to or delegate to (use switch_model / the "
            "sub-agent `model` arg):\n" + lines + "\n")


# Signals that a prompt is hard/deep (→ thinking model) or trivial (→ cheap model).
_HARD_RE = re.compile(
    r"\b(debug|why\b|root[- ]?cause|race condition|deadlock|concurren\w*|segfault|"
    r"memory leak|optimi[sz]e|performance|architect\w*|redesign|design a|refactor|"
    r"algorithm|prove|failing test|flaky|stuck|trace through|investigate|reason about|"
    r"vulnerab\w*|exploit|complex|tricky|subtle)\b", re.I)
_SIMPLE_RE = re.compile(
    r"\b(rename|typo|reformat|format the|add a comment|docstring|bump|version|"
    r"trivial|one[- ]?liner|change the (?:text|label|string|colou?r|title)|"
    r"update the readme|fix the import|add an import|add a log)\b", re.I)


def _role_models() -> tuple[str, str, str]:
    """(default, thinking, cheap) model ids resolved from the live menu."""
    menu = available_menu()
    default = menu[0][0]
    thinking = next((m for m, d in menu if "reason" in d or "think" in d), default)
    cheap = next((m for m, d in menu if "cheap" in d or "fast" in d), default)
    return default, thinking, cheap


def route_model(prompt: str) -> tuple[str, str]:
    """Heuristic fallback router: (model_id, one-line reason)."""
    default, thinking, cheap = _role_models()
    hard = len(_HARD_RE.findall(prompt))
    simple = len(_SIMPLE_RE.findall(prompt))
    if hard or len(prompt) > 800:
        why = f"deep/hard work ({hard} signal{'s' if hard != 1 else ''})" if hard \
              else "long, detailed prompt"
        return thinking, why
    if simple:
        return cheap, "simple mechanical edit"
    return default, "general coding task"


ROUTER_SYSTEM = (
    "You are kode's model-router. Given a coding task and a list of available "
    "models, choose the single best one, balancing capability against cost. "
    "Prefer the cheapest model that will clearly succeed; escalate to a "
    "reasoning/'thinking' model only for genuinely hard debugging, tricky "
    "concurrency, or non-trivial design. Reply with ONLY a JSON object: "
    '{"model": "<exact id from the list>", "reason": "<=10 words"}'
)


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates code fences)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


_FOLLOWUP_RE = re.compile(
    r"^(ok|okay|yes|yep|yeah|sure|go|go ahead|do it|try( it| that)?|continue|"
    r"proceed|next|and |also |now |then |fix it|great|thanks|thank you|nice|"
    r"perfect|good|keep going|more|again|please do)\b", re.I)


def looks_like_followup(prompt: str, first_turn: bool) -> bool:
    """A follow-up steers the current task and should NOT trigger re-routing
    (routing a bare 'ok continue' with no context misjudges it)."""
    if first_turn:
        return False
    p = prompt.strip()
    return len(p) < 24 or bool(_FOLLOWUP_RE.match(p))


def default_router_model() -> str:
    """The cheap/fast model the router itself runs on (env-overridable)."""
    env = os.environ.get("KODE_ROUTER_MODEL")
    if env:
        return env
    return _role_models()[2]  # the "cheap & fast" menu entry


def default_temperature(model: str) -> float:
    """Moonshot/Kimi and most reasoning models prefer a higher temperature."""
    m = model.lower()
    if "kimi" in m or "thinking" in m or "reason" in m or "deepseek-r" in m:
        return 0.6
    return 0.3


def model_context_limit(model: str) -> int:
    """The model's real context window from the catalog (env pin wins)."""
    if _CTX_ENV:
        return int(_CTX_ENV)
    for m in fetch_models():
        if m["id"] == model and m.get("context_length"):
            return int(m["context_length"])
    return CONTEXT_LIMIT


# Models that need explicit cache_control breakpoints on OpenRouter. Moonshot/
# Kimi, DeepSeek, etc. cache automatically, so they're deliberately excluded.
_CACHE_PREFIXES = ("anthropic/", "google/")


def _mark_cache(msg: dict) -> dict:
    """Copy `msg` with a cache_control breakpoint on its text content."""
    content = msg.get("content")
    if isinstance(content, str) and content:
        out = dict(msg)
        out["content"] = [{"type": "text", "text": content,
                           "cache_control": {"type": "ephemeral"}}]
        return out
    return msg  # tool_calls / empty content can't be cheaply marked; skip


def apply_cache_control(messages: list, model: str) -> list:
    """Add prompt-cache breakpoints for models that bill by them.

    Two breakpoints: the static system prefix and the growing conversation tail.
    A no-op (returns the original list) for auto-caching models."""
    if not model.startswith(_CACHE_PREFIXES):
        return messages
    out = list(messages)
    sys_idx = max((i for i, m in enumerate(out) if m.get("role") == "system"),
                  default=-1)
    if sys_idx >= 0:
        out[sys_idx] = _mark_cache(out[sys_idx])
    if out and len(out) - 1 != sys_idx:
        out[-1] = _mark_cache(out[-1])
    return out


def build_system_prompt(workspace: Path) -> str:
    """Base prompt + live environment info + any project instruction docs."""
    env = (
        f"\nEnvironment:\n"
        f"- Workspace: {workspace}\n"
        f"- OS: {platform.system()} {platform.release()} ({platform.machine()})\n"
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"- Shell: {os.environ.get('SHELL', 'unknown')}\n"
    )
    docs = ""
    for name in PROJECT_DOCS:
        p = workspace / name
        if p.exists():
            body = "\n".join(p.read_text(errors="replace").splitlines()[:200])
            docs += f"\n--- Project instructions ({name}) ---\n{body}\n"
    return BASE_PROMPT + env + _menu_block() + docs


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {}


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def load_api_key() -> str | None:
    """Key from env first, then the saved ~/.kode/key file."""
    env = os.environ.get("OPENROUTER_API_KEY")
    if env:
        return env.strip()
    if KEY_FILE.exists():
        return KEY_FILE.read_text().strip() or None
    return None


def save_api_key(key: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_text(key.strip())
    try:
        os.chmod(KEY_FILE, 0o600)
    except OSError:
        pass


def validate_api_key(key: str) -> tuple[bool, str]:
    """Cheap auth check against OpenRouter's key-info endpoint (no token cost)."""
    try:
        r = requests.get("https://openrouter.ai/api/v1/key",
                         headers={"Authorization": f"Bearer {key}"}, timeout=15)
    except requests.RequestException as e:
        return False, str(e)
    if r.status_code == 200:
        d = r.json().get("data", {})
        limit = d.get("limit")
        usage = d.get("usage")
        info = f"usage ${usage:.4f}" if usage is not None else "ok"
        if limit is not None:
            info += f" / limit ${limit}"
        return True, info
    return False, f"HTTP {r.status_code}"


def load_project_config(workspace: Path) -> dict:
    """Per-repo overrides from .kode.toml or .kode.json in the workspace."""
    toml = workspace / ".kode.toml"
    js = workspace / ".kode.json"
    try:
        if toml.exists():
            import tomllib
            return tomllib.loads(toml.read_text())
        if js.exists():
            return json.loads(js.read_text())
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]ignoring bad project config: {e}[/yellow]")
    return {}


def save_project_config(workspace: Path, updates: dict) -> None:
    js = workspace / ".kode.json"
    cur = {}
    if js.exists():
        try:
            cur = json.loads(js.read_text())
        except Exception:  # noqa: BLE001
            pass
    cur.update(updates)
    js.write_text(json.dumps(cur, indent=2))


def project_context(workspace: Path) -> str | None:
    """Cached project scan; regenerated only when key files change."""
    sig_parts = []
    for name in ("README.md", "package.json", "pyproject.toml", "Cargo.toml",
                 "go.mod", "CLAUDE.md", "KODE.md"):
        p = workspace / name
        if p.exists():
            sig_parts.append(f"{name}:{int(p.stat().st_mtime)}")
    if not sig_parts:
        return None
    sig = "|".join(sig_parts)
    cache_dir = CONFIG_DIR / "initcache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(workspace)).strip("-")[-60:]
    cache = cache_dir / f"{slug}.json"
    if cache.exists():
        try:
            d = json.loads(cache.read_text())
            if d.get("sig") == sig:
                return d["text"]
        except Exception:  # noqa: BLE001
            pass
    lines = ["Project context (auto-scanned):", "", tools.list_dir(".")]
    for name in ("README.md", "package.json", "pyproject.toml", "CLAUDE.md", "KODE.md"):
        p = workspace / name
        if p.exists():
            body = "\n".join(p.read_text(errors="replace").splitlines()[:40])
            lines.append(f"\n--- {name} ---\n{body}")
    text = "\n".join(lines)
    cache.write_text(json.dumps({"sig": sig, "text": text}))
    return text


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class Agent:
    def __init__(self, api_key: str, model: str, yolo: bool = False,
                 system: str | None = None, budget: float | None = None,
                 auto_route: bool = False):
        self.api_key = api_key
        self.model = model
        self.yolo = yolo
        self.budget = budget
        self.auto_route = auto_route
        self.router_model = default_router_model()
        self.approved: set[str] = set()  # tools the user chose "always allow" for
        self.messages = [{"role": "system", "content": system or BASE_PROMPT}]
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.last_prompt_tokens = 0  # context size of the most recent request
        self.api_cost = 0.0          # authoritative cost reported by OpenRouter
        self.temperature = default_temperature(model)
        self._pricing: dict | None = None
        self._ctx_limit: int | None = None  # lazy per-model context window
        self.autosave_path: Path | None = None
        self.interrupted = False
        self._budget_warned = False
        self._acct_lock = threading.Lock()  # sub-agents update counters concurrently
        self.checkpointer = None            # set in main() once workspace is known
        self.session_start_ckpt: str | None = None
        self.turn_ckpts: list[dict] = []    # {hash, prompt, msg_len} per turn
        self.last_input: str | None = None  # for /retry
        self.title: str | None = None       # auto-named from first prompt
        self.bash_allow: list[str] = []      # command prefixes auto-approved

    # ---- pricing / cost --------------------------------------------------- #
    def _price(self) -> tuple[float, float]:
        if self._pricing is None:
            self._pricing = {"in": 0.0, "out": 0.0}
            for m in fetch_models():
                if m["id"] == self.model:
                    p = m.get("pricing", {})
                    self._pricing = {
                        "in": float(p.get("prompt", 0) or 0),
                        "out": float(p.get("completion", 0) or 0),
                    }
                    break
        return self._pricing["in"], self._pricing["out"]

    def cost_usd(self) -> float:
        # Prefer OpenRouter's authoritative per-request cost (accounts for cache
        # discounts and reasoning-token pricing); fall back to price × tokens.
        if self.api_cost > 0:
            return self.api_cost
        pin, pout = self._price()
        return self.prompt_tokens * pin + self.completion_tokens * pout

    # ---- persistence ------------------------------------------------------ #
    def snapshot(self) -> dict:
        return {
            "model": self.model,
            "workspace": str(tools.WORKSPACE),
            "saved_at": time.time(),
            "messages": self.messages,
            "todos": tools.TODOS,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "last_prompt_tokens": self.last_prompt_tokens,
            "api_cost": self.api_cost,
            "temperature": self.temperature,
            "title": self.title,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2))
        try:  # transcripts can contain secrets bash printed — keep them private
            os.chmod(path, 0o600)
        except OSError:
            pass

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self.messages = data["messages"]
        self.model = data.get("model", self.model)
        self.prompt_tokens = data.get("prompt_tokens", 0)
        self.completion_tokens = data.get("completion_tokens", 0)
        self.last_prompt_tokens = data.get("last_prompt_tokens", 0)
        self.api_cost = data.get("api_cost", 0.0)
        self.temperature = data.get("temperature", default_temperature(self.model))
        self.title = data.get("title")
        self._pricing = None
        self._ctx_limit = None
        tools.TODOS[:] = data.get("todos", [])
        self.repair_history()

    def autosave(self) -> None:
        if self.autosave_path is not None:
            try:
                self.save(self.autosave_path)
            except Exception:  # noqa: BLE001
                pass

    # ---- HTTP with retry/backoff ----------------------------------------- #
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/quinnrallen-hub/kode",
            "X-Title": "kode",
        }

    def _post(self, payload: dict, stream: bool):
        """POST with retry on transient failures (429 / 5xx / network)."""
        last_err = ""
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(API_URL, headers=self._headers(), json=payload,
                                     stream=stream, timeout=600)
            except requests.RequestException as e:
                last_err = str(e)
            else:
                if resp.status_code == 200:
                    return resp
                if resp.status_code in (408, 409, 429, 500, 502, 503, 504):
                    last_err = f"{resp.status_code}: {resp.text[:200]}"
                else:
                    raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text[:500]}")
            wait = min(2 ** attempt, 20) + random.uniform(0, 0.5)
            console.print(f"[dim]retry {attempt + 1}/{MAX_RETRIES} in {wait:.1f}s "
                          f"({last_err[:80]})[/dim]")
            time.sleep(wait)
        raise RuntimeError(f"OpenRouter failed after {MAX_RETRIES} retries: {last_err}")

    def _account(self, usage: dict | None) -> None:
        if not usage:
            return
        with self._acct_lock:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            if usage.get("prompt_tokens"):
                self.last_prompt_tokens = usage["prompt_tokens"]
            if usage.get("cost"):
                self.api_cost += usage["cost"]

    # ---- non-streaming completion (used by compact + sub-agents) --------- #
    def context_limit(self) -> int:
        if self._ctx_limit is None:
            self._ctx_limit = model_context_limit(self.model)
        return self._ctx_limit

    def _complete(self, messages: list, tools_spec=None, model: str | None = None) -> dict:
        m = model or self.model
        payload = {"model": m, "messages": apply_cache_control(messages, m),
                   "temperature": self.temperature}
        if tools_spec:
            payload["tools"] = tools_spec
            payload["tool_choice"] = "auto"
        resp = self._post(payload, stream=False)
        data = resp.json()
        self._account(data.get("usage"))
        return data["choices"][0]["message"]

    # ---- streaming model call -------------------------------------------- #
    def _stream(self) -> dict:
        payload = {
            "model": self.model,
            "messages": apply_cache_control(self.messages, self.model),
            "tools": tools.TOOLS_SPEC,
            "tool_choice": "auto",
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        resp = self._post(payload, stream=True)

        content, reasoning = "", ""
        tc_acc: dict[int, dict] = {}
        last_render = 0.0
        stopped = False
        broke = False
        finish = None

        # Only the tail is kept live (avoids re-rendering the whole markdown each
        # frame and flickering past the terminal height). A transient Live means
        # the plain-text stream is cleared and reprinted once as Markdown at the end.
        def view() -> Group:
            parts = []
            if reasoning:
                tail = "\n".join(reasoning.strip().splitlines()[-4:])
                parts.append(Text("… " + tail, style="dim italic"))
            if content:
                tail = "\n".join(content.splitlines()[-24:])
                parts.append(Text(tail))
            return Group(*parts) if parts else Group(Text("thinking…", style="dim"))

        def consume(live) -> None:
            nonlocal content, reasoning, finish, last_render
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", "replace")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                self._account(chunk.get("usage"))
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                if choices[0].get("finish_reason"):
                    finish = choices[0]["finish_reason"]
                delta = choices[0].get("delta") or {}
                if delta.get("reasoning"):
                    reasoning += delta["reasoning"]
                if delta.get("content"):
                    content += delta["content"]
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tc_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
                if live is not None and time.time() - last_render > 0.08:
                    live.update(view())
                    last_render = time.time()

        try:
            # Live streaming view only when attached to a terminal; piped/one-shot
            # output just accumulates and prints once (no double-render).
            if console.is_terminal:
                with Live(console=console, refresh_per_second=12, transient=True) as live:
                    consume(live)
            else:
                consume(None)
        except KeyboardInterrupt:
            stopped = True
            self.interrupted = True
        except requests.RequestException as e:  # mid-stream drop — salvage
            broke = True
            console.print(f"[yellow]stream interrupted ({e}); "
                          f"keeping partial reply[/yellow]")
        finally:
            resp.close()

        # Print the final answer once, as Markdown (Live was transient).
        if reasoning.strip():
            console.print(Text(reasoning.strip(), style="dim italic"))
        if content.strip():
            console.print(Markdown(content))

        if finish == "length":
            console.print("[yellow]⚠ response truncated (hit max length)[/yellow]")
        if stopped:
            console.print("[yellow]⏹ stopped[/yellow]")

        # On interrupt/broken stream, drop partial tool calls to keep history valid.
        tool_calls = [] if (stopped or broke) else [
            {"id": s["id"] or f"call_{i}", "type": "function",
             "function": {"name": s["name"], "arguments": s["args"] or "{}"}}
            for i, s in sorted(tc_acc.items())
        ]
        msg = {"role": "assistant",
               "content": content or ("(stopped)" if stopped else None)}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    # ---- history hygiene -------------------------------------------------- #
    def repair_history(self) -> None:
        """Ensure every assistant tool_call is followed by a tool result.

        A Ctrl-C between the model requesting tools and them running would
        otherwise leave orphaned tool_calls that make the next request 400."""
        out: list = []
        i, msgs = 0, self.messages
        while i < len(msgs):
            m = msgs[i]
            out.append(m)
            if m.get("role") == "assistant" and m.get("tool_calls"):
                ids = [tc["id"] for tc in m["tool_calls"]]
                j, seen = i + 1, set()
                while j < len(msgs) and msgs[j].get("role") == "tool":
                    seen.add(msgs[j].get("tool_call_id"))
                    out.append(msgs[j])
                    j += 1
                for tid in ids:
                    if tid not in seen:
                        out.append({"role": "tool", "tool_call_id": tid,
                                    "content": "(interrupted before execution)"})
                i = j
                continue
            i += 1
        self.messages = out

    # ---- lightweight context pruning ------------------------------------- #
    _READ_TOOLS = {"read_file", "grep", "glob_files", "list_dir", "fetch_url"}

    def prune_context(self, keep_recent: int = 8) -> None:
        """Stub out stale read/grep/list outputs to slow context growth.

        A later read of the same target supersedes earlier ones (dedupe); large
        read-type results beyond the recent window are elided. Both are safely
        re-fetchable, so this cheap pass makes full compaction rare."""
        meta: dict[str, tuple[str, str]] = {}
        for m in self.messages:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    try:
                        a = json.loads(tc["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        a = {}
                    key = a.get("path") or a.get("pattern") or a.get("url") or ""
                    meta[tc["id"]] = (tc["function"]["name"], key)

        tool_idxs = [i for i, m in enumerate(self.messages) if m.get("role") == "tool"]
        recent = set(tool_idxs[-keep_recent:])
        last_for: dict[tuple[str, str], int] = {}
        for i in tool_idxs:
            name, key = meta.get(self.messages[i].get("tool_call_id"), ("", ""))
            if name in self._READ_TOOLS and key:
                last_for[(name, key)] = i

        for i in tool_idxs:
            if i in recent:
                continue
            m = self.messages[i]
            body = m.get("content") or ""
            if body.startswith("[elided:") or body.startswith("ERROR"):
                continue
            name, key = meta.get(m.get("tool_call_id"), ("", ""))
            if name not in self._READ_TOOLS:
                continue
            superseded = bool(key) and last_for.get((name, key), i) != i
            if superseded or len(body) > 800:
                why = "superseded by a later call" if superseded else "old output"
                m["content"] = f"[elided: {name} {key} — {why}; re-run if needed]"

    # ---- context compaction ---------------------------------------------- #
    def maybe_compact(self) -> None:
        threshold = int(self.context_limit() * 0.8)  # auto-compact near the limit
        if self.last_prompt_tokens and self.last_prompt_tokens > threshold:
            console.print(f"[dim]context {self.last_prompt_tokens} tok > "
                          f"{threshold} (80% of {self.context_limit()}); compacting…[/dim]")
            self.compact()

    def compact(self) -> None:
        """Summarize older turns into one system note, keep the recent tail."""
        self.repair_history()
        if len(self.messages) <= 4:
            return
        # Keep the system message and the last user turn onward intact.
        cut = len(self.messages) - 1
        while cut > 1 and self.messages[cut]["role"] != "user":
            cut -= 1
        head, tail = self.messages[1:cut], self.messages[cut:]
        if not head:
            return
        ask = head + [{"role": "user", "content":
                       "Summarize everything above as concise handoff notes: the "
                       "task, decisions made, files created/changed, and what's "
                       "left to do. Be specific; this replaces the transcript."}]
        try:
            m = self._complete(ask)
            summary = (m.get("content") or m.get("reasoning") or "").strip() or "(no summary)"
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]compact failed: {e}[/yellow]")
            return
        self.messages = (
            [self.messages[0],
             {"role": "system", "content": "Summary of earlier conversation:\n" + summary}]
            + tail
        )
        console.print("[dim]compacted.[/dim]")

    # ---- one user turn ---------------------------------------------------- #
    def run_turn(self, user_input: str) -> None:
        self.interrupted = False
        first_turn = not any(m["role"] == "user" for m in self.messages)
        if self.auto_route and not looks_like_followup(user_input, first_turn):
            with console.status("[dim]router choosing a model…[/dim]", spinner="dots"):
                picked, why = self.route_via_agent(user_input)
            if picked != self.model:
                console.print(f"[magenta]⇄ router → {picked}[/magenta]  [dim]{why}[/dim]")
                self.model = picked
                self._pricing = None
                self._ctx_limit = None
                self.temperature = default_temperature(picked)
            else:
                console.print(f"[dim]router kept {picked} — {why}[/dim]")
        self.repair_history()
        self.prune_context()
        self.maybe_compact()
        # Checkpoint the workspace state entering this turn (for /rewind, /revert).
        if self.checkpointer:
            h = (self.checkpointer.snapshot(f"before: {user_input[:60]}")
                 or self.checkpointer._head())
            self.turn_ckpts.append({"hash": h, "prompt": user_input,
                                    "msg_len": len(self.messages)})
        self.last_input = user_input
        if self.title is None and user_input.strip():
            self.title = user_input.strip().replace("\n", " ")[:50]
        self.messages.append({"role": "user", "content": user_input})
        steps = 0
        while True:
            if steps >= MAX_STEPS:
                try:
                    more = console.input(
                        f"[yellow]hit {MAX_STEPS} tool steps — keep going? "
                        f"[y/N] › [/yellow]").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    more = "n"
                if more not in ("y", "yes"):
                    console.print("[yellow]stopped at step limit.[/yellow]")
                    return
                steps = 0
            steps += 1
            msg = self._stream()
            self.messages.append(msg)
            self._check_budget()
            if self.interrupted:
                return
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return
            for tc in tool_calls:
                self._run_tool(tc)
                if self.interrupted:
                    self.repair_history()
                    return

    # ---- sub-agents ------------------------------------------------------- #
    def spawn(self, task: str, model: str | None = None) -> str:
        """Run one read-only sub-agent to completion and return its answer."""
        sub = [
            {"role": "system", "content":
             "You are a focused read-only research sub-agent. Investigate using "
             "read_file/grep/glob_files/list_dir/fetch_url and return a concise, "
             "concrete answer. You cannot modify files."},
            {"role": "user", "content": task},
        ]
        spec = [t for t in tools.TOOLS_SPEC
                if t["function"]["name"] in READONLY_TOOLS]
        nudged = False
        for _ in range(SUBAGENT_STEPS):
            msg = self._complete(sub, spec, model=model)
            sub.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                answer = (msg.get("content") or msg.get("reasoning") or "").strip()
                if answer:
                    return answer
                if not nudged:  # reasoning model returned an empty turn — prod it once
                    nudged = True
                    sub.append({"role": "user",
                                "content": "Give your final answer now, concisely."})
                    continue
                return "(sub-agent returned nothing)"
            for tc in calls:
                name = tc["function"]["name"]
                try:
                    a = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    a = {}
                fn = tools.TOOL_FUNCS.get(name)
                res = fn(**a) if fn and name in READONLY_TOOLS else f"ERROR: {name} not allowed"
                sub.append({"role": "tool", "tool_call_id": tc["id"], "content": res})
        return "(sub-agent hit step limit)"

    def spawn_many(self, tasks: list) -> str:
        """Run several read-only sub-agents in parallel; return all answers."""
        tasks = tasks[:MAX_PARALLEL_AGENTS]
        if not tasks:
            return "ERROR: no tasks given"
        results: list[str | None] = [None] * len(tasks)

        def work(i, spec):
            results[i] = self.spawn(spec.get("task", ""), model=spec.get("model"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            futs = [ex.submit(work, i, t) for i, t in enumerate(tasks)]
            concurrent.futures.wait(futs)
        return "\n\n".join(
            f"### sub-agent {i + 1}"
            + (f" [{t.get('model')}]" if t.get("model") else "")
            + f": {t.get('task', '')[:80]}\n{results[i]}"
            for i, t in enumerate(tasks)
        )

    # ---- dedicated model-router agent ------------------------------------ #
    def route_via_agent(self, prompt: str) -> tuple[str, str]:
        """Ask a small dedicated router agent which model should handle `prompt`.

        Falls back to the keyword heuristic if the router errors or returns junk."""
        menu = available_menu()
        listing = "\n".join(f"- {mid}: {desc}" for mid, desc in menu)
        msgs = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content":
             f"Available models:\n{listing}\n\nCurrently running: {self.model} "
             f"(keep it unless a change is clearly worth it).\n\nTask:\n"
             f"{prompt[:1500]}\n\nRespond with JSON only."},
        ]
        try:
            m = self._complete(msgs, model=self.router_model)
            data = _extract_json((m.get("content") or m.get("reasoning") or ""))
            picked = data.get("model", "")
            ids = {x["id"] for x in fetch_models()}
            valid = {mid for mid, _ in menu}
            if picked in valid or (picked in ids):
                reason = str(data.get("reason", "")).strip()[:60] or "router pick"
                return picked, f"router: {reason}"
        except Exception:  # noqa: BLE001
            pass
        model, why = route_model(prompt)  # heuristic fallback
        return model, f"heuristic: {why}"

    def switch_model(self, model: str, reason: str) -> str:
        """Let the agent change its own model. Validates against the catalog."""
        ids = {m["id"] for m in fetch_models()}
        if ids and model not in ids:
            near = [i for i in ids if model.split("/")[-1].lower() in i.lower()][:5]
            hint = ("did you mean: " + ", ".join(near)) if near else \
                   ("options: " + ", ".join(mid for mid, _ in available_menu()))
            return f"ERROR: '{model}' is not a valid model id. {hint}"
        old = self.model
        self.model = model
        self._pricing = None
        self._ctx_limit = None
        self.temperature = default_temperature(model)
        console.print(f"[magenta]⇄ model {old} → {model}[/magenta]  [dim]{reason}[/dim]")
        return (f"Switched from {old} to {model} for the rest of the task "
                f"(temperature now {self.temperature}). Reason: {reason}")

    def _bash_allowed(self, command: str) -> bool:
        cmd = command.strip()
        return any(cmd == p or cmd.startswith(p + " ") or cmd.startswith(p)
                   for p in self.bash_allow if p)

    # ---- confirmation + diff --------------------------------------------- #
    def _confirm(self, tool: str, preview) -> tuple[bool, str]:
        if self.yolo or tool in self.approved:
            return True, ""
        console.print(preview)
        ans = console.input(
            "[bold]approve?[/bold] [green]y[/green]/[cyan]a[/cyan]lways/"
            "[red]n[/red]  (or type feedback) › "
        ).strip()
        low = ans.lower()
        if low in ("y", "yes", ""):
            return True, ""
        if low in ("a", "always"):
            self.approved.add(tool)
            return True, ""
        if low in ("n", "no"):
            return False, ""
        return False, ans  # treated as feedback to the model

    def _run_tool(self, tc: dict) -> None:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}

        _render_tool_call(name, args)

        auto_ok = (name == "bash" and self._bash_allowed(args.get("command", "")))
        if name in tools.MUTATING and not auto_ok:
            ok, feedback = self._confirm(name, _preview_for(name, args))
            if not ok:
                result = ("User declined this action. Do not attempt it again "
                          "unless the user explicitly asks."
                          + (f" User feedback: {feedback}" if feedback else ""))
                console.print(f"[yellow]↳ skipped[/yellow]")
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": result})
                return

        # Tools handled by the agent itself (need self), not the tools module.
        if name in ("spawn_agent", "spawn_agents", "switch_model"):
            try:
                if name == "spawn_agent":
                    with console.status("[dim]sub-agent working…[/dim]", spinner="dots"):
                        result = self.spawn(args.get("task", ""), model=args.get("model"))
                elif name == "spawn_agents":
                    tsk = args.get("tasks", [])
                    with console.status(f"[dim]{len(tsk)} sub-agents working in "
                                        f"parallel…[/dim]", spinner="dots"):
                        result = self.spawn_many(tsk)
                else:  # switch_model
                    result = self.switch_model(args.get("model", ""),
                                               args.get("reason", ""))
            except Exception as e:  # noqa: BLE001
                result = f"ERROR: {type(e).__name__}: {e}"
        else:
            func = tools.TOOL_FUNCS.get(name)
            if func is None:
                result = f"ERROR: unknown tool {name}"
            else:
                try:
                    result = func(**args)
                except Exception as e:  # noqa: BLE001
                    result = f"ERROR: {type(e).__name__}: {e}"

        if name == "todo_write":
            _render_todos()
        else:
            _render_tool_result(name, result)
        self.messages.append(
            {"role": "tool", "tool_call_id": tc["id"], "content": result})
        self._check_budget()

    def _check_budget(self) -> None:
        if self.budget and not self._budget_warned and self.cost_usd() > self.budget:
            self._budget_warned = True
            console.print(f"[red]⚠ budget ${self.budget:.2f} exceeded "
                          f"(${self.cost_usd():.4f} spent)[/red]")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _render_tool_call(name: str, args: dict) -> None:
    if name == "bash":
        detail = ("[bg] " if args.get("background") else "") + args.get("command", "")
    elif name in ("read_file", "write_file", "edit_file", "multi_edit",
                  "list_dir", "glob_files"):
        detail = args.get("path", args.get("pattern", ""))
        if name == "multi_edit":
            detail += f"  ({len(args.get('edits', []))} edits)"
    elif name == "grep":
        detail = f"{args.get('pattern','')}  {args.get('glob','')}".strip()
    elif name == "fetch_url":
        detail = args.get("url", "")
    elif name == "web_search":
        detail = args.get("query", "")
    elif name == "spawn_agent":
        detail = args.get("task", "")[:80] + (
            f"  [{args['model']}]" if args.get("model") else "")
    elif name == "spawn_agents":
        detail = f"{len(args.get('tasks', []))} parallel: " + " | ".join(
            t.get("task", "")[:30] for t in args.get("tasks", [])[:4])
    elif name == "switch_model":
        detail = f"→ {args.get('model','')}  ({args.get('reason','')[:50]})"
    elif name == "todo_write":
        detail = f"{len(args.get('todos', []))} items"
    else:
        detail = ""
    console.print(Text.assemble(("\n● ", "cyan"), (name, "bold"),
                                ("  " + detail, "cyan dim")))


# Tools whose output is worth previewing a few lines of; others get a 1-line summary.
_CONTENT_TOOLS = {"read_file", "grep", "glob_files", "list_dir", "bash",
                  "fetch_url", "web_search", "spawn_agent", "spawn_agents"}


def _render_tool_result(name: str, result: str) -> None:
    if result.startswith("ERROR"):
        msg = re.sub(r"^ERROR:?\s*", "", result.splitlines()[0])
        console.print(f"  [red]⎿ ✗ {msg[:160]}[/red]")
        return
    lines = result.splitlines() or ["(no output)"]
    if name in _CONTENT_TOOLS:
        show = lines[:6]
        console.print(f"  [dim]⎿ {show[0][:150]}[/dim]")
        for ln in show[1:]:
            console.print(f"  [dim]  {ln[:150]}[/dim]")
        if len(lines) > 6:
            console.print(f"  [dim]  … +{len(lines) - 6} more lines[/dim]")
    else:
        console.print(f"  [green dim]⎿ {lines[0][:150]}[/green dim]")


def _render_todos() -> None:
    mark = {"done": "[green]✓[/green]", "in_progress": "[yellow]▸[/yellow]",
            "pending": "[dim]○[/dim]"}
    console.print("  [bold]plan[/bold]")
    for t in tools.TODOS:
        style = "dim" if t["status"] == "done" else ""
        txt = f"[{style}]{t['content']}[/{style}]" if style else t["content"]
        console.print(f"    {mark.get(t['status'], '○')} {txt}")


def _diff_text(old: str, new: str, path: str) -> Text:
    out = Text()
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(),
                                fromfile=path, tofile=path, lineterm="")
    for ln in diff:
        if ln.startswith("+") and not ln.startswith("+++"):
            out.append(ln + "\n", style="green")
        elif ln.startswith("-") and not ln.startswith("---"):
            out.append(ln + "\n", style="red")
        elif ln.startswith("@@"):
            out.append(ln + "\n", style="cyan")
        else:
            out.append(ln + "\n", style="dim")
    return out or Text("(no textual change)", style="dim")


def _preview_for(name: str, args: dict):
    try:
        if name == "bash":
            return Panel(args.get("command", ""), title="will run",
                         border_style="yellow", padding=(0, 1), title_align="left")
        if name == "write_file":
            p = tools._safe_path(args.get("path", ""))
            old = p.read_text() if p.exists() else ""
            body = _diff_text(old, args.get("content", ""), args.get("path", ""))
            return Panel(body, title=f"write {args.get('path','')}",
                         border_style="yellow", padding=(0, 1), title_align="left")
        if name == "edit_file":
            p = tools._safe_path(args.get("path", ""))
            old = p.read_text() if p.exists() else ""
            new = old.replace(args.get("old", ""), args.get("new", ""))
            body = _diff_text(old, new, args.get("path", ""))
            return Panel(body, title=f"edit {args.get('path','')}",
                         border_style="yellow", padding=(0, 1), title_align="left")
        if name == "multi_edit":
            p = tools._safe_path(args.get("path", ""))
            new = old = p.read_text() if p.exists() else ""
            for e in args.get("edits", []):
                new = new.replace(e.get("old", ""), e.get("new", ""))
            body = _diff_text(old, new, args.get("path", ""))
            return Panel(body, title=f"multi_edit {args.get('path','')} "
                         f"({len(args.get('edits', []))} edits)",
                         border_style="yellow", padding=(0, 1), title_align="left")
    except Exception as e:  # noqa: BLE001
        return Text(f"(could not build preview: {e})", style="dim")
    return Text(str(args), style="dim")


# --------------------------------------------------------------------------- #
# Slash commands
# --------------------------------------------------------------------------- #
def cmd_init(agent: Agent) -> None:
    """Scan the workspace and inject a project-context message."""
    lines = ["Project context (auto-scanned):", ""]
    lines.append("Top-level:")
    lines.append(tools.list_dir("."))
    for cand in ("README.md", "package.json", "pyproject.toml", "Cargo.toml",
                 "go.mod", "CLAUDE.md"):
        p = tools.WORKSPACE / cand
        if p.exists():
            snippet = "\n".join(p.read_text(errors="replace").splitlines()[:40])
            lines.append(f"\n--- {cand} ---\n{snippet}")
    git = tools.bash("git rev-parse --is-inside-work-tree 2>/dev/null && "
                     "git status --short && git log --oneline -5")
    if git and not git.startswith("("):
        lines.append(f"\n--- git ---\n{git}")
    agent.messages.append({"role": "system", "content": "\n".join(lines)})
    console.print("[dim]project context loaded into the conversation[/dim]")


_MODELS_CACHE: list[dict] | None = None


def fetch_models(force: bool = False) -> list[dict]:
    """All OpenRouter models (cached for the session)."""
    global _MODELS_CACHE
    if _MODELS_CACHE is None or force:
        try:
            _MODELS_CACHE = requests.get(MODELS_URL, timeout=20).json()["data"]
        except Exception:  # noqa: BLE001
            return _MODELS_CACHE or []
    return _MODELS_CACHE


def _per_million(model: dict, key: str) -> str:
    try:
        v = float(model.get("pricing", {}).get(key, 0) or 0) * 1_000_000
    except (TypeError, ValueError):
        return "?"
    return "free" if v == 0 else f"${v:.2f}"


def render_models(models: list[dict], current: str) -> None:
    tbl = Table(box=None, padding=(0, 2))
    tbl.add_column("#", style="cyan", justify="right")
    tbl.add_column("id")
    tbl.add_column("ctx", justify="right", style="dim")
    tbl.add_column("$in/M", justify="right", style="dim")
    tbl.add_column("$out/M", justify="right", style="dim")
    for i, m in enumerate(models, 1):
        ctx = m.get("context_length") or 0
        mark = " [green]●[/green]" if m["id"] == current else ""
        ctx_s = f"{ctx // 1000}k" if ctx >= 1000 else str(ctx)
        tbl.add_row(str(i), m["id"] + mark, ctx_s,
                    _per_million(m, "prompt"), _per_million(m, "completion"))
    console.print(tbl)


def _set_model(agent: Agent, model_id: str) -> None:
    agent.model = model_id
    agent._pricing = None
    agent._ctx_limit = None
    agent.temperature = default_temperature(model_id)
    cfg = load_config()
    cfg["model"] = model_id
    save_config(cfg)
    console.print(f"[green]model → {model_id}[/green]  "
                  f"[dim]in {_per_million(next((m for m in fetch_models() if m['id']==model_id), {}), 'prompt')}/M "
                  f"· out {_per_million(next((m for m in fetch_models() if m['id']==model_id), {}), 'completion')}/M[/dim]")


def cmd_model(agent: Agent, session: PromptSession, arg: str) -> None:
    """Switch model: exact id, a filter term, or blank for an interactive pick."""
    arg = arg.strip()
    models = fetch_models()
    ids = {m["id"] for m in models}

    if arg and arg in ids:               # exact id → switch immediately
        _set_model(agent, arg)
        return
    if not models:                       # offline: allow blind set
        if arg:
            _set_model(agent, arg)
        else:
            console.print(f"[dim]model = {agent.model} (catalog unavailable)[/dim]")
        return

    flt = arg or session.prompt("filter models (e.g. 'kimi', 'free', "
                                "'claude'; blank = all) › ").strip()
    matches = [m for m in models if flt.lower() in m["id"].lower()] if flt else list(models)
    matches.sort(key=lambda m: m["id"])
    if not matches:
        console.print(f"[yellow]no models match '{flt}'[/yellow]")
        return
    if len(matches) > 80:
        console.print(f"[yellow]{len(matches)} matches — narrow the filter "
                      f"(showing first 80)[/yellow]")
        matches = matches[:80]
    render_models(matches, agent.model)
    pick = session.prompt("pick # or paste an id (blank = cancel) › ").strip()
    if not pick:
        console.print("[dim]cancelled[/dim]")
    elif pick.isdigit() and 1 <= int(pick) <= len(matches):
        _set_model(agent, matches[int(pick) - 1]["id"])
    elif pick in ids:
        _set_model(agent, pick)
    else:
        console.print(f"[yellow]'{pick}' is not a valid choice[/yellow]")


def sessions_meta() -> list[dict]:
    """Metadata for every saved session, newest first."""
    out = []
    for p in SESSION_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        first = next((m.get("content") for m in data.get("messages", [])
                      if m.get("role") == "user" and m.get("content")), "")
        out.append({
            "name": p.stem, "path": p, "mtime": p.stat().st_mtime,
            "msgs": len(data.get("messages", [])),
            "workspace": data.get("workspace", ""),
            "cost": data.get("api_cost", 0.0),
            "model": data.get("model", ""),
            "title": data.get("title") or (first or "").replace("\n", " ")[:50],
            "preview": (first or "").replace("\n", " ")[:60],
        })
    return sorted(out, key=lambda d: d["mtime"], reverse=True)


def latest_session(workspace: str | None = None) -> Path | None:
    metas = sessions_meta()
    if workspace:
        metas = [m for m in metas if m["workspace"] == workspace]
    return metas[0]["path"] if metas else None


def prune_sessions(keep: int = SESSION_KEEP) -> int:
    """Delete the oldest sessions beyond `keep`, never touching auto-* files."""
    named = [m for m in sessions_meta() if not m["name"].startswith("auto-")]
    victims = named[keep:]
    for m in victims:
        try:
            m["path"].unlink()
        except OSError:
            pass
    return len(victims)


def render_sessions(metas: list[dict]) -> None:
    tbl = Table(title="saved sessions", title_justify="left", box=None, padding=(0, 2))
    tbl.add_column("#", style="cyan")
    tbl.add_column("title")
    tbl.add_column("when", style="dim")
    tbl.add_column("msgs", justify="right", style="dim")
    tbl.add_column("$", justify="right", style="dim")
    for i, m in enumerate(metas, 1):
        when = time.strftime("%m-%d %H:%M", time.localtime(m["mtime"]))
        tbl.add_row(str(i), m.get("title") or m["name"], when, str(m["msgs"]),
                    f"{m.get('cost', 0):.3f}")
    console.print(tbl)


def cmd_save(agent: Agent, name: str) -> None:
    name = name.strip() or time.strftime("%Y%m%d-%H%M%S")
    path = SESSION_DIR / f"{name}.json"
    agent.save(path)
    console.print(f"[dim]saved session → {path}[/dim]")


def cmd_load(agent: Agent, name: str) -> None:
    path = SESSION_DIR / f"{name.strip()}.json"
    if not path.exists():
        avail = ", ".join(p.stem for p in SESSION_DIR.glob("*.json")) or "none"
        console.print(f"[yellow]no session '{name}'. available: {avail}[/yellow]")
        return
    agent.load(path)
    console.print(f"[dim]loaded {path.name} ({len(agent.messages)} msgs)[/dim]")


def cmd_resume(agent: Agent, session: PromptSession, arg: str) -> None:
    """Pick a session to resume: by name, by number, or interactively."""
    metas = sessions_meta()
    if not metas:
        console.print("[yellow]no saved sessions yet[/yellow]")
        return
    arg = arg.strip()
    chosen = None
    if arg.isdigit() and 1 <= int(arg) <= len(metas):
        chosen = metas[int(arg) - 1]
    elif arg:
        chosen = next((m for m in metas if m["name"] == arg), None)
        if chosen is None:
            console.print(f"[yellow]no session '{arg}'[/yellow]")
            return
    else:
        render_sessions(metas)
        pick = session.prompt("resume # (blank to cancel) › ").strip()
        if not pick.isdigit() or not (1 <= int(pick) <= len(metas)):
            console.print("[dim]cancelled[/dim]")
            return
        chosen = metas[int(pick) - 1]
    agent.load(chosen["path"])
    console.print(f"[dim]resumed {chosen['name']} — {len(agent.messages)} msgs, "
                  f"${agent.cost_usd():.4f} so far[/dim]")
    _replay_tail(agent)


def _replay_tail(agent: Agent, n: int = 6) -> None:
    """Show the last few messages so the user has context on resume."""
    tail = [m for m in agent.messages if m.get("role") in ("user", "assistant")][-n:]
    for m in tail:
        content = m.get("content")
        if not content:
            continue
        who = "[bold]›[/bold]" if m["role"] == "user" else "[cyan]⏺[/cyan]"
        snippet = content if len(content) < 400 else content[:400] + "…"
        console.print(f"{who} [dim]{snippet}[/dim]")


def cmd_diff(agent: Agent) -> None:
    if not agent.checkpointer or not agent.checkpointer.enabled:
        console.print("[yellow]checkpoints disabled (git unavailable)[/yellow]")
        return
    d = agent.checkpointer.diff(agent.session_start_ckpt)
    console.print(d if len(d) < 6000 else d[:6000] + "\n… [truncated]")


def cmd_rewind(agent: Agent, arg: str) -> None:
    """Undo the last N turns: restore files + truncate the conversation."""
    if not agent.checkpointer or not agent.checkpointer.enabled:
        console.print("[yellow]checkpoints disabled (git unavailable)[/yellow]")
        return
    n = int(arg) if arg.strip().isdigit() else 1
    if n < 1 or n > len(agent.turn_ckpts):
        console.print(f"[yellow]can rewind 1..{len(agent.turn_ckpts)} turns[/yellow]")
        return
    target = agent.turn_ckpts[-n]
    agent.checkpointer.reset(target["hash"])
    agent.messages = agent.messages[:target["msg_len"]]
    del agent.turn_ckpts[-n:]
    agent.repair_history()
    console.print(f"[magenta]⏪ rewound {n} turn(s)[/magenta] "
                  f"[dim]— files restored, conversation truncated[/dim]")


def cmd_revert(agent: Agent) -> None:
    if not agent.checkpointer or not agent.checkpointer.enabled:
        console.print("[yellow]checkpoints disabled (git unavailable)[/yellow]")
        return
    if agent.checkpointer.reset(agent.session_start_ckpt):
        console.print("[magenta]⏮ reverted all file changes to session start[/magenta] "
                      "[dim](conversation kept)[/dim]")


def cmd_jobs(agent: Agent, session: PromptSession, arg: str) -> None:
    parts = arg.split()
    if parts and parts[0] == "kill" and len(parts) > 1 and parts[1].isdigit():
        console.print(f"[dim]{tools.kill_job(int(parts[1]))}[/dim]")
        return
    jobs = tools.list_jobs()
    if not jobs:
        console.print("[dim]no background jobs[/dim]")
        return
    tbl = Table(box=None, padding=(0, 2))
    tbl.add_column("pid", style="cyan")
    tbl.add_column("status")
    tbl.add_column("command", style="dim")
    for j in jobs:
        col = "green" if j["status"] == "running" else "dim"
        tbl.add_row(str(j["pid"]), f"[{col}]{j['status']}[/{col}]", j["command"][:60])
    console.print(tbl)
    console.print("[dim]/jobs kill <pid> to stop one[/dim]")


def cmd_allow(agent: Agent, arg: str) -> None:
    """Add a bash command prefix to the per-project auto-approve allowlist."""
    prefix = arg.strip()
    if not prefix:
        console.print("[dim]allowlist: " + (", ".join(agent.bash_allow) or "empty")
                      + "[/dim]\n[dim]usage: /allow pytest[/dim]")
        return
    if prefix not in agent.bash_allow:
        agent.bash_allow.append(prefix)
        save_project_config(tools.WORKSPACE, {"bash_allow": agent.bash_allow})
    console.print(f"[dim]auto-approving bash commands starting with '{prefix}'[/dim]")


def run_onboarding() -> None:
    """First-run setup wizard: key → default model → confirmation mode."""
    console.print(Panel.fit(
        "[bold cyan]Welcome to kode[/bold cyan]\n"
        "[dim]a terminal coding agent powered by OpenRouter[/dim]\n"
        "[dim]quick setup — about 20 seconds[/dim]",
        border_style="cyan", padding=(1, 3)))
    cfg = load_config()

    # 1 — API key
    console.print("\n[bold cyan]1/3[/bold cyan]  [bold]OpenRouter API key[/bold]  "
                  "[dim]get one at https://openrouter.ai/keys[/dim]")
    if load_api_key():
        console.print("  [green]✓[/green] [dim]already configured[/dim]")
    else:
        import getpass
        while True:
            try:
                key = getpass.getpass("  paste key (hidden, Enter to skip) › ").strip()
            except (EOFError, KeyboardInterrupt):
                key = ""
            if not key:
                console.print("  [yellow]skipped — add it later with /key[/yellow]")
                break
            with console.status("  [dim]validating…[/dim]", spinner="dots"):
                ok, info = validate_api_key(key)
            if ok:
                save_api_key(key)
                console.print(f"  [green]✓[/green] [dim]saved ({info})[/dim]")
                break
            console.print(f"  [red]✗ {info}[/red] [dim]— try again[/dim]")

    # 2 — default model
    console.print("\n[bold cyan]2/3[/bold cyan]  [bold]Default model[/bold]")
    menu = available_menu()
    for i, (mid, desc) in enumerate(menu, 1):
        tag = " [green](recommended)[/green]" if i == 1 else ""
        console.print(f"  [cyan]{i}[/cyan]  {mid}  [dim]— {desc}[/dim]{tag}")
    auto_n = len(menu) + 1
    console.print(f"  [cyan]{auto_n}[/cyan]  auto-route  "
                  f"[dim]— a router agent picks per prompt[/dim]")
    pick = console.input(f"  choose [1-{auto_n}, default 1] › ").strip() or "1"
    if pick == str(auto_n):
        cfg["auto_route"] = True
        cfg["model"] = menu[0][0]
    else:
        cfg["auto_route"] = False
        idx = int(pick) - 1 if pick.isdigit() and 1 <= int(pick) <= len(menu) else 0
        cfg["model"] = menu[idx][0]

    # 3 — confirmation mode
    console.print("\n[bold cyan]3/3[/bold cyan]  [bold]How should kode apply changes?[/bold]")
    console.print("  [cyan]1[/cyan]  confirm  [dim]— review each file write / command "
                  "(recommended)[/dim]")
    console.print("  [cyan]2[/cyan]  yolo     [dim]— auto-approve everything[/dim]")
    cfg["yolo"] = console.input("  choose [1-2, default 1] › ").strip() == "2"

    save_config(cfg)
    console.print("\n[green]✓ all set.[/green]  [dim]change anything anytime: "
                  "/key · /model · /route · /auto[/dim]")


def cmd_key(agent: Agent, arg: str) -> None:
    """Set / validate the OpenRouter API key and save it to ~/.kode/key."""
    key = arg.strip()
    if not key:
        import getpass
        try:
            key = getpass.getpass("paste OpenRouter key (hidden) › ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("[dim]cancelled[/dim]"); return
    if not key:
        cur = "set" if agent.api_key else "not set"
        console.print(f"[dim]key is {cur}. usage: /key sk-or-...[/dim]")
        return
    if not key.startswith("sk-or-"):
        console.print("[yellow]that doesn't look like an OpenRouter key "
                      "(expected sk-or-…)[/yellow]")
    with console.status("[dim]validating…[/dim]", spinner="dots"):
        ok, info = validate_api_key(key)
    if not ok:
        console.print(f"[red]key rejected: {info}[/red]")
        return
    agent.api_key = key
    save_api_key(key)
    console.print(f"[green]key saved[/green] [dim]({info}) → {KEY_FILE}[/dim]")


def cmd_usage(agent: Agent) -> None:
    """Cost per day and per model across all saved sessions."""
    metas = sessions_meta()
    by_day: dict[str, float] = {}
    by_model: dict[str, float] = {}
    total = 0.0
    for m in metas:
        day = time.strftime("%Y-%m-%d", time.localtime(m["mtime"]))
        by_day[day] = by_day.get(day, 0) + m["cost"]
        by_model[m["model"]] = by_model.get(m["model"], 0) + m["cost"]
        total += m["cost"]
    tbl = Table(title="usage across sessions", title_justify="left", box=None,
                padding=(0, 2))
    tbl.add_column("by day", style="cyan")
    tbl.add_column("$", justify="right")
    for day in sorted(by_day, reverse=True)[:14]:
        tbl.add_row(day, f"{by_day[day]:.4f}")
    console.print(tbl)
    tbl2 = Table(box=None, padding=(0, 2))
    tbl2.add_column("by model", style="cyan")
    tbl2.add_column("$", justify="right")
    for mdl, c in sorted(by_model.items(), key=lambda x: -x[1])[:8]:
        tbl2.add_row(mdl or "?", f"{c:.4f}")
    console.print(tbl2)
    console.print(f"[bold]total: ${total:.4f}[/bold] across {len(metas)} sessions")


HELP = """[bold cyan]kode commands[/bold cyan]
  [green]/help[/green]              this help
  [green]/setup[/green]             re-run the first-time setup wizard
  [green]/key[/green] [sk-or-…]     set/validate your OpenRouter key (saved to ~/.kode/key)
  [green]/init[/green]              scan the project and add it to context
  [green]/model[/green] [id|filter]  switch model (blank/filter = browse catalog)
  [green]/route[/green]             toggle router agent (a cheap model picks per prompt)
  [green]/tools[/green]             list tools the model can call
  [green]/auto[/green]              toggle YOLO mode (skip all confirmations)
  [green]/undo[/green]              revert the last file write/edit
  [green]/diff[/green]              show all file changes this session
  [green]/rewind[/green] [n]        undo last n turns: restore files + conversation
  [green]/revert[/green]            discard all of this session's file changes
  [green]/retry[/green] [model]     re-run the last turn (optionally on another model)
  [green]/jobs[/green] [kill <pid>] list / kill background bash jobs
  [green]/allow[/green] <prefix>    auto-approve bash commands starting with prefix
  [green]/usage[/green]             cost per day / per model across sessions
  [green]/plan[/green]              re-show the current task plan
  [green]/compact[/green]           summarize history to reclaim context
  [green]/temp[/green] <0-2>        set sampling temperature
  [green]/cost[/green]              tokens + $ + context size
  [green]/budget[/green] <usd>      warn once session cost passes this
  [green]/export[/green] [file]     write the conversation to a markdown file
  [green]/save[/green] [name]       save this conversation
  [green]/sessions[/green] [prune]  list (or prune old) saved sessions
  [green]/resume[/green] [#|name]   resume a session (blank = pick from a list)
  [green]/load[/green] <name>       restore a saved conversation
  [green]/clear[/green]             reset the conversation
  [green]/exit[/green]              quit (or Ctrl-D)

[dim]@path[/dim]  in a message inlines that file's contents (Tab completes paths).
[dim]!cmd[/dim]   runs a shell command directly (no model turn, no tokens).
[dim]Ctrl-C[/dim] during a reply stops it cleanly. Long turns auto-compact.
Multi-line: end a line with [bold]\\[/bold] to keep typing; blank line sends."""


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
# Slash commands with a one-line description shown in the auto-fill menu.
CMD_META = {
    "/help": "show all commands",
    "/setup": "re-run the setup wizard",
    "/key": "set your OpenRouter API key",
    "/init": "scan the project into context",
    "/model": "switch model / browse catalog",
    "/models": "browse the model catalog",
    "/route": "toggle auto model routing",
    "/tools": "list callable tools",
    "/auto": "toggle YOLO (skip confirmations)",
    "/undo": "revert the last file change",
    "/diff": "show this session's file changes",
    "/rewind": "undo last n turns (files + chat)",
    "/revert": "discard all session file changes",
    "/retry": "re-run the last turn",
    "/jobs": "list / kill background jobs",
    "/allow": "auto-approve a bash prefix",
    "/usage": "cost per day / per model",
    "/plan": "show the task plan",
    "/cost": "tokens + $ this session",
    "/compact": "summarize history to save context",
    "/temp": "set sampling temperature",
    "/budget": "set a spend warning",
    "/export": "write the chat to a markdown file",
    "/save": "save this conversation",
    "/sessions": "list saved sessions",
    "/resume": "reopen a saved session",
    "/load": "restore a saved conversation",
    "/clear": "reset the conversation",
    "/exit": "quit",
}
SLASH_CMDS = list(CMD_META)


class KodeCompleter(Completer):
    """Live auto-fill menu: slash commands (with descriptions), model ids, @paths."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        stripped = text.lstrip()
        word = document.get_word_before_cursor(pattern=re.compile(r"[@/\w.:/-]+"))

        # /model <partial-id>  — only if the catalog is already cached (non-blocking)
        if stripped.startswith(("/model ", "/models ")) and _MODELS_CACHE is not None:
            frag = stripped.split(None, 1)[1] if " " in stripped else ""
            for m in _MODELS_CACHE:
                if frag.lower() in m["id"].lower():
                    price = _per_million(m, "prompt")
                    yield Completion(m["id"], start_position=-len(frag),
                                     display=m["id"], display_meta=f"{price}/M in")
            return

        if word.startswith("/") and stripped == word:
            for c in SLASH_CMDS:
                if c.startswith(word):
                    yield Completion(c, start_position=-len(word),
                                     display=c, display_meta=CMD_META.get(c, ""))
        elif word.startswith("@"):
            frag = word[1:]
            base = os.path.dirname(frag) or "."
            try:
                entries = os.listdir(tools.WORKSPACE / base)
            except OSError:
                return
            for name in sorted(entries):
                if name.startswith("."):
                    continue
                rel = os.path.join(base, name) if base != "." else name
                if rel.startswith(frag):
                    full = tools.WORKSPACE / rel
                    disp = rel + ("/" if full.is_dir() else "")
                    yield Completion("@" + disp, start_position=-len(word))


def expand_mentions(text: str) -> str:
    """Append the contents of any @path files referenced in the message.

    Requires whitespace/start before @ so emails (user@host) aren't matched."""
    blocks, seen = [], set()
    for m in re.findall(r"(?:^|\s)@([\w./\-]+)", text):
        if m in seen:
            continue
        seen.add(m)
        try:
            p = tools._safe_path(m)
        except ValueError:
            continue
        if p.is_file():
            body = "\n".join(p.read_text(errors="replace").splitlines()[:400])
            blocks.append(f"\n--- Contents of {m} ---\n{body}")
    return text + ("\n" + "\n".join(blocks) if blocks else "")


def read_message(session: PromptSession) -> str:
    first = session.prompt("\n› ")
    if not first.endswith("\\"):
        return first
    buf = [first.rstrip("\\")]
    while True:
        nxt = session.prompt("  ")
        if nxt == "":
            break
        buf.append(nxt.rstrip("\\"))
    return "\n".join(buf)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(prog="kode")
    parser.add_argument("workspace", nargs="?", default=os.getcwd())
    parser.add_argument("--model", default=cfg.get("model", DEFAULT_MODEL))
    parser.add_argument("--yolo", action="store_true", default=cfg.get("yolo", False))
    parser.add_argument("--budget", type=float, default=cfg.get("budget"),
                        help="warn once session cost exceeds this many USD")
    parser.add_argument("--route", "--auto-route", action="store_true",
                        dest="route", default=cfg.get("auto_route", False),
                        help="auto-pick a model per prompt (cheap/thinking/default)")
    parser.add_argument("-r", "--resume", "--continue", nargs="?",
                        const="__latest__", default=None, dest="resume",
                        help="resume a session (no value = most recent for this workspace)")
    parser.add_argument("--list-sessions", action="store_true",
                        help="list saved sessions and exit")
    parser.add_argument("-p", "--print", dest="oneshot", metavar="PROMPT",
                        nargs="?", const="", default=None,
                        help="one-shot: run a single turn and exit (reads stdin if no PROMPT)")
    parser.add_argument("--json", action="store_true",
                        help="one-shot: emit a JSON result instead of prose")
    parser.add_argument("--version", action="version", version=f"kode {__version__}")
    args = parser.parse_args()

    if args.list_sessions:
        metas = sessions_meta()
        render_sessions(metas) if metas else console.print("[dim]no sessions[/dim]")
        return

    api_key = load_api_key()

    # First run (no key, no config): walk through the setup wizard.
    if (api_key is None and not CONFIG_FILE.exists()
            and args.oneshot is None and sys.stdin.isatty()):
        run_onboarding()
        cfg = load_config()
        api_key = load_api_key()

    if not api_key and args.oneshot is not None:
        console.print("[red]No API key. Set OPENROUTER_API_KEY or run kode "
                      "interactively and use /key.[/red]")
        sys.exit(1)

    tools.set_workspace(args.workspace)
    os.chdir(tools.WORKSPACE)

    # Resolution order: per-project config > global config (incl. onboarding) > CLI.
    pcfg = load_project_config(tools.WORKSPACE)
    model = pcfg.get("model", cfg.get("model", args.model))
    yolo = args.yolo or cfg.get("yolo", False) or pcfg.get("yolo", False)
    budget = args.budget if args.budget is not None else pcfg.get("budget")

    agent = Agent(api_key, model, yolo=yolo, budget=budget,
                  auto_route=(args.route or cfg.get("auto_route", False)
                              or pcfg.get("auto_route", False)),
                  system=build_system_prompt(tools.WORKSPACE))
    agent.bash_allow = list(pcfg.get("bash_allow", []))
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(tools.WORKSPACE)).strip("-")[-60:]
    agent.autosave_path = SESSION_DIR / f"auto-{slug}.json"

    # Cached project context (scanned once, refreshed when key files change).
    ctx = project_context(tools.WORKSPACE)
    if ctx:
        agent.messages.append({"role": "system", "content": ctx})

    # Git-backed checkpoints for /rewind, /diff, /revert.
    agent.checkpointer = checkpoint.Checkpointer(tools.WORKSPACE, CONFIG_DIR)
    if agent.checkpointer.setup():
        agent.session_start_ckpt = agent.checkpointer._head()

    # One-shot / pipe mode: run a single turn and exit (no REPL).
    if args.oneshot is not None:
        prompt = args.oneshot or (sys.stdin.read() if not sys.stdin.isatty() else "")
        if not prompt.strip():
            console.print("[red]no prompt (pass text or pipe stdin)[/red]"); sys.exit(1)
        _run_oneshot(agent, prompt, as_json=args.json)
        return
    menu_style = Style.from_dict({
        "completion-menu.completion": "bg:#1c1c1c #b0b0b0",
        "completion-menu.completion.current": "bg:#00afaf #ffffff",
        "completion-menu.meta.completion": "bg:#1c1c1c #6c6c6c",
        "completion-menu.meta.completion.current": "bg:#008787 #ffffff",
        "scrollbar.background": "bg:#3a3a3a",
    })

    kb = KeyBindings()

    _word_re = re.compile(r"[@/\w.:/-]+")

    @kb.add("backspace")
    def _(event):
        # Deleting a char normally closes the menu; re-open it so the auto-fill
        # list keeps updating as you backspace through /commands and @paths.
        b = event.current_buffer
        b.delete_before_cursor(1)
        word = b.document.get_word_before_cursor(pattern=_word_re)
        if word.startswith(("/", "@")):
            b.start_completion(select_first=False)
        elif b.complete_state is not None:
            b.cancel_completion()

    session = PromptSession(
        history=FileHistory(os.path.expanduser("~/.kode_history")),
        completer=KodeCompleter(), complete_while_typing=True,
        key_bindings=kb, style=menu_style)

    resumed = None
    if args.resume is not None:
        path = (latest_session(str(tools.WORKSPACE)) if args.resume == "__latest__"
                else SESSION_DIR / f"{args.resume}.json")
        if path and path.exists():
            agent.load(path)
            resumed = path.stem
        else:
            console.print(f"[yellow]no session to resume "
                          f"({args.resume}); starting fresh[/yellow]")

    mode = "[red]● yolo[/red]" if agent.yolo else "[green]● confirm[/green]"
    model_label = ("auto-route" if agent.auto_route else
                   agent.model.split("/")[-1])
    ck_on = agent.checkpointer and agent.checkpointer.enabled
    ckpt = "[green]● checkpoints[/green]" if ck_on else "[dim]○ no git[/dim]"
    n_saved = len(sessions_meta())

    rows = [
        f"[bold cyan]▐ kode[/bold cyan] [dim]v{__version__}[/dim]",
        f"[dim]dir  [/dim] {tools.WORKSPACE}",
        f"[dim]model[/dim] [green]{model_label}[/green]   {mode}   {ckpt}",
    ]
    if n_saved:
        rows.append(f"[dim]{n_saved} saved session(s) · /resume to reopen[/dim]")
    rows.append("[dim]/help for commands · @file to attach · Ctrl-C stop · Ctrl-D quit[/dim]")
    console.print(Panel("\n".join(rows), border_style="cyan",
                        padding=(0, 2), title="", expand=False))

    if not agent.api_key:
        console.print("[yellow]⚠ no API key — run [bold]/key[/bold] (or [bold]/setup[/bold]) "
                      "to add your OpenRouter key[/yellow]")

    if resumed:
        console.print(f"[dim]resumed [cyan]{resumed}[/cyan] — {len(agent.messages)} "
                      f"msgs, ${agent.cost_usd():.4f} so far[/dim]")
        _replay_tail(agent)

    while True:
        try:
            user = read_message(session).strip()
        except (EOFError, KeyboardInterrupt):
            agent.repair_history()
            agent.autosave()
            console.print("\n[dim]bye (autosaved)[/dim]")
            break
        if not user:
            continue

        if user.startswith("!"):  # shell passthrough — no model turn, no tokens
            _run_shell_passthrough(user[1:].strip())
            continue

        if user.startswith("/"):
            cmd, _, rest = user[1:].partition(" ")
            if cmd in ("exit", "quit"):
                console.print("[dim]bye[/dim]"); break
            elif cmd == "help":
                console.print(HELP)
            elif cmd == "init":
                cmd_init(agent)
            elif cmd == "clear":
                agent.messages = [agent.messages[0]]
                tools.TODOS.clear()
                console.print("[dim]cleared[/dim]")
            elif cmd in ("model", "models"):
                cmd_model(agent, session, rest)
            elif cmd == "auto":
                agent.yolo = not agent.yolo
                cfg["yolo"] = agent.yolo; save_config(cfg)
                console.print(f"[dim]YOLO mode {'ON' if agent.yolo else 'OFF'}[/dim]")
            elif cmd == "route":
                agent.auto_route = not agent.auto_route
                cfg["auto_route"] = agent.auto_route; save_config(cfg)
                extra = (f" via router agent [{agent.router_model}]"
                         if agent.auto_route else "")
                console.print(f"[dim]auto-route {'ON' if agent.auto_route else 'OFF'}"
                              f"{extra}[/dim]")
            elif cmd == "undo":
                console.print(f"[dim]{tools.undo_last()}[/dim]")
            elif cmd == "diff":
                cmd_diff(agent)
            elif cmd == "rewind":
                cmd_rewind(agent, rest)
            elif cmd == "revert":
                cmd_revert(agent)
            elif cmd == "jobs":
                cmd_jobs(agent, session, rest)
            elif cmd == "allow":
                cmd_allow(agent, rest)
            elif cmd == "usage":
                cmd_usage(agent)
            elif cmd == "key":
                cmd_key(agent, rest)
            elif cmd == "setup":
                run_onboarding()
                ncfg = load_config()
                agent.api_key = load_api_key() or agent.api_key
                if ncfg.get("model"):
                    _set_model(agent, ncfg["model"])
                agent.yolo = ncfg.get("yolo", agent.yolo)
                agent.auto_route = ncfg.get("auto_route", agent.auto_route)
            elif cmd == "retry":
                if not agent.last_input:
                    console.print("[yellow]nothing to retry[/yellow]"); continue
                if rest.strip():
                    cmd_model(agent, session, rest.strip())
                if agent.turn_ckpts:  # roll back the last turn's files + messages
                    cmd_rewind(agent, "1")
                retry_input = agent.last_input
                console.print(f"[dim]retrying: {retry_input[:60]}[/dim]")
                _run_user_turn(agent, retry_input)
                continue
            elif cmd == "plan":
                _render_todos()
            elif cmd == "tools":
                tbl = Table(show_header=False, box=None, padding=(0, 2))
                for t in tools.TOOLS_SPEC:
                    f = t["function"]
                    tbl.add_row(f"[cyan]{f['name']}[/cyan]", f["description"])
                console.print(tbl)
            elif cmd == "cost":
                console.print(
                    f"[dim]{agent.prompt_tokens} in + {agent.completion_tokens} out "
                    f"= ${agent.cost_usd():.4f} · context {agent.last_prompt_tokens} tok"
                    + (f" · budget ${agent.budget:.2f}" if agent.budget else "")
                    + "[/dim]")
            elif cmd == "compact":
                agent.compact()
            elif cmd == "temp":
                if rest.strip():
                    try:
                        agent.temperature = max(0.0, min(2.0, float(rest.strip())))
                    except ValueError:
                        console.print("[yellow]usage: /temp 0.6[/yellow]"); continue
                console.print(f"[dim]temperature = {agent.temperature}[/dim]")
            elif cmd == "budget":
                if rest.strip():
                    try:
                        agent.budget = float(rest.strip()); agent._budget_warned = False
                        cfg["budget"] = agent.budget; save_config(cfg)
                    except ValueError:
                        console.print("[yellow]usage: /budget 5.00[/yellow]"); continue
                console.print(f"[dim]budget = "
                              f"{('$%.2f' % agent.budget) if agent.budget else 'none'}[/dim]")
            elif cmd == "export":
                cmd_export(agent, rest)
            elif cmd == "save":
                cmd_save(agent, rest)
            elif cmd == "load":
                cmd_load(agent, rest)
            elif cmd == "resume":
                cmd_resume(agent, session, rest)
            elif cmd == "sessions":
                if rest.strip().startswith("prune"):
                    parts = rest.split()
                    keep = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else SESSION_KEEP
                    console.print(f"[dim]pruned {prune_sessions(keep)} old sessions[/dim]")
                else:
                    metas = sessions_meta()
                    render_sessions(metas) if metas else console.print("[dim]no sessions[/dim]")
            else:
                console.print(f"[yellow]unknown /{cmd} — try /help[/yellow]")
            continue

        # Run the turn. In YOLO + TTY mode, capture anything typed while it runs
        # and replay it as follow-up turns (confirm mode can't — it reads stdin).
        queued: list[str] = []
        stop = threading.Event()
        reader = _start_input_capture(agent, queued, stop)
        _run_user_turn(agent, user)
        if reader:
            stop.set(); reader.join(timeout=0.5)
        for q in queued:
            if q.startswith("/"):
                console.print(f"[dim]skipped queued command {q} — run it after[/dim]")
                continue
            console.print(f"[dim]↵ queued: {q[:60]}[/dim]")
            _run_user_turn(agent, q)


def _start_input_capture(agent: Agent, queued: list, stop: threading.Event):
    """Background reader that collects lines typed during a YOLO turn."""
    if not (agent.yolo and sys.stdin.isatty()):
        return None
    import select

    def _rd():
        while not stop.is_set():
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
            except Exception:  # noqa: BLE001
                return
            if r:
                line = sys.stdin.readline()
                if line.strip():
                    queued.append(line.strip())

    t = threading.Thread(target=_rd, daemon=True)
    t.start()
    return t


def _run_oneshot(agent: Agent, prompt: str, as_json: bool = False) -> None:
    """Run a single turn and exit — for `kode -p` and piped input."""
    global console
    if as_json:  # keep stdout clean for the JSON; stream UI goes to stderr
        console = Console(stderr=True)
    agent.checkpointer and agent.checkpointer.snapshot("oneshot start")
    start = agent.session_start_ckpt
    try:
        agent.run_turn(expand_mentions(prompt))
    except Exception as e:  # noqa: BLE001
        if as_json:
            print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        else:
            console.print(f"[red]{type(e).__name__}: {e}[/red]")
        sys.exit(1)
    agent.autosave()
    answer = next((m.get("content") for m in reversed(agent.messages)
                   if m.get("role") == "assistant" and m.get("content")), "")
    if as_json:
        changed = agent.checkpointer.changed_files(start) if agent.checkpointer else []
        print(json.dumps({
            "answer": answer, "model": agent.model,
            "cost_usd": round(agent.cost_usd(), 6),
            "files_changed": changed,
        }, indent=2))


def _run_shell_passthrough(command: str) -> None:
    """Run a shell command directly (inherits the terminal) — the `!cmd` prefix."""
    if not command:
        return
    import subprocess as sp
    try:
        rc = sp.run(command, shell=True, cwd=str(tools.WORKSPACE)).returncode
    except KeyboardInterrupt:
        console.print("[yellow]^C[/yellow]"); return
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]{type(e).__name__}: {e}[/red]"); return
    if rc:
        console.print(f"[dim]exit {rc}[/dim]")


def cmd_export(agent: Agent, arg: str) -> None:
    """Write the conversation to a markdown file (default: timestamped in workspace)."""
    name = arg.strip() or f"kode-transcript-{datetime.now():%Y%m%d-%H%M%S}.md"
    if not name.endswith(".md"):
        name += ".md"
    path = Path(name)
    if not path.is_absolute():
        path = tools.WORKSPACE / path
    out = ["# kode transcript", "",
           f"- model: `{agent.model}`",
           f"- exported: {datetime.now():%Y-%m-%d %H:%M}", ""]
    for m in agent.messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):  # tolerate cache-control part form
            content = " ".join(p.get("text", "") for p in content
                               if isinstance(p, dict))
        if role == "user":
            out.append(f"## 🧑 User\n\n{content}\n")
        elif role == "assistant":
            if content:
                out.append(f"## 🤖 Assistant\n\n{content}\n")
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                out.append(f"> 🔧 `{fn['name']}` — `{fn.get('arguments', '')[:200]}`\n")
        elif role == "tool":
            out.append(f"> ⎿ `{(content or '')[:400].splitlines()[0] if content else ''}`\n")
    try:
        path.write_text("\n".join(out))
    except OSError as e:
        console.print(f"[red]export failed: {e}[/red]"); return
    console.print(f"[green]exported {len(agent.messages)} messages → {path}[/green]")


def _run_user_turn(agent: Agent, user: str) -> None:
    """Run one turn with mention-expansion, bell on long turns, and autosave."""
    if not agent.api_key:
        console.print("[yellow]no API key set — run [bold]/key[/bold] to add your "
                      "OpenRouter key[/yellow]")
        return
    t0 = time.time()
    try:
        agent.run_turn(expand_mentions(user))
        agent.autosave()
        if time.time() - t0 > 20:      # ring the terminal bell after a long turn
            sys.stdout.write("\a"); sys.stdout.flush()
        console.print(f"[dim]— {agent.completion_tokens} out tok · "
                      f"${agent.cost_usd():.4f} · ctx {agent.last_prompt_tokens} "
                      f"· autosaved —[/dim]")
    except KeyboardInterrupt:
        agent.repair_history()
        agent.autosave()
        console.print("\n[yellow]interrupted (autosaved)[/yellow]")
    except Exception as e:  # noqa: BLE001
        agent.repair_history()
        agent.autosave()
        console.print(f"[red]{type(e).__name__}: {e}[/red]")


if __name__ == "__main__":
    main()
