"""Tool implementations for the kode agent.

Each tool is a plain Python function. TOOLS_SPEC is the OpenAI/OpenRouter
function-calling schema advertised to the model; TOOL_FUNCS maps names to
the callables. Every function returns a string (what the model sees back).
"""
from __future__ import annotations

import fnmatch
import html
import ipaddress
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

# Root the agent is allowed to touch. Defaults to CWD at import time.
WORKSPACE = Path(os.environ.get("KODE_WORKSPACE", os.getcwd())).resolve()

MAX_READ_BYTES = 400_000       # refuse to read files larger than this
MAX_OUTPUT_CHARS = 20_000
MAX_FETCH_BYTES = 5_000_000    # cap on fetch_url download size

# Undo journal: list of (path, previous_content_or_None). None means the file
# did not exist before, so undo deletes it.
_UNDO_STACK: list[tuple[Path, str | None]] = []

# mtime of each file the agent last read/wrote — used to detect external edits.
_READ_MTIMES: dict[str, float] = {}

# Latest plan written by the model via todo_write; the UI renders it.
TODOS: list[dict] = []

# Background bash jobs started this session: {pid, command, log, proc}.
JOBS: list[dict] = []

# Directories we never descend into for glob/grep/scan.
_IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
                "dist", "build", ".mypy_cache", ".pytest_cache", ".next"}

# Files whose contents must not be sent to the model. Override with
# KODE_ALLOW_SECRETS=1 (e.g. when you're intentionally debugging a .env).
_SECRET_GLOBS = (".env", ".env.*", "*.pem", "*.key", "id_rsa", "id_ed25519",
                 "id_dsa", "*.p12", "*.pfx", ".git-credentials",
                 ".netrc", "*.keystore")

# Shell commands blocked unconditionally (even in YOLO mode) — catastrophic ops.
_DANGEROUS_CMD = [
    re.compile(r"\brm\s+(-[a-z]*\s+)*-?[a-z]*f[a-z]*\s+(-[a-z]+\s+)*/(\s|$|\*)"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\b[^\n]*\bof=/dev/"),
    re.compile(r">\s*/dev/(sd|nvme|hd|mmcblk)"),
    re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),  # fork bomb
    re.compile(r"\bchmod\s+-R\s+[0-7]{3,4}\s+/(\s|$)"),
]


def set_workspace(path) -> Path:
    """Point the agent at a new workspace root (keeps the global a Path)."""
    global WORKSPACE
    WORKSPACE = Path(os.path.abspath(path)).resolve()
    _READ_MTIMES.clear()
    return WORKSPACE


def _record_undo(p: Path) -> None:
    _UNDO_STACK.append((p, p.read_text() if p.exists() else None))


def _touch(p: Path) -> None:
    """Remember a file's mtime after we read or wrote it."""
    try:
        _READ_MTIMES[str(p)] = p.stat().st_mtime
    except OSError:
        pass


def _check_stale(p: Path) -> str | None:
    """Return an error if the file changed on disk since the agent last saw it."""
    key = str(p)
    if p.exists() and key in _READ_MTIMES:
        if p.stat().st_mtime > _READ_MTIMES[key] + 1e-6:
            return (f"ERROR: {os.path.relpath(p, WORKSPACE)} changed on disk since "
                    f"you last read it. read_file it again before writing.")
    return None


def _is_secret(p: Path) -> bool:
    if os.environ.get("KODE_ALLOW_SECRETS") == "1":
        return False
    return any(fnmatch.fnmatch(p.name, g) for g in _SECRET_GLOBS)


def _safe_path(path: str) -> Path:
    """Resolve `path` and ensure it stays inside WORKSPACE."""
    p = (WORKSPACE / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    if WORKSPACE not in p.parents and p != WORKSPACE:
        raise ValueError(f"path escapes workspace {WORKSPACE}: {p}")
    return p


def _clip(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(text)} chars total]"
    return text


# --------------------------------------------------------------------------- #
# Tool functions
# --------------------------------------------------------------------------- #
def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    if p.is_dir():
        return f"ERROR: {path} is a directory (use list_dir)"
    if _is_secret(p):
        return (f"ERROR: {path} looks like a secrets file; refusing to send its "
                f"contents. Set KODE_ALLOW_SECRETS=1 to override.")
    size = p.stat().st_size
    if size > MAX_READ_BYTES:
        return (f"ERROR: {path} is {size} bytes (> {MAX_READ_BYTES}); read a slice "
                f"with offset/limit or grep it instead.")
    _touch(p)
    data = p.read_text(errors="replace").splitlines()
    chunk = data[offset : offset + limit]
    numbered = "\n".join(f"{i + offset + 1}\t{line}" for i, line in enumerate(chunk))
    tail = "" if offset + limit >= len(data) else f"\n... [{len(data)} lines total]"
    return _clip(numbered + tail) or "(empty file)"


def write_file(path: str, content: str) -> str:
    p = _safe_path(path)
    stale = _check_stale(p)
    if stale:
        return stale
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    _record_undo(p)
    p.write_text(content)
    _touch(p)
    verb = "Overwrote" if existed else "Created"
    return f"{verb} {path} ({len(content)} bytes, {content.count(chr(10)) + 1} lines)"


def _fuzzy_find(text: str, old: str):
    """Locate `old` in `text` ignoring each line's leading/trailing whitespace.

    Cheaper models often botch indentation in the `old` snippet; this recovers a
    match when it's line-aligned and unique. Returns a (start, end) char span, or
    None if there's no match or more than one."""
    olines = old.strip("\n").splitlines()
    if not olines:
        return None
    tlines = text.splitlines(keepends=True)
    norm_o = [l.strip() for l in olines]
    norm_t = [l.rstrip("\n").strip() for l in tlines]
    n = len(norm_o)
    hits = [i for i in range(len(norm_t) - n + 1) if norm_t[i:i + n] == norm_o]
    if len(hits) != 1:
        return None
    i = hits[0]
    start = sum(len(tlines[j]) for j in range(i))
    end = start + sum(len(tlines[j]) for j in range(i, i + n))
    return start, end


def _apply_edit(text: str, old: str, new: str, replace_all: bool) -> tuple[str, str]:
    """Return (new_text, note). Raises ValueError with a model-facing message."""
    count = text.count(old)
    if count == 0:
        span = _fuzzy_find(text, old)
        if span is None:
            raise ValueError("old string not found. Read the file and match exactly.")
        s, e = span
        matched = text[s:e]
        repl = new
        if matched.endswith("\n") and not repl.endswith("\n"):
            repl += "\n"  # keep the trailing newline the matched region had
        return text[:s] + repl + text[e:], " (whitespace-tolerant match)"
    if count > 1 and not replace_all:
        raise ValueError(f"old string appears {count} times; make it unique "
                         f"or set replace_all=true")
    return text.replace(old, new), ""


def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    stale = _check_stale(p)
    if stale:
        return stale
    text = p.read_text()
    try:
        result, note = _apply_edit(text, old, new, replace_all)
    except ValueError as e:
        return f"ERROR: {e}"
    _record_undo(p)
    p.write_text(result)
    _touch(p)
    return f"Edited {path}{note}"


def list_dir(path: str = ".") -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: not found: {path}"
    entries = []
    for item in sorted(p.iterdir()):
        if item.name.startswith(".") and item.name not in (".env", ".gitignore"):
            continue
        entries.append(f"{item.name}/" if item.is_dir() else item.name)
    return "\n".join(entries) or "(empty)"


def _is_dangerous(command: str) -> bool:
    return any(rx.search(command) for rx in _DANGEROUS_CMD)


def _clean_jobs(logdir: Path, keep: int = 20, max_age_days: int = 7) -> None:
    logs = sorted(logdir.glob("job-*.log"), key=lambda p: p.stat().st_mtime)
    cutoff = time.time() - max_age_days * 86400
    for old in logs[:-keep]:
        old.unlink(missing_ok=True)
    for p in logdir.glob("job-*.log"):
        if p.stat().st_mtime < cutoff:
            p.unlink(missing_ok=True)


def bash(command: str, timeout: int = 120, background: bool = False) -> str:
    if _is_dangerous(command):
        return ("ERROR: refused — this command matches a catastrophic pattern "
                "(rm -rf /, mkfs, dd to a device, fork bomb, …). Run it yourself "
                "if you really mean to.")
    if background:
        logdir = WORKSPACE / ".kode-jobs"
        logdir.mkdir(exist_ok=True)
        _clean_jobs(logdir)
        log = logdir / f"job-{int(time.time())}.log"
        fh = open(log, "w")
        p = subprocess.Popen(command, shell=True, cwd=str(WORKSPACE),
                             stdout=fh, stderr=subprocess.STDOUT)
        JOBS.append({"pid": p.pid, "command": command, "log": log, "proc": p})
        rel = os.path.relpath(log, WORKSPACE)
        return (f"started in background (pid {p.pid}); output streaming to {rel}. "
                f"read that file to check progress.")
    return _run_foreground(command, timeout)


def _run_foreground(command: str, timeout: int) -> str:
    """Run a command, echoing output live (dimmed) while capturing it.

    A watchdog thread enforces the timeout even if the command produces no
    output. Set KODE_BASH_QUIET=1 to suppress the live echo."""
    proc = subprocess.Popen(command, shell=True, cwd=str(WORKSPACE),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    timed_out = threading.Event()

    def _kill():
        timed_out.set()
        try:
            proc.kill()
        except OSError:
            pass

    timer = threading.Timer(timeout, _kill)
    timer.start()
    live = os.environ.get("KODE_BASH_QUIET") != "1" and sys.stderr.isatty()
    chunks: list[str] = []
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            chunks.append(line)
            if live:
                sys.stderr.write("\x1b[2m│ " + line.rstrip("\n")[:400] + "\x1b[0m\n")
                sys.stderr.flush()
    finally:
        proc.wait()
        timer.cancel()
    out = "".join(chunks).strip()
    if timed_out.is_set():
        return _clip(out) + f"\nERROR: command timed out after {timeout}s"
    rc = proc.returncode
    return _clip(out or f"(no output, exit {rc})") + (f"\n[exit {rc}]" if rc else "")


def list_jobs() -> list[dict]:
    """Background jobs with live running/exited status."""
    out = []
    for j in JOBS:
        rc = j["proc"].poll()
        out.append({"pid": j["pid"], "command": j["command"], "log": j["log"],
                    "status": "running" if rc is None else f"exited {rc}"})
    return out


def kill_job(pid: int) -> str:
    for j in JOBS:
        if j["pid"] == pid:
            if j["proc"].poll() is None:
                j["proc"].terminate()
                return f"killed job {pid}"
            return f"job {pid} already exited"
    return f"no job with pid {pid}"


def multi_edit(path: str, edits: list) -> str:
    """Apply several find/replace edits to one file, in order, atomically.

    Each edit is {old, new}. If any `old` is missing or ambiguous, nothing is
    written."""
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    stale = _check_stale(p)
    if stale:
        return stale
    text = original = p.read_text()
    for k, e in enumerate(edits):
        try:
            text, _ = _apply_edit(text, e.get("old", ""), e.get("new", ""),
                                  bool(e.get("replace_all")))
        except ValueError as err:
            return f"ERROR: edit {k + 1}: {err} (no changes written)"
    if text == original:
        return "no changes (edits produced identical content)"
    _record_undo(p)
    p.write_text(text)
    _touch(p)
    return f"Applied {len(edits)} edits to {path}"


def grep(pattern: str, path: str = ".", glob: str = "") -> str:
    # Prefer ripgrep when available: faster and honours .gitignore.
    if _have_rg():
        args = ["rg", "--line-number", "--no-heading", "--color=never"]
        for d in _IGNORE_DIRS:
            args += ["--glob", f"!**/{d}/**"]
        if glob:
            args += ["--glob", glob]
        args += [pattern, str(_safe_path(path))]
    else:
        args = ["grep", "-rn", "--color=never"]
        for d in _IGNORE_DIRS:
            args.append(f"--exclude-dir={d}")
        if glob:
            args += [f"--include={glob}"]
        args += [pattern, str(_safe_path(path))]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    if r.returncode == 1:
        return "(no matches)"
    if r.returncode not in (0, 1) and r.stderr:
        return f"ERROR: {r.stderr.strip()[:300]}"
    return _clip(r.stdout.strip()) or "(no matches)"


def _have_rg() -> bool:
    if not hasattr(_have_rg, "_v"):
        from shutil import which
        _have_rg._v = which("rg") is not None
    return _have_rg._v


def glob_files(pattern: str, path: str = ".") -> str:
    """Find files by name pattern (e.g. '*.py', 'test_*.py') recursively."""
    root = _safe_path(path)
    matches = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
        for fn in filenames:
            if fnmatch.fnmatch(fn, pattern):
                matches.append(os.path.relpath(os.path.join(dirpath, fn), WORKSPACE))
    matches.sort()
    return _clip("\n".join(matches)) or "(no matches)"


def _is_public_host(host: str) -> bool:
    """False if the host resolves to a private/loopback/link-local address."""
    try:
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast):
                return False
    except (socket.gaierror, ValueError):
        return False
    return True


def fetch_url(url: str, max_chars: int = 12000) -> str:
    """Fetch a public URL; HTML is crudely stripped to text."""
    if not url.startswith(("http://", "https://")):
        return "ERROR: url must start with http(s)://"
    host = urlparse(url).hostname or ""
    if os.environ.get("KODE_ALLOW_LOCAL_FETCH") != "1" and not _is_public_host(host):
        return (f"ERROR: refusing to fetch {host} (private/loopback address). "
                f"Set KODE_ALLOW_LOCAL_FETCH=1 to override.")
    try:
        r = requests.get(url, timeout=30, stream=True,
                         headers={"User-Agent": "kode/1.0"})
        chunks, total = [], 0
        for c in r.iter_content(8192, decode_unicode=False):
            chunks.append(c)
            total += len(c)
            if total > MAX_FETCH_BYTES:
                break
        r._content = b"".join(chunks)  # let .text decode what we collected
    except requests.RequestException as e:
        return f"ERROR: {e}"
    text = r.text
    if "html" in r.headers.get("content-type", ""):
        text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"
    return f"[{r.status_code}] {text}" if text else f"[{r.status_code}] (empty body)"


def web_search(query: str, max_results: int = 6) -> str:
    """Search the web via DuckDuckGo's HTML endpoint (no API key needed).

    Returns a ranked list of title / url / snippet. Use fetch_url to read a hit."""
    try:
        r = requests.post("https://html.duckduckgo.com/html/", data={"q": query},
                          timeout=20, headers={"User-Agent": "Mozilla/5.0 (kode)"})
    except requests.RequestException as e:
        return f"ERROR: {e}"
    if r.status_code != 200:
        return f"ERROR: search returned HTTP {r.status_code}"
    results = []
    for m in re.finditer(r'result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r.text, re.S):
        title = html.unescape(re.sub(r"<[^>]+>", "", m.group(2)).strip())
        um = re.search(r"uddg=([^&]+)", m.group(1))
        url = unquote(um.group(1)) if um else m.group(1)
        if title:
            results.append((title, url))
        if len(results) >= max_results:
            break
    if not results:
        return "(no results)"
    snippets = [html.unescape(re.sub(r"<[^>]+>", "", s).strip())
                for s in re.findall(r'result__snippet"[^>]*>(.*?)</a>', r.text, re.S)]
    out = []
    for i, (title, url) in enumerate(results):
        snip = snippets[i][:300] if i < len(snippets) else ""
        out.append(f"{i + 1}. {title}\n   {url}" + (f"\n   {snip}" if snip else ""))
    return "\n".join(out)


def todo_write(todos: list) -> str:
    """Record/replace the task plan. Each item: {content, status}.

    status is one of: pending | in_progress | done."""
    global TODOS
    TODOS = [
        {"content": t.get("content", ""), "status": t.get("status", "pending")}
        for t in todos
    ]
    done = sum(1 for t in TODOS if t["status"] == "done")
    return f"Plan updated: {done}/{len(TODOS)} done"


def undo_last() -> str:
    """Revert the most recent write_file/edit_file. UI-driven, not a model tool."""
    if not _UNDO_STACK:
        return "nothing to undo"
    p, prev = _UNDO_STACK.pop()
    rel = os.path.relpath(p, WORKSPACE)
    if prev is None:
        if p.exists():
            p.unlink()
        return f"removed {rel} (was newly created)"
    p.write_text(prev)
    return f"reverted {rel}"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
TOOL_FUNCS = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "multi_edit": multi_edit,
    "list_dir": list_dir,
    "glob_files": glob_files,
    "bash": bash,
    "grep": grep,
    "fetch_url": fetch_url,
    "web_search": web_search,
    "todo_write": todo_write,
}

# Tools that change disk state — the UI asks for confirmation before running.
# (spawn_agent is handled in agent.py, not here.)
MUTATING = {"write_file", "edit_file", "multi_edit", "bash"}

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file. Returns line-numbered content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "0-indexed start line"},
                    "limit": {"type": "integer", "description": "max lines to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact substring in a file. `old` must match exactly and be unique unless replace_all=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_edit",
            "description": "Apply several exact find/replace edits to a single file in one atomic operation. Fails without writing if any `old` is missing/ambiguous.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old": {"type": "string"},
                                "new": {"type": "string"},
                                "replace_all": {"type": "boolean"},
                            },
                            "required": ["old", "new"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files/dirs in a directory (hidden files skipped).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the workspace and return combined stdout/stderr. Set background=true for long-running processes (servers, builds); output goes to a log file you can read.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "background": {"type": "boolean"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Recursively search file contents for a regex pattern. Optional glob like '*.py'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_files",
            "description": "Find files by name pattern recursively (e.g. '*.py', 'test_*'). Ignores vendor/build dirs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a web page or API endpoint (docs, references). HTML is stripped to text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for a query and get ranked title/url/snippet results. Use when you need to find current docs, error messages, or references and don't already know the URL; follow up with fetch_url to read a result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": "default 6"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_agent",
            "description": "Delegate a self-contained read-only investigation to a sub-agent (e.g. 'find where X is configured and summarize'). It can read/grep/glob/fetch but not modify files, and returns a concise answer. Use it to keep your own context small. Optionally run it on a different model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "the question/task for the sub-agent"},
                    "model": {"type": "string", "description": "optional model id to run this sub-agent on (e.g. a cheaper or a thinking model)"},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_agents",
            "description": "Fan out SEVERAL independent read-only investigations at once; they run in parallel and you get all answers back together. Use when a task splits into independent questions (e.g. investigate 3 modules). Each may specify its own model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"},
                                "model": {"type": "string", "description": "optional model id for this sub-agent"},
                            },
                            "required": ["task"],
                        },
                    },
                },
                "required": ["tasks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_model",
            "description": "Switch the model YOU (the main agent) run on for the rest of the task. Use a cheaper/faster model for simple mechanical work, or a stronger 'thinking' model for hard debugging or design. State why. The switch persists until you change it again.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "the OpenRouter model id to switch to"},
                    "reason": {"type": "string", "description": "one line: why this model for what's next"},
                },
                "required": ["model", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": "Record or update your task plan for a multi-step job. Call it as you start and finish steps so the user can follow along.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "done"],
                                },
                            },
                            "required": ["content", "status"],
                        },
                    }
                },
                "required": ["todos"],
            },
        },
    },
]
