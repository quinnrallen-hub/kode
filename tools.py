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
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests

# Root the agent is allowed to touch. Defaults to CWD at import time.
WORKSPACE = Path(os.environ.get("KODE_WORKSPACE", os.getcwd())).resolve()

MAX_READ_BYTES = 400_000       # refuse a *full* read of files larger than this
MAX_READ_HARD_CAP = 50_000_000  # never load a file bigger than this, even sliced
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
    # Only refuse a *full* read of a big file; an explicit offset/limit slice is
    # allowed (and bounded again by _clip below), so the model isn't told to do
    # something the guard would keep rejecting.
    explicit_slice = offset > 0 or limit < 2000
    if size > MAX_READ_BYTES and not explicit_slice:
        return (f"ERROR: {path} is {size} bytes (> {MAX_READ_BYTES}); read a slice "
                f"with offset/limit (e.g. offset=0, limit=200) or grep it instead.")
    if size > MAX_READ_HARD_CAP:
        return (f"ERROR: {path} is {size} bytes (> {MAX_READ_HARD_CAP}); too large to "
                f"read even sliced — grep it for what you need.")
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
        log = logdir / f"job-{time.time_ns()}.log"  # ns → no same-second collision
        fh = open(log, "w")
        p = subprocess.Popen(command, shell=True, cwd=str(WORKSPACE),
                             stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        fh.close()  # the child holds its own dup; the parent doesn't need this one
        JOBS.append({"pid": p.pid, "command": command, "log": log, "proc": p})
        rel = os.path.relpath(log, WORKSPACE)
        return (f"started in background (pid {p.pid}); output streaming to {rel}. "
                f"read that file to check progress.")
    return _run_foreground(command, timeout)


def _run_foreground(command: str, timeout: int) -> str:
    """Run a command, echoing output live (dimmed) while capturing it.

    A watchdog thread enforces the timeout even if the command produces no
    output. Set KODE_BASH_QUIET=1 to suppress the live echo."""
    # start_new_session=True puts the command in its own process group so the
    # watchdog can kill the whole tree — a grandchild holding the pipe open would
    # otherwise keep us blocked in the read loop long past the timeout.
    proc = subprocess.Popen(command, shell=True, cwd=str(WORKSPACE),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)
    timed_out = threading.Event()

    def _kill():
        timed_out.set()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
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


def _strip_html(text: str) -> str:
    """Crudely turn an HTML document into readable plain text."""
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text


def _safe_get(url: str, *, timeout: int = 30, max_bytes: int = MAX_FETCH_BYTES,
              headers: dict | None = None):
    """Perform a GET on an arbitrary/user-influenced URL, safely.

    Follows redirects by hand, re-checking the host at every hop (SSRF guard),
    and caps the download at max_bytes. Returns a `requests.Response` whose body
    is already collected, or a string starting with "ERROR:" on refusal/failure.
    """
    if not url.startswith(("http://", "https://")):
        return "ERROR: url must start with http(s)://"
    allow_local = os.environ.get("KODE_ALLOW_LOCAL_FETCH") == "1"
    hdrs = {"User-Agent": "kode/1.0"}
    if headers:
        hdrs.update(headers)
    try:
        # Follow redirects by hand, re-checking the host at every hop: otherwise a
        # public URL could 302 us to http://169.254.169.254/ or a loopback service.
        r = None
        for _ in range(6):
            host = urlparse(url).hostname or ""
            if not allow_local and not _is_public_host(host):
                return (f"ERROR: refusing to fetch {host} (private/loopback address). "
                        f"Set KODE_ALLOW_LOCAL_FETCH=1 to override.")
            r = requests.get(url, timeout=timeout, stream=True, allow_redirects=False,
                             headers=hdrs)
            if r.is_redirect and r.headers.get("location"):
                nxt = urljoin(url, r.headers["location"])
                r.close()
                if not nxt.startswith(("http://", "https://")):
                    return "ERROR: redirect to a non-http(s) URL blocked"
                url = nxt
                continue
            break
        else:
            return "ERROR: too many redirects"
        chunks, total = [], 0
        for c in r.iter_content(8192, decode_unicode=False):
            chunks.append(c)
            total += len(c)
            if total > max_bytes:
                break
        r._content = b"".join(chunks)  # let .text/.content see what we collected
    except requests.RequestException as e:
        return f"ERROR: {e}"
    return r


def fetch_url(url: str, max_chars: int = 12000) -> str:
    """Fetch a public URL; HTML is crudely stripped to text."""
    r = _safe_get(url)
    if isinstance(r, str):
        return r
    text = r.text
    if "html" in r.headers.get("content-type", ""):
        text = _strip_html(text)
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


# --------------------------------------------------------------------------- #
# Fetchers — targeted retrieval from well-known APIs. Fixed-host API calls hit
# requests directly (with a timeout); anything that then chases an arbitrary URL
# (e.g. a wayback snapshot) goes through _safe_get so the SSRF guard applies.
# --------------------------------------------------------------------------- #
def _gh_headers() -> dict:
    """GitHub API headers, with a token from `gh auth token` when available."""
    h = {"User-Agent": "kode/1.0", "Accept": "application/vnd.github+json"}
    try:
        r = subprocess.run(["gh", "auth", "token"], capture_output=True,
                           text=True, timeout=5)
        tok = r.stdout.strip()
        if r.returncode == 0 and tok:
            h["Authorization"] = f"Bearer {tok}"
    except Exception:  # noqa: BLE001 — gh missing/offline: fall back to keyless
        pass
    return h


def fetch_github(repo: str, kind: str = "file", ref: str = "",
                 path: str = "", number: int = 0) -> str:
    """Fetch from api.github.com for owner/repo.

    kind: file | issues | prs | releases. For kind=file, `path` is the in-repo
    path (and `ref` an optional branch/tag/sha). For issues/prs, `number` fetches
    a single one; otherwise the most recent are listed. Tries `gh auth token`."""
    if "/" not in repo:
        return "ERROR: repo must be 'owner/name'"
    base = f"https://api.github.com/repos/{repo}"
    if kind == "file":
        if not path:
            return "ERROR: kind=file needs a path"
        url = f"{base}/contents/{path.lstrip('/')}"
        if ref:
            url += f"?ref={ref}"
    elif kind == "issues":
        url = f"{base}/issues/{number}" if number else f"{base}/issues?state=all&per_page=15"
    elif kind == "prs":
        url = f"{base}/pulls/{number}" if number else f"{base}/pulls?state=all&per_page=15"
    elif kind == "releases":
        url = f"{base}/releases?per_page=10"
    else:
        return "ERROR: kind must be file | issues | prs | releases"
    try:
        r = requests.get(url, timeout=30, headers=_gh_headers())
    except requests.RequestException as e:
        return f"ERROR: {e}"
    if r.status_code == 404:
        return f"ERROR: not found (404): {repo} {kind} {path or number}"
    if r.status_code != 200:
        return f"ERROR: GitHub returned HTTP {r.status_code}: {r.text[:200]}"
    try:
        data = r.json()
    except ValueError:
        return f"ERROR: non-JSON response from GitHub"
    if kind == "file":
        import base64
        if isinstance(data, list):
            return _clip("\n".join(f"{e['type']:5} {e['name']}" for e in data))
        if data.get("encoding") == "base64":
            try:
                body = base64.b64decode(data["content"]).decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                return "ERROR: could not decode file content"
            return _clip(body)
        return _clip(str(data.get("content", "")))
    if isinstance(data, dict):  # single issue/pr
        return _clip(f"#{data.get('number')} {data.get('title')} "
                     f"[{data.get('state')}]\n{data.get('html_url')}\n\n"
                     f"{data.get('body') or ''}")
    if kind == "releases":
        lines = [f"{d.get('tag_name')}  {d.get('name') or ''}  ({d.get('published_at','')[:10]})"
                 for d in data]
        return _clip("\n".join(lines)) or "(no releases)"
    lines = [f"#{d.get('number')} [{d.get('state')}] {d.get('title')}  {d.get('html_url')}"
             for d in data]
    return _clip("\n".join(lines)) or "(none)"


def fetch_docs(package: str, ecosystem: str = "") -> str:
    """Fetch package registry metadata: PyPI, npm, or crates.io.

    If ecosystem is omitted, tries pypi, then npm, then crates."""
    order = [ecosystem] if ecosystem else ["pypi", "npm", "crates"]
    errs = []
    for eco in order:
        if eco == "pypi":
            url = f"https://pypi.org/pypi/{package}/json"
        elif eco == "npm":
            url = f"https://registry.npmjs.org/{package}"
        elif eco == "crates":
            url = f"https://crates.io/api/v1/crates/{package}"
        else:
            return "ERROR: ecosystem must be pypi | npm | crates"
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "kode/1.0"})
        except requests.RequestException as e:
            errs.append(f"{eco}: {e}")
            continue
        if r.status_code != 200:
            errs.append(f"{eco}: HTTP {r.status_code}")
            continue
        try:
            d = r.json()
        except ValueError:
            errs.append(f"{eco}: non-JSON")
            continue
        if eco == "pypi":
            info = d.get("info", {})
            return _clip(f"{info.get('name')} {info.get('version')} (PyPI)\n"
                         f"{info.get('summary') or ''}\n"
                         f"Home: {info.get('home_page') or info.get('project_url') or ''}\n"
                         f"Requires-Python: {info.get('requires_python') or '-'}\n"
                         f"License: {info.get('license') or '-'}\n\n"
                         f"{(info.get('description') or '')[:4000]}")
        if eco == "npm":
            latest = (d.get("dist-tags") or {}).get("latest", "")
            ver = (d.get("versions") or {}).get(latest, {})
            return _clip(f"{d.get('name')} {latest} (npm)\n"
                         f"{d.get('description') or ''}\n"
                         f"Home: {d.get('homepage') or ''}\n"
                         f"License: {ver.get('license') or d.get('license') or '-'}\n\n"
                         f"{(d.get('readme') or '')[:4000]}")
        if eco == "crates":
            c = d.get("crate", {})
            return _clip(f"{c.get('name')} {c.get('max_stable_version') or c.get('max_version')} (crates.io)\n"
                         f"{c.get('description') or ''}\n"
                         f"Docs: {c.get('documentation') or ''}\n"
                         f"Repo: {c.get('repository') or ''}\n"
                         f"Downloads: {c.get('downloads')}")
    return "ERROR: package not found — " + "; ".join(errs)


def fetch_error(query: str, max_results: int = 5) -> str:
    """Search Stack Overflow for an error/question via the Stack Exchange API.

    Returns top questions with links, score, answered flag; includes the accepted
    answer body for the best hit when cheap to do so."""
    try:
        r = requests.get(
            "https://api.stackexchange.com/2.3/search/advanced",
            params={"order": "desc", "sort": "relevance", "q": query,
                    "site": "stackoverflow", "pagesize": max_results,
                    "filter": "withbody"},
            timeout=30, headers={"User-Agent": "kode/1.0"})
    except requests.RequestException as e:
        return f"ERROR: {e}"
    if r.status_code != 200:
        return f"ERROR: Stack Exchange returned HTTP {r.status_code}"
    try:
        items = r.json().get("items", [])
    except ValueError:
        return "ERROR: non-JSON response from Stack Exchange"
    if not items:
        return "(no results)"
    out = []
    for it in items:
        ans = "answered" if it.get("is_answered") else "unanswered"
        out.append(f"[{it.get('score')}] {html.unescape(it.get('title',''))} "
                   f"({ans})\n   {it.get('link')}")
    top = items[0]
    acc = top.get("accepted_answer_id")
    if acc:
        try:
            ar = requests.get(
                f"https://api.stackexchange.com/2.3/answers/{acc}",
                params={"site": "stackoverflow", "filter": "withbody"},
                timeout=30, headers={"User-Agent": "kode/1.0"})
            a_items = ar.json().get("items", []) if ar.status_code == 200 else []
            if a_items:
                body = _strip_html(a_items[0].get("body", "")).strip()
                out.append(f"\n--- accepted answer for top result ---\n{body[:3000]}")
        except (requests.RequestException, ValueError):
            pass
    return _clip("\n".join(out))


def fetch_readme(target: str) -> str:
    """Fetch a README: 'owner/repo' (GitHub) or a bare package name (PyPI)."""
    if "/" in target:
        try:
            r = requests.get(f"https://api.github.com/repos/{target}/readme",
                             timeout=30, headers=_gh_headers())
        except requests.RequestException as e:
            return f"ERROR: {e}"
        if r.status_code == 404:
            return f"ERROR: no README found for {target}"
        if r.status_code != 200:
            return f"ERROR: GitHub returned HTTP {r.status_code}"
        try:
            d = r.json()
            import base64
            body = base64.b64decode(d.get("content", "")).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return "ERROR: could not decode README"
        return _clip(body)
    # bare name → PyPI description
    try:
        r = requests.get(f"https://pypi.org/pypi/{target}/json",
                         timeout=30, headers={"User-Agent": "kode/1.0"})
    except requests.RequestException as e:
        return f"ERROR: {e}"
    if r.status_code != 200:
        return f"ERROR: package '{target}' not found on PyPI (HTTP {r.status_code})"
    try:
        info = r.json().get("info", {})
    except ValueError:
        return "ERROR: non-JSON response from PyPI"
    desc = info.get("description") or info.get("summary") or ""
    return _clip(f"{info.get('name')} {info.get('version')}\n\n{desc}") or "(no description)"


def _dig(data, keypath: str):
    """Walk a dotted/indexed key path like 'items[0].name' into JSON data."""
    cur = data
    for token in re.findall(r"[^.\[\]]+", keypath):
        if isinstance(cur, list):
            cur = cur[int(token)]
        elif isinstance(cur, dict):
            cur = cur[token]
        else:
            raise KeyError(token)
    return cur


def fetch_json(url: str, keypath: str = "") -> str:
    """Fetch a URL (safely), parse JSON, optionally filter by a key path
    like 'items[0].name', and pretty-print the result."""
    import json as _json
    r = _safe_get(url)
    if isinstance(r, str):
        return r
    try:
        data = r.json()
    except ValueError:
        return "ERROR: response body is not valid JSON"
    if keypath:
        try:
            data = _dig(data, keypath)
        except (KeyError, IndexError, ValueError) as e:
            return f"ERROR: key path '{keypath}' not found ({e})"
    try:
        return _clip(_json.dumps(data, indent=2, ensure_ascii=False))
    except (TypeError, ValueError):
        return _clip(str(data))


def fetch_mdn(query: str, max_results: int = 5) -> str:
    """Search MDN Web Docs; returns title / url / summary of the top hits."""
    try:
        r = requests.get("https://developer.mozilla.org/api/v1/search",
                         params={"q": query}, timeout=30,
                         headers={"User-Agent": "kode/1.0"})
    except requests.RequestException as e:
        return f"ERROR: {e}"
    if r.status_code != 200:
        return f"ERROR: MDN returned HTTP {r.status_code}"
    try:
        docs = r.json().get("documents", [])
    except ValueError:
        return "ERROR: non-JSON response from MDN"
    if not docs:
        return "(no results)"
    out = []
    for d in docs[:max_results]:
        out.append(f"{d.get('title')}\n   https://developer.mozilla.org{d.get('mdn_url')}"
                   f"\n   {d.get('summary','').strip()}")
    return _clip("\n".join(out))


def fetch_wayback(url: str, timestamp: str = "") -> str:
    """Fetch the closest Internet Archive snapshot of a URL, stripped to text."""
    params = {"url": url}
    if timestamp:
        params["timestamp"] = timestamp
    try:
        r = requests.get("https://archive.org/wayback/available", params=params,
                         timeout=30, headers={"User-Agent": "kode/1.0"})
    except requests.RequestException as e:
        return f"ERROR: {e}"
    if r.status_code != 200:
        return f"ERROR: wayback returned HTTP {r.status_code}"
    try:
        snap = ((r.json().get("archived_snapshots") or {}).get("closest") or {})
    except ValueError:
        return "ERROR: non-JSON response from wayback"
    snap_url = snap.get("url")
    if not snap_url or not snap.get("available"):
        return f"ERROR: no wayback snapshot found for {url}"
    resp = _safe_get(snap_url)  # arbitrary URL → go through the SSRF guard
    if isinstance(resp, str):
        return resp
    text = resp.text
    if "html" in resp.headers.get("content-type", ""):
        text = _strip_html(text)
    return _clip(f"[snapshot {snap.get('timestamp','')}] {snap_url}\n\n{text.strip()}")


def fetch_rfc(number: int, offset: int = 0, limit: int = 0) -> str:
    """Fetch the plain-text of an IETF RFC by number, optionally line-sliced."""
    try:
        n = int(number)
    except (TypeError, ValueError):
        return "ERROR: number must be an integer RFC number"
    try:
        r = requests.get(f"https://www.rfc-editor.org/rfc/rfc{n}.txt",
                         timeout=30, headers={"User-Agent": "kode/1.0"})
    except requests.RequestException as e:
        return f"ERROR: {e}"
    if r.status_code == 404:
        return f"ERROR: RFC {n} not found"
    if r.status_code != 200:
        return f"ERROR: rfc-editor returned HTTP {r.status_code}"
    lines = r.text.splitlines()
    if offset or limit:
        end = offset + limit if limit else len(lines)
        lines = lines[offset:end]
    return _clip("\n".join(lines))


def fetch_manpage(name: str, section: str = "") -> str:
    """Fetch a man page: local `man` first, else man7.org, stripped to text."""
    args = ["man", "-P", "cat"]
    if section:
        args.append(section)
    args.append(name)
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            # strip backspace-overstrike bolding that some man setups emit
            out = re.sub(r".\x08", "", r.stdout)
            return _clip(out.strip())
    except Exception:  # noqa: BLE001 — man missing/slow: fall through to the web
        pass
    sec = section or "1"
    url = f"https://man7.org/linux/man-pages/man{sec}/{name}.{sec}.html"
    resp = _safe_get(url)
    if isinstance(resp, str):
        return resp
    if resp.status_code != 200:
        return f"ERROR: no man page for '{name}' locally or on man7.org"
    return _clip(_strip_html(resp.text).strip())


def fetch_pdf(url: str, pages: str = "") -> str:
    """Download a PDF (safely) and extract its text. `pages` is an optional
    range like '1-3' or '2'. Requires the pypdf package."""
    try:
        import pypdf  # noqa: F401 — imported lazily so it's not a hard dep
    except ImportError:
        return ("ERROR: pypdf is not installed — run `pip install pypdf` to enable "
                "fetch_pdf.")
    resp = _safe_get(url, max_bytes=MAX_FETCH_BYTES)
    if isinstance(resp, str):
        return resp
    import io
    try:
        reader = pypdf.PdfReader(io.BytesIO(resp.content))
    except Exception as e:  # noqa: BLE001
        return f"ERROR: could not parse PDF ({e})"
    n = len(reader.pages)
    start, end = 0, n
    if pages:
        m = re.match(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s*$", pages)
        if not m:
            return "ERROR: pages must look like '2' or '1-3'"
        start = max(0, int(m.group(1)) - 1)
        end = int(m.group(2)) if m.group(2) else start + 1
        end = min(end, n)
    parts = []
    for i in range(start, end):
        try:
            parts.append(reader.pages[i].extract_text() or "")
        except Exception:  # noqa: BLE001
            parts.append("")
    text = "\n".join(parts).strip()
    return _clip(f"[{n} pages, showing {start + 1}-{end}]\n\n{text}") or "(no extractable text)"


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
        _READ_MTIMES.pop(str(p), None)
        return f"removed {rel} (was newly created)"
    p.write_text(prev)
    _touch(p)  # keep the stale-guard from flagging our own revert as an outside edit
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
    "fetch_github": fetch_github,
    "fetch_docs": fetch_docs,
    "fetch_error": fetch_error,
    "fetch_readme": fetch_readme,
    "fetch_json": fetch_json,
    "fetch_mdn": fetch_mdn,
    "fetch_wayback": fetch_wayback,
    "fetch_rfc": fetch_rfc,
    "fetch_manpage": fetch_manpage,
    "fetch_pdf": fetch_pdf,
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
            "name": "fetch_github",
            "description": "Fetch from GitHub for an owner/repo: file contents, issues, PRs, or releases. Use when you need source, an issue/PR thread, or release notes from a specific repo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "'owner/name'"},
                    "kind": {"type": "string", "enum": ["file", "issues", "prs", "releases"]},
                    "path": {"type": "string", "description": "in-repo path when kind=file"},
                    "ref": {"type": "string", "description": "branch/tag/sha for kind=file"},
                    "number": {"type": "integer", "description": "a specific issue/PR number"},
                },
                "required": ["repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_docs",
            "description": "Fetch package metadata (version, summary, description, links) from PyPI, npm, or crates.io. Use to check a library's current version or what it does. Ecosystem auto-detected if omitted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "ecosystem": {"type": "string", "enum": ["pypi", "npm", "crates"]},
                },
                "required": ["package"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_error",
            "description": "Search Stack Overflow for an error message or how-to question; returns top questions (with links, score, answered flag) plus the accepted answer for the best hit. Use when debugging an unfamiliar error.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": "default 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_readme",
            "description": "Fetch the README for a GitHub 'owner/repo', or the long description for a bare PyPI package name. Use for a quick overview of a project or library.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_json",
            "description": "Fetch a URL, parse it as JSON, optionally drill in with a key path like 'items[0].name', and pretty-print. Use for JSON API endpoints where you want a specific field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "keypath": {"type": "string", "description": "e.g. 'items[0].name'"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_mdn",
            "description": "Search MDN Web Docs for web platform APIs (JS/CSS/HTTP); returns title/url/summary of top hits. Use for authoritative web-standards references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": "default 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_wayback",
            "description": "Fetch an archived Internet Archive snapshot of a URL (stripped to text). Use when a page is down, changed, or you need a historical version.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "timestamp": {"type": "string", "description": "optional YYYYMMDD to target"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_rfc",
            "description": "Fetch the plain text of an IETF RFC by number, optionally line-sliced. Use for protocol/spec references (HTTP, TLS, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "offset": {"type": "integer", "description": "0-indexed start line"},
                    "limit": {"type": "integer", "description": "max lines"},
                },
                "required": ["number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_manpage",
            "description": "Fetch a Unix man page (local `man` first, else man7.org), as text. Use for CLI flags and system-call details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "section": {"type": "string", "description": "optional man section, e.g. '2'"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_pdf",
            "description": "Download a PDF and extract its text (optional page range like '1-3'). Requires pypdf. Use to read PDF docs/specs/papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "pages": {"type": "string", "description": "e.g. '2' or '1-3'"},
                },
                "required": ["url"],
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
            "name": "spawn_swarm",
            "description": "Fan a BROAD goal out to a swarm of parallel read-only workers (up to 10). A planner splits the goal into independent angles, each worker investigates one, and a synthesis pass merges everything into one report. Use for wide investigations (audit a codebase, survey a design space, map a large module) where you don't already know the exact sub-questions — if you do, prefer spawn_agents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "the overall goal to investigate"},
                    "n": {"type": "integer", "description": "number of workers, 2-10 (default 6)"},
                    "model": {"type": "string", "description": "optional model id the workers run on (e.g. a cheap model for breadth)"},
                },
                "required": ["task"],
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
