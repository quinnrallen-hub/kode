"""Git-backed checkpoints for kode.

Uses a *shadow* git repo (a separate GIT_DIR whose work-tree is the workspace)
so we can snapshot every turn and roll back without ever touching the user's own
git history. .gitignore in the workspace is still honored; we add a few coarse
excludes on top. If git isn't available the whole thing no-ops gracefully.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from shutil import which

_EXCLUDES = [".git", "node_modules", "__pycache__", ".venv", "venv", "dist",
             "build", ".mypy_cache", ".pytest_cache", ".next", ".kode-jobs"]


class Checkpointer:
    def __init__(self, workspace: Path, home: Path):
        self.workspace = Path(workspace).resolve()
        slug = hashlib.sha1(str(self.workspace).encode()).hexdigest()[:16]
        self.git_dir = Path(home) / "shadow" / slug
        self.enabled = which("git") is not None
        self.reason = "" if self.enabled else "git not installed"
        # Refuse to snapshot an unbounded tree (home dir, filesystem root): a
        # `git add -A` there would index everything under it — huge and slow, and
        # would sweep in caches/secrets. Checkpoints just switch off; kode still
        # runs, you only lose per-turn file rollback in that directory.
        if self.enabled and os.environ.get("KODE_ALLOW_BROAD_CKPT") != "1":
            broad = {Path.home().resolve(), Path(self.workspace.anchor).resolve()}
            if self.workspace in broad and not (self.workspace / ".git").exists():
                self.enabled = False
                self.reason = ("workspace is your home/root dir — checkpoints off "
                               "(set KODE_ALLOW_BROAD_CKPT=1 to force)")

    # -- low-level ---------------------------------------------------------- #
    def _git(self, *args: str, check: bool = False) -> subprocess.CompletedProcess:
        env = {**os.environ,
               "GIT_DIR": str(self.git_dir),
               "GIT_WORK_TREE": str(self.workspace),
               "GIT_AUTHOR_NAME": "kode", "GIT_AUTHOR_EMAIL": "kode@localhost",
               "GIT_COMMITTER_NAME": "kode", "GIT_COMMITTER_EMAIL": "kode@localhost"}
        return subprocess.run(["git", *args], env=env, capture_output=True,
                              text=True, check=check)

    def setup(self) -> bool:
        """Init the shadow repo (if needed) and record a baseline snapshot."""
        if not self.enabled:
            return False
        if not (self.git_dir / "HEAD").exists():
            self.git_dir.mkdir(parents=True, exist_ok=True)
            self._git("init", "-q")
            info = self.git_dir / "info"
            info.mkdir(exist_ok=True)
            (info / "exclude").write_text("\n".join(_EXCLUDES) + "\n")
        self.snapshot("session start")
        return True

    # -- operations --------------------------------------------------------- #
    def snapshot(self, label: str) -> str | None:
        """Commit the current worktree. Returns the short hash, or None if
        nothing changed since the last snapshot."""
        if not self.enabled:
            return None
        self._git("add", "-A")
        # any staged change?
        if self._git("diff", "--cached", "--quiet").returncode == 0 and self._head():
            return None
        r = self._git("commit", "-q", "-m", label, "--allow-empty-message")
        if r.returncode != 0:
            return None
        return self._head()

    def _head(self) -> str | None:
        r = self._git("rev-parse", "--short", "HEAD")
        return r.stdout.strip() if r.returncode == 0 else None

    def diff(self, since: str | None) -> str:
        if not self.enabled:
            return "(checkpoints disabled — git unavailable)"
        base = since or self._first_commit()
        if not base:
            return "(no checkpoints yet)"
        self._git("add", "-A")
        out = self._git("diff", "--stat", base).stdout
        detail = self._git("diff", base).stdout
        return (out + "\n" + detail).strip() or "(no changes since checkpoint)"

    def reset(self, to_hash: str) -> bool:
        """Restore the worktree to a snapshot (tracked files added later are
        removed; never touches excluded/untracked files)."""
        if not self.enabled or not to_hash:
            return False
        return self._git("reset", "--hard", to_hash).returncode == 0

    def changed_files(self, since: str | None) -> list[str]:
        if not self.enabled:
            return []
        base = since or self._first_commit()
        if not base:
            return []
        self._git("add", "-A")
        r = self._git("diff", "--name-only", base)
        return [f for f in r.stdout.splitlines() if f]

    def _first_commit(self) -> str | None:
        r = self._git("rev-list", "--max-parents=0", "HEAD")
        return r.stdout.split()[0] if r.returncode == 0 and r.stdout.strip() else None
