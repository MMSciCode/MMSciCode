"""runner.py — execute each sample's own unit tests against insert.py outputs.

Pipeline position
-----------------
After ``insert.py`` has spliced model output into a sandbox tree under
``intermediate_outputs/patched/<lang>__<sample>__<mode>/``, this script
takes those patched files, plants them into a fresh copy of the
dataset sample, and runs that sample's unit tests in the language's
expected environment (conda env, R env, gcc) — exactly the way each
sample's authors set them up.

Per-language test invariants observed in the migrated dataset
-------------------------------------------------------------
- Python/R tests import or source the unsuffixed working file. The sandbox is
  first materialized from ``*_reference`` files, then model patches are overlaid
  onto the target source file.
- C: the patched source is syntax-checked with the matching compiler
  (``gcc``/``g++``) against the real headers, then the sample's cmake
  ``realtest_*`` targets are built and run, and the pre-built
  simple_test_golden / simple_test_masked binaries are re-run
  (informationally) when available.

Each (sample, mode) result is independent; failures in one don't
affect others. The summary JSON is the canonical machine output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


_LANG_DIR = {"python": "Python", "c": "C_CPP", "r": "R"}
_LANG_EXT = {"python": ".py", "c": ".c", "r": ".R"}
# Source/header extensions to overlay per language. Compared case-insensitively
# (via ``.suffix.lower()``) so uppercase ``.R`` and lowercase ``.r`` both match,
# and so C/C++ targets that live in ``.cpp``/``.hpp``/``.h``/... are overlaid
# too (the single-extension glob used to skip them, leaving the golden
# reference in the sandbox and scoring the model on the wrong file).
_LANG_EXTS = {
    "python": (".py",),
    "c": (".c", ".cpp", ".cc", ".cxx", ".c++", ".h", ".hpp", ".hh", ".hxx"),
    "r": (".r",),
}
_DEFAULT_CONDA_ROOT = Path("/opt/conda")
_DEFAULT_TIMEOUT_S = 600


# ---------------------------------------------------------------------------
# Path / discovery helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _resolve_data_root(explicit: Optional[Path]) -> Path:
    if explicit:
        return explicit.resolve()
    here = Path(__file__).resolve().parent
    for cand in (here / "data", here.parent / "data", Path.cwd() / "data"):
        if cand.is_dir() and any((cand / d).is_dir() for d in _LANG_DIR.values()):
            return cand.resolve()
    raise FileNotFoundError("Could not locate data root. Pass --data-root explicitly.")


def _sample_dir(data_root: Path, lang_dir: str, sample_id: str) -> Path:
    """Resolve both legacy `Python/<sample>` and HF `Python/data/<sample>` layouts."""
    for cand in (data_root / lang_dir / sample_id,
                 data_root / lang_dir / "data" / sample_id):
        if cand.is_dir():
            return cand
    return data_root / lang_dir / sample_id


def _parse_key(name: str) -> Optional[tuple[str, str, str, str]]:
    """``python__037__00-foo__direct`` -> (lang, sample, func_part, mode).

    Per-function flow only — keys are exactly 4 ``__``-separated parts."""
    parts = name.split("__")
    if len(parts) != 4:
        return None
    lang, sample, func_part, mode = parts
    if lang not in _LANG_DIR or not sample or not func_part or not mode:
        return None
    return lang, sample, func_part, mode


def _conda_env_path(env_name: str, conda_root: Path) -> Optional[Path]:
    cand = conda_root / "envs" / env_name
    return cand if cand.is_dir() else None


# Regex patterns that signal "the env can't run the tests" rather than "the
# model's code is buggy". Used to demote a 0/0 generic error into the more
# accurate `env_broken` bucket so it doesn't pollute model-quality stats.
_ENV_BROKEN_PATTERNS = (
    re.compile(r"\bModuleNotFoundError\b"),
    re.compile(r"\bImportError:\s+(?:cannot import name|No module|libstdc)"),
    # ABI/version mismatches surface as ``AttributeError: module 'numpy' has
    # no attribute 'BitGenerator'`` (and similar) during ``import pandas``
    # at test discovery time. These are env problems, not model bugs.
    re.compile(r"AttributeError:\s+module ['\"](?:numpy|pandas|torch|scipy|sklearn|matplotlib|tensorflow)[\"'.]"),
    re.compile(r"there is no package called"),
    re.compile(r"loadNamespace\([^)]*\)\s*:\s*there is no"),
    re.compile(r"package or namespace load failed for"),
    re.compile(r"DLL load failed"),
    # R/C native-extension crashes before a parseable test summary are not a
    # model-quality signal. Keep them separate from ordinary assertion fails.
    re.compile(r"\*\*\* caught segfault \*\*\*"),
    re.compile(r"\bC stack usage\b.*\btoo close to the limit\b"),
    re.compile(r"\bAn irrecoverable exception occurred\. R is aborting now\b"),
)


def _looks_like_env_failure(log: str) -> bool:
    return any(p.search(log) for p in _ENV_BROKEN_PATTERNS)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    status: str           # "pass" | "fail" | "error" | "compile_ok" | "compile_fail" | "skipped" | "timeout" | "env_missing" | "env_broken"
    n_tests: int = 0
    n_passed: int = 0
    n_failed: int = 0
    n_errors: int = 0
    duration_s: float = 0.0
    log_excerpt: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class RunnerResult:
    key: str
    sample_id: str
    language: str
    mode: str
    sandbox: str
    func_part: str = ""        # "<idx:02d>-<slug>" for the per-function flow
    function_name: str = ""    # canonical name from prompt.json, when available
    tests: list[TestResult] = field(default_factory=list)
    overall: str = "unknown"   # "pass" | "fail" | "compile_ok" | "compile_fail" | "error" | "skipped"


# ---------------------------------------------------------------------------
# Sandbox: copy dataset sample, then overlay patched files on top
# ---------------------------------------------------------------------------

def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)


def _apply_patches(sandbox: Path, patched_dir: Path, language: str) -> list[str]:
    """Overlay every ``*<ext>`` file under ``patched_dir`` onto ``sandbox``.

    Skips ``*_masked.<ext>`` snapshots — those are insert.py bookkeeping.
    Returns a list of human-readable notes about what was overlaid.
    """
    notes: list[str] = []
    exts = _LANG_EXTS[language]

    for src in patched_dir.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix.lower() not in exts:
            continue
        if src.stem.endswith("_masked"):  # skip insert.py masked snapshots (any ext)
            continue
        rel = src.relative_to(patched_dir)
        dst = sandbox / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        notes.append(f"overlay {rel}")

    return notes


def _materialize_references(sandbox: Path, status: dict) -> list[str]:
    """Copy reference implementations onto unsuffixed working paths in sandbox."""
    notes: list[str] = []
    for target in status.get("target_functions", []):
        src_rel = target.get("src_file") or target.get("src") or target.get("file_path") or target.get("original_file")
        ref_rel = target.get("reference_file") or target.get("reference") or target.get("golden")
        if not src_rel or not ref_rel:
            continue
        src = sandbox / src_rel
        ref = sandbox / ref_rel
        if not ref.is_file():
            continue
        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ref, src)
        notes.append(f"materialize reference {ref_rel} -> {src_rel}")
    return notes


# ---------------------------------------------------------------------------
# Subprocess runner with timeout + log capture
# ---------------------------------------------------------------------------

def _run_cmd(
    cmd: list[str],
    cwd: Path,
    env: Optional[dict[str, str]] = None,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> tuple[int, str, float]:
    """Run a command; return (returncode, combined_output, elapsed_s).

    Returncode -1 means timeout, -2 means the binary couldn't be launched.
    """
    start = time.monotonic()
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            text=True,
            errors="replace",
            check=False,
        )
        return proc.returncode, proc.stdout or "", time.monotonic() - start
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        return -1, out + f"\n[timeout after {timeout_s}s]\n", time.monotonic() - start
    except FileNotFoundError as e:
        return -2, f"[binary not found: {e}]\n", time.monotonic() - start
    except OSError as e:
        # PermissionError, ENOTDIR (cwd missing), ENOEXEC, etc. — surfacing
        # these as a runner result (rather than letting them propagate and
        # kill the worker) keeps the rest of the batch alive.
        return -3, f"[OSError: {type(e).__name__}: {e}]\n", time.monotonic() - start


def _tail(text: str, n: int = 60) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text
