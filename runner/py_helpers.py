# ---------------------------------------------------------------------------
# Python runner
# ---------------------------------------------------------------------------

_UNITTEST_TAIL_RE = re.compile(
    r"Ran\s+(\d+)\s+tests?\s+in\s+([0-9.]+)s\s*\n+"
    r"(OK|FAILED)(?:\s*\(([^)]*)\))?",
    re.MULTILINE,
)
_UNITTEST_COUNTS_RE = re.compile(r"(failures|errors|skipped)=(\d+)")
# `unittest` collapses any module that fails to import (ModuleNotFoundError,
# SyntaxError at top level, etc.) into a single ``_FailedTest`` node, so
# ``Ran N tests`` undercounts. Detect this and recover the real method count
# by scanning the source file for ``def test_*``.
_FAILED_TEST_NODE_RE = re.compile(
    r"\b(?:test_\w+|unit_test_\w+)\s*\(unittest\.loader\._FailedTest\)"
)
_FAILED_IMPORT_NAME_RE = re.compile(
    r"Failed to import test module:\s*(\S+)"
)


_TEST_METHOD_DEF_RE = re.compile(r"^\s*def\s+test_\w+\s*\(", re.MULTILINE)


def _parse_unittest_output(text: str) -> tuple[int, int, int, int]:
    """Return (n_total, n_passed, n_failed, n_errors). Best-effort.

    ``Ran N tests`` counts top-level test methods, but ``failures=X errors=Y``
    counts every individual fail/error — including subtests. A test method
    decorated with ``self.subTest(...)`` that errors three times shows up as
    ``Ran 1 tests`` + ``errors=3``. Naive subtraction
    (``n_total - failures - errors - skipped``) then goes negative. Clamp at
    zero: if a top-level test had any subtest fail/error, it didn't fully
    pass, so 0 passed is the right floor.
    """
    m = _UNITTEST_TAIL_RE.search(text)
    if not m:
        return 0, 0, 0, 0
    n_total = int(m.group(1))
    failures = errors = skipped = 0
    if m.group(4):
        for k, v in _UNITTEST_COUNTS_RE.findall(m.group(4)):
            if k == "failures":
                failures = int(v)
            elif k == "errors":
                errors = int(v)
            elif k == "skipped":
                skipped = int(v)
    n_passed = max(0, n_total - failures - errors - skipped)
    return n_total, n_passed, failures, errors


def _python_static_test_methods(py_path: Path) -> int:
    """Count ``def test_*`` definitions in a Python test file."""
    try:
        src = py_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return len(_TEST_METHOD_DEF_RE.findall(src))


def _module_to_path(module: str, code_dir: Path) -> Optional[Path]:
    """Resolve dotted module name to its source file under code_dir."""
    p = code_dir / (module.replace(".", "/") + ".py")
    return p if p.is_file() else None


def _recover_static_test_count(
    out: str, modules: list[str], code_dir: Path
) -> tuple[int, int]:
    """When unittest reports ``_FailedTest`` nodes (module-level import
    failures collapsed into a single test), scan the source files of those
    modules for ``def test_*`` and return ``(extra_total, extra_errors)``
    so the caller can replace the misleading dynamic counts with the real
    method count. The collapsed node already counts as 1, so we only add
    ``static - 1`` per failed module."""
    failed_basenames: set[str] = set()
    for m in _FAILED_TEST_NODE_RE.finditer(out):
        failed_basenames.add(m.group(0).split("(")[0].strip())
    for m in _FAILED_IMPORT_NAME_RE.finditer(out):
        failed_basenames.add(m.group(1))
    if not failed_basenames:
        return 0, 0

    extra = 0
    for mod in modules:
        base = mod.rpartition(".")[2]
        if base not in failed_basenames:
            continue
        py = _module_to_path(mod, code_dir)
        if py is None:
            continue
        static = _python_static_test_methods(py)
        if static > 1:
            extra += static - 1
    return extra, extra


def _find_python_code_dir(sandbox: Path) -> Optional[Path]:
    """Locate the directory that holds ``unit_test/test_*.py`` (or
    ``unit_test_*.py``) for this sample.

    Two layouts ship in the dataset:
      - ICLR samples:           ``<sandbox>/code/unit_test/``
      - Nature Comms samples:   ``<sandbox>/<project>/unit_test/``

    Return the directory whose direct child is ``unit_test/``. Prefer the
    legacy ``code/`` location, then any single top-level subdir with a
    ``unit_test/`` inside, then a recursive fallback.
    """
    legacy = sandbox / "code"
    if (legacy / "unit_test").is_dir():
        return legacy
    for sub in sorted(sandbox.iterdir()):
        if sub.is_dir() and (sub / "unit_test").is_dir():
            return sub
    for ut in sandbox.rglob("unit_test"):
        if ut.is_dir() and any(ut.glob("*.py")):
            return ut.parent
    return None


def _python_test_modules(status: dict, code_dir: Path) -> tuple[list[str], list[str]]:
    """Return (modules_to_run, dropped_modules).

    Source order: ``unit_test_status.json`` if present (authoritative for
    intent), filtered to modules whose ``.py`` actually exists. If status
    lists nothing, glob ``unit_test/`` for both ``test_*.py`` (ICLR
    convention) and ``unit_test_*.py`` (Nature Comms convention).

    A dataset entry like ``code/unit_test/test_foo.py`` may be shipped as
    ``test_foo.py.disabled`` — filter those out.
    """
    listed = (status.get("file_structure", {}) or {}).get("test_files", []) or []
    candidates: list[tuple[str, Path]] = []  # (module_name, .py path under code_dir)
    for f in listed:
        p = Path(f)
        parts = list(p.with_suffix("").parts)
        # Strip a leading project-dir prefix (``code`` or e.g. ``SkinGPT-4``)
        # when it matches code_dir's basename, so the resulting module path is
        # importable from cwd=code_dir.
        if parts and parts[0] in ("code", code_dir.name):
            parts = parts[1:]
        if not parts:
            continue
        mod = ".".join(parts)
        candidates.append((mod, code_dir / Path(*parts).with_suffix(".py")))

    if not candidates:
        # Fallback: discover real test files under unit_test/.
        test_dir = code_dir / "unit_test"
        if test_dir.is_dir():
            seen: set[str] = set()
            for pattern in ("test_*.py", "unit_test_*.py"):
                for f in sorted(test_dir.glob(pattern)):
                    if f.stem in seen:
                        continue
                    seen.add(f.stem)
                    candidates.append((f"unit_test.{f.stem}", f))

    keep, drop = [], []
    for mod, py in candidates:
        if py.is_file():
            keep.append(mod)
        else:
            drop.append(mod)
    return keep, drop


_TEST_REFERENCE_COMPARE = (
    r"os\.environ\.get\(['\"]TEST_REFERENCE['\"]\)\s*==\s*['\"]true['\"]"
)
_TEST_REFERENCE_NOT_COMPARE = (
    r"os\.environ\.get\(['\"]TEST_REFERENCE['\"]\)\s*!=\s*['\"]true['\"]"
)
_PY_TEST_REF_ASSIGN_RE = re.compile(
    rf"^(\s{{8,}})([^#\n]*?=\s*){_TEST_REFERENCE_COMPARE}(\s*(?:#.*)?)$"
)
_PY_TEST_REF_NOT_ASSIGN_RE = re.compile(
    rf"^(\s{{8,}})([^#\n]*?=\s*){_TEST_REFERENCE_NOT_COMPARE}(\s*(?:#.*)?)$"
)
_PY_TEST_REF_IF_RE = re.compile(
    rf"^(\s{{8,}})if\s+{_TEST_REFERENCE_COMPARE}\s*:(\s*(?:#.*)?)$"
)
_PY_TEST_REF_NOT_IF_RE = re.compile(
    rf"^(\s{{8,}})if\s+{_TEST_REFERENCE_NOT_COMPARE}\s*:(\s*(?:#.*)?)$"
)
_PY_SUBMISSION_REF_EXPR = (
    "os.environ.get('TEST_REFERENCE') == 'true' "
    "or os.environ.get('TEST_SUBMISSION') == 'true'"
)
_PY_SUBMISSION_MASKED_EXPR = (
    "os.environ.get('TEST_REFERENCE') != 'true' "
    "and os.environ.get('TEST_SUBMISSION') != 'true'"
)


def _python_if_block_imports_reference(lines: list[str], idx: int,
                                       indent_len: int) -> bool:
    """Return True when a TEST_REFERENCE branch is selecting golden imports.

    Submission evaluation needs target-code imports but golden-style assertions.
    Therefore sandbox rewriting must not broaden import-selection guards.
    """
    for line in lines[idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        cur_indent = len(line) - len(line.lstrip())
        if cur_indent <= indent_len:
            break
        if ((stripped.startswith("import ") or stripped.startswith("from "))
                and ("golden" in stripped or "reference" in stripped)):
            return True
    return False
