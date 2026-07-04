# ---------------------------------------------------------------------------
# Per-(sample, mode) orchestration
# ---------------------------------------------------------------------------

def _load_status(sample_dir: Path) -> dict:
    p = sample_dir / "unit_test_status.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(_read(p))
    except json.JSONDecodeError:
        return {}


def _patched_c_files(insert_summary: list[dict], key: str,
                     patched_dir: Path) -> list[str]:
    """Return repo-relative C/C++ source/header file paths that were patched.

    Prefers ``insert.py``'s ``_summary.json`` when present (authoritative).
    Falls back to globbing the C/C++ source/header extensions under
    ``patched_dir`` (excluding ``*_masked.*`` snapshots) — works when insert.py
    was invoked per-key without writing a roll-up summary. The glob matches
    ``.cpp``/``.hpp``/... (case-insensitively), not only ``.c``.
    """
    for entry in insert_summary:
        if entry.get("key") == key:
            ok = [f["file_path"] for f in entry.get("files", []) if f.get("success")]
            if ok:
                return ok
    # Fallback: filesystem discovery
    exts = _LANG_EXTS["c"]
    return sorted(
        str(p.relative_to(patched_dir))
        for p in patched_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts and not p.stem.endswith("_masked")
    )


def _overall_status(tests: list[TestResult]) -> str:
    if not tests:
        return "error"
    if all(t.status in ("pass", "compile_ok") for t in tests):
        return "pass" if any(t.status == "pass" for t in tests) else "compile_ok"
    if any(t.status == "timeout" for t in tests):
        return "timeout"
    # Env problems must take priority over fail/compile_fail so a single env-broken
    # test does not get masked when sibling tests in the same key happen to also
    # have non-zero counts (Phase G post-filter relies on this label to surface
    # infra-blocked attempts cleanly). Reviewed in iter2_reviews/pipeline_integrity_review.md.
    if any(t.status == "env_missing" for t in tests):
        return "env_missing"
    if any(t.status == "env_broken" for t in tests):
        return "env_broken"
    if any(t.status in ("fail", "compile_fail") for t in tests):
        return "compile_fail" if all(t.status in ("compile_ok", "compile_fail") for t in tests) else "fail"
    return "error"


def run_one(
    patched_dir: Path,
    data_root: Path,
    insert_summary: list[dict],
    sandbox_root: Path,
    conda_root: Path,
    timeout_s: int,
    keep_sandbox: bool,
) -> RunnerResult:
    parsed = _parse_key(patched_dir.name)
    if not parsed:
        raise ValueError(f"cannot parse key: {patched_dir.name}")
    lang, sample_id, func_part, mode = parsed

    sample_src = _sample_dir(data_root, _LANG_DIR[lang], sample_id)
    if not sample_src.is_dir():
        raise FileNotFoundError(f"sample dir not found: {sample_src}")

    sandbox = sandbox_root / patched_dir.name
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)
    _copy_tree(sample_src, sandbox)
    status_json = _load_status(sample_src)
    overlay_notes = _materialize_references(sandbox, status_json)
    overlay_notes.extend(_apply_patches(sandbox, patched_dir, lang))

    if lang == "python":
        tests = _run_python(sandbox, status_json, conda_root, timeout_s)
    elif lang == "r":
        tests = _run_r(sandbox, status_json, conda_root, timeout_s)
    elif lang == "c":
        tests = _run_c(sandbox, sample_id,
                       _patched_c_files(insert_summary, patched_dir.name,
                                        patched_dir),
                       timeout_s, status_json)
    else:
        tests = [TestResult(name=lang, status="error",
                            log_excerpt=f"unknown language: {lang}")]

    # Annotate the first result with overlay notes
    if tests:
        tests[0].notes = list(tests[0].notes) + overlay_notes

    # Try to recover the canonical function name from the patched-dir's
    # prompt.json (insert.py copies prompt metadata when --out-dir is used,
    # but in our flow it lives next to the infer output). Best-effort.
    function_name = ""
    pj = patched_dir / "prompt.json"
    if pj.is_file():
        try:
            function_name = json.loads(_read(pj)).get("function_name", "") or ""
        except json.JSONDecodeError:
            pass

    res = RunnerResult(
        key=patched_dir.name,
        sample_id=sample_id,
        language=lang,
        mode=mode,
        func_part=func_part,
        function_name=function_name,
        sandbox=str(sandbox),
        tests=tests,
    )
    res.overall = _overall_status(tests)

    if not keep_sandbox:
        shutil.rmtree(sandbox, ignore_errors=True)
        res.sandbox = "(removed)"

    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_insert_summary(patched_root: Path) -> list[dict]:
    p = patched_root / "_summary.json"
    if not p.is_file():
        return []
    try:
        return json.loads(_read(p))
    except json.JSONDecodeError:
        return []
