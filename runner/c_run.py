# ---------------------------------------------------------------------------
# C runner
#
# For each sample this runs, in order:
#
#   1) a syntax check of each patched source file with the right compiler
#      (gcc for C, g++ for C++) and standard/macros derived from the sample's
#      CMakeLists; missing-header/lib failures are demoted to env_broken.
#   2) the ``realtest_*`` targets declared by the sample's CMakeLists, when
#      present.
#   3) optional: run the pre-built simple_test_masked binary (informational).
# ---------------------------------------------------------------------------

# The raw syntax-check flags (compiler standard + -D macros) are derived from
# the sample's own unit_test/CMakeLists.txt rather than hardcoded per sample,
# so #ifdef/#error-guarded sources (e.g. dimension macros) compile for every
# sample, not just one.
_CPP_SUFFIXES = (".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx")
_DEF_CALL_RE = re.compile(
    r"(add_definitions|add_compile_definitions|target_compile_definitions)\s*\(([^)]*)\)",
    re.IGNORECASE)
_SET_RE = re.compile(r"set\s*\(\s*([A-Za-z_]\w*)\s+([^)]*)\)", re.IGNORECASE)
_STD_C_RE = re.compile(r"set\s*\(\s*CMAKE_C_STANDARD\s+(\d+)", re.IGNORECASE)
_STD_CXX_RE = re.compile(r"set\s*\(\s*CMAKE_CXX_STANDARD\s+(\d+)", re.IGNORECASE)
_MACRO_TOKEN_RE = re.compile(r"^(?:-D)?([A-Za-z_]\w*(?:=[^\s]+)?)$")

# C/C++ compile errors that mean "build env is missing deps" rather than
# "the model's code is wrong" — demoted to env_broken so they don't pollute
# model-quality stats (mirrors the py/r `_looks_like_env_failure` handling).
_C_ENV_PATTERNS = (
    re.compile(r"fatal error:\s*\S+:\s*No such file or directory"),
    re.compile(r"\bcannot find -l\S+"),
)


def _cmake_text(sandbox: Path) -> str:
    parts = []
    for cml in ((sandbox / "unit_test" / "CMakeLists.txt"), (sandbox / "CMakeLists.txt")):
        if cml.is_file():
            try:
                parts.append(cml.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    return "\n".join(parts)


def _c_macro_flags(sandbox: Path) -> list[str]:
    """Union of -D macros declared in the sample's CMakeLists (add_definitions /
    *_compile_definitions), resolving set() variables like ${REAL_LIB_DEFS}."""
    text = _cmake_text(sandbox)
    if not text.strip():
        return []
    var_map = {m.group(1): m.group(2).split() for m in _SET_RE.finditer(text)}

    def resolve(tokens):
        out = []
        for t in tokens:
            mv = re.fullmatch(r"\$\{([A-Za-z_]\w*)\}", t)
            out.extend(var_map.get(mv.group(1), []) if mv else [t])
        return out

    flags, seen = [], set()
    for m in _DEF_CALL_RE.finditer(text):
        call, toks = m.group(1).lower(), m.group(2).split()
        if call == "target_compile_definitions" and toks:
            toks = toks[1:]  # first token is the target name
        for t in resolve(toks):
            if t.upper() in ("PRIVATE", "PUBLIC", "INTERFACE") or t.startswith("${"):
                continue
            mt = _MACRO_TOKEN_RE.match(t)
            if mt:
                flag = f"-D{mt.group(1)}"
                if flag not in seen:
                    seen.add(flag)
                    flags.append(flag)
    return flags


def _is_cpp_source(src: Path, status: dict) -> bool:
    s = src.suffix.lower()
    if s in _CPP_SUFFIXES:
        return True
    if s == ".h":  # ambiguous — treat as C++ when the sample is a C++ project
        env = status.get("environment") or {}
        cstd = (env.get("c_standard") or "").lower()
        return bool(env.get("cpp_standard")) and cstd in ("", "n/a", "na")
    return False


def _c_std_flags(src: Path, status: dict, cmake_text: str) -> list[str]:
    env = status.get("environment") or {}
    if _is_cpp_source(src, status):
        m = _STD_CXX_RE.search(cmake_text)
        std = f"c++{m.group(1)}" if m else (env.get("cpp_standard") or "").lower().replace(" ", "")
        std = std if std.startswith("c++") else "c++17"
        return [f"-std={std}", "-x", "c++"]
    m = _STD_C_RE.search(cmake_text)
    if m:
        std = f"c{m.group(1)}"
    else:
        cstd = (env.get("c_standard") or "").lower().replace(" ", "")
        std = cstd if re.fullmatch(r"(c|gnu)\d+", cstd or "") else "c99"
    return [f"-std={std}"]


def _looks_like_c_env_failure(log: str) -> bool:
    return any(p.search(log) for p in _C_ENV_PATTERNS)


def _run_c(sandbox: Path, sample_id: str, patched_files: list[str],
           timeout_s: int, status: dict | None = None) -> list[TestResult]:
    results: list[TestResult] = []
    status = status or {}
    cmake_text = _cmake_text(sandbox)
    macro_flags = _c_macro_flags(sandbox)

    if not patched_files:
        return [TestResult(name="c:compile", status="error",
                           log_excerpt="no patched C/C++ files reported by insert.py",
                           notes=["expected patched_files from _summary.json"])]

    # 1) Syntax-check each patched source with the right compiler + standard.
    gcc = shutil.which("gcc") or "/usr/bin/gcc"
    gpp = shutil.which("g++") or "/usr/bin/g++"
    n_ok = n_fail = n_env = 0
    excerpts: list[str] = []
    notes: list[str] = []
    t0 = time.monotonic()
    for rel in patched_files:
        src = sandbox / rel
        if not src.is_file():
            n_fail += 1
            excerpts.append(f"[missing] {rel}")
            continue
        compiler = gpp if _is_cpp_source(src, status) else gcc
        inc = src.parent
        cmd = [compiler, *_c_std_flags(src, status, cmake_text),
               "-fsyntax-only", "-c", str(src), "-I", str(inc), *macro_flags]
        rc, out, _ = _run_cmd(cmd, cwd=sandbox, timeout_s=timeout_s)
        if rc == 0:
            n_ok += 1
        elif _looks_like_c_env_failure(out):
            n_env += 1
            excerpts.append(f"[env rc={rc}] {rel}\n{_tail(out, 20)}")
        else:
            n_fail += 1
            excerpts.append(f"[fail rc={rc}] {rel}\n{_tail(out, 30)}")
    elapsed = time.monotonic() - t0

    if n_fail > 0:
        status_str = "compile_fail"
    elif n_env > 0:
        status_str = "env_broken"  # couldn't validate the model due to env deps
    else:
        status_str = "compile_ok"

    if n_env:
        notes.append(f"{n_env} file(s) failed on missing headers/libs "
                     "(counted as env, not model error)")
    results.append(TestResult(
        name="c:compile-syntax",
        status=status_str,
        n_tests=len(patched_files),
        n_passed=n_ok,
        n_failed=n_fail,
        n_errors=n_env,
        duration_s=round(elapsed, 3),
        log_excerpt=_tail("\n".join(excerpts) or "all files compiled cleanly", 80),
        notes=notes + [f"macros: {' '.join(macro_flags) or '(none)'}",
                       f"files: {patched_files}"],
    ))

    # 2) Informational: run pre-built simple_test_masked from the dataset
    #    if it exists. This does NOT exercise model output; it confirms the
    #    bundled test harness still passes its own pre-built golden vs masked.
    pre = sandbox / "unit_test"
    bin_masked = pre / "simple_test_masked"
    if bin_masked.is_file() and os.access(bin_masked, os.X_OK):
        rc, out, elapsed_b = _run_cmd([str(bin_masked)], cwd=pre, timeout_s=timeout_s)
        results.append(TestResult(
            name="c:bundled simple_test_masked",
            status="pass" if rc == 0 else "fail",
            n_tests=1,
            n_passed=1 if rc == 0 else 0,
            n_failed=0 if rc == 0 else 1,
            duration_s=round(elapsed_b, 3),
            log_excerpt=_tail(out, 60),
            notes=["informational: bundled harness, doesn't link to model output"],
        ))

    # 3) Run any realtest_* targets the sample's CMakeLists declares.
    results.extend(_run_c_realtests(pre, timeout_s))

    return results


_REAL_RESULT_RE = re.compile(r"Results:\s+(\d+)/(\d+)\s+passed")




def _run_c_realtests(unit_test_dir: Path, timeout_s: int) -> list[TestResult]:
    """Build (via cmake) and run any ``realtest_*`` targets defined in
    ``CMakeLists.txt``. Returns one TestResult per realtest binary.

    Assumes the sample's ``unit_test/CMakeLists.txt`` declares
    ``add_executable(realtest_<name> ...)``. If none are declared, returns
    an empty list (the sample declares no realtest_* targets).
    """
    cml = unit_test_dir / "CMakeLists.txt"
    if not cml.is_file():
        return []

    # Discover realtest_* target names from CMakeLists.txt.
    text = cml.read_text(encoding="utf-8", errors="replace")
    targets = sorted(set(re.findall(
        r"add_executable\(\s*(realtest_[A-Za-z0-9_]+)", text
    )))
    if not targets:
        return []

    # Out-of-tree build inside the sandbox. Wipe any stale source-dir
    # CMakeCache.txt that the dataset may ship with — those carry
    # absolute paths from the dataset's prep machine and break cmake.
    for stale in (unit_test_dir / "CMakeCache.txt",
                  unit_test_dir / "CMakeFiles"):
        if stale.is_file():
            stale.unlink()
        elif stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)

    build = unit_test_dir / "_build_real"
    if build.exists():
        shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True, exist_ok=True)

    rc, out_cm, _ = _run_cmd(["cmake", ".."], cwd=build, timeout_s=timeout_s)
    if rc != 0:
        return [TestResult(
            name=f"c:realtest cmake",
            status="error",
            log_excerpt=_tail(out_cm, 40),
            notes=["cmake failed; cannot run realtest_* binaries"],
        )]

    results: list[TestResult] = []
    for tgt in targets:
        rc, out_mk, _ = _run_cmd(["make", tgt], cwd=build, timeout_s=timeout_s)
        if rc != 0:
            results.append(TestResult(
                name=f"c:realtest {tgt} (build)",
                status="compile_fail",
                log_excerpt=_tail(out_mk, 40),
                notes=["model output broke the build of " + tgt],
            ))
            continue
        binpath = build / tgt
        rc_run, out_run, elapsed = _run_cmd([str(binpath)], cwd=build,
                                            timeout_s=timeout_s)
        m = _REAL_RESULT_RE.search(out_run)
        if m:
            n_pass = int(m.group(1))
            n_total = int(m.group(2))
            n_fail = n_total - n_pass
        else:
            n_pass = 0
            n_total = 0
            n_fail = 0
        # Distinguish exit modes:
        #   rc==0  — clean exit; trust the parsed counts
        #   rc==-1 — our timeout sentinel (set by _run_cmd on TimeoutExpired)
        #   rc<0   — terminated by signal (Python convention: -11 = SIGSEGV)
        #   rc>0   — non-zero clean exit (main returned 1 because some test failed)
        signaled = rc_run is not None and rc_run < 0 and rc_run != -1
        extra_notes = [f"realtest target: {tgt}"]
        if rc_run == -1:
            status_str = "timeout"
        elif signaled:
            # The patched function crashed before main could print results.
            # Treat as max-fail: no test passed, charge all planned tests.
            status_str = "fail"
            n_total = max(n_total, 1)  # ensure denominator non-zero so report shows 0/1
            n_pass = 0
            n_fail = n_total
            sig = -rc_run
            sig_name = {11: "SIGSEGV", 6: "SIGABRT", 8: "SIGFPE",
                        9: "SIGKILL", 13: "SIGPIPE", 14: "SIGALRM"}.get(
                            sig, f"signal {sig}")
            extra_notes.append(f"binary terminated by {sig_name} (rc={rc_run})")
        elif n_total == 0:
            status_str = "error"
            extra_notes.append("no Results: line in stdout")
        elif n_fail == 0 and rc_run == 0:
            status_str = "pass"
        else:
            status_str = "fail"
        results.append(TestResult(
            name=f"c:realtest {tgt}",
            status=status_str,
            n_tests=n_total,
            n_passed=n_pass,
            n_failed=n_fail,
            duration_s=round(elapsed, 3),
            log_excerpt=_tail(out_run, 30),
            notes=extra_notes,
        ))
    return results
