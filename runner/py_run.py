

def _enable_submission_mode_in_python_tests(code_dir: Path) -> list[str]:
    """Let sandboxed tests assert correctness for patched submissions.

    Many migrated Python tests use TEST_REFERENCE for two different purposes:
    selecting golden imports at module import time, and deciding whether test
    methods should run correctness assertions or expect NotImplementedError.
    Runner keeps TEST_REFERENCE=false so imports resolve to the patched target,
    then this sandbox-only rewrite makes assertion guards also accept the
    separate TEST_SUBMISSION=true flag.
    """
    test_dir = code_dir / "unit_test"
    if not test_dir.is_dir():
        return []

    notes: list[str] = []
    for path in sorted(test_dir.glob("*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines(keepends=True)
        changed = False
        new_lines: list[str] = []
        for i, line in enumerate(lines):
            eol = "\n" if line.endswith("\n") else ""
            body = line[:-1] if eol else line
            m_assign = _PY_TEST_REF_ASSIGN_RE.match(body)
            if m_assign:
                indent, lhs, trailing = m_assign.groups()
                new_lines.append(
                    f"{indent}{lhs}({_PY_SUBMISSION_REF_EXPR}){trailing}{eol}"
                )
                changed = True
                continue

            m_not_assign = _PY_TEST_REF_NOT_ASSIGN_RE.match(body)
            if m_not_assign:
                indent, lhs, trailing = m_not_assign.groups()
                new_lines.append(
                    f"{indent}{lhs}({_PY_SUBMISSION_MASKED_EXPR}){trailing}{eol}"
                )
                changed = True
                continue

            m_if = _PY_TEST_REF_IF_RE.match(body)
            if (m_if
                    and not _python_if_block_imports_reference(
                        [ln[:-1] if ln.endswith("\n") else ln for ln in lines],
                        i,
                        len(m_if.group(1)),
                    )):
                indent, trailing = m_if.groups()
                new_lines.append(
                    f"{indent}if {_PY_SUBMISSION_REF_EXPR}:{trailing}{eol}"
                )
                changed = True
                continue

            m_not_if = _PY_TEST_REF_NOT_IF_RE.match(body)
            if m_not_if:
                indent, trailing = m_not_if.groups()
                new_lines.append(
                    f"{indent}if {_PY_SUBMISSION_MASKED_EXPR}:{trailing}{eol}"
                )
                changed = True
                continue

            new_lines.append(line)

        if changed:
            with path.open("w", encoding="utf-8", newline="\n") as f:
                f.write("".join(new_lines))
            notes.append(
                f"enabled TEST_SUBMISSION assertion mode in "
                f"{path.relative_to(code_dir)}"
            )
    return notes


def _run_python(sandbox: Path, status: dict, conda_root: Path,
                timeout_s: int) -> list[TestResult]:
    code_dir = _find_python_code_dir(sandbox)
    if code_dir is None:
        return [TestResult(name="python", status="error",
                           log_excerpt=f"no unit_test/ dir found under {sandbox}",
                           notes=["expected <sandbox>/code/unit_test/ or "
                                  "<sandbox>/<project>/unit_test/"])]

    modules, dropped = _python_test_modules(status, code_dir)
    if not modules:
        return [TestResult(name="python", status="error",
                           log_excerpt=f"no importable test_*.py / unit_test_*.py modules found under {code_dir}/unit_test/",
                           notes=["status.json listed: " + ", ".join(dropped) if dropped
                                  else "check unit_test_status.json file_structure.test_files"])]
    drop_notes = [f"dropped (file missing): {m}" for m in dropped]

    env_name = ((status.get("environment") or {}).get("conda_env_name")) or ""
    env_path = _conda_env_path(env_name, conda_root) if env_name else None
    if env_name and env_path is None:
        return [TestResult(
            name="python",
            status="env_missing",
            log_excerpt=f"conda env '{env_name}' does not exist under {conda_root}/envs/",
            notes=[f"status.json requested env_name='{env_name}' but it is not provisioned"],
        )]
    if env_path is None:
        py = shutil.which("python3") or shutil.which("python") or sys.executable
        notes = [f"no conda env_name in status.json; using system {py}"]
    else:
        py = Path(str(env_path / "bin" / "python"))
        notes = [f"using conda env '{env_name}' at {env_path}"]
    notes.extend(_enable_submission_mode_in_python_tests(code_dir))

    # GLIBCXX probe (mirrors phase_d_golden.py). If the env's libstdc++.so.6
    # exposes GLIBCXX_3.4.29 but the system one doesn't, prepend env_lib to
    # LD_LIBRARY_PATH so PIL / libLerc loads correctly. (iter5e: previously
    # only phase_d had this; runner.py was hitting the system libstdc++.)
    extra_env = {}
    if env_path is not None:
        env_lib = Path(env_path) / "lib"
        env_libstdcpp = env_lib / "libstdc++.so.6"
        sys_libstdcpp = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
        if env_libstdcpp.is_file() and sys_libstdcpp.is_file():
            try:
                sys_syms = subprocess.check_output(
                    ["strings", str(sys_libstdcpp)], text=True, stderr=subprocess.DEVNULL)
                env_syms = subprocess.check_output(
                    ["strings", str(env_libstdcpp)], text=True, stderr=subprocess.DEVNULL)
                if "GLIBCXX_3.4.29" in env_syms and "GLIBCXX_3.4.29" not in sys_syms:
                    existing_lp = os.environ.get("LD_LIBRARY_PATH", "")
                    extra_env["LD_LIBRARY_PATH"] = str(env_lib) + (
                        (":" + existing_lp) if existing_lp else "")
                    notes.append(f"prepended {env_lib} to LD_LIBRARY_PATH (GLIBCXX_3.4.29)")
            except Exception:
                pass

    cmd = [str(py), "-m", "unittest", *modules, "-v"]
    rc, out, elapsed = _run_cmd(
        cmd,
        cwd=code_dir,
        env={"TEST_REFERENCE": "false", "TEST_GOLDEN": "false",
             "TEST_SUBMISSION": "true", "PYTHONDONTWRITEBYTECODE": "1",
             **extra_env},
        timeout_s=timeout_s,
    )

    n_total, n_pass, n_fail, n_err = _parse_unittest_output(out)

    # ImportError-folding correction: when a test module fails to import,
    # unittest reports the whole module as a single ``_FailedTest`` (n_tests=1
    # for the entire file) regardless of how many ``def test_*`` it actually
    # has. Restore the static method count so the denominator reflects what
    # the test file *intends* to exercise. Otherwise an env fix that lets
    # imports succeed looks like a magical 10× test-count increase.
    extra_total, extra_err = _recover_static_test_count(out, modules, code_dir)
    static_recovered = extra_total > 0
    if static_recovered:
        n_total += extra_total
        n_err += extra_err

    if rc == -1:
        status_str = "timeout"
    elif rc == -2:
        status_str = "error"
    elif rc == 0 and n_total > 0 and n_fail == 0 and n_err == 0:
        status_str = "pass"
    elif _looks_like_env_failure(out):
        # Env-broken runs often surface n_total>0 because unittest reports
        # _FailedTest rows for modules whose import raised ModuleNotFoundError.
        # Catching env_broken before "fail" keeps Phase G's E-class signal
        # honest. See iter2_reviews/pipeline_integrity_review.md L22.
        status_str = "env_broken"
    elif n_total > 0:
        status_str = "fail"
    else:
        status_str = "error"

    extra_notes = [f"cmd: {shlex.join(cmd)}"]
    if static_recovered:
        extra_notes.append(
            f"recovered +{extra_total} test method(s) statically because "
            f"unittest collapsed import-failed module(s) into _FailedTest"
        )

    return [TestResult(
        name="python:" + ",".join(modules),
        status=status_str,
        n_tests=n_total,
        n_passed=n_pass,
        n_failed=n_fail,
        n_errors=n_err,
        duration_s=round(elapsed, 3),
        log_excerpt=_tail(out, 80),
        notes=notes + drop_notes + extra_notes,
    )]
