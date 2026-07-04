# ---------------------------------------------------------------------------
# R runner
# ---------------------------------------------------------------------------

# testthat 3.x summary line e.g. "[ FAIL 0 | WARN 0 | SKIP 0 | PASS 12 ]"
_TESTTHAT_BRACKET_RE = re.compile(
    r"\[\s*FAIL\s+(\d+)\s*\|\s*WARN\s+\d+\s*\|\s*SKIP\s+(\d+)\s*\|\s*PASS\s+(\d+)\s*\]",
)
# testthat 2.x style "OK: 12  Failed: 0  Warnings: 0  Skipped: 0"
_TESTTHAT_LINE_RE = re.compile(
    r"OK:\s*(\d+).*?Failed:\s*(\d+).*?Skipped:\s*(\d+)",
    re.IGNORECASE,
)


def _parse_testthat_output(text: str) -> tuple[int, int, int, int]:
    """Return (n_total, n_passed, n_failed, n_skipped). 0s if unparseable."""
    fails = passes = skipped = 0
    found = False
    for m in _TESTTHAT_BRACKET_RE.finditer(text):
        # last match wins (final summary)
        fails, skipped, passes = int(m.group(1)), int(m.group(2)), int(m.group(3))
        found = True
    if not found:
        m = _TESTTHAT_LINE_RE.search(text)
        if m:
            passes, fails, skipped = int(m.group(1)), int(m.group(2)), int(m.group(3))
            found = True
    if not found:
        return 0, 0, 0, 0
    n_total = passes + fails + skipped
    return n_total, passes, fails, skipped


_R_EXPECT_CALL = re.compile(r"\bexpect_\w+\s*\(")


def _r_static_expect_count(test_dir: Path) -> int:
    """Count ``expect_*(`` call sites across every ``test_*.R`` file.

    testthat's ``[ FAIL n | WARN n | SKIP n | PASS n ]`` line reports only
    the ``expect_*`` calls that were actually executed. When a ``test_that``
    block aborts (e.g. the function under test throws an R error mid-block),
    later ``expect_*`` calls in that block don't fire and don't count — so
    the denominator silently shrinks in proportion to how badly the model
    fails. Using the static call count locks the denominator to the file,
    making pass-rate comparisons across runs and models stable.
    """
    n = 0
    for f in sorted(test_dir.glob("test_*.R")):
        try:
            n += len(_R_EXPECT_CALL.findall(
                f.read_text(encoding="utf-8", errors="replace")
            ))
        except OSError:
            continue
    return n




def _find_r_test_root(sandbox: Path) -> Optional[Path]:
    """Locate the directory whose direct child is ``tests/testthat/`` for
    this sample.

    Two layouts ship in the dataset:
      - sample-rooted:        ``<sandbox>/tests/testthat/``           (some R samples)
      - project-rooted:       ``<sandbox>/<project>/tests/testthat/`` (most Nature Comms R samples)

    R tests in this dataset hardcode ``../../<subdir>/<file>_golden.R``
    relative paths from inside ``tests/testthat/``, so the chosen root is
    used as the Rscript working directory — that makes the relative paths
    resolve identically to the dataset's own ``validate_tests.R``.
    """
    if (sandbox / "tests" / "testthat").is_dir():
        return sandbox
    for sub in sorted(sandbox.iterdir()):
        if sub.is_dir() and (sub / "tests" / "testthat").is_dir():
            return sub
    for tt in sandbox.rglob("testthat"):
        if tt.is_dir() and tt.parent.name == "tests":
            return tt.parent.parent
    return None


def _run_r(sandbox: Path, status: dict, conda_root: Path,
           timeout_s: int) -> list[TestResult]:
    test_dir_rel = "tests/testthat"
    test_root = _find_r_test_root(sandbox)
    if test_root is None:
        return [TestResult(name="r", status="error",
                           log_excerpt=f"no {test_dir_rel}/ dir found under {sandbox}",
                           notes=["expected <sandbox>/tests/testthat/ or "
                                  "<sandbox>/<project>/tests/testthat/"])]
    test_dir = test_root / test_dir_rel

    env_name = ((status.get("environment") or {}).get("conda_env_name")) or ""
    env_path = _conda_env_path(env_name, conda_root) if env_name else None
    if env_name and env_path is None:
        return [TestResult(
            name=f"r:{test_dir_rel}",
            status="env_missing",
            log_excerpt=f"conda env '{env_name}' does not exist under {conda_root}/envs/",
            notes=[f"status.json requested env_name='{env_name}' but it is not provisioned"],
        )]
    if env_path is None:
        rscript = shutil.which("Rscript") or "/usr/bin/Rscript"
        notes = [f"no conda env_name in status.json; using system {rscript}"]
    else:
        rscript = str(env_path / "bin" / "Rscript")
        notes = [f"using conda env '{env_name}' at {env_path}"]
    notes.append(f"r test root: {test_root.relative_to(sandbox) or '.'}")

    # Run from the test root (sample dir for legacy layout, <project>/ for
    # Nature Comms layout) so the test's ``../../<subdir>/<file>_golden.R``
    # relative path resolves identically to the dataset's validate_tests.R.
    # ``progress`` reporter emits the bracket summary line
    # ``[ FAIL N | WARN N | SKIP N | PASS N ]`` that we parse below;
    # ``stop_on_failure=FALSE`` keeps Rscript's exit code at 0 even when tests
    # fail, so we drive status off counts not exit code.
    expr = (f'testthat::test_dir("{test_dir_rel}", '
            f'reporter = "progress", stop_on_failure = FALSE)')
    cmd = [rscript, "-e", expr]
    rc, out, elapsed = _run_cmd(
        cmd,
        cwd=test_root,
        env={"TEST_REFERENCE": "false", "TEST_GOLDEN": "false"},
        timeout_s=timeout_s,
    )

    n_executed, n_pass, n_fail, n_skip = _parse_testthat_output(out)
    n_executed_total = n_pass + n_fail + n_skip

    # When a test_that block aborts mid-block because the function under test
    # throws, testthat reports only the expect_*() calls that already ran in
    # that block. The remaining expect_*() calls — present in the file, just
    # not reached — silently shrink the denominator in proportion to how
    # badly the model fails. To stabilise the denominator across runs and
    # models, fall back to the static count of expect_*() call sites in the
    # test directory; the difference is charged as failed tests (since they
    # were planned but didn't pass).
    n_planned_static = _r_static_expect_count(test_dir)
    static_used = False
    # Only pad the denominator when at least one test_that block FAILED.
    # If testthat reports fail=0, every executed expect_*() passed cleanly
    # and any "unreached" static call sites are most likely conditional
    # branches, helper-internal asserts taken on a different branch, or
    # dead code — NOT mid-block aborts. Padding those into the denominator
    # would penalize a clean pass.
    if (n_fail > 0
            and n_executed_total > 0
            and n_planned_static > n_executed_total):
        unreached = n_planned_static - n_executed_total
        n_fail += unreached
        n_total = n_planned_static
        static_used = True
    else:
        n_total = n_executed_total

    if rc == -1:
        status_str = "timeout"
    elif rc == -2:
        status_str = "error"
    elif n_total == 0:
        # Couldn't parse anything from testthat — most often because R itself
        # crashed before reaching the bracket summary (missing package,
        # invalid lockfile, etc.).
        status_str = "env_broken" if _looks_like_env_failure(out) else "error"
    elif n_fail == 0 and rc == 0:
        status_str = "pass"
    else:
        status_str = "fail"

    extra_notes = [
        f"cmd: {shlex.join(cmd)}",
        f"testthat reports: pass={n_pass} fail={n_executed_total - n_pass - n_skip} "
        f"skip={n_skip} (executed={n_executed_total})",
    ]
    if static_used:
        extra_notes.append(
            f"recovered +{n_planned_static - n_executed_total} unreached expect_*() "
            f"call(s) from static scan; denominator pinned to {n_planned_static}"
        )

    return [TestResult(
        name=f"r:{test_dir_rel}",
        status=status_str,
        n_tests=n_total,
        n_passed=n_pass,
        n_failed=n_fail,
        n_errors=0,
        duration_s=round(elapsed, 3),
        log_excerpt=_tail(out, 80),
        notes=notes + extra_notes,
    )]
