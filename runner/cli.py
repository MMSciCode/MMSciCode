

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--patched", type=Path, default=None,
                    help="A single intermediate_outputs/patched/<key>/ dir.")
    ap.add_argument("--all", action="store_true",
                    help="Iterate every <key>/ dir under --patched-root.")
    ap.add_argument("--patched-root", type=Path, default=None,
                    help="Root that holds <key>/ dirs from insert.py "
                         "(default: <repo>/intermediate_outputs/patched).")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Auto-detected if omitted (prefers repository data/).")
    ap.add_argument("--sandbox-root", type=Path, default=None,
                    help="Where to materialize per-key run sandboxes "
                         "(default: <repo>/intermediate_outputs/run_sandbox).")
    ap.add_argument("--conda-root", type=Path, default=_DEFAULT_CONDA_ROOT,
                    help="conda installation root (envs/<name>/bin/python).")
    ap.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT_S,
                    help="Per-command timeout in seconds.")
    ap.add_argument("--keep-sandbox", action="store_true",
                    help="Don't delete the sandbox after each run.")
    ap.add_argument("--summary", type=Path, default=None,
                    help="Write aggregate JSON summary here (default: under --sandbox-root).")
    ap.add_argument("--filter", type=str, default=None,
                    help="Only run keys containing this substring.")
    args = ap.parse_args(argv)

    if not args.patched and not args.all:
        ap.error("either --patched or --all is required")

    _root = Path(__file__).resolve().parent
    data_root = _resolve_data_root(args.data_root)
    patched_root = (args.patched_root
                    or _root / "intermediate_outputs" / "patched").resolve()
    sandbox_root = (args.sandbox_root
                    or _root / "intermediate_outputs" / "run_sandbox").resolve()
    sandbox_root.mkdir(parents=True, exist_ok=True)

    insert_summary = _load_insert_summary(patched_root)

    if args.all:
        keys = sorted(p for p in patched_root.iterdir()
                      if p.is_dir() and _parse_key(p.name) is not None)
    else:
        keys = [args.patched.resolve()]
    if args.filter:
        keys = [k for k in keys if args.filter in k.name]
    if not keys:
        print("no patched dirs found", file=sys.stderr)
        return 2

    results: list[RunnerResult] = []
    for k in keys:
        print(f"==> {k.name}", flush=True)
        try:
            res = run_one(
                patched_dir=k,
                data_root=data_root,
                insert_summary=insert_summary,
                sandbox_root=sandbox_root,
                conda_root=args.conda_root,
                timeout_s=args.timeout,
                keep_sandbox=args.keep_sandbox,
            )
        except Exception as e:
            res = RunnerResult(
                key=k.name, sample_id="?", language="?", mode="?",
                sandbox="(none)",
                tests=[TestResult(name="setup", status="error",
                                  log_excerpt=f"{type(e).__name__}: {e}")],
                overall="error",
            )
        results.append(res)
        # one-line per (sample, mode) status
        line = (f"   overall={res.overall}  "
                + "  ".join(
                    f"[{t.name}: {t.status} "
                    + (f"{t.n_passed}/{t.n_tests}" if t.n_tests else "-")
                    + f" {t.duration_s}s]"
                    for t in res.tests))
        print(line, flush=True)

    summary_path = args.summary or (sandbox_root / "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )
    print(f"\nsummary -> {summary_path}", flush=True)

    # Exit non-zero only if every key failed; mixed runs return 0.
    if all(r.overall in ("error", "fail", "compile_fail", "timeout",
                         "env_missing", "env_broken") for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
