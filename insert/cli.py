

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(r: RunResult) -> None:
    print(f"\n==> {r.key}")
    for fr in r.files:
        status = "OK" if fr.success else "FAIL"
        print(f"  [{status}] {fr.file_path}  ({', '.join(fr.functions)})")
        if fr.success:
            print(f"      patched -> {fr.output_path}")
            print(f"      masked  -> {fr.masked_path}")
        else:
            print(f"      error  : {fr.error}")
        for n in fr.notes:
            print(f"      note   : {n}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--infer-output", type=Path, default=None,
                    help="Path to a single infer output dir.")
    ap.add_argument("--all", action="store_true",
                    help="Process every <lang>__<sample>__<mode>/ dir under --infer-root.")
    ap.add_argument("--infer-root", type=Path,
                    default=Path(__file__).resolve().parent / "intermediate_outputs" / "infer_preview",
                    help="Where to scan for infer output dirs when --all is given.")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Dataset root containing Python/, C_CPP/, R/ (auto-detected if omitted).")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "intermediate_outputs" / "patched",
                    help="Write patched files to this tree (default: intermediate_outputs/patched). "
                         "Pass --in-place to write back to the dataset instead.")
    ap.add_argument("--in-place", action="store_true",
                    help="Write patched files back into --data-root in-place. "
                         "DANGEROUS: overwrites the dataset; only use intentionally. "
                         "(historical: a v6 run without this flag would write in-place by default and we "
                         "had to restore 54 _masked snapshots.)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and validate, but do not write anything.")
    ap.add_argument("--summary", type=Path, default=None,
                    help="Optional path to write a JSON summary report.")
    args = ap.parse_args(argv)

    if not args.all and args.infer_output is None:
        ap.error("either --infer-output or --all is required")

    data_root = _resolve_data_root(args.data_root)
    # In-place mode is opt-in via explicit flag. Default writes to a side tree
    # so accidental runs never clobber the dataset.
    out_dir = None if args.in_place else args.out_dir.resolve()

    # Collect targets
    if args.all:
        if not args.infer_root.is_dir():
            print(f"error: infer-root not found: {args.infer_root}", file=sys.stderr)
            return 2
        targets = sorted(p for p in args.infer_root.iterdir()
                         if p.is_dir() and _parse_key(p.name) is not None)
    else:
        if not args.infer_output.is_dir():
            print(f"error: infer-output not found: {args.infer_output}", file=sys.stderr)
            return 2
        targets = [args.infer_output.resolve()]

    if not targets:
        print("nothing to do", file=sys.stderr)
        return 1

    print(f"data_root = {data_root}")
    print(f"out_dir   = {out_dir or '(in-place)'}")
    print(f"dry_run   = {args.dry_run}")
    print(f"targets   = {len(targets)}")

    results: list[RunResult] = []
    n_ok_files = 0
    n_fail_files = 0
    n_process_fail = 0
    for t in targets:
        try:
            r = process(t, data_root=data_root, out_dir=out_dir, dry_run=args.dry_run)
        except Exception as e:
            # Bug fix (codex review CRITICAL): process-level failures must NOT
            # disappear from the summary. Track + emit synthetic failed RunResult
            # so Phase G triage sees them as P:insert-process-fail.
            print(f"\n==> {t.name}\n  [FAIL] could not process: {e}", file=sys.stderr)
            n_process_fail += 1
            n_fail_files += 1
            try:
                key_parts = _parse_key(t.name)
                lang = key_parts[0] if key_parts else ""
                sample = key_parts[1] if key_parts else ""
                mode = key_parts[3] if key_parts else ""
            except Exception:
                lang, sample, mode = "", "", ""
            results.append(RunResult(
                key=t.name, sample_id=sample, language=lang, mode=mode,
                files=[FilePatchResult(file_path="", functions=[],
                                       success=False, output_path="", masked_path="",
                                       notes=[],
                                       error=f"process-level exception: {type(e).__name__}: {e}")],
            ))
            continue
        results.append(r)
        _print_result(r)
        for fr in r.files:
            if fr.success:
                n_ok_files += 1
            else:
                n_fail_files += 1

    print(f"\nfiles patched: {n_ok_files} ok, {n_fail_files} failed (process-level fail: {n_process_fail})")

    if args.summary:
        payload = [
            {
                "key": r.key,
                "sample_id": r.sample_id,
                "language": r.language,
                "mode": r.mode,
                "files": [asdict(fr) for fr in r.files],
            }
            for r in results
        ]
        _write(args.summary, json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"summary: {args.summary}")

    return 0 if n_fail_files == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
