

def run_function(
    *, sample_dir: Path, func_index: int, mode: str, out_root: Path,
    model: str, base_url: str, api_key: str,
    temperature: float, max_tokens: int,
    thinking: str, dry_run: bool,
) -> dict:
    """Run inference for ONE (sample, function, mode). The model sees the
    golden source with only this function masked; everything else (sibling
    target functions, helpers, other files) is at golden state."""
    problem: Problem = load_function_problem(sample_dir, func_index)
    func_part = func_part_for(sample_dir, func_index)
    prompt = build_prompt(problem, mode=mode)

    fn = problem.functions[0]
    sub = out_root / f"{problem.language}__{problem.sample_id}__{func_part}__{mode}"
    sub.mkdir(parents=True, exist_ok=True)

    _write_json(sub / "prompt.json", {
        "sample_id": problem.sample_id,
        "language": problem.language,
        "mode": mode,
        "function_index": func_index,
        "function_name": fn.name,
        "func_part": func_part,
        "system": prompt["system"],
        "user": prompt["user"],
        "target_functions": [
            {"name": f.name, "file_path": f.file_path} for f in problem.functions
        ],
    })

    if dry_run:
        print(f"  [dry-run] {sub}  user_chars={len(prompt['user']):,}  fn={fn.name}")
        return {"sample": problem.sample_id, "function": fn.name,
                "function_index": func_index, "func_part": func_part,
                "mode": mode, "dry_run": True}

    print(f"  -> {model} on {problem.sample_id} :: {fn.name} [{mode}]  "
          f"user_chars={len(prompt['user']):,}")
    text, meta = call_model(
        system=prompt["system"], user=prompt["user"],
        model=model, base_url=base_url, api_key=api_key,
        temperature=temperature, max_tokens=max_tokens,
        thinking=thinking,
    )

    (sub / "response.txt").write_text(text, encoding="utf-8")
    if meta.get("reasoning_content"):
        (sub / "reasoning_content.txt").write_text(meta["reasoning_content"], encoding="utf-8")
    meta_to_save = {k: v for k, v in meta.items() if k != "reasoning_content"}
    meta_to_save["reasoning_chars"] = len(meta.get("reasoning_content") or "")
    _write_json(sub / "response_meta.json", meta_to_save)

    fnames = [f.name for f in problem.functions]  # length 1
    extractions = extract_code_blocks(text, problem.language, fnames)

    ext_summary = {}
    for fn_name, info in extractions.items():
        if info.code:
            (sub / f"{fn_name}{_EXT_FOR.get(problem.language, '.txt')}").write_text(
                info.code, encoding="utf-8")
        d = asdict(info)
        d["code_chars"] = len(info.code)
        d.pop("code", None)
        ext_summary[fn_name] = d
    _write_json(sub / "extraction.json", ext_summary)

    n_ok = sum(1 for e in extractions.values() if e.code and e.signature_matches)
    print(f"     done in {meta['duration_s']}s  "
          f"extracted {n_ok}/{len(fnames)} signature-valid")
    return {
        "sample": problem.sample_id, "function": fn.name,
        "function_index": func_index, "func_part": func_part,
        "mode": mode,
        "duration_s": meta["duration_s"],
        "n_funcs": len(fnames), "n_ok": n_ok,
        "out_dir": str(sub),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _collect_all_samples() -> list[Path]:
    root = Path(os.environ.get("MMSCI_DATA_ROOT") or Path(__file__).resolve().parent / "data")
    samples = []
    for lang_dir in ("Python", "C_CPP", "R"):
        nested = root / lang_dir / "data"
        d = nested if nested.is_dir() else root / lang_dir
        if not d.is_dir():
            continue
        for child in sorted(d.iterdir()):
            if child.is_dir():
                samples.append(child)
    return samples


def _main() -> None:
    pipeline_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description="Run model inference on MMSciCode prompts.")
    ap.add_argument("sample_dir", nargs="?",
                    help="One sample directory (e.g. data/Python/037_...)")
    ap.add_argument("--all", action="store_true", help="Run all samples under data/ or MMSCI_DATA_ROOT.")
    ap.add_argument("--mode", choices=["direct", "cot", "both"], default="direct")
    ap.add_argument("--out-dir",
                    default=str(pipeline_dir / "intermediate_outputs" / "infer_preview"),
                    help="Where to write infer outputs. Default chains into "
                         "insert.py's --infer-root so the 3 stages connect "
                         "without arguments.")
    ap.add_argument("--env-file", default=str(pipeline_dir / ".env"))
    ap.add_argument("--model", default=None,
                    help="Override OPENAI_MODEL (default: gpt-5.4)")
    ap.add_argument("--base-url", default=None,
                    help="Override OPENAI_BASE_URL (default: https://api.openai.com/v1)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--thinking", choices=["disabled", "enabled"], default=None,
                    help="Provider-specific thinking mode for compatible endpoints.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build prompts and validate wiring without calling the API.")
    args = ap.parse_args()

    load_dotenv(Path(args.env_file))
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = args.model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    temperature = (args.temperature
                   if args.temperature is not None
                   else float(os.environ.get("OPENAI_TEMPERATURE", "0.0")))
    thinking = (args.thinking
                if args.thinking is not None
                else os.environ.get("OPENAI_THINKING", ""))
    # Thinking mode emits hidden reasoning tokens that count against the
    # response budget; double the default cap so the visible answer isn't
    # starved. An explicit ``--max-tokens`` always wins.
    base_budget = int(os.environ.get("OPENAI_MAX_TOKENS", "8192"))
    default_budget = base_budget * 2 if thinking == "enabled" else base_budget
    max_tokens = (args.max_tokens
                  if args.max_tokens is not None
                  else default_budget)

    if not args.dry_run and not api_key:
        sys.exit(
            "OPENAI_API_KEY is empty.\n"
            f"  1) cp {pipeline_dir / '.env.example'} {pipeline_dir / '.env'}\n"
            "  2) edit .env and set OPENAI_API_KEY"
        )

    if args.all:
        samples = _collect_all_samples()
    else:
        if not args.sample_dir:
            ap.error("sample_dir is required unless --all is given")
        samples = [Path(args.sample_dir).resolve()]

    modes = ["direct", "cot"] if args.mode == "both" else [args.mode]
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"model={model}  thinking={thinking}  base_url={base_url}  out={out_root}")
    print(f"samples={len(samples)}  modes={modes}  dry_run={args.dry_run}")

    summary = []
    for sample in samples:
        try:
            funcs_meta = list_function_specs(sample)
        except Exception as e:
            print(f"  !! {sample.name} skipped: {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue
        print(f"\n==> {sample.name}  ({len(funcs_meta)} target function(s))")
        for func_idx in range(len(funcs_meta)):
            for mode in modes:
                try:
                    r = run_function(
                        sample_dir=sample, func_index=func_idx, mode=mode,
                        out_root=out_root,
                        model=model, base_url=base_url, api_key=api_key,
                        temperature=temperature, max_tokens=max_tokens,
                        thinking=thinking, dry_run=args.dry_run,
                    )
                    summary.append(r)
                except Exception as e:
                    fn_name = (funcs_meta[func_idx].get("function_name", f"#{func_idx}")
                               if 0 <= func_idx < len(funcs_meta) else f"#{func_idx}")
                    print(f"  !! {sample.name} :: {fn_name} [{mode}] failed: "
                          f"{type(e).__name__}: {e}", file=sys.stderr)
                    summary.append({
                        "sample": sample.name, "function": fn_name,
                        "function_index": func_idx, "mode": mode,
                        "error": f"{type(e).__name__}: {e}",
                    })

    _write_json(out_root / "_run_summary.json", summary)
    print(f"\nsummary written to {out_root / '_run_summary.json'}")


if __name__ == "__main__":
    _main()
