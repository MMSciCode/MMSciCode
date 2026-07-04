

_DIRECT_INSTRUCTIONS = """## Your task

Produce the **complete implementation of every function listed above**. Follow \
these rules:

1. Match the declared signature exactly (parameter names, default values, \
   return type/shape).
2. Use only symbols that are already imported in the masked file, plus the \
   standard library of the target language. Do not add new top-level imports \
   unless strictly necessary — if you must, justify it in a comment.
3. Honor the paper's formulas and the provided symbol mapping. Prefer \
   numerically stable, vectorised forms when idiomatic for the language.
4. Preserve any side effects on struct/object fields that the surrounding code \
   relies on (e.g. setting flags like `isForceValid = false`).
5. Do **not** print, log, or add test scaffolding.

## Output format

Return one fenced code block per target function, tagged with the language and \
the function name, e.g.:

```{lang}:{example_name}
<full function definition, drop-in replacement for the masked version>
```

Only output the code blocks — no prose, no explanation, no extra text before or \
after."""


_COT_INSTRUCTIONS = """## Your task

Produce the **complete implementation of every function listed above**, but \
first reason through it.

### Phase 1 — Reasoning (write inside `<reasoning>` tags)

For each target function, work through:

1. **Restate the algorithm** in plain English, grounded in the paper quote and \
   formula above.
2. **Symbol map**: enumerate every symbol in the formula and state the \
   corresponding variable name and type in the code.
3. **Input / output contract**: declared parameter types, return type, array \
   shape(s), and any side effects on shared structs/objects.
4. **Edge cases**: boundary inputs (empty / NaN / insufficient data), numerical \
   stability, and what the paper implies should happen.
5. **Execution plan**: a short numbered list of the steps the body will take.

### Phase 2 — Implementation

Return one fenced code block per target function, tagged with the language and \
the function name, e.g.:·

```{lang}:{example_name}
<full function definition, drop-in replacement for the masked version>
```

### Rules

1. Match the declared signature exactly (parameter names, default values, \
   return type/shape).
2. Use only symbols that are already imported in the masked file, plus the \
   standard library of the target language. Do not add new top-level imports \
   unless strictly necessary — if you must, justify it in a comment.
3. Honor the paper's formulas and the symbol mapping.
4. Preserve any side effects on struct/object fields that the surrounding code \
   relies on.
5. Do **not** print, log, or add test scaffolding.
6. After the reasoning block, output only the code blocks — no further prose."""


def _enforce_prompt_budget(user_prompt: str,
                           max_tokens: int = _MAX_USER_PROMPT_TOKENS) -> str:
    """Hard-cap the user prompt at ``max_tokens``.

    Strategy: while over budget, truncate the largest fenced code block
    (typically the masked-file block) and append an explicit `... (truncated to
    fit token budget)` marker so the model knows content was elided. Iterates
    until the whole prompt is under budget (so overflow that no single block can
    absorb is still removed), then applies a character-level backstop. Char/token
    conversions use the same ~3 chars/token ratio as ``_approx_tokens`` so the
    amount removed matches the amount over budget. This is a final backstop
    after `_smart_truncate_file`."""
    pattern = re.compile(r"(```[a-zA-Z_]*\n)(.*?)(\n```)", re.DOTALL)
    _MARK = "\n\n# ... (block truncated to fit token budget)"
    for _ in range(32):  # bounded; each pass shrinks the largest remaining block
        if _approx_tokens(user_prompt) <= max_tokens:
            return user_prompt
        matches = list(pattern.finditer(user_prompt))
        if not matches:
            break
        matches.sort(key=lambda m: len(m.group(2)), reverse=True)
        biggest = matches[0]
        body = biggest.group(2)
        if len(body) <= 200:
            break  # largest block already minimal — blocks can't help further
        over_tokens = _approx_tokens(user_prompt) - max_tokens
        # ~3 chars/token (matches _approx_tokens) + 10% margin so we cross under.
        remove = min(len(body) - 200, int(over_tokens * 3 * 1.1) + 1)
        new_body = body[: len(body) - remove] + _MARK
        new_block = biggest.group(1) + new_body + biggest.group(3)
        user_prompt = (user_prompt[: biggest.start()] + new_block
                       + user_prompt[biggest.end():])
    # Character-level backstop if still over (e.g. no/again-minimal blocks).
    if _approx_tokens(user_prompt) > max_tokens:
        max_chars = max_tokens * 3
        if len(user_prompt) > max_chars:
            user_prompt = (user_prompt[:max_chars]
                           + "\n\n... (prompt truncated to fit token budget)")
    return user_prompt


def build_prompt(problem: Problem, mode: str = "direct") -> dict[str, str]:
    """Render a prompt dict {system, user} for the given problem and mode."""
    mode = mode.lower()
    if mode not in {"direct", "cot"}:
        raise ValueError(f"mode must be 'direct' or 'cot', got {mode!r}")

    lang = _LANG_LABEL[problem.language]
    header = [
        f"# Paper-Grounded Function Implementation ({_LANG_DISPLAY[problem.language]})",
        "",
        "You are given (1) a function-implementation task extracted from a "
        "peer-reviewed scientific paper, (2) the relevant paper excerpts, and "
        "(3) the original source file with the target function's body masked. "
        "Your job is to fill in the masked body so that the function realises "
        "the algorithm described in the paper.",
        "",
        "## Paper",
        f"- **Title**: {problem.paper_title}",
    ]
    if problem.paper_subject:
        header.append(f"- **Subject / domain**: {problem.paper_subject}")
    if problem.paper_url:
        header.append(f"- **URL**: {problem.paper_url}")
    header.append(f"- **Language**: {_LANG_DISPLAY[problem.language]}")

    if problem.env_info:
        header.append(f"- **Execution environment**: {json.dumps(problem.env_info, ensure_ascii=False)}")

    sections = ["\n".join(header)]

    if problem.paper_context.strip():
        sections.append("## Paper excerpts\n\n" + problem.paper_context.strip())

    sections.append("## Target function(s)\n\n" +
                    "\n\n---\n\n".join(_format_function_block(f) for f in problem.functions))

    example_name = problem.functions[0].name if problem.functions else "func"
    instr = _COT_INSTRUCTIONS if mode == "cot" else _DIRECT_INSTRUCTIONS
    instr = instr.replace("{lang}", lang).replace("{example_name}", example_name)
    sections.append(instr)

    user_prompt = "\n\n".join(sections)
    user_prompt = _enforce_prompt_budget(user_prompt,
                                         max_tokens=_MAX_USER_PROMPT_TOKENS)
    return {"system": SYSTEM_PROMPT, "user": user_prompt}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _problem_to_public_dict(p: Problem) -> dict[str, Any]:
    d = asdict(p)
    d["codebase_root"] = str(p.codebase_root)
    return d


def _main():
    ap = argparse.ArgumentParser(description="Build MMSciCode prompts.")
    ap.add_argument("sample_dir", nargs="?",
                    help="Path to a single sample directory (e.g. data/R/s41467-024-45976-8)")
    ap.add_argument("--all", action="store_true",
                    help="Render all samples under data/ or MMSCI_DATA_ROOT.")
    ap.add_argument("--mode", choices=["direct", "cot", "both"], default="both")
    ap.add_argument("--out", type=str, default=None,
                    help="Output file (single sample) or directory (with --all)")
    ap.add_argument("--show", action="store_true",
                    help="Print the rendered user prompt to stdout")
    args = ap.parse_args()

    if args.all:
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
    else:
        if not args.sample_dir:
            ap.error("sample_dir is required unless --all is given")
        samples = [Path(args.sample_dir).resolve()]

    modes = ["direct", "cot"] if args.mode == "both" else [args.mode]
    out_root = Path(args.out).resolve() if args.out else None
    if args.all and out_root:
        out_root.mkdir(parents=True, exist_ok=True)

    if out_root and not args.all:
        # Per-function flow always writes one file per (sample × func × mode);
        # treat --out as a directory.
        out_root.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        funcs_meta = list_function_specs(sample)
        for func_idx in range(len(funcs_meta)):
            problem = load_function_problem(sample, func_idx)
            func_part = func_part_for(sample, func_idx)
            fn_name = problem.functions[0].name
            for mode in modes:
                prompt = build_prompt(problem, mode=mode)
                bundle = {
                    "sample_id": problem.sample_id,
                    "language": problem.language,
                    "mode": mode,
                    "function_index": func_idx,
                    "function_name": fn_name,
                    "func_part": func_part,
                    "system": prompt["system"],
                    "user": prompt["user"],
                    "target_functions": [
                        {"name": f.name, "file_path": f.file_path}
                        for f in problem.functions
                    ],
                }

                if args.show:
                    hdr = f"{problem.sample_id} :: {fn_name} [{mode}]"
                    print(f"\n{'=' * 72}\n{hdr}\n{'=' * 72}")
                    print(prompt["user"])

                if out_root:
                    fname = (f"{problem.language}__{problem.sample_id}"
                             f"__{func_part}__{mode}.json")
                    path = out_root / fname
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("w", encoding="utf-8") as f:
                        json.dump(bundle, f, ensure_ascii=False, indent=2)
                    print(f"wrote {path}")


if __name__ == "__main__":
    _main()
