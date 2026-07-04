

def load_problem(sample_dir: str | Path,
                 func_index: int | None = None) -> Problem:
    sample_dir = Path(sample_dir).resolve()
    lang = _detect_language(sample_dir)

    core = _load_json(sample_dir / "selected_core_functions.json")
    all_funcs_meta = list(core["selected_core_functions"])
    funcs_meta = all_funcs_meta
    if func_index is not None:
        if not (0 <= func_index < len(all_funcs_meta)):
            raise IndexError(f"func_index {func_index} out of range [0, {len(all_funcs_meta)})")
        funcs_meta = [all_funcs_meta[func_index]]

    # Resolve codebase root:
    # - Python: sample_dir (paths in selected_core_functions.json are "code/...")
    # - C/R: sample_dir (paths are repo-relative inside the sample)
    codebase_root = sample_dir

    # Paper metadata
    paper_title, paper_url, paper_subject = "", "", ""
    if (sample_dir / "paper_info.json").exists():
        p = _load_json(sample_dir / "paper_info.json")
        paper_title = p.get("title", "")
        paper_url = p.get("paper_url", "") or p.get("code_url", "")
    elif (sample_dir / "article_info.json").exists():
        p = _load_json(sample_dir / "article_info.json")
        paper_title = p.get("title", "")
        paper_url = p.get("article_url", "")
        paper_subject = p.get("subject", "")
    elif (sample_dir / "article_metadata.json").exists():
        p = _load_json(sample_dir / "article_metadata.json")
        paper_title = p.get("title", "")
        paper_url = p.get("article_url", "")
        paper_subject = p.get("subject", "")

    # Fallback subject from selected_core_functions.json
    if not paper_subject:
        paper_subject = core.get("project_type", "")

    paper_context = _build_paper_context(sample_dir, lang, funcs_meta)

    # Env info
    env_info: dict[str, Any] = {}
    status_data: dict[str, Any] = {}
    status_path = sample_dir / "unit_test_status.json"
    if status_path.exists():
        status_data = _load_json(status_path)
        env_info = status_data.get("environment", {})

    status_targets = list(status_data.get("target_functions", [])) if status_data.get("schema_version") == 2 else []

    def reference_for(meta: dict, rel_file: str) -> Path | None:
        name = meta.get("function_name", "")
        for target in status_targets:
            if target.get("function_name") != name:
                continue
            target_src = target.get("src_file") or target.get("file_path") or target.get("original_file") or ""
            if target_src and target_src != rel_file and Path(target_src).name != Path(rel_file).name:
                continue
            ref = _explicit_reference_path(sample_dir, target.get("reference_file"))
            if ref is not None:
                return ref
        return None

    # Build FunctionSpec list
    functions: list[FunctionSpec] = []
    for meta in funcs_meta:
        rel_file = meta["file_path"]
        abs_file = sample_dir / rel_file
        if not abs_file.exists():
            # Some datasets record only the basename; resolve by recursive search.
            basename = Path(rel_file).name
            candidates = [p for p in sample_dir.rglob(basename)
                          if "_golden" not in p.name and p.is_file()]
            if not candidates:
                raise FileNotFoundError(abs_file)
            abs_file = candidates[0]
            rel_file = str(abs_file.relative_to(sample_dir))
        reference_abs = reference_for(meta, rel_file)
        masked = _build_masked_file_content(abs_file, meta["function_name"], lang, reference_abs)
        masked = _smart_truncate_file(masked, meta["function_name"], lang,
                                      max_tokens=_MASKED_FILE_TOKEN_BUDGET)
        complete_abs = reference_abs if reference_abs is not None else _golden_path(abs_file)
        masked_full_src = _read_file(complete_abs) if complete_abs is not None else _read_file(abs_file)
        sig = _extract_signature(masked_full_src, meta["function_name"], lang)

        # Cross-cutting: pull return signature from `*_golden` if available, and
        # enumerate the other top-level helpers in the same file.
        target_names = {m["function_name"] for m in funcs_meta}
        siblings = _list_helpers(masked_full_src, target_names, lang)
        returns: list[str] = []
        if complete_abs is not None:
            returns = _extract_returns(_read_file(complete_abs), meta["function_name"], lang)

        if lang == "c":
            extras = _collect_c_headers(sample_dir, rel_file)
        elif lang == "python":
            extras = _collect_python_sibling_helpers(sample_dir, rel_file)
        else:
            extras = _collect_r_sibling_helpers(sample_dir)

        # Cross-file context: when the sample has target functions in OTHER
        # files, include those files at golden state so the model can call
        # them. Without this, multi-file samples lose the cross-file signal
        # the per-sample flow used to provide via concatenated function blocks.
        extras.extend(_collect_other_target_files(
            sample_dir, all_funcs_meta, rel_file, lang,
            current_target_fn=meta["function_name"]
        ))

        # iter5e: enforce per-entry + total cap on the whole extras list.
        # _collect_c_headers / sibling_helpers don't truncate; cross-file
        # entries individually are already capped at _CROSS_FILE_PER_FILE_BUDGET
        # (4k) but C headers can be 5-15k. Hard-cap each + stop accumulating
        # once total > _CROSS_FILE_TOTAL_BUDGET.
        capped_extras: list[dict] = []
        cum = 0
        for ex in extras:
            content = ex.get("content", "")
            if _approx_tokens(content) > _CROSS_FILE_PER_FILE_BUDGET:
                comment = "//" if lang == "c" else "#"
                content = _hard_cap_truncate(content, _CROSS_FILE_PER_FILE_BUDGET, comment)
                ex = {**ex, "content": content}
            this_n = _approx_tokens(content) + 50  # 50 ≈ markdown wrapper overhead
            if cum + this_n > _CROSS_FILE_TOTAL_BUDGET and capped_extras:
                capped_extras.append({
                    "title": "(remaining cross-file entries dropped to fit budget)",
                    "path": "...", "content": "",
                })
                break
            capped_extras.append(ex)
            cum += this_n
        extras = capped_extras

        functions.append(FunctionSpec(
            name=meta["function_name"],
            file_path=rel_file,
            language=lang,
            description=meta.get("description", ""),
            paper_section=meta.get("paper_section", ""),
            article_reference=meta.get("article_reference", ""),
            formula=meta.get("formula", ""),
            key_terms_mapping=meta.get("key_terms_mapping", {}),
            implementation_instruction=meta.get("implementation_instruction", ""),
            masked_file_content=masked,
            signature_block=sig,
            return_signatures=returns,
            siblings_in_file=siblings,
            extra_context=extras,
        ))

    return Problem(
        sample_id=sample_dir.name,
        language=lang,
        codebase_root=codebase_root,
        paper_title=paper_title,
        paper_url=paper_url,
        paper_subject=paper_subject,
        paper_context=paper_context,
        functions=functions,
        env_info=env_info,
    )


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

_LANG_LABEL = {"python": "python", "c": "c", "r": "r"}
_LANG_DISPLAY = {"python": "Python", "c": "C", "r": "R"}

# iter4/iter5d token budgets (Qwen tokenizer; ~4 chars/token approx).
# - _MAX_USER_PROMPT_TOKENS: hard ceiling on the rendered user prompt; the
#   final guard hard-truncates the masked-file block above this.
# - _MASKED_FILE_TOKEN_BUDGET: per-function masked-file budget that the
#   smart truncator targets (target + callers + callees + sig stubs).
# - _PAPER_EXCERPTS_BUDGET: per-function paper excerpts budget. Without this
#   the article_content.json's full Methods + Results dump can hit 36k tokens
#   on its own (60878-z, 46773-z, 57887-3 etc). iter5d cap = 8k.
_MAX_USER_PROMPT_TOKENS = 25_000
_MASKED_FILE_TOKEN_BUDGET = 7_000
_PAPER_EXCERPTS_BUDGET = 8_000
# iter5e: per-section ceilings of 8k each (paper excerpts / masked / cross-file).
# Wrapper overhead may push max to ~8,008 — acceptable.
_CROSS_FILE_PER_FILE_BUDGET = 4_000     # each cross-file entry — headroom for multiple files
_CROSS_FILE_TOTAL_BUDGET = 8_000        # total cross-file section ceiling

SYSTEM_PROMPT = """You are an expert research software engineer and computational \
scientist. You implement functions from peer-reviewed scientific papers with the \
same rigor as the paper's authors: you read the cited passages carefully, map \
the paper's mathematical notation to concrete code, respect the exact function \
signature and return contract, and use only libraries that are already imported \
or part of the standard language runtime. You do not invent behaviour that is \
not justified by the paper or the existing codebase."""


def _format_key_terms(mapping: dict[str, str]) -> str:
    if not mapping:
        return "(none provided)"
    lines = [f"- `{k}` → {v}" for k, v in mapping.items()]
    return "\n".join(lines)


def _format_function_block(fn: FunctionSpec) -> str:
    lang = _LANG_LABEL[fn.language]
    parts = [
        f"### Function: `{fn.name}`",
        f"- **File**: `{fn.file_path}`",
        f"- **Paper section**: {fn.paper_section or '(unspecified)'}",
        "",
        f"**What it does** — {fn.description}",
    ]
    if fn.article_reference:
        parts.append(f"\n**Paper quote** — \"{fn.article_reference}\"")
    if fn.formula:
        parts.append(f"\n**Governing formula** — `{fn.formula}`")
    if fn.key_terms_mapping:
        parts.append(f"\n**Paper symbol ↔ code variable**\n{_format_key_terms(fn.key_terms_mapping)}")
    if fn.implementation_instruction:
        parts.append(f"\n**Implementation instruction** — {fn.implementation_instruction}")
    parts.append(f"\n**Function signature (from the existing codebase)**\n```{lang}\n{fn.signature_block}\n```")

    # Cross-cutting context — only rendered when extractable from the sample.
    if fn.return_signatures:
        bullets = "\n".join(f"- `{r}`" for r in fn.return_signatures)
        parts.append(
            f"\n**Return contract (extracted from the reference implementation)** "
            f"— your function MUST return values in this shape. Match the "
            f"arity, ordering, and types exactly; the unit tests unpack the "
            f"result positionally.\n{bullets}"
        )
    if fn.siblings_in_file:
        bullets = "\n".join(f"- `{s}`" for s in fn.siblings_in_file)
        parts.append(
            f"\n**Other functions defined in this same file** — these are in "
            f"scope and may be called directly. Read their definitions in the "
            f"masked file below before calling them; some may be near-duplicates "
            f"of one another and only one variant is correct for the pinned "
            f"environment.\n{bullets}"
        )

    parts.append(
        f"\n**Relevant source with the target body masked** — this is the file "
        f"you must edit. Independent large blocks may be stubbed (signature only) "
        f"to stay under budget; the target function, its direct callers, and its "
        f"direct callees are shown in full. Replace the masked body and keep the "
        f"surrounding context byte-for-byte where it is shown in full.\n"
        f"```{lang}\n{fn.masked_file_content}\n```"
    )
    if fn.extra_context:
        parts.append("\n**Supporting / cross-file context** — type/struct definitions, helper headers, and other files in this sample that contain target functions (shown at golden state so this function can call them).")
        for ex in fn.extra_context:
            label = ex.get("title") or ex.get("path", "")
            parts.append(f"\n_{label}_\n<file path=\"{ex['path']}\">\n```{lang}\n{ex['content']}\n```\n</file>")
    return "\n".join(parts)
