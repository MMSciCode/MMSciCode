

def _collect_other_target_files(sample_dir: Path,
                                all_funcs_meta: list[dict],
                                current_file_rel: str,
                                lang: str,
                                current_target_fn: str = "",
                                max_tokens_per_file: Optional[int] = None) -> list[dict]:
    """For per-function flow on multi-file samples: gather the reference content
    of every other file that hosts a target function. Without this, the model
    only sees its own file and has no signal that helpers in sibling files
    exist — even though they're at golden state during evaluation, the model
    can't call them if it doesn't know they exist.

    iter5e: each cross-file entry is now also routed through the same
    `_smart_truncate_file` used for the main masked file, anchored at the
    current target's name. This keeps cross-file extras under
    `_CROSS_FILE_PER_FILE_BUDGET` (default 4k tokens) so the whole
    cross-file section can't blow past 8k even when many sibling files exist.
    """
    if max_tokens_per_file is None:
        max_tokens_per_file = _CROSS_FILE_PER_FILE_BUDGET
    out: list[dict] = []
    seen: set[str] = {current_file_rel}
    cumulative = 0
    for meta in all_funcs_meta:
        rel = meta.get("file_path", "")
        if not rel or rel in seen:
            continue
        abs_file = sample_dir / rel
        if not abs_file.is_file():
            basename = Path(rel).name
            candidates = [p for p in sample_dir.rglob(basename)
                          if "_golden" not in p.name and p.is_file()]
            if not candidates:
                continue
            abs_file = candidates[0]
            rel = str(abs_file.relative_to(sample_dir))
            if rel in seen:
                continue
        golden = _golden_path(abs_file)
        src_path = golden if golden is not None else abs_file
        try:
            content = _read_file(src_path)
        except OSError:
            continue
        # iter5e: truncate this cross-file content. We anchor on the CURRENT
        # target function (so callees of the current target stay full); this
        # often stubs the sibling-target functions to one-liners, which is OK
        # — model just needs to know they exist + their signatures.
        if current_target_fn:
            try:
                content = _smart_truncate_file(content, current_target_fn,
                                               lang, max_tokens=max_tokens_per_file)
            except Exception:
                pass
        # Hard cap as backstop: if smart-truncate failed (e.g. unknown lang
        # detector), still chop to the budget so a single 30k-line C++ file
        # doesn't blow up the prompt.
        if _approx_tokens(content) > max_tokens_per_file:
            keep = max_tokens_per_file * 3
            content = content[:keep] + "\n\n/* ... cross-file content truncated to fit budget ... */\n"

        # Stop adding more files once we're past the section budget.
        cumulative += _approx_tokens(content)
        if cumulative > _CROSS_FILE_TOTAL_BUDGET:
            out.append({
                "title": "(remaining cross-file context omitted to fit budget)",
                "path": "...",
                "content": "",
            })
            break
        # Mark contained target functions, since the per-function flow may be
        # generating one of them in a different invocation; here those
        # functions appear at golden state so the current function can call
        # into them safely.
        names = [m["function_name"] for m in all_funcs_meta
                 if m.get("file_path") == meta.get("file_path")]
        title = (f"Other target file (reference, possibly truncated) — contains "
                 f"target function(s): {', '.join(names) if names else '?'}")
        out.append({
            "title": title,
            "path": rel,
            "content": content,
        })
        seen.add(rel)
    return out


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------

def slugify_func_name(name: str) -> str:
    """Filename-safe slug for a function name. Used in per-function output keys."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return s or "fn"


def list_function_specs(sample_dir: str | Path) -> list[dict]:
    """Return the raw selected_core_functions list (with function_name, file_path, ...)."""
    sample_dir = Path(sample_dir).resolve()
    return list(_load_json(sample_dir / "selected_core_functions.json")["selected_core_functions"])


def func_part_for(sample_dir: str | Path, func_index: int) -> str:
    """Compose the `{idx:02d}-{slug}` token used inside the 4-part run key."""
    fns = list_function_specs(sample_dir)
    if not (0 <= func_index < len(fns)):
        raise IndexError(f"func_index {func_index} out of range [0, {len(fns)})")
    return f"{func_index:02d}-{slugify_func_name(fns[func_index]['function_name'])}"


def load_function_problem(sample_dir: str | Path, func_index: int) -> Problem:
    """Per-function variant of load_problem. Returns a Problem with exactly
    ONE FunctionSpec — the func_index'th entry in selected_core_functions.json.

    The masked_file_content for that spec is the GOLDEN file with ONLY that
    function masked (other functions, target or helper, keep golden bodies)."""
    return load_problem(sample_dir, func_index=func_index)
