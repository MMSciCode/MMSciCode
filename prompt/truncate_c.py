

def _smart_truncate_c(masked_src: str, target_fn: str, max_tokens: int) -> str:
    """iter5e C/C++ truncation — drop unrelated functions aggressively.

    Strategy:
      1. Detect every function definition (multi-line aware, C++ Class::method).
      2. Find target. Its leaf (after `::`) is also matched, so a inventory
         entry like `MyClass::method` finds the right def.
      3. Keep target full. Keep direct callers + direct callees of target full.
      4. Stub every other function to a 1-line signature + `/* body omitted */`.
      5. Drop module-level driver between functions (after the FIRST function);
         keep only `#include` / `#define` / `typedef` / `struct` / `enum`
         declarations in between.
      6. Always preserve the file's preamble (everything before the first func).
      7. If the result still exceeds max_tokens, also truncate the target body
         (keep first 60% + last 20%, drop middle) — last-resort budget guard.
    """
    if _approx_tokens(masked_src) <= max_tokens:
        return masked_src

    lines = masked_src.splitlines(keepends=True)
    fns = _detect_c_functions(masked_src)
    if not fns:
        return masked_src

    target_leaf = target_fn.split("::")[-1]
    target = None
    for f in fns:
        if f["name"] == target_fn or f["leaf"] == target_leaf:
            target = f
            break
    if target is None:
        return masked_src

    target_names = {target["name"], target["leaf"]}
    target_body = "".join(lines[target["start"]:target["end"] + 1])

    # Collect callees (names called in target body)
    callees: set[str] = set()
    for f in fns:
        if f is target:
            continue
        if re.search(rf"\b{re.escape(f['leaf'])}\s*\(", target_body):
            callees.add(f["name"])

    # Collect callers (functions whose body calls target leaf)
    callers: set[str] = set()
    for f in fns:
        if f is target:
            continue
        body = "".join(lines[f["start"]:f["end"] + 1])
        if re.search(rf"\b{re.escape(target_leaf)}\s*\(", body):
            callers.add(f["name"])

    keep_full = {target["name"]} | callees | callers

    # Build output: preamble + each function (full or stub) + safe inter-func decls.
    fns_sorted = sorted(fns, key=lambda f: f["start"])
    out: list[str] = []
    # Preamble (before first function)
    out.extend(lines[:fns_sorted[0]["start"]])

    SAFE_PREFIXES = ("#include", "#define", "#if", "#ifdef", "#ifndef", "#else",
                     "#elif", "#endif", "#pragma", "typedef ", "struct ", "enum ",
                     "extern ", "namespace ", "using ", "// ", "/*", "*/", "*")

    for idx, f in enumerate(fns_sorted):
        if f["name"] in keep_full or f["leaf"] in keep_full:
            out.extend(lines[f["start"]:f["end"] + 1])
            out.append("\n")
        else:
            header = "".join(lines[f["start"]:f["header_end"] + 1])
            # Replace what's inside `{...}` with a comment
            header = re.sub(r"\{[^}]*$", "{ /* body omitted */ }",
                           header.rstrip("\n")) + "\n"
            out.append(header)

        # Inter-function block: keep only safe decls
        nxt_start = fns_sorted[idx + 1]["start"] if idx + 1 < len(fns_sorted) else None
        if nxt_start is not None:
            for ln in lines[f["end"] + 1:nxt_start]:
                stripped = ln.lstrip()
                if not stripped or stripped.startswith(SAFE_PREFIXES):
                    out.append(ln)

    out.append("\n/* ---- other helpers omitted to fit token budget ---- */\n")
    result = "".join(out)

    # Last-resort: if still over budget, truncate the target body itself.
    if _approx_tokens(result) > max_tokens:
        # Re-detect target in the new result
        new_fns = _detect_c_functions(result)
        new_target = next((f for f in new_fns
                           if f["name"] == target["name"] or f["leaf"] == target_leaf), None)
        if new_target is not None:
            new_lines = result.splitlines(keepends=True)
            body_start = new_target["header_end"] + 1
            body_end = new_target["end"]
            body_lines = new_lines[body_start:body_end]
            n_body = len(body_lines)
            if n_body > 30:
                keep_head = int(n_body * 0.6)
                keep_tail = int(n_body * 0.2)
                middle_marker = ["    /* ---- middle of target body truncated to fit budget ---- */\n"]
                new_body = body_lines[:keep_head] + middle_marker + body_lines[-keep_tail:]
                new_lines = new_lines[:body_start] + new_body + new_lines[body_end:]
                result = "".join(new_lines)
    return result


def _hard_cap_truncate(text: str, max_tokens: int, comment_prefix: str = "#") -> str:
    """Last-resort hard truncation: keep first 70% + last 20% of text by tokens.

    Used when the language-specific smart truncators couldn't reduce a file
    below max_tokens (e.g. paren-disambig fn names that don't resolve, target
    body that's the whole file, etc.). Better to cut into the middle than
    silently emit a 20k-token block."""
    if _approx_tokens(text) <= max_tokens:
        return text
    tok = _get_qwen_tokenizer()
    if tok is not None:
        ids = tok.encode(text)
        head_n = int(max_tokens * 0.75)
        tail_n = max_tokens - head_n - 20  # leave 20 token for marker
        if tail_n > 0:
            head = tok.decode(ids[:head_n])
            tail = tok.decode(ids[-tail_n:])
            return (head
                    + f"\n\n{comment_prefix} ... ({len(ids) - head_n - tail_n} "
                      f"tokens of unrelated content truncated) ...\n\n"
                    + tail)
    # Fallback: char-based
    keep = max_tokens * 3
    head_chars = int(keep * 0.75)
    tail_chars = keep - head_chars
    return (text[:head_chars]
            + f"\n\n{comment_prefix} ... (middle truncated to fit budget) ...\n\n"
            + text[-tail_chars:])


def _smart_truncate_file(file_src: str, target_fn: str, lang: str,
                         max_tokens: int = 7000) -> str:
    """Reduce ``file_src`` around ``target_fn`` to fit ``max_tokens``.

    Always keeps:
      * the target's signature, masked stub, and docstring (when present)
      * direct callees of the target body (functions referenced inside it)
      * direct callers of the target (functions whose body references it)
    Stubs other functions to a 1-line signature + body-omitted marker.
    Drops module-level driver / orchestration code.
    iter5e: applies a hard char-based truncation as last-resort backstop —
    smart truncators may fail to reduce (e.g. paren-disambig name not found,
    file has zero detectable functions); always returns ≤ max_tokens.

    Falls back to the input unchanged on parser failure or unknown language."""
    try:
        if lang == "python":
            result = _smart_truncate_python(file_src, target_fn, max_tokens)
        elif lang == "r":
            result = _smart_truncate_r(file_src, target_fn, max_tokens)
        elif lang == "c":
            result = _smart_truncate_c(file_src, target_fn, max_tokens)
        else:
            result = file_src
    except Exception:
        # Defensive: never let a truncation bug break prompt generation.
        result = file_src

    # iter5e hard-cap backstop: never return a file > max_tokens.
    if _approx_tokens(result) > max_tokens:
        comment = "//" if lang == "c" else "#"
        result = _hard_cap_truncate(result, max_tokens, comment)
    return result


def _build_masked_file_content(file_abs: Path, func_name: str, lang: str,
                               reference_abs: Path | None = None) -> str:
    """Per-function masked file: read the reference version of `file_abs`, then
    mask ONLY the named function. All other functions in the file (target or
    helper) keep their reference bodies — that's the "complete context" guarantee
    for the per-function evaluation flow.

    Falls back to whichever copy is on disk when no reference/golden sibling
    exists (e.g. helper-only modules)."""
    complete = reference_abs if reference_abs is not None and reference_abs.is_file() else _golden_path(file_abs)
    base = _read_file(complete) if complete is not None else _read_file(file_abs)
    if lang == "python":
        masked = _mask_python_function_body(base, func_name)
    elif lang == "r":
        masked = _mask_r_function_body(base, func_name)
    elif lang == "c":
        masked = _mask_c_function_body(base, func_name)
    else:
        return base
    # Guard: if masking was a no-op but the file actually holds a real body for
    # the target, the masker failed to locate it and would leak the answer into
    # the prompt. Fail loudly instead of silently emitting an unmasked file.
    if masked == base and _target_has_real_def(base, func_name, lang):
        raise RuntimeError(
            f"masking left target {func_name!r} unmasked in "
            f"{complete if complete is not None else file_abs} — refusing to "
            f"emit an unmasked prompt (would leak the reference implementation)")
    return masked


# ---------------------------------------------------------------------------
# Paper-context extraction
# ---------------------------------------------------------------------------

def _article_section(article: dict, title_contains: str) -> str:
    """Flatten a section whose title contains the given substring (case-insensitive)."""
    want = title_contains.lower()
    out: list[str] = []

    def visit(sec: dict):
        if want in sec.get("title", "").lower():
            out.append(f"### {sec['title']}\n{sec.get('content', '')}")
            for ss in sec.get("subsections", []):
                out.append(f"#### {ss['title']}\n{ss.get('content', '')}")
        else:
            for ss in sec.get("subsections", []):
                visit(ss)

    for sec in article.get("sections", []):
        visit(sec)
    return "\n\n".join(s for s in out if s.strip())


def _flatten_article(art: dict) -> list[dict]:
    """Walk the article tree, return a flat list of {title, content, depth, top}.

    `top` is the H1 (top-level section title) the entry belongs to —
    used for "Methods" / "Results" / etc filtering. Subsections inherit
    their parent's top-level title.
    """
    out: list[dict] = []

    def visit(sec: dict, depth: int, top: str):
        title = (sec.get("title") or "").strip()
        content = (sec.get("content") or "").strip()
        if depth == 0:
            top = title
        if content:
            out.append({"title": title, "content": content, "depth": depth, "top": top})
        for ss in sec.get("subsections", []) or []:
            visit(ss, depth + 1, top)

    for sec in art.get("sections", []) or []:
        visit(sec, 0, "")
    return out


def _keywords_from_text(text: str) -> set[str]:
    """Extract simple unigram/bigram keywords for section relevance scoring."""
    words = [
        w for w in re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", text.lower())
        if len(w) >= 4
    ]
    out = set(words)
    for a, b in zip(words, words[1:]):
        if len(a) >= 4 and len(b) >= 4:
            out.add(f"{a} {b}")
    return out
