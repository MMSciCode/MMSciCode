

def _add_source_keywords(sample_dir: Path, funcs_meta: list[dict], lang: str,
                         target_keywords: set[str]) -> None:
    """Augment relevance keywords with target signatures and docstrings."""
    for meta in funcs_meta:
        rel = (meta.get("file_path") or meta.get("src_file")
               or meta.get("original_file") or "")
        name = meta.get("function_name") or ""
        if not rel or not name:
            continue
        src = sample_dir / rel
        if not src.is_file():
            basename = Path(rel).name
            matches = [p for p in sample_dir.rglob(basename)
                       if "_golden" not in p.name and "_reference" not in p.name
                       and p.is_file()]
            if not matches:
                continue
            src = matches[0]
        ref = _golden_path(src) or src
        try:
            sig = _extract_signature(_read_file(ref), name, lang)
        except OSError:
            continue
        target_keywords.update(_keywords_from_text(sig))


def _score_relevance(entry: dict, target_keywords: set[str]) -> int:
    """Higher = more relevant to target functions."""
    title_blob = entry["title"].lower()
    body_blob = entry["content"][:1000].lower()
    score = 0
    for kw in target_keywords:
        if not kw:
            continue
        if kw in title_blob:
            score += 4 if " " in kw else 3
        if kw in body_blob:
            score += 2 if " " in kw else 1
    return score


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate `text` so the result is ≤ max_tokens (Qwen-real)."""
    if _approx_tokens(text) <= max_tokens:
        return text
    tok = _get_qwen_tokenizer()
    if tok is not None:
        try:
            ids = tok.encode(text)
            keep = max_tokens - 10  # leave room for marker
            return tok.decode(ids[:keep]) + "\n\n... (section truncated to fit token budget)"
        except Exception:
            pass
    # Fallback char-based — use char/3 to avoid overshoot
    keep = max_tokens * 3
    return text[:keep] + "\n\n... (section truncated to fit token budget)"


def _build_paper_context(sample_dir: Path, lang: str, funcs_meta: list[dict],
                        max_tokens: Optional[int] = None) -> str:
    """Build paper excerpts under a token budget.

    iter5d strategy:
    1. Always include abstract (small, high-signal, lossy to drop).
    2. Build priority tiers from article_content.json:
       Tier 0 — sections matching target fn's `paper_section` field
       Tier 1 — subsections under "Methods" / top-level Methods content
       Tier 2 — subsections under "Results" / top-level Results content
       Tier 3 — everything else (Introduction, Discussion, etc) — only if
                a target's paper_section explicitly named it
    3. Within each tier, score by keyword overlap with target function names
       and `paper_section` values; sort desc.
    4. Add entries until budget exhausted; partial-truncate the last one.
    """
    if max_tokens is None:
        max_tokens = _PAPER_EXCERPTS_BUDGET
    article_json = sample_dir / "article_content.json"
    if not article_json.exists():
        return ""
    art = _load_json(article_json)

    # Build target keywords from function names, metadata, and source
    # signatures/docstrings. The source signal matters for long Methods
    # sections whose subsection titles use implementation terms rather than
    # the exact selected_core_functions label.
    target_keywords: set[str] = set()
    explicit_paper_sections: set[str] = set()
    for f in funcs_meta:
        # function_name → split on _, ., (, ) — drop short tokens
        fn = (f.get("function_name") or "").lower()
        target_keywords.update(_keywords_from_text(fn.replace("_", " ")))
        for field in ("description", "article_reference", "formula",
                      "implementation_instruction"):
            target_keywords.update(_keywords_from_text(f.get(field) or ""))
        # paper_section
        ps = (f.get("paper_section") or "").lower().strip()
        if ps:
            # Take the WHOLE thing as a phrase + the head (before " - ")
            explicit_paper_sections.add(ps)
            target_keywords.update(_keywords_from_text(ps))
            head = ps.split(" - ")[0].split(":")[0].strip()
            if head and head != ps:
                explicit_paper_sections.add(head)
                target_keywords.update(_keywords_from_text(head))
    _add_source_keywords(sample_dir, funcs_meta, lang, target_keywords)

    # Flatten article
    entries = _flatten_article(art)

    # Tier each entry
    def tier_of(e: dict) -> int:
        top_lc = e["top"].lower()
        title_lc = e["title"].lower()
        # Tier 0: explicit paper_section match (title is in or contains a paper_section value)
        for ps in explicit_paper_sections:
            if ps and (ps in title_lc or title_lc in ps):
                return 0
        # Tier 1: under Methods
        if "method" in top_lc or "method" in title_lc:
            return 1
        # Tier 2: under Results
        if "result" in top_lc or "result" in title_lc:
            return 2
        # Tier 3: explicitly mentioned via paper_section (intro/discussion etc)
        for ps in explicit_paper_sections:
            if ps and (ps in top_lc or top_lc in ps):
                return 3
        # Tier 4: dropped (intro / discussion not asked for)
        return 4

    scored = []
    for i, e in enumerate(entries):
        t = tier_of(e)
        if t >= 4:
            continue
        rel = _score_relevance(e, target_keywords)
        # Sort key: tier asc, relevance desc, depth asc (parent before child),
        # original position asc (preserve paper order within ties).
        scored.append(((t, -rel, e["depth"], i), e))
    scored.sort(key=lambda x: x[0])
    if not scored and entries:
        # Some Nature "Matters arising" articles and parser fallbacks have no
        # explicit Methods/Results headings. Keep the available article text
        # rather than returning an empty Paper excerpts block.
        scored = [
            ((3, -_score_relevance(e, target_keywords), e["depth"], i), e)
            for i, e in enumerate(entries)
        ]
        scored.sort(key=lambda x: x[0])

    parts: list[str] = []
    used = 0
    stub_budget = min(400, max_tokens // 20) if scored else 0
    content_budget = max_tokens - stub_budget
    omitted: list[dict] = []
    included_titles: set[tuple[str, str]] = set()

    # Abstract first
    abstract = (art.get("abstract") or "").strip()
    if abstract:
        ab_block = f"### Abstract\n{abstract}"
        ab_tok = _approx_tokens(ab_block)
        if ab_tok < content_budget:
            parts.append(ab_block)
            used += ab_tok

    # Add ranked entries until budget hit
    for idx, (_key, e) in enumerate(scored):
        if used >= content_budget:
            omitted.extend(x[1] for x in scored[idx:])
            break
        title = e["title"]
        content = e["content"]
        prefix = "###" if e["depth"] == 0 else "####"
        block = f"{prefix} {title}\n{content}"
        block_tok = _approx_tokens(block)
        if used + block_tok <= content_budget:
            parts.append(block)
            used += block_tok
            included_titles.add((e["top"], title))
        else:
            remaining = content_budget - used
            if remaining > 200:  # only worth partial-include if at least 200 tokens left
                truncated = _truncate_to_tokens(block, remaining)
                parts.append(truncated)
                used += _approx_tokens(truncated)
                included_titles.add((e["top"], title))
            omitted.extend(x[1] for x in scored[idx + 1:])
            break

    # Preserve a lightweight map of omitted Methods/Results subsections. This
    # avoids the old "silent cliff" where a 30k-token Methods section was cut
    # to 8k with no hint that other named algorithm sections existed.
    seen_stub_titles: set[tuple[str, str]] = set()
    for e in omitted:
        key = (e["top"], e["title"])
        if key in included_titles or key in seen_stub_titles:
            continue
        title_lc = (e["top"] + " " + e["title"]).lower()
        if not ("method" in title_lc or "result" in title_lc):
            continue
        prefix = "###" if e["depth"] == 0 else "####"
        stub = f"{prefix} {e['title']} (omitted, low relevance or over budget)"
        stub_tok = _approx_tokens(stub)
        if used + stub_tok > max_tokens:
            break
        parts.append(stub)
        used += stub_tok
        seen_stub_titles.add(key)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Per-language extra context (e.g. C struct definitions)
# ---------------------------------------------------------------------------

_C_HEADER_MAX = 8


def _collect_c_headers(sample_dir: Path, target_file_rel: str = "") -> list[dict]:
    """Return header files likely needed to understand the target's types:
    the headers sitting in the target source file's own directory (and a
    neighbouring ``include/`` if present). Generalizes across samples instead
    of matching a fixed set of header names. Downstream budget logic trims the
    result further, so we cap only the count here."""
    out: list[dict] = []
    seen: set[Path] = set()
    dirs: list[Path] = []
    if target_file_rel:
        tdir = (sample_dir / target_file_rel).parent
        dirs += [tdir, tdir / "include", tdir.parent / "include"]
    for d in dirs:
        if not d.is_dir():
            continue
        headers = (sorted(d.glob("*.h")) + sorted(d.glob("*.hpp"))
                   + sorted(d.glob("*.hh")) + sorted(d.glob("*.hxx")))
        for p in headers:
            if p in seen or not p.is_file():
                continue
            seen.add(p)
            try:
                txt = _read_file(p)
            except OSError:
                continue
            out.append({
                "title": p.name,
                "path": str(p.relative_to(sample_dir)),
                "content": txt,
            })
            if len(out) >= _C_HEADER_MAX:
                return out
    return out


def _collect_python_sibling_helpers(sample_dir: Path,
                                    target_file_rel: str) -> list[dict]:
    """Include same-directory helpers the target function may call."""
    out = []
    target_abs = sample_dir / target_file_rel
    target_dir = target_abs.parent
    # Just include the target file's imports / module-level helpers via the
    # whole file — the caller already has the masked file. Nothing extra needed.
    _ = target_dir  # placeholder for future extension
    return out


def _collect_r_sibling_helpers(sample_dir: Path) -> list[dict]:
    return []
