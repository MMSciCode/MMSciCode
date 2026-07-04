

# --- per-language slicers (used when one block defines multiple functions) ---

def _slice_python(code: str, name: str) -> str | None:
    lines = code.splitlines()
    pat = re.compile(rf"^(?P<indent>[ \t]*)def\s+{re.escape(name)}\s*\(")
    start = None
    indent = ""
    # capture leading decorators
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if m:
            start = i
            indent = m.group("indent")
            # walk back over `@decorator` lines at the same indent
            while start > 0 and lines[start - 1].lstrip().startswith("@") and \
                    lines[start - 1].startswith(indent):
                start -= 1
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if not ln.strip():
            continue
        # next top-level (same-indent) `def`/`class`/decorator ends our function
        if ln.startswith(indent) and len(ln) - len(ln.lstrip()) == len(indent):
            stripped = ln.lstrip()
            if stripped.startswith(("def ", "class ", "@")):
                end = j; break
        # any line at shallower indent ends our function
        elif not ln.startswith(indent) or (len(ln) - len(ln.lstrip())) < len(indent):
            if ln.lstrip():
                end = j; break
    return "\n".join(lines[start:end]).rstrip() + "\n"


def _slice_braced(code: str, name: str, lang: str) -> str | None:
    """Slice an R or C function definition by walking braces from its header."""
    lines = code.splitlines(keepends=True)
    if lang == "r":
        pat = re.compile(rf"^\s*{re.escape(name)}\s*(<-|=)\s*function\s*\(")
    else:
        pat = re.compile(rf"^[^/].*\b{re.escape(name)}\s*\(")

    start = None
    for i, ln in enumerate(lines):
        if pat.match(ln) and not ln.lstrip().startswith(("//", "*", "#")):
            for look in range(i, min(i + 8, len(lines))):
                if "{" in lines[look]:
                    start = i; break
            if start is not None:
                break
    if start is None:
        return None
    open_line = start
    while open_line < len(lines) and "{" not in lines[open_line]:
        open_line += 1
    depth = 0
    found = False
    end = open_line
    for j in range(open_line, len(lines)):
        for ch in lines[j]:
            if ch == "{":
                depth += 1; found = True
            elif ch == "}":
                depth -= 1
                if found and depth == 0:
                    end = j; break
        if found and depth == 0:
            break
    return "".join(lines[start:end + 1])


def _slice_function(code: str, name: str, lang: str) -> str | None:
    if lang == "python":
        return _slice_python(code, name)
    if lang in ("r", "c"):
        return _slice_braced(code, name, lang)
    return None


def _clean_code(code: str, lang: str, name: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    if code.startswith("﻿"):
        code = code.lstrip("﻿"); notes.append("stripped BOM")
    code = code.strip("\n").rstrip() + "\n"

    if lang == "python":
        try:
            ast.parse(code)
        except SyntaxError as e:
            notes.append(f"python ast.parse failed: {e.msg} (line {e.lineno})")
    elif lang == "c":
        if "{" in code and code.count("{") != code.count("}"):
            notes.append(f"unbalanced braces: {code.count('{')} '{{' vs {code.count('}')} '}}'")
    return code, notes


def extract_code_blocks(
    response_text: str, language: str, func_names: list[str],
) -> dict[str, Extraction]:
    cleaned_text = _strip_reasoning(response_text)
    aliases = _LANG_ALIASES.get(language, {language})

    raw_blocks = []
    for m in _FENCE_RE.finditer(cleaned_text):
        tag_raw = m.group(1) or ""
        body = m.group(2)
        flang, fname = _parse_fence_tag(tag_raw)
        defines = [fn for fn in func_names if _block_defines(body, fn, language)]
        raw_blocks.append({
            "tag": tag_raw, "lang": flang, "named": fname,
            "body": body, "defines": defines,
        })

    out: dict[str, Extraction] = {}

    # Pass 1: explicit `lang:func_name` fence
    for fn in func_names:
        for b in raw_blocks:
            if b["lang"] in aliases and b["named"] == fn:
                code, notes = _clean_code(b["body"], language, fn)
                out[fn] = Extraction(
                    name=fn, language=language, code=code,
                    extraction_method="lang_and_name_tag",
                    signature_matches=_block_defines(code, fn, language),
                    block_tag=b["tag"], notes=notes,
                )
                break

    # Pass 2: any block that defines this function (slice if multi-defined)
    for fn in func_names:
        if fn in out:
            continue
        for b in raw_blocks:
            if fn in b["defines"]:
                if len(b["defines"]) == 1:
                    body = b["body"]; method = "by_definition_whole_block"
                else:
                    sliced = _slice_function(b["body"], fn, language)
                    body = sliced if sliced else b["body"]
                    method = "by_definition_sliced" if sliced else "by_definition_multi_unsliced"
                code, notes = _clean_code(body, language, fn)
                out[fn] = Extraction(
                    name=fn, language=language, code=code,
                    extraction_method=method,
                    signature_matches=_block_defines(code, fn, language),
                    block_tag=b["tag"], notes=notes,
                )
                break

    # Pass 3 (single-target fallback): take the first lang-tagged block
    if len(func_names) == 1 and func_names[0] not in out:
        fn = func_names[0]
        candidates = [b for b in raw_blocks if b["lang"] in aliases or not b["lang"]]
        if candidates:
            b = candidates[0]
            code, notes = _clean_code(b["body"], language, fn)
            out[fn] = Extraction(
                name=fn, language=language, code=code,
                extraction_method="fallback_first_lang_block",
                signature_matches=_block_defines(code, fn, language),
                block_tag=b["tag"], notes=notes + ["no name-tagged or defining block found"],
            )

    # Fill in any still-missing functions as not-found
    for fn in func_names:
        if fn not in out:
            out[fn] = Extraction(
                name=fn, language=language, code="",
                extraction_method="not_found",
                signature_matches=False, block_tag="",
                notes=[
                    f"no fenced block tagged {language}:{fn}, no block defining {fn}, "
                    f"and no usable single-block fallback. Check response.txt manually."
                ],
            )
    return out


# ---------------------------------------------------------------------------
# Per-sample driver
# ---------------------------------------------------------------------------

def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
