

_R_TOPLEVEL_FN = re.compile(
    # iter5e: allow leading whitespace — many R scripts wrap function
    # definitions inside `local({ ... })` or namespace assignments,
    # which indents the def. Without `\s*` head, indented defs were
    # invisible to the truncator (s41467-025-59529-0 had 1700 lines
    # of indented helper definitions skipped).
    r"^\s*([A-Za-z_][A-Za-z_0-9.]*)\s*(?:<-|=)\s*function\s*\(",
    re.MULTILINE,
)


def _smart_truncate_r(masked_src: str, target_fn: str, max_tokens: int) -> str:
    """Brace-balance scan: keep target + direct callees/callers, stub the rest,
    drop top-level driver lines outside any function."""
    if _approx_tokens(masked_src) <= max_tokens:
        return masked_src

    lines = masked_src.splitlines()
    # Find every top-level function definition with its name and span.
    fns: list[dict] = []  # {name, start, end}
    i = 0
    while i < len(lines):
        m = _R_TOPLEVEL_FN.match(lines[i])
        if not m:
            i += 1
            continue
        name = m.group(1)
        # walk forward to find matching `}`
        depth = 0
        seen_open = False
        end = i
        in_str: str | None = None
        for j in range(i, len(lines)):
            s = lines[j]
            k = 0
            while k < len(s):
                ch = s[k]
                nxt = s[k + 1] if k + 1 < len(s) else ""
                if in_str is not None:
                    if ch == "\\" and nxt:
                        k += 2; continue
                    if ch == in_str:
                        in_str = None
                    k += 1; continue
                if ch == "#":
                    break
                if ch in ("'", '"'):
                    in_str = ch; k += 1; continue
                if ch == "{":
                    depth += 1; seen_open = True
                elif ch == "}":
                    depth -= 1
                    if seen_open and depth == 0:
                        end = j
                        break
                k += 1
            if seen_open and depth == 0:
                end = j
                break
        fns.append({"name": name, "start": i, "end": end})
        i = end + 1

    if not fns:
        return masked_src

    target_idx = next((k for k, f in enumerate(fns) if f["name"] == target_fn), None)
    if target_idx is None:
        return masked_src

    target_body = "\n".join(lines[fns[target_idx]["start"]:fns[target_idx]["end"] + 1])
    # Find names called inside target body
    other_names = {f["name"] for f in fns if f["name"] != target_fn}
    callees = {n for n in other_names if re.search(rf"\b{re.escape(n)}\s*\(", target_body)}
    # Find callers (functions referencing target)
    callers: set[str] = set()
    for f in fns:
        if f["name"] == target_fn:
            continue
        body = "\n".join(lines[f["start"]:f["end"] + 1])
        if re.search(rf"\b{re.escape(target_fn)}\s*\(", body):
            callers.add(f["name"])
    keep_full = {target_fn} | callees | callers

    # Reconstruct: keep imports/library calls/source() lines outside functions
    # only if they're short and look like setup; drop anything else outside fns.
    out_parts: list[str] = []
    cursor = 0
    for f in fns:
        # Pre-function setup — keep only library()/source()/require()
        pre = lines[cursor:f["start"]]
        for ln in pre:
            if re.match(r"^\s*(library|require|source|suppressMessages|"
                        r"suppressWarnings)\s*\(", ln):
                out_parts.append(ln)
            elif re.match(r"^\s*#", ln) and len(ln) < 200:
                out_parts.append(ln)
        # Function block
        if f["name"] in keep_full:
            out_parts.append("\n".join(lines[f["start"]:f["end"] + 1]))
        else:
            # 1-line stub
            sig_line = lines[f["start"]]
            # Find header span (until `{`)
            head_end = f["start"]
            while head_end <= f["end"] and "{" not in lines[head_end]:
                head_end += 1
            header = "\n".join(lines[f["start"]:head_end + 1])
            out_parts.append(
                header
                + "\n  # (body omitted to fit token budget)\n}"
            )
        cursor = f["end"] + 1
    # Trailing driver — drop entirely
    out_parts.append(
        "\n# ---- Top-level driver code and unrelated helpers omitted to fit "
        "the token budget ----\n"
    )
    return "\n".join(out_parts)


def _detect_c_functions(src: str) -> list[dict]:
    """Find all C/C++ function definitions in src.

    Handles multi-line arg lists, Class::method names, multi-line return-type
    specifiers. Skips prototypes (ends with `;` instead of `{`).

    Returns list of {name, start (line idx), header_end (line idx of `{`),
                     end (line idx of closing `}`), body_start (line idx after `{`)}
    """
    fns: list[dict] = []
    text = src
    # Use the loose regex — finds opening `(` of a possible function header.
    # We then walk forward to balance parens and find `{` to confirm.
    for m in _C_FUNC_HEADER_LOOSE.finditer(text):
        name = m.group("name")
        leaf = name.split("::")[-1]
        if leaf in _C_KEYWORDS or name in _C_KEYWORDS:
            continue
        # Reject if header begins with a keyword that's a control-flow stmt
        lead_tokens = m.group("lead").split()
        if lead_tokens and lead_tokens[0] in _C_KEYWORDS - {"static", "inline", "extern", "const", "void"}:
            continue
        # Find paren-balanced close after the `(`
        i = m.end()  # right after '('
        depth = 1
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        if depth != 0:
            continue
        # Skip optional whitespace + initializer-list (C++ ctor) + `const`/`override`/etc
        j = i
        # Find next `{` or `;` ignoring strings/comments-ish
        k = j
        while k < len(text):
            ch = text[k]
            if ch == "{":
                # function definition
                break
            if ch == ";":
                # prototype, skip
                k = -1
                break
            k += 1
        if k < 0:
            continue
        if k >= len(text):
            continue
        # Walk braces to find matching `}`
        depth = 1
        body_open = k
        kk = k + 1
        while kk < len(text) and depth > 0:
            ch = text[kk]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            kk += 1
        if depth != 0:
            continue
        body_close = kk - 1
        # Convert byte positions to line indices
        start_line = text.count("\n", 0, m.start())
        body_open_line = text.count("\n", 0, body_open)
        body_close_line = text.count("\n", 0, body_close)
        fns.append({
            "name": name, "leaf": leaf,
            "start": start_line,
            "header_end": body_open_line,
            "end": body_close_line,
            "char_start": m.start(),
            "char_end": body_close + 1,
        })
    return fns
