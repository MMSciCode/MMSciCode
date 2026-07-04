

def _locate_c_start(lines: list[str], name: str) -> Optional[int]:
    """Find the start line of C/C++ function `name`.

    Tries several name forms in priority order to handle real-world C++ patterns:
      1. `name` as written (matches `void Class::method(` on one line)
      2. `Class__DOUBLE_COLON__method` substitution (some masking tools rewrite `::`)
      3. Bare method name (last segment after `::`) — for in-class definitions
         and the `void Class::\\n method(...)` multi-line style; requires that
         a preceding line contain `Class::` to confirm we matched the right one.

    After locating the function-name line, walks back over preamble lines
    (return type, template params, `Class::` qualifier when on its own line)
    so the splice replaces the full declaration.
    """
    # Each candidate is (label_for_call_check, compiled_regex). label is the
    # bare-name string used by the call-skip heuristic; if no fixed name applies
    # (e.g. template-class fallback), pass empty string to disable that check.
    candidates: list[tuple[str, re.Pattern, bool]] = []
    candidates.append((name, re.compile(rf"\b{re.escape(name)}\s*\("), False))
    if "::" in name:
        # `__DOUBLE_COLON__` substitution (some masking tools rewrite `::`).
        # Drop `\b` anchor — substring match across underscore-glued tokens.
        dc = name.replace("::", "__DOUBLE_COLON__")
        candidates.append((dc, re.compile(rf"{re.escape(dc)}\s*\("), False))
        # Template-class form: `Class<...>::method` — spec name strips template
        # params but the source has them.
        cls, method = name.rsplit("::", 1)
        candidates.append((
            method,
            re.compile(rf"\b{re.escape(cls)}<[^>]*>::{re.escape(method)}\s*\("),
            False,
        ))
        # Bare-method fallback: in-class defs and multi-line `Class::\n method`.
        # Verifies a preceding line carries the qualifier (set bare_method=True).
        candidates.append((
            method,
            re.compile(rf"\b{re.escape(method)}\s*\("),
            True,
        ))

    for cand_label, pat, bare_method in candidates:
        qualifier = name.rsplit("::", 1)[0] if bare_method else None
        for i, ln in enumerate(lines):
            stripped = ln.lstrip()
            if stripped.startswith(("//", "*", "#")):
                continue
            if not pat.search(ln):
                continue
            # Bare-call check: only skip if line is a complete-on-one-line call
            # (ends with `;`). Multi-line decls like `funcName(arg1,` must NOT
            # be skipped because the bare name happens to start the line.
            if (cand_label and stripped.startswith(cand_label + "(")
                    and ln.rstrip().endswith(";")):
                continue
            # `{` should appear within ~12 lines (multi-line argument lists need slack)
            has_open = any("{" in lines[k] for k in range(i, min(i + 12, len(lines))))
            if not has_open:
                continue
            # For bare-method fallback, verify a preceding line carries the
            # `Class::` qualifier so we don't match an unrelated free function
            # of the same name.
            if bare_method:
                qual_ok = False
                for k in range(max(0, i - 4), i):
                    if "::" in lines[k] and qualifier.split("::")[-1] in lines[k]:
                        qual_ok = True; break
                if not qual_ok:
                    continue
            return _walk_back_c_preamble(lines, i)
    return None


def _walk_back_c_preamble(lines: list[str], start: int) -> int:
    """Walk back over preceding non-empty lines that look like a multi-line
    function preamble (return type / template params / `Class::` qualifier on
    its own line). Stops at any line containing `;`, `{`, or `}`, or a comment.

    Examples handled:
      template<int DIM>           ← walked back
      void Muscle::               ← walked back
      ApplyForceToBody(args)      ← original `start`
    """
    j = start - 1
    walked = 0
    while j >= 0 and walked < 4:
        prev = lines[j]
        prev_stripped = prev.strip()
        if not prev_stripped:
            j -= 1; walked += 1; continue
        if prev_stripped.startswith(("//", "*", "#")):
            break
        if any(c in prev for c in (";", "{", "}")):
            break
        start = j
        j -= 1; walked += 1
    return start


def _locate_braced(src: str, name: str, lang: str) -> Optional[tuple[int, int]]:
    """Locate a C or R function by name; return inclusive line range."""
    lines = src.splitlines()
    if lang == "r":
        pat = re.compile(rf"^\s*{re.escape(name)}\s*(<-|=)\s*function\s*\(")
        start = None
        for i, ln in enumerate(lines):
            if pat.match(ln):
                start = i; break
        if start is None:
            return None
        # Walk back over `#` comment lines (roxygen)
        b = start
        while b > 0 and lines[b - 1].lstrip().startswith("#"):
            b -= 1
        start = b
    else:  # c / c++
        start = _locate_c_start(lines, name)
        if start is None:
            return None

    # Find the opening `{` line
    open_line = start
    while open_line < len(lines) and "{" not in lines[open_line]:
        open_line += 1
    if open_line >= len(lines):
        return None

    # Brace-balance, ignoring braces inside strings and line comments.
    depth = 0
    end = open_line
    found = False
    in_block_comment = False
    for j in range(open_line, len(lines)):
        s = lines[j]
        i = 0
        in_str = None
        while i < len(s):
            ch = s[i]
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue
            if in_str:
                if ch == "\\" and nxt:
                    i += 2; continue
                if ch == in_str:
                    in_str = None
                i += 1; continue
            if ch == "/" and nxt == "/":  # line comment
                break
            if ch == "/" and nxt == "*":
                in_block_comment = True; i += 2; continue
            if ch == "#" and lang == "r":  # R line comment
                break
            if ch in ("'", '"'):
                in_str = ch; i += 1; continue
            if ch == "{":
                depth += 1; found = True
            elif ch == "}":
                depth -= 1
                if found and depth == 0:
                    end = j
                    return (start, end)
            i += 1
    return None


# ---------------------------------------------------------------------------
# Decorator preservation (Python)
# ---------------------------------------------------------------------------

def _extracted_python_has_decorators(code: str) -> bool:
    for ln in code.splitlines():
        s = ln.strip()
        if not s:
            continue
        return s.startswith("@")
    return False


def _masked_python_decorators(src_lines: list[str], def_line: int) -> list[str]:
    """Return decorator lines immediately above the `def` line (same indent)."""
    if def_line >= len(src_lines):
        return []
    indent = len(src_lines[def_line]) - len(src_lines[def_line].lstrip())
    out: list[str] = []
    j = def_line - 1
    while j >= 0:
        ln = src_lines[j]
        if not ln.strip():
            j -= 1; continue
        cur_indent = len(ln) - len(ln.lstrip())
        if cur_indent != indent or not ln.lstrip().startswith("@"):
            break
        out.insert(0, ln)
        j -= 1
    return out


# ---------------------------------------------------------------------------
# Splicing
# ---------------------------------------------------------------------------

def _normalize_block(code: str) -> str:
    """Strip BOM, normalize EOLs, drop trailing blank lines."""
    if code.startswith("﻿"):
        code = code[1:]
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = code.rstrip()
    return code


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
_THINK_OPEN_RE = re.compile(r"<think>.*$", re.S | re.I)
_FENCE_RE = re.compile(r"```[a-zA-Z]*\n(.*?)\n```", re.S)


def _strip_prose(code: str) -> str:
    """Remove <think>...</think> blocks and code-fence wrappers from agent output.

    Some Haiku traces leak reasoning text alongside code (visible in calc_f_el
    submissions: a `<think>` block embedded mid-file). The splicer's brace
    counter then sees prose-balanced gibberish and rejects the whole splice.
    Strip these wrappers up front so only real code reaches `_splice`.
    """
    # Remove balanced <think>...</think>
    code = _THINK_RE.sub("", code)
    # Remove unclosed <think> tail (model truncated mid-thought)
    code = _THINK_OPEN_RE.sub("", code)
    # If wrapped in ```...```, extract first fence
    m = _FENCE_RE.search(code)
    if m and m.group(1).strip():
        code = m.group(1)
    return code
