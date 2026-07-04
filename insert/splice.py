

def _extract_c_body(code: str, name: str) -> str:
    """Pull the function body out of agent's C/C++ submission.

    Three submission forms in the wild:
      A. Full function: `void Foo() { body }` — strip signature, return `body`
      B. Body only: `{ body }` — strip outer braces, return `body`
      C. Bare statements: no braces — return as-is

    The splice path injects whatever this returns between the original
    function's `{` and `}`, so the source's signature is preserved (handles
    multi-line `void Class::\\n method(...)` decls and bizarre golden-file
    formatting we can't easily reconstruct from the agent's output).
    """
    code = code.strip()
    if not code:
        return ""
    bare_name = name.split("::")[-1]
    has_sig = bool(re.search(rf"\b{re.escape(bare_name)}\s*\(", code))
    if has_sig:
        first_open = code.find("{")
        last_close = code.rfind("}")
        if first_open >= 0 and last_close > first_open:
            return code[first_open + 1 : last_close].strip("\n")
        # Signature without braces — fall through, treat as bare statements
    if code.startswith("{") and code.endswith("}"):
        return code[1:-1].strip("\n")
    return code


def _splice(src: str, name: str, lang: str, new_code: str) -> tuple[Optional[str], list[str]]:
    """Replace function `name` in `src` with `new_code`. Returns (patched_src, notes)."""
    notes: list[str] = []
    new_code = _normalize_block(new_code)
    if lang in ("c", "r"):
        new_code = _strip_prose(new_code)
    if not new_code:
        return None, ["extracted code is empty"]

    if lang == "python":
        rng = _locate_python(src, name)
    elif lang in ("c", "r"):
        rng = _locate_braced(src, name, lang)
    else:
        return None, [f"unsupported language: {lang}"]

    if rng is None:
        return None, [f"function `{name}` not found in masked source"]

    start, end = rng
    src_lines = src.splitlines()

    # C/C++ splice path: replace the BODY between the original function's `{`
    # and matching `}`, preserving the source's signature lines verbatim. This
    # handles agent submissions in three forms (full, body-only, bare statements)
    # and avoids reconstructing multi-line `void Class::\n method(...)` decls
    # from whatever the agent emitted.
    if lang == "c":
        open_line = start
        while open_line < len(src_lines) and "{" not in src_lines[open_line]:
            open_line += 1
        if open_line >= len(src_lines) or open_line > end:
            return None, ["could not locate opening `{` for splice"]
        body = _extract_c_body(new_code, name)
        # Preserve `{` opener (with anything before it on that line, e.g. ` {`)
        open_text = src_lines[open_line]
        open_idx = open_text.index("{")
        open_prefix = open_text[: open_idx + 1]
        # Preserve `}` closer (with anything after it on that line)
        close_text = src_lines[end]
        close_idx = close_text.rindex("}")
        close_suffix = close_text[close_idx:]
        body_lines = body.split("\n") if body else []
        new_lines = (
            src_lines[:open_line]
            + [open_prefix]
            + body_lines
            + [close_suffix]
            + src_lines[end + 1 :]
        )
        patched = "\n".join(new_lines)
        if src.endswith("\n") and not patched.endswith("\n"):
            patched += "\n"
        return patched, notes

    # Python: re-align the extracted snippet's indentation to the masked file's
    # function indent. The model often returns class-method-indented code (4
    # spaces of leading whitespace on every line because the def lives inside
    # a class). Naively prepending the target indent on top of that produces
    # 8-space indent and a nested-function definition that ast.parse accepts
    # but tests can't import. Dedent first, then prepend, so the splice always
    # lands at exactly the target indent regardless of what the model returned.
    if lang == "python":
        first_def_line = src_lines[start]
        target_indent = first_def_line[: len(first_def_line) - len(first_def_line.lstrip())]

        # 1) Strip any common leading whitespace the model emitted.
        new_code = textwrap.dedent(new_code)

        # 2) If the masked source had decorators above `def` and the model
        #    didn't return them, prepend them at base 0 (we'll re-indent below).
        if not _extracted_python_has_decorators(new_code):
            def_idx = next(
                (k for k in range(start, end + 1)
                 if re.match(rf"^\s*def\s+{re.escape(name)}\s*\(", src_lines[k])),
                None,
            )
            if def_idx is not None and def_idx > start:
                decos = [
                    d[len(target_indent):] if d.startswith(target_indent) else d
                    for d in src_lines[start:def_idx]
                ]
                if decos:
                    notes.append(f"preserved {len(decos)} decorator(s) from masked file")
                    new_code = "\n".join(decos) + "\n" + new_code

        # 3) Re-apply the target indent uniformly.
        if target_indent:
            new_code = "\n".join(
                target_indent + ln if ln else ln for ln in new_code.splitlines()
            )

    # Reassemble: keep src[:start] + new_code + src[end+1:]
    before = "\n".join(src_lines[:start])
    after = "\n".join(src_lines[end + 1:])

    parts = []
    if before:
        parts.append(before)
    parts.append(new_code)
    if after:
        parts.append(after)
    patched = "\n".join(parts)

    # Preserve trailing newline if the original had one
    if src.endswith("\n") and not patched.endswith("\n"):
        patched += "\n"

    return patched, notes


# ---------------------------------------------------------------------------
# Strict syntax validation
# ---------------------------------------------------------------------------

def _balanced(src: str, lang: str) -> tuple[bool, str]:
    """Check brace/paren/string balance for C/R. Returns (ok, reason)."""
    depth_brace = 0
    depth_paren = 0
    in_str: Optional[str] = None
    in_block_comment = False
    line_no = 1
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if ch == "\n":
            line_no += 1; i += 1; continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False; i += 2; continue
            i += 1; continue
        if in_str is not None:
            if ch == "\\" and nxt:
                i += 2; continue
            if ch == in_str:
                in_str = None
            i += 1; continue
        if ch == "/" and nxt == "/":
            j = src.find("\n", i)
            i = n if j == -1 else j; continue
        if ch == "/" and nxt == "*":
            in_block_comment = True; i += 2; continue
        if ch == "#" and lang == "r":
            j = src.find("\n", i)
            i = n if j == -1 else j; continue
        if ch in ("'", '"'):
            in_str = ch; i += 1; continue
        if ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1
            if depth_brace < 0:
                return False, f"unbalanced `}}` at line {line_no}"
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
            if depth_paren < 0:
                return False, f"unbalanced `)` at line {line_no}"
        i += 1
    if in_block_comment:
        return False, "unterminated /* ... */ comment"
    if in_str is not None:
        return False, f"unterminated {in_str} string"
    if depth_brace != 0:
        return False, f"brace imbalance ({depth_brace:+d})"
    if depth_paren != 0:
        return False, f"paren imbalance ({depth_paren:+d})"
    return True, ""


def _validate(text: str, lang: str) -> tuple[bool, str]:
    if lang == "python":
        try:
            ast.parse(text)
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} (line {e.lineno})"
        return True, ""
    if lang in ("c", "r"):
        return _balanced(text, lang)
    return False, f"unsupported language: {lang}"


# ---------------------------------------------------------------------------
# Per-(sample, mode) processor
# ---------------------------------------------------------------------------

@dataclass
class FilePatchResult:
    file_path: str
    functions: list[str]
    success: bool
    output_path: str
    masked_path: str
    notes: list[str]
    error: str = ""


@dataclass
class RunResult:
    key: str
    sample_id: str
    language: str
    mode: str
    files: list[FilePatchResult]


def _parse_key(name: str) -> Optional[tuple[str, str, str, str]]:
    """`python__037_..._00-foo__direct` -> (lang, sample, func_part, mode).

    Per-function flow only — keys are exactly 4 `__`-separated parts."""
    parts = name.split("__")
    if len(parts) != 4:
        return None
    lang, sample, func_part, mode = parts
    if lang not in _LANG_DIR or not sample or not func_part or not mode:
        return None
    return lang, sample, func_part, mode


def _golden_sibling(path: Path) -> Path:
    """`code/source.py` -> `code/source_golden.py`."""
    return path.with_name(path.stem + "_golden" + path.suffix)
