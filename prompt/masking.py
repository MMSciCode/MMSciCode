

# ---------------------------------------------------------------------------
# Masked-file construction: keep full file for Python/R (they already contain
# a stub); for C we synthesise a masked file by replacing the function body.
# ---------------------------------------------------------------------------

def _mask_python_function_body(src: str, func_name: str) -> str:
    """Replace the body of `func_name` with a NotImplementedError stub,
    preserving signature, docstring (if any), and decorators. Returns `src`
    unchanged if the function isn't found."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src

    # ``function_name`` sometimes carries a human parenthetical annotation, e.g.
    # ``forward (FNet)`` (the class) or ``LlamaAttention.forward (with ... masking)``
    # (a note). Normalise: a single-identifier parenthetical on an unqualified
    # name is the class -> ``Class.method``; anything else is dropped.
    pm = re.match(r"^(.*?)\s*\((.*)\)\s*$", func_name)
    if pm:
        head, paren = pm.group(1).strip(), pm.group(2).strip()
        if "." not in head and re.fullmatch(r"\w+", paren):
            func_name = f"{paren}.{head}"
        else:
            func_name = head

    target = None
    if "." in func_name:
        cls_name, _, method_name = func_name.partition(".")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for child in node.body:
                    if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and child.name == method_name):
                        target = child
                        break
                if target is not None:
                    break
        # The method may be a *nested* def inside the class (e.g. a closure
        # created in ``__init__`` and stored as an attribute). Fall back to a
        # recursive search within the class body.
        if target is None:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == cls_name:
                    for sub in ast.walk(node):
                        if (isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and sub.name == method_name):
                            target = sub
                            break
                    if target is not None:
                        break
    if target is None:
        for node in ast.walk(tree):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == func_name):
                target = node
                break
    if target is None or not target.body:
        return src

    lines = src.splitlines(keepends=True)
    body = target.body
    first = body[0]
    has_doc = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )

    end_line = target.end_lineno  # 1-indexed inclusive
    if end_line is None:
        return src

    _MSG = '("Function implementation has been masked for testing")\n'

    # Text on the body's first line that precedes the first statement. For a
    # normal multi-line def this is just indentation; for a single-line def
    # (``def f(x): return x``) it is the ``def ...:`` header itself.
    header_prefix = lines[first.lineno - 1][:first.col_offset]

    if header_prefix.strip():
        # Body shares the header's line. Keep the header up to and including
        # ``:`` and drop everything after it. (A docstring on the same line is
        # too rare/awkward to mask cleanly — leave the source unchanged.)
        if has_doc:
            return src
        head = header_prefix.rstrip()
        if not head.endswith(":"):
            return src  # can't locate the header colon; don't risk breaking it
        # Indent one level past the def, matching the file's tab/space style.
        def_ws = lines[target.lineno - 1][:target.col_offset]
        indent = def_ws + ("\t" if "\t" in def_ws else "    ")
        stub = f'{indent}raise NotImplementedError{_MSG}'
        return ("".join(lines[:first.lineno - 1]) + head + "\n"
                + stub + "".join(lines[end_line:]))

    # Body starts on its own line (normal case). Reuse the body's own leading
    # whitespace (tabs or spaces) so we never mix indentation styles.
    indent = header_prefix
    stub = f'{indent}raise NotImplementedError{_MSG}'
    keep_until_idx = first.end_lineno if has_doc else first.lineno - 1
    return "".join(lines[:keep_until_idx]) + stub + "".join(lines[end_line:])


def _mask_r_function_body(src: str, func_name: str) -> str:
    """Replace the body of `func_name <- function(...)` with a stop() stub.
    Walks brace balance (skipping strings and `#` line comments) to find the
    matching `}`."""
    lines = src.splitlines(keepends=True)
    pat = re.compile(rf"^\s*{re.escape(func_name)}\s*(<-|=)\s*function\s*\(")
    start = next((i for i, ln in enumerate(lines) if pat.match(ln)), None)
    if start is None:
        return src

    open_line = start
    while open_line < len(lines) and "{" not in lines[open_line]:
        open_line += 1
    if open_line >= len(lines):
        return src

    header_indent = len(lines[start]) - len(lines[start].lstrip())
    body_indent = " " * (header_indent + 2)

    depth = 0
    seen_open = False
    in_str: str | None = None
    end_line = open_line
    found = False
    for j in range(open_line, len(lines)):
        s = lines[j]
        i = 0
        while i < len(s):
            ch = s[i]
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if in_str is not None:
                if ch == "\\" and nxt:
                    i += 2; continue
                if ch == in_str:
                    in_str = None
                i += 1; continue
            if ch == "#":
                break
            if ch in ("'", '"'):
                in_str = ch; i += 1; continue
            if ch == "{":
                depth += 1; seen_open = True
            elif ch == "}":
                depth -= 1
                if seen_open and depth == 0:
                    end_line = j; found = True; break
            i += 1
        if found:
            break
    if not found:
        return src

    open_text = lines[open_line]
    brace_pos = open_text.index("{")
    head = open_text[: brace_pos + 1] + "\n"

    close_text = lines[end_line]
    close_pos = close_text.index("}")
    tail = close_text[close_pos:]

    stub = f'{body_indent}stop("Function implementation has been masked")\n'

    return (
        "".join(lines[:open_line])
        + head
        + stub
        + tail
        + "".join(lines[end_line + 1:])
    )


def _strip_c_code(line: str, state: dict) -> str:
    """Return `line` with string/char literals and comments blanked out, so a
    brace or the function name inside a literal/comment is never counted.
    `state` carries block-comment status across lines: {'blockcomment': bool}."""
    out = []
    i, n = 0, len(line)
    in_str = None  # '"' or "'"
    while i < n:
        ch = line[i]
        nxt = line[i + 1] if i + 1 < n else ""
        if state.get("blockcomment"):
            if ch == "*" and nxt == "/":
                state["blockcomment"] = False
                i += 2; continue
            i += 1; continue
        if in_str is not None:
            if ch == "\\" and nxt:
                i += 2; continue
            if ch == in_str:
                in_str = None
            i += 1; continue
        if ch == "/" and nxt == "/":
            break  # line comment
        if ch == "/" and nxt == "*":
            state["blockcomment"] = True
            i += 2; continue
        if ch in ('"', "'"):
            in_str = ch
            out.append(" ")  # blank the literal
            i += 1; continue
        out.append(ch)
        i += 1
    return "".join(out)


def _mask_c_function_body(src: str, func_name: str) -> str:
    """Replace the body of `func_name` with a TODO placeholder, preserving the
    rest of the file verbatim.

    Robust to (a) qualified C++ names (``Class::method`` / ``ns::Class::method``)
    whose out-of-class definition is preferred over a same-named call or a
    nested occurrence, and (b) braces/`;`/the name appearing inside strings,
    char literals or comments (string/comment-aware scanning)."""
    lines = src.splitlines(keepends=True)
    parts = re.split(r"::|\.", func_name)
    bare = parts[-1]
    # The reference/golden copy we mask sometimes renames the target with a
    # ``_golden`` / ``_reference`` suffix and/or flattens the namespace with
    # underscores (``cpihmc::elec_num::mc_evolve`` -> ``elec_num_mc_evolve_golden``).
    # Build candidate spellings from progressively longer trailing joins.
    cands: list[str] = []
    for k in range(len(parts), 0, -1):
        joined = "_".join(parts[-k:])
        for suf in ("_golden", "_reference", ""):
            if joined + suf not in cands:
                cands.append(joined + suf)
    name_alts = "|".join(re.escape(n) for n in cands)
    # ``(?<!\w)`` rejects mid-identifier hits (``initialize_fold`` when masking
    # ``fold``) but still allows a ``::``/``.`` qualifier immediately before the
    # name, so the out-of-class definition ``Ret Class::method(`` is matched.
    name_re = re.compile(rf"(?<!\w)(?:{name_alts})\s*\(")

    def body_open_line(i):
        """Index of the line holding the body's opening `{`, or None if a `;`
        (declaration/prototype/statement) closes it first."""
        st = {}
        for j in range(i, min(i + 15, len(lines))):
            for ch in _strip_c_code(lines[j], st):
                if ch == "{":
                    return j
                if ch == ";":
                    return None
        return None

    candidates = []
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith(("//", "*", "/*")):
            continue
        if not name_re.search(_strip_c_code(ln, {})):
            continue
        oc = body_open_line(i)
        if oc is None:
            continue
        qualified = bool(re.search(rf"\w\s*(?:::|\.)\s*{re.escape(bare)}\s*\(",
                                   _strip_c_code(ln, {})))
        candidates.append((i, oc, qualified))
    if not candidates:
        return src  # unmodified

    # Prefer the qualified out-of-class definition (skips spurious same-name
    # occurrences such as a nested/inline mention); else the first definition.
    qual = [c for c in candidates if c[2]]
    start, open_line, _ = qual[0] if qual else candidates[0]

    # Brace-balance from the opening `{`, string/comment-aware.
    st = {}
    depth = 0
    found_open = False
    end_line = None
    for j in range(open_line, len(lines)):
        for ch in _strip_c_code(lines[j], st):
            if ch == "{":
                depth += 1; found_open = True
            elif ch == "}":
                depth -= 1
                if found_open and depth == 0:
                    end_line = j
                    break
        if end_line is not None:
            break
    if end_line is None:
        return src

    header = "".join(lines[start:open_line])
    open_txt = lines[open_line]
    bpos = open_txt.index("{")
    head = header + open_txt[:bpos + 1]
    if not head.endswith("\n"):
        head += "\n"
    close_txt = lines[end_line]
    cpos = close_txt.index("}")
    closing = close_txt[cpos:]
    if not closing.endswith("\n"):
        closing += "\n"
    todo = "    /* TODO: Implement per paper description. */\n"
    return "".join(lines[:start]) + head + todo + closing + "".join(lines[end_line + 1:])


_STUB_MARKERS = ("has been masked", "NotImplementedError", "TODO: Implement",
                 "not implemented", "masked for")


def _target_has_real_def(base: str, func_name: str, lang: str) -> bool:
    """True if `base` contains a NON-stub definition of the target function.

    Used to turn a silent no-op mask (masker couldn't locate the target, so it
    returned the source verbatim and the real body would leak into the prompt)
    into a hard error. Mirrors each masker's locate logic so it only fires when
    the masker genuinely should have masked something but didn't; a legitimate
    no-op (target simply absent from this file) returns False."""
    if lang == "python":
        try:
            tree = ast.parse(base)
        except SyntaxError:
            return False
        pm = re.match(r"^(.*?)\s*\((.*)\)\s*$", func_name)
        if pm:
            head, paren = pm.group(1).strip(), pm.group(2).strip()
            func_name = f"{paren}.{head}" if ("." not in head and re.fullmatch(r"\w+", paren)) else head
        bare = func_name.rsplit(".", 1)[-1]
        cls = func_name.rsplit(".", 1)[0] if "." in func_name else None

        def _is_real(fn):
            stmts = [s for s in fn.body if not (isinstance(s, ast.Expr)
                     and isinstance(getattr(s, "value", None), ast.Constant))]
            return not (len(stmts) == 1 and isinstance(stmts[0], (ast.Raise, ast.Pass)))

        scope = tree
        if cls:
            scope = next((n for n in ast.walk(tree)
                          if isinstance(n, ast.ClassDef) and n.name == cls), None)
            if scope is None:
                return False
        for node in ast.walk(scope):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == bare and _is_real(node)):
                return True
        return False
    if lang == "r":
        bare = func_name.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
        m = re.search(rf"(?m)^\s*{re.escape(bare)}\s*(?:<-|<<-|=)\s*function", base)
        if not m:
            return False
        return not any(s in base[m.end(): m.end() + 200] for s in _STUB_MARKERS)
    if lang == "c":
        parts = re.split(r"::|\.", func_name)
        alts = []
        for k in range(len(parts), 0, -1):
            j = "_".join(parts[-k:])
            for suf in ("", "_golden", "_reference"):
                if j + suf not in alts:
                    alts.append(j + suf)
        alt_re = "|".join(re.escape(a) for a in alts)
        for m in re.finditer(rf"(?<!\w)(?:{alt_re})\s*\([^;{{]*\)[^;{{]*\{{", base):
            if not any(s in base[m.end(): m.end() + 200] for s in _STUB_MARKERS):
                return True
        return False
    return False


# ---------------------------------------------------------------------------
# Smart truncation of the masked file (iter4)
#
# Many of our "long-tail" prompts (~1.7% of 1152) push past 64k tokens because
# the masked file dump contains thousands of lines of unrelated module-level
# driver code, plotting helpers, or sibling functions the target never touches.
# `_smart_truncate_file` reduces that file to:
#   - the target function's signature + masked stub + docstring (always kept)
#   - direct callees (functions referenced inside the target body) — kept full
#   - direct callers (functions whose body references the target) — kept full
#   - other functions — replaced with a 1-line signature stub
#   - module-level driver/orchestration code outside any function — dropped
# Falls back to the original file when parsing fails, so the worst case is
# "no shrink", not "broken prompt".
# ---------------------------------------------------------------------------

# iter5e: lazy-load real Qwen tokenizer for accurate budget bookkeeping.
# Empirical chars/Qwen-token ratio is 2.6-2.8 for English natural language,
# 3.0-3.5 for code. Old char/4 heuristic let paper-excerpts overshoot 42%;
# even char/3 is off by ~6% on R Methods sections. Real tokenizer is the
# only way to hit a hard 8k cap reliably.
_QWEN_TOK = None
# Optional real tokenizer for exact prompt token budgets. Set
# MMSCI_QWEN_TOKENIZER to a local directory or a HuggingFace model id (e.g.
# "Qwen/Qwen2.5-7B") to enable it; requires `transformers`. When unset (the
# default), token counts use the char/3 heuristic in `_approx_tokens` — no
# external dependency and no network access required.
_QWEN_TOK_PATH = os.environ.get("MMSCI_QWEN_TOKENIZER", "")


def _get_qwen_tokenizer():
    global _QWEN_TOK
    if _QWEN_TOK is None:
        if not _QWEN_TOK_PATH:
            _QWEN_TOK = False  # not configured; use heuristic
        else:
            try:
                from transformers import AutoTokenizer
                _QWEN_TOK = AutoTokenizer.from_pretrained(_QWEN_TOK_PATH)
            except Exception:
                _QWEN_TOK = False  # disable, fall back to heuristic
    return _QWEN_TOK if _QWEN_TOK else None


def _approx_tokens(s: str) -> int:
    """Token count for budget bookkeeping. Uses real Qwen tokenizer when
    available (cached); falls back to char/3 heuristic if transformers not
    importable. char/3 covers both natural language and code reasonably."""
    tok = _get_qwen_tokenizer()
    if tok is not None:
        try:
            return len(tok.encode(s))
        except Exception:
            pass
    return len(s) // 3 + 1
