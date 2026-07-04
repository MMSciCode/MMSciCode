

def _extract_signature(src: str, func_name: str, lang: str) -> str:
    if lang == "python":
        return _extract_python_signature(src, func_name)
    if lang == "r":
        return _extract_r_signature(src, func_name)
    if lang == "c":
        return _extract_c_signature(src, func_name)
    raise ValueError(lang)


# ---------------------------------------------------------------------------
# Cross-cutting extractors (run on every sample mechanically — no per-sample
# hand-coded rules). Two pieces of context:
#
#   1) Return-signature contract — from `*_reference.<ext>` (or legacy
#      `*_golden.<ext>`). The reference
#      implementation's `return` statement(s) tell the model exactly what
#      shape/values are expected. Without this the only signal is a
#      docstring on the masked file, which may be stale.
#
#   2) Sibling helper inventory — list every other top-level function in the
#      masked file (name + arg list). Helps the model make an informed
#      choice when multiple similar helpers exist (e.g. two helpers that
#      compute the same thing but only one is bug-free in the pinned env).
# ---------------------------------------------------------------------------

def _reference_path(masked_abs: Path) -> Path | None:
    """Return ``foo_reference.py`` next to ``foo.py`` if it exists."""
    cand = masked_abs.with_name(masked_abs.stem + "_reference" + masked_abs.suffix)
    return cand if cand.is_file() else None


def _golden_path(masked_abs: Path) -> Path | None:
    """Return the preferred complete implementation next to ``foo.py``.

    New samples use ``*_reference``. Keep ``*_golden`` as a legacy fallback so
    old datasets and partially migrated trees still build prompts.
    """
    ref = _reference_path(masked_abs)
    if ref is not None:
        return ref
    cand = masked_abs.with_name(masked_abs.stem + "_golden" + masked_abs.suffix)
    return cand if cand.is_file() else None


def _explicit_reference_path(sample_dir: Path, rel: str | None) -> Path | None:
    if not rel:
        return None
    p = sample_dir / rel
    return p if p.is_file() else None


def _extract_python_returns(golden_src: str, func_name: str) -> list[str]:
    try:
        tree = ast.parse(golden_src)
    except SyntaxError:
        return []
    target = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == func_name), None)
    if target is None:
        return []
    src_lines = golden_src.splitlines()
    out: list[str] = []
    for node in ast.walk(target):
        if isinstance(node, ast.Return) and node.value is not None:
            try:
                out.append("return " + ast.unparse(node.value))  # type: ignore[attr-defined]
            except AttributeError:
                # py < 3.9: snap the source line (may not capture multi-line)
                end = getattr(node, "end_lineno", node.lineno) or node.lineno
                snippet = "\n".join(src_lines[node.lineno - 1:end]).strip()
                out.append(snippet)
    return out


def _list_python_helpers(masked_src: str, exclude: set[str]) -> list[str]:
    try:
        tree = ast.parse(masked_src)
    except SyntaxError:
        return []
    src_lines = masked_src.splitlines()
    out: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name in exclude:
            continue
        # Reconstruct signature: prefer ast.unparse (3.9+); else snap the
        # source line that defines the function (handles single-line headers
        # — multi-line headers degrade to first line, still useful).
        try:
            sig = f"def {node.name}({ast.unparse(node.args)})"  # type: ignore[attr-defined]
        except AttributeError:
            line = src_lines[node.lineno - 1].strip()
            sig = re.sub(r":\s*(#.*)?$", "", line)
        first_doc = (ast.get_docstring(node) or "").split("\n", 1)[0].strip()
        out.append(f"{sig}  # {first_doc}" if first_doc else sig)
    return out


_R_FUNC_HEADER = re.compile(
    r"^\s*([A-Za-z_][A-Za-z_0-9.]*)\s*(?:<-|=)\s*function\s*(\([^)]*\))",
    re.MULTILINE,
)


def _r_function_body(src: str, func_name: str) -> str | None:
    lines = src.splitlines()
    pat = re.compile(rf"^\s*{re.escape(func_name)}\s*(<-|=)\s*function\s*\(")
    start = next((i for i, ln in enumerate(lines) if pat.match(ln)), None)
    if start is None:
        return None
    depth = 0
    seen_open = False
    for i in range(start, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1; seen_open = True
            elif ch == "}":
                depth -= 1
                if seen_open and depth == 0:
                    return "\n".join(lines[start:i + 1])
    return None


def _extract_r_returns(golden_src: str, func_name: str) -> list[str]:
    body = _r_function_body(golden_src, func_name)
    if body is None:
        return []
    # Explicit return(...) calls — handle nested parens
    out: list[str] = []
    for m in re.finditer(r"\breturn\s*\(", body):
        depth = 1
        i = m.end()
        while i < len(body) and depth > 0:
            if body[i] == "(":
                depth += 1
            elif body[i] == ")":
                depth -= 1
            i += 1
        out.append("return(" + body[m.end():i - 1].strip() + ")")
    if out:
        return out
    # No explicit return — last non-comment line of the body (R returns last expr)
    lines = [ln.strip() for ln in body.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    if len(lines) >= 2 and lines[-1] == "}":
        lines = lines[:-1]
    return [lines[-1]] if lines else []


def _list_r_helpers(masked_src: str, exclude: set[str]) -> list[str]:
    out: list[str] = []
    for m in _R_FUNC_HEADER.finditer(masked_src):
        name = m.group(1)
        if name in exclude:
            continue
        out.append(f"{name} <- function{m.group(2)}")
    return out


_C_FUNC_HEADER = re.compile(
    # `<return-type-tokens> <name>(<args>) {` — at column 0, not inside
    # a comment or preprocessor line, body open brace possibly on the next line.
    r"^(?P<rt>[a-zA-Z_][\w\s\*]*?)\s+\b(?P<name>[a-zA-Z_]\w*)\s*\((?P<args>[^)]*)\)\s*$",
    re.MULTILINE,
)
# iter5e: looser version — allow Class::method names and multi-line arg lists
# (just match the OPENING `(`; caller is responsible for balancing parens).
# Anchored at start-of-line so it doesn't match calls inside other functions.
_C_FUNC_HEADER_LOOSE = re.compile(
    r"^(?P<lead>(?:[a-zA-Z_]\w*\s+|\*\s*|&\s*)+)"   # one-or-more lead tokens (return type qualifiers etc)
    r"(?P<name>[a-zA-Z_]\w*(?:::[a-zA-Z_]\w*)*)"     # name, possibly Class::method
    r"\s*\(",
    re.MULTILINE,
)
_C_KEYWORDS = {"if", "while", "for", "switch", "return", "sizeof", "typedef",
               "static", "inline", "extern", "const", "void", "do", "else"}


def _extract_c_returns(golden_src: str, func_name: str) -> list[str]:
    """Find a function definition for ``func_name`` and list its `return ...;` lines."""
    lines = golden_src.splitlines()
    pat = re.compile(rf"\b{re.escape(func_name)}\s*\(")
    start = None
    for i, ln in enumerate(lines):
        if pat.search(ln) and not ln.lstrip().startswith(("//", "*", "#")):
            for look in range(i, min(i + 6, len(lines))):
                if "{" in lines[look]:
                    start = i
                    break
            if start is not None:
                break
    if start is None:
        return []
    depth = 0
    seen_open = False
    end = start
    for i in range(start, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1; seen_open = True
            elif ch == "}":
                depth -= 1
                if seen_open and depth == 0:
                    end = i; break
        if seen_open and depth == 0:
            break
    body = "\n".join(lines[start:end + 1])
    return [f"return {m.group(1).strip()};"
            for m in re.finditer(r"\breturn\s+([^;]+);", body)
            if m.group(1).strip()]


def _list_c_helpers(masked_src: str, exclude: set[str]) -> list[str]:
    out: list[str] = []
    lines = masked_src.splitlines()
    for i, ln in enumerate(lines):
        m = _C_FUNC_HEADER.match(ln)
        if not m:
            continue
        name = m.group("name")
        if name in exclude or name in _C_KEYWORDS:
            continue
        # Confirm it's a definition, not a prototype: the `{` should appear
        # within the next few lines (or on this line via trailing text).
        opens = "{" in ln
        if not opens:
            for j in range(i + 1, min(i + 4, len(lines))):
                if lines[j].lstrip().startswith("{"):
                    opens = True; break
                if lines[j].strip().endswith(";"):
                    break  # prototype, skip
        if not opens:
            continue
        out.append(f"{m.group('rt').strip()} {name}({m.group('args').strip()})")
    return out


def _extract_returns(golden_src: str, func_name: str, lang: str) -> list[str]:
    if lang == "python":
        return _extract_python_returns(golden_src, func_name)
    if lang == "r":
        return _extract_r_returns(golden_src, func_name)
    if lang == "c":
        return _extract_c_returns(golden_src, func_name)
    return []


def _list_helpers(masked_src: str, exclude: set[str], lang: str) -> list[str]:
    if lang == "python":
        return _list_python_helpers(masked_src, exclude)
    if lang == "r":
        return _list_r_helpers(masked_src, exclude)
    if lang == "c":
        return _list_c_helpers(masked_src, exclude)
    return []
