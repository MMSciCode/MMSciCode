

def _smart_truncate_python(masked_src: str, target_fn: str, max_tokens: int) -> str:
    """Drop module-level driver code, stub unrelated helpers, keep target +
    direct callers/callees of the target."""
    if _approx_tokens(masked_src) <= max_tokens:
        return masked_src
    try:
        tree = ast.parse(masked_src)
    except SyntaxError:
        return masked_src

    src_lines = masked_src.splitlines(keepends=True)

    # Resolve target_fn (dotted Class.method or plain) — pick the FunctionDef
    # whose body contains the masked stub (or the first match by name).
    method_name = target_fn.partition(".")[2] if "." in target_fn else target_fn
    plain_name = method_name or target_fn

    target_node: ast.AST | None = None
    if "." in target_fn:
        cls_name = target_fn.partition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for child in node.body:
                    if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and child.name == plain_name):
                        target_node = child
                        break
                if target_node is not None:
                    break
    if target_node is None:
        for node in ast.walk(tree):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == plain_name):
                target_node = node
                break
    if target_node is None:
        return masked_src

    # Names called by target (callees)
    callees: set[str] = set()
    for n in ast.walk(target_node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                callees.add(f.id)
            elif isinstance(f, ast.Attribute):
                callees.add(f.attr)

    # Top-level FunctionDef + ClassDef nodes; record direct callers (functions
    # whose body references the target name).
    keep_full: set[int] = set()  # ids of nodes to keep full
    stub_only: list[ast.AST] = []  # nodes to compress to a one-line signature

    target_call_names = {plain_name, target_fn}

    def _refs_target(node: ast.AST) -> bool:
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                f = n.func
                nm = (
                    f.id if isinstance(f, ast.Name)
                    else f.attr if isinstance(f, ast.Attribute)
                    else None
                )
                if nm in target_call_names:
                    return True
        return False

    # Walk top-level nodes (under module + classes) to collect candidates
    top_nodes: list[ast.AST] = list(tree.body)
    for top in tree.body:
        if isinstance(top, ast.ClassDef):
            top_nodes.extend(top.body)

    for node in top_nodes:
        if node is target_node:
            keep_full.add(id(node))
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in callees or _refs_target(node):
                keep_full.add(id(node))

    # Containment: include the enclosing class of the target node (we walk
    # at module-level; class-method targets won't appear in tree.body, so we
    # need to keep the parent class). The strategy: rebuild line-by-line.

    # For each top-level FunctionDef/ClassDef/AsyncFunctionDef, decide:
    #   - keep full (in keep_full)
    #   - stub it (replace body with a `# ... (other functions stubbed)` line)
    # For other top-level statements (imports, constants), KEEP imports and
    # `from ... import ...` lines, top-level Assign of small literals, and
    # top-level `if __name__ == "__main__":` only if it's short.

    def _is_module_driver(node: ast.AST) -> bool:
        # Anything that's not an import / class / function / simple assign:
        # treat as driver code to drop.
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False
        if isinstance(node, ast.Assign):
            # Keep simple top-level constants
            try:
                src = ast.get_source_segment(masked_src, node) or ""
                return len(src.splitlines()) > 8 or _approx_tokens(src) > 200
            except Exception:
                return True
        if isinstance(node, ast.AnnAssign):
            return False
        if isinstance(node, ast.Expr):
            # Module-level docstrings (constant Expr) — keep
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return False
            return True
        if isinstance(node, ast.If):
            # `if __name__ == "__main__":` — drop body but keep the guard? Just drop.
            return True
        # Try / Match / With / For / While at module level: driver
        return True

    def _segment(node: ast.AST) -> str:
        try:
            seg = ast.get_source_segment(masked_src, node)
        except Exception:
            seg = None
        if seg is None:
            # Reconstruct from line numbers
            start = (node.lineno or 1) - 1
            end = (getattr(node, "end_lineno", None) or node.lineno) - 1
            seg = "".join(src_lines[start:end + 1]).rstrip("\n")
        return seg

    def _signature_stub(node: ast.AST) -> str:
        # Single-line signature + brief stub line
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                args = ast.unparse(node.args)  # type: ignore[attr-defined]
            except AttributeError:
                args = "..."
            kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            doc = (ast.get_docstring(node) or "").strip().split("\n", 1)[0]
            indent_col = node.col_offset
            indent = " " * indent_col
            body_indent = " " * (indent_col + 4)
            head = f"{indent}{kw} {node.name}({args}):"
            doc_line = (
                f'{body_indent}"""{doc}"""' if doc
                else f"{body_indent}# (body omitted to fit token budget)"
            )
            stub_line = f"{body_indent}..."
            if doc:
                return f"{head}\n{doc_line}\n{stub_line}"
            return f"{head}\n{stub_line}"
        if isinstance(node, ast.ClassDef):
            # Stub a class: keep the class header + a pass
            indent = " " * node.col_offset
            return f"{indent}class {node.name}:\n{indent}    ...  # (class body omitted)"
        return ""

    def _render_class(cls: ast.ClassDef, target_inside: bool) -> str:
        """If the target is inside this class, keep the class header + the
        target method full + signature-stubs for siblings. Otherwise return
        either the full class (if it's small / referenced) or a stub."""
        full_seg = _segment(cls)
        if not target_inside and id(cls) in keep_full:
            return full_seg
        if not target_inside:
            return _signature_stub(cls)
        # Target is inside: rebuild
        header_line = src_lines[cls.lineno - 1].rstrip("\n")
        # Class docstring?
        body_parts: list[str] = []
        cls_indent = cls.col_offset
        first_stmt = cls.body[0] if cls.body else None
        has_cls_doc = (
            isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)
        )
        if has_cls_doc:
            body_parts.append(_segment(first_stmt))
        for child in cls.body:
            if has_cls_doc and child is first_stmt:
                continue
            if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and id(child) in keep_full):
                body_parts.append(_segment(child))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body_parts.append(_signature_stub(child))
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                body_parts.append(_segment(child))
            else:
                # Skip nested classes / driver code in classes we're trimming
                continue
        # Re-indent body parts to ensure class indent + 4 spaces (Python
        # AST already gives correctly indented segments via get_source_segment,
        # so just stitch).
        body_text = "\n".join(body_parts)
        if not body_text.strip():
            body_text = " " * (cls_indent + 4) + "pass"
        return header_line + "\n" + body_text

    # Determine if the target lives inside a top-level class
    target_class: ast.ClassDef | None = None
    for top in tree.body:
        if isinstance(top, ast.ClassDef):
            for child in top.body:
                if child is target_node:
                    target_class = top
                    break
        if target_class is not None:
            break

    # Module-level docstring (if any)
    pieces: list[str] = []
    body_idx = 0
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        pieces.append(_segment(tree.body[0]))
        body_idx = 1

    # Collect imports first (always keep)
    for node in tree.body[body_idx:]:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            pieces.append(_segment(node))

    # Then top-level constants/small assigns
    for node in tree.body[body_idx:]:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if not _is_module_driver(node):
                pieces.append(_segment(node))

    # Then classes and functions in source order
    seen_target = False
    for node in tree.body[body_idx:]:
        if isinstance(node, ast.ClassDef):
            target_inside = (target_class is node)
            seg = _render_class(node, target_inside)
            pieces.append(seg)
            if target_inside:
                seen_target = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if id(node) in keep_full:
                pieces.append(_segment(node))
                if node is target_node:
                    seen_target = True
            else:
                pieces.append(_signature_stub(node))

    if not seen_target:
        # Safety net: if for any reason the target wasn't emitted, fall back
        # to the original masked src (better long than missing the stub).
        return masked_src

    omitted_marker = (
        "\n# ---- Module-level driver code and unrelated helpers omitted "
        "to fit the token budget ----\n"
    )
    out = "\n\n".join(p for p in pieces if p and p.strip())
    out = out + "\n" + omitted_marker
    # If still over budget, return as-is (we did our best); the final guard
    # will hard-truncate the user prompt.
    return out
