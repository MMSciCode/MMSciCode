"""Build Direct-Answer and CoT prompts for MMSciCode samples.

Usage (CLI):
    python build_prompt.py <sample_dir> --mode direct|cot [--out <path>]
    python build_prompt.py --all [--out-dir <dir>]

Library:
    from build_prompt import build_prompt, load_problem
    problem = load_problem("/path/to/sample_dir")
    prompt  = build_prompt(problem, mode="cot")
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FunctionSpec:
    name: str
    file_path: str                      # repo-relative
    language: str                       # "python" | "c" | "r"
    description: str
    paper_section: str
    article_reference: str
    formula: str
    key_terms_mapping: dict[str, str]
    implementation_instruction: str
    masked_file_content: str            # full file with body stubbed (model sees this)
    signature_block: str                # signature + docstring / declaration only
    # Cross-cutting context — derived mechanically from each sample's own
    # ground truth (`*_golden`, masked source AST). Empty when not derivable.
    return_signatures: list[str] = field(default_factory=list)   # from *_golden body
    siblings_in_file: list[str] = field(default_factory=list)    # other top-level fns
    extra_context: list[dict[str, str]] = field(default_factory=list)
    # extra_context entries: {"title": str, "path": str, "content": str}


@dataclass
class Problem:
    sample_id: str
    language: str                       # "python" | "c" | "r"
    codebase_root: Path
    paper_title: str
    paper_url: str
    paper_subject: str
    paper_context: str                  # extracted section(s) / abstract
    functions: list[FunctionSpec]
    env_info: dict[str, Any]


# ---------------------------------------------------------------------------
# Language detection & sample loading
# ---------------------------------------------------------------------------

def _detect_language(sample_dir: Path) -> str:
    parts = {p.lower() for p in sample_dir.parts}
    if "python" in parts:
        return "python"
    if "c_cpp" in parts or "cpp" in parts:
        return "c"
    if "r" in parts:
        return "r"
    # fallback: inspect files
    if (sample_dir / "code").exists() and any(sample_dir.glob("code/*.py")):
        return "python"
    if any(sample_dir.rglob("*.R")):
        return "r"
    if any(sample_dir.rglob("*.c")) or any(sample_dir.rglob("*.cpp")):
        return "c"
    raise ValueError(f"Could not detect language for {sample_dir}")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Source extraction: function signature + docstring
# ---------------------------------------------------------------------------

def _extract_python_signature(src: str, func_name: str) -> str:
    """Return `def func(...):` + docstring (up to the first statement of body).

    Handles dot-notation `Class.method` by walking the AST to find the matching
    method inside the named class (mirrors `_mask_python_function_body`)."""
    lines = src.splitlines()

    # Paren-disambig suffix `name (ClassName)` or `Class.method (note)`.
    # If the parens contain a single Python identifier, treat it as the class
    # name and rewrite to dotted form. Otherwise strip the suffix entirely so
    # the regex can still match by leaf name (description text in parens
    # is informational only, not part of the actual function name).
    paren_match = re.match(r"^([\w.]+)\s*\(([^)]+)\)\s*$", func_name)
    if paren_match:
        leaf = paren_match.group(1)
        note = paren_match.group(2).strip()
        if re.fullmatch(r"[A-Za-z_]\w*", note) and "." not in leaf:
            # `forward (FNet)` -> `FNet.forward`
            func_name = f"{note}.{leaf}"
        else:
            # `LlamaAttention.forward (with block_list masking)` -> first match
            func_name = leaf

    # Dotted name: Class.method — locate via AST so we hit the right method
    # rather than the first global/method that happens to share the name.
    if "." in func_name:
        cls_name, _, method_name = func_name.partition(".")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            tree = None
        if tree is not None:
            target = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == cls_name:
                    for child in node.body:
                        if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and child.name == method_name):
                            target = child
                            break
                    if target is not None:
                        break
            if target is not None:
                start = target.lineno - 1  # 0-indexed
                # Locate header end on this/next lines (multi-line headers exist).
                header_end = start
                while (header_end < len(lines)
                       and not re.search(r"\)\s*(->[^:]+)?\s*:\s*(#.*)?$",
                                         lines[header_end])):
                    header_end += 1
                block = lines[start:header_end + 1]
                # Capture docstring if present (uses AST docstring for accuracy)
                doc_node = target.body[0] if target.body else None
                has_doc = (
                    doc_node is not None
                    and isinstance(doc_node, ast.Expr)
                    and isinstance(doc_node.value, ast.Constant)
                    and isinstance(doc_node.value.value, str)
                )
                if has_doc:
                    doc_start = doc_node.lineno - 1
                    doc_end = (doc_node.end_lineno or doc_node.lineno) - 1
                    # Include any blank-line gap between header and docstring
                    for k in range(header_end + 1, doc_start):
                        block.append(lines[k])
                    block.extend(lines[doc_start:doc_end + 1])
                return "\n".join(block)
        # Fall through: try plain method-name lookup as a last resort
        method_pat = re.compile(rf"^\s*def\s+{re.escape(method_name)}\s*\(")
        start = next((i for i, ln in enumerate(lines) if method_pat.match(ln)), None)
        if start is None:
            return f"# def {func_name}(...): not found"
        header_end = start
        while (header_end < len(lines)
               and not re.search(r"\)\s*(->[^:]+)?\s*:\s*(#.*)?$",
                                 lines[header_end])):
            header_end += 1
        return "\n".join(lines[start:header_end + 1])

    pat = re.compile(rf"^\s*def\s+{re.escape(func_name)}\s*\(")
    start = next((i for i, ln in enumerate(lines) if pat.match(ln)), None)
    if start is None:
        return f"# def {func_name}(...): not found"

    # Header may span multiple lines until we hit the closing ') :'
    header_end = start
    while header_end < len(lines) and not re.search(r"\)\s*(->[^:]+)?\s*:\s*(#.*)?$",
                                                    lines[header_end]):
        header_end += 1
    block = lines[start:header_end + 1]

    # Capture docstring if present
    j = header_end + 1
    # skip blank lines
    while j < len(lines) and lines[j].strip() == "":
        block.append(lines[j]); j += 1
    if j < len(lines):
        stripped = lines[j].lstrip()
        if stripped.startswith(('"""', "'''")):
            quote = stripped[:3]
            block.append(lines[j])
            # single-line docstring
            if stripped.count(quote) >= 2 and len(stripped) > 3:
                pass
            else:
                j += 1
                while j < len(lines):
                    block.append(lines[j])
                    if quote in lines[j]:
                        break
                    j += 1
    return "\n".join(block)


def _extract_r_signature(src: str, func_name: str) -> str:
    """Return `func_name <- function(...) {` plus any leading roxygen comments."""
    lines = src.splitlines()
    pat = re.compile(rf"^\s*{re.escape(func_name)}\s*(<-|=)\s*function\s*\(")
    start = next((i for i, ln in enumerate(lines) if pat.match(ln)), None)
    if start is None:
        return f"# {func_name} <- function(...): not found"

    # Walk back over contiguous `#` comments (roxygen / description)
    begin = start
    while begin > 0 and lines[begin - 1].lstrip().startswith("#"):
        begin -= 1

    # Walk forward until we see the opening `{` to get full header
    end = start
    while end < len(lines) and "{" not in lines[end]:
        end += 1
    return "\n".join(lines[begin:end + 1])


def _extract_c_signature(src: str, func_name: str) -> str:
    """Return the C function declaration line up to and including `{`."""
    lines = src.splitlines()
    # Match lines with the function name followed by `(` (not a call like foo()...)
    pat = re.compile(rf"^[^/].*\b{re.escape(func_name)}\s*\(")
    # Accept only declaration-style lines (contain a return type token before the name)
    start = None
    for i, ln in enumerate(lines):
        if pat.match(ln) and not ln.lstrip().startswith(("//", "*", "#")):
            # Heuristic: is this a declaration? Look for the matching '{' within a few lines.
            for look in range(i, min(i + 6, len(lines))):
                if "{" in lines[look]:
                    start = i
                    break
            if start is not None:
                break
    if start is None:
        return f"// {func_name}(...): not found"

    end = start
    while end < len(lines) and "{" not in lines[end]:
        end += 1
    return "\n".join(lines[start:end + 1])
