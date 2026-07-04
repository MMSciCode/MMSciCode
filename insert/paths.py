"""Splice infer.py-extracted function code back into the masked source file.

Naming convention enforced by this tool:
  xx_reference.<ext> — original full implementation (preferred)
  xx_golden.<ext>   — legacy full implementation fallback
  xx_masked.<ext>   — masked-stub version (snapshot of original `xx.<ext>`)
  xx.<ext>          — final patched file (model output spliced in)

Usage:
  # patch a single infer result into a side output tree
  python insert.py --infer-output intermediate_outputs/infer_preview/python__037..._direct

  # patch every result into intermediate_outputs/patched
  python insert.py --all

  # patch every result into a custom tree
  python insert.py --all --out-dir intermediate_outputs/patched

  # show the plan without writing
  python insert.py --all --dry-run
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LANG_DIR = {"python": "Python", "c": "C_CPP", "r": "R"}
_LANG_EXT = {"python": ".py", "c": ".c", "r": ".R"}


def _read(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if text.startswith("﻿"):
        text = text[1:]
    return text


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _masked_sibling(path: Path) -> Path:
    """`code/source.py` -> `code/source_masked.py`."""
    return path.with_name(path.stem + "_masked" + path.suffix)


def _reference_sibling(path: Path) -> Path:
    """`code/source.py` -> `code/source_reference.py`."""
    return path.with_name(path.stem + "_reference" + path.suffix)


def _resolve_data_root(explicit: Optional[Path]) -> Path:
    if explicit:
        return explicit.resolve()
    here = Path(__file__).resolve().parent
    for cand in (here / "data", here.parent / "data", Path.cwd() / "data"):
        if cand.is_dir() and any((cand / d).is_dir() for d in _LANG_DIR.values()):
            return cand.resolve()
    raise FileNotFoundError(
        "Could not locate data root. Pass --data-root explicitly."
    )


def _sample_dir(data_root: Path, lang_dir: str, sample_id: str) -> Path:
    """Resolve both legacy `Python/<sample>` and HF `Python/data/<sample>` layouts."""
    for cand in (data_root / lang_dir / sample_id,
                 data_root / lang_dir / "data" / sample_id):
        if cand.is_dir():
            return cand
    return data_root / lang_dir / sample_id


# ---------------------------------------------------------------------------
# Function locators (return (start_line, end_line) inclusive, 0-indexed)
# ---------------------------------------------------------------------------

def _locate_python(src: str, name: str) -> Optional[tuple[int, int]]:
    """Use ast to find the function and its decorators.

    Accepts four name forms:
      1. ``foo``                              — module-level function
      2. ``Class.method``                     — method inside a class
      3. ``method (ClassName)`` or
         ``Class.method (description)``       — selected_core_functions sometimes
                                                appends a parenthetical
                                                disambiguator (`forward (FNet)`,
                                                `LlamaAttention.forward (with block_list)`,
                                                `training_step (BoosterAlignmentTrainer)`)
      4. ``Free Form Heading``                — paper-section labels; sanitize
                                                to snake_case as last resort.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return _locate_python_regex(src, name)

    # Form 3: peel trailing parenthetical disambiguator. E.g. `forward (FNet)`
    # → name `forward`, disambiguator `FNet`. If the disambiguator is itself a
    # class name in the AST, prefer that class's method; otherwise fall back to
    # the bare name.
    paren_match = re.match(r"^(.+?)\s*\(([^()]+)\)\s*$", name)
    if paren_match:
        base = paren_match.group(1).strip()
        disambig = paren_match.group(2).strip()
        # If disambig looks like a ClassName, scope the search to that class.
        if disambig and disambig.replace("_", "").isalnum() and disambig[:1].isupper():
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == disambig:
                    method_name = base.split(".")[-1]
                    for child in node.body:
                        if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and child.name == method_name):
                            return _node_range(child)
        # Otherwise just resolve `base` (which may be `Class.method` or `method`).
        return _locate_python(src, base)

    # Form 2: Class.method — look inside the named class for the method.
    if "." in name:
        cls_name, _, method_name = name.partition(".")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for child in node.body:
                    if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and child.name == method_name):
                        return _node_range(child)
        # Fall through — also try matching method_name alone in case the
        # masked source has it at module scope.
        return _locate_python(src, method_name)

    # Form 1: plain identifier. Walk the whole tree; first match wins.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return _node_range(node)

    # Form 4: heading with spaces / non-identifier characters. Try the
    # snake_case sanitization and recurse once.
    if " " in name or not name.isidentifier():
        sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()
        if sanitized and sanitized != name:
            return _locate_python(src, sanitized)

    return None


def _node_range(node) -> Optional[tuple[int, int]]:
    """Return (start_line, end_line) inclusive 0-indexed for a function node."""
    start = node.lineno  # 1-indexed
    if node.decorator_list:
        start = min(d.lineno for d in node.decorator_list)
    end = getattr(node, "end_lineno", None)
    if end is None:
        return None
    return (start - 1, end - 1)


def _locate_python_regex(src: str, name: str) -> Optional[tuple[int, int]]:
    lines = src.splitlines()
    pat = re.compile(rf"^(\s*)def\s+{re.escape(name)}\s*\(")
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if not m:
            continue
        indent = len(m.group(1))
        # walk back over decorators at the same indent
        start = i
        j = i - 1
        while j >= 0 and lines[j].lstrip().startswith("@") and (len(lines[j]) - len(lines[j].lstrip())) == indent:
            start = j
            j -= 1
        # walk forward until a non-blank line with indent <= function indent
        end = i
        k = i + 1
        # First, finish the def header (may span multiple lines).
        while k < len(lines) and not re.search(r"\)\s*(->[^:]+)?\s*:\s*(#.*)?$", lines[k - 1]):
            if k >= len(lines):
                break
            end = k
            k += 1
        # Now consume the body
        for kk in range(end + 1, len(lines)):
            stripped = lines[kk].rstrip()
            if not stripped:
                end = kk
                continue
            cur_indent = len(lines[kk]) - len(lines[kk].lstrip())
            if cur_indent <= indent:
                break
            end = kk
        # Trim trailing blank lines
        while end > start and lines[end].strip() == "":
            end -= 1
        return (start, end)
    return None
