"""infer.py - call an OpenAI or OpenAI-compatible API on MMSciCode prompts.

Pipeline stage 2: model generation + per-language code extraction.

Reads a sample directory (or `--all`), rebuilds the prompt via
`build_prompt.load_problem` + `build_prompt.build_prompt`, calls the OpenAI
chat-completions endpoint by default, then extracts a clean
implementation per target function. **Does not** patch the original source
files — only writes artifacts under `intermediate_outputs/infer_preview/`
(the default chains directly into `insert.py`'s `--infer-root`).

Usage:
    # 1) cp .env.example .env  &&  fill in OPENAI_API_KEY
    # 2) call once for a single sample
    python infer.py data/Python/037_... --mode cot
    # 3) or run all samples in both modes
    python infer.py --all --mode both

Outputs (per sample × mode), under
intermediate_outputs/infer_preview/<lang>__<sample_id>__<mode>/:
    prompt.json           - exact {system, user} sent to the model
    response.txt          - raw assistant text
    response_meta.json    - latency, usage, finish_reason
    extraction.json       - per-function extraction status + notes
    <func_name>.<ext>     - cleaned, drop-in implementation per function
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from build_prompt import (
    build_prompt, load_function_problem, list_function_specs,
    func_part_for, Problem,
)


# ---------------------------------------------------------------------------
# .env loader (no python-dotenv dependency)
# ---------------------------------------------------------------------------

def load_dotenv(path: Path) -> None:
    """Lightweight .env loader. Existing env vars take precedence."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


def call_model(
    *, system: str, user: str, model: str, base_url: str, api_key: str,
    temperature: float = 0.0, max_tokens: int = 8192, attempts: int = 3,
    thinking: str = "",
) -> tuple[str, dict]:
    """Call an OpenAI-compatible chat-completions endpoint with simple retries.

    `thinking` controls the internal chain-of-thought on OpenAI-compatible
    endpoints that expose a thinking/reasoning toggle (passed through as
    `extra_body={"thinking": {"type": ...}}`):
        - "disabled": no hidden reasoning; the entire token budget
          goes to visible content. Required for our prompt-level CoT — we want
          reasoning to appear in `<reasoning>` tags inside `content`, not be
          hidden in `reasoning_content`.
        - "enabled": the model thinks first. `reasoning_content` is returned
          alongside `content` and saved to meta for inspection.
    Endpoints that do not support this field ignore it; leave it empty ("")
    to omit the field entirely.

    Returns (assistant_text, meta_dict).
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai SDK is required. Install with `pip install 'openai>=1.30'`."
        ) from e

    client = OpenAI(api_key=api_key, base_url=base_url)
    extra_body = {"thinking": {"type": thinking}} if thinking in {"disabled", "enabled"} else None

    last_err: Exception | None = None
    for i in range(attempts):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                extra_body=extra_body,
            )
            dur = time.time() - t0
            choice = resp.choices[0]
            text = choice.message.content or ""
            reasoning = getattr(choice.message, "reasoning_content", None) or ""
            usage = getattr(resp, "usage", None)
            meta = {
                "model": model,
                "thinking": thinking,
                "duration_s": round(dur, 2),
                "finish_reason": choice.finish_reason,
                "usage": _safe_dump(usage),
                "reasoning_content": reasoning,
                "attempt": i + 1,
            }
            return text, meta
        except Exception as e:
            last_err = e
            if i + 1 < attempts:
                wait = 2.0 * (2 ** i)
                print(f"  ! API call failed ({type(e).__name__}: {e}); "
                      f"retrying in {wait:.0f}s...", file=sys.stderr)
                time.sleep(wait)
    assert last_err is not None
    raise last_err


def _safe_dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    try:
        return dict(obj)
    except Exception:
        return str(obj)


# ---------------------------------------------------------------------------
# Code-block extraction
# ---------------------------------------------------------------------------

@dataclass
class Extraction:
    name: str
    language: str
    code: str
    extraction_method: str
    signature_matches: bool
    block_tag: str = ""
    notes: list[str] = field(default_factory=list)


_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_LANG_ALIASES: dict[str, set[str]] = {
    "python": {"python", "py", "python3"},
    "r": {"r", "rscript"},
    "c": {"c", "cpp", "c++", "cxx"},
}
_EXT_FOR: dict[str, str] = {"python": ".py", "r": ".R", "c": ".c"}


def _parse_fence_tag(tag: str) -> tuple[str, str]:
    """`python:foo` -> ('python','foo'); `python` -> ('python',''); `` -> ('','')."""
    tag = tag.strip()
    if ":" in tag:
        lang, name = tag.split(":", 1)
        return lang.strip().lower(), name.strip()
    return tag.lower(), ""


def _strip_reasoning(text: str) -> str:
    """Remove <reasoning>...</reasoning> and <think>...</think> sections."""
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def _block_defines(code: str, name: str, lang: str) -> bool:
    if lang == "python":
        return re.search(rf"^\s*def\s+{re.escape(name)}\s*\(", code, re.MULTILINE) is not None
    if lang == "r":
        return re.search(rf"^\s*{re.escape(name)}\s*(<-|=)\s*function\s*\(",
                         code, re.MULTILINE) is not None
    if lang == "c":
        # Match `<type> NAME(` style declaration with body opening brace nearby.
        for m in re.finditer(rf"\b{re.escape(name)}\s*\(", code):
            head = code.rfind("\n", 0, m.start())
            line = code[head + 1: m.start()]
            # Heuristic: declaration lines have a return-type token before the name
            # and aren't comment / preprocessor / string
            if line.lstrip().startswith(("//", "*", "#", '"', "'")):
                continue
            if re.search(r"\b(if|for|while|switch|return)\b", line):
                continue
            # Look ahead for an opening brace within ~8 lines
            tail = code[m.end(): m.end() + 800]
            if "{" in tail:
                return True
        return False
    return False
