#!/usr/bin/env python3
from __future__ import annotations

from stage_loader import load_stage_files


load_stage_files(__file__, [
    "prompt/models.py",
    "prompt/returns.py",
    "prompt/masking.py",
    "prompt/truncate_py.py",
    "prompt/truncate_r.py",
    "prompt/truncate_c.py",
    "prompt/context.py",
    "prompt/loader.py",
    "prompt/problem.py",
    "prompt/render.py",
], globals())
