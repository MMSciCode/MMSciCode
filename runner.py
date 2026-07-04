#!/usr/bin/env python3
from __future__ import annotations

from stage_loader import load_stage_files


load_stage_files(__file__, [
    "runner/core.py",
    "runner/py_helpers.py",
    "runner/py_run.py",
    "runner/r_run.py",
    "runner/c_run.py",
    "runner/orchestrate.py",
    "runner/cli.py",
], globals())
