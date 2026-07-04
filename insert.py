#!/usr/bin/env python3
from __future__ import annotations

from stage_loader import load_stage_files


load_stage_files(__file__, [
    "insert/paths.py",
    "insert/locate.py",
    "insert/splice.py",
    "insert/patch.py",
    "insert/cli.py",
], globals())
