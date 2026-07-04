#!/usr/bin/env python3
from __future__ import annotations

from stage_loader import load_stage_files


load_stage_files(__file__, [
    "infer/client.py",
    "infer/extract.py",
    "infer/driver.py",
], globals())
