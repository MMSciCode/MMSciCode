from __future__ import annotations

from pathlib import Path


def load_stage_files(wrapper_file: str | Path, files: list[str], namespace: dict) -> None:
    """Execute a shallow list of stage fragments in the wrapper namespace."""
    wrapper_path = Path(wrapper_file).resolve()
    root = wrapper_path.parent
    paths = [root / name for name in files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing stage files: {missing}")
    namespace["__file__"] = str(wrapper_path)
    for path in paths:
        source = path.read_text(encoding="utf-8")
        exec(compile(source, str(path), "exec"), namespace)
