#!/usr/bin/env python3
"""Map Docker images to benchmark samples.

Each sample records the execution environment it needs in
``unit_test_status.json`` under ``environment.conda_env_name``. The dataset's
``index.tsv`` maps that environment id to a Docker image name. This script
joins the two so batch runners know which image to launch for each sample.

The join is sample -> env -> image, which is the correct direction for
*shared* environments: many R samples share ``r_test_env`` and many C/C++
samples share ``cpp_test_env``, so a single image legitimately maps to
multiple samples. (The previous image -> sample direction could not express
that and silently dropped every shared-env sample.)

Samples whose ``environment`` has no ``conda_env_name`` (e.g. standalone
C/C++ samples compiled with the system toolchain) have no Docker image and
are simply omitted from the output.
"""
import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


def data_dirs(data_root):
    """Return {sample_id: sample_dir} across the three language trees."""
    out = {}
    for lang in ("Python", "C_CPP", "R"):
        nested = data_root / lang / "data"
        d = nested if nested.is_dir() else data_root / lang
        if not d.is_dir():
            continue
        for child in sorted(d.iterdir()):
            if child.is_dir():
                out[child.name] = child
    return out


def sample_env_name(sample_dir):
    """Read environment.conda_env_name from the sample's unit_test_status.json."""
    p = sample_dir / "unit_test_status.json"
    if not p.is_file():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    return ((data.get("environment") or {}).get("conda_env_name") or "").strip()


def load_env_to_image(index_path):
    """Build {env_id: image} from index.tsv."""
    mapping = {}
    with index_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            env = (row.get("env") or "").strip()
            image = (row.get("image") or "").strip()
            if env and image:
                mapping[env] = image
    return mapping


def docker_images():
    try:
        p = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return set()
    return {line.strip() for line in p.stdout.splitlines() if line.strip()}


def resolve_index(explicit, data_root, script_dir, root):
    if explicit:
        return explicit
    env_root = os.environ.get("MMSCI_DATA_ROOT", "")
    candidates = []
    if env_root:
        candidates.append(Path(env_root) / "index.tsv")
    candidates += [data_root / "index.tsv", script_dir / "index.tsv", root / "index.tsv"]
    for cand in candidates:
        if cand.is_file():
            return cand
    return candidates[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Dataset root (defaults to $MMSCI_DATA_ROOT, then ./data).")
    ap.add_argument("--index", type=Path, default=None,
                    help="index.tsv path (defaults to <data-root>/index.tsv).")
    ap.add_argument("--available-only", action="store_true",
                    help="Only emit rows whose image exists in local `docker images`.")
    args = ap.parse_args()

    root = args.root.resolve()
    script_dir = Path(__file__).resolve().parent
    env_data_root = os.environ.get("MMSCI_DATA_ROOT", "")
    if args.data_root:
        data_root = args.data_root
    elif env_data_root:
        data_root = Path(env_data_root)
    else:
        data_root = root / "data"
    data_root = data_root.resolve()

    index = resolve_index(args.index, data_root, script_dir, root)
    if not index.is_file():
        raise FileNotFoundError(
            f"index.tsv not found at {index}. Pass --index or set MMSCI_DATA_ROOT "
            f"to the downloaded dataset root."
        )

    env_to_image = load_env_to_image(index)
    samples = data_dirs(data_root)
    available = docker_images() if args.available_only else None

    for sample_id in sorted(samples):
        sample_dir = samples[sample_id]
        env = sample_env_name(sample_dir)
        if not env:
            continue
        image = env_to_image.get(env)
        if not image:
            continue
        if available is not None and image not in available:
            continue
        rel_sample = sample_dir.relative_to(data_root)
        print(f"{image}\t{rel_sample}")


if __name__ == "__main__":
    main()
