#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_pipeline_once.sh SAMPLE_DIR [MODE] [OUT_DIR]

Runs infer.py -> insert.py -> runner.py for one MMSciCode sample.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
    usage
    exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$script_dir"
data_root="${MMSCI_DATA_ROOT:-$root_dir/data}"

sample_arg="$1"
mode="${2:-direct}"
out_arg="${3:-}"

if [[ "$mode" != "direct" && "$mode" != "cot" && "$mode" != "both" ]]; then
    echo "mode must be direct, cot, or both" >&2
    exit 2
fi

for f in infer.py build_prompt.py insert.py runner.py; do
    if [[ ! -f "$script_dir/$f" ]]; then
        echo "missing eval script file: $script_dir/$f" >&2
        exit 2
    fi
done

if [[ ! -d "$data_root" ]]; then
    echo "data root not found: $data_root" >&2
    exit 2
fi

env_file="${MMSCI_EVAL_ENV_FILE:-$script_dir/.env}"
if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck source=/dev/null
    . "$env_file"
    set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is empty; set it in .env or the environment" >&2
    exit 2
fi

resolve_sample() {
    local v="$1"
    if [[ -d "$v" ]]; then
        realpath "$v"
        return
    fi
    if [[ -d "$root_dir/$v" ]]; then
        realpath "$root_dir/$v"
        return
    fi
    for d in \
        "$data_root/Python/$v" "$data_root/Python/data/$v" \
        "$data_root/C_CPP/$v" "$data_root/C_CPP/data/$v" \
        "$data_root/R/$v" "$data_root/R/data/$v"; do
        if [[ -d "$d" ]]; then
            realpath "$d"
            return
        fi
    done
    echo "sample not found: $v" >&2
    return 1
}

sample_dir="$(resolve_sample "$sample_arg")"
sample_name="$(basename "$sample_dir")"
stamp="$(date +%Y%m%d_%H%M%S)"
if [[ -n "$out_arg" ]]; then
    run_dir="$(realpath -m "$out_arg")"
else
    run_dir="$script_dir/outputs/${stamp}__${sample_name}__${mode}"
fi

infer_dir="$run_dir/infer"
infer_ready_dir="$run_dir/infer_ready"
patched_dir="$run_dir/patched"
sandbox_dir="$run_dir/sandbox"
conda_link_root="$run_dir/conda_root"

mkdir -p "$infer_dir" "$infer_ready_dir" "$patched_dir" "$sandbox_dir" "$conda_link_root/envs"

python_bin="${PYTHON:-python3}"

if [[ "${MMSCI_INSTALL_OPENAI:-1}" == "1" ]]; then
    if ! "$python_bin" - <<'PY' >/dev/null 2>&1
import openai
PY
    then
        "$python_bin" -m pip install -q 'openai>=1.30'
    fi
fi

env_name="$("$python_bin" - "$sample_dir" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "unit_test_status.json"
if not p.is_file():
    raise SystemExit(0)
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
print(((data.get("environment") or {}).get("conda_env_name")) or "")
PY
)"

if [[ -n "$env_name" ]]; then
    linked=0
    if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX:-}" ]]; then
        ln -sfn "$CONDA_PREFIX" "$conda_link_root/envs/$env_name"
        linked=1
    fi
    for cand in "/opt/conda/envs/$env_name"; do
        if [[ -d "$cand" ]]; then
            ln -sfn "$cand" "$conda_link_root/envs/$env_name"
            linked=1
            break
        fi
    done
    if [[ "$linked" != "1" ]]; then
        active_python="$(command -v "$python_bin" || true)"
        if [[ -n "$active_python" ]]; then
            active_env="$(cd "$(dirname "$active_python")/.." && pwd)"
            if [[ -d "$active_env/bin" ]]; then
                ln -sfn "$active_env" "$conda_link_root/envs/$env_name"
            fi
        fi
    fi
fi

echo "sample=$sample_name"
echo "mode=$mode"
echo "out=$run_dir"
echo "data_root=$data_root"

"$python_bin" "$script_dir/infer.py" "$sample_dir" \
    --mode "$mode" \
    --out-dir "$infer_dir" \
    --env-file "$env_file"

ready_count=0
while IFS= read -r -d '' d; do
    if [[ -f "$d/extraction.json" ]]; then
        ln -sfn "$d" "$infer_ready_dir/$(basename "$d")"
        ready_count=$((ready_count + 1))
    fi
done < <(find "$infer_dir" -mindepth 1 -maxdepth 1 -type d -print0)

if [[ "$ready_count" -eq 0 ]]; then
    echo "no usable inference outputs under $infer_dir" >&2
    echo "check $infer_dir/_run_summary.json and response/error output above" >&2
    exit 1
fi

"$python_bin" "$script_dir/insert.py" \
    --all \
    --infer-root "$infer_ready_dir" \
    --data-root "$data_root" \
    --out-dir "$patched_dir" \
    --summary "$run_dir/insert_summary.json"

"$python_bin" "$script_dir/runner.py" \
    --all \
    --patched-root "$patched_dir" \
    --data-root "$data_root" \
    --sandbox-root "$sandbox_dir" \
    --conda-root "$conda_link_root" \
    --timeout "${MMSCI_EVAL_TIMEOUT:-600}" \
    --summary "$run_dir/runner_summary.json"

echo "summary=$run_dir/runner_summary.json"
