#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_in_image.sh IMAGE SAMPLE [MODE] [OUT_DIR]

SAMPLE can be a data/... path or a sample directory name.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]]; then
    usage
    exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$script_dir"
data_root="$(realpath -m "${MMSCI_DATA_ROOT:-$root_dir/data}")"

image="$1"
sample="$2"
mode="${3:-direct}"
out_dir="${4:-$script_dir/outputs/$(date +%Y%m%d_%H%M%S)__${image//[:\/]/_}}"

env_file="${MMSCI_EVAL_ENV_FILE:-$script_dir/.env}"
if [[ ! -f "$env_file" ]]; then
    echo "missing env file: $env_file" >&2
    echo "copy .env.example to .env and set OPENAI_API_KEY" >&2
    exit 2
fi

if [[ ! -d "$data_root" ]]; then
    echo "data root not found: $data_root" >&2
    exit 2
fi

resolve_host_sample() {
    local v="$1"
    if [[ -d "$v" ]]; then
        realpath "$v"
        return
    fi
    if [[ -d "$root_dir/$v" ]]; then
        realpath "$root_dir/$v"
        return
    fi
    if [[ -d "$data_root/$v" ]]; then
        realpath "$data_root/$v"
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

sample_host="$(resolve_host_sample "$sample")"
case "$sample_host" in
    "$data_root"/*) sample_ctr="/workspace/data/${sample_host#"$data_root"/}" ;;
    "$root_dir"/*) sample_ctr="/workspace/${sample_host#"$root_dir"/}" ;;
    *) echo "sample must be under $root_dir or $data_root: $sample_host" >&2; exit 2 ;;
esac

out_host="$(realpath -m "$out_dir")"
mkdir -p "$out_host"
case "$out_host" in
    "$root_dir"/*) out_ctr="/workspace/${out_host#"$root_dir"/}" ;;
    *) echo "out dir must be under $root_dir: $out_host" >&2; exit 2 ;;
esac

docker_args=(--rm -w /workspace)
if [[ "${MMSCI_DOCKER_GPU:-0}" == "1" ]]; then
    docker_args+=(--gpus all)
fi

docker run "${docker_args[@]}" \
    -v "$root_dir:/workspace" \
    -v "$data_root:/workspace/data" \
    -v "$env_file:/run/secrets/mmsci_openai.env:ro" \
    -e MMSCI_EVAL_ENV_FILE=/run/secrets/mmsci_openai.env \
    -e MMSCI_DATA_ROOT=/workspace/data \
    -e MMSCI_EVAL_TIMEOUT="${MMSCI_EVAL_TIMEOUT:-600}" \
    -e MMSCI_INSTALL_OPENAI="${MMSCI_INSTALL_OPENAI:-1}" \
    "$image" \
    bash /workspace/run_pipeline_once.sh "$sample_ctr" "$mode" "$out_ctr"
