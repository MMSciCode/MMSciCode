#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_completed_images.sh [MODE] [OUT_ROOT]

Runs available Docker images listed in the dataset index.tsv by default.
Each line is logged and failures do not stop later images.

The image list is read from $MMSCI_IMAGE_INDEX, defaulting to
$MMSCI_DATA_ROOT/index.tsv (the index.tsv shipped with the dataset).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$script_dir"
mode="${1:-direct}"
out_root="${2:-$script_dir/outputs/batch_$(date +%Y%m%d_%H%M%S)}"
data_root="$(realpath -m "${MMSCI_DATA_ROOT:-$root_dir/data}")"
index_file="${MMSCI_IMAGE_INDEX:-$data_root/index.tsv}"
manifest="$out_root/images.manifest.tsv"
state="$out_root/eval.state.tsv"
failed="$out_root/eval.failed.tsv"

mkdir -p "$out_root"
python3 "$script_dir/map_images.py" \
    --available-only \
    --data-root "$data_root" \
    --index "$index_file" > "$manifest"

if [[ ! -s "$manifest" ]]; then
    echo "no matching local images found" >&2
    exit 1
fi

printf "image\tsample\tstatus\tseconds\tout_dir\n" > "$state"
: > "$failed"

while IFS=$'\t' read -r image sample; do
    [[ -n "$image" && -n "$sample" ]] || continue
    safe="${image//[:\/]/_}"
    run_out="$out_root/$safe"
    t0="$(date +%s)"
    echo "==> $image $sample"
    if MMSCI_DATA_ROOT="$data_root" "$script_dir/run_in_image.sh" "$image" "$sample" "$mode" "$run_out"; then
        dt="$(( $(date +%s) - t0 ))"
        printf "%s\t%s\tpass\t%s\t%s\n" "$image" "$sample" "$dt" "$run_out" >> "$state"
    else
        dt="$(( $(date +%s) - t0 ))"
        printf "%s\t%s\tfail\t%s\t%s\n" "$image" "$sample" "$dt" "$run_out" >> "$state"
        printf "%s\t%s\t%s\n" "$image" "$sample" "$run_out" >> "$failed"
    fi
done < "$manifest"

echo "state=$state"
echo "failed=$failed"
