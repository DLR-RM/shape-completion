#!/usr/bin/env bash
input_dir="${1:-}"
output_dir="${2:-}"
shift 2 2>/dev/null || true
command=("$@")

find "$input_dir" -name "model.obj" -type f | shuf | while read -r obj_path; do
    obj_id_path=$(dirname "$obj_path")
    obj_category_path=$(dirname "$obj_id_path")
    obj_id=$(basename "$obj_id_path")
    obj_category=$(basename "$obj_category_path")

    if [ -d "$output_dir/$obj_category/$obj_id" ] || [ -d "$output_dir/$obj_id" ]; then
        echo "Skipping $obj_category/$obj_id"
        continue
    fi
 
    full_command=("${command[@]}" --object-path "$obj_path" --output-dir "$output_dir")
    "${full_command[@]}"
done