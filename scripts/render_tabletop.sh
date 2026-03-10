#!/bin/bash

# Parse all arguments: extract flags, collect positional args, pass through extras
debug=false
verbose=false
show_help=false
positional=()
passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) show_help=true; shift ;;
    --debug) debug=true; shift ;;
    --verbose) verbose=true; shift ;;
    --) shift ;;  # skip separator
    --*=*) passthrough+=("$1"); shift ;;  # --key=value style
    --*)  # --key value style: capture flag and its value together
      passthrough+=("$1")
      if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
        passthrough+=("$2")
        shift
      fi
      shift ;;
    *) positional+=("$1"); shift ;;
  esac
done

# Handle --help: print wrapper help (blenderproc intercepts --help, can't pass through)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
if $show_help; then
  cat <<'EOF'
Usage: render_tabletop.sh [SPLIT] [RUNS] [VIEWS] [SHARD] [OUT_DIR] [SCENE] [OPTIONS]

Render tabletop scenes with physics simulation using BlenderProc.

Positional arguments:
  SPLIT       Dataset split: train, val, test (default: train)
  RUNS        Number of scenes to render (default: 100)
  VIEWS       Camera views per scene (default: 10)
  SHARD       Shard index for parallel rendering (default: 0)
  OUT_DIR     Output directory name under data_root (default: tabletop.v3)
  SCENE       Scene preset: "packed" (upright, surface) or "pile" (random, volume)

Wrapper flags:
  --debug     Run with blenderproc debug (opens Blender GUI)
  --verbose   Enable verbose output
  --help      Show this help message

Passthrough options (passed to render_blenderproc.py):
  --placement MODE      Object placement: surface, volume, sequential, tower
  --num-objects MIN MAX Number of objects per scene
  --scale MIN MAX       Object scale range
  --physics / --no-physics  Enable/disable physics simulation
  --seed N              Random seed

For full script options, see: process/scripts/render_blenderproc.py
EOF
  exit 0
fi

# Assign positional arguments
split="${positional[0]:-train}"
runs="${positional[1]:-100}"
views="${positional[2]:-10}"
shard="${positional[3]:-0}"
out_dir="${positional[4]:-tabletop.v3}"
scene="${positional[5]:-}"  # Optional: "packed" or "pile" preset

# Build blenderproc command and script flags
# Note: blenderproc passes unknown args through to the script via parse_known_args()
blenderproc_cmd="run"
script_flags=()
$debug && blenderproc_cmd="debug"
$verbose && script_flags+=(--verbose)

data_root="${DATA_ROOT:?Set DATA_ROOT to your datasets directory}"

# Use venv's blenderproc if available, otherwise fall back to PATH
if [ -x "$REPO_ROOT/.venv/bin/blenderproc" ]; then
  BLENDERPROC="$REPO_ROOT/.venv/bin/blenderproc"
else
  BLENDERPROC="blenderproc"
fi

# Build scene-specific arguments
scene_args=()
if [ "$scene" == "packed" ]; then
  # VGN packed: upright objects, surface placement, no physics stacking
  scene_args=(--scene packed)
elif [ "$scene" == "pile" ]; then
  # VGN pile: random SO3, volume placement, containment walls, physics settling
  scene_args=(--scene pile)
fi

for ((i=1; i<=runs; i++)); do
  seed=$((i + shard * runs))
  "$BLENDERPROC" $blenderproc_cmd "$REPO_ROOT/process/scripts/render_blenderproc.py" \
    --object-path "$data_root/$out_dir/${split}_objs.txt" \
    --metadata-path "$data_root/shapenet/ShapeNetCore.v1/taxonomy.json" \
    --output-dir "$data_root/$out_dir/$split/$shard" \
    --hdri-path "$data_root/haven" --cc-material-path "$data_root/cc_textures" --hdri-strength random \
    --depth --kinect --kinect-sim --normals --diffuse --segmentation \
    --camera.extrinsics "$views" --camera.sampler shell --camera.inplane-rotation 45 --camera.jitter -0.1 0.1 \
    --clear-normals --normalize --scale 0.05 0.5 --distort 0.1 --rotation True --materials --colors auto \
    --displacement 0.5 --replace 0.5 --num_objects 1 15 --surface plane --physics \
    --writer coco --seed "$seed" "${scene_args[@]}" "${script_flags[@]}" "${passthrough[@]}";
done
